from typing import Tuple

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from kornia.color.hsv import rgb_to_hsv
from ssar.analysis.efficient_quantile import quantile
from ssar.features.processing import cart2pol, median_filter_2d, normalize, onset_envelope, spectral_flux, standardize


@torch.jit.script
def redogram(video, bins=32):
    return torch.stack([torch.histogram(frame, bins=bins, density=True) for frame in video[:, 0]])


@torch.jit.script
def greenogram(video, bins=32):
    return torch.stack([torch.histogram(frame, bins=bins, density=True) for frame in video[:, 1]])


@torch.jit.script
def blueogram(video, bins=32):
    return torch.stack([torch.histogram(frame, bins=bins, density=True) for frame in video[:, 2]])


@torch.jit.script
def huestogram(video, bins=32):
    hue = rgb_to_hsv(video)[:, 0]
    return torch.stack([torch.histogram(frame, bins=bins, density=True) for frame in hue])


@torch.jit.script
def saturogram(video, bins=32):
    sat = rgb_to_hsv(video)[:, 1]
    return torch.stack([torch.histogram(frame, bins=bins, density=True) for frame in sat])


@torch.jit.script
def valueogram(video, bins=32):
    val = rgb_to_hsv(video)[:, 2]
    return torch.stack([torch.histogram(frame, bins=bins, density=True) for frame in val])


@torch.jit.script
def visual_variance(video):
    return video.std((1, 2, 3)) ** 2


@torch.jit.script
def absdiff(video, stride=64):
    y = []
    for i in range(0, len(video), stride):
        y_part = torch.diff(video[i : i + stride + 1], dim=0).abs().sum((1, 2, 3))
        if len(y_part) < 1:
            continue
        y.append(y_part)
    y = torch.cat((*y, y[-1][[-1]]))
    return y


@torch.jit.script
def fft(video, stride=64):
    _, _, h, w = video.shape
    y = []
    for i in range(0, len(video), stride):
        y_part = torch.fft.rfft2(video[i : i + stride + 1], norm="forward")[..., : h // 2, : w // 2]
        y.append(y_part)
    y = torch.cat(y)
    return y


def video_spectrogram(video):
    _, _, h, w = video.shape
    freqs = torch.abs(fft(video))
    freqs = freqs.clamp(quantile(freqs, 0.0015), quantile(freqs, 0.9985)).cpu().numpy()
    radius, flags = max(h, w) // 4, cv2.WARP_FILL_OUTLIERS
    freqs = torch.from_numpy(
        np.array([[cv2.linearPolar(channel, (0, 0), radius, flags) for channel in frame] for frame in freqs])
    )
    freqs = freqs.mean((1, 2))[:, 2:]
    return freqs


def low_frequency_power(video):
    spec = video_spectrogram(video)
    t, f = spec.shape
    return spec[:, : f // 3].sum(1)


def mid_frequency_power(video):
    spec = video_spectrogram(video)
    t, f = spec.shape
    return spec[:, f // 3 : 2 * f // 3].sum(1)


def high_frequency_power(video):
    spec = video_spectrogram(video)
    t, f = spec.shape
    return spec[:, 2 * f // 3 :].sum(1)


def adaptive_frequency_power(video, k=10):
    spec = video_spectrogram(video)
    max_var_freqs = torch.topk(spec.std(0), k=k).indices
    return spec[:, max_var_freqs].sum(1)


def _optical_flow_cpu_worker(prev_next):
    i, (prev, next) = prev_next
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)
    flow_frame = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    magang = cart2pol(flow_frame[..., 0], flow_frame[..., 1])
    flow_frame = np.stack(magang)
    return flow_frame


def optical_flow_cpu(video):
    dev = video.device
    video = video.cpu().numpy()
    flow = []
    with mp.Pool(mp.cpu_count()) as pool:
        flow = [torch.from_numpy(f) for f in pool.map(_optical_flow_cpu_worker, list(enumerate(zip(video, video[1:]))))]
    flow = torch.stack(flow)
    flow = torch.cat((flow[[0]], flow))
    flow[:, 0] = standardize(flow[:, 0])
    flow[:, 1] = normalize(flow[:, 1])
    return flow.to(dev)


@torch.jit.script
def directogram(flow, bins: int = 8):
    bin_width = 256 // bins
    angle_bins = torch.linspace(0, 255, bins, device=flow.device, dtype=torch.uint8)[None, None, None, :]
    flow = flow.mul(255).to(torch.uint8)

    # TODO figure out a more memory efficient way of doing this
    bin_idxs = torch.argmax((torch.abs(angle_bins - flow[:, 1, :, :, None]) <= bin_width).to(torch.uint8), dim=-1)

    dg = torch.zeros((flow.shape[0], bins), device=flow.device, dtype=torch.int64)
    for t in range(flow.shape[0]):
        for bin in range(bins):
            dg[t, bin] = torch.sum(flow[t, 0][bin_idxs[t] == bin])
    dg = dg.float() / 255.0

    dg = median_filter_2d(dg[None, None]).squeeze()

    return dg


def video_flow_onsets(video):
    flow = optical_flow_cpu(video.permute(0, 2, 3, 1).cpu())
    spec = directogram(flow)
    flux = spectral_flux(spec)
    onset = onset_envelope(flux)
    return onset


def video_spectral_onsets(video):
    spec = video_spectrogram(video)
    flux = spectral_flux(spec)
    onset = onset_envelope(flux)
    return onset


# if __name__ == "__main__":
#     with torch.inference_mode():
#         from glob import glob
#         from pathlib import Path

#         import cv2
#         import decord as de
#         import matplotlib.pyplot as plt

#         de.bridge.set_bridge("torch")

#         for vfile in glob("/home/hans/datasets/audiovisual/maua256/*"):
#             v = de.VideoReader(vfile)
#             video = v[:]
#             video = video.permute(0, 3, 1, 2).div(255).cuda()
#             t, c, h, w = video.shape
#             print("\n", Path(vfile).stem, video.shape)

#             try:
#                 freqs = video_spectrogram(video)

#                 fig, ax = plt.subplots(4, 1, figsize=(16, 10))
#                 ax[0].imshow(normalize(freqs).T.numpy()[::-1])
#                 ax[0].axis("off")
#                 ax[1].plot(adaptive_frequency_power(video))
#                 ax[2].plot(video_flow_onsets(video))
#                 ax[3].plot(video_spectral_onsets(video))
#                 plt.tight_layout()
#                 plt.show()
#             except Exception as e:
#                 print(e)
