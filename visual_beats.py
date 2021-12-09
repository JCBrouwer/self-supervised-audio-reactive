"""
PyTorch implementation of video rhythm features introduced in "Visual Rhythm and Beat"
@article{davisVisualRhythmBeat2018a,
  title = {Visual Rhythm and Beat},
  author = {Davis, Abe and Agrawala, Maneesh},
  date = {2018-08-10},
  journaltitle = {ACM Transactions on Graphics},
  shortjournal = {ACM Trans. Graph.},
  volume = {37},
  number = {4},
  pages = {1--11},
  issn = {0730-0301, 1557-7368},
  doi = {10.1145/3197517.3201371},
  url = {https://dl.acm.org/doi/10.1145/3197517.3201371},
}
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv


@torch.jit.script
def normalize(tensor):
    result = tensor - tensor.min()
    result = result / result.max()
    return result


@torch.jit.script
def standardize(array):
    result = torch.clamp(array, torch.quantile(array, 0.25), torch.quantile(array, 0.75) + 1e-10)
    result = normalize(result)
    return result


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


# TODO differentiable + GPU accelerated version with PTFlow https://ptlflow.readthedocs.io/en/latest/
def optical_flow(video):
    dev = video.device
    video = video.cpu().numpy()
    flow = []
    for prev, next in zip(video, video[1:]):
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)
        flow_frame = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 5, 15, 3, 7, 1.5, 10)
        mag, ang = cart2pol(flow_frame[..., 0], flow_frame[..., 1])
        flow.append(torch.stack((torch.from_numpy(mag), torch.from_numpy(ang))))
    flow = torch.stack(flow)
    flow = torch.cat((flow[[0]], flow))
    flow[:, 0] = standardize(flow[:, 0])
    return flow


@torch.jit.script
def median_filter_2d(
    x, k: Tuple[int, int] = (3, 3), s: Tuple[int, int] = (1, 1), p: Tuple[int, int, int, int] = (1, 1, 1, 1)
):
    x = F.pad(x, p, mode="reflect")
    x = x.unfold(2, k[0], s[0]).unfold(3, k[1], s[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


@torch.jit.script
def directogram(flow, bins: int = 64):
    bin_width = 2 * torch.pi / bins
    angle_bins = torch.linspace(-torch.pi, torch.pi, bins, device=flow.device)[None, None, None, :]

    bin_idxs = torch.argmax((torch.abs(angle_bins - flow[:, 1, :, :, None]) <= bin_width).int(), dim=-1)

    dg = torch.zeros((flow.shape[0], bins), device=flow.device)
    for t in range(flow.shape[0]):
        for bin in range(bins):
            dg[t, bin] = torch.sum(flow[t, 0][bin_idxs[t] == bin])

    dg = median_filter_2d(dg[None, None]).squeeze()

    return dg


@torch.jit.script
def spectral_flux(gram):
    return torch.diff(gram, dim=0, append=torch.zeros((1, gram.shape[1]), device=gram.device))


@torch.jit.script
def onset_envelope(flux):
    u = torch.sum(0.5 * (flux + torch.abs(flux)), dim=1)
    u = torch.clamp(u, 0, torch.quantile(u, 0.98))
    u /= u.max()
    return u


def tempogram(onset):
    return


if __name__ == "__main__":
    from time import time

    t = time()
    video, audio, info = tv.io.read_video("/home/hans/datasets/audiovisual/maua/airhead2.mp4")
    video = tv.transforms.functional.resize(video.permute(0, 3, 1, 2), 128).permute(0, 2, 3, 1).cuda()
    print("load:", time() - t, "sec")

    t = time()
    flow = optical_flow(video)
    print("flow:", time() - t, "sec")

    hsv = np.ones((flow.shape[0], flow.shape[2], flow.shape[3], 3), dtype=np.float32)
    hsv[..., 0] = normalize(flow[:, 1].cpu().numpy())
    hsv[..., 2] = flow[:, 0].cpu().numpy()
    hsv = (hsv * 255).astype(np.uint8)
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_HSV2RGB) for f in hsv])
    tv.io.write_video(
        "output/flow.mp4",
        video_array=rgb,
        fps=info["video_fps"],
        video_codec="h264",
        audio_array=audio,
        audio_fps=info["audio_fps"],
        audio_codec="aac",
    )

    t = time()
    direct = directogram(flow)
    print("directogram:", time() - t, "sec")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.imshow(direct.T.cpu().numpy())
    plt.savefig("output/directogram.pdf")
    plt.tight_layout()
    plt.axis("off")
    plt.close()

    t = time()
    decelleration = spectral_flux(direct)
    print("decelleration:", time() - t, "sec")

    plt.figure(figsize=(12, 6))
    plt.imshow(decelleration.T.cpu().numpy())
    plt.savefig("output/decelleration.pdf")
    plt.tight_layout()
    plt.axis("off")
    plt.close()

    t = time()
    onsets = onset_envelope(decelleration)
    print("impact:", time() - t, "sec")

    plt.figure(figsize=(12, 6))
    plt.plot(onsets.cpu().numpy())
    plt.savefig("output/onsets.pdf")
    plt.tight_layout()
    plt.axis("off")
    plt.close()
