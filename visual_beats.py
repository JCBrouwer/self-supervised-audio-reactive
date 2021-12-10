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

from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision as tv
from kornia.filters import gaussian_blur2d
from torch.nn.functional import pad
from torchvision.transforms.functional import resize


@torch.jit.script
def normalize(tensor):
    result = tensor - tensor.min()
    result = result / result.max()
    return result


@torch.jit.script
def standardize(array):
    result = torch.clamp(array, torch.quantile(array, 0.05), torch.quantile(array, 0.95) + 1e-10)
    result = normalize(result)
    return result


@torch.jit.script
def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    return rho, phi


@torch.jit.script
def pyramid_expand(image, upscale: int = 2):
    dim = image.dim()
    if dim == 3:
        h, w, _ = image.shape
        image = image.permute(2, 0, 1).unsqueeze(0)
    elif dim == 2:
        h, w = image.shape
        image = image[None, None]
    else:
        raise Exception("Only 2 and 3 dimensional images are supported: [H,W] or [H,W,C]")

    out_shape = [int(torch.ceil(h * upscale)), int(torch.ceil(w * upscale))]
    resized = resize(image, out_shape, antialias=True)

    sigma = 2 * upscale / 6.0
    ks = int(sigma * 4 + 1)
    out = gaussian_blur2d(resized, kernel_size=(ks, ks), sigma=(sigma, sigma)).squeeze()

    if dim == 3:
        out = out.permute(1, 2, 0)
    return out


@torch.jit.script
def pyramid_reduce(image, downscale: int = 2):
    dim = image.dim()
    if dim == 3:
        h, w, _ = image.shape
        image = image.permute(2, 0, 1).unsqueeze(0)
    elif dim == 2:
        h, w = image.shape
        image = image[None, None]
    else:
        raise Exception("Only 2 and 3 dimensional images are supported: [H,W] or [H,W,C]")

    sigma = 2 * downscale / 6.0
    ks = int(sigma * 4 + 1)
    blurred = gaussian_blur2d(image, kernel_size=(ks, ks), sigma=(sigma, sigma))

    out_shape = [int(torch.ceil(h / downscale)), int(torch.ceil(w / downscale))]
    out = resize(blurred, out_shape, antialias=True).squeeze()

    if dim == 3:
        out = out.permute(1, 2, 0)
    return out


@torch.jit.script
def to_grayscale(im):
    # magic numbers from https://en.wikipedia.org/wiki/Grayscale
    return 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]


@torch.jit.script
def lucas_kanade(im1, im2):
    """Implementation of the Lucas-Kanade optical flow algorithm
    ported from https://stackoverflow.com/a/14325821

    Args:
        im1 : First image
        im2 : Second image

    Returns:
        flow : Optical flow between the two images
    """
    win = 2

    im1, im2 = to_grayscale(im1), to_grayscale(im2)

    I_x, I_y, I_t = torch.zeros_like(im1), torch.zeros_like(im1), torch.zeros_like(im1)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = torch.stack(
        (
            gaussian_blur2d((I_x * I_x)[None, None], (5, 5), (3.0, 3.0)).squeeze(),
            gaussian_blur2d((I_y * I_y)[None, None], (5, 5), (3.0, 3.0)).squeeze(),
            gaussian_blur2d((I_x * I_y)[None, None], (5, 5), (3.0, 3.0)).squeeze(),
            gaussian_blur2d((I_x * I_t)[None, None], (5, 5), (3.0, 3.0)).squeeze(),
            gaussian_blur2d((I_y * I_t)[None, None], (5, 5), (3.0, 3.0)).squeeze(),
        ),
        dim=-1,
    )
    del I_x, I_y, I_t
    cum_params = torch.cumsum(torch.cumsum(params, dim=0), dim=1)
    del params
    win_params = (
        cum_params[2 * win + 1 :, 2 * win + 1 :]
        - cum_params[2 * win + 1 :, : -1 - 2 * win]
        - cum_params[: -1 - 2 * win, 2 * win + 1 :]
        + cum_params[: -1 - 2 * win, : -1 - 2 * win]
    )
    del cum_params
    det = win_params[..., 0] * win_params[..., 1] - win_params[..., 2] ** 2
    flow_x = torch.where(
        det != 0,
        (win_params[..., 1] * win_params[..., 3] - win_params[..., 2] * win_params[..., 4]) / det,
        torch.zeros_like(det),
    )
    flow_y = torch.where(
        det != 0,
        (win_params[..., 0] * win_params[..., 4] - win_params[..., 2] * win_params[..., 3]) / det,
        torch.zeros_like(det),
    )
    flow = torch.zeros(im1.shape + (2,), device=im1.device, dtype=im1.dtype)
    flow[win + 1 : -1 - win, win + 1 : -1 - win, 0] = flow_x[:-1, :-1]
    flow[win + 1 : -1 - win, win + 1 : -1 - win, 1] = flow_y[:-1, :-1]
    return flow


@torch.jit.script
def lucas_kanade_gaussian_pyramid(im1, im2, levels: int = 4):

    pyramid = torch.jit.annotate(List[List[torch.Tensor]], [])
    cur = [im1, im2]
    for _ in range(levels):
        cur = [pyramid_reduce(a) for a in cur]
        pyramid.append(cur)

    flow = lucas_kanade(cur[0], cur[1])

    for pyr1, pyr2 in pyramid[-2::-1]:
        flow = 2 * pyramid_expand(flow)
        flow = flow[: pyr1.shape[0], : pyr1.shape[1]]  # account for shapes not quite matching
        flow += lucas_kanade(pyr1, pyr2)

    return flow


@torch.jit.script
def optical_flow(video, n_pyr: int = 4):
    flow = []
    for prev, next in zip(video[1:], video[:-1]):
        flow.append(lucas_kanade_gaussian_pyramid(prev, next))
    flow = torch.stack(flow)
    flow = torch.stack(cart2pol(flow[..., 0], flow[..., 1]), dim=1)
    flow = torch.cat((flow[[0]], flow))
    flow[:, 0] = standardize(flow[:, 0])
    flow[:, 1] = normalize(flow[:, 1])
    return flow


def optical_flow_cpu(video):
    dev = video.device
    video = video.cpu().numpy()
    flow = []
    for prev, next in zip(video, video[1:]):
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)
        flow_frame = torch.from_numpy(
            cv2.calcOpticalFlowFarneback(
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
        )
        flow.append(torch.stack(cart2pol(flow_frame[..., 0], flow_frame[..., 1])))
    flow = torch.stack(flow)
    flow = torch.cat((flow[[0]], flow))
    flow[:, 0] = standardize(flow[:, 0])
    flow[:, 1] = normalize(flow[:, 1])
    return flow.to(dev)


@torch.jit.script
def median_filter_2d(
    x, k: Tuple[int, int] = (3, 3), s: Tuple[int, int] = (1, 1), p: Tuple[int, int, int, int] = (1, 1, 1, 1)
):
    x = pad(x, p, mode="reflect")
    x = x.unfold(2, k[0], s[0]).unfold(3, k[1], s[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


@torch.jit.script
def directogram(flow, bins: int = 64):
    bin_width = 1 / bins
    angle_bins = torch.linspace(0, 1, bins, device=flow.device)[None, None, None, :]

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
    with torch.inference_mode():
        from time import time

        t = time()
        video, audio, info = tv.io.read_video("/home/hans/datasets/audiovisual/maua/airhead2.mp4")
        video = (
            tv.transforms.functional.resize(video.permute(0, 3, 1, 2), 128).permute(0, 2, 3, 1).float().div(255).cuda()
        )
        print("load:", time() - t, "sec")

        # t = time()
        # flow = optical_flow(video)
        # print("flow:", time() - t, "sec")

        # hsv = np.ones((flow.shape[0], flow.shape[2], flow.shape[3], 3), dtype=np.float32)
        # hsv[..., 0] = flow[:, 1].cpu().numpy()
        # hsv[..., 2] = flow[:, 0].cpu().numpy()
        # hsv = (hsv * 255).astype(np.uint8)
        # rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_HSV2RGB) for f in hsv])
        # tv.io.write_video(
        #     "output/flow_lk.mp4",
        #     video_array=rgb,
        #     fps=info["video_fps"],
        #     video_codec="h264",
        #     audio_array=audio,
        #     audio_fps=info["audio_fps"],
        #     audio_codec="aac",
        # )

        t = time()
        flow = optical_flow_cpu(video)
        print("flow:", time() - t, "sec")

        hsv = np.ones((flow.shape[0], flow.shape[2], flow.shape[3], 3), dtype=np.float32)
        hsv[..., 0] = flow[:, 1].cpu().numpy()
        hsv[..., 2] = flow[:, 0].cpu().numpy()
        hsv = (hsv * 255).astype(np.uint8)
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_HSV2RGB) for f in hsv])
        tv.io.write_video(
            "output/flow_fb.mp4",
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
        plt.tight_layout()
        plt.axis("off")
        plt.savefig("output/directogram.pdf")
        plt.close()

        t = time()
        decelleration = spectral_flux(direct)
        print("decelleration:", time() - t, "sec")

        plt.figure(figsize=(12, 6))
        plt.imshow(decelleration.T.cpu().numpy())
        plt.tight_layout()
        plt.axis("off")
        plt.savefig("output/decelleration.pdf")
        plt.close()

        t = time()
        onsets = onset_envelope(decelleration)
        print("impact:", time() - t, "sec")

        plt.figure(figsize=(12, 6))
        plt.plot(onsets.cpu().numpy())
        plt.tight_layout()
        plt.axis("off")
        plt.savefig("output/onsets.pdf")
        plt.close()
