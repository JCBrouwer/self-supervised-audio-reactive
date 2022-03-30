from math import ceil
from time import time

import numpy as np
import torch
import triton
import triton.language as tl
from torch.nn.functional import conv1d


def diff_conv1d(x, mode="replicate"):
    dim = len(x.shape)
    while len(x.shape) < 3:
        x = x[:, None]

    channels = x.shape[1] * x.shape[2]
    kernel = torch.tensor([-1, 1]).view(1, 1, -1).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c * h, w)
    x = x.transpose(0, 2)

    x = conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t - 1, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return torch.cat((t, t[[-1]]))


def absdiff_kernel_pytorch(x, BLOCK_T=32, BLOCK_E=32):
    x = x.view(len(x), -1)
    T, E = x.shape
    stride_xt, stride_xe = x.stride()
    x = x.flatten()

    y = torch.zeros(T, device=x.device)
    stride_yt = y.stride(0)

    x_ptr, y_ptr = 0, 0

    for t in range(ceil(T / BLOCK_T)):
        for e in range(ceil(E / BLOCK_E)):

            t_idxs = t * BLOCK_T + torch.arange(0, BLOCK_T)
            e_idxs = e * BLOCK_E + torch.arange(0, BLOCK_E)

            x_idx_t = x_ptr + (t_idxs[:, None] * stride_xt + e_idxs[None, :] * stride_xe)
            mask_t = (t_idxs < T)[:, None] & (e_idxs < E)[None, :]
            if not mask_t.all():
                x_idx_t = x_idx_t.clamp(0, T * E - 1)

            x_idx_t1 = x_ptr + ((t_idxs[:, None] + 1) * stride_xt + e_idxs[None, :] * stride_xe)
            mask_t1 = ((t_idxs + 1) < T)[:, None] & (e_idxs < E)[None, :]
            if not mask_t1.all():
                x_idx_t1 = x_idx_t1.clamp(0, T * E - 1)

            x_t = x[x_idx_t]
            x_t[~mask_t] = 0.0

            x_t1 = x[x_idx_t1]
            x_t1[~mask_t1] = 0.0

            y_t = torch.sum(torch.abs(x_t1 - x_t), axis=1)

            y_idxs = y_ptr + t_idxs * stride_yt

            if (y_idxs > T - 1).any():
                y_idxs = y_idxs.clamp(0, T - 1)
                skip = y_idxs != T - 1
                y_idxs = y_idxs[skip]
                y_t = y_t[skip]

            y[y_idxs] += y_t

    y[-1] = y[-2]
    return y


@triton.jit
def absdiff_kernel(x_ptr, y_ptr, T, E, stride_xt, stride_xe, stride_yt, **meta):
    """Compute absolute difference between consecutive elements along time axis of a matrix, reducing element axis"""
    BLOCK_T, BLOCK_E = meta["BLOCK_T"], meta["BLOCK_E"]
    t, e = tl.program_id(0), tl.program_id(1)
    t_idxs = t * BLOCK_T + tl.arange(0, BLOCK_T)
    e_idxs = e * BLOCK_E + tl.arange(0, BLOCK_E)

    x_idx_t = x_ptr + (t_idxs[:, None] * stride_xt + e_idxs[None, :] * stride_xe)
    mask_t = (t_idxs < T)[:, None] & (e_idxs < E)[None, :]

    x_idx_t1 = x_ptr + ((t_idxs[:, None] + 1) * stride_xt + e_idxs[None, :] * stride_xe)
    mask_t1 = ((t_idxs + 1) < T)[:, None] & (e_idxs < E)[None, :]

    x_t = tl.load(x_idx_t, mask=mask_t, other=0.0)
    x_t1 = tl.load(x_idx_t1, mask=mask_t1, other=0.0)

    y_t = tl.sum(tl.abs(x_t1 - x_t), axis=1)

    y_idxs = y_ptr + t_idxs * stride_yt

    tl.atomic_add(y_idxs, y_t, mask=(t_idxs + 1) < T)


def absdiff(x):
    x = x.view(len(x), -1)
    n_timestamps, n_elements = x.shape
    y = torch.zeros(len(x), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(n_timestamps, meta["BLOCK_T"]), triton.cdiv(n_elements, meta["BLOCK_E"]))
    absdiff_kernel[grid](x, y, n_timestamps, n_elements, x.stride(0), x.stride(1), y.stride(0), BLOCK_T=32, BLOCK_E=32)
    y[-1] = y[-2]
    return y


if __name__ == "__main__":
    np.set_printoptions(precision=1, linewidth=200)

    T, C, H, W = 472, 3, 64, 128
    x = torch.randn((T, C, H, W), device="cuda")
    y = torch.randn(T, device="cuda")

    pt = torch.diff(x, dim=0).abs().sum((1, 2, 3))
    pt = torch.cat((pt, pt[[-1]]))

    t = time()
    for _ in range(20):
        eff = absdiff_kernel_pytorch(x)
        torch.cuda.synchronize()
    print((time() - t) / 20, "s")

    rel_diff = (pt - eff).abs().div(torch.maximum(pt, eff))
    print("relative differences", rel_diff.min().item(), rel_diff.mean().item(), rel_diff.max().item())
    assert torch.allclose(pt, eff, rtol=2e-4)

    t = time()
    for _ in range(20):
        trt = absdiff(x)
        torch.cuda.synchronize()
    print((time() - t) / 20, "s")

    rel_diff = (pt - trt).abs().div(torch.maximum(pt, trt))
    print("relative differences", rel_diff.min().item(), rel_diff.mean().item(), rel_diff.max().item())
    assert torch.allclose(pt, trt)
