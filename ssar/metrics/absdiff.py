from math import ceil

import numpy as np
import torch
import triton
import triton.language as tl
from torch.nn.functional import conv1d


def absdiff_kernel_test(T=27, E=53, stride_xt=53, stride_xe=1, stride_yt=1, BLOCK_T=8, BLOCK_E=8):
    x = torch.arange(T * E) + torch.rand(T * E)
    print(x.numpy())
    y = torch.zeros(T)
    x_ptr, y_ptr = 0, 0
    for t in range(ceil(T / BLOCK_T)):
        for e in range(ceil(E / BLOCK_E)):
            t_idxs = t * BLOCK_T + torch.arange(0, BLOCK_T)
            e_idxs = e * BLOCK_E + torch.arange(0, BLOCK_E)

            x_idx_t = x_ptr + (t_idxs[:, None] * stride_xt + e_idxs[None, :] * stride_xe)
            mask_t = (t_idxs < T)[:, None] & (e_idxs < E)[None, :]
            print(x_idx_t.numpy())
            if not mask_t.all():
                print(mask_t.numpy())
                x_idx_t = x_idx_t.clamp(0, T * E - 1)

            x_idx_t1 = x_ptr + ((t_idxs[:, None] + 1) * stride_xt + e_idxs[None, :] * stride_xe)
            mask_t1 = ((t_idxs + 1) < T)[:, None] & (e_idxs < E)[None, :]
            print(x_idx_t1.numpy())
            if not mask_t1.all():
                print(mask_t1.numpy())
                x_idx_t1 = x_idx_t1.clamp(0, T * E - 1)

            x_t = x[x_idx_t]
            print(x_t.numpy())
            x_t[~mask_t] = 0.0

            x_t1 = x[x_idx_t1]
            print(x_idx_t1.numpy())
            x_t1[~mask_t1] = 0.0

            y_t = torch.sum(torch.abs(x_t1 - x_t), axis=1)
            print(y_t.numpy())

            y_idxs = y_ptr + t_idxs * stride_yt
            y_idxs = y_idxs.clamp(0, T)
            skip = y_idxs != T
            y_idxs = y_idxs[skip]
            y[y_idxs] = y_t[skip]
    print(y[: T + 3].numpy())
    print(torch.diff(x[: T * E].reshape(T, E), dim=0).abs_().sum(1).numpy())


np.set_printoptions(precision=1, linewidth=200)
absdiff_kernel_test()
exit()


@triton.jit
def absdiff_kernel(x_ptr, y_ptr, T, E, stride_xt, stride_xe, stride_yt, **meta):
    """Compute absolute difference between consecutive elements along time axis of a matrix, reducing element axis"""
    BLOCK_T, BLOCK_E = meta["BLOCK_T"], meta["BLOCK_E"]
    t_idxs = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    e_idxs = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)

    x_idx_t = x_ptr + (t_idxs[:, None] * stride_xt + e_idxs[None, :] * stride_xe)
    mask_t = (t_idxs < T)[:, None] & (e_idxs < E)[None, :]

    x_idx_t1 = x_ptr + ((t_idxs[:, None] + 1) * stride_xt + e_idxs[None, :] * stride_xe)
    mask_t1 = ((t_idxs + 1) < T)[:, None] & (e_idxs < E)[None, :]

    x_t = tl.load(x_idx_t, mask=mask_t, other=0.0)
    x_t1 = tl.load(x_idx_t1, mask=mask_t1, other=0.0)

    y_t = tl.sum(tl.abs(x_t1 - x_t), axis=1)

    tl.atomic_add(y_ptr + t_idxs * stride_yt, y_t, mask=(t_idxs + 1) < T)


from time import time


def absdiff(x):
    x = x.view(len(x), -1)
    n_timestamps, n_elements = x.shape
    print(n_timestamps, n_elements)
    y = torch.zeros(len(x), device=x.device, dtype=x.dtype)
    print(y.shape)
    grid = lambda meta: (triton.cdiv(n_timestamps, meta["BLOCK_T"]), triton.cdiv(n_elements, meta["BLOCK_E"]))
    print(x.stride(), y.stride())
    t = time()
    absdiff_kernel[grid](x, y, n_timestamps, n_elements, x.stride(0), x.stride(1), y.stride(0), BLOCK_T=32, BLOCK_E=32)
    torch.cuda.synchronize()
    print(time() - t)
    return y


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
