"""
Implementation of Gunnar Farneback's optical flow algorithm
ported from https://github.com/ericPrince/optical-flow
"""

from typing import List

import numpy as np
import scipy.ndimage
import torch
from torch.nn.functional import conv1d, pad

from visual_beats import cart2pol, normalize, pyramid_expand, pyramid_reduce, to_grayscale


@torch.jit.script
def correlate1d(x, w, dim: int):
    x = x.transpose(dim, -1)
    shape = x.shape
    x = torch.flatten(x, 0, -2)[None]  # [1, c = (? * ? * ...), t]

    x = pad(x, pad=(len(w) // 2, len(w) // 2), mode="constant", value=0.0)
    x = conv1d(x, w[None, None].repeat(x.shape[1], 1, 1), groups=x.shape[1])

    x = x.squeeze()  # [c, t]
    x = x.reshape(shape)
    x = x.transpose(-1, dim)
    return x


@torch.jit.script
def poly_exp(f, c, sigma: float):
    """Calculates the local polynomial expansion of a 2D signal, as described by Farneback

    Uses separable normalized correlation

    $f ~ x^T A x + B^T x + C$

    If f[i, j] and c[i, j] are the signal value and certainty of pixel (i, j) then
    A[i, j] is a 2x2 array representing the quadratic term of the polynomial, B[i, j]
    is a 2-element array representing the linear term, and C[i, j] is a scalar
    representing the constant term.

    Args:
        f (Tensor): Itorchut signal
        c (Tensor): Certainty of signal
        sigma (float): Standard deviation of applicability Gaussian kernel

    Returns:
        A (Tensor): Quadratic term of polynomial expansion
        B (Tensor): Linear term of polynomial expansion
        C (Tensor): Constant term of polynomial expansion
    """
    # Calculate applicability kernel (1D because it is separable)
    n = int(4 * sigma + 1)
    x = torch.arange(-n, n + 1, dtype=torch.int, device=f.device)
    a = torch.exp(-(x ** 2) / (2 * sigma ** 2))  # a: applicability kernel [n]

    # b: calculate b from the paper. Calculate separately for X and Y dimensions
    # [n, 6]
    one = torch.ones(a.shape, device=f.device)
    bx = torch.stack([one, x, one, x ** 2, one, x], dim=-1)
    by = torch.stack([one, one, x, one, x ** 2, x], dim=-1)

    # Pre-calculate product of certainty and signal
    cf = c * f

    # G and v are used to calculate "r" from the paper: v = G*r
    # r is the parametrization of the 2nd order polynomial for f
    G = torch.empty(list(f.shape) + [bx.shape[-1]] * 2, device=f.device)
    v = torch.empty(list(f.shape) + [bx.shape[-1]], device=f.device)

    # Apply separable cross-correlations

    # Pre-calculate quantities recommended in paper
    ab = torch.einsum("i,ij->ij", a, bx)
    abb = torch.einsum("ij,ik->ijk", ab, bx)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = correlate1d(c, abb[..., i, j], dim=0)

        v[..., i] = correlate1d(cf, ab[..., i], dim=0)

    # Pre-calculate quantities recommended in paper
    ab = torch.einsum("i,ij->ij", a, by)
    abb = torch.einsum("ij,ik->ijk", ab, by)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = correlate1d(G[..., i, j], abb[..., i, j], dim=1)

        v[..., i] = correlate1d(v[..., i], ab[..., i], dim=1)

    # Solve r for each pixel
    r = torch.linalg.solve(G, v)

    # Quadratic term
    A = torch.empty(list(f.shape) + [2, 2], device=f.device)
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    # Linear term
    B = torch.empty(list(f.shape) + [2], device=f.device)
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    # constant term
    C = r[..., 0]

    return A, B, C


@torch.jit.script
def flow_for_scale(f1, f2, c1, c2, d, sigma: float, sigma_flow: float, num_iter: int, model: str, mu: float):
    """Single iteration of Farneback's algorithm

    Args:
        f1 (Tensor): First image
        f2 (Tensor): Second image
        c1 (Tensor): Certainty of first image
        c2 (Tensor): Certainty of second image
        d (Tensor): Initial displacement field. Defaults to None.
        sigma (float): Polynomial expansion applicability Gaussian kernel sigma. Defaults to 4.0.
        sigma_flow (float): Applicability window Gaussian kernel sigma for polynomial matching. Defaults to 4.0.
        num_iter (int): Number of iterations to run. Defaults to 3.
        model (str): Optical flow parametrization to use. Options ["constant", "affine", "eight_param"]. Defaults to "constant".
        mu (float): Weighting term for usage of global parametrization. Defaults to using value recommended in Farneback's thesis.

    Returns:
        flow (Tensor): Optical flow field. d[i, j] is the (y, x) displacement for pixel (i, j)
    """
    # Calculate the polynomial expansion at each point in the images
    A1, B1, C1 = poly_exp(f1, c1, sigma)
    A2, B2, C2 = poly_exp(f2, c2, sigma)

    # Pixel coordinates of each point in the images
    fh, fw = f1.shape
    x = torch.stack(
        torch.meshgrid(torch.arange(fh, device=f1.device), torch.arange(fw, device=f1.device), indexing="xy"), dim=-1
    )

    # Set up applicability convolution window
    n_flow = int(4 * sigma_flow + 1)
    xw = torch.arange(-n_flow, n_flow + 1, device=f1.device)
    w = torch.exp(-(xw ** 2) / (2 * sigma_flow ** 2))

    # Evaluate warp parametrization model at pixel coordinates
    if model == "constant":
        S = torch.eye(2, device=f1.device)

    elif model in ("affine", "eight_param"):
        S = torch.empty(list(x.shape) + [6 if model == "affine" else 8], device=f1.device)

        S[..., 0, 0] = 1
        S[..., 0, 1] = x[..., 0]
        S[..., 0, 2] = x[..., 1]
        S[..., 0, 3] = 0
        S[..., 0, 4] = 0
        S[..., 0, 5] = 0

        S[..., 1, 0] = 0
        S[..., 1, 1] = 0
        S[..., 1, 2] = 0
        S[..., 1, 3] = 1
        S[..., 1, 4] = x[..., 0]
        S[..., 1, 5] = x[..., 1]

        if model == "eight_param":
            S[..., 0, 6] = x[..., 0] ** 2
            S[..., 0, 7] = x[..., 0] * x[..., 1]

            S[..., 1, 6] = x[..., 0] * x[..., 1]
            S[..., 1, 7] = x[..., 1] ** 2

    else:
        raise ValueError("Invalid parametrization model")

    S_T = S.swapaxes(-1, -2)

    # Iterate convolutions to estimate the optical flow
    for _ in range(num_iter):
        # Set d~ as displacement field fit to nearest pixel (and constrain to not being off image). Note we are setting
        # certainty to 0 for points that would have been off-image had we not constrained them
        x_ = x + d.long()

        # Constrain d~ to be on-image, and find points that would have been off-image
        x_2 = x_.clone()
        x_2[..., 0] = x_2[..., 0].clamp(0, x_2.shape[0] - 1)
        x_2[..., 1] = x_2[..., 1].clamp(0, x_2.shape[1] - 1)
        off_f = torch.any(x_ != x_2, dim=-1)
        x_ = x_2

        # Set certainty to 0 for off-image points
        c_ = c1[x_[..., 0], x_[..., 1]]
        c_[off_f] = 0

        # Calculate A and delB for each point, according to paper
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        A *= c_[..., None, None]  # recommendation in paper: add in certainty by applying to A and delB

        delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d[..., None])[..., 0]
        delB *= c_[..., None]  # recommendation in paper: add in certainty by applying to A and delB

        # Pre-calculate quantities recommended by paper
        A_T = A.swapaxes(-1, -2)
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ delB[..., None])[..., 0]

        # If mu is 0, it means the global/average parametrized warp should not be calculated, and the parametrization
        # should apply to the local calculations
        if mu == 0.0:
            # Apply separable cross-correlation to calculate linear equation
            # for each pixel: G*d = h
            G = correlate1d(ATA, w, dim=0)
            G = correlate1d(G, w, dim=1)

            h = correlate1d(ATb, w, dim=0)
            h = correlate1d(h, w, dim=1)

            sol = torch.linalg.pinv(G) @ h[..., None]
            d = (S @ sol).squeeze()

        # if mu is not 0, it should be used to regularize the least squares problem and "force" the background warp onto
        # uncertain pixels
        else:
            # Calculate global parametrized warp
            G_avg = torch.mean(ATA, dim=(0, 1))
            h_avg = torch.mean(ATb, dim=(0, 1))
            p_avg = torch.linalg.solve(G_avg, h_avg)
            d_avg = (S @ p_avg[..., None])[..., 0]

            # Default value for mu is to set mu to 1/2 the trace of G_avg
            if mu == -1:
                mu = 1 / 2 * torch.trace(G_avg)

            # Apply separable cross-correlation to calculate linear equation
            G = correlate1d(A_T @ A, w, dim=0)
            G = correlate1d(G, w, dim=1)

            h = correlate1d((A_T @ delB[..., None])[..., 0], w, dim=0)
            h = correlate1d(h, w, dim=1)

            # Refine estimate of displacement field
            sol = torch.linalg.pinv(G + mu * torch.eye(2, device=f1.device)) @ (h + mu * d_avg)[..., None]
            d = S @ sol

    return d


@torch.jit.script
def farneback(
    f1,
    f2,
    n_pyr: int = 1,
    sigma: float = 4.0,
    sigma_flow: float = 4.0,
    num_iter: int = 1,
    model: str = "constant",
    mu: float = 0.0,
):
    """Calculates optical flow with the algorithm described by Gunnar Farneback

    Args:
        f1 (Tensor): First image
        f2 (Tensor): Second image
        n_pyr (int, optional): Number of pyramid levels to iterate over. Defaults to 4.
        sigma (float, optional): Polynomial expansion applicability Gaussian kernel sigma. Defaults to 4.0.
        sigma_flow (float, optional): Applicability window Gaussian kernel sigma for polynomial matching. Defaults to 4.0.
        num_iter (int, optional): Number of iterations to run. Defaults to 3.
        model (str, optional): Optical flow parametrization to use. Options ["constant", "affine", "eight_param"]. Defaults to "constant".
        mu (int, optional): Weighting term for usage of global parametrization. Set to -1 to use value recommended in Farneback's thesis.

    Returns:
        flow (Tensor): Optical flow field. d[i, j] is the (y, x) displacement for pixel (i, j)
    """
    if f1.dim() == 4:
        f1, f2 = f1.squeeze().permute(1, 2, 0), f2.squeeze().permute(1, 2, 0)
    if f1.dim() == 3:
        f1, f2 = to_grayscale(f1), to_grayscale(f2)

    h, w = f1.shape
    c = torch.minimum(torch.ones((h, w)), 1 / 5 * torch.minimum(torch.arange(h)[:, None], torch.arange(w)))
    c = torch.minimum(c, 1 / 5 * torch.minimum(h - 1 - torch.arange(h)[:, None], w - 1 - torch.arange(w)))
    c = c.to(f1.device)

    # calculate optical flow using pyramids
    pyramid = torch.jit.annotate(List[List[torch.Tensor]], [])
    cur = [f1, f2, c]
    for _ in range(n_pyr):
        cur = [pyramid_reduce(a) for a in cur]
        pyramid.append(cur)

    flow = torch.empty(0, device=f1.device)
    for pyr1, pyr2, c in pyramid[::-1]:  # note: reversed(...) because we start with the smallest pyramid
        if len(flow) == 0:
            flow = torch.zeros(list(pyr1.shape) + [2], device=f1.device)
        else:
            # account for shapes not quite matching
            flow = 2 * pyramid_expand(flow)
            flow = flow[: pyr1.shape[0], : pyr2.shape[1]]
        flow = flow_for_scale(
            pyr1, pyr2, c, c, flow, sigma=sigma, sigma_flow=sigma_flow, num_iter=num_iter, model=model, mu=mu
        )

    return flow


def poly_exp_np(f, c, sigma):
    """
    Calculates the local polynomial expansion of a 2D signal, as described by Farneback

    Uses separable normalized correlation

    $f ~ x^T A x + B^T x + C$

    If f[i, j] and c[i, j] are the signal value and certainty of pixel (i, j) then
    A[i, j] is a 2x2 array representing the quadratic term of the polynomial, B[i, j]
    is a 2-element array representing the linear term, and C[i, j] is a scalar
    representing the constant term.

    Parameters
    ----------
    f
        Input signal
    c
        Certainty of signal
    sigma
        Standard deviation of applicability Gaussian kernel

    Returns
    -------
    A
        Quadratic term of polynomial expansion
    B
        Linear term of polynomial expansion
    C
        Constant term of polynomial expansion
    """
    # Calculate applicability kernel (1D because it is separable)
    n = int(4 * sigma + 1)
    x = np.arange(-n, n + 1, dtype=int)
    a = np.exp(-(x ** 2) / (2 * sigma ** 2))  # a: applicability kernel [n]

    # b: calculate b from the paper. Calculate separately for X and Y dimensions
    # [n, 6]
    bx = np.stack([np.ones(a.shape), x, np.ones(a.shape), x ** 2, np.ones(a.shape), x], axis=-1)
    by = np.stack(
        [
            np.ones(a.shape),
            np.ones(a.shape),
            x,
            np.ones(a.shape),
            x ** 2,
            x,
        ],
        axis=-1,
    )

    # Pre-calculate product of certainty and signal
    cf = c * f

    # G and v are used to calculate "r" from the paper: v = G*r
    # r is the parametrization of the 2nd order polynomial for f
    G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    # Apply separable cross-correlations

    # Pre-calculate quantities recommended in paper
    ab = np.einsum("i,ij->ij", a, bx)
    abb = np.einsum("ij,ik->ijk", ab, bx)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(c, abb[..., i, j], axis=0, mode="constant", cval=0)

        v[..., i] = scipy.ndimage.correlate1d(cf, ab[..., i], axis=0, mode="constant", cval=0)

    # Pre-calculate quantities recommended in paper
    ab = np.einsum("i,ij->ij", a, by)
    abb = np.einsum("ij,ik->ijk", ab, by)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0)

        v[..., i] = scipy.ndimage.correlate1d(v[..., i], ab[..., i], axis=1, mode="constant", cval=0)

    # Solve r for each pixel
    r = np.linalg.solve(G, v)

    # Quadratic term
    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    # Linear term
    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    # constant term
    C = r[..., 0]

    # b: [n, n, 6]
    # r: [f, f, 6]
    # f: [f, f]
    # e = b*r - f

    return A, B, C


def flow_for_scale_np(f1, f2, sigma, c1, c2, sigma_flow, num_iter=1, d=None, model="constant", mu=-1):
    """
    Calculates optical flow with an algorithm described by Gunnar Farneback

    Parameters
    ----------
    f1
        First image
    f2
        Second image
    sigma
        Polynomial expansion applicability Gaussian kernel sigma
    c1
        Certainty of first image
    c2
        Certainty of second image
    sigma_flow
        Applicability window Gaussian kernel sigma for polynomial matching
    num_iter
        Number of iterations to run (defaults to 1)
    d: (optional)
        Initial displacement field
    p: (optional)
        Initial global displacement model parameters
    model: ['constant', 'affine', 'eight_param']
        Optical flow parametrization to use
    mu: (optional)
        Weighting term for usage of global parametrization. Defaults to
        using value recommended in Farneback's thesis

    Returns
    -------
    d
        Optical flow field. d[i, j] is the (y, x) displacement for pixel (i, j)
    """

    # TODO: add initial warp parameters as optional input?

    # Calculate the polynomial expansion at each point in the images
    A1, B1, C1 = poly_exp_np(f1, c1, sigma)
    A2, B2, C2 = poly_exp_np(f2, c2, sigma)

    # Pixel coordinates of each point in the images
    x = np.stack(np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])), axis=-1).astype(int)

    # Initialize displacement field
    if d is None:
        d = np.zeros(list(f1.shape) + [2])

    # Set up applicability convolution window
    n_flow = int(4 * sigma_flow + 1)
    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-(xw ** 2) / (2 * sigma_flow ** 2))

    # Evaluate warp parametrization model at pixel coordinates
    if model == "constant":
        S = np.eye(2)

    elif model in ("affine", "eight_param"):
        S = np.empty(list(x.shape) + [6 if model == "affine" else 8])

        S[..., 0, 0] = 1
        S[..., 0, 1] = x[..., 0]
        S[..., 0, 2] = x[..., 1]
        S[..., 0, 3] = 0
        S[..., 0, 4] = 0
        S[..., 0, 5] = 0

        S[..., 1, 0] = 0
        S[..., 1, 1] = 0
        S[..., 1, 2] = 0
        S[..., 1, 3] = 1
        S[..., 1, 4] = x[..., 0]
        S[..., 1, 5] = x[..., 1]

        if model == "eight_param":
            S[..., 0, 6] = x[..., 0] ** 2
            S[..., 0, 7] = x[..., 0] * x[..., 1]

            S[..., 1, 6] = x[..., 0] * x[..., 1]
            S[..., 1, 7] = x[..., 1] ** 2

    else:
        raise ValueError("Invalid parametrization model")

    S_T = S.swapaxes(-1, -2)

    # Iterate convolutions to estimate the optical flow
    for _ in range(num_iter):
        # Set d~ as displacement field fit to nearest pixel (and constrain to not
        # being off image). Note we are setting certainty to 0 for points that
        # would have been off-image had we not constrained them
        d_ = d.astype(int)
        x_ = x + d_

        # Constrain d~ to be on-image, and find points that would have
        # been off-image
        x_2 = x_.copy()
        x_2[..., 0] = np.clip(x_2[..., 0], 0, x_2.shape[0] - 1)
        x_2[..., 1] = np.clip(x_2[..., 1], 0, x_2.shape[1] - 1)
        off_f = np.any(x_ != x_2, axis=-1)
        x_ = x_2

        # Set certainty to 0 for off-image points
        c_ = c1[x_[..., 0], x_[..., 1]]
        c_[off_f] = 0

        # Calculate A and delB for each point, according to paper
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        A *= c_[..., None, None]  # recommendation in paper: add in certainty by applying to A and delB

        delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
        delB *= c_[..., None]  # recommendation in paper: add in certainty by applying to A and delB

        # Pre-calculate quantities recommended by paper
        A_T = A.swapaxes(-1, -2)
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ delB[..., None])[..., 0]

        # If mu is 0, it means the global/average parametrized warp should not be
        # calculated, and the parametrization should apply to the local calculations
        if mu == 0:
            # Apply separable cross-correlation to calculate linear equation
            # for each pixel: G*d = h
            G = scipy.ndimage.correlate1d(ATA, w, axis=0, mode="constant", cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)

            h = scipy.ndimage.correlate1d(ATb, w, axis=0, mode="constant", cval=0)
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)

            # print(np.linalg.pinv(G).shape, h[..., None].shape)
            sol = np.linalg.pinv(G) @ h[..., None]
            # print(S.shape, sol.shape)
            d = (S @ sol).squeeze()

        # if mu is not 0, it should be used to regularize the least squares problem
        # and "force" the background warp onto uncertain pixels
        else:
            # Calculate global parametrized warp
            G_avg = np.mean(ATA, axis=(0, 1))
            h_avg = np.mean(ATb, axis=(0, 1))
            p_avg = np.linalg.solve(G_avg, h_avg)
            d_avg = (S @ p_avg[..., None])[..., 0]

            # Default value for mu is to set mu to 1/2 the trace of G_avg
            if mu == -1:
                mu = 1 / 2 * np.trace(G_avg)

            # Apply separable cross-correlation to calculate linear equation
            G = scipy.ndimage.correlate1d(A_T @ A, w, axis=0, mode="constant", cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)

            h = scipy.ndimage.correlate1d((A_T @ delB[..., None])[..., 0], w, axis=0, mode="constant", cval=0)
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)

            # Refine estimate of displacement field
            # print((G + mu * np.eye(2)).shape)
            # print(np.linalg.pinv(G + mu * np.eye(2)).shape, (h + mu * d_avg)[..., None].shape)
            sol = np.linalg.pinv(G + mu * np.eye(2)) @ (h + mu * d_avg)[..., None]
            # print(S.shape, sol.shape)
            d = (S @ sol).squeeze()

    # TODO: return global displacement parameters and/or global displacement if mu != 0

    return d


def farneback_np(f1, f2, n_pyr, sigma, sigma_flow, num_iter, model, mu):
    c1 = np.minimum(1, 1 / 5 * np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
    c1 = np.minimum(
        c1,
        1 / 5 * np.minimum(f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None], f1.shape[1] - 1 - np.arange(f1.shape[1])),
    )
    c2 = c1
    n_pyr = 6
    d = None
    for pyr1, pyr2, c1_, c2_ in reversed(
        list(zip(*list(map(partial(skimage.transform.pyramid_gaussian, max_layer=n_pyr), [f1, f2, c1, c2]))))
    ):
        if d is not None:
            # TODO: account for shapes not quite matching
            d = 2 * skimage.transform.pyramid_expand(d, channel_axis=-1)
            d = d[: pyr1.shape[0], : pyr2.shape[1]]

        d = flow_for_scale_np(
            pyr1,
            pyr2,
            c1=c1_,
            c2=c2_,
            d=d,
            sigma=sigma,
            sigma_flow=sigma_flow,
            num_iter=num_iter,
            model=model,
            mu=mu,
        )
    return d


if __name__ == "__main__":
    from functools import partial
    from time import time

    import cv2
    import matplotlib
    import skimage
    import torch
    import torchvision as tv

    from farneback import to_grayscale

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def report(flow, ext):
        print(flow[..., 0].min().item(), flow[..., 0].mean().item(), flow[..., 0].max().item())
        print(flow[..., 1].min().item(), flow[..., 1].mean().item(), flow[..., 1].max().item())
        flow = torch.stack(cart2pol(flow[..., 0], flow[..., 1]))
        print(flow[0].min().item(), flow[0].mean().item(), flow[0].max().item())
        print(flow[1].min().item(), flow[1].mean().item(), flow[1].max().item())
        print()

        hsv = np.ones((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
        hsv[..., 0] = normalize(flow[1]).cpu().numpy()
        hsv[..., 2] = normalize(flow[0]).cpu().numpy()
        hsv = (hsv * 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(f"output/flow_test_{ext}.pdf")
        plt.close()

    video, audio, info = tv.io.read_video("/home/hans/datasets/audiovisual/maua/airhead2.mp4")
    video = tv.transforms.functional.resize(video.permute(0, 3, 1, 2), 256).permute(0, 2, 3, 1)

    f1 = to_grayscale(video[225]).numpy()
    f2 = to_grayscale(video[226]).numpy()
    print(f1.min(), f1.mean(), f1.max(), f1.shape)
    print()

    t = time()
    flow3 = farneback_np(
        f1,
        f2,
        n_pyr=6,
        sigma=4.0,
        sigma_flow=4.0,
        num_iter=10,
        model="constant",
        mu=0,
    )
    print(time() - t)
    report(torch.from_numpy(flow3), "npy")

    t = time()
    flow = farneback(
        torch.from_numpy(f1).cpu(),
        torch.from_numpy(f2).cpu(),
        n_pyr=6,
        sigma=4.0,
        sigma_flow=4.0,
        num_iter=10,
        model="constant",
        mu=0,
    )
    print(time() - t)
    report(flow, "pth")

    t = time()
    flow2 = cv2.calcOpticalFlowFarneback(
        f1.astype(np.uint8),
        f2.astype(np.uint8),
        None,
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    print(time() - t)
    report(torch.from_numpy(flow2), "cv2")
