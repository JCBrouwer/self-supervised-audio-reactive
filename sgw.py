#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:39:12 2019

@author: vayer
"""

import torch


def sgw_gpu(
    xs: torch.Tensor, xt: torch.Tensor, device: torch.device, nproj: int = 200, P: torch.Tensor = torch.tensor([])
):
    """Returns SGW between xs and xt eq (4) in [1]. Only implemented with the 0 padding operator Delta
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignore if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix. If None creates a new projection matrix
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    Example
    ----------
    import numpy as np
    import torch
    from sgw_pytorch import sgw

    n_samples=300
    Xs=np.random.rand(n_samples,2)
    Xt=np.random.rand(n_samples,1)
    xs=torch.from_numpy(Xs).to(torch.float32)
    xt=torch.from_numpy(Xt).to(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P=np.random.randn(2,500)
    sgw_gpu(xs,xt,device,P=torch.from_numpy(P).to(torch.float32))
    """
    xsp, xtp = sink_(xs, xt, device, nproj, P)
    d = gromov_1d(xsp, xtp)
    return d


def _cost(xsp: torch.Tensor, xtp: torch.Tensor):
    """Returns the GM cost eq (3) in [1]
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the target
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    xs = xsp
    xt = xtp

    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2

    xt2 = xt * xt
    xt3 = xt2 * xt
    xt4 = xt2 * xt2

    X = torch.sum(xs, 0)
    X2 = torch.sum(xs2, 0)
    X3 = torch.sum(xs3, 0)
    X4 = torch.sum(xs4, 0)

    Y = torch.sum(xt, 0)
    Y2 = torch.sum(xt2, 0)
    Y3 = torch.sum(xt3, 0)
    Y4 = torch.sum(xt4, 0)

    xxyy_ = torch.sum((xs2) * (xt2), 0)
    xxy_ = torch.sum((xs2) * (xt), 0)
    xyy_ = torch.sum((xs) * (xt2), 0)
    xy_ = torch.sum((xs) * (xt), 0)

    n = xs.shape[0]

    C2 = 2 * X2 * Y2 + 2 * (n * xxyy_ - 2 * Y * xxy_ - 2 * X * xyy_ + 2 * xy_ * xy_)

    power4_x = 2 * n * X4 - 8 * X3 * X + 6 * X2 * X2
    power4_y = 2 * n * Y4 - 8 * Y3 * Y + 6 * Y2 * Y2

    C = (1 / (n ** 2)) * (power4_x + power4_y - 2 * C2)

    return C


def gromov_1d(xs: torch.Tensor, xt: torch.Tensor):
    """Solves the Gromov in 1D (eq (2) in [1] for each proj
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the target
    fast: use the O(nlog(n)) cost or not
    Returns
    -------
    toreturn : tensor, shape (n_proj,1)
           The SGW cost for each proj
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    xs2, i_s = torch.sort(xs, dim=0)
    xt_asc, i_t = torch.sort(xt, dim=0)
    xt_desc, i_t = torch.sort(xt, dim=0, descending=True)
    l1 = _cost(xs2, xt_asc)
    l2 = _cost(xs2, xt_desc)
    toreturn = torch.mean(torch.min(l1, l2))
    return toreturn


def sink_(
    xs: torch.Tensor, xt: torch.Tensor, device: torch.device, nproj: int = 200, P: torch.Tensor = torch.tensor([])
):  # Delta operator (here just padding)
    """Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections.
    Only implemented with the 0 padding Delta=Delta_pad operator (see [1])
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignored if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix
    Returns
    -------
    xsp : tensor, shape (n,n_proj)
           Projected source samples
    xtp : tensor, shape (n,n_proj)
           Projected target samples
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    dim_d = xs.shape[1]
    dim_p = xt.shape[1]

    if dim_d < dim_p:
        random_projection_dim = dim_p
        xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p - dim_d)).to(device)), dim=1)
        xt2 = xt
    else:
        random_projection_dim = dim_d
        xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d - dim_p)).to(device)), dim=1)
        xs2 = xs

    if len(P) == 0:
        P = torch.randn(random_projection_dim, nproj)
    p = P / torch.sqrt(torch.sum(P ** 2, 0, True))

    xsp = torch.matmul(xs2, p.to(device))
    xtp = torch.matmul(xt2, p.to(device))

    return xsp, xtp
