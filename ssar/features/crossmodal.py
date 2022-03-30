from typing import Optional

import numpy as np
import torch
from torchsort import soft_rank


def spearman_correlation(x, y, regularization: str = "l2", regularization_strength: float = 0.01):
    x = soft_rank(x.unsqueeze(0), regularization, regularization_strength).squeeze() / x.shape[-1]
    return correlation(x, y)


@torch.jit.script
def correlation(x, y):
    pred_n = x - x.mean()
    target_n = y - y.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return pred_n @ target_n


@torch.jit.script
def autocorrelation_correlation(X, Y):
    X = X - X.mean()
    X = X / X.norm(p=2, dim=1, keepdim=True)
    Xac = X @ X.T

    Y = Y - Y.mean()
    Y = Y / Y.norm(p=2, dim=1, keepdim=True)
    Yac = Y @ Y.T

    T = len(X)
    triuy, triux = torch.triu_indices(T, T, offset=1).unbind(0)
    similarity = correlation(Xac[triuy, triux], Yac[triuy, triux])
    return similarity


@torch.jit.script
def rv(Ms, modified: bool = True):
    """
    This function computes the RV matrix correlation coefficients between pairs of arrays. The number and order of
    objects (rows) for the two arrays must match. The number of variables in each array may vary. The RV2 coefficient is
    a modified version of the RV coefficient with values -1 <= RV2 <= 1. RV2 is independent of object and variable size.

    Reference: `Matrix correlations for high-dimensional data - the modified RV-coefficient`_
    .. _Matrix correlations for high-dimensional data - the modified RV-coefficient: https://academic.oup.com/bioinformatics/article/25/3/401/244239
    .. _Hoggorm implementation: https://github.com/olivertomic/hoggorm
    """

    Mss = []
    for M in Ms:
        M = M - M.mean()
        MMt = M @ M.T
        if modified:
            MMt = MMt - torch.diag(torch.diag(MMt))
        Mss.append(MMt)

    C = torch.eye(len(Ms), dtype=Ms[0].dtype, device=Ms[0].device)
    for idx in torch.triu_indices(len(Ms), len(Ms), offset=1):
        Rv = torch.trace(Mss[idx[0]].T @ Mss[idx[1]]) / torch.sqrt(
            torch.trace(Mss[idx[0]].T @ Mss[idx[0]]) * torch.trace(Mss[idx[1]].T @ Mss[idx[1]])
        )
        C[idx[0], idx[1]] = C[idx[1], idx[0]] = Rv

    return C


@torch.jit.script
def matrix_rank(X, tol: float = 1e-8) -> int:
    return (torch.linalg.svdvals(X) > tol).sum().item()


@torch.jit.script
def smi(
    X,
    Y,
    n_components: Optional[int] = None,
    projection: str = "orthogonal",
    significance: bool = False,
    B: int = 10_000,
):
    """
    Similarity of Matrices Index (SMI)

    A similarity index for comparing coupled data matrices.
    A two-step process starts with extraction of stable subspaces using Principal Component Analysis or some other
    method yielding two orthonormal bases. These bases are compared using Orthogonal Projection (OP / ordinary least
    squares) or Procrustes Rotation (PR). The result is a similarity measure that can be adjusted to various data sets
    and contexts and which includes explorative plotting and permutation based testing of matrix subspace equality.

    Reference: `A similarity index for comparing coupled matrices`_
    .. _A similarity index for comparing coupled matrices: https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.3049
    .. _Hoggorm implementation: https://github.com/olivertomic/hoggorm

    significance=True:
        Significance estimation for Similarity of Matrices Index (SMI)

        For each combination of components significance is estimated by sampling from a null distribution of no
        similarity, i.e. when the rows of one matrix is permuted B times and corresponding SMI values are computed. If
        the vector replicates is included, replicates will be kept together through permutations.
    """
    X = X - X.mean()
    Y = Y - Y.mean()

    rankX = matrix_rank(X) if n_components is None else n_components
    rankY = matrix_rank(Y) if n_components is None else n_components

    UX, _, _ = torch.linalg.svd(X - X.mean(0))
    UY, _, _ = torch.linalg.svd(Y - Y.mean(0))

    m = torch.empty(())  # please torch.jit

    # Compute SMI values
    if projection == "orthogonal":
        m = (
            torch.arange(rankX, device=X.device)[:, None]
            .tile(1, rankX)
            .min(torch.arange(rankY, device=X.device)[None, :].tile(rankY, 1))
            .add(1)
            .reshape(rankX, rankY)
        )
        smi = (UX[:, :rankX].T @ UY[:, :rankY]).square().cumsum(1).cumsum(0) / m

    else:  # procrustes
        smi = torch.zeros((rankX, rankY))
        TU = UX[:, :rankX].T @ UY[:, :rankY]
        for p in range(rankX):
            for q in range(rankY):
                smi[p, q] = torch.linalg.svdvals(TU[: p + 1, : q + 1]).mean().square()

    # Recover wrong calculations (due to numerics)
    smi[smi > 1] = 1
    smi[smi < 0] = 0

    P = torch.zeros((rankX, rankY))

    if significance:
        BUX = UX.clone()

        if projection == "orthogonal":
            for __ in range(B):
                BUX = BUX[torch.randperm(len(BUX))]
                smiB = (BUX[:, :rankX].T @ UY[:, :rankY]).square().cumsum(1).cumsum(0) / m
                P[smi > torch.maximum(smiB, 1 - smiB)] += 1  # Increase P-value if non-significant permutation

        else:  # procrustes
            for __ in range(B):
                BUX = BUX[torch.randperm(len(BUX))]
                smiB = torch.zeros((rankX, rankY))
                TU = BUX[:, :rankX].T @ UY[:, :rankY]
                for p in range(rankX):
                    for q in range(rankY):
                        smiB[p, q] = torch.linalg.svdvals(TU[: p + 1, : q + 1]).mean().square()
                P[smi > torch.maximum(smiB, 1 - smiB)] += 1  # Increase P-value if non-significant permutation

    return smi, P / B


if __name__ == "__main__":
    X = np.random.rand(100, 300)
    X = X - X.mean()
    U, s, V = np.linalg.svd(X, 0)
    Y = np.dot(np.dot(np.delete(U, 2, 1), np.diag(np.delete(s, 2))), np.delete(V, 2, 0))
    X, Y = torch.from_numpy(X).float().cuda(), torch.from_numpy(Y).float().cuda()
    Y2 = torch.rand(100, 300, device="cuda")

    smiOP, sigOP = smi(X, Y, n_components=10, significance=True, B=100)
    smiPR, sigPR = smi(X, Y, n_components=10, projection="procrustes", significance=True, B=100)
    smiOP, sigOP = smi(X, Y2, n_components=10, significance=True, B=100)
    smiPR, sigPR = smi(X, Y2, n_components=10, projection="procrustes", significance=True, B=100)
