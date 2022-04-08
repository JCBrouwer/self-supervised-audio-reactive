import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg as la
from scipy import signal
from scipy import special as ss

from .transition import transition
from .unroll import variable_unroll_matrix, variable_unroll_matrix_sequential


def init_leg_t(N, dt=1.0, discretization="bilinear"):
    A, B = transition("lmu", N)
    C = np.ones((1, N))
    D = np.zeros((1,))

    # dt, discretization options
    A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
    B = B.squeeze(-1)

    vals = np.arange(0.0, 1.0, dt)
    E = ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T

    return torch.FloatTensor(A), torch.FloatTensor(B), torch.FloatTensor(E)


def encode_leg_t(fs, A, B):
    fs = fs.unsqueeze(-1)
    u = fs * B  # (length, ..., N)
    c = torch.zeros(u.shape[1:], device=fs.device)
    cs = []
    for f in fs:
        c = F.linear(c, A) + B * f
        cs.append(c)
    return torch.stack(cs, dim=0)


def init_leg_s(N, max_length=1024, measure="legs", discretization="bilinear"):
    A, B = transition(measure, N)
    B = B.squeeze(-1)
    A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
    B_stacked = np.empty((max_length, N), dtype=B.dtype)
    for t in range(1, max_length + 1):
        At = A / t
        Bt = B / t
        if discretization == "forward":
            A_stacked[t - 1] = np.eye(N) + At
            B_stacked[t - 1] = Bt
        elif discretization == "backward":
            A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
            B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
        elif discretization == "bilinear":
            A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
            B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
        else:  # ZOH
            A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
            B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)

    vals = np.linspace(0.0, 1.0, max_length)
    E = (B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T

    return torch.FloatTensor(A_stacked), torch.FloatTensor(B_stacked), torch.FloatTensor(E)


def encode_leg_s(fs, A, B, fast=False):
    L = fs.shape[0]

    fs = fs.unsqueeze(-1)
    u = torch.transpose(fs, 0, -2)
    u = u * B[:L]
    u = torch.transpose(u, 0, -2)  # (length, ..., N)

    if fast:
        result = variable_unroll_matrix(A[:L], u)
    else:
        result = variable_unroll_matrix_sequential(A[:L], u)
    return result


class HiPPOTimeseries(nn.Module):
    """Efficient timeseries parameterization using 'High-order Polynomial Projection Operators'"""

    def __init__(self, f, N=256, invariance="s") -> None:
        super().__init__()
        if invariance == "t":
            A, B, E = init_leg_t(N, dt=1 / len(f))
            c = encode_leg_t(f, A.to(f), B.to(f))
        elif invariance == "s":
            A, B, E = init_leg_s(N, max_length=len(f))
            c = encode_leg_s(f, A.to(f), B.to(f))
        self.register_buffer("E", E)
        self.c = nn.Parameter(c)

    def forward(self):
        return (self.E @ self.c.unsqueeze(-1)).squeeze(-1)[-1]


if __name__ == "__main__":
    from ssar.features.audio import onsets
    from ssar.features.generate import get_random_audio

    for i in range(1, 11):
        while True:
            try:
                audio, sr = get_random_audio()
                break
            except:
                pass
        f = onsets(audio, sr).squeeze()

        f_train_t = HiPPOTimeseries(f, invariance="t").cuda()
        f_init_t = f_train_t()
        opt_t = torch.optim.Adam(f_train_t.parameters(), lr=1e-5)

        f_train_s = HiPPOTimeseries(f, invariance="s").cuda()
        f_init_s = f_train_s()
        opt_s = torch.optim.Adam(f_train_s.parameters(), lr=1e-5)

        for j in range(10000):
            opt_t.zero_grad()
            loss_t = F.mse_loss(f, f_train_t())
            loss_t.backward()
            opt_t.step()

            opt_s.zero_grad()
            loss_s = F.mse_loss(f, f_train_s())
            loss_s.backward()
            opt_s.step()

            if i == 1:
                print(f"t: {loss_t.item():.4f}".ljust(10), f"s: {loss_s.item():.4f}")

        vals = np.linspace(0.0, 1.0, len(f))

        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        for j, (title, fa) in enumerate(
            [("init t", f_init_t), ("init s", f_init_s), ("train t", f_train_t()), ("train s", f_train_s())]
        ):
            ax.T.flatten()[j].plot(vals, f.detach().cpu(), "k--", linewidth=0.5)
            ax.T.flatten()[j].plot(
                vals[: len(f)], fa.detach().cpu()[: len(f)], alpha=0.5, color="orange" if "s" in title else "blue"
            )
            ax.T.flatten()[j].set_xlabel("Time (normalized)", labelpad=-10)
            ax.T.flatten()[j].set_xticks([0, 1])
            ax.T.flatten()[j].set_title(title)
        plt.tight_layout()
        plt.savefig(f"output/hippo-onsets{i}_param.png")
        plt.close()
