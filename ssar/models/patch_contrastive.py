from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy


class PatchSampler1d(torch.jit.ScriptModule):
    def __init__(self, n_patches: int, patch_len: int):
        super().__init__()
        self.n_patches, self.patch_len = n_patches, patch_len

    @torch.jit.script_method
    def forward(self, sequences: List[Tensor], target: Tensor = None) -> Tuple[List[Tensor], Tensor]:
        if target is None:
            target = torch.zeros_like(sequences[0])
        seq_batch_patches = torch.jit.annotate(List[List[Tensor]], [[] for _ in sequences])
        tar_batch_patches = torch.jit.annotate(List[Tensor], [])
        for b in range(target.shape[0]):
            seq_patches = torch.jit.annotate(List[List[Tensor]], [[] for _ in sequences])
            tar_patches = torch.jit.annotate(List[Tensor], [])
            for _ in range(self.n_patches):
                start_idx = torch.randint(0, target.shape[1] - self.patch_len, (1,)).item()
                for s, seq in enumerate(sequences):
                    seq_patches[s].append(seq[b, start_idx : start_idx + self.patch_len])
                tar_patches.append(target[b, start_idx : start_idx + self.patch_len])
            for s, seq in enumerate(seq_patches):
                seq_batch_patches[s].append(torch.stack(seq))
            tar_batch_patches.append(torch.stack(tar_patches))
        return [torch.stack(seq_batch) for seq_batch in seq_batch_patches], torch.stack(tar_batch_patches)  # B, P, S, F


class PatchSampler2d(torch.jit.ScriptModule):
    def __init__(self, patch_size: int, n_channels: int = 32, patch_scaling: float = 0.5):
        super().__init__()
        self.patch_size, self.n_channels, self.patch_scaling = patch_size, n_channels, patch_scaling

    @torch.jit.script_method
    def forward(self, input):
        P, S, C, H, W = input.shape
        max_size = max(W, H)
        min_size = min(W, H, self.patch_size)
        patches = torch.jit.annotate(List[Tensor], [])
        for p in range(P):
            size = int(torch.rand([1]) ** self.patch_scaling * (max_size - min_size) + min_size)
            pixel = torch.randint(0, (W - size + 1) * (H - size + 1), ())
            offsety, offsetx = torch.div(pixel, (W - size + 1), rounding_mode="floor"), pixel % (W - size + 1)
            channels = torch.randperm(C)[: self.n_channels]
            patch = input[p, :, channels, offsety : offsety + size, offsetx : offsetx + size]
            patches.append(F.adaptive_avg_pool2d(patch, min_size).flatten(1, 3))
        return torch.stack(patches)  # P, S, F


def patch_nce_loss(f_q, f_k, tau: float = 0.07):
    """
    Input: f_q (B, S, F) sampled feature patches
    Input: f_k (B, S, F) sampled feature patches
    Input: tau is the temperature used in PatchNCE loss.
    Output: PatchNCE loss
    """
    B, S, C = f_q.shape
    f_q, f_k = f_q.permute(0, 2, 1), f_k.permute(0, 2, 1)

    # calculate v * v+: BxSx1
    l_pos = (f_k * f_q).sum(dim=1)[:, :, None]

    # calculate v * v-: BxSxS
    l_neg = torch.bmm(f_q.transpose(1, 2), f_k)

    # The diagonal entries are not negatives. Remove them.
    identity_matrix = torch.eye(S, device=f_q.device, dtype=torch.bool)[None, :, :]
    l_neg.masked_fill_(identity_matrix, -float("inf"))

    # calculate logits: (B)x(S)x(S+1)
    logits = torch.cat((l_pos, l_neg), dim=2) / tau

    # return PatchNCE loss
    predictions = logits.flatten(0, 1)
    targets = torch.zeros(B * S, device=f_q.device, dtype=torch.long)
    return cross_entropy(predictions, targets)


class LazyCombinationsPatchContrastor(torch.jit.ScriptModule):
    def __init__(self, latent_dim, sequences):
        super().__init__()
        self.latent_dim = latent_dim
        B, P = sequences[0].shape[0], sequences[0].shape[1]
        self.feature_heads = nn.ModuleList(
            [
                ContrastiveHead(B, P, torch.prod(torch.tensor(seq.shape[2:])).item(), self.latent_dim)
                for seq in sequences
            ]
        ).to(sequences[0].device)

    @torch.jit.script_method
    def forward(self, sequences: List[Tensor]):

        embeddings = torch.jit.annotate(List[Tensor], [])
        for idx, feature_head in enumerate(self.feature_heads):
            embeddings.append(feature_head(sequences[idx]))

        loss = 0
        for e1, embed1 in enumerate(embeddings):
            for e2, embed2 in enumerate(embeddings):
                if e1 == e2:
                    continue
                loss += patch_nce_loss(embed1, embed2)

        return loss


class CombinationsPatchContrastor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.contrastor = None
        self.latent_dim = latent_dim

    def forward(self, sequences):
        if self.contrastor is None:
            self.contrastor = LazyPatchContrastor(self.latent_dim, sequences)
        return self.contrastor(sequences)


class LazyPatchContrastor(torch.jit.ScriptModule):
    def __init__(self, latent_dim, sequences, target):
        super().__init__()
        self.latent_dim = latent_dim
        B, P, S, target_channels = target.shape
        self.feature_heads = nn.ModuleList(
            [
                ContrastiveHead(B, P, torch.prod(torch.tensor(seq.shape[2:])).item(), self.latent_dim)
                for seq in sequences
            ]
        ).to(target.device)
        self.target_head = ContrastiveHead(B, P, S * target_channels, self.latent_dim).to(target.device)

    @torch.jit.script_method
    def forward(self, sequences: List[Tensor], target: Tensor):
        target_embedding = self.target_head(target)

        loss = 0
        for idx, feature_head in enumerate(self.feature_heads):
            loss += patch_nce_loss(feature_head(sequences[idx]), target_embedding)

        return loss


class PatchContrastor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.contrastor = None
        self.latent_dim = latent_dim

    def forward(self, sequences, target):
        if self.contrastor is None:
            self.contrastor = LazyPatchContrastor(self.latent_dim, sequences, target)
        return self.contrastor(sequences, target)


class ContrastiveHead(torch.jit.ScriptModule):
    def __init__(self, batch_size: int, n_patches: int, in_nc: int, out_nc: int):
        super().__init__()
        self.mlp = nn.Sequential(
            Reshape(batch_size * n_patches, -1),
            nn.Linear(in_nc, out_nc),
            nn.LeakyReLU(0.2),
            nn.Linear(out_nc, out_nc),
            Reshape(batch_size, n_patches, -1),
        )

    @torch.jit.script_method
    def forward(self, x):
        return self.mlp(x)


class Reshape(torch.jit.ScriptModule):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    @torch.jit.script_method
    def forward(self, x):
        return x.reshape(*self.shape)
