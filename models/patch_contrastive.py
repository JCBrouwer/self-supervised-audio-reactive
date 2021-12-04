import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.nn import functional as F


class PatchSampler1d(nn.Module):
    def __init__(self, n_patches: int, patch_len: int):
        super().__init__()
        self.n_patches, self.patch_len = n_patches, patch_len

    def forward(self, x, y):
        x_batch_patches, y_batch_patches = [], []
        for b in range(x.shape[0]):
            x_patches, y_patches = [], []
            for _ in range(self.n_patches):
                start_idx = torch.randint(0, x.shape[1] - self.patch_len, (1,), device=x.device)
                x_patches.append(x[b, start_idx : start_idx + self.patch_len])
                y_patches.append(y[b, start_idx : start_idx + self.patch_len])
            x_batch_patches.append(torch.stack(x_patches))
            y_batch_patches.append(torch.stack(y_patches))
        return torch.stack(x_batch_patches), torch.stack(y_batch_patches)  # B, P, S, F


class PatchSampler2d(nn.Module):
    def __init__(self, patch_size: int, n_channels: int = 32, patch_scaling: float = 0.5):
        super().__init__()
        self.patch_size, self.n_channels, self.patch_scaling = patch_size, n_channels, patch_scaling

    def forward(self, input):
        P, S, C, H, W = input.shape
        max_size = max(W, H)
        min_size = min(W, H, self.patch_size)
        cutouts = []
        for p in range(P):
            size = int(torch.rand([1]) ** self.patch_scaling * (max_size - min_size) + min_size)
            pixel = torch.randint(0, (W - size + 1) * (H - size + 1), ())
            offsety, offsetx = torch.div(pixel, (W - size + 1), rounding_mode="floor"), pixel % (W - size + 1)
            channels = torch.randperm(C)[: self.n_channels]
            cutout = input[p, :, channels, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, min_size).flatten(1, 3))
        return torch.stack(cutouts)  # P, S, F


def patch_nce_loss(f_q, f_k, tau=0.07):
    """
    Input: f_q (BxSxC) and sampled features from H(G_enc(x))
    Input: f_k (BxSxC) are sampled features from H(G_enc(G(x))
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


class PatchContrastor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_heads = None
        self.target_head = None

    def initialize(self, sequences, targets):
        B, P, S, target_channels = targets.shape
        self.feature_heads = nn.ModuleList(
            [ContrastiveHead(B, P, np.prod(seq.shape[2:]), self.latent_dim) for seq in sequences]
        ).to(targets.device)
        self.target_head = ContrastiveHead(B, P, S * target_channels, self.latent_dim).to(targets.device)

    def forward(self, sequences, targets):
        if self.feature_heads is None:
            self.initialize(sequences, targets)

        target_embedding = self.target_head(targets)

        loss = 0
        for s, seq in enumerate(sequences):
            loss += patch_nce_loss(self.feature_heads[s](seq), target_embedding)

        return loss


class ContrastiveHead(nn.Module):
    def __init__(self, batch_size, n_patches, in_nc, out_nc):
        super().__init__()
        self.mlp = nn.Sequential(
            Reshape(batch_size * n_patches, -1),
            nn.Linear(in_nc, out_nc),
            nn.LeakyReLU(0.2),
            nn.Linear(out_nc, out_nc),
            Reshape(batch_size, n_patches, -1),
        )

    def forward(self, x):
        return self.mlp(x)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)
