import os
import sys
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio
from scipy import stats

from usrlt import CausalCNNEncoderClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_checkpoint = "cache/encoder.pt"


def sqrtm(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def frechet_distance(feats1, feats2, eps=1e-6):
    mu1, sigma1 = torch.mean(feats1, axis=0), torch.cov(feats1)
    mu2, sigma2 = torch.mean(feats2, axis=0), torch.cov(feats2)

    # Product might be almost singular
    covmean = sqrtm(sigma1 @ sigma2)
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if torch.is_complex(covmean):
        if not torch.allclose(torch.diagonal(covmean).imag, 0, atol=1e-3):
            m = torch.max(torch.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m:.3f}")
        covmean = covmean.real

    diff = mu1 - mu2

    return diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)


def calculate_fcd(dataloader, model):
    encoder = joblib.load(encoder_checkpoint).encoder.eval().to(device)

    real_feats, fake_feats = [], []
    for f, l in dataloader:
        l = l.to(device).flatten(2).transpose(1, 2)
        real_feats.append(encoder(l).cpu())

        lf = model(f.to(device)).flatten(2).transpose(1, 2)
        fake_feats.append(encoder(lf).cpu())
    real_feats, fake_feats = torch.cat(real_feats), torch.cat(fake_feats)

    return frechet_distance(real_feats, fake_feats)


if __name__ == "__main__":
    if not os.path.exists(encoder_checkpoint):
        X = np.load("cache/audio2latent_192frames_train_lats.npy", mmap_mode="r")
        XO = np.mean(X, dim=(1, 2))
        B, T, N, L = X.shape
        encoder = CausalCNNEncoderClassifier(
            compared_length=96,
            nb_random_samples=10,
            batch_size=16,
            nb_steps=200,
            in_channels=N * L,
            channels=128,
            out_channels=64,
            reduced_size=32,
            depth=10,
            cuda=True,
        )
        encoder.fit_encoder(X, XO, verbose=True)
        joblib.dump(encoder, encoder_checkpoint, compress=9)
    else:
        encoder = joblib.load(encoder_checkpoint)

    encoder = encoder.encoder.eval().to(device)

    from test_supervised import load_a2l

    with torch.inference_mode():
        ckpts = sys.argv[1:]

        X = np.load("cache/audio2latent_192frames_val_lats.npy", mmap_mode="r")
        F = np.load("cache/audio2latent_192frames_val_feats.npy", mmap_mode="r")

        bs = 16
        T = X.shape[1]
        real_feats = []
        for i in range(0, X.shape[0], bs):
            x = torch.from_numpy(X[i : i + bs].copy()).float()
            xo = torch.mean(x, dim=(1, 2))
            x = x - xo[:, None, None, :]
            x = x.to(device)
            x = x.flatten(2).transpose(1, 2)
            real_feats.append(encoder(x).cpu())
        real_feats = torch.cat(real_feats)

        print("ckpt,frechet_context_distance,laplace_b")
        for ckpt in ckpts:
            a2l = load_a2l(ckpt)

            fake_feats, latent_residuals = [], []
            for i in range(0, X.shape[0], bs):
                f = torch.from_numpy(F[i : i + bs].copy()).float().to(device)
                fx = a2l(f)
                latent_residuals.append(np.random.choice(fx.flatten().cpu().numpy(), size=100000))
                fx = fx.flatten(2).transpose(1, 2)
                fake_feats.append(encoder(fx).cpu())

            distance = frechet_distance(real_feats, torch.cat(fake_feats))

            loc, scale = stats.laplace.fit(np.concatenate(latent_residuals), loc=0, scale=0.1)

            print(f"{Path(ckpt).stem},{distance},{scale}")
