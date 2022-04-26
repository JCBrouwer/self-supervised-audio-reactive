# fmt: off
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

from .features.audio import (chromagram, drop_strength, mfcc, onsets, pulse,
                             rms, spectral_contrast, spectral_flatness,
                             tonnetz)
from .features.correlation import rv2 as correlation
from .features.processing import gaussian_filter
from .models.hippo import hippo
from .train import STYLEGAN_CKPT, normalize_gradients

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from ops.video import VideoWriter

afns = [chromagram, tonnetz, mfcc, spectral_contrast, spectral_flatness, rms, drop_strength, onsets, pulse]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fmt: on


@torch.no_grad()
def latent2video(
    audio_file,
    residuals,
    noise,
    out_file,
    stylegan_file,
    fps=24,
    output_size=(512, 512),
    batch_size=8,
    offset=0,
    duration=40,
    seed=123,
):
    start_frame = int(fps * offset)
    end_frame = int(fps * (offset + duration)) if duration is not None else len(residuals)

    noise1, noise2, noise3, noise4 = noise

    mapper = StyleGAN2Mapper(model_file=stylegan_file, inference=False)
    base_latent = mapper(torch.from_numpy(np.random.RandomState(seed).randn(1, 512))).to(device)
    del mapper

    synthesizer = StyleGAN2Synthesizer(
        model_file=stylegan_file, inference=False, output_size=output_size, strategy="stretch", layer=0
    )
    synthesizer.eval().to(device)

    with VideoWriter(
        output_file=out_file,
        output_size=output_size,
        fps=fps,
        audio_file=audio_file,
        audio_offset=offset,
        audio_duration=duration,
    ) as video:
        for i in tqdm(range(start_frame, end_frame, batch_size), unit_scale=batch_size):
            for frame in (
                synthesizer(
                    latents=base_latent + residuals[i : i + batch_size],
                    noise1=noise1[i : i + batch_size, None],
                    noise2=noise2[i : i + batch_size, None],
                    noise3=noise2[i : i + batch_size, None],
                    noise4=noise3[i : i + batch_size, None],
                    noise5=noise3[i : i + batch_size, None],
                    noise6=noise4[i : i + batch_size, None],
                    noise7=noise4[i : i + batch_size, None],
                )
                .add(1)
                .div(2)
            ):
                video.write(frame.unsqueeze(0))


class HiPPOTimeseries(torch.nn.Module):
    """Efficient timeseries parameterization using 'High-order Polynomial Projection Operators'"""

    def __init__(self, f, N=256, invariance="s") -> None:
        super().__init__()
        if invariance == "t":
            A, B, E = hippo.init_leg_t(N, dt=1 / len(f))
            c = hippo.encode_leg_t(f, A.to(f), B.to(f))
        elif invariance == "s":
            A, B, E = hippo.init_leg_s(N, max_length=len(f))
            c = hippo.encode_leg_s(f, A.to(f), B.to(f))
        self.register_buffer("E", E)
        self.c = torch.nn.Parameter(c)

    def forward(self):
        return (self.E @ self.c.unsqueeze(-1)).squeeze(-1)[-1]


class FixedLatentNoiseDecoder(torch.nn.Module):
    def __init__(self, latents, hidden_size=12, n_latent_split=3, n_noise=4):
        super().__init__()

        self.S = n_latent_split
        self.H = hidden_size
        assert (
            len(latents) == self.S * self.H
        ), f"Number of latent vectors supplied does not equal n_latent_split * hidden_size ({self.S * self.H})"
        self.latents = latents
        self.W = self.latents.shape[1] // self.S

    def forward(self, x):
        latents = []
        for i in range(self.S):
            env = x[..., i * self.H : (i + 1) * self.H]
            with torch.no_grad():
                lat = self.latents[i * self.H : (i + 1) * self.H, i * self.W : (i + 1) * self.W]
            latents.append(torch.einsum("BTH,HWL->BTWL", env, lat))
        latents = torch.cat(latents, dim=2)
        latents = latents - latents.mean(dim=1, keepdim=True)  # latents --> residuals

        noise_envs = x[..., self.S * self.H :]
        noise = []
        B, T, _ = x.shape
        for i in range(noise_envs.shape[-1] // 2):
            mu, sig = noise_envs[..., 2 * i : 2 * (i + 1)].unbind(-1)
            size = 2 ** (i + 2)
            mu = mu[..., None, None].expand(B, T, size, size)
            sig = sig[..., None, None].expand(B, T, size, size)
            with torch.no_grad():
                n = gaussian_filter(torch.randn_like(mu).transpose(1, 0), 5).transpose(1, 0)
            n = mu + sig * n
            noise.append(n)

        return latents, noise


def optimize(
    audio_file,
    fps=24,
    n_steps=512,
    n_params=512,
    hidden_size=6,
    n_latent_split=1,
    n_noise=4,
    lr=1e-4,
    log_steps=16,
    eval_steps=128,
):
    audio, sr = torchaudio.load(audio_file)
    audio = audio.mean(0)
    audio = resample(audio, sr, 1024 * fps)
    sr = 1024 * fps

    n_envelopes = n_latent_split * hidden_size + 2 * n_noise

    features = {afn.__name__: afn(audio, sr).to(device) for afn in afns}
    for n, f in features.items():
        print(
            n.ljust(20),
            f"{tuple(f.shape)}".ljust(15),
            f"{f.min().item():.4f}".ljust(10),
            f"{f.mean().item():.4f}".ljust(10),
            f"{f.max().item():.4f}".ljust(10),
        )
    print()

    n_frames = features["rms"].shape[0]
    pred_names = ["latents", "noise 4x4", "noise 8x8", "noise 16x16", "noise 32x32"]

    envelopes = HiPPOTimeseries(torch.rand((n_frames, n_envelopes), device=device), N=n_params)
    envelopes = envelopes.to(device)

    mapper = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False)
    decoder_latents = mapper(torch.from_numpy(np.random.RandomState(42).randn(n_latent_split * hidden_size, 512))).to(
        device
    )
    del mapper
    decoder = FixedLatentNoiseDecoder(
        decoder_latents, hidden_size=hidden_size, n_latent_split=n_latent_split, n_noise=n_noise
    )

    optimizer = torch.optim.Adam(envelopes.parameters(), lr=lr)

    with tqdm(range(n_steps)) as pbar:
        for it in pbar:
            optimizer.zero_grad()

            envs = envelopes().T
            latents, noise = decoder(envs.unsqueeze(0))
            latents = latents.squeeze(0)
            noise = [n.squeeze(0) for n in noise]
            predictions = [normalize_gradients(latents)] + [normalize_gradients(n, 0.25) for n in noise]

            loss = 0
            for nn, p in zip(pred_names, predictions):
                for n, f in features.items():
                    contribution = 1 - correlation(p.flatten(1), f.flatten(1))
                    if n == "onsets":
                        contribution = 20 * contribution
                    loss = loss + contribution

                    if it % log_steps == 0:
                        pbar.write(nn.ljust(20) + n.ljust(20) + f"{contribution.item():.4f}")
            if it % log_steps == 0:
                pbar.write(f"total: {loss.item():.4f}\n")

            loss.backward()
            optimizer.step()

            if it % eval_steps == 0:
                fig, ax = plt.subplots(3, 5, figsize=(20, 12))
                [x.axis("off") for x in ax.flatten()]
                for a, (n, f) in enumerate(features.items()):
                    if n == "onsets":
                        pbar.write("onset autocorrelation:")
                        pbar.write(str((f @ f.T).min().item()))
                        pbar.write(str((f @ f.T).mean().item()))
                        pbar.write(str((f @ f.T).max().item()))
                        pbar.write("")
                    ax.flatten()[a].imshow((f @ f.T).detach().cpu().numpy(), cmap="inferno")
                    ax.flatten()[a].set_title(n)
                for a, (n, f) in enumerate(zip(pred_names, predictions)):
                    ax.flatten()[-5 + a].imshow((f.flatten(1) @ f.flatten(1).T).detach().cpu().numpy(), cmap="inferno")
                    ax.flatten()[-5 + a].set_title(n)
                fig.tight_layout()
                plt.savefig(f"output/hippo_{it}_{Path(audio_file).stem}_autocorrelations.pdf")
                plt.close()

                fig, ax = plt.subplots(envs.shape[1], 1, figsize=(16, 4 * envs.shape[1]))
                for e, env in enumerate(envs.unbind(1)):
                    ax.flatten()[e].plot(env.detach().cpu().numpy())
                fig.tight_layout()
                plt.savefig(f"output/hippo_{it}_{Path(audio_file).stem}_envelopes.pdf")
                plt.close()

                latent2video(
                    audio_file,
                    latents,
                    noise,
                    out_file=f"output/hippo_{it}_{Path(audio_file).stem}.mp4",
                    stylegan_file=STYLEGAN_CKPT,
                    fps=fps,
                    duration=None,
                    output_size=(512, 512),
                )


if __name__ == "__main__":
    optimize("/home/hans/datasets/wavefunk/naamloos.wav")
