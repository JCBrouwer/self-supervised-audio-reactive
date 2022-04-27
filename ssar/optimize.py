# fmt: off
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import librosa as rosa
import matplotlib
import numpy as np
import torch
import torchaudio
from torch.nn.functional import one_hot, pad
from torchaudio.functional import resample
from tqdm import tqdm

from .features.audio import (chromagram, drop_strength, mfcc, onsets, pulse,
                             rms, spectral_contrast, spectral_flatness,
                             tonnetz)
from .features.processing import gaussian_filter
from .features.rosa.segment import (laplacian_segmentation,
                                    laplacian_segmentation_rosa)
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
    latents,
    noise,
    out_file,
    stylegan_file,
    fps=24,
    output_size=(1024, 1024),
    batch_size=16,
    offset=0,
    duration=None,
):
    start_frame = int(fps * offset)
    end_frame = int(fps * (offset + duration)) if duration is not None else len(latents)

    noise = sum([[noise[0]]] + [[n, n] for n in noise[1:]], [])

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
                    latents=latents[i : i + batch_size],
                    **{f"noise{i}": n[i : i + batch_size, None] for i, n in enumerate(noise)},
                )
                .add(1)
                .div(2)
            ):
                video.write(frame.unsqueeze(0))


class HiPPOTimeseries(torch.nn.Module):
    """Efficient timeseries parameterization using 'High-order Polynomial Projection Operators'"""

    def __init__(self, f, N=512, invariance="s", padding=128) -> None:
        super().__init__()
        self.padding = padding
        f = pad(f.T, (padding, padding)).T  # pad to avoid noisy edges

        if invariance == "t":
            A, B, E = hippo.init_leg_t(N, dt=1 / len(f))
            c = hippo.encode_leg_t(f, A.to(f), B.to(f))
        elif invariance == "s":
            A, B, E = hippo.init_leg_s(N, max_length=len(f))
            c = hippo.encode_leg_s(f, A.to(f), B.to(f))

        self.register_buffer("E", E)
        self.c = torch.nn.Parameter(c)

    def forward(self):
        return (self.E @ self.c.unsqueeze(-1)).squeeze(-1)[-1].T[self.padding : -self.padding]


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
            env = x[:, i * self.H : (i + 1) * self.H]
            env = env - env.min(dim=0).values
            env = env / (env.max(dim=0).values + 1e-8)
            env = env / (env.sum(dim=1, keepdim=True) + 1e-8)
            with torch.no_grad():
                lat = self.latents[i * self.H : (i + 1) * self.H, i * self.W : (i + 1) * self.W]
            latents.append(torch.einsum("TH,HWL->TWL", env, lat))
        latents = torch.cat(latents, dim=1)

        noise_envs = x[:, self.S * self.H :]
        noise = []
        T, _ = x.shape
        for i in range(noise_envs.shape[-1] // 2):
            mu, sig = noise_envs[:, 2 * i : 2 * (i + 1)].unbind(-1)
            size = 2 ** (i + 2)
            mu = mu[:, None, None].expand(T, size, size)
            sig = sig[:, None, None].expand(T, size, size)
            with torch.no_grad():
                n = gaussian_filter(torch.randn_like(mu), 5)
            n = mu + sig * n
            noise.append(n)

        return latents, noise


def autocorrelation(A):
    A = A - A.mean(0)
    A = A / A.std(0)
    A = A.flatten(1)
    return A @ A.T


def rv2(X, Y):
    XX = autocorrelation(X)
    XX = XX - torch.diag(torch.diag(XX))

    YY = autocorrelation(Y)
    YY = YY - torch.diag(torch.diag(YY))

    return torch.trace(XX.T @ YY) / torch.sqrt(torch.trace(XX.T @ XX) * torch.trace(YY.T @ YY))


def abscos(X, Y):
    XX = autocorrelation(X)
    XX = XX / XX.norm(p=2)

    YY = autocorrelation(Y)
    YY = YY / YY.norm(p=2)

    return torch.abs(XX.flatten() @ YY.flatten())


class MutualInformation(torch.nn.Module):
    def __init__(self, max_val=16, sigma=0.4, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()

        self.max_val = max_val
        self.sigma = 2 * sigma ** 2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.register_buffer("bins", torch.linspace(0, max_val, num_bins, device=device)[None, None].float())

    def marginal_pdf(self, values):
        residuals = values - self.bins
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def joint_pdf(self, kernel_values1, kernel_values2):

        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def get_mutual_information(self, x1, x2):
        """
        x1: 1, T, C
        x2: 1, T, C
        return: scalar
        """
        assert x1.max() <= self.max_val and x2.max() <= self.max_val, "Values must be in range [0, max_val]!"
        pdf_x1, kernel_values1 = self.marginal_pdf(x1)
        pdf_x2, kernel_values2 = self.marginal_pdf(x2)
        pdf_x1x2 = self.joint_pdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        return mutual_information

    def forward(self, input1, input2):
        """
        input1: T, C
        input2: T, C
        return: scalar
        """
        return self.get_mutual_information(input1.unsqueeze(0), input2.unsqueeze(0))


mutual_information = MutualInformation()


def optimize(
    audio_file,
    fps=24,
    n_steps=1024,
    n_params=512,
    hidden_size=6,
    n_latent_split=1,
    n_noise=4,
    lr=1e-4,
    log_steps=64,
    eval_steps=256,
):
    with torch.no_grad():
        audio, sr = torchaudio.load(audio_file)
        audio = audio.mean(0)
        audio = resample(audio, sr, 1024 * fps)
        sr = 1024 * fps

        n_envelopes = n_latent_split * hidden_size + 2 * n_noise
        pred_names = ["envelopes", "latents", "noise 4x4", "noise 8x8", "noise 16x16", "noise 32x32"]

        features = {afn.__name__: afn(audio, sr).to(device) for afn in afns}
        n_frames = features["rms"].shape[0]
        feature_weights = {}
        for name, feature in features.items():
            ac = autocorrelation(feature)
            ac -= ac.min()
            ac /= ac.max()
            feature_weights[name] = 1 / ac.mean()
        median_weight = torch.median(torch.tensor(list(feature_weights.values())))

        ks = [2, 3, 4, 5, 6, 7, 8, 8, 8, 12, 12, 12, 16, 16, 16]
        segmentations = laplacian_segmentation_rosa(audio.numpy(), sr, n_frames, ks=ks)
        for i, (k, seg) in enumerate(zip(ks, segmentations.unbind(1))):
            seg = seg.unsqueeze(1).to(device)
            features[f"segmentation_{i}_{k}"] = seg
            feature_weights[f"segmentation_{i}_{k}"] = median_weight

        _, beats = rosa.beat.beat_track(y=audio.numpy(), sr=sr, trim=False, hop_length=1024)
        beats = list(beats)
        feature_segmentations = {}
        for name, feature in features.items():
            if not "segmentation" in name:
                print(name, feature.shape)
                feature_segmentations[name] = laplacian_segmentation(feature, beats)
                print(feature_segmentations[name].shape)

        envelopes = HiPPOTimeseries(torch.rand((n_frames, n_envelopes), device=device), N=n_params)
        envelopes = envelopes.to(device)

        mapper = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False)
        decoder_latents = mapper(
            torch.from_numpy(np.random.RandomState(42).randn(n_latent_split * hidden_size, 512))
        ).to(device)
        del mapper
        decoder = FixedLatentNoiseDecoder(
            decoder_latents, hidden_size=hidden_size, n_latent_split=n_latent_split, n_noise=n_noise
        )

        optimizer = torch.optim.Adam(envelopes.parameters(), lr=lr)

        uuid = str(uuid4())[:6]
        save_base = f"output/hippo_{Path(audio_file).stem}_{uuid}"
        shutil.copy(__file__, f"{save_base}.py")

    with tqdm(range(n_steps)) as pbar:
        for it in pbar:
            optimizer.zero_grad()

            envs = envelopes()
            latents, noise = decoder(envs)
            predictions = (
                [normalize_gradients(envs)]
                + [normalize_gradients(latents)]
                + [normalize_gradients(n, 0.25) for n in noise]
            )

            penalty = 0
            for p1 in range(len(predictions)):
                for p2 in range(p1 + 1, len(predictions)):
                    penalty = penalty + abscos(predictions[p1], predictions[p2])
            penalty = 10 * penalty

            loss, loss_seg = 0, 0
            for nn, p in zip(pred_names, predictions):
                for n, f in features.items():
                    contribution = feature_weights[n] * (1 - rv2(p, f))
                    loss = loss + contribution

                    contribution_seg = mutual_information(feature_segmentations[n], laplacian_segmentation(f, beats))
                    loss_seg = loss_seg + contribution_seg

                    if it % log_steps == 0:
                        pbar.write(
                            nn.ljust(20) + n.ljust(20) + f"{contribution.item():.4f}    {contribution_seg.item():.4f}"
                        )

            (loss + loss_seg + penalty).backward()
            optimizer.step()

            if it % log_steps == 0:
                pbar.write(f"total  : {loss.item():.4f}")
                pbar.write(f"seg    : {loss_seg.item():.4f}")
                pbar.write(f"penalty: {penalty.item():.4f}\n")

                fig, ax = plt.subplots(5, 6, figsize=(20, 16))
                [x.axis("off") for x in ax.flatten()]
                for a, (n, f) in enumerate(features.items()):
                    ax.flatten()[a].imshow((f @ f.T).detach().cpu().numpy(), cmap="inferno")
                    ax.flatten()[a].set_title(n)
                for a, (n, f) in enumerate(zip(pred_names, predictions)):
                    ax.flatten()[-6 + a].imshow((f.flatten(1) @ f.flatten(1).T).detach().cpu().numpy(), cmap="inferno")
                    ax.flatten()[-6 + a].set_title(n)
                fig.tight_layout()
                plt.savefig(f"{save_base}_{it}_autocorrelations.pdf")
                plt.close()

                fig, ax = plt.subplots(envs.shape[1], 1, figsize=(16, 4 * envs.shape[1]))
                for e, env in enumerate(envs.unbind(1)):
                    ax.flatten()[e].plot(env.detach().cpu().numpy())
                fig.tight_layout()
                plt.savefig(f"{save_base}_{it}_envelopes.pdf")
                plt.close()

            if (it + 1) % eval_steps == 0:
                latent2video(
                    audio_file, latents, noise, out_file=f"{save_base}_{it}.mp4", stylegan_file=STYLEGAN_CKPT, fps=fps
                )


if __name__ == "__main__":
    optimize("/home/hans/datasets/wavefunk/naamloos.wav")
