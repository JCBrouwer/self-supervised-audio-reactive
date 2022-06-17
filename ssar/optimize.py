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
from torch.nn.functional import binary_cross_entropy, cross_entropy,mse_loss, l1_loss, log_softmax, one_hot, pad
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

sys.path.append("/home/hans/code/maua/")
from maua.GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from maua.ops.video import VideoWriter

afns = [chromagram, tonnetz, mfcc, spectral_contrast, rms, drop_strength, onsets]#, spectral_flatness, pulse]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fmt: on


@torch.no_grad()
def latent2mp4(
    audio_file,
    latents,
    noise,
    out_file,
    stylegan_file,
    fps=24,
    output_size=(1024, 1024),
    batch_size=8,
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
                    **{f"noise{j}": n[i : i + batch_size, None] for j, n in enumerate(noise)},
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
    def __init__(self, latents, n_latent_split=3, n_latent_groups=3, n_latent_per_group=3):
        super().__init__()

        self.S = n_latent_split
        self.G = n_latent_groups
        self.H = n_latent_per_group
        assert (
            len(latents) == self.S * self.G * self.H
        ), f"Number of latent vectors supplied does not equal n_latent_split * n_latent_per_split ({self.S * self.G * self.H})"
        self.latents = latents
        self.W = self.latents.shape[1] // self.S

    def forward(self, x):
        latents = []
        for i in range(self.S):
            env = x[:, i * (self.G * self.H) : (i + 1) * (self.G * self.H)]
            env = env.reshape(env.shape[0], self.G, self.H)
            env = env.softmax(dim=2)  # winner takes all within latent group
            env = env / (env.sum(dim=(1, 2), keepdim=True) + 1e-8)  # total weighted average normalized to 1
            with torch.no_grad():
                lat = self.latents[i * (self.G * self.H) : (i + 1) * (self.G * self.H), i * self.W : (i + 1) * self.W]
                lat = lat.reshape(self.G, self.H, lat.shape[-2], lat.shape[-1])
            latents.append(torch.einsum("TGH,GHWL->TWL", env, lat))
        latents = torch.cat(latents, dim=1)

        noise_envs = x[:, self.S * self.G * self.H :]
        noise = []
        T, _ = x.shape
        for i in range(noise_envs.shape[-1] // 2):
            mu, sig = noise_envs[:, 2 * i : 2 * (i + 1)].unbind(-1)
            size = 2 ** (i + 2)
            mu = mu[:, None, None].expand(T, size, size)
            sig = sig[:, None, None].expand(T, size, size)
            with torch.no_grad():
                n = gaussian_filter(torch.randn_like(mu), 2)
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


def auction_lap(X, eps=None, max_iters=250):
    """
    From https://github.com/bkj/auction-lap
    https://dspace.mit.edu/bitstream/handle/1721.1/3265/P-2108-26912652.pdf

    X: n-by-n matrix w/ integer entries
    eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """
    eps = 1 / X.shape[0] if eps is None else eps

    cost = torch.zeros((1, X.shape[1]), device=X.device)
    assignment = torch.zeros(X.shape[0], device=X.device, dtype=torch.long) - 1
    bids = torch.zeros(X.shape, device=X.device)

    counter = 0
    while (assignment == -1).any():
        counter += 1

        # Bidding
        unassigned = (assignment == -1).nonzero().squeeze(1)

        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]

        bid_increments = first_value - second_value + eps

        bids_ = bids[unassigned]
        bids_.zero_()
        bids_.scatter_(dim=1, index=first_idx.contiguous().view(-1, 1), src=bid_increments.view(-1, 1))

        # Assignment
        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

        high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]

        cost[:, have_bidder] += high_bids

        assignment[(assignment.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        assignment[high_bidders] = have_bidder.squeeze()

        if counter > max_iters:
            assignment = torch.arange(len(assignment), device=X.device)  # abort
            break

    return assignment


def lap_loss(targets, predictions):
    total_loss = 0
    for b in range(len(targets)):
        target, prediction = targets[b], predictions[b]
        # print((target.argmax(1) == prediction.argmax(1)).sum().item() / len(target), end=" --> ")

        reassignment = auction_lap(target.T @ prediction)
        prediction = prediction[:, reassignment]
        # print((target.argmax(1) == prediction.argmax(1)).sum().item() / len(target))

        # target = target - target.mean(0)
        # target = target / (target.norm(p=2, dim=1, keepdim=True) + 1e-8)
        # prediction = prediction - prediction.mean(0)
        # prediction = prediction / (prediction.norm(p=2, dim=1, keepdim=True) + 1e-8)
        # loss = 1 - torch.einsum("ti,tj->t", prediction, target).mean()
        loss = mse_loss(prediction, target)

        total_loss = total_loss + loss
    return total_loss / len(targets)


def optimize(
    audio_file,
    fps=24,
    n_steps=512,
    n_params=512,
    n_latent_split=1,
    n_latent_groups=1,
    n_latent_per_group=6,
    n_noise=6,
    lr=1e-3,
    log_steps=16,
    eval_steps=128,
    prediction_similarity_penalty=0,
    lambda_rv2=1,
    lambda_lap=0,
    use_audio_segmentation_features=True,
):
    with torch.no_grad():
        audio, sr = torchaudio.load(audio_file)
        audio = audio.mean(0)
        audio = audio[: 40 * sr]
        audio = resample(audio, sr, 1024 * fps)
        sr = 1024 * fps

        n_envelopes = n_latent_split * n_latent_groups * n_latent_per_group + 2 * n_noise
        pred_names = ["envelopes", "latents"] + [f"noise {2**(i+2)}" for i in range(n_noise)]

        features = {afn.__name__: afn(audio, sr).to(device) for afn in afns}
        n_frames = features["rms"].shape[0]
        feature_weights = {}
        for name, feature in features.items():
            ac = autocorrelation(feature)
            ac -= ac.min()
            ac /= ac.max()
            feature_weights[name] = 1 / ac.mean()

        ks = [2, 4, 6, 8, 12, 16]
        if use_audio_segmentation_features:
            segmentations = laplacian_segmentation_rosa(audio.numpy(), sr, n_frames, ks=ks)
            features[f"rosa_segmentation"] = segmentations.float().to(device)
            feature_weights[f"rosa_segmentation"] = torch.max(torch.tensor(list(feature_weights.values())))

        _, beats = rosa.beat.beat_track(y=audio.numpy(), sr=sr, trim=False, hop_length=1024)
        beats = list(beats)
        feature_segmentations = {}
        for name, feature in features.items():
            if not "segmentation" in name:
                feature_segmentations[name] = laplacian_segmentation(feature, beats, ks=ks)
            else:
                feature_segmentations[name] = [one_hot(f).float() for f in feature.long().unbind(1)]

        envelopes = HiPPOTimeseries(torch.rand((n_frames, n_envelopes), device=device), N=n_params)
        envelopes = envelopes.to(device)

        mapper = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False)
        decoder_latents = mapper(
            torch.from_numpy(
                np.random.RandomState(42).randn(n_latent_split * n_latent_groups * n_latent_per_group, 512)
            )
        ).to(device)
        del mapper
        decoder = FixedLatentNoiseDecoder(decoder_latents, n_latent_split, n_latent_groups, n_latent_per_group)

        optimizer = torch.optim.Adam(envelopes.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr / 100)

        uuid = str(uuid4())[:6]
        save_base = f"output/optimization/hippo_{Path(audio_file).stem}_{uuid}"
        shutil.copy(__file__, f"{save_base}.py")

    with tqdm(range(n_steps)) as pbar:
        for it in pbar:
            optimizer.zero_grad()

            envs = envelopes()
            latents, noise = decoder(envs)
            predictions = (
                [normalize_gradients(envs, 1)]
                + [normalize_gradients(latents, 10)]
                + [normalize_gradients(n, 0.25) for n in noise]
            )

            loss_rv2, loss_seg = torch.zeros((), device=device), torch.zeros((), device=device)
            for nn, p in zip(pred_names, predictions):

                segmentations = laplacian_segmentation(p.flatten(1), beats, ks=ks)

                for n, f in features.items():
                    if lambda_rv2:
                        contribution_rv2 = feature_weights[n] * (1 - rv2(p, f))
                        loss_rv2 = loss_rv2 + contribution_rv2

                    if lambda_lap:
                        contribution_seg = lap_loss(feature_segmentations[n], segmentations).mean()
                        loss_seg = loss_seg + contribution_seg

                    if it % log_steps == 0:
                        log_str = nn.ljust(20) + n.ljust(20)
                        if lambda_rv2:
                            log_str += f"{lambda_rv2} * {contribution_rv2.item():.4f}".ljust(10)
                        if lambda_lap:
                            log_str += f"{lambda_lap} * {contribution_seg.item():.4f}"
                        pbar.write(log_str)

            if prediction_similarity_penalty:
                penalty = 0
                for p1 in range(len(predictions)):
                    for p2 in range(p1 + 1, len(predictions)):
                        penalty = penalty + abscos(predictions[p1], predictions[p2])

            loss = 0
            if lambda_rv2:
                loss = loss + lambda_rv2 * loss_rv2
            if lambda_lap:
                loss = loss + lambda_lap * loss_seg
            if prediction_similarity_penalty:
                loss = loss + prediction_similarity_penalty * penalty
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if it % log_steps == 0:
                pbar.write(f"total  : {loss.item():.4f}")
                if lambda_rv2:
                    pbar.write(f"rv2    : {loss_rv2.item():.4f}")
                if lambda_lap:
                    pbar.write(f"seg    : {loss_seg.item():.4f}")
                if prediction_similarity_penalty:
                    pbar.write(f"penalty: {penalty.item():.4f}\n")

                cols = len(predictions)
                rows = int(np.ceil((len(features) + 2 * cols) / cols))
                fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
                [x.axis("off") for x in ax.flatten()]
                for a, (n, f) in enumerate(features.items()):
                    ax.flatten()[a].imshow(autocorrelation(f).detach().cpu().numpy(), cmap="inferno")
                    ax.flatten()[a].set_title(n)
                for a, (n, f) in enumerate(zip(pred_names, predictions)):
                    ax.flatten()[-2 * cols + a].imshow(
                        autocorrelation(f.flatten(1)).detach().cpu().numpy(), cmap="inferno"
                    )
                    ax.flatten()[-2 * cols + a].set_title(n)
                segmentations = [
                    torch.stack(
                        [s.argmax(1).float() for s in laplacian_segmentation(p.flatten(1), beats, ks=ks)], dim=1
                    )
                    for p in predictions
                ]
                seg_names = [f"segmentation {pn}" for pn in pred_names]
                for a, (n, f) in enumerate(zip(seg_names, segmentations)):
                    ax.flatten()[-cols + a].imshow(autocorrelation(f.flatten(1)).detach().cpu().numpy(), cmap="inferno")
                    ax.flatten()[-cols + a].set_title(n)
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
                latent2mp4(
                    audio_file,
                    latents,
                    noise,
                    out_file=f"{save_base}_{it + 1}.mp4",
                    stylegan_file=STYLEGAN_CKPT,
                    fps=fps,
                    duration=40,
                )


if __name__ == "__main__":
    optimize("/home/hans/datasets/wavefunk/naamloos.wav")
