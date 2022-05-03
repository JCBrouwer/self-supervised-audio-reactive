import json
import random
import sys
from pathlib import Path
from uuid import uuid4

import librosa as rosa
import matplotlib
import numpy as np
import scipy
import torch
import torchaudio
from torchaudio.functional import resample
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import tqdm

from .features.audio import chromagram, drop_strength, mfcc, onsets, rms, spectral_contrast, spectral_flatness, tonnetz
from .features.processing import emphasize, gaussian_filter, normalize
from .features.rosa.segment import laplacian_segmentation, laplacian_segmentation_rosa
from .train import STYLEGAN_CKPT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from ops.video import VideoWriter

afns = [chromagram, tonnetz, mfcc, spectral_contrast, spectral_flatness, rms, drop_strength, onsets]
unit_features = ["rms", "drop_strength", "onsets", "spectral_flatness"]
all_features = ["chromagram", "tonnetz", "mfcc", "spectral_contrast"] + unit_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def salience_weighted(envelope, short_sigma=5, long_sigma=80):
    if envelope.dim() > 1:
        envelope = envelope.squeeze(1)
    short = gaussian_filter(envelope, short_sigma, mode="reflect", causal=0)
    long = gaussian_filter(envelope, long_sigma, mode="reflect", causal=0)
    weighted = (short / long) ** 2 * envelope
    if weighted.dim() < 2:
        weighted = weighted.unsqueeze(1)
    return weighted


def spline_loop_latents(y, size, n_loops=1):
    y = torch.cat((y, y[[0]]))
    t_in = torch.linspace(0, 1, len(y)).to(y)
    t_out = torch.linspace(0, n_loops, size).to(y) % 1
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


def latent_patch(latents, ws, segmentations, features, tempo, fps, base, k, nb, feat, where, merge, seed):
    feature = features[feat]
    segmentation = segmentations[(feat, k)]
    permutation = np.random.RandomState(seed).permutation(len(ws))

    if base == "segmentation":
        selection = permutation[:k]
        selectseq = selection[segmentation.cpu().numpy()]
        sequence = ws[selectseq]
        sequence = gaussian_filter(sequence, 5)
    elif base == "feature":
        n_select = feature.shape[1]
        if n_select == 1:
            selection = permutation[:2]
            sequence = feature[..., None] * ws[selection][[0]] + (1 - feature[..., None]) * ws[selection][[1]]
        else:
            selection = permutation[:n_select]
            sequence = torch.einsum("TN,NWL->TWL", feature, ws[selection])
    elif base == "loop":
        selection = permutation[:k]
        n_loops = len(latents) / fps / 60 / tempo / 4 / nb
        sequence = spline_loop_latents(ws[selection], len(latents), n_loops=n_loops)
    sequence = gaussian_filter(sequence, 1)

    if where == "low":
        lays = slice(0, 6)
    elif where == "mid":
        lays = slice(6, 12)
    elif where == "high":
        lays = slice(12, 18)
    elif where == "lowmid":
        lays = slice(0, 12)
    elif where == "midhigh":
        lays = slice(6, 18)
    elif where == "all":
        lays = slice(0, 18)

    if merge == "average":
        latents[:, lays] += sequence[:, lays]
        latents[:, lays] /= 2
    elif merge == "modulate":
        modulation = features[np.random.RandomState(seed).choice(unit_features)][..., None]
        latents[:, lays] *= 1 - modulation
        latents[:, lays] += modulation * sequence[:, lays]
    else:  # overwrite
        latents[:, lays] = sequence[:, lays]

    return latents


class Noise(torch.nn.Module):
    def __init__(self, length, size):
        super().__init__()
        self.length = length
        self.size = size


class Blend(Noise):
    def __init__(self, length, size, modulator):
        super().__init__(length, size)
        self.register_buffer("vec1", torch.randn((modulator.shape[1], size[0], size[1])))
        self.register_buffer("vec2", torch.randn((modulator.shape[1], size[0], size[1])))
        self.register_buffer("modulator", modulator)

    def forward(self, i, b):
        mod = self.modulator[i : i + b]
        mod = mod.reshape(len(mod), -1)
        left = torch.einsum("MHW,BM->BHW", self.vec1, mod)
        right = torch.einsum("MHW,BM->BHW", self.vec2, 1 - mod)
        return left + right


class Multiply(Noise):
    def __init__(self, length, size, modulator):
        super().__init__(length, size)
        self.register_buffer("vec", torch.randn((modulator.shape[1], size[0], size[1])))
        self.register_buffer("modulator", modulator)

    def forward(self, i, b):
        mod = self.modulator[i : i + b]
        mod = mod.reshape(len(mod), -1)
        left = torch.einsum("MHW,BM->BHW", self.vec, mod)
        return left


def rotation_around_random_axis(tensor, angles):  # 2D rotation around random axis with angles in radians
    b = angles.shape[0]
    n = tensor.numel()

    idx = torch.arange(n, device=tensor.device, dtype=torch.long).unsqueeze(0)
    index = torch.cat([torch.cat([torch.ones_like(idx) * bb, idx, idx], dim=0) for bb in range(b)], dim=1)

    values = torch.ones((b, n), device=tensor.device, dtype=tensor.dtype)

    i = np.random.randint(0, n)
    values[:, i] = torch.cos(angles)
    values[:, (i + 1) % n] = torch.cos(angles)
    values = torch.cat((values, torch.sin(angles)[:, None], -torch.sin(angles)[:, None]), dim=1)

    top_right = torch.ones((3, b), device=tensor.device, dtype=torch.long)
    top_right[0] = torch.arange(b, device=tensor.device, dtype=torch.long)
    top_right[1] *= i
    top_right[2] *= (i + 1) % n

    bottom_left = torch.ones((3, b), device=tensor.device, dtype=torch.long)
    bottom_left[0] = torch.arange(b, device=tensor.device, dtype=torch.long)
    bottom_left[1] *= (i + 1) % n
    bottom_left[2] *= i

    index = torch.cat([index, top_right, bottom_left], dim=1)

    R = torch.sparse_coo_tensor(index, values.flatten(), (b, n, n))
    return R


class Loop(Noise):
    def __init__(self, length, size, n_loops=1):
        super().__init__(length, size)
        self.n_loops = n_loops
        self.loop_len = length / self.n_loops
        self.d_theta = 2 * np.pi / self.loop_len
        self.register_buffer("vec", torch.randn((1, size[0] * size[1], 1)))

    def forward(self, i, b):
        angles = torch.arange(i % self.loop_len, (i + b) % self.loop_len, device=self.vec.device) * self.d_theta
        R = rotation_around_random_axis(self.vec, angles)
        out = torch.bmm(R, self.vec.tile(b, 1, 1))
        out = out.reshape(b, *self.size)
        return out


class Average(Noise):
    def __init__(self, left, right):
        super().__init__(left.length, left.size)
        self.left = left
        self.right = right

    def forward(self, i, b):
        return (self.left(i, b) + self.right(i, b)) / 2


class Modulate(Noise):
    def __init__(self, left, right, modulator):
        super().__init__(left.length, left.size)
        self.left = left
        self.right = right
        self.register_buffer("modulator", normalize(modulator.mean(1)))

    def forward(self, i, b):
        mod = self.modulator[i : i + b, None, None]
        return self.left(i, b) * mod + self.right(i, b) * (1 - mod)


class ScaleBias(Noise):
    def __init__(self, base, scale, bias):
        super().__init__(base.length, base.size)
        self.base = base
        self.scale = scale
        self.bias = bias

    def forward(self, i, b):
        return self.scale * self.base(i, b) + self.bias


def noise_patch(noise, features, tempo, fps, base, nb, feat, where, merge, mu, sig, seed):
    if where == "low":
        lays = range(0, 6)
    elif where == "mid":
        lays = range(6, 12)
    elif where == "high":
        lays = range(12, 17)
    elif where == "lowmid":
        lays = range(0, 12)
    elif where == "midhigh":
        lays = range(6, 17)
    elif where == "all":
        lays = range(0, 17)

    feature = features[feat]

    for n in lays:

        if base == "blend":
            new_noise = Blend(length=len(feature), size=noise[n].size, modulator=feature)
        elif base == "multiply":
            new_noise = Multiply(length=len(feature), size=noise[n].size, modulator=feature)
        elif base == "loop":
            n_loops = len(feature) / fps / 60 / tempo / 4 / nb
            new_noise = Loop(length=len(feature), size=noise[n].size, n_loops=n_loops)

        if merge == "average":
            noise[n] = Average(left=noise[n], right=new_noise)
        elif merge == "modulate":
            noise[n] = Modulate(
                left=noise[n], right=new_noise, modulator=features[np.random.RandomState(seed).choice(unit_features)]
            )
        else:  # overwrite
            noise[n] = new_noise

        noise[n] = ScaleBias(noise[n], scale=sig, bias=mu)

    return noise


@torch.inference_mode()
def generate(audio_file="/home/hans/datasets/wavefunk/naamloos.wav", fps=24, dur=None):
    audio, sr = torchaudio.load(audio_file)
    audio = audio.mean(0)
    if dur is not None:
        audio = audio[: dur * sr]
    audio = resample(audio, sr, 1024 * fps)
    sr = 1024 * fps

    features = {afn.__name__: afn(audio, sr).to(device) for afn in afns}
    n_frames = features["rms"].shape[0]

    ks = [2, 4, 6, 8, 12, 16]
    onset_env = onsets(audio, sr).squeeze().numpy()
    prior = scipy.stats.lognorm(loc=70, scale=500, s=1)
    tempo = float(rosa.beat.tempo(onset_envelope=onset_env, max_tempo=200, prior=prior, ac_size=60, hop_length=1024))
    beats = list(rosa.beat.beat_track(onset_envelope=onset_env, trim=False, hop_length=1024, bpm=tempo)[1])

    segmentations = {}
    for name, feature in features.items():
        segs = laplacian_segmentation(feature, beats, ks=ks)
        for k, s in enumerate(segs):
            segmentations[(name, ks[k])] = s.argmax(1)
    for k, rosa_seg in enumerate(laplacian_segmentation_rosa(audio.numpy(), sr, n_frames, ks=ks).to(device).unbind(1)):
        segmentations[("rosa", ks[k])] = rosa_seg.to(device)

    features = {k: normalize(salience_weighted(gaussian_filter(af, sigma=2))) for k, af in features.items()}

    ws = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False)(torch.randn(180, 512)).to(device)

    min_patches, max_patches = 7, 15
    latent_patch_spec = [
        dict(
            base=random.choice(["segmentation", "feature", "loop"]),
            k=random.choice(ks),
            nb=random.choice([4, 8, 16, 32]),  # loop every nb bars
            feat=random.choice(all_features),
            where=random.choice(["low", "mid", "high", "lowmid", "midhigh", "all"]),
            merge=random.choice(["average", "modulate"]),  # , "overwrite"]),
            seed=np.random.randint(0, 2 ** 32),
        )
        for _ in range(random.randint(min_patches, max_patches))
    ]

    base_selection = np.random.permutation(len(ws))[: np.random.randint(3, 15)]
    latents = spline_loop_latents(ws[base_selection], n_frames)
    for spec in latent_patch_spec:
        latents = latent_patch(latents, ws, segmentations, features, tempo, fps, **spec)

    noise_patch_spec = [
        dict(
            base=random.choice(["blend", "multiply", "loop"]),
            nb=random.choice([4, 8, 16, 32]),  # loop every nb bars
            feat=random.choice(all_features),
            where=random.choice(["low", "mid", "high", "lowmid", "midhigh", "all"]),
            merge=random.choice(["average", "modulate",  # , "overwrite"]),
            mu=np.random.normal(),
            sig=np.random.normal(loc=1, scale=1 / 3),
            seed=np.random.randint(0, 2 ** 32),
        )
        for _ in range(random.randint(min_patches, max_patches))
    ]

    noise = [Loop(length=n_frames, size=(2 ** i, 2 ** i)) for i in sum([[2]] + [[n, n] for n in range(3, 11)], [])]
    for spec in noise_patch_spec:
        noise = noise_patch(noise, features, tempo, fps, **spec)
    noise = [n.to(device) for n in noise]

    print("\nlatent patches")
    [print(p) for p in latent_patch_spec]
    print("\nnoise patches")
    [print(p) for p in noise_patch_spec]
    print()

    out_file = f"output/{Path(audio_file).stem}_RandomPatches++_{str(uuid4())[:6]}.mp4"
    out_size = (1024, 1024)
    batch_size = 16

    G = StyleGAN2Synthesizer(
        model_file=STYLEGAN_CKPT, inference=False, output_size=out_size, strategy="stretch", layer=0
    )
    G.eval().to(device)

    with VideoWriter(
        output_file=out_file,
        output_size=out_size,
        fps=fps,
        audio_file=audio_file,
        audio_offset=0,
        audio_duration=dur,
    ) as video:
        for i in tqdm(range(0, len(latents), batch_size), unit_scale=batch_size):
            L = latents[i : i + batch_size].to(device)

            N = {}
            for j, noise_module in enumerate(noise):
                N[f"noise{j}"] = noise_module.forward(i, batch_size)[:, None]
            #     print(j, N[f"noise{j}"].min().item(), N[f"noise{j}"].mean().item(), N[f"noise{j}"].max().item())
            # print()

            for frame in G(latents=L, **N).add(1).div(2):
                video.write(frame.unsqueeze(0))

            if i == 0:
                with open(out_file.replace(".mp4", ".json"), mode="w") as f:
                    f.write(json.dumps({"latent_patches": latent_patch_spec, "noise_patches": noise_patch_spec}))


if __name__ == "__main__":
    generate()
