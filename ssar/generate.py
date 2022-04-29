# fmt: off
import random
import sys
from pathlib import Path
from uuid import uuid4

import librosa as rosa
import matplotlib
import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import tqdm

from .train import STYLEGAN_CKPT

from .features.audio import (chromagram, drop_strength, mfcc, onsets, rms,
                             spectral_contrast, spectral_flatness, tonnetz)
from .features.processing import gaussian_filter, normalize
from .features.rosa.segment import (laplacian_segmentation,
                                    laplacian_segmentation_rosa)
from .optimize import latent2mp4
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper

afns = [chromagram, tonnetz, mfcc, spectral_contrast, spectral_flatness, rms, drop_strength, onsets]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fmt: on


def spline_loop_latents(y, size):
    y = torch.cat((y, y[[0]]))
    t_in = torch.linspace(0, 1, len(y)).to(y)
    t_out = torch.linspace(0, 1, size).to(y)
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


def salience_weighted(envelope, short_sigma=5, long_sigma=80):
    envelope = envelope.squeeze(1)
    short = gaussian_filter(envelope, short_sigma, mode="reflect", causal=0)
    long = gaussian_filter(envelope, long_sigma, mode="reflect", causal=0)
    weighted = (short / long) ** 2 * envelope
    if weighted.dim() < 2:
        weighted = weighted.unsqueeze(1)
    return weighted


@torch.inference_mode()
def generate(audio_file="/home/hans/datasets/wavefunk/naamloos.wav", fps=24, dur=None):
    audio, sr = torchaudio.load(audio_file)
    audio = audio.mean(0)
    if dur is not None:
        audio = audio[: dur * sr]
    audio = resample(audio, sr, 1024 * fps)
    sr = 1024 * fps

    features = {afn.__name__: normalize(salience_weighted(afn(audio, sr).to(device))) for afn in afns}
    n_frames = features["rms"].shape[0]

    ks = [2, 4, 6, 8, 12, 16]
    beats = list(
        rosa.beat.beat_track(onset_envelope=onsets(audio, sr).squeeze().numpy(), sr=sr, trim=False, hop_length=1024)[1]
    )

    segmentations = {}
    for name, feature in features.items():
        for k, s in enumerate(laplacian_segmentation(feature, beats, ks=ks)):
            segmentations[(name, ks[k])] = s.argmax(1)
    for k, rosa_seg in enumerate(laplacian_segmentation_rosa(audio.numpy(), sr, n_frames, ks=ks).to(device).unbind(1)):
        segmentations[("rosa", ks[k])] = rosa_seg.to(device)

    print(list(features.keys()))
    print(list(segmentations.keys()))

    unit_features = ["rms", "drop_strength", "onsets", "spectral_flatness"]
    all_features = ["chromagram", "tonnetz", "mfcc", "spectral_contrast"] + unit_features
    get_intermodulator = lambda: random.choice(unit_features)

    ws = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False)(torch.randn(180, 512)).to(device)

    selection = np.random.permutation(len(ws))[: np.random.randint(3, 15)]
    latents = spline_loop_latents(ws[selection], n_frames)

    n_patches = random.randint(7, 15)
    for _ in range(n_patches):
        k = random.choice(ks)
        feat = random.choice(all_features)
        print(f"{feat} {k}", end=" ")

        seg = segmentations[(feat, k)].cpu().numpy()
        selection = np.random.permutation(len(ws))[:k]
        selectseq = selection[seg]
        segseq = ws[selectseq]
        segseq = gaussian_filter(segseq, 5)

        if np.random.rand() < 0.5:
            if k in [2, 4]:
                lays = slice(12, 18)
            elif k in [6, 8]:
                lays = slice(6, 12)
            elif k in [12, 16]:
                lays = slice(0, 6)
        else:
            lays = random.choice([slice(12, 18), slice(6, 12), slice(0, 6), slice(0, 18), slice(0, 12), slice(6, 18)])

        if np.random.rand() < 1 / 3:
            print("overwrite")
            latents[:, lays] = segseq[:, lays]
        elif np.random.rand() < 2 / 3:
            print("average")
            latents[:, lays] = segseq[:, lays]
            latents[:, lays] /= 2
        else:
            print("modulate")
            modulation = features[get_intermodulator()][..., None]
            latents[:, lays] *= 1 - modulation
            latents[:, lays] += modulation * segseq[:, lays]
    latents = gaussian_filter(latents, 1)

    noise = [gaussian_filter(torch.randn((len(latents), 2 ** i, 2 ** i)), 10) for i in range(2, 8)]
    noise = [n / n.square().mean().sqrt() for n in noise]

    latent2mp4(
        audio_file,
        latents,
        noise,
        out_file=f"output/new_rand_{str(uuid4())[:6]}.mp4",
        stylegan_file=STYLEGAN_CKPT,
        fps=fps,
        duration=dur,
    )


if __name__ == "__main__":
    generate()
