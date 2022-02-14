from glob import glob
from pathlib import Path

import audioreactive as ar
import numpy as np
import torch

dict(title="Intro", start=3 * 60 + 41, end=4 * 60 + 51)
dict(title="Drop", start=4 * 60 + 51, end=5 * 60 + 38)

from .base import get_bends


def initialize(args):
    ending = args.audio_file.split(".")[-1]
    drums, drsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=1, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 1.5, causal=0)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/other.wav"), args.offset, args.duration)
    args.focus_onsets = ar.onsets(focus, focsr, args.n_frames, smooth=1, clip=95, power=1)
    args.focus_onsets = ar.compress(args.focus_onsets, 0.5, 0.5)
    args.focus_onsets = ar.gaussian_filter(args.focus_onsets, 1.5)

    bass, basr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/bass.wav"), args.offset, args.duration)
    args.bass_onsets = ar.onsets(bass, basr, args.n_frames, smooth=2, clip=95, power=1)
    args.bass_onsets = ar.compress(args.bass_onsets, 0.5, 0.5)
    args.bass_onsets = ar.gaussian_filter(args.bass_onsets, 1.5)

    args.focus_chroma = torch.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)

    args.drop_weight = 0.5 * ar.rms(args.audio, args.sr, args.n_frames, smooth=100) ** 2

    args.main_weight = (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=2.5), 0.5, 0.5) * args.drop_weight
    )

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = 0.75 * ar.normalize(args.low_noise_mod)

    ar.plot_signals(
        [
            args.drum_onsets,
            (args.drum_onsets + args.main_weight) / 2,
            args.low_noise_mod,
            args.bass_onsets,
            args.focus_onsets,
            args.focus_chroma,
            args.drop_weight,
        ]
    )
    return args


def get_latents(selection, args):
    colors = selection[np.random.permutation(range(len(selection)))]

    A = ar.load_latents(
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-33-1024-145000|Reflex Angle.npy"
    )
    B = ar.load_latents(
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-33-1024-138000|Reflex Angle.npy"
    )
    colors = torch.cat((A, colors, B))

    structure_file = "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-34-1024-125000|reflex_angle_structure_latents.npy"
    structure = ar.load_latents(structure_file)[[2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

    intro_selection = torch.cat((structure[: int(len(structure) / 2)], A, A, A))
    intro_selection = intro_selection[np.random.permutation(range(len(intro_selection)))]
    drop_selection = torch.cat((structure[int(len(structure) / 2) :], B, B, B))
    intro_selection = drop_selection[np.random.permutation(range(len(drop_selection)))]

    intro_latents = ar.slerp_loops(intro_selection, args.n_frames, args.bpm / 60 * args.duration / 64, smoothing=16)
    idx = np.random.choice(range(len(selection)))
    intro_latents = (1 - 0.5 * args.drum_onsets[:, None, None]) * intro_latents + selection[
        [idx]
    ] * 0.5 * args.drum_onsets[:, None, None]

    drop_latents = ar.slerp_loops(drop_selection, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)

    freqs = (args.focus_chroma % selection.shape[0]).long().numpy()
    reactive_latents = torch.from_numpy(selection.numpy()[freqs, :, :]).float()
    reactive_latents = ar.gaussian_filter(reactive_latents, 2)

    drop_latents = (1 - args.main_weight[:, None, None]) * drop_latents + reactive_latents * args.main_weight[
        :, None, None
    ]

    latents = args.drop_weight[:, None, None] * drop_latents + (1 - args.drop_weight[:, None, None]) * intro_latents

    color_loops = ar.slerp_loops(colors, args.n_frames, args.bpm / 60 * args.duration / 32, smoothing=16)
    latents[:, 12:] = color_loops[:, 12:]

    latents = ar.gaussian_filter(latents, 2.5)

    return latents


def get_noise(height, width, scale, num_scales, args):
    if height > 256:
        return None

    noise = ar.gaussian_filter(torch.randn((args.n_frames, 1, height, width), device="cuda"), 20).cpu()
    noise_noisy = ar.gaussian_filter(torch.randn((args.n_frames, 1, height, width), device="cuda"), 5).cpu()
    noise_noisier = ar.gaussian_filter(torch.randn((args.n_frames, 1, height, width), device="cuda"), 1.5).cpu()

    if height > 8:
        onsets = (args.drum_onsets[:, None, None, None] + args.main_weight[:, None, None, None]) / 2
        noise = onsets * noise_noisier + (1 - onsets) * noise
        noise = (
            args.low_noise_mod[:, None, None, None] * noise_noisy
            + (1 - args.low_noise_mod[:, None, None, None]) * noise
        )
    noise = noise / noise.std()
    noise = noise / 3

    return noise
