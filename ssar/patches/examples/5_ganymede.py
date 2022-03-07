from glob import glob
from pathlib import Path

import audioreactive as ar
import numpy as np
import torch

dict(title="A Section", start=9 * 60 + 22, end=10 * 60 + 40)
dict(title="B section", start=10 * 60 + 40, end=12 * 60 + 1)
B_SEC = (10 * 60 + 40) - (9 * 60 + 22)

from .base import *


def initialize(args):
    ending = args.audio_file.split(".")[-1]
    drums, drsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=1, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 4, causal=0.1)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/other.wav"), args.offset, args.duration)

    args.focus_chroma = torch.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)

    try:
        args.drop_weight = ar.rms(args.audio, args.sr, args.n_frames, smooth=100) ** 2
    except IndexError:
        print(f"IndexError in ar.rms() on {args.n_frames} frames")
        args.drop_weight = 0.125 * torch.ones(args.n_frames)

    args.main_weight = 0.8 * (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=3), 0.5, 0.5) * args.drop_weight
    )

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = 0.75 * ar.normalize(args.low_noise_mod)

    ar.plot_signals(
        [
            args.drum_onsets,
            (args.drum_onsets + args.main_weight) / 2,
            args.low_noise_mod,
            args.focus_chroma,
            args.drop_weight,
        ]
    )
    return args


def get_latents(selection, args):
    colors = selection
    structure_file = (
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-34-1024-125000|Ganymede_structure.npy"
    )
    structure = ar.load_latents(structure_file)[[12, 1, 7, 2, 11, 3, 2, 5, 2, 7, 8, 9, 10, 11, 12]]
    structure = structure[np.random.permutation(range(len(structure)))]

    intro_selection = structure[: int(len(structure) / 2)]
    drop_selection = structure[int(len(structure) / 2) :]

    intro_latents = ar.slerp_loops(intro_selection, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=16)
    idx = np.random.choice(range(len(selection)))
    intro_latents = (1 - 0.5 * args.drum_onsets[:, None, None]) * intro_latents + selection[
        [idx]
    ] * 0.5 * args.drum_onsets[:, None, None]

    drop_latents = ar.slerp_loops(drop_selection, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=4)

    latents = args.drop_weight[:, None, None] * drop_latents + (1 - args.drop_weight[:, None, None]) * intro_latents

    freqs = (args.focus_chroma % structure.shape[0]).long().numpy()
    reactive_latents = torch.from_numpy(structure.numpy()[freqs, :, :]).float()
    reactive_latents = ar.gaussian_filter(reactive_latents, 2)

    latents = (1 - args.main_weight[:, None, None]) * latents + reactive_latents * args.main_weight[:, None, None]

    color_loopsA = ar.slerp_loops(
        colors[: int(round(len(colors) / 2))], args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=16
    )
    color_loopsB = ar.slerp_loops(
        colors[int(round(len(colors) / 2)) :], args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=4
    )
    color_loops = ar.gaussian_filter(
        torch.cat((color_loopsA[: int(B_SEC * args.fps)], color_loopsB[int(B_SEC * args.fps) :])), 5
    )
    latents[:, 8:] = color_loops[:, 8:]

    latents = ar.gaussian_filter(latents, 3)

    return latents
