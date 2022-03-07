from time import time

import audioreactive as ar
import numpy as np
import torch

dict(title="A Section", start=15 * 60 + 9, end=16 * 60 + 43)
dict(title="B Section", start=16 * 60 + 43, end=17 * 60 + 8)
from .base import *


def initialize(args):
    ending = args.audio_file.split(".")[-1]
    drums, drsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=2, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 3, causal=0.1)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/other.wav"), args.offset, args.duration)
    args.focus_chroma = torch.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)

    try:
        args.drop_weight = ar.rms(args.audio, args.sr, args.n_frames, smooth=100) ** 2
    except IndexError:
        print(f"IndexError in ar.rms() on {args.n_frames} frames")
        args.drop_weight = 0.125 * torch.ones(args.n_frames)

    args.main_weight = (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=3), 0.5, 0.5) * args.drop_weight
    )

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = ar.normalize(args.low_noise_mod)

    ar.plot_signals([args.drum_onsets, args.low_noise_mod, args.focus_chroma, args.drop_weight])
    return args


def get_latents(selection, args):
    colors = ar.load_latents("/home/hans/datasets/frequent/redrocks/lats/dohr-013|Wilco Collab.npy")
    more_colors = ar.load_latents("/home/hans/datasets/frequent/redrocks/lats/dohr-006_wilco.npy")
    structure = torch.cat(
        (
            colors[[0, 1]],
            more_colors[[0, 1, 2]],
            colors[[0, 1]],
            more_colors[[3, 4]],
            colors[[0, 1]],
            more_colors[[5, 6, 7]],
            colors[[0, 1]],
            more_colors[[8, 9]],
        )
    )

    intro_selection = structure[: int(len(structure) / 2)]
    drop_selection = structure[int(len(structure) / 2) :]

    intro_latents = ar.slerp_loops(intro_selection, args.n_frames, args.bpm / 60 * args.duration / 64, smoothing=16)
    idx = np.random.choice(range(len(selection)))
    intro_latents = (1 - 0.5 * args.drum_onsets[:, None, None]) * intro_latents + selection[
        [idx]
    ] * 0.5 * args.drum_onsets[:, None, None]

    drop_latents = ar.slerp_loops(drop_selection, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=2)

    rea = ar.load_latents("/home/hans/datasets/frequent/redrocks/lats/lyreca-006.npy")[[12, 16, 18, 26, 34, 41, 45, 53]]
    freqs = (args.focus_chroma % rea.shape[0]).long().numpy()
    reactive_latents = torch.from_numpy(rea.numpy()[freqs, :, :]).float()
    reactive_latents = ar.gaussian_filter(reactive_latents, 2)

    latents = args.drop_weight[:, None, None] * drop_latents + (1 - args.drop_weight[:, None, None]) * intro_latents
    latents = (1 - args.main_weight[:, None, None]) * latents + reactive_latents * args.main_weight[:, None, None]
    latents = ar.gaussian_filter(latents, 1)

    return latents
