from time import time

import audioreactive as ar
import numpy as np
import torch

dict(title="Intro", start=25 * 60 + 9, end=26 * 60 + 30)
dict(title="A Section", start=26 * 60 + 30, end=27 * 60 + 33)
dict(title="B Section", start=27 * 60 + 33, end=28 * 60 + 35)
from .base import *

start = 25 * 60 + 9
A = 26 * 60 + 30 - start
B = 27 * 60 + 33 - start


def initialize(args):
    ending = args.audio_file.split(".")[-1]
    drums, drsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=1, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 2, causal=0)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/other.wav"), args.offset, args.duration)
    args.focus_chroma = torch.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)

    try:
        args.drop_weight = ar.rms(args.audio, args.sr, args.n_frames, smooth=100) ** 2
    except IndexError:
        print(f"IndexError in ar.rms() on {args.n_frames} frames")
        args.drop_weight = 0.125 * torch.ones(args.n_frames)

    args.main_weight = (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=2.5), 0.5, 0.5) * args.drop_weight
    )

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = ar.normalize(args.low_noise_mod)

    ar.plot_signals([args.drum_onsets, args.low_noise_mod, args.focus_chroma, args.drop_weight])
    return args


def get_latents(selection, args):
    col1 = ar.load_latents("/home/hans/datasets/frequent/redrocks/lats/dohr-013|Axon.npy")
    orb = col1[[3]]
    col1 = col1[[0, 1, 2, 4]]
    col2 = ar.load_latents("/home/hans/datasets/frequent/redrocks/lats/dohr-006_axon3.npy")
    col3 = ar.load_latents("/home/hans/datasets/frequent/redrocks/lats/dohr-006_axon2.npy")

    # structure_file = "/home/hans/datasets/frequent/redrocks/lats/strucs/lyreca-006.npy"
    # structure = ar.load_latents(structure_file)[[0, 1, 3, 5, 11, 21, 22, 25, 33, 35, 36, 37, 44]]
    # structure = structure[np.random.permutation(range(len(structure)))]
    # print(torch.arange(len(structure))[::2])
    # print(torch.arange(len(structure))[1::2])
    # nustruc = torch.zeros_like(torch.cat((structure, *[orb] * len(structure))))
    # nustruc[::2] = orb
    # nustruc[1::2] = structure
    # structure = nustruc

    structure = torch.cat([col1] * 4 + [orb] * 4)
    structure = structure[np.random.permutation(range(len(structure)))]

    intro_selection = structure[: int(len(structure) / 2)]
    drop_selection = structure[int(len(structure) / 2) :]

    intro_latents = ar.slerp_loops(intro_selection, args.n_frames, args.bpm / 60 * args.duration / 64, smoothing=16)
    idx = np.random.choice(range(len(selection)))
    intro_latents = (1 - 0.5 * args.drum_onsets[:, None, None]) * intro_latents + selection[
        [idx]
    ] * 0.5 * args.drum_onsets[:, None, None]

    drop_latents = ar.slerp_loops(drop_selection, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)

    freqs = (args.focus_chroma % structure.shape[0]).long().numpy()
    reactive_latents = torch.from_numpy(structure.numpy()[freqs, :, :]).float()
    reactive_latents = ar.gaussian_filter(reactive_latents, 2)

    drop_latents = (1 - args.main_weight[:, None, None]) * drop_latents + reactive_latents * args.main_weight[
        :, None, None
    ]

    latents = args.drop_weight[:, None, None] * drop_latents + (1 - args.drop_weight[:, None, None]) * intro_latents

    color_loops1 = ar.slerp_loops(col1, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=16)
    color_loops2 = ar.slerp_loops(col2, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=8)
    color_loops3 = ar.slerp_loops(col3, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=4)
    color_loops = torch.cat(
        (
            color_loops1[: int(A * args.fps)],
            color_loops2[int(A * args.fps) : int(B * args.fps)],
            color_loops3[int(B * args.fps) :],
        )
    )
    latents[:, 8:] = color_loops[:, 8:]

    latents = ar.gaussian_filter(latents, 2.5)

    return latents
