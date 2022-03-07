from time import time

import audioreactive as ar
import numpy as np
import torch

dict(title="Intro", start=28 * 60 + 57, end=29 * 60 + 31)
dict(title="1st Movement", start=29 * 60 + 31, end=30 * 60 + 37)
dict(title="2nd Movement", start=30 * 60 + 37, end=31 * 60 + 27)
dict(title="Ambience", start=31 * 60 + 27, end=32 * 60 + 20)
dict(title="Final Movement", start=32 * 60 + 20, end=34 * 60 + 8)
from .base import *

start = 28 * 60 + 50
A = 29 * 60 + 31 - start
B = 30 * 60 + 37 - start
C = 32 * 60 + 20 - start


def get_latents(selection, args):
    c1 = ar.load_latents(
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-33-1024-138000|Forest Fires.npy"
    )
    c2 = ar.load_latents(
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-33-1024-138000_forestfiresC.npy"
    )
    c3 = ar.load_latents(
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-33-1024-138000_forestfiresB.npy"
    )
    c = torch.cat((c1, c2))
    c = c[np.random.permutation(range(len(c)))]
    col1 = c1
    col2 = c[:6]
    col3 = c[6:12]
    col4 = c[12:]

    try:
        structure_file = np.random.choice(
            glob(f"/home/hans/datasets/frequent/redrocks/lats/strucs/{Path(args.latent_file).stem[:4]}*.npy")
        )
        structure = ar.load_latents(structure_file)
    except:
        structure = torch.cat([selection] * 4)
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
    color_loops2 = ar.slerp_loops(col2, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=12)
    color_loops3 = ar.slerp_loops(col3, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=8)
    color_loops4 = ar.slerp_loops(col4, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=4)
    color_loops = torch.cat(
        (
            color_loops1[: int(A * args.fps)],
            color_loops2[int(A * args.fps) : int(B * args.fps)],
            color_loops3[int(B * args.fps) : int(C * args.fps)],
            color_loops4[int(C * args.fps) :],
        )
    )
    latents[:, 8:] = color_loops[:, 8:]

    latents = ar.gaussian_filter(latents, 2.5)

    return latents
