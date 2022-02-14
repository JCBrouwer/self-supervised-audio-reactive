import audioreactive as ar
import numpy as np
import torch
from time import time

dict(title="1st Movement", start=6 * 60 + 5, end=7 * 60 + 9)
# [transition into 2nd movement]
dict(title="2nd Movement", start=7 * 60 + 25, end=8 * 60 + 13)
dict(title="Outro", start=8 * 60 + 13, end=9 * 60 + 7)
from .base import *


def initialize(args):
    ending = args.audio_file.split(".")[-1]
    drums, drsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=2, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = 0.888 * ar.gaussian_filter(args.drum_onsets, 3, causal=0)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/other.wav"), args.offset, args.duration)

    args.focus_chroma = torch.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)

    try:
        args.drop_weight = 0.888 * ar.rms(args.audio, args.sr, args.n_frames, smooth=100) ** 2
    except IndexError:
        print(f"IndexError in ar.rms() on {args.n_frames} frames")
        args.drop_weight = 0.125 * torch.ones(args.n_frames)

    args.main_weight = 0.888 * (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=3), 0.5, 0.5) * args.drop_weight
    )

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = 0.888 * ar.normalize(args.low_noise_mod)

    ar.plot_signals(
        [
            args.drum_onsets,
            (args.drum_onsets + args.main_weight) / 2,
            args.low_noise_mod,
            # args.bass_onsets,
            # args.focus_onsets,
            args.focus_chroma,
            args.drop_weight,
        ]
    )
    return args
