import audioreactive as ar
import numpy as np
import torch


start = 12 * 60 + 20
end = 14 * 60 + 50
dict(title="Intro", start=12 * 60 + 26, end=13 * 60 + 21)
dict(title="Drop", start=13 * 60 + 21, end=14 * 60 + 43)

DROP = (13 * 60 + 21) - (12 * 60 + 20)

from .base import *


def initialize(args):
    ending = args.audio_file.split(".")[-1]
    drums, drsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=2, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 2, causal=0.25)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/other.wav"), args.offset, args.duration)
    args.focus_chroma = torch.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)

    try:
        args.drop_weight = ar.rms(args.audio, args.sr, args.n_frames, smooth=66, clip=66) ** 2
    except IndexError:
        print(f"IndexError in ar.rms() on {args.n_frames} frames")
        args.drop_weight = 0.125 * torch.ones(args.n_frames)

    args.main_weight = (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=2.5), 0.5, 0.5) * args.drop_weight
    )

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = ar.normalize(args.low_noise_mod)

    global DROP_FRAME
    DROP_FRAME = DROP * args.fps
    args.drum_onsets[: DROP_FRAME - args.fps] *= 0.333
    args.drop_weight[: DROP_FRAME - args.fps] *= 0.6
    args.main_weight[: DROP_FRAME - args.fps] *= 0.333
    args.low_noise_mod[: DROP_FRAME - args.fps] *= 0.6

    args.drop_weight[DROP_FRAME : DROP_FRAME + 500] /= 0.35
    args.drop_weight[args.drop_weight > 1] = 1

    ar.plot_signals([args.drum_onsets, args.main_weight, args.low_noise_mod, args.focus_chroma, args.drop_weight])
    return args


def get_latents(selection, args):
    colors = ar.load_latents(
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-33-1024-138000|Tension Coil.npy"
    )[[0, 1, 3]]
    color = colors[[2]]
    structure = ar.load_latents(
        "/home/hans/datasets/frequent/redrocks/lats/cyphept-CYPHEPT-2q5b2lk6-36-1024-103000.npy"
    )

    intro_selection = structure[: int(len(structure) / 2)]
    drop_selection = structure[int(len(structure) / 2) :]

    intro_latents = ar.slerp_loops(intro_selection, args.n_frames, args.bpm / 60 * args.duration / 32, smoothing=16)

    drop_latents = ar.slerp_loops(drop_selection, args.n_frames, args.bpm / 60 * args.duration / 8, smoothing=1)
    idx = np.random.choice(range(len(selection)))
    drop_latents = (1 - 0.5 * args.drum_onsets[:, None, None]) * drop_latents + selection[
        [idx]
    ] * 0.5 * args.drum_onsets[:, None, None]

    latents = args.drop_weight[:, None, None] * drop_latents + (1 - args.drop_weight[:, None, None]) * intro_latents

    color_loops = ar.slerp_loops(colors, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)
    color_loops[DROP_FRAME:, 10:] = color[:, 10:]
    color_loops = ar.gaussian_filter(color_loops, 2)

    freqs = (args.focus_chroma % drop_selection.shape[0]).long().numpy()
    reactive_latents1 = torch.from_numpy(drop_selection.numpy()[freqs, :, :]).float()
    reactive_latents1 = ar.gaussian_filter(reactive_latents1, 2)

    freqs = (args.focus_chroma % intro_selection.shape[0]).long().numpy()
    reactive_latents2 = torch.from_numpy(intro_selection.numpy()[freqs, :, :]).float()
    reactive_latents2 = ar.gaussian_filter(reactive_latents2, 1)

    reactive_latents = torch.cat((reactive_latents1[:DROP_FRAME], reactive_latents2[DROP_FRAME:]))

    latents[:, 8:] = color_loops[:, 8:]
    latents = (1 - args.main_weight[:, None, None]) * latents + reactive_latents * args.main_weight[:, None, None]

    latents = ar.gaussian_filter(latents, 2)

    return latents
