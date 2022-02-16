import random

import audioreactive as ar
import numpy as np
import torch as th

dict(title="Set intro", start=0 * 60, end=0 * 60 + 40)
dict(title="Mirage Intro", start=0 * 60 + 40, end=1 * 60 + 54)
dict(title="Mirage Main Section", start=1 * 60 + 55, end=2 * 60 + 38)
# [transition into outro]
dict(title="Mirage Outro", start=2 * 60 + 58, end=3 * 60 + 41)


def initialize(args):
    ending = args.audio_file.split(".")[-1]
    drums, drsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=2, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 2, causal=0)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/other.wav"), args.offset, args.duration)
    args.focus_chroma = th.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)

    bass, basr, _ = ar.load_audio(args.audio_file.replace("." + ending, "/bass.wav"), args.offset, args.duration)
    args.bass_onsets = ar.onsets(bass, basr, args.n_frames, smooth=2, clip=95, power=1)
    args.bass_onsets = ar.compress(args.bass_onsets, 0.5, 0.5)
    args.bass_onsets = ar.gaussian_filter(args.bass_onsets, 2)

    try:
        args.drop_weight = 0.5 * ar.rms(args.audio, args.sr, args.n_frames, smooth=100, clip=0.6) ** 2
    except:
        args.drop_weight = th.zeros(args.n_frames)

    args.main_weight = 0.666 * (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=3), 0.5, 0.5) * args.drop_weight
    )

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = 0.75 * ar.normalize(args.low_noise_mod)

    ar.plot_signals([args.drum_onsets, args.main_weight, args.low_noise_mod, args.focus_chroma, args.drop_weight])
    return args


def get_latents(selection, args):
    intro_selection = ar.load_latents(
        "/home/hans/code/maua-stylegan2/workspace/cyphept-CYPHEPT-2q5b2lk6-33-1024-138000_mirage_green_latents.npy"
    )
    drop_selection = ar.load_latents(
        "/home/hans/code/maua-stylegan2/workspace/cyphept-CYPHEPT-2q5b2lk6-33-1024-138000_mirage_red_latents.npy"
    )

    lats = "/home/hans/datasets/frequent/redrocks/lats/"
    redblue = ar.load_latents(lats + "cyphept-CYPHEPT-2q5b2lk6-35-1024-112000_red_blue_mirage.npy")
    bluered = ar.load_latents(lats + "cyphept-CYPHEPT-2q5b2lk6-35-1024-112000_blue_red_mirage.npy")
    sunset = th.cat((redblue, bluered))
    sunset = sunset[np.random.permutation(range(len(sunset)))]
    if np.random.rand() < 0.5:
        sunset_latents = ar.chroma_weight_latents(ar.chroma(args.audio, args.sr, args.n_frames), sunset[:12])
    else:
        sunset_latents = ar.slerp_loops(sunset[:12], args.n_frames, args.bpm / 60 * args.duration / 32, smoothing=16)

    intro_latents = ar.slerp_loops(intro_selection, args.n_frames, args.bpm / 60 * args.duration / 64, smoothing=16)
    idx = np.random.choice(range(len(selection)))
    intro_latents = (1 - 0.5 * args.drum_onsets[:, None, None]) * intro_latents + selection[
        [idx]
    ] * 0.5 * args.drum_onsets[:, None, None]

    drop_latents = ar.slerp_loops(drop_selection, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)

    freqs = (args.focus_chroma % selection.shape[0]).long().numpy()
    reactive_latents = th.from_numpy(selection.numpy()[freqs, :, :]).float()
    reactive_latents = ar.gaussian_filter(reactive_latents, 2)

    drop_latents = (1 - args.main_weight[:, None, None]) * drop_latents + reactive_latents * args.main_weight[
        :, None, None
    ]

    latents = args.drop_weight[:, None, None] * drop_latents + (1 - args.drop_weight[:, None, None]) * intro_latents

    latents[:, 14:] = sunset_latents[:, 14:]

    latents = ar.gaussian_filter(latents, 2.5)

    return latents


def get_noise(height, width, scale, num_scales, args):
    if height > 256:
        return None

    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 20).cpu()
    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 5).cpu()
    noise_noisier = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 1.5).cpu()

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


def get_bends(args):
    transform = th.nn.Sequential(
        th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda"))
    )
    bends = [{"layer": 0, "transform": transform}]

    return bends
