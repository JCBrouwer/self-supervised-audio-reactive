import audioreactive as ar
import numpy as np
import torch as th

th.backends.cudnn.benchmark = True
th.set_grad_enabled(False)


dict(title="Intro", start=54 * 60 + 23, end=55 * 60 + 17)
dict(title="Build", start=55 * 60 + 15, end=55 * 60 + 51)
dict(title="1st Drop", start=55 * 60 + 51, end=56 * 60 + 36)
dict(title="2nd Drop", start=56 * 60 + 36, end=57 * 60 + 20)
dict(title="Bridge", start=57 * 60 + 20, end=58 * 60 + 5)
dict(title="2nd Movement", start=58 * 60 + 5, end=59 * 60 + 12)
dict(title="Outro", start=59 * 60 + 19, end=60 * 60)

start = 54 * 60 + 15
first = 56 * 60 + 36 - start
second = 186


def initialize(args):
    drums, drsr, _ = ar.load_audio(args.audio_file.replace(".flac", "/drums.wav"), args.offset, args.duration)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=200, clip=94, smooth=1, power=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 0.5, causal=0)

    focus, focsr, _ = ar.load_audio(args.audio_file.replace(".flac", "/other.wav"), args.offset, args.duration)
    args.focus_onsets = ar.onsets(focus, focsr, args.n_frames, smooth=1, clip=95, power=1)
    args.focus_onsets = ar.compress(args.focus_onsets, 0.5, 0.5)
    args.focus_onsets = ar.gaussian_filter(args.focus_onsets, 0.5)

    bass, basr, _ = ar.load_audio(args.audio_file.replace(".flac", "/bass.wav"), args.offset, args.duration)
    args.bass_onsets = ar.onsets(bass, basr, args.n_frames, smooth=2, clip=95, power=1)
    args.bass_onsets = ar.compress(args.bass_onsets, 0.5, 0.5)
    args.bass_onsets = ar.gaussian_filter(args.bass_onsets, 1)

    args.focus_chroma = th.argmax(ar.chroma(focus, focsr, args.n_frames), dim=1)
    args.chroma = th.argmax(ar.chroma(args.audio, args.sr, args.n_frames), dim=1)

    args.drop_weight = ar.rms(args.audio, args.sr, args.n_frames, smooth=100) ** 2
    args.drop_weight[: int(first * args.fps)] *= 0.8
    args.drop_weight[int(second * args.fps) :] *= 0.7
    args.drop_weight = ar.gaussian_filter(args.drop_weight, 2)
    args.drop_weight = ar.normalize(args.drop_weight)

    args.main_weight = (
        ar.compress(ar.onsets(args.audio, args.sr, args.n_frames, clip=97, smooth=2.5), 0.5, 0.5) * args.drop_weight
    )

    args.high_noise_mod = ar.percentile_clip(args.main_weight ** 2, 97)
    # args.high_noise_mod *= 3.333

    args.low_noise_mod = (1 - args.drop_weight) * args.main_weight
    args.low_noise_mod = ar.normalize(args.low_noise_mod)

    ar.plot_signals(
        [
            args.drum_onsets,
            (args.drum_onsets + args.main_weight) / 2,
            args.bass_onsets,
            args.focus_onsets,
            args.focus_chroma,
            args.chroma,
            args.drop_weight,
            args.main_weight,
            args.high_noise_mod,
            args.low_noise_mod,
        ]
    )
    return args


# [ 8  6  0 11  2  5  7 10  1  4  3  9]
# [17, 8, 15, 0, 9, 24, 10, 11, 14, 18, 12, 5, 6]
# [0, 7, 4, 13, 6, 14]
# [10, 8, 12, 11, 9]
# [ 3  2 13 10 14  8  9 11  4  7  6  0 12  5  1]


def get_latents(selection, args):
    lats = "/home/hans/datasets/frequent/redrocks/lats/"

    druk = ar.load_latents(lats + "neurout2-117_intro_latents.npy")
    druk = druk[np.random.permutation(len(druk))]

    rustig = ar.load_latents(lats + "neurout2-117_drop_latents.npy")
    rustig = rustig[np.random.permutation(len(rustig))]

    third = ar.load_latents(lats + "neurout2-117_third.npy")
    reac = ar.load_latents(lats + "neurout2-117_reac.npy")
    reac2 = ar.load_latents(lats + "neurout2-117_reac2.npy")

    selection = ar.load_latents("workspace/cosmic-tuba-neurout2-665_A_latents.npy")[[2, 4, 7, 8, 10, 13, 14, 15]]
    selection = selection[np.random.permutation(range(len(selection)))]
    intro_structure = ar.load_latents("workspace/cosmic-tuba-intro-structure.npy")
    intro_structure = intro_structure[np.random.permutation(range(len(intro_structure)))]
    drop_structure = ar.load_latents("workspace/cosmic-tuba-drop-structure.npy")
    drop_structure = drop_structure[np.random.permutation(range(len(drop_structure)))]

    drop_color = ar.load_latents("workspace/cosmic-tuba-drop-color.npy")

    blue_idxs = [0, 7, 4, 13, 6, 14]
    np.random.shuffle(blue_idxs)
    blues = drop_color[blue_idxs]

    red_idxs = [10, 8, 12, 11, 9]
    np.random.shuffle(red_idxs)
    reds = drop_color[red_idxs]

    intro_latents = ar.slerp_loops(rustig, args.n_frames, args.bpm / 60 * args.duration / 64, smoothing=8)
    intro_latents = (1 - 0.5 * args.drum_onsets[:, None, None]) * intro_latents + druk[
        [np.random.choice(range(len(druk)))]
    ] * 0.5 * args.drum_onsets[:, None, None]

    drop_latents = ar.slerp_loops(drop_structure, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)
    drop_latents2 = ar.slerp_loops(druk, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)
    drop_latents3 = ar.slerp_loops(intro_structure, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)
    drop_latents[int(first * args.fps) : int(second * args.fps)] = drop_latents2[
        int(first * args.fps) : int(second * args.fps)
    ]
    drop_latents[int(second * args.fps) :] = drop_latents3[int(second * args.fps) :]

    freqs = (args.focus_chroma % selection.shape[0]).long().numpy()
    reactive_latents = th.from_numpy(selection.numpy()[freqs, :, :]).float()
    reactive_latents = ar.gaussian_filter(reactive_latents, 2)

    freqs2 = (args.focus_chroma % reac.shape[0]).long().numpy()
    reactive_latents2 = th.from_numpy(reac.numpy()[freqs2, :, :]).float()
    reactive_latents2 = ar.gaussian_filter(reactive_latents2, 2)

    freqs3 = (args.focus_chroma % reac2.shape[0]).long().numpy()
    reactive_latents3 = th.from_numpy(reac2.numpy()[freqs3, :, :]).float()
    reactive_latents3 = ar.gaussian_filter(reactive_latents3, 2)

    reactive_latents[int(first * args.fps) : int(second * args.fps)] = reactive_latents2[
        int(first * args.fps) : int(second * args.fps)
    ]
    reactive_latents[int(second * args.fps) :] = reactive_latents3[int(second * args.fps) :]

    latents = args.drop_weight[:, None, None] * drop_latents + (1 - args.drop_weight[:, None, None]) * intro_latents
    latents = (1 - args.main_weight[:, None, None]) * latents + reactive_latents * args.main_weight[:, None, None]

    blue_loops = ar.slerp_loops(blues, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)
    red_loops = ar.slerp_loops(reds, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)
    third_loops = ar.slerp_loops(third, args.n_frames, args.bpm / 60 * args.duration / 16, smoothing=4)

    layr = 8
    latents[:, layr:] = blue_loops[:, layr:]
    latents[int(first * args.fps) : int(second * args.fps), layr:] = red_loops[
        int(first * args.fps) : int(second * args.fps), layr:
    ]
    latents[int(second * args.fps) :, layr:] = third_loops[int(second * args.fps) :, layr:]

    latents = ar.gaussian_filter(latents, 2.5)

    return latents


def get_noise(height, width, scale, num_scales, args):
    if height > 128:
        return None

    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 20).cpu()
    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 5).cpu()
    noise_noisier = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 1.5).cpu()

    noise = (
        args.high_noise_mod[:, None, None, None] * noise_noisier
        + (1 - args.high_noise_mod[:, None, None, None]) * noise
    )
    noise = (
        args.low_noise_mod[:, None, None, None] * noise_noisy + (1 - args.low_noise_mod[:, None, None, None]) * noise
    )
    if width > 16:
        onsets = (args.drum_onsets[:, None, None, None] + args.main_weight[:, None, None, None]) / 2
        noise = onsets * noise_noisier + (1 - onsets) * noise
    noise /= noise.std()
    noise /= 1.666

    return noise


# def get_bends(args):
#     transform = th.nn.Sequential(
#         th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda"))
#     )
#     bends = [{"layer": 0, "transform": transform}]

#     return bends
