# %%
import glob
import time
import uuid

import generate
import librosa as rosa
import matplotlib.pyplot as plt
import numpy as np
import render
import scipy.interpolate as interp
import torch as th
import torch.nn.functional as F
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

th.set_grad_enabled(False)
VERBOSE = True


def info(arr):
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")


def percentile_clip(signal, percentile):
    peaks = argrelextrema(signal, np.greater)
    signal = signal.clip(0, np.percentile(signal[peaks], percentile))
    signal /= signal.max()
    return signal


def normalize(signal):
    signal -= signal.min()
    signal /= signal.max()
    return signal


def wrap(tensor, start, length):
    if start + length > tensor.shape[0]:
        return th.cat((tensor[start : tensor.shape[0]], tensor[0 : (start + length) % tensor.shape[0]]))
    return tensor[start : start + length]


def interpolate_signal_to_length(signals, new_length):
    squeeze = False
    if not isinstance(signals, tuple):
        signals = [signals]
        squeeze = True

    interpolated_signals = tuple(
        (
            th.from_numpy(
                interp.interp1d(np.linspace(0, new_length, num=len(x)), x.numpy())(
                    np.linspace(0, new_length, num=new_length)
                )
            )
            if len(x.shape) == 1
            else th.cat(
                [
                    th.from_numpy(
                        interp.interp1d(np.linspace(0, new_length, num=len(x_i)), x_i.numpy())(
                            np.linspace(0, new_length, num=new_length)
                        )
                    )[None, :]
                    for x_i in x
                ],
                axis=0,
            )
            for x in signals
        )
    )
    if squeeze:
        interpolated_signals = interpolated_signals[0]
    return interpolated_signals


def plot_signals(signals):
    if not VERBOSE:
        return
    plt.figure(figsize=(30, 6 * len(signals)))
    for sbplt, signal in enumerate(signals):
        plt.subplot(len(signals), 1, sbplt + 1)
        plt.plot(signal.squeeze())
        plt.tight_layout()
    plt.show()


def plot_spectra(spectra, chroma=False):
    if not VERBOSE:
        return
    plt.figure(figsize=(30, 8 * len(spectra)))
    for sbplt, spectrum in enumerate(spectra):
        # plt.subplot(len(spectrum), 1, sbplt + 1)
        rosa.display.specshow(spectrum, y_axis="chroma" if chroma else None, x_axis="time")
        # plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    start_time = time.time()

    checkpoint = "/home/hans/modelzoo/neurout_665_stylegan2.pt"

    main_audio_file = "/home/hans/datasets/wavefunk/Ouroboromorphism.flac"
    bpm = 70
    fps = 30

    offset = 0
    duration = None  # 27.40 * 4

    if duration is None:
        raw_audio, sr = rosa.load(main_audio_file, offset=offset)
        duration = rosa.core.get_duration(rosa.load(main_audio_file)[0])
    else:
        raw_audio, sr = rosa.load(main_audio_file, offset=offset, duration=duration)
    # if bpm is None:
    onset = rosa.onset.onset_strength(raw_audio, sr=sr)
    # bpm = rosa.beat.tempo(onset_envelope=onset, sr=sr)
    # bpm = round(bpm * 10.0) / 10.0

    num_frames = len(onset)  # int(round(duration * fps))
    fps = num_frames / duration
    num_bars = bpm / 60.0 * duration / 4.0

    # stylegan output size and batch size
    size = 1920
    batch_size = 16

    cache_pls = True
    load_latents = True
    latent_file = "workspace/ourolatents.npy"

    load_noise = False
    noise_file = "workspace/ouronoise.npy"

    # %%
    # generate latents
    if not load_latents:
        # load multitrack audio files
        (
            (kick_audio, _),
            (snare_audio, _),
            (hats_audio, _),
            (amen_audio, _),
            (bass_audio, _),
            (taiko_audio, _),
            (synth_audio, _),
            (plucks_audio, _),
            (vox1_audio, _),
            (vox2_audio, _),
            (fx_audio, _),
        ) = Pool(nodes=10).map(
            rosa.load,
            [
                "workspace/ourokick.wav",
                "workspace/ourosnare.wav",
                "workspace/ourohats.wav",
                "workspace/ouroamen.wav",
                "workspace/ourobass.wav",
                "workspace/ourotaiko.wav",
                "workspace/ourosynth.wav",
                "workspace/ouroplucks.wav",
                "workspace/ourovox.wav",
                "workspace/ourovoxstutter.wav",
                "workspace/ourofx.wav",
            ],
            kwargs={"offset": offset, "duration": duration},
        )

        # %%
        def get_base_latent_loops(base_latent_selection, loop_starting_latents, num_frames, num_loops, smoothing):
            base_latents = []
            for n in range(len(base_latent_selection)):
                for val in np.arange(0, num_frames // num_loops // len(base_latent_selection), 1):
                    base_latents.append(
                        generate.slerp(
                            val,
                            base_latent_selection[(n + loop_starting_latents) % len(base_latent_selection)][0],
                            base_latent_selection[(n + loop_starting_latents + 1) % len(base_latent_selection)][0],
                        ).numpy()
                    )

            base_latents = gaussian_filter(np.array(base_latents), [smoothing, 0], mode="wrap")

            base_latents = np.concatenate(
                [base_latents] * int(num_loops)
                + [base_latents[[-1]]] * (int(num_frames) - int(num_loops) * len(base_latents)),
                axis=0,
            )
            base_latents = np.concatenate([base_latents[:, None, :]] * 18, axis=1)
            base_latents = th.from_numpy(base_latents)
            return base_latents

        # intro latents
        intro_latent_selection = th.from_numpy(np.load("workspace/ourointro_latents.npy"))
        intro_latents = get_base_latent_loops(
            intro_latent_selection, loop_starting_latents=3, num_frames=num_frames, num_loops=num_bars * 2, smoothing=8
        )

        # %%
        def get_vocal_modulation_signal(vox_audio):
            vox_spec = rosa.amplitude_to_db(np.abs(rosa.stft(vox_audio)), ref=np.max)
            plot_spectra([vox_spec])
            vox_spec = normalize(vox_spec)
            vox_spec_weight = vox_spec.mean(0) ** 2
            vox_chroma = rosa.feature.chroma_cens(y=vox_audio, sr=sr)
            vox_chroma_weight = gaussian_filter(vox_chroma.sum(axis=0), [6])
            vox_chroma_weight = normalize(vox_chroma_weight)
            vox_chroma_weight = vox_chroma_weight ** 3
            vox_weight = percentile_clip(vox_spec_weight * vox_chroma_weight, 95)
            vox_weight = gaussian_filter(vox_weight, [4])
            vox_weight = th.from_numpy(vox_weight)
            plot_signals([vox_weight])
            return vox_weight

        vox1_weight = interpolate_signal_to_length(get_vocal_modulation_signal(vox1_audio), num_frames)
        vox2_weight = interpolate_signal_to_length(get_vocal_modulation_signal(vox2_audio), num_frames)
        vox1_weight, vox2_weight = vox1_weight[:, None, None], vox2_weight[:, None, None]
        intro_latents = 2 / 5.0 * intro_latents + 3 / 5.0 * intro_latent_selection[None, 0].numpy()
        vox_factor = 0.777
        intro_latents = (
            (1 - vox_factor * vox1_weight) * (1 - vox_factor * vox2_weight) * intro_latents
            + vox_factor * vox1_weight * intro_latent_selection[None, 0]
            + vox_factor * vox2_weight * intro_latent_selection[None, 3]
        )

        # %%
        def get_synth_modulation_signal(synth_audio):
            synth_chroma = rosa.feature.chroma_cens(y=synth_audio, sr=sr)
            synth_chroma = gaussian_filter(synth_chroma, [0, 10])
            synth_chroma = percentile_clip(synth_chroma, 80)
            synth_chroma /= synth_chroma.sum(0) + 1e-8

            synth_onset = rosa.onset.onset_strength(y=synth_audio, sr=sr, hop_length=512)
            synth_onset = percentile_clip(synth_onset, 80)
            synth_onset = gaussian_filter(synth_onset, [6])
            synth_onset = normalize(synth_onset)
            synth_onset = synth_onset ** 3

            synth_weight = synth_chroma * synth_onset

            synth_weight = th.from_numpy(synth_weight)
            plot_signals([synth_weight.sum(0)])
            return synth_weight

        synth_weight = interpolate_signal_to_length(get_synth_modulation_signal(synth_audio), num_frames)
        synth_weight = synth_weight[:, :, None, None]

        synth_latents = th.cat((intro_latent_selection, intro_latent_selection[:4]))
        synth_latents = (synth_weight * synth_latents[:, None, :, :]).sum(0) + (1 - synth_weight.sum(0)) * intro_latents
        synth_layers = [0, 12]
        intro_latents[:, synth_layers[0] : synth_layers[1], :] = synth_latents[:, synth_layers[0] : synth_layers[1], :]

        # %%
        def get_plucks_modulation_signal(plucks_audio):
            plucks_spec = rosa.amplitude_to_db(np.abs(rosa.stft(plucks_audio)), ref=np.max)

            # plot_spectra([plucks_spec])

            plucks_spec = normalize(plucks_spec)
            plucks_spec_weight = plucks_spec.mean(0)
            # plucks_spec_weight = normalize(plucks_spec_weight)
            # plucks_spec_weight = gaussian_filter(plucks_spec_weight, [1])

            plucks_chroma = rosa.feature.chroma_cens(y=plucks_audio, sr=sr)
            # plucks_chroma = gaussian_filter(plucks_chroma, [0, 2])
            plucks_chroma = plucks_chroma[sorted(np.argpartition(plucks_chroma.sum(1), -7)[-7:])]
            plucks_chroma = percentile_clip(plucks_chroma, 80)
            plucks_chroma /= plucks_chroma.sum(0) + 1e-8
            # plucks_chroma = normalize(plucks_chroma)

            # plot_spectra([plucks_chroma[:, 5000:7300]], chroma=True)

            plucks_weight = rosa.onset.onset_strength(y=plucks_audio, sr=sr, hop_length=512)
            # plucks_weight = plucks_weight ** 2
            plucks_weight = percentile_clip(plucks_weight, 75)
            plucks_weight[1000:3000] *= 0.1
            # plucks_weight[5000:5900] *= 1
            plucks_weight[5900:] *= 0.1
            # plucks_weight[5900:] = plucks_weight[5900:] ** 2
            # plucks_weight = gaussian_filter(plucks_weight, [2])
            plucks_weight = normalize(plucks_weight)

            plucks_weight = plucks_weight * plucks_chroma.sum(0) * plucks_spec_weight
            plucks_weight = percentile_clip(plucks_weight, 75)
            plucks_weight[1000:3000] *= 0.7
            plucks_weight[5000:5900] *= 0.83
            plucks_weight = gaussian_filter(plucks_weight, [1.5])
            # plucks_weight = plucks_weight ** 2

            plucks_weight = th.from_numpy(plucks_weight)
            plot_signals([th.cat((plucks_weight[1000:2500], plucks_weight[5000:7400]))])
            return plucks_weight

        plucks_weight = interpolate_signal_to_length(get_plucks_modulation_signal(plucks_audio), num_frames)
        plucks_weight = plucks_weight[:, None, None]

        # pluck_factor = 0.75
        # plucks_latents = (pluck_factor * plucks_weight * intro_latent_selection[[5, 6, 7, 5, 6, 7, 5], None]).sum(
        #     0
        # ) + th.clamp((1 - pluck_factor * plucks_weight.sum(0)), 0, 3) * intro_latents
        # plucks_layers = [0, 18]
        # intro_latents[:, plucks_layers[0] : plucks_layers[1], :] = plucks_latents[
        #     :, plucks_layers[0] : plucks_layers[1], :
        # ]

        # %%
        # drop latents
        drop_latent_selection = th.from_numpy(np.load("workspace/ourodrop_latents.npy"))
        drop_loops = get_base_latent_loops(
            drop_latent_selection, loop_starting_latents=0, num_frames=num_frames, num_loops=num_bars * 2, smoothing=5
        )
        layer_num = 2
        drop_latents = drop_loops
        drop_latents[:, layer_num:, :] = drop_latent_selection[None, 3, layer_num:, :]

        # %%
        def get_drums_modulation_signal(drum_audio_list):
            kick, snare, hats, amen, taiko = [
                rosa.onset.onset_strength(y=y, sr=sr, hop_length=512) for y in drum_audio_list
            ]

            kick = gaussian_filter(kick, [4])
            snare = gaussian_filter(snare, [4])
            hats = gaussian_filter(hats, [3])

            amen[4720:5500] = gaussian_filter(amen[4720:5500], [5])
            amen[9420:10000] = gaussian_filter(amen[9420:10000], [6])
            amen = gaussian_filter(amen, [2])

            taiko = gaussian_filter(taiko, [4])

            kick = percentile_clip(kick, 97)
            snare = percentile_clip(snare, 97)
            hats = percentile_clip(hats, 95)
            amen = percentile_clip(amen, 90)
            taiko = percentile_clip(taiko, 95)

            kick = th.from_numpy(kick)
            snare = th.from_numpy(snare)
            hats = th.from_numpy(hats)
            amen = th.from_numpy(amen)
            taiko = th.from_numpy(taiko)

            # plot_signals([kick, snare, hats, taiko, amen])
            plot_signals([amen[4500:5500], amen[9300:10000]])

            return kick, snare, hats, amen, taiko

        kick, snare, hats, amen, taiko = interpolate_signal_to_length(
            get_drums_modulation_signal([kick_audio, snare_audio, hats_audio, amen_audio, taiko_audio]), num_frames
        )
        kick, snare, hats, amen, taiko = (
            kick[:, None, None],
            snare[:, None, None],
            hats[:, None, None],
            amen[:, None, None],
            taiko[:, None, None],
        )
        drums_layers = [7, 15]
        drop_latents[:, drums_layers[0] : drums_layers[1], :] = (
            (1 - kick)
            * (1 - snare)
            * (1 - hats)
            * (1 - amen)
            * (1 - taiko)
            * drop_latents[:, drums_layers[0] : drums_layers[1], :]
            + drop_latent_selection[1][None, drums_layers[0] : drums_layers[1], :] * kick
            + drop_latent_selection[5][None, drums_layers[0] : drums_layers[1], :] * amen
            + drop_latent_selection[0][None, drums_layers[0] : drums_layers[1], :] * snare
            + drop_latent_selection[0][None, drums_layers[0] : drums_layers[1], :] * hats
            + drop_latent_selection[0][None, drums_layers[0] : drums_layers[1], :] * taiko
        )

        # %%
        def get_bass_modulation_signal(bass_audio):
            bass_chroma = rosa.feature.chroma_cens(y=bass_audio, sr=sr)
            bass_chroma = gaussian_filter(bass_chroma, [0, 3])
            bass_chroma /= bass_chroma.sum(0) + 1e-8

            bass_onset = rosa.onset.onset_strength(y=bass_audio, sr=sr, hop_length=512)
            bass_onset = gaussian_filter(bass_onset, [7])
            bass_onset = bass_onset ** 2
            bass_onset = percentile_clip(bass_onset, 95)

            bass_weight = bass_onset * bass_chroma
            bass_weight[:, :2400] *= 0.6
            bass_onset = gaussian_filter(bass_onset, [1])

            bass_weight = th.from_numpy(bass_weight)
            plot_signals([bass_weight.sum(0)])
            return bass_weight

        bass_weight = interpolate_signal_to_length(get_bass_modulation_signal(bass_audio), num_frames)
        bass_weight = bass_weight[:, :, None, None]
        bass_latents = th.from_numpy(
            np.concatenate((np.load("workspace/my_latents2.npy"), np.load("workspace/my_latents.npy")))
        )
        bass_latents = (bass_weight * bass_latents[:, None, :, :]).sum(0) + (1 - bass_weight.sum(0)) * drop_latents
        bass_layers = [0, 4]
        drop_latents[:, bass_layers[0] : bass_layers[1], :] = bass_latents[:, bass_layers[0] : bass_layers[1], :]

        # %%
        def get_fx_modulation_signal(fx_audio):
            fx_spec = rosa.amplitude_to_db(np.abs(rosa.stft(fx_audio)), ref=np.max)
            fx_spec = normalize(fx_spec)
            fx_spec_weight = fx_spec.mean(0)  # ** 2

            fx_chroma = rosa.feature.chroma_cens(y=fx_audio, sr=sr)
            fx_chroma = gaussian_filter(fx_chroma, [0, 3])
            fx_chroma /= fx_chroma.sum(0) + 1e-8
            fx_chroma = fx_chroma.sum(0)

            fx_weight = fx_spec_weight * fx_chroma
            fx_weight = gaussian_filter(fx_weight, [2])
            fx_weight = percentile_clip(fx_weight, 95)
            fx_weight = fx_weight ** 2

            fx_weight = th.from_numpy(fx_weight)
            plot_signals([fx_weight])
            return fx_weight

        fx_weight = interpolate_signal_to_length(get_fx_modulation_signal(fx_audio), num_frames)
        fx_weight = fx_weight[:, None, None]
        fx_latents = (1 - fx_weight) * drop_latents + fx_weight * (
            drop_loops / 2 + drop_latent_selection[[2]] / 2
        ) * 1.5

        fx_layers = [6, 18]
        drop_latents[:, fx_layers[0] : fx_layers[1], :] = fx_latents[:, fx_layers[0] : fx_layers[1], :]

        color_layer = 9
        drop_latents[: int(4000 / 7406 * len(drop_latents)), color_layer:, :] = (
            drop_loops[:, color_layer:, :] * 3 / 5 + drop_latent_selection[[2], color_layer:, :] * 2 / 5
        )[: int(4000 / 7406 * len(drop_latents))]
        drop_latents[int(4000 / 7406 * len(drop_latents)) :, color_layer:, :] = (
            drop_loops[:, color_layer:, :] * 3 / 5 + drop_latent_selection[[4], color_layer:, :] * 2 / 5
        )[int(4000 / 7406 * len(drop_latents)) :]

        # %%
        # mix intro & drop latents
        drop_weight = bass_weight.sum(0) + kick + snare + (hats / 3) + taiko + amen
        drop_weight = gaussian_filter(drop_weight.numpy().squeeze(), [2])
        drop_weight = np.sqrt(drop_weight)
        drop_weight[: int(1700 / 7406 * len(drop_weight))] *= 0.5
        drop_weight[int(2460 / 7406 * len(drop_weight)) : int(2550 / 7406 * len(drop_weight))] = 0.63
        drop_weight[int(4000 / 7406 * len(drop_weight)) : int(5000 / 7406 * len(drop_weight))] *= 0.5
        drop_weight = percentile_clip(drop_weight, 50)
        drop_weight[int(1700 / 7406 * len(drop_weight)) : int(3400 / 7406 * len(drop_weight))] = np.maximum(
            0.5, drop_weight[int(1700 / 7406 * len(drop_weight)) : int(3400 / 7406 * len(drop_weight))]
        )
        drop_weight[int(5100 / 7406 * len(drop_weight)) : int(6550 / 7406 * len(drop_weight))] = np.maximum(
            0.8, drop_weight[int(5100 / 7406 * len(drop_weight)) : int(6550 / 7406 * len(drop_weight))]
        )
        drop_weight[7737:7750] = 0
        drop_weight = gaussian_filter(drop_weight, [2])
        drop_weight = th.from_numpy(drop_weight)[:, None, None]

        plot_signals([drop_weight])

        latents = drop_weight * drop_latents + (1 - drop_weight) * intro_latents

        bass_weight[:, :2400] *= 0.3
        high_noise_mod = (amen + 2 * kick + 2 * snare + 0.4 * plucks_weight ** 2 + 1.5 * bass_weight.sum(0))[:, None]
        low_noise_mod = (9 * plucks_weight + hats + fx_weight + vox1_weight + vox2_weight + synth_weight.sum(0))[
            :, None
        ]

        # %%
        if cache_pls:
            print("saving latents...")
            np.save(latent_file, latents.numpy())
            np.save("ourohighnoise.npy", high_noise_mod.numpy())
            np.save("ourolownoise.npy", low_noise_mod.numpy())
        # %%
    else:
        print("loading latents...")
        latents = th.from_numpy(np.load(latent_file))
        if not load_noise:
            high_noise_mod = th.from_numpy(np.load("ourohighnoise.npy"))
            low_noise_mod = th.from_numpy(np.load("ourolownoise.npy"))

    # generate noise
    if not load_noise:

        def generate_noise(num_frames, bpm, duration, high_mod, low_mod, min_h, min_w):
            if duration > 60:
                num_loops = int(bpm * duration / 60 / 2) + 1
            else:
                num_loops = int(bpm * duration / 60 / 2) + 1
            noise_vec = np.random.normal(
                size=(int(num_frames / (num_loops - (1 if duration > 60 else 0))), 1, min_h * 32, min_w * 32)
            )
            loop_len = len(noise_vec)

            noise_noisy = th.from_numpy(gaussian_filter(noise_vec, [1, 0, 0, 0], mode="wrap")).float().cuda()
            noise_vox = th.from_numpy(gaussian_filter(noise_vec, [7, 0, 0, 0], mode="wrap")).float().cuda()
            noise_smooth = th.from_numpy(gaussian_filter(noise_vec, [20, 0, 0, 0], mode="wrap")).float().cuda()

            # calculate noise values for G
            noise = []
            for power in range(0, 9):
                if power > 5:
                    noise.append(None)
                    continue

                high_noise = F.interpolate(noise_noisy, size=(2 ** power * min_h, 2 ** power * min_w)).cpu()
                low_noise = F.interpolate(noise_vox, size=(2 ** power * min_h, 2 ** power * min_w)).cpu()
                base_noise = F.interpolate(noise_smooth, size=(2 ** power * min_h, 2 ** power * min_w)).cpu()

                high_noise = th.cat(
                    [high_noise] * num_loops + [high_noise[: num_frames - num_loops * loop_len]], axis=0
                )
                low_noise = th.cat([low_noise] * num_loops + [low_noise[: num_frames - num_loops * loop_len]], axis=0)
                base_noise = th.cat(
                    [base_noise] * num_loops + [base_noise[: num_frames - num_loops * loop_len]], axis=0
                )

                noise.append(
                    (
                        high_mod * high_noise
                        + (1 - high_mod) * base_noise
                        + low_mod * low_noise
                        + (1 - low_mod) * base_noise
                    ).float()
                )

            del noise_noisy, noise_vox, noise_smooth
            return noise

        min_w = int(2 if size == 512 else (4 if size == 1024 else 8))
        min_h = int(min_w / 2.0 if size == 1920 else min_w)
        noise = generate_noise(num_frames, bpm, duration, high_noise_mod, low_noise_mod, min_h, min_w)

        if cache_pls:
            print("saving noise...")
            [
                np.save(noise_file.replace(".npy", f"{ns}.npy"), noise_scale.numpy())
                for ns, noise_scale in enumerate(noise)
                if noise_scale is not None
            ]
    else:
        print("loading noise...")
        noise = [
            th.from_numpy(np.load(noise_scale_file))
            for noise_scale_file in sorted(glob.glob(noise_file.replace(".npy", "*.npy")))
        ]
        noise += [None] * (9 - len(noise))

    print(f"Preprocessing time: {(time.time() - start_time):.2f}s")

    checkpoint_title = checkpoint.split("/")[-1].split(".")[0].lower()
    track_title = main_audio_file.split("/")[-1].split(".")[0].lower()
    title = f"output/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"
    render.generate_video(
        main_audio_file,
        checkpoint,
        latents,
        noise,
        audio_offset=offset,
        audio_duration=duration,
        bpm=70,
        size=size,
        batch_size=batch_size,
        output_file=title,
    )
