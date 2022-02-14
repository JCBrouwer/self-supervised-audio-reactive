# %%
import os
import uuid
import copy
import json
import render
import generate
import numpy as np
import torch as th
import madmom as mm
import librosa as rosa
import scipy.signal as signal
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

th.set_grad_enabled(False)
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["figure.facecolor"] = "black"
VERBOSE = False

os.environ["MIX"] = "ellipsis_cbc"
os.environ["CKPT"] = "neurout2-338"
# VERBOSE = True


def info(arr):
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")


def percentile_clip(y, percentile):
    try:
        peaks = signal.argrelextrema(y, np.greater)
        y = y.clip(0, np.percentile(y[peaks], percentile))
        y /= y.max()
    except:
        print("WARNING: percentile clipping failed, no local maxima?")
    return y


def normalize(y):
    y -= y.min()
    y /= y.max()
    return y


def causal_gaussian(arr, sigmas):
    # 4x gaussian_filter smooth size cuz g_f defaults to truncate=4
    smooth_size = max(4, int(5 * sigmas[0]))
    kernel = signal.hann(smooth_size)[smooth_size // 2 :]
    for _ in range(len(sigmas) - 1):
        kernel = kernel[:, None]
    return signal.convolve(arr, kernel, mode="same") / (np.sum(kernel) + 1e-8)


def plot_signals(signals, vlines=None):
    if not VERBOSE:
        return
    info(signals)
    plt.figure(figsize=(30, 6 * len(signals)))
    for sbplt, y in enumerate(signals):
        plt.subplot(len(signals), 1, sbplt + 1)
        if vlines is not None:
            plt.vlines(vlines, 0.0, 1.0)
        plt.plot(y.squeeze())
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
    checkpoint = f"/home/hans/modelzoo/train_checks/{os.environ['CKPT']}.pt"
    size = 1920
    batch_size = 16
    noise_scales = 6

    file_root = os.environ["MIX"]
    main_audio_file = f"../../datasets/cantbecancelled/{file_root}.wav"
    metadata_file = f"workspace/{main_audio_file.split('/')[-1].split('.')[0]}_metadata.json"
    intro_file = f"workspace/{file_root}_intro_latents.npy"
    drop_file = f"workspace/{file_root}_drop_latents.npy"
    latent_file = f"workspace/{file_root}_latents.npy"
    noise_file = f"workspace/{file_root}_noise.npy"

    bpm = None

    render_track = None
    offset = 0
    duration = None
    if duration is None:
        duration = rosa.get_duration(filename=main_audio_file)

    if os.path.exists(metadata_file):
        with open(metadata_file) as json_file:
            data = json.load(json_file)

        bpm = data["bpm"]
        total_frames = data["total_frames"]
        raw_tracks = data["tracks"]
    else:
        audio, sr = rosa.load(main_audio_file)
        onset = rosa.onset.onset_strength(audio, sr=sr)
        bpm = rosa.beat.tempo(onset_envelope=onset, sr=sr)[0]
        bpm = np.round(bpm * 10.0) / 10.0

        total_frames = len(onset)
        raw_tracks = [["0:00", bpm]]

        with open(metadata_file, "w") as outfile:
            json.dump({"bpm": bpm, "total_frames": total_frames, "tracks": raw_tracks}, outfile)

    fps = 24
    if fps is None:
        fps = total_frames / rosa.get_duration(filename=main_audio_file)
    smf = fps / 43.066666

    num_frames = int(round(duration * fps))

    # preprocess track times --> frame indexes
    tracks = []
    for t, (t_time, t_bpm) in enumerate(raw_tracks):
        start_time = int(t_time.split(":")[0]) * 60 + int(t_time.split(":")[1])
        if t + 1 < len(raw_tracks):
            stop_time = int(raw_tracks[t + 1][0].split(":")[0]) * 60 + int(raw_tracks[t + 1][0].split(":")[1])

        else:
            stop_time = rosa.get_duration(filename=main_audio_file)
        # if not (stop_time > offset and start_time < offset + duration):
        #     continue
        # else:
        if stop_time < offset:
            start_time, stop_time = -1, -1
        elif start_time > offset + duration:
            start_time, stop_time = -1, -1
        elif start_time < offset:
            start_time = offset
        elif stop_time > offset + duration:
            stop_time = offset + duration

        fac = 1.0 / duration * num_frames

        start_frame = int(round(offset * fac))
        stop_frame = int(round((offset + duration) * fac))

        tracks.append(
            [
                t,
                t_bpm,
                stop_time - start_time,
                int(round(start_time * fac)) - start_frame,
                int(round(stop_time * fac)) - start_frame,
            ]
        )
        # print(
        #     t,
        #     t_bpm,
        #     stop_time - start_time,
        #     int(round(start_time * fac)) - start_frame,
        #     int(round(stop_time * fac)) - start_frame,
        # )

    if render_track is not None:
        offset = tracks[render_track][-2]
        duration = tracks[render_track][-1] - tracks[render_track][-2]
        num_frames = int(round(duration * fps))

    # %%
    prms = [
        {
            "intro_num_beats": 64,
            "intro_loop_smoothing": 40,
            "intro_loop_factor": 0.7,
            "drop_num_beats": 32,
            "drop_loop_smoothing": 16,
            "drop_loop_factor": 1,
            "onset_smooth": 1,
            "onset_clip": 90,
            "freq_mod": 10,
            "freq_mod_offset": 0,
            "freq_smooth": 6,
            "freq_latent_smooth": 5,
            "freq_latent_layer": 0,
            "freq_latent_weight": 3,
            "high_freq_mod": 10,
            "high_freq_mod_offset": 0,
            "high_freq_smooth": 3,
            "high_freq_latent_smooth": 6,
            "high_freq_latent_layer": 1,
            "high_freq_latent_weight": 2,
            "rms_smooth": 3,
            "bass_smooth": 3,
            "bass_clip": 90,
            "drop_clip": 70,
            "drop_smooth": 2,
            "drop_weight": 1,
            "high_noise_clip": 97,
            "high_noise_weight": 2.5,
            "low_noise_weight": 1.5,
            "transition_window": int(round(fps)),
        }
        for _ in range(len(tracks))
    ]

    with open(metadata_file.replace("metadata", "params"), "w") as outfile:
        json.dump(prms, outfile)

    # %%
    if not eval(os.environ["RENDER"]):

        # %%
        if not os.path.exists(f"workspace/{file_root}_onsets.npy"):
            print(f"processing audio files...")
            main_audio, sr = rosa.load(main_audio_file)  # , offset=offset, duration=duration)

            sig = mm.audio.signal.Signal(main_audio_file, num_channels=1)  # , start=offset, stop=offset + duration)
            sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
            stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, ciruclar_shift=True)
            spec = mm.audio.spectrogram.Spectrogram(stft, ciruclar_shift=True)
            log_filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24)

            onsets = np.sum(
                [
                    mm.features.onsets.high_frequency_content(log_filt_spec),
                    mm.features.onsets.spectral_diff(log_filt_spec),
                    mm.features.onsets.spectral_flux(log_filt_spec),
                    mm.features.onsets.superflux(log_filt_spec),
                    mm.features.onsets.complex_flux(log_filt_spec),
                    mm.features.onsets.modified_kullback_leibler(log_filt_spec),
                    mm.features.onsets.phase_deviation(log_filt_spec),
                    mm.features.onsets.weighted_phase_deviation(spec),
                    mm.features.onsets.normalized_weighted_phase_deviation(spec),
                    mm.features.onsets.complex_domain(spec),
                    mm.features.onsets.rectified_complex_domain(spec),
                ],
                axis=0,
            )
            onsets = np.clip(signal.resample(onsets, num_frames), onsets.min(), onsets.max())

            pitches, magnitudes = rosa.core.piptrack(y=main_audio, sr=sr, hop_length=512, fmin=40, fmax=4000)
            pitches_mean = pitches.mean(0)
            pitches_mean = np.clip(signal.resample(pitches_mean, num_frames), pitches_mean.min(), pitches_mean.max())
            average_pitch = np.average(pitches, axis=0, weights=magnitudes + 1e-8)
            average_pitch = np.clip(
                signal.resample(average_pitch, num_frames), average_pitch.min(), average_pitch.max()
            )

            high_pitch_cutoff = 40
            high_pitches, high_magnitudes = pitches[high_pitch_cutoff:], magnitudes[high_pitch_cutoff:]
            high_pitches_mean = high_pitches.mean(0)
            high_pitches_mean = np.clip(
                signal.resample(high_pitches_mean, num_frames), high_pitches_mean.min(), high_pitches_mean.max(),
            )
            high_average_pitch = np.average(high_pitches, axis=0, weights=high_magnitudes + 1e-8)
            high_average_pitch = np.clip(
                signal.resample(high_average_pitch, num_frames), high_average_pitch.min(), high_average_pitch.max()
            )

            rms = rosa.feature.rms(S=np.abs(rosa.stft(y=main_audio, hop_length=512)))[0]
            rms = np.clip(signal.resample(rms, num_frames), rms.min(), rms.max())
            rms = normalize(rms)

            bass_audio = signal.sosfilt(signal.butter(12, 100, "lp", fs=sr, output="sos"), main_audio)
            bass_spec = np.abs(rosa.stft(bass_audio))
            bass_sum = bass_spec.sum(0)
            bass_sum = np.clip(signal.resample(bass_sum, num_frames), bass_sum.min(), bass_sum.max())
            bass_sum = normalize(bass_sum)

            np.save(f"workspace/{file_root}_onsets.npy", onsets)
            np.save(f"workspace/{file_root}_pitches_mean.npy", pitches_mean)
            np.save(f"workspace/{file_root}_average_pitch.npy", average_pitch)
            np.save(f"workspace/{file_root}_high_pitches_mean.npy", high_pitches_mean)
            np.save(f"workspace/{file_root}_high_average_pitch.npy", high_average_pitch)
            np.save(f"workspace/{file_root}_rms.npy", rms)
            np.save(f"workspace/{file_root}_bass_sum.npy", bass_sum)
        else:
            onsets = np.load(f"workspace/{file_root}_onsets.npy")[start_frame:stop_frame]
            pitches_mean = np.load(f"workspace/{file_root}_pitches_mean.npy")[start_frame:stop_frame]
            average_pitch = np.load(f"workspace/{file_root}_average_pitch.npy")[start_frame:stop_frame]
            high_pitches_mean = np.load(f"workspace/{file_root}_high_pitches_mean.npy")[start_frame:stop_frame]
            high_average_pitch = np.load(f"workspace/{file_root}_high_average_pitch.npy")[start_frame:stop_frame]
            rms = np.load(f"workspace/{file_root}_rms.npy")[start_frame:stop_frame]
            bass_sum = np.load(f"workspace/{file_root}_bass_sum.npy")[start_frame:stop_frame]

        # %%

        def get_latent_loops(base_latent_selection, loop_starting_latents, n_frames, num_loops, smoothing):
            base_latents = []
            for n in range(len(base_latent_selection)):
                for val in np.linspace(0.0, 1.0, int(n_frames // max(1, num_loops) // len(base_latent_selection))):
                    base_latents.append(
                        generate.slerp(
                            val,
                            base_latent_selection[(n + loop_starting_latents) % len(base_latent_selection)][0],
                            base_latent_selection[(n + loop_starting_latents + 1) % len(base_latent_selection)][0],
                        ).numpy()
                    )
            base_latents = gaussian_filter(np.array(base_latents), [smoothing * smf, 0], mode="wrap")
            base_latents = np.concatenate([base_latents] * int(n_frames / len(base_latents)), axis=0)
            base_latents = np.concatenate([base_latents[:, None, :]] * 18, axis=1)
            base_latents = th.from_numpy(base_latents)
            return base_latents

        full_latents = []
        full_high_mod = []
        full_low_mod = []

        # %%
        for t, t_bpm, t_dur, t_start, t_stop in tracks:
            # print(t, t_bpm, t_dur, t_start, t_stop)
            if t_start < 0:
                continue
            if render_track is None:
                rms_slice = rms[t_start:t_stop]
                onsets_slice = onsets[t_start:t_stop]
                average_pitch_slice = average_pitch[t_start:t_stop]
                pitches_slice = pitches_mean[t_start:t_stop]
                high_average_pitch_slice = high_average_pitch[t_start:t_stop]
                high_pitches_slice = high_pitches_mean[t_start:t_stop]
                bass_sum_slice = bass_sum[t_start:t_stop]
            else:
                if t != render_track:
                    continue
                info(rms)
                rms_slice = rms
                onsets_slice = onsets
                average_pitch_slice = average_pitch
                pitches_slice = pitches_mean
                high_average_pitch_slice = high_average_pitch
                high_pitches_slice = high_pitches_mean
                bass_sum_slice = bass_sum
            print(t, t_bpm, t_dur, t_start, t_stop)
            # print(prms[t])

            # %%
            def wrapping_slice(tensor, start, length, return_indices=False):
                if start + length <= tensor.shape[0]:
                    indices = th.arange(start, start + length)
                else:
                    indices = th.cat(
                        (th.arange(start, tensor.shape[0]), th.arange(0, (start + length) % tensor.shape[0]))
                    )
                if tensor.shape[0] == 1:
                    indices = th.zeros(1, dtype=th.int64)
                if return_indices:
                    return indices
                return tensor[indices]

            # %%
            # if t == 0:
            intro_selection = th.from_numpy(np.load(intro_file))
            # gen = render.load_stylegan(checkpoint, 1920)
            # intro_selection = gen(th.randn((len(intro_selection), 512)), noise=None, map_latents=True).cpu()
            # np.save(f"workspace/{file_root}_random_intro_latents.npy", intro_selection.numpy())
            intro_loops = get_latent_loops(
                base_latent_selection=wrapping_slice(intro_selection, t, 16),
                loop_starting_latents=t,
                n_frames=t_stop - t_start,
                num_loops=t_bpm / 60.0 * t_dur / prms[t]["intro_num_beats"],
                smoothing=prms[t]["intro_loop_smoothing"] * smf,
            )
            intro_latents = th.cat(
                [
                    prms[t]["intro_loop_factor"] * intro_loops
                    + (1 - prms[t]["intro_loop_factor"]) * intro_selection[[t % len(intro_selection)], :],
                    prms[t]["intro_loop_factor"] * intro_loops[0 : (t_stop - t_start) - len(intro_loops)]
                    + (1 - prms[t]["intro_loop_factor"]) * intro_selection[[t % len(intro_selection)], :],
                ]
            )

            # if t == 0:
            drop_selection = th.from_numpy(np.load(drop_file))
            # drop_selection = gen(th.randn((len(drop_selection), 512)), noise=None, map_latents=True).cpu()
            # np.save(f"workspace/{file_root}_random_drop_latents.npy", drop_selection.numpy())
            drop_loops = get_latent_loops(
                base_latent_selection=wrapping_slice(drop_selection, t, 16),
                loop_starting_latents=t,
                n_frames=t_stop - t_start,
                num_loops=t_bpm / 60.0 * t_dur / prms[t]["drop_num_beats"],
                smoothing=prms[t]["drop_loop_smoothing"] * smf,
            )
            drop_latents = th.cat(
                [
                    prms[t]["drop_loop_factor"] * drop_loops
                    + (1 - prms[t]["drop_loop_factor"]) * drop_selection[[t % len(drop_selection)], :],
                    prms[t]["drop_loop_factor"] * drop_loops[0 : (t_stop - t_start) - len(drop_loops)]
                    + (1 - prms[t]["drop_loop_factor"]) * drop_selection[[t % len(drop_selection)], :],
                ]
            )

            # %%
            rms_slice = causal_gaussian(rms_slice, [prms[t]["rms_smooth"] * smf])

            main_weight = causal_gaussian(
                normalize(rms_slice) * normalize(onsets_slice), [prms[t]["onset_smooth"] * smf * 86 / t_bpm],
            )
            main_weight = percentile_clip(main_weight, prms[t]["onset_clip"])
            main_weight = th.from_numpy(main_weight)[:, None, None]
            plot_signals([main_weight])

            # %%
            freqs = (average_pitch_slice + pitches_slice) / prms[t]["freq_mod"]
            freqs = causal_gaussian(freqs, [prms[t]["freq_smooth"] * smf])

            plot_signals([average_pitch_slice, pitches_slice, freqs])
            freqs = (freqs + t + prms[t]["freq_mod_offset"]) % (drop_selection.shape[0] - 1)
            freqs = freqs.astype(np.uint8)

            plot_signals([freqs])
            reactive_latents = drop_selection.numpy()[freqs, :, :]
            reactive_latents = gaussian_filter(reactive_latents, [prms[t]["freq_latent_smooth"] * smf, 0, 0])
            reactive_latents = th.from_numpy(reactive_latents)

            layr = prms[t]["freq_latent_layer"]
            drop_latents[:, layr:, :] = (1 - main_weight) * drop_latents[:, layr:, :] + reactive_latents[
                :, layr:, :
            ] * prms[t]["freq_latent_weight"] * main_weight

            # %%
            high_freqs = (high_average_pitch_slice + high_pitches_slice) / prms[t]["high_freq_mod"]
            high_freqs = causal_gaussian(high_freqs, [prms[t]["high_freq_smooth"] * smf])
            plot_signals([high_average_pitch_slice, high_pitches_slice, high_freqs])
            high_freqs = (high_freqs + t + prms[t]["high_freq_mod_offset"]) % (intro_selection.shape[0] - 1)
            high_freqs = high_freqs.astype(np.uint8)

            plot_signals([high_freqs])
            reactive_latents = intro_selection.numpy()[high_freqs, :, :]
            reactive_latents = gaussian_filter(reactive_latents, [prms[t]["high_freq_latent_smooth"] * smf, 0, 0])
            reactive_latents = th.from_numpy(reactive_latents)

            layr = prms[t]["high_freq_latent_layer"]
            intro_latents[:, layr:, :] = (1 - main_weight) * intro_latents[:, layr:, :] + reactive_latents[
                :, layr:, :
            ] * prms[t]["high_freq_latent_weight"] * main_weight

            # %%
            bass_weight = causal_gaussian(bass_sum_slice, [prms[t]["bass_smooth"] * smf])
            bass_weight = percentile_clip(bass_weight, prms[t]["bass_clip"])

            # %%
            drop_weight = rms_slice ** 2 + bass_weight
            drop_weight = percentile_clip(drop_weight, prms[t]["drop_clip"])
            drop_weight = causal_gaussian(drop_weight, [prms[t]["drop_smooth"] * smf])
            drop_weight = normalize(drop_weight) * prms[t]["drop_weight"]
            drop_weight = th.from_numpy(drop_weight)[:, None, None]
            plot_signals([drop_weight])

            # %%
            latents = drop_weight * drop_latents + (1 - drop_weight) * intro_latents

            color_layer = 6
            latents[:, color_layer:, :] = (
                latents[:, color_layer:, :] * 3 / 5 + drop_selection[[t], color_layer:, :] * 2 / 5
            )
            latents[:, color_layer:, :] = (
                latents[:, color_layer:, :] * 3 / 5 + drop_selection[[t], color_layer:, :] * 2 / 5
            )

            full_latents.append(latents)

            # %%
            high_noise_mod = percentile_clip(main_weight.numpy() ** 2, prms[t]["high_noise_clip"])
            high_noise_mod *= prms[t]["high_noise_weight"]
            high_noise_mod = th.from_numpy(high_noise_mod)[:, None].float()
            full_high_mod.append(high_noise_mod)
            plot_signals([high_noise_mod])

            # %%
            low_noise_mod = (1 - drop_weight) * main_weight
            low_noise_mod = normalize(low_noise_mod)
            low_noise_mod *= prms[t]["low_noise_weight"]
            low_noise_mod = low_noise_mod[:, None].float()
            full_low_mod.append(low_noise_mod)
            plot_signals([low_noise_mod])

        # %%
        latents = th.cat(full_latents).numpy()
        high_noise_mod = th.cat(full_high_mod).numpy()
        low_noise_mod = th.cat(full_low_mod).numpy()
        if render_track is None:
            for t, t_bpm, t_dur, t_start, t_stop in tracks:
                if t_start <= 0:
                    continue

                transition_window = prms[t]["transition_window"]
                transin = max(0, int(round(t_start - transition_window)))
                transout = min(len(latents), int(round(t_start + transition_window)))

                # interp from start of transition window to new track start_time to make transition smoother
                transition = []
                linsp = th.linspace(0.0, 1.0, transout - transin)[:, None]
                for val in linsp:
                    transition.append(
                        th.cat(
                            [
                                generate.slerp(
                                    val,
                                    latents[transin, 0],
                                    latents[transout + (1 if t != len(tracks) - 1 else -1), 0],
                                )[None, :]
                            ]
                            * 18,
                            axis=0,
                        )[None, :]
                    )
                transition = th.cat(transition)
                half_len = int(round((transout - transin) / 2))
                trans_linsp = th.cat([th.linspace(0.0, 1.0, half_len), th.linspace(1.0, 0.0, half_len)])[:, None, None]
                latents[transin:transout] = (1 - trans_linsp) * latents[transin:transout] + trans_linsp * transition

                high_noise_mod[transin:transout] = gaussian_filter(high_noise_mod[transin:transout], [fps, 0, 0, 0])
                low_noise_mod[transin:transout] = gaussian_filter(low_noise_mod[transin:transout], [fps, 0, 0, 0])

        latents = th.from_numpy(gaussian_filter(latents, [3.5 * smf, 0, 0]))
        high_noise_mod = th.from_numpy(gaussian_filter(high_noise_mod, [3.5 * smf, 0, 0, 0]))
        low_noise_mod = th.from_numpy(gaussian_filter(low_noise_mod, [2.5 * smf, 0, 0, 0]))

        np.save(latent_file, latents.numpy().astype(np.float32))

        print()
        info(latents)
        info(high_noise_mod)
        info(low_noise_mod)

        # %%
        def generate_noise(num_frames, bpm, duration, high_mod, low_mod, min_h, min_w):
            num_loops = round(bpm * duration / 60 / 4)
            noise_vec = np.random.normal(size=(int(num_frames // num_loops), 1, min_h * 32, min_w * 32))
            loop_len = len(noise_vec)
            num_loops = int(num_frames // loop_len)

            noise_noisy = th.from_numpy(gaussian_filter(noise_vec, [2 * smf, 0, 0, 0], mode="wrap")).float().cuda()
            noise_vox = th.from_numpy(gaussian_filter(noise_vec, [10 * smf, 0, 0, 0], mode="wrap")).float().cuda()
            noise_smooth = th.from_numpy(gaussian_filter(noise_vec, [20 * smf, 0, 0, 0], mode="wrap")).float().cuda()

            # calculate noise values for G
            noise = []
            for power in range(0, 9):
                if power >= noise_scales:
                    noise.append(None)
                    continue

                high_noise = F.interpolate(noise_noisy, size=(2 ** power * min_h, 2 ** power * min_w)).cpu()
                low_noise = F.interpolate(noise_vox, size=(2 ** power * min_h, 2 ** power * min_w)).cpu()
                base_noise = F.interpolate(noise_smooth, size=(2 ** power * min_h, 2 ** power * min_w)).cpu()

                noise_scale = th.zeros((num_frames, *base_noise.shape[1:]))
                for idx in range(0, len(noise_scale), loop_len):
                    len_ns = len(noise_scale[idx : idx + loop_len])
                    noise_scale[idx : idx + loop_len] = high_mod[idx : idx + loop_len] * high_noise[:len_ns]
                    noise_scale[idx : idx + loop_len] += (1 - high_mod[idx : idx + loop_len]) * base_noise[:len_ns]
                    noise_scale[idx : idx + loop_len] += low_mod[idx : idx + loop_len] * low_noise[:len_ns]
                    noise_scale[idx : idx + loop_len] += (1 - low_mod[idx : idx + loop_len]) * base_noise[:len_ns]
                noise.append(noise_scale.float())

            del noise_noisy, noise_vox, noise_smooth
            return noise

        min_w = int(2 if size == 512 else (4 if size == 1024 else 8))
        min_h = int(min_w / 2.0 if size == 1920 else min_w)
        noise = generate_noise(len(latents), bpm, duration, high_noise_mod, low_noise_mod, min_h, min_w)

        [
            np.save(noise_file.replace(".npy", f"{ns}.npy"), noise_scale.numpy().astype(np.float32))
            for ns, noise_scale in enumerate(noise)
            if noise_scale is not None
        ]
        [info(noise_scale) for ns, noise_scale in enumerate(noise) if noise_scale is not None]
    # %%
    else:
        # %%
        print("loading latents...")
        latents = th.from_numpy(np.load(latent_file))
        # info(latents)

        print("loading noise...")
        noise = [th.from_numpy(np.load(noise_file.replace(".npy", f"{ns}.npy"))) for ns in range(noise_scales)]
        noise += [None] * (9 - len(noise))
        # [info(noise_scale) for ns, noise_scale in enumerate(noise) if noise_scale is not None]

        print(f"rendering {num_frames} frames...")
        checkpoint_title = checkpoint.split("/")[-1].split(".")[0].lower()
        track_title = main_audio_file.split("/")[-1].split(".")[0].lower()
        title = f"/home/hans/neurout/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"
        if render_track is not None:
            title.replace(".mp4", f"_track{render_track}.mp4")
        render.generate_video(
            main_audio_file,
            checkpoint,
            latents,
            noise,
            audio_offset=offset,
            audio_duration=duration,
            bpm=bpm,
            size=size,
            batch_size=batch_size,
            output_file=title,
        )
# %%
# import ffmpeg

# input_args = {"hwaccel": "nvdec", "vcodec": "h264_cuvid", "c:v": "h264_cuvid"}

# output_args = {
#     "vcodec": "hevc_nvenc",
#     "c:v": "hevc_nvenc",
#     "preset": "veryslow",
#     "crf": 20,
#     "b:v": "20M",
#     # "vcodec": "libx264"
#     # "preset": "slow"
# }

# # (ffmpeg.input("Input/my_video.mp4", **input_args).output("Output/out.mkv", **output_args).run())

# audio = ffmpeg.input(main_audio_file)  # , acodec="pcm_s16le")
# video = (
#     ffmpeg.input(
#         "pipe:", format="rawvideo", pix_fmt="rgb24", s="128x64", framerate=len(latents) / duration, **input_args
#     )
#     .output("noise_test.mp4", s="1920x1080", framerate=len(latents) / duration, **output_args)
#     .overwrite_output()
#     .run_async(pipe_stdin=True)  # , pipe_stdout=True, pipe_stderr=True)
# )
# # video = (
# #     ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s="128x64", framerate=fps)
# #     .output(
# #         audio, "noise_test.mp4", s="352x240", framerate=fps, vcodec="libx264", preset="fast", audio_bitrate="320K"
# #     )
# #     .overwrite_output()
# #     .run_async(pipe_stdin=True)  # , pipe_stdout=True, pipe_stderr=True)
# # )
# for n in range(len(latents)):
#     video.stdin.write((noise[4][n] * 255).numpy().astype(np.uint8).tobytes())
# video.stdin.close()
# video.wait()

