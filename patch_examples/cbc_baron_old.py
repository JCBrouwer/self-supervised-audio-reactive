# %%
import os
import uuid
import json
import glob
import tqdm
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
from pathos.multiprocessing import ProcessingPool as Pool

th.set_grad_enabled(False)
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["figure.facecolor"] = "black"
VERBOSE = False

os.environ["MIX"] = "baron_cbc_mix2"
os.environ["CKPT"] = "dohr-013"
# os.environ["GENERATE"] = "True"
# VERBOSE = True


def info(arr):
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")


def percentile_clip(y, percentile):
    peaks = signal.argrelextrema(y, np.greater)
    y = y.clip(0, np.percentile(y[peaks], percentile))
    y /= y.max()
    return y


def normalize(y):
    y -= y.min()
    y /= y.max()
    return y


def causal_gaussian(arr, sigmas):
    # 4x gaussian_filter smooth size cuz g_f defaults to truncate=4, 2x cuz causal only looks at half
    smooth_size = int(4 * sigmas[0])
    kernel = signal.hann(smooth_size)[smooth_size // 2 :]  # causal gaussian kernel
    for _ in range(len(sigmas) - 1):
        kernel = kernel[:, None]
    return signal.convolve(arr, kernel, mode="same") / np.sum(kernel)


def wrap(tensor, start, length):
    if start + length > tensor.shape[0]:
        return th.cat((tensor[start : tensor.shape[0]], tensor[0 : (start + length) % tensor.shape[0]]))
    return tensor[start : start + length]


def plot_signals(signals, vlines=None):
    if not VERBOSE:
        return
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

    cache_pls = True
    load_latents = not eval(os.environ["GENERATE"])
    load_noise = not eval(os.environ["GENERATE"])

    file_root = os.environ["MIX"]
    main_audio_file = f"../../datasets/cantbecancelled/{file_root}.wav"
    bpm = None  # 70

    offset = 0
    duration = 900
    if duration is None:
        duration = rosa.get_duration(filename=main_audio_file)

    metadata_file = f"workspace/{main_audio_file.split('/')[-1].split('.')[0]}_metadata.json"

    if os.path.exists(metadata_file):
        with open(metadata_file) as json_file:
            data = json.load(json_file)

        bpm = data["bpm"]
        total_frames = data["total_frames"]
        raw_track_times = data["tracks"]
    else:
        audio, sr = rosa.load(main_audio_file)
        onset = rosa.onset.onset_strength(audio, sr=sr)
        bpm = rosa.beat.tempo(onset_envelope=onset, sr=sr)[0]
        bpm = np.round(bpm * 10.0) / 10.0

        total_frames = len(onset)
        raw_track_times = [["0:00", bpm]]

        with open(metadata_file, "w") as outfile:
            json.dump({"bpm": bpm, "total_frames": total_frames, "tracks": raw_track_times}, outfile)

    fps = 30
    if fps is None:
        fps = total_frames / rosa.get_duration(filename=main_audio_file)
    smf = fps / 43.066666

    num_frames = int(round(duration * fps))

    # stylegan output size and batch size
    size = 1920
    batch_size = 18

    intro_file = f"workspace/{file_root}_intro_latents.npy"
    drop_file = f"workspace/{file_root}_drop_latents.npy"
    high_noise_mod_file = f"workspace/{file_root}_high_noise_mod.npy"
    low_noise_mod_file = f"workspace/{file_root}_low_noise_mod.npy"

    latent_file = f"workspace/{file_root}_latents.npy"
    noise_file = f"workspace/{file_root}_noise.npy"

    # %%
    if not load_latents:
        print(f"loading audio files... offset={offset} & duration={duration}")
        main_audio, sr = rosa.load(main_audio_file, offset=offset, duration=duration)

        # preprocess track times --> frame indexes
        tracks = []
        for t, (t_time, t_bpm) in enumerate(raw_track_times):
            start_time = int(t_time.split(":")[0]) * 60 + int(t_time.split(":")[1])
            if t + 1 < len(raw_track_times):
                stop_time = int(raw_track_times[t + 1][0].split(":")[0]) * 60 + int(
                    raw_track_times[t + 1][0].split(":")[1]
                )
            else:
                stop_time = duration
            if not (stop_time > offset and start_time < offset + duration):
                continue
            else:
                if start_time < offset:
                    start_time = offset
                if stop_time > offset + duration:
                    stop_time = offset + duration
            start_time -= offset
            stop_time -= offset
            fac = 1.0 / duration * num_frames

            tracks.append([t, int(start_time * fac), int(stop_time * fac), bpm])
            print(t, int(start_time * fac), int(stop_time * fac), t_bpm)

    # %%
    if not load_latents:

        def get_latent_loops(base_latent_selection, loop_starting_latents, num_frames, num_loops, smoothing):
            base_latents = []
            for n in range(len(base_latent_selection)):
                for val in np.linspace(0.0, 1.0, int(num_frames // max(1, num_loops) // len(base_latent_selection))):
                    base_latents.append(
                        generate.slerp(
                            val,
                            base_latent_selection[(n + loop_starting_latents) % len(base_latent_selection)][0],
                            base_latent_selection[(n + loop_starting_latents + 1) % len(base_latent_selection)][0],
                        ).numpy()
                    )
            base_latents = gaussian_filter(np.array(base_latents), [smoothing * smf, 0], mode="wrap")
            base_latents = np.concatenate([base_latents] * int(num_frames / len(base_latents)), axis=0)
            base_latents = np.concatenate([base_latents[:, None, :]] * 18, axis=1)
            base_latents = th.from_numpy(base_latents)
            return base_latents

        def get_loops_per_track(loop_latents_file, tracks, num_beats, loop_smoothing, loop_factor):
            latent_selection = th.from_numpy(np.load(loop_latents_file))
            latents = th.zeros((num_frames, 18, 512))
            transition_times = []
            for t, t_start, t_stop, t_bpm in tracks:
                loops = get_latent_loops(
                    latent_selection,
                    loop_starting_latents=t,
                    num_frames=t_stop - t_start,
                    num_loops=t_bpm / 60.0 * (stop_time - start_time) / num_beats,
                    smoothing=loop_smoothing,
                )

                latents[t_start : t_start + len(loops)] = (
                    loop_factor * loops + (1 - loop_factor) * latent_selection[[t if t > 4 else -(t + 1)], :]
                )

                # loops is shorter due to rounding, so append last partial loop length
                latents[t_start + len(loops) : t_stop] = (
                    loop_factor * loops[0 : t_stop - (t_start + len(loops))]
                    + (1 - loop_factor) * latent_selection[[t if t > 4 else -(t + 1)], :]
                )

                transition_times.append([t_start + len(loops), t_stop])

            # interp from start of transition window to new track start_time to make transition smoother
            transition_window = 200
            for t, (transition_start, transition_end) in enumerate(transition_times):
                transition = []
                linsp = th.linspace(0.0, 1.0, transition_end - (transition_start - transition_window))[:, None]

                for val in linsp:
                    transition.append(
                        th.cat(
                            [
                                generate.slerp(
                                    val,
                                    latents[transition_start - transition_window, 0],
                                    latents[transition_end + (1 if t != len(transition_times) - 1 else -1), 0],
                                )[None, :]
                            ]
                            * 18,
                            axis=0,
                        )[None, :]
                    )
                transition = th.cat(transition)
                latents[transition_start - transition_window : transition_end] = (1 - linsp[:, None]) * latents[
                    transition_start - transition_window : transition_end
                ] + linsp[:, None] * transition
            return latents, latent_selection

        intro_latents, intro_selection = get_loops_per_track(
            intro_file, tracks, num_beats=32, loop_smoothing=20 * smf, loop_factor=0.7
        )
        drop_latents, drop_selection = get_loops_per_track(
            intro_file, tracks, num_beats=16, loop_smoothing=4 * smf, loop_factor=0.5
        )

        # %%
        sig = mm.audio.signal.Signal(main_audio_file, start=offset, stop=offset + duration, num_channels=1)
        frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
        stft = mm.audio.stft.ShortTimeFourierTransform(frames, ciruclar_shift=True)
        spec = mm.audio.spectrogram.Spectrogram(stft, ciruclar_shift=True)
        log_filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24)
        onset = np.sum(
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
        onset = signal.resample(onset, num_frames)

        # %%
        rms = gaussian_filter(rosa.feature.rms(S=np.abs(rosa.stft(y=main_audio, hop_length=512)))[0], [120 * smf])
        rms = signal.resample(rms, num_frames)
        rms = normalize(rms)

        main_weight = np.zeros_like(onset)
        for t, t_start, t_stop, t_bpm in tracks:
            main_weight[t_start:t_stop] = gaussian_filter(
                rms[t_start:t_stop] * onset[t_start:t_stop], [2 * smf * 86 / t_bpm]
            )
            main_weight[t_start:t_stop] = percentile_clip(main_weight[t_start:t_stop], 95)
        main_weight = gaussian_filter(main_weight, [2 * smf])
        main_weight = th.from_numpy(main_weight)[:, None, None]
        plot_signals([main_weight])

        pitches, magnitudes = rosa.core.piptrack(y=main_audio, sr=sr, hop_length=512, fmin=40, fmax=4000)
        freqs = np.average(pitches, axis=0, weights=magnitudes + 1e-8)
        freqs = signal.resample(pitches.mean(0), num_frames)
        freqs /= 10
        freqs = gaussian_filter(freqs, [3 * smf])
        plot_signals([freqs])

        step_fun_per_track = np.zeros((num_frames))
        for t, t_start, t_stop, t_bpm in tracks:
            step_fun_per_track[t_start:] += 1
        freqs += step_fun_per_track
        freqs = freqs % (drop_selection.shape[0] - 1)
        freqs = freqs.astype(np.uint8)
        plot_signals([freqs])

        reactive_latents = drop_selection.numpy()[freqs, :, :]
        reactive_latents = gaussian_filter(reactive_latents, [6 * smf, 0, 0])
        reactive_latents = th.from_numpy(reactive_latents)

        main_layers = [3, 18]
        drop_latents[:, main_layers[0] : main_layers[1], :] = (1 - main_weight) * drop_latents[
            :, main_layers[0] : main_layers[1], :
        ] + reactive_latents[:, main_layers[0] : main_layers[1], :] * 1.5 * main_weight

        # %%
        bass_audio = signal.sosfilt(signal.butter(12, 100, "lp", fs=sr, output="sos"), main_audio)
        bass_spec = np.abs(rosa.stft(bass_audio))
        bass_spec = normalize(bass_spec)
        bass_weight = signal.resample(bass_spec.sum(0), num_frames)
        bass_weight = gaussian_filter(bass_weight, [180 * smf])
        bass_weight = percentile_clip(bass_weight, 80)

        # %%
        drop_weight = rms ** 2 + bass_weight

        # win_size = 100
        for t, t_start, t_stop, t_bpm in tracks:
            # win_start = max(0, t_start - win_size)
            # win_stop = min(len(drop_weight), t_stop + win_size)
            # drop_weight[t_start:t_stop] = percentile_clip(drop_weight[win_start:win_stop], 50)[
            #     t_start
            #     - win_start
            #     - (1 if t_stop == win_stop else 0) : -(win_stop - t_stop)
            #     - (1 if t_stop == win_stop else 0)
            # ]
            drop_weight[t_start:t_stop] = percentile_clip(drop_weight[t_start:t_stop], 70)
        drop_weight = gaussian_filter(drop_weight, [45 * smf])
        plot_signals([drop_weight])
        drop_weight = th.from_numpy(drop_weight)[:, None, None]

        # %%
        latents = drop_weight * drop_latents + (1 - drop_weight) * intro_latents
        latents = gaussian_filter(latents, [2 * smf, 0, 0])
        latents = th.from_numpy(latents).float()

        high_noise_mod = percentile_clip(main_weight.numpy() ** 2, 95)
        high_noise_mod *= 1
        high_noise_mod = th.from_numpy(high_noise_mod)[:, None].float()
        plot_signals([high_noise_mod])

        low_noise_mod = (1 - drop_weight) * main_weight
        low_noise_mod = normalize(low_noise_mod)
        low_noise_mod = low_noise_mod[:, None].float()
        plot_signals([1 - drop_weight, low_noise_mod])

        # %%
        if cache_pls:
            print("saving latents...")
            np.save(latent_file, latents.numpy().astype(np.float32))
            np.save(high_noise_mod_file, high_noise_mod.numpy().astype(np.float32))
            np.save(low_noise_mod_file, low_noise_mod.numpy().astype(np.float32))
        # %%
    else:
        # %%
        print("loading latents...")
        latents = th.from_numpy(np.load(latent_file))
        if not load_noise:
            high_noise_mod = th.from_numpy(np.load(high_noise_mod_file))
            low_noise_mod = th.from_numpy(np.load(low_noise_mod_file))

    # %%
    # generate noise
    noise_scales = 7
    if not load_noise:
        print("generating noise...")

        def generate_noise(num_frames, bpm, duration, high_mod, low_mod, min_h, min_w):
            num_loops = round(bpm * duration / 60 / 4)
            noise_vec = np.random.normal(size=(int(num_frames // num_loops), 1, min_h * 32, min_w * 32))
            loop_len = len(noise_vec)
            num_loops = int(num_frames // loop_len)

            noise_noisy = th.from_numpy(gaussian_filter(noise_vec, [2, 0, 0, 0], mode="wrap")).float().cuda()
            noise_vox = th.from_numpy(gaussian_filter(noise_vec, [10, 0, 0, 0], mode="wrap")).float().cuda()
            noise_smooth = th.from_numpy(gaussian_filter(noise_vec, [20, 0, 0, 0], mode="wrap")).float().cuda()

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
        noise = generate_noise(num_frames, bpm, duration, high_noise_mod, low_noise_mod, min_h, min_w)

        if cache_pls:
            print("saving noise...")
            [
                np.save(noise_file.replace(".npy", f"{ns}.npy"), noise_scale.numpy().astype(np.float32))
                for ns, noise_scale in enumerate(noise)
                if noise_scale is not None
            ]
            # [info(noise_scale) for ns, noise_scale in enumerate(noise) if noise_scale is not None]
    # %%
    else:
        # %%
        print("loading noise...")
        noise = [th.from_numpy(np.load(noise_file.replace(".npy", f"{ns}.npy"))) for ns in range(noise_scales)]
        noise += [None] * (9 - len(noise))
        # [info(noise_scale) for ns, noise_scale in enumerate(noise) if noise_scale is not None]

    # %%
    if not eval(os.environ["GENERATE"]):
        print(f"rendering {num_frames} frames...")
        checkpoint_title = checkpoint.split("/")[-1].split(".")[0].lower()
        track_title = main_audio_file.split("/")[-1].split(".")[0].lower()
        title = f"/home/hans/neurout/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"
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

