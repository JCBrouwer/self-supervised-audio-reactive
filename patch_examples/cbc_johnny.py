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
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from pathos.multiprocessing import ProcessingPool as Pool

th.set_grad_enabled(False)
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["figure.facecolor"] = "black"
VERBOSE = False

os.environ["MIX"] = "johnnysmall_cbc_mix2"
os.environ["CKPT"] = "dohr-105"
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

    fps = None
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
        tracks.append(
            [t, int(round(start_time * fac)), int(round(stop_time * fac)), t_bpm, start_time, stop_time,]
        )
        # print(t, int(round(start_time * fac)), int(round(stop_time * fac)), t_bpm, start_time, stop_time)

    if render_track is not None:
        offset = tracks[render_track][-2]
        duration = tracks[render_track][-1] - tracks[render_track][-2]
        num_frames = int(round(duration * fps))

    # %%
    if not eval(os.environ["RENDER"]):
        # %%
        print(f"loading audio files... offset={offset} & duration={duration}")
        main_audio, sr = rosa.load(main_audio_file, offset=offset, duration=duration)

        sig = mm.audio.signal.Signal(main_audio_file, start=offset, stop=offset + duration, num_channels=1)
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
        onsets = signal.resample(onsets, num_frames)

        pitches, magnitudes = rosa.core.piptrack(y=main_audio, sr=sr, hop_length=512, fmin=40, fmax=4000)
        average_pitch = np.average(pitches, axis=0, weights=magnitudes + 1e-8)
        average_pitch = signal.resample(average_pitch, num_frames)

        rms = rosa.feature.rms(S=np.abs(rosa.stft(y=main_audio, hop_length=512)))[0]
        rms = gaussian_filter(rms, [120 * smf])
        rms = signal.resample(rms, num_frames)
        rms = normalize(rms)

        bass_audio = signal.sosfilt(signal.butter(12, 100, "lp", fs=sr, output="sos"), main_audio)
        bass_spec = np.abs(rosa.stft(bass_audio))
        bass_sum = signal.resample(bass_spec.sum(0), num_frames)
        bass_sum = normalize(bass_sum)

        # %%
        prms = {
            "intro_num_beats": dict(zip(range(len(tracks)), [32] * len(tracks))),
            "intro_loop_smoothing": dict(zip(range(len(tracks)), [20] * len(tracks))),
            "intro_loop_factor": dict(zip(range(len(tracks)), [0.7] * len(tracks))),
            "drop_num_beats": dict(zip(range(len(tracks)), [16] * len(tracks))),
            "drop_loop_smoothing": dict(zip(range(len(tracks)), [4] * len(tracks))),
            "drop_loop_factor": dict(zip(range(len(tracks)), [0.5] * len(tracks))),
            "onset_smooth": dict(zip(range(len(tracks)), [2.25] * len(tracks))),
            "onset_clip": dict(zip(range(len(tracks)), [95] * len(tracks))),
            "freq_mod": dict(zip(range(len(tracks)), [10] * len(tracks))),
            "freq_mod_offset": dict(zip(range(len(tracks)), [0] * len(tracks))),
            "freq_smooth": dict(zip(range(len(tracks)), [4] * len(tracks))),
            "freq_latent_smooth": dict(zip(range(len(tracks)), [4] * len(tracks))),
            "freq_latent_layer": dict(zip(range(len(tracks)), [0] * len(tracks))),
            "freq_latent_weight": dict(zip(range(len(tracks)), [1] * len(tracks))),
            "bass_smooth": dict(zip(range(len(tracks)), [120] * len(tracks))),
            "bass_clip": dict(zip(range(len(tracks)), [90] * len(tracks))),
            "drop_clip": dict(zip(range(len(tracks)), [80] * len(tracks))),
            "drop_smooth": dict(zip(range(len(tracks)), [4] * len(tracks))),
            "drop_weight": dict(zip(range(len(tracks)), [1] * len(tracks))),
            "noise_clip": dict(zip(range(len(tracks)), [99] * len(tracks))),
            "high_noise_weight": dict(zip(range(len(tracks)), [2.5] * len(tracks))),
            "low_noise_weight": dict(zip(range(len(tracks)), [0.7] * len(tracks))),
            "transition_window": dict(zip(range(len(tracks)), [int(round(2 * fps))] * len(tracks))),
        }

        prms["drop_clip"][0] = 100
        prms["drop_smooth"][0] = 20
        prms["drop_weight"][0] = 0.5
        prms["intro_num_beats"][0] = 64
        prms["intro_loop_smoothing"][0] = 30
        prms["high_noise_weight"][0] = 1.5

        with open(metadata_file.replace("metadata", "params"), "w") as outfile:
            json.dump(prms, outfile)

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

        # %%
        full_latents = []
        full_high_mod = []
        full_low_mod = []

        # %%
        for t, t_start, t_stop, t_bpm, start_time, stop_time in tracks:
            if render_track is not None and t != render_track:
                continue

            # %%
            intro_selection = th.from_numpy(np.load(intro_file))
            intro_loops = get_latent_loops(
                base_latent_selection=intro_selection,
                loop_starting_latents=t,
                n_frames=t_stop - t_start,
                num_loops=t_bpm / 60.0 * (stop_time - start_time) / prms["intro_num_beats"][t],
                smoothing=prms["intro_loop_smoothing"][t] * smf,
            )
            intro_latents = th.cat(
                [
                    prms["intro_loop_factor"][t] * intro_loops
                    + (1 - prms["intro_loop_factor"][t]) * intro_selection[[t], :],
                    prms["intro_loop_factor"][t] * intro_loops[0 : (t_stop - t_start) - len(intro_loops)]
                    + (1 - prms["intro_loop_factor"][t]) * intro_selection[[t], :],
                ]
            )

            drop_selection = th.from_numpy(np.load(drop_file))
            drop_loops = get_latent_loops(
                base_latent_selection=drop_selection,
                loop_starting_latents=t,
                n_frames=t_stop - t_start,
                num_loops=t_bpm / 60.0 * (stop_time - start_time) / prms["drop_num_beats"][t],
                smoothing=prms["drop_loop_smoothing"][t] * smf,
            )
            drop_latents = th.cat(
                [
                    prms["drop_loop_factor"][t] * drop_loops
                    + (1 - prms["drop_loop_factor"][t]) * drop_selection[[t], :],
                    prms["drop_loop_factor"][t] * drop_loops[0 : (t_stop - t_start) - len(drop_loops)]
                    + (1 - prms["drop_loop_factor"][t]) * drop_selection[[t], :],
                ]
            )

            # %%
            main_weight = gaussian_filter(
                rms[t_start:t_stop] * onsets[t_start:t_stop], [prms["onset_smooth"][t] * smf * 86 / t_bpm],
            )
            main_weight = percentile_clip(main_weight, prms["onset_clip"][t])
            main_weight = th.from_numpy(main_weight)[:, None, None]
            plot_signals([main_weight])

            # %%
            freqs = (average_pitch[t_start:t_stop] + pitches[:, t_start:t_stop].mean(0)) / prms["freq_mod"][t]
            freqs = gaussian_filter(freqs, [prms["freq_smooth"][t] * smf])
            freqs = (freqs + t + prms["freq_mod_offset"][t]) % (drop_selection.shape[0] - 1)
            freqs = freqs.astype(np.uint8)
            plot_signals([freqs])

            # %%
            reactive_latents = drop_selection.numpy()[freqs, :, :]
            reactive_latents = gaussian_filter(reactive_latents, [prms["freq_latent_smooth"][t] * smf, 0, 0])
            reactive_latents = th.from_numpy(reactive_latents)

            # %%
            layr = prms["freq_latent_layer"][t]
            drop_latents[:, layr:, :] = (1 - main_weight) * drop_latents[:, layr:, :] + reactive_latents[
                :, layr:, :
            ] * prms["freq_latent_weight"][t] * main_weight

            # %%
            bass_weight = gaussian_filter(bass_sum[t_start:t_stop], [prms["bass_smooth"][t] * smf])
            bass_weight = percentile_clip(bass_weight, prms["bass_clip"][t])

            # %%
            drop_weight = rms[t_start:t_stop] ** 2 + bass_weight
            drop_weight = percentile_clip(drop_weight, prms["drop_clip"][t])
            drop_weight = gaussian_filter(drop_weight, [prms["drop_smooth"][t] * smf])
            drop_weight = normalize(drop_weight) * prms["drop_weight"][t]
            drop_weight = th.from_numpy(drop_weight)[:, None, None]
            plot_signals([drop_weight])

            # %%
            latents = drop_weight * drop_latents + (1 - drop_weight) * intro_latents
            full_latents.append(latents)

            # %%
            high_noise_mod = percentile_clip(main_weight.numpy() ** 2, prms["noise_clip"][t])
            high_noise_mod *= prms["high_noise_weight"][t]
            high_noise_mod = th.from_numpy(high_noise_mod)[:, None].float()
            full_high_mod.append(high_noise_mod)
            plot_signals([high_noise_mod])

            # %%
            low_noise_mod = (1 - drop_weight) * main_weight
            low_noise_mod = normalize(low_noise_mod)
            low_noise_mod *= prms["low_noise_weight"][t]
            low_noise_mod = low_noise_mod[:, None].float()
            full_low_mod.append(low_noise_mod)
            plot_signals([low_noise_mod])

        # %%
        latents = th.cat(full_latents).numpy()
        high_noise_mod = th.cat(full_high_mod).numpy()
        low_noise_mod = th.cat(full_low_mod).numpy()
        for t, t_start, t_stop, t_bpm, start_time, stop_time in tracks:
            if (render_track is not None and t != render_track) or t_start == 0:
                continue

            transition_window = prms["transition_window"][t]
            transin = max(0, int(round(t_start - transition_window)))
            transout = min(len(latents), int(round(t_start)))

            # interp from start of transition window to new track start_time to make transition smoother
            transition = []
            linsp = th.linspace(0.0, 1.0, transout - transin)[:, None]
            for val in linsp:
                transition.append(
                    th.cat(
                        [
                            generate.slerp(
                                val, latents[transin, 0], latents[transout + (1 if t != len(tracks) - 1 else -1), 0],
                            )[None, :]
                        ]
                        * 18,
                        axis=0,
                    )[None, :]
                )
            transition = th.cat(transition)
            latents[transin:transout] = (1 - linsp[:, None]) * latents[transin:transout] + linsp[:, None] * transition

            high_noise_mod[transin:transout] = gaussian_filter(high_noise_mod[transin:transout], [fps, 0, 0, 0])
            low_noise_mod[transin:transout] = gaussian_filter(low_noise_mod[transin:transout], [fps, 0, 0, 0])

        latents = th.from_numpy(gaussian_filter(latents, [2, 0, 0]))
        high_noise_mod = th.from_numpy(gaussian_filter(high_noise_mod, [2, 0, 0, 0]))
        low_noise_mod = th.from_numpy(gaussian_filter(low_noise_mod, [2, 0, 0, 0]))

        np.save(latent_file, latents.numpy().astype(np.float32))

        info(latents)
        info(high_noise_mod)
        info(low_noise_mod)

        # %%
        def generate_noise(num_frames, bpm, duration, high_mod, low_mod, min_h, min_w):
            num_loops = round(bpm * duration / 60 / 4)
            noise_vec = np.random.normal(size=(int(num_frames // num_loops), 1, min_h * 32, min_w * 32))
            loop_len = len(noise_vec)
            num_loops = int(num_frames // loop_len)

            noise_noisy = th.from_numpy(gaussian_filter(noise_vec, [1.5, 0, 0, 0], mode="wrap")).float().cuda()
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

