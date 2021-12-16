import librosa as rosa
import madmom as mm
import numpy as np
import torch
from scipy.signal import convolve2d, spectrogram
from scipy.sparse import coo_matrix
from skimage.transform import resize

from visual_beats import normalize, video_onsets

SR = 16384


def fourier_tempogram(audio=None, onsets=None):
    if onsets is not None:
        return (
            torch.abs(torch.from_numpy(rosa.feature.fourier_tempogram(onset_envelope=onsets, sr=SR)))
            .float()
            .permute(1, 0)
        )
    return torch.abs(torch.from_numpy(rosa.feature.fourier_tempogram(y=audio, sr=SR))).float().permute(1, 0)


def tempogram(audio=None, onsets=None):
    if onsets is not None:
        return torch.from_numpy(rosa.feature.tempogram(onset_envelope=onsets, sr=SR)).float().permute(1, 0)
    return torch.from_numpy(rosa.feature.tempogram(y=audio, sr=SR)).float().permute(1, 0)


def tonnetz(audio):
    return torch.from_numpy(rosa.feature.tonnetz(y=audio, sr=SR)).float().permute(1, 0)


def mfcc(audio):
    return torch.from_numpy(rosa.feature.mfcc(y=audio, sr=SR)).float().permute(1, 0)


def onsets(audio, margin=8, fmin=40, fmax=16384, type="mm"):
    y_perc = rosa.effects.percussive(y=audio, margin=margin)
    if type == "rosa":
        onset = rosa.onset.onset_strength(y=y_perc, sr=SR, fmin=fmin, fmax=fmax)
    elif type == "mm":
        sig = mm.audio.signal.Signal(y_perc, num_channels=1, sample_rate=SR)
        sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=512)
        stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, circular_shift=True)
        spec = mm.audio.spectrogram.Spectrogram(stft, circular_shift=True)
        filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24, fmin=fmin, fmax=fmax)
        onset = np.sum(
            [
                normalize(mm.features.onsets.spectral_diff(filt_spec)),
                normalize(mm.features.onsets.spectral_flux(filt_spec)),
                normalize(mm.features.onsets.superflux(filt_spec)),
                normalize(mm.features.onsets.complex_flux(filt_spec)),
                normalize(mm.features.onsets.modified_kullback_leibler(filt_spec)),
            ],
            axis=0,
        )[None]
    onset = torch.from_numpy(onset).float().permute(1, 0)
    onset = torch.clamp(onset, 0, torch.quantile(onset, 0.97))
    onset = normalize(onset)
    return onset


def rms(audio):
    return torch.from_numpy(rosa.feature.rms(audio, SR)).float().permute(1, 0)


def raw_chroma(audio, type="avg", nearest_neighbor=True):
    if type == "cens":
        ch = rosa.feature.chroma_cens(y=audio, sr=SR, fmin=40)
    elif type == "cqt":
        ch = rosa.feature.chroma_cqt(y=audio, sr=SR, fmin=40)
    elif type == "stft":
        ch = rosa.feature.chroma_stft(y=audio, sr=SR)
    elif type == "deep":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=SR)
        ch = mm.audio.chroma.DeepChromaProcessor().process(sig).T
    elif type == "clp":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=SR)
        ch = mm.audio.chroma.CLPChromaProcessor().process(sig).T
    elif type == "avg":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=SR)
        cens = rosa.feature.chroma_cens(y=audio, sr=SR, fmin=40)
        ch = np.mean(
            np.stack(
                [
                    cens,
                    rosa.feature.chroma_cqt(y=audio, sr=SR, fmin=40),
                    rosa.feature.chroma_stft(y=audio, sr=SR),
                    resize(mm.audio.chroma.DeepChromaProcessor().process(sig).T, cens.shape),
                    resize(mm.audio.chroma.CLPChromaProcessor().process(sig).T, cens.shape),
                ]
            ),
            axis=0,
        )
    else:
        raise Exception(f"Uknown type: {type}")

    if nearest_neighbor:
        ch = np.minimum(ch, rosa.decompose.nn_filter(ch, aggregate=np.median, metric="cosine"))

    return ch


def chroma(audio, margin=8, type="avg"):
    y_harm = rosa.effects.harmonic(y=audio, margin=margin)
    chroma = raw_chroma(y_harm, type=type)
    chroma = torch.from_numpy(chroma).float().permute(1, 0)
    return chroma


def hpcp(audio):
    return torch.from_numpy(raw_hpcp(audio, sr=SR)).float()


# The MIT License (MIT)

# Copyright (c) 2015 jvbalen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def raw_hpcp(
    y,
    sr,
    win_size=4096,
    hop_size=512,
    window="blackman",
    precision="float32",
    f_min=100,
    f_max=5000,
    global_thr=80,  # in dB below the highest peak
    local_thr=30,  # in dB below 0
    bins_per_octave=12,
    whitening=True,
    filter_width=1 / 3.0,  # in octaves
    harmonic_decay=0.6,
    harmonic_tolerance=2 / 3.0,  # in semitones
    norm_frames=False,
    final_thr=0.0,
):

    # spectrogram
    Y, k, f, t = stft(y, sr, win_size=win_size, hop_size=hop_size, window=window, precision=precision)

    # prune spectrogram to [f_min, f_max]
    Y_lim, k, f = prune_spectrogram(Y, k, f, f_min=f_min, f_max=f_max)

    # threshold spectrogram based on dB magnitudes
    Y_dB = dB(Y_lim)
    Y_thr = global_thresholding(Y_dB, thr=global_thr)
    if local_thr < global_thr:
        Y_thr = local_thresholding(Y_thr, thr=local_thr)

    # peak interpolation
    Y_pks, F, peaks = spectral_peaks(Y_thr, k, sr, win_size)

    # multi-octave pitch profile based on linear magnitudes
    Y_lin = lin_mag(Y_pks, global_thr)
    pp = pitch_profile(Y_lin, F, peaks, bins_per_octave)
    if whitening:
        pp = whiten(pp, bins_per_octave=bins_per_octave, filter_width=filter_width)

    # harmonic summation
    hpp = sum_harmonics(
        pp, harmonic_decay=harmonic_decay, harmonic_tolerance=harmonic_tolerance, bins_per_octave=bins_per_octave
    )

    # fold to chromagram/hpcp
    pcp = fold_octaves(hpp, bins_per_octave=bins_per_octave)
    if norm_frames:
        pcp = normalize_frames(pcp, final_thr)

    return pcp


def stft(x, sr, win_size=4096, hop_size=1024, window="blackman", precision="float32"):
    f, t, X = spectrogram(x, sr, nperseg=win_size, noverlap=win_size - hop_size, window=window)
    X = X.astype(precision).T

    # keep bin numbers k
    k = np.arange(len(f))
    return X, k, f, t


def prune_spectrogram(X, k, f, f_min=100, f_max=5000):
    f_band = np.all([f > f_min, f < f_max], axis=0)
    return X[:, f_band], k[f_band], f[f_band]


def bin2hz(k, sr, win_size):
    return k * sr / win_size


def dB(x):
    return 20.0 * np.log10(x)


def global_thresholding(X, thr=80):
    X = X - np.max(X) + thr
    X[X < 0] = 0
    return X


def local_thresholding(X, thr=30):
    n_frames, n_bins = X.shape
    X[X < np.tile(np.max(X, axis=1).reshape((-1, 1)) - thr, (1, n_bins))] = 0
    return X


def spectral_peaks(X, k, sr, win_size):
    n_frames, n_bins = X.shape
    precision = X.dtype

    A1 = np.zeros((n_frames, n_bins), dtype=precision)
    A2 = np.zeros((n_frames, n_bins), dtype=precision)
    A3 = np.zeros((n_frames, n_bins), dtype=precision)
    A1[:, 1:-1] = X[:, :-2]
    A2[:, 1:-1] = X[:, 1:-1]
    A3[:, 1:-1] = X[:, 2:]
    peaks = np.all([A2 > A1, A2 > A3], axis=0)

    # Bin number of each peak
    K = k * peaks

    # Compute deviations D of spectral peaks, in bins
    D = np.zeros((n_frames, n_bins), dtype=precision)
    D[peaks] = 0.5 * (A1[peaks] - A3[peaks]) / (A1[peaks] - 2 * A2[peaks] + A3[peaks])

    # Vompute adjusted frequencies and amplitudes
    F = bin2hz(K + D, sr, win_size)
    A = np.zeros((n_frames, n_bins), dtype=precision)
    A[peaks] = A2[peaks] - D[peaks] / 4 * (A1[peaks] - A3[peaks])

    return A, F, peaks


def lin_mag(x, x_max):
    return 10 ** ((x - x_max) / 20)


def pitch_profile(X, F, peaks, bins_per_octave):
    n_frames, n_bins = X.shape
    T = np.ones((n_frames, n_bins)) * np.arange(n_frames).reshape((-1, 1))  # t in frames, not seconds
    pitch = hz2midi(F)
    pitch_in_bins = bins_per_octave * pitch / 12

    # fill sparse matrix with spectral peak amplitudes in the right bins
    pp = coo_matrix((X[peaks], (T[peaks].astype(int), pitch_in_bins[peaks].astype(int))))
    return pp.toarray()


def hz2midi(f):
    m = np.zeros(f.shape)
    m[f > 0] = 69 + 12.0 * np.log2(f[f > 0] / 440)
    return m


def whiten(X, bins_per_octave, filter_width=1 / 3.0):
    filter_width_in_bins = int(bins_per_octave * filter_width)

    # moving average filter kernel
    filter_kernel = np.ones((1, filter_width_in_bins), dtype=X.dtype)
    filter_kernel = filter_kernel / np.sum(filter_kernel)

    # subtract moving average
    X = X - convolve2d(X, filter_kernel, mode="same")
    X[X < 0] = 0
    return X


def sum_harmonics(X, harmonic_decay=0.6, harmonic_tolerance=1, bins_per_octave=120):
    w = harmonic_summation_kernel(
        harmonic_decay=harmonic_decay, harmonic_tolerance=harmonic_tolerance, bins_per_octave=bins_per_octave
    )
    w = w.astype(X.dtype).reshape((1, -1))

    # sum harmonics in X using convolution with precomputed kernel w
    return convolve2d(X, w, mode="same")


def harmonic_summation_kernel(harmonic_decay=0.6, harmonic_tolerance=1, bins_per_octave=120, n_octaves=4):
    # f/f0 (log, in octaves) for a linspace of constant Q bins symmetrically around f0
    f_ratio_octaves = 1.0 * np.arange(-n_octaves * bins_per_octave, n_octaves * bins_per_octave + 1) / bins_per_octave

    # f/f0 (in Hz)
    f_ratio = 2 ** f_ratio_octaves

    # harmonic number and harmonic deviation
    n_harm = np.round(f_ratio)
    d_harm = abs(f_ratio - n_harm)

    w = cosine_window(d_harm, tol=harmonic_tolerance) * attenuation(n_harm, r=harmonic_decay)
    return w / np.sum(w)


def attenuation(n, r=0.6):
    n = np.array(np.round(n))
    w = np.zeros(n.shape)
    w[n > 0] = r ** (n[n > 0] - 1)
    return w


def cosine_window(d, tol=1.0):
    # width of the cosine-weighted window around each of the harmonics
    width = np.log(2 ** (tol / 12.0))
    w = np.zeros(d.shape)
    w[d < width] = np.cos(d[d < width] * (np.pi / 2) / width) ** 2
    return w


def fold_octaves(X, bins_per_octave):
    n_frames, n_bins = X.shape

    # fold multi-octave pitch profile at every C
    folds = np.arange(0, n_bins, bins_per_octave)  # every C
    return np.array([X[:, fold : fold + bins_per_octave] for fold in folds[:-1]]).sum(axis=0)


def normalize_frames(X, thr):
    X = X - np.min(X, axis=1).reshape((-1, 1))
    X_max = np.max(X, axis=1)
    X = X[X_max > 0] / (X_max[X_max > 0]).reshape((-1, 1))
    if thr > 0:
        X = (1 - thr) * (X - thr) * (X > thr)
    return X


if __name__ == "__main__":
    import sys

    import librosa.display
    import matplotlib
    import torchaudio
    import torchvision as tv

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video, audio, info = tv.io.read_video(sys.argv[1])
    print(video.shape, audio.shape, info)
    video = tv.transforms.functional.resize(video.permute(0, 3, 1, 2), 128).permute(0, 2, 3, 1).float().div(255)
    audio = torchaudio.transforms.Resample(info["audio_fps"], 16384)(audio).mean(0).numpy()

    ftempo = fourier_tempogram(audio)
    tempo = tempogram(audio)
    ons = onsets(audio)
    vid_ons = video_onsets(video)
    vid_tempo = tempogram(onsets=vid_ons)
    vid_ftempo = fourier_tempogram(onsets=vid_ons)
    chrom = chroma(audio)
    hpc = hpcp(audio)
    tonn = tonnetz(audio)
    mfc = mfcc(audio)
    vol = rms(audio)

    print(f"audio fourier tempogram {ftempo.shape}")
    print(f"audio tempogram {tempo.shape}")
    print(f"audio onsets {ons.shape}")
    print(f"video tempogram {vid_tempo.shape}")
    print(f"video fourier tempogram {vid_ftempo.shape}")
    print(f"video onsets {vid_ons.shape}")
    print(f"chroma {chrom.shape}")
    print(f"hpcp {hpc.shape}")
    print(f"rms {vol.shape}")
    print(f"tonnetz {tonn.shape}")
    print(f"mfcc {mfc.shape}")

    fig, ax = plt.subplots(4, 3, figsize=(64, 48))
    ax = ax.flatten()
    librosa.display.specshow(ftempo.permute(1, 0).numpy(), sr=SR, y_axis="fourier_tempo", ax=ax[0], cmap="magma")
    librosa.display.specshow(tempo.permute(1, 0).numpy(), sr=SR, y_axis="tempo", ax=ax[1], cmap="magma")
    ax[2].plot(ons.numpy().squeeze())
    ax[2].set_ylabel("onsets")
    librosa.display.specshow(vid_ftempo.permute(1, 0).numpy(), sr=SR, y_axis="fourier_tempo", ax=ax[3], cmap="magma")
    librosa.display.specshow(vid_tempo.permute(1, 0).numpy(), sr=SR, y_axis="tempo", ax=ax[4], cmap="magma")
    ax[5].plot(vid_ons.numpy().squeeze())
    ax[5].set_ylabel("video onsets")
    librosa.display.specshow(chrom.permute(1, 0).numpy(), sr=SR, y_axis="chroma", ax=ax[6], cmap="magma")
    librosa.display.specshow(hpc.permute(1, 0).numpy(), sr=SR, ax=ax[7], cmap="magma")
    ax[7].set_ylabel("hpcp")
    ax[8].plot(vol.numpy().squeeze())
    ax[8].set_ylabel("rms")
    librosa.display.specshow(tonn.permute(1, 0).numpy(), sr=SR, y_axis="tonnetz", ax=ax[9], cmap="magma")
    librosa.display.specshow(mfc.permute(1, 0).numpy(), sr=SR, ax=ax[10], cmap="magma")
    ax[10].set_ylabel("mfcc")
    ax[11].axis("off")
    plt.tight_layout()
    plt.savefig("output/audiovisual_features.jpg")
