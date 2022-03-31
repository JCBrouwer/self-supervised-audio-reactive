import numpy as np
import torch
from torch.nn.functional import conv1d

from ..processing import median_filter_2d
from .convert import cq_to_chroma, hz_to_mel, mel_to_hz


def stft(
    y, n_fft=2048, hop_length=1024, center=True, window=torch.hann_window, pad_mode="reflect", return_complex=True
):
    print(y.shape, y.dtype)
    y_stft = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window(n_fft).to(y),
        center=center,
        pad_mode=pad_mode,
        return_complex=return_complex,
    )
    print(y_stft.shape, y_stft.dtype)
    return y_stft


def istft(y_stft, n_fft=2048, hop_length=1024, center=True, window=torch.hann_window, return_complex=True, length=None):
    print(y_stft.shape, y_stft.dtype)
    return torch.istft(
        y_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window(n_fft).to(y_stft),
        center=center,
        return_complex=return_complex,
        length=length,
    )


def spectrogram(y, n_fft=2048, hop_length=1024, power=1, window=torch.hann_window, center=True, pad_mode="reflect"):
    S = torch.abs(stft(y, n_fft=n_fft, hop_length=hop_length, center=center, window=window, pad_mode=pad_mode)) ** power
    return S


def melspectrogram(
    y, sr, n_fft=2048, hop_length=1024, window=torch.hann_window, center=True, pad_mode="reflect", power=2.0, fmax=None
):
    S = spectrogram(y, n_fft=n_fft, hop_length=hop_length, power=power, window=window, center=center, pad_mode=pad_mode)
    mel_basis = mel(sr, n_fft, fmax=fmax)
    return mel_basis @ S


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = torch.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, dtype=torch.float):
    if fmax is None:
        fmax = float(sr) / 2

    n_mels = int(n_mels)
    weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = torch.linspace(0, float(sr) / 2, int(1 + n_fft // 2))

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = torch.diff(mel_f)

    ramps = mel_f.reshape(-1, 1) - fftfreqs

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = torch.maximum(torch.zeros(()), torch.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]

    return weights


def magphase(D, power=1.0):
    mag = torch.abs(D)
    mag = mag ** power
    phase = torch.exp(1.0j * torch.angle(D))
    return mag, phase


def softmask(X, X_ref, power=torch.ones(()), split_zeros=False):
    # Re-scale the input arrays relative to the larger value
    dtype = torch.float
    Z = torch.maximum(X, X_ref).to(dtype)
    bad_idx = Z < torch.finfo(dtype).tiny
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    if np.isfinite(power):
        mask = (X / Z) ** power
        ref_mask = (X_ref / Z) ** power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask


def hpss(S, ks=31, power=2.0, margin=1.0):
    if torch.is_complex(S):
        S, phase = magphase(S)
    else:
        phase = 1.0

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = torch.empty_like(S)
    harm[:] = median_filter_2d(S[None, None], k=(1, ks), p=(ks // 2, ks // 2, 0, 0)).squeeze()

    perc = torch.empty_like(S)
    perc[:] = median_filter_2d(S[None, None], k=(ks, 1), p=(0, 0, ks // 2, ks // 2)).squeeze()

    split_zeros = margin == 1 and margin == 1
    mask_harm = softmask(harm, perc * margin, power=power, split_zeros=split_zeros)
    mask_perc = softmask(perc, harm * margin, power=power, split_zeros=split_zeros)
    return ((S * mask_harm) * phase, (S * mask_perc) * phase)


from .constantq import cqt


def chroma_cqt(
    y,
    sr,
    hop_length=1024,
    fmin=None,
    threshold=0.0,
    tuning=None,
    n_chroma=12,
    n_octaves=7,
    window=torch.hann_window,
    bins_per_octave=36,
):

    # Build the CQT if we don't have one already
    C = torch.abs(
        cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_octaves * bins_per_octave,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
        )
    )

    # Map to chroma
    cq_to_chr = cq_to_chroma(C.shape[0], bins_per_octave=bins_per_octave, n_chroma=n_chroma, fmin=fmin, window=window)
    chroma = cq_to_chr.dot(C)

    if threshold is not None:
        chroma[chroma < threshold] = 0.0

    return chroma


def chroma_cens(
    y,
    sr,
    hop_length=1024,
    fmin=None,
    tuning=None,
    n_chroma=12,
    n_octaves=7,
    bins_per_octave=36,
    window=None,
    win_len_smooth=41,
    smoothing_window=torch.hann_window,
):
    chroma = chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        n_chroma=n_chroma,
        n_octaves=n_octaves,
        window=window,
    )
    chroma = chroma / torch.norm(chroma, p=1, dim=0)  # L1-Normalization

    QUANT_STEPS = [0.4, 0.2, 0.1, 0.05]  # Quantize amplitudes
    QUANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    chroma_quant = torch.zeros_like(chroma)

    for cur_quant_step_idx, cur_quant_step in enumerate(QUANT_STEPS):
        chroma_quant += (chroma > cur_quant_step) * QUANT_WEIGHTS[cur_quant_step_idx]

    if win_len_smooth:
        win = smoothing_window(win_len_smooth + 2)
        win /= torch.sum(win)
        cens = conv1d(chroma_quant.unsqueeze(0), win, padding="same").squeeze(0)  # Apply temporal smoothing
    else:
        cens = chroma_quant

    return cens / torch.norm(cens, p=2, dim=0)  # L2-Normalization
