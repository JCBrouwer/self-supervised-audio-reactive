import librosa as rosa
import torch
from torch.nn.functional import pad, sigmoid
from torch_dct import dct

from .processing import gaussian_filter, normalize
from .rosa.convert import power_to_db
from .rosa.helpers import sync_agg
from .rosa.spectral import chroma_cens, hpss, istft, melspectrogram, stft


def onset_strength(audio, sr, hop_length=1024, n_fft=2048):
    S = torch.abs(melspectrogram(audio, sr, n_fft=n_fft, hop_length=hop_length, fmax=11025.0))
    S = power_to_db(S)

    onset_env = S[:, 1:] - S[:, :-1]
    onset_env = torch.maximum(torch.zeros(()), onset_env)
    onset_env = sync_agg(onset_env, [slice(None)], aggregate=torch.mean, axis=0)

    pad_width = 1 + n_fft // (2 * hop_length)  # Counter-act framing effects. Shift the onsets by n_fft / hop_length

    onset_env = pad(onset_env, (int(pad_width), 0, 0, 0), mode="constant")
    onset_env = onset_env[:, : S.shape[1]]

    return onset_env.squeeze()


def harmonic(audio, margin=8.0):
    y_stft = stft(audio)
    stft_harm = hpss(y_stft, margin=margin)[0]
    y_harm = istft(stft_harm, length=len(audio))
    return y_harm


def percussive(audio, margin=8.0):
    y_stft = stft(audio)
    stft_perc = hpss(y_stft, margin=margin)[1]
    y_perc = istft(stft_perc, length=len(audio))
    return y_perc


def chromagram(audio, sr):
    harm = harmonic(audio)
    return chroma_cens(harm, sr, hop_length=1024)


def tonnetz(y, sr):
    chroma = chromagram(y, sr)
    dim_map = torch.linspace(0, 12, num=chroma.shape[0], endpoint=False)  # Generate Transformation matrix
    scale = torch.tensor([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])
    V = scale.reshape(-1, 1) * dim_map
    V[::2] -= 0.5  # Even rows compute sin()
    R = torch.tensor([1, 1, 1, 1, 0.5, 0.5])  # Fifths  # Minor  # Major
    phi = R[:, None] * torch.cos(torch.pi * V)
    return phi.dot(chroma / chroma.norm(p=1, dim=0))


def mfcc(y, sr, n_mfcc=20, **kwargs):
    S = power_to_db(melspectrogram(y, sr, **kwargs))
    M = dct(S.T, norm="ortho").T[:n_mfcc]
    return M


if __name__ == "__main__":
    import torchaudio as ta

    audio, sr = ta.load("/home/hans/datasets/wavefunk/Ouroboromorphism_49_109.flac")
    SR = 1024 * 24
    audio, sr = ta.functional.resample(audio, sr, SR), SR
    audio = audio.mean(0).cuda()
    print(audio.shape, sr)

    chromanp = rosa.feature.chroma_cens(audio.squeeze().cpu().numpy(), sr, hop_length=1024)
    print(chromanp.min(), chromanp.mean(), chromanp.max(), chromanp.shape)
    chromath = chromagram(audio, sr)
    print(chromath.min(), chromath.mean(), chromath.max(), chromath.shape)
    print(chromanp - chromath.numpy())

    import matplotlib.pyplot as plt

    fig, ax = plt.suubplots(2, 1, figsize=(12, 6))
    ax[0].imshow(chromath)
    ax[1].imshow(chromanp)
    plt.show()
    exit()

    onsnp = rosa.onset.onset_strength(audio.squeeze().cpu().numpy(), sr, hop_length=1024)
    print(onsnp.min(), onsnp.mean(), onsnp.max(), onsnp.shape)
    onsth = onset_strength(audio, sr)
    print(onsth.min(), onsth.mean(), onsth.max(), onsth.shape)
    print(onsnp - onsth.numpy())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(onsth)
    plt.plot(onsnp)
    plt.show()

exit()


def pulse(audio, sr):
    return rosa.beat.plp(audio, sr, win_length=1024, tempo_min=60, tempo_max=180, hop_length=1024)


def rms(audio):
    return rosa.feature.rms(audio, hop_length=1024)


def drop_strength(audio):
    return sigmoid(normalize(gaussian_filter(rms(audio), 10)) * 10 - 5)


def percussive(audio, margin=8.0):
    return rosa.effects.percussive(audio, margin=margin)


def mfcc(audio, sr):
    return rosa.feature.mfcc(audio, sr, hop_length=1024)


def tonnetz(audio, sr):
    harm = harmonic(audio)
    return rosa.feature.tonnetz(harm, sr, hop_length=1024)


def spectral_contrast(audio, sr):
    return rosa.feature.spectral_contrast(audio, sr, hop_length=1024)


def spectral_flatness(audio, sr):
    return rosa.feature.spectral_flatness(audio, hop_length=1024)


def low_pass(audio, sr):
    pass


def mid_pass(audio, sr):
    pass


def high_pass(audio, sr):
    pass
