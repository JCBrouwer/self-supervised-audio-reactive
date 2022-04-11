import random
import sys
import traceback
from glob import glob
from uuid import uuid4

import decord as de
import numpy as np
import torch
import torchaudio as ta
import torchvision as tv
from einops import rearrange
from ssar.features.processing import emphasize, high_pass, low_pass, mid_pass, normalize
from ssar.features.rosa.beat import onset_strength, plp
from ssar.supervised.data import audio2features, gaussian_filter
from ssar.supervised.latent_augmenter import LatentAugmenter, spline_loop_latents
from torch.nn.functional import interpolate
from torchvision.transforms.functional import resize
from tqdm import tqdm

from .audio import (
    chromagram,
    drop_strength,
    harmonic,
    mfcc,
    onsets,
    percussive,
    rms,
    spectral_contrast,
    spectral_flatness,
    tonnetz,
)

sys.path.append("../maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2

de.bridge.set_bridge("torch")


ckpt = "cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl"
G = StyleGAN2(ckpt)
G = G.eval().cuda()
LA = LatentAugmenter(ckpt, n_patches=5)
FPS = 24
DUR = 16
N = DUR * FPS
SR = 1024 * FPS


@torch.inference_mode()
def get_random_audio():
    audio_file = random.choice(glob("/home/hans/datasets/audiovisual/maua256/*"))
    a = de.AudioReader(audio_file)
    audio = a[:].squeeze()
    sr = round(len(audio) / a.duration())
    del a
    audio = audio.contiguous().cuda()
    offset_time = np.random.rand() * (len(audio) / sr - DUR)
    audio = audio[round(offset_time * sr) : round((offset_time + DUR) * sr)]
    audio = ta.functional.resample(audio, sr, SR)
    return audio, SR


@torch.inference_mode()
def get_video(latents, noise):
    noises = G.synthesizer.make_noise_pyramid(noise)
    noises["noise0"] = noises["noise0"][[0]].tile(len(noises["noise0"]), 1, 1, 1)
    inputs = {"latents": latents, **noises}
    resize_fn = lambda x: resize(x, size=256, antialias=True)
    video = torch.cat([frames.cpu() for frames in G.render(inputs=inputs, batch_size=16, postprocess_fn=resize_fn)])
    return video


@torch.inference_mode()
def low_correlation():
    audio, sr = get_random_audio()

    zs = torch.randn((np.random.randint(DUR // 8, DUR), 512), device="cuda")
    ws = G.mapper(zs)

    latents = spline_loop_latents(ws, N)

    noise = torch.randn((np.random.randint(DUR // 8, DUR), 1, 16 * 16))
    noise = spline_loop_latents(noise, N).reshape(N, 1, 16, 16)

    return audio, sr, get_video(latents, noise), FPS


@torch.inference_mode()
def audio2features(audio, sr, clamp=True, smooth=True):
    audio_harm, audio_perc = harmonic(audio), percussive(audio)
    multi_features = [
        gaussian_filter(mfcc(audio, sr), 10),
        gaussian_filter(chromagram(audio_harm, sr), 10),
        gaussian_filter(tonnetz(audio_harm, sr), 10),
        gaussian_filter(spectral_contrast(audio, sr), 10),
    ]
    single_features = [
        gaussian_filter(spectral_flatness(audio, sr), 10),
        onset_strength(audio_perc, sr),
        onset_strength(low_pass(audio_perc, sr), sr),
        onset_strength(mid_pass(audio_perc, sr), sr),
        onset_strength(high_pass(audio_perc, sr), sr),
        plp(audio, sr),
        rms(audio_harm, sr),
        rms(low_pass(audio_harm, sr), sr),
        rms(mid_pass(audio_harm, sr), sr),
        rms(high_pass(audio_harm, sr), sr),
        drop_strength(audio, sr),
        drop_strength(low_pass(audio, sr), sr),
        drop_strength(mid_pass(audio, sr), sr),
        drop_strength(high_pass(audio, sr), sr),
    ]
    features = multi_features + [sf.reshape(-1, 1) for sf in single_features]
    features = torch.cat(features, axis=1)
    if clamp:
        features = features.clamp(torch.quantile(features, q=0.1, dim=0), torch.quantile(features, q=0.975, dim=0))
    if smooth:
        features = gaussian_filter(features, 2)
    return features


@torch.inference_mode()
def medium_correlation():
    audio, sr = get_random_audio()

    audio_features = audio2features(audio, sr).unsqueeze(0)

    residual, offset = LA(audio_features)

    latents = (offset + residual).squeeze()

    noise = torch.randn((np.random.randint(DUR // 8, DUR), 1, 16 * 16))
    noise = spline_loop_latents(noise, N).reshape(N, 1, 16, 16)

    return audio, sr, get_video(latents, noise), FPS


@torch.inference_mode()
def test_set_correlation():
    video_file = random.choice(glob("/home/hans/datasets/audiovisual/maua256/*"))

    v = de.VideoReader(video_file)
    fps = v.get_avg_fps()
    h, w, _ = v[0].shape
    sr = round(1024 * fps)
    del v

    av = de.AVReader(video_file, sample_rate=sr, height=256, width=round(w / h * 256))
    audio, video = av[:]
    del av

    audio = torch.cat(audio, dim=1).squeeze().contiguous().cuda()
    video = video.permute(0, 3, 1, 2).div(255).contiguous().cuda()

    offset_time = np.random.rand() * (len(audio) / sr - DUR)

    video = video[round(offset_time * fps) : round((offset_time + DUR) * fps)]
    video = rearrange(video, "T C H W -> (C H) W T")
    video = interpolate(video, size=N, mode="linear", align_corners=False)
    video = rearrange(video, "(C H) W T -> T C H W", C=3)

    audio = audio[round(offset_time * sr) : round((offset_time + DUR) * sr)]
    audio = ta.functional.resample(audio, sr, 1024 * FPS)

    return audio, 1024 * FPS, video, FPS


@torch.inference_mode()
def high_chroma_correlation():
    audio, sr = get_random_audio()

    chroma = chromagram(audio, sr)
    chroma = gaussian_filter(chroma, FPS / 24)
    chroma = emphasize(chroma, strength=3, percentile=75)
    chroma /= chroma.sum(1, keepdim=True)

    ws = G.mapper(torch.randn((12, 512), device="cuda"))

    latents = torch.einsum("TC,CNL->TNL", chroma, ws)
    latents = gaussian_filter(latents, FPS / 24)

    noise = gaussian_filter(torch.randn((len(chroma), 1, 32, 32), device="cuda"), 3 * FPS)
    noise /= noise.std()

    return audio, sr, get_video(latents, noise), FPS


@torch.inference_mode()
def high_onset_correlation():
    audio, sr = get_random_audio()

    ons = onsets(audio, sr).squeeze()
    ons = gaussian_filter(ons, FPS / 24)
    ons = emphasize(ons, strength=2, percentile=75)
    ons = normalize(ons)

    ws = G.mapper(torch.randn((2, 512), device="cuda"))

    latents = ws[[0]] * ons.reshape(-1, 1, 1) + ws[[1]] * (1 - ons.reshape(-1, 1, 1))
    latents = gaussian_filter(latents, FPS / 24)

    noise = gaussian_filter(torch.randn((len(ons), 1, 64, 64), device="cuda"), 3 * FPS / 24)
    noise /= noise.std()
    noise *= ons.reshape(-1, 1, 1, 1)

    return audio, sr, get_video(latents, noise), FPS


@torch.inference_mode()
def high_both_correlation():
    audio, sr = get_random_audio()

    chroma = chromagram(audio, sr)
    chroma = gaussian_filter(chroma, FPS / 24)
    chroma = emphasize(chroma, strength=3, percentile=75)
    chroma /= chroma.sum(1, keepdim=True)

    ons = onsets(audio, sr).squeeze()
    ons = gaussian_filter(ons, FPS / 24)
    ons = emphasize(ons, strength=2, percentiel=75)
    ons = normalize(ons)

    ws = G.mapper(torch.randn((14, 512), device="cuda"))

    chroma_latents = torch.einsum("TC,CNL->TNL", chroma, ws[:12])
    onset_latents = ws[[12]] * ons.reshape(-1, 1, 1) + ws[[13]] * (1 - ons.reshape(-1, 1, 1))
    latents = (chroma_latents + onset_latents) / 2
    latents = gaussian_filter(latents, FPS / 24)

    noise = torch.randn((len(ons), 1, 64, 64), device="cuda")
    noise = gaussian_filter(noise, 3 * FPS / 24)
    noise = noise / noise.std()
    noise = noise * ons.reshape(-1, 1, 1, 1)

    return audio, sr, get_video(latents, noise), FPS


if __name__ == "__main__":
    od = "output/maua_correlation_test2"
    groups = [
        low_correlation,
        medium_correlation,
        test_set_correlation,
        high_chroma_correlation,
        high_onset_correlation,
        high_both_correlation,
    ]
    num = 100
    with torch.inference_mode(), tqdm(total=len(groups) * num) as progress:
        for i in range(100):
            for correlation in groups:
                try:
                    audio, sr, video, fps = correlation()
                    audio, video = audio.cpu(), video.cpu()
                    audio, video = ta.functional.resample(audio, SR, 22050).unsqueeze(0), video.permute(0, 2, 3, 1)
                    file = f"{od}/{correlation.__name__}_{str(uuid4())[:6]}.mp4"
                    tv.io.write_video(file, video.mul(255), fps, audio_array=audio, audio_fps=22050, audio_codec="aac")
                except KeyboardInterrupt:
                    exit(0)
                except:
                    progress.write(f"\nError: {correlation.__name__}")
                    progress.write(traceback.format_exc())
                    progress.write("\n")

                progress.update()
