import random
import sys
import traceback
from glob import glob
from time import time
from uuid import uuid4

import decord as de
import numpy as np
import torch
import torchaudio as ta
import torchvision as tv
from einops import rearrange
from ssar.metrics.chroma import my_chromagram
from ssar.metrics.rhythmic import my_audio_onsets
from ssar.supervised.data import audio2features, gaussian_filter
from ssar.supervised.latent_augmenter import LatentAugmenter, spline_loop_latents
from torch.nn.functional import interpolate
from torchvision.transforms.functional import resize
from tqdm import tqdm

sys.path.append("../maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2

de.bridge.set_bridge("torch")


ckpt = "cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl"
G = StyleGAN2(ckpt)
G = G.eval().cuda()
LA = LatentAugmenter(ckpt, n_patches=5)
FPS = 24
DUR = 12
N = DUR * FPS
SR = 1024 * FPS


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


def get_video(latents, noise):
    noises = G.synthesizer.make_noise_pyramid(noise)
    noises["noise0"] = noises["noise0"][[0]].tile(len(noises["noise0"]), 1, 1, 1)
    inputs = {"latents": latents, **noises}
    resize_fn = lambda x: resize(x, size=256, antialias=True)
    video = torch.cat([frames.cpu() for frames in G.render(inputs=inputs, batch_size=32, postprocess_fn=resize_fn)])
    return video


def low_correlation():
    audio, sr = get_random_audio()
    zs = torch.randn((np.random.randint(DUR // 8, DUR), 512), device="cuda")
    ws = G.mapper(zs)
    latents = spline_loop_latents(ws, N)
    noise = torch.randn((np.random.randint(DUR // 8, DUR), 1, 16 * 16))
    noise = spline_loop_latents(noise, N).reshape(N, 1, 16, 16)
    return audio, sr, get_video(latents, noise), FPS


def medium_correlation():
    audio, sr = get_random_audio()
    audio_features = audio2features(audio, sr, fps=FPS).unsqueeze(0).cuda()
    residual, offset = LA(audio_features)
    latents = (offset + residual).squeeze()
    noise = torch.randn((np.random.randint(DUR // 8, DUR), 1, 16 * 16))
    noise = spline_loop_latents(noise, N).reshape(N, 1, 16, 16)
    return audio, sr, get_video(latents, noise), FPS


def high_chroma_correlation():
    audio, sr = get_random_audio()
    chroma = my_chromagram(audio.cpu().numpy(), sr).cuda()
    chroma /= chroma.sum(1, keepdim=True)
    ws = G.mapper(torch.randn((12, 512), device="cuda"))
    latents = torch.einsum("TC,CNL->TNL", chroma, ws)
    latents = gaussian_filter(latents, FPS / 24)
    noise = gaussian_filter(torch.randn((len(chroma), 1, 32, 32), device="cuda"), 3 * FPS)
    noise /= noise.std()
    return audio, sr, get_video(latents, noise), FPS


def high_onset_correlation():
    audio, sr = get_random_audio()
    ons = my_audio_onsets(audio.cpu().numpy(), sr).squeeze().cuda()
    ws = G.mapper(torch.randn((2, 512), device="cuda"))
    latents = ws[[0]] * ons.reshape(-1, 1, 1) + ws[[1]] * (1 - ons.reshape(-1, 1, 1))
    latents = gaussian_filter(latents, FPS / 24)
    noise = gaussian_filter(torch.randn((len(ons), 1, 64, 64), device="cuda"), 3 * FPS / 24)
    noise /= noise.std()
    noise *= ons.reshape(-1, 1, 1, 1)
    return audio, sr, get_video(latents, noise), FPS


def high_both_correlation():
    audio, sr = get_random_audio()

    chroma = my_chromagram(audio.cpu().numpy(), sr).cuda()
    chroma /= chroma.sum(1, keepdim=True)

    ons = my_audio_onsets(audio.cpu().numpy(), sr).squeeze().cuda()

    ws = G.mapper(torch.randn((14, 512), device="cuda"))

    latents = (
        torch.einsum("TC,CNL->TNL", chroma[:12], ws)
        + ws[[13]] * ons.reshape(-1, 1, 1)
        + ws[[14]] * (1 - ons.reshape(-1, 1, 1))
    ) / 2

    latents = gaussian_filter(latents, FPS / 24)

    noise = gaussian_filter(torch.randn((len(ons), 1, 64, 64), device="cuda"), 3 * FPS / 24)
    noise /= noise.std()
    noise *= ons.reshape(-1, 1, 1, 1)

    return audio, sr, get_video(latents, noise), FPS


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


if __name__ == "__main__":
    od = "output/maua_correlation_test2"
    with torch.inference_mode(), tqdm(total=6 * 120) as progress:
        for i in range(120):
            for correlation in [
                low_correlation,
                medium_correlation,
                test_set_correlation,
                high_chroma_correlation,
                high_onset_correlation,
                high_both_correlation,
            ]:
                try:
                    t = time()
                    audio, sr, video, fps = correlation()
                    audio, video = audio.cpu(), video.cpu()
                    audio, video = ta.functional.resample(audio, SR, 22050).unsqueeze(0), video.permute(0, 2, 3, 1)
                    file = f"{od}/{correlation.__name__}_{str(uuid4())[:6]}.mp4"
                    tv.io.write_video(file, video.mul(255), fps, audio_array=audio, audio_fps=22050, audio_codec="aac")
                except:
                    progress.write(f"\nError: {correlation.__name__}")
                    progress.write(traceback.format_exc())
                    progress.write("\n")
                progress.update()
