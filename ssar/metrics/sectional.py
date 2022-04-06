from glob import glob
from time import time

import decord as de
import torch
from sklearn.utils import shuffle
from ssar.metrics.chroma import chromatic_reactivity
from ssar.metrics.rhythmic import rhythmic_reactivity
from torch.nn.functional import interpolate
from torchaudio.functional import resample

de.bridge.set_bridge("torch")


def load_audio_video(path, downsample=4, resample_fps=24):
    v = de.VideoReader(path)
    fps = v.get_avg_fps()
    l, (h, w, c) = len(v), v[0].shape
    dur = l / fps
    del v
    sr = round(1024 * fps)
    av = de.AVReader(path, sample_rate=sr, width=w // downsample, height=h // downsample)
    audio, video = av[:]
    del av
    audio = torch.cat(audio, dim=1).squeeze().contiguous()
    video = video.permute(0, 3, 1, 2).div(255).contiguous()
    if resample_fps and fps != resample_fps:
        audio = resample(audio, sr, 1024 * resample_fps).contiguous()
        video = video.permute(1, 2, 3, 0).reshape(c * h, w, l)
        video = interpolate(video, size=round(dur * resample_fps))
        video = video.reshape(c, h, w, -1).permute(3, 0, 1, 2).contiguous()
    return audio, sr, video, fps


if __name__ == "__main__":
    with torch.inference_mode():
        for video_file in shuffle(glob("/home/hans/datasets/audiovisual/maua/*")):
            print("loading:", video_file)
            t = time()
            audio, sr, video, fps = load_audio_video(video_file)
            audio, video = audio.cuda(), video.cuda()
            print("took:", time() - t)
            print("rhythmic reactivity:", rhythmic_reactivity(audio, sr, video, fps).item())
            print("chromatic reactivity:", chromatic_reactivity(audio, sr, video, fps).item())
            print()
