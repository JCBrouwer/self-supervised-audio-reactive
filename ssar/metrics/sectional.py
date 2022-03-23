from glob import glob
from time import time

import matplotlib.pyplot as plt
import torch
from sklearn.utils import shuffle
from ssar.metrics.chroma import chromatic_reactivity
from ssar.metrics.rhythmic import rhythmic_reactivity
import decord as de

de.bridge.set_bridge("torch")

if __name__ == "__main__":
    for video_file in shuffle(glob("/home/hans/datasets/audiovisual/maua/*")):
        print("loading:", video_file)
        t = time()
        v = de.VideoReader(video_file)
        fps = v.get_avg_fps()
        h, w, c = v[0].shape
        sr = round(1024 * fps)
        av = de.AVReader(video_file, sample_rate=sr, width=w // 4, height=h // 4)
        audio, video = av[:]
        audio = torch.cat(audio, dim=1).squeeze().float().cuda()
        video = video.permute(0, 3, 1, 2).float().cuda()
        print("took:", time() - t)
        print("rhythmic reactivity:", rhythmic_reactivity(audio, sr, video, fps).item())
        print("chromatic reactivity:", chromatic_reactivity(audio, sr, video, fps).item())
        print()
