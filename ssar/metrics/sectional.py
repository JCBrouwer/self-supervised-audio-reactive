from glob import glob
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tvl
from sklearn.utils import shuffle
from ssar.metrics.chroma import chromatic_reactivity
from ssar.metrics.rhythmic import rhythmic_reactivity
from torchaudio.prototype.io import Streamer

if __name__ == "__main__":
    for video_file in shuffle(glob("/home/hans/datasets/audiovisual/maua/*")):
        print("loading:", video_file)
        t = time()
        streamer = Streamer(src=video_file)
        fps = 24
        sr = 1024 * fps
        info = streamer.get_src_stream_info(0)
        w, h = info.width, info.height
        streamer.add_basic_audio_stream(frames_per_chunk=round(sr), sample_rate=sr)
        streamer.add_basic_video_stream(frames_per_chunk=round(fps), width=w // 4, height=h // 4, frame_rate=fps)
        audio, video = [], []
        for audio_chunk, video_chunk in streamer.stream():
            audio.append(audio_chunk.mean(1).cuda(non_blocking=True))
            video.append(video_chunk.div(255.0).cuda(non_blocking=True))
        audio, video = torch.cat(audio), torch.cat(video)
        print("took:", time() - t)

        print("rhythmic reactivity:", rhythmic_reactivity(audio, sr, video, fps).item())
        print("chromatic reactivity:", chromatic_reactivity(audio, sr, video, fps).item())
        print()
