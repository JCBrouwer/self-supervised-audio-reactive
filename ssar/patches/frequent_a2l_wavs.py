import sys
from glob import glob
from pathlib import Path

import joblib
import torch
from scipy.io.wavfile import write as write_wav

sys.path.append("/home/hans/code/maua-stylegan2/")
from audioreactive.frequent.timestamps import timestamps

sys.path.append("/home/hans/code/maua/")
from maua.audiovisual.audioreactive.audio import load_audio

if __name__ == "__main__":

    for noise_file in glob("/home/hans/datasets/frequent/final/*_noise.npy"):
        noise = joblib.load(noise_file)
        print(noise_file)
        title = Path(noise_file).stem.replace("_noise", "")
        for width, idx in [(4, 0), (8, 2), (16, 4), (32, 6)]:
            out_file = f"/home/hans/datasets/audio2latent/Frequent @ Red Rocks - {title} - Noise {width}.npy"
            joblib.dump(noise[idx], out_file)

    exit(0)

    audio_file = "/home/hans/datasets/frequent/redrocks/tipper red rocks set sketch v8.flac"
    FPS = 24

    for i, song_info in enumerate(timestamps):
        title, start_time, end_time = song_info["title"], song_info["start"], song_info["end"]

        offset = start_time
        duration = end_time - start_time

        audio, sr, duration = load_audio(audio_file, offset, duration)

        write_wav(f"/home/hans/datasets/audio2latent/Frequent @ Red Rocks - {title}.wav", sr, audio.numpy())
        write_wav(f"/home/hans/datasets/audio2latent/Frequent @ Red Rocks - {title}.wav", sr, audio.numpy())
