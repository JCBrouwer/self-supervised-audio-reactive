import sys

import torch
from scipy.io.wavfile import write as write_wav

sys.path.append("/home/hans/code/maua-stylegan2/")
from audioreactive.frequent.timestamps import timestamps

sys.path.append("/home/hans/code/maua/maua")
from audiovisual.audioreactive.audio import load_audio

if __name__ == "__main__":
    audio_file = "/home/hans/datasets/frequent/redrocks/tipper red rocks set sketch v8.flac"
    FPS = 24

    for i, song_info in enumerate(timestamps):
        title, start_time, end_time = song_info["title"], song_info["start"], song_info["end"]

        offset = start_time
        duration = end_time - start_time

        audio, sr, duration = load_audio(audio_file, offset, duration)

        write_wav(f"/home/hans/datasets/audio2latent/Frequent @ Red Rocks - {title}.wav", sr, audio.numpy())
