import matplotlib

matplotlib.use("Agg")

import random
import sys

import joblib
import librosa as rosa
import librosa.display as rosa_display
import matplotlib.pyplot as plt
import numpy as np
import torch
from npy_append_array import NpyAppendArray as NpArr
from scipy.io.wavfile import write as write_wav
from tqdm import trange

from .random_patches import RandomPatch

sys.path.append("/home/hans/code/maua-stylegan2/")
from audioreactive import load_latents
from audioreactive.frequent.timestamps import timestamps

sys.path.append("/home/hans/code/maua/maua")
from audiovisual.audioreactive.audio import load_audio
from audiovisual.patches.primitives import ModulationSum


def get_latent_selection(latent_file, ckpt, ckpt2=None, shuffle_latents=True, n=36):
    if latent_file is not None:
        latent_selection = load_latents(latent_file)
    else:
        raise Exception("NO LATENT FILE?!")
    if shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
    return latent_selection.float()


if __name__ == "__main__":
    audio_file = "/home/hans/datasets/frequent/redrocks/tipper red rocks set sketch v8.flac"
    FPS = 24

    with torch.inference_mode():

        for i, song_info in enumerate(timestamps):
            title, file, ckpt, lat_file = song_info["title"], song_info["file"], song_info["ckpt"], song_info["latents"]
            start_time, end_time, bpm = song_info["start"], song_info["end"], song_info["bpm"]
            output_file = f"/home/hans/neurout/ssar/frequent/{title}.mp4"

            offset = start_time
            duration = end_time - start_time
            fps = FPS

            audio, sr, duration = load_audio(audio_file, offset, duration)
            n_frames = int(round(duration * FPS))

            print("\n\n")
            print(title)
            latents, noises, patches, feats = [], [], [], []
            for i in range(5):
                print()
                patch = RandomPatch.randomize()
                patches.append(patch)

                patch_module, f, p = patch(
                    audio=audio,
                    sr=sr,
                    dur=duration,
                    fps=fps,
                    audio_file=audio_file,
                    latent_selection=get_latent_selection(lat_file, ckpt),
                    size=64,
                )
                feats.append([f, p])

                if not hasattr(patch_module, "modulation"):
                    patch_module.modulation = torch.ones((1))
                if "Latent" in patch_module.__class__.__name__:
                    latents.append(patch_module)
                else:
                    noises.append(patch_module)

            latent = ModulationSum(latents)
            noise = ModulationSum(noises)
            # synthesizer = StyleGAN2Synthesizer(model_file, False, (512, 512), "stretch", 0).cuda()
            # with VideoWriter(output_file, synthesizer.output_size, fps, audio_file, offset, dur, "slow")
            with NpArr(output_file.replace(".mp4", "_noise.npy")) as noise_file, NpArr(
                output_file.replace(".mp4", "_latent.npy")
            ) as latent_file:
                for _ in trange(round(duration * fps)):
                    lat = latent()
                    noi = noise()
                    # frame = synthesizer(latent_w_plus=lat.cuda(), **synthesizer.make_noise_pyramid(noi.unsqueeze(0).cuda())).add(1).div(2)
                    # video.write(frame)
                    latent_file.append(np.ascontiguousarray(lat.numpy()))
                    if noise is not None:
                        noise_file.append(np.ascontiguousarray(noi.unsqueeze(0).numpy()))

            joblib.dump(patches, output_file.replace(".mp4", f"_patches.pkl"), compress=9)
            with open(output_file.replace(".mp4", f"_patches.txt"), "w") as f:
                for patch in patches:
                    f.write(f"{patch}")

            fig, ax = plt.subplots(2, len(feats), figsize=(8 * len(feats), 16))
            for i in range(len(feats)):
                for j in range(len(feats[i])):
                    af = feats[i][j]
                    if isinstance(af, float):
                        rosa_display.specshow(
                            rosa.feature.tempogram(audio.numpy(), sr), sr=sr, x_axis="time", y_axis="tempo", ax=ax[j, i]
                        )
                        ax[j, i].axhline(
                            af, color="w", linestyle="--", alpha=1, label="Estimated tempo={:g}".format(af)
                        )
                        ax[j, i].legend(loc="upper right")
                    elif len(af.squeeze().shape) == 1:
                        ax[j, i].plot(af.squeeze().numpy())
                    elif af.shape[1] == 6:
                        rosa_display.specshow(
                            np.rollaxis(af.numpy(), 1, 0), sr=sr, x_axis="time", y_axis="tonnetz", ax=ax[j, i]
                        )
                    elif af.shape[1] == 12:
                        rosa_display.specshow(
                            np.rollaxis(af.numpy(), 1, 0), sr=sr, x_axis="time", y_axis="chroma", ax=ax[j, i]
                        )
                    else:
                        raise Exception(f"what? {af.shape} {patches[i].feature.__class__.__name__}")
                    ax[j, i].set(
                        title=patches[i].source.__class__.__name__
                        + " "
                        + patches[i].filter.__class__.__name__
                        + " "
                        + patches[i].target.__class__.__name__
                        + (" " + patches[i].postprocess.__class__.__name__ if j else "")
                    )
            plt.tight_layout()
            plt.savefig(output_file.replace(".mp4", ".pdf"))
