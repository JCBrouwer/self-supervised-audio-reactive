import os
import sys
from time import time

import librosa as rosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio as ta
from dtaidistance import dtw
from torchvision.transforms.functional import resize

SSAR_DIR = os.path.abspath(os.path.dirname(__file__)) + "/../../"

sys.path.append(SSAR_DIR)
from ssar.analysis.chatterjee import xi
from ssar.analysis.visual_beats import video_onsets
from ssar.supervised.data import gaussian_filter

sys.path.append(SSAR_DIR + "/../maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2
from ops.video import write_video


def info(x, l):
    print(
        l.ljust(25),
        f"{x.min().item():.4f}",
        f"{x.mean().item():.4f} +/- {x.std().item():.4f}",
        f"{x.max().item():.4f}",
        tuple(x.shape),
        x.dtype,
    )


def percentile_clip(signal, percent):
    result = []
    if len(signal.shape) < 2:
        signal = signal.unsqueeze(1)
    for sig in signal.unbind(1):
        locs = torch.arange(0, sig.shape[0])
        peaks = torch.ones(sig.shape, dtype=bool)
        main = sig.take(locs)

        plus = sig.take((locs + 1).clamp(0, sig.shape[0] - 1))
        minus = sig.take((locs - 1).clamp(0, sig.shape[0] - 1))
        peaks &= torch.gt(main, plus)
        peaks &= torch.gt(main, minus)

        sig = sig.clamp(0, torch.quantile(sig[peaks], percent / 100))
        sig /= sig.max()
        result.append(sig)
    return torch.stack(result, dim=1)


def normalize(x):
    y = x - x.min()
    y = y / y.max()
    return y


def onsets(audio, sr):
    perc = rosa.effects.percussive(audio, margin=8.0)
    ons = rosa.onset.onset_strength(perc, sr, hop_length=1024)
    ons = torch.from_numpy(ons)
    ons = gaussian_filter(ons, 4)
    ons = percentile_clip(ons, (1 - Q) * 100)
    ons = torch.clamp(ons, torch.quantile(ons, 4 * Q, dim=0).item(), 1)
    ons = gaussian_filter(ons, 2)
    return normalize(ons)


Q = 0.025
SR = 24576 * 2


def rhythmic_reactivity(rhythmic_file="/home/hans/datasets/audiovisual/wishlist/spiruen-drums.flac", debug=False):
    t = time()
    audio, sr = ta.load(rhythmic_file)
    audio, sr = ta.functional.resample(audio, sr, SR).mean(0).numpy(), SR

    if debug:
        print(f"load audio".ljust(25), f"{time()-t:3f} s")
        info(audio, "audio")
        plt.figure()
        rosa.display.specshow(
            librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=1024)), ref=np.max),
            sr=sr,
            hop_length=1024,
            x_axis="time",
            y_axis="log",
        )
        plt.show(block=False)

    t = time()
    ons = onsets(audio, sr)
    ons = ons.reshape(-1, 1, 1).cuda()

    if debug:
        info(ons, "onsets")
        print(f"onsets".ljust(25), f"{time()-t:3f} s")
        plt.figure()
        plt.plot(ons.cpu().squeeze())
        plt.show(block=False)

    G = StyleGAN2(SSAR_DIR + "/cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl")
    G = G.eval().cuda()

    @torch.inference_mode()
    def get_inputs():
        ws = G.mapper(torch.randn((2, 512), device="cuda"))
        latents = ws[[0]] * ons + ws[[1]] * (1 - ons)
        latents = gaussian_filter(latents, 1)

        noise = gaussian_filter(torch.randn((len(ons), 1, 64, 64), device="cuda"), 3)
        noise /= noise.std()
        noise *= ons.reshape(-1, 1, 1, 1)

        return latents, noise

    @torch.inference_mode()
    def get_video(latents, noise):
        noises = G.synthesizer.make_noise_pyramid(noise)
        noises["noise0"] = noises["noise0"][[0]].tile(len(noises["noise0"]), 1, 1, 1)
        inputs = {"latents": latents, **noises}
        resize_fn = lambda x: resize(x, size=128, antialias=True)
        video = torch.cat([frames.cpu() for frames in G.render(inputs=inputs, batch_size=32, postprocess_fn=resize_fn)])
        return video

    def vonsets(v):
        o = video_onsets(v.permute(0, 2, 3, 1))
        o = gaussian_filter(o, 2)
        o = percentile_clip(o, (1 - Q) * 100)
        o = torch.clamp(o, torch.quantile(o, 4 * Q, dim=0).item(), 1)
        o = gaussian_filter(o, 2)
        return normalize(o)

    def absdiff(v):
        d = torch.diff(v, dim=0, append=v[[-1]]).abs().sum(dim=(1, 2, 3))
        d = gaussian_filter(d, 2)
        d = percentile_clip(d, (1 - Q) * 100)
        d = torch.clamp(d, torch.quantile(d, 4 * Q, dim=0).item(), 1)
        d = gaussian_filter(d, 2)
        return normalize(d)

    metrics = {"flow onsets": vonsets, "abs diff": absdiff, "average": None}

    n_trials = 1

    xis, dtws, envs = {name: [] for name in metrics}, {name: [] for name in metrics}, {name: [] for name in metrics}
    for i in range(n_trials):
        latents, noise = get_inputs()
        video = get_video(latents, noise)
        write_video(video, SSAR_DIR + f"/output/wishlist_rhythmic_rand{i}.mp4", fps=SR / 1024, audio_file=rhythmic_file)

        for name, metric in list(metrics.items())[:-1]:
            t = time()
            env = metric(video)
            print(name, time() - t)
            envs[name].append(env)
            t = time()
            print(type(ons), type(env))
            xis[name].append(xi(ons.cpu().squeeze(), env))
            print("xi", time() - t)
            t = time()
            dtws[name].append(
                dtw.distance_fast(
                    ons.cpu().squeeze().numpy().astype(np.double), env.cpu().squeeze().numpy().astype(np.double)
                )
            )
            print("dtw", time() - t)
        env = np.mean([envs[name] for name in list(metrics.keys())[:-1]], axis=0)
        envs["average"].append(env)
        xis["average"].append(xi(ons.cpu().squeeze(), env))
        dtws["average"].append(dtw.distance(ons.cpu().squeeze(), env))

    fig, ax = plt.subplots(len(xis), 1, figsize=(16, 4 * len(xis)))
    colors = ["tab:blue", "tab:green", "tab:orange"]
    for i, ((name, x), d) in enumerate(zip(xis.items(), dtws.values())):
        for env in envs[name]:
            ax[i].plot(env, color=colors[i], alpha=1 / len(envs[name]), linewidth=0.75)
        ax[i].plot(ons.cpu().squeeze(), color="black", ls="--", alpha=1, linewidth=0.75)
        ax[i].set_title(f"{name}, xi: {np.mean(x):.2f} +/- {np.std(x):.2f}, dtw: {np.mean(d):.2f} +/- {np.std(d):.2f}")
    plt.tight_layout()
    plt.savefig(SSAR_DIR + "/output/wishlist_rhythmic_comparison.pdf")
    exit()

    latents, noise = get_inputs()
    xis = []
    for i in range(n_trials):
        latperm = np.random.permutation(latents.shape[-1])
        lats = latents[:, :, latperm]

        l, _, h, w = noise.shape
        noiseperm = np.random.permutation(h * w)
        nois = noise[..., noiseperm // h, noiseperm % h].reshape(l, 1, h, w)

        video = get_video(lats, nois)
        write_video(video, SSAR_DIR + f"/output/wishlist_rhythmic_perm{i}.mp4", fps=24, audio_file=rhythmic_file)
        vons = video_metric(video)
        xis.append(xi(ons, vons))
    print(
        "permutations".ljust(25),
        f"{np.min(xis):.3f}, {np.percentile(xis, 25):.3f}, {np.mean(xis):.3f} +/- {np.std(xis):.3f}, {np.percentile(xis, 75):.3f}, {np.max(xis):.3f}",
    )


if __name__ == "__main__":
    rhythmic_reactivity(debug=True)
