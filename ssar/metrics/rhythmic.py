import os
import sys

import joblib
import librosa as rosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio as ta
from dtaidistance import dtw
from torch.nn.functional import mse_loss
from torchvision.transforms.functional import resize
from tqdm import tqdm

dtwd = lambda x, y: dtw.distance_fast(
    x.squeeze().cpu().numpy().astype(np.double), y.squeeze().cpu().numpy().astype(np.double)
)

SSAR_DIR = os.path.abspath(os.path.dirname(__file__)) + "/../../"

sys.path.append(SSAR_DIR)
from ssar.analysis.chatterjee import xi
from ssar.analysis.visual_beats import video_onsets
from ssar.supervised.data import gaussian_filter

sys.path.append(SSAR_DIR + "/../maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2
from ops.video import write_video

COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]


def info(x, l=""):
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


def postprocess(x):
    x = gaussian_filter(x, 2 * F, causal=0)
    x = percentile_clip(x, (1 - Q) * 100)
    x = torch.clamp(x, torch.quantile(x, 4 * Q, dim=0).item(), 1)
    x = gaussian_filter(x, F)
    return normalize(x)


def my_audio_onsets(audio, sr):
    perc = rosa.effects.percussive(audio, margin=8.0)
    ons = rosa.onset.onset_strength(perc, sr, hop_length=1024)
    ons = torch.from_numpy(ons)
    ons = gaussian_filter(ons, 2 * F, causal=0)
    ons = percentile_clip(ons, (1 - Q) * 100)
    ons = torch.clamp(ons, torch.quantile(ons, 4 * Q, dim=0).item(), 1)
    ons[ons > 0.75] *= 2
    ons = gaussian_filter(ons, F)
    return normalize(ons)


def flow_onsets(video):
    flow = video_onsets(video.permute(0, 2, 3, 1))
    return postprocess(flow)


def absolute_difference(video):
    diff = torch.diff(video, dim=0, append=video[[-1]]).abs().sum(dim=(1, 2, 3))
    return postprocess(diff)


def cossim(x, y):
    return (x / torch.norm(x, 2)).dot(y / torch.norm(y, 2))


Q = 0.025
F = 1
SR = 24576 * F


def generate_rhythmic_onsets(rhythmic_file="/home/hans/datasets/audiovisual/wishlist/spiruen-drums.flac", debug=False):
    audio, sr = ta.load(rhythmic_file)
    audio, sr = ta.functional.resample(audio, sr, SR).mean(0).numpy(), SR

    if debug:
        plt.figure()
        rosa.display.specshow(
            librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=1024)), ref=np.max),
            sr=sr,
            hop_length=1024,
            x_axis="time",
            y_axis="log",
        )
        plt.savefig(SSAR_DIR + "/output/wishlist_rhythmic_spectrogram.pdf")
        plt.close()

    ons = my_audio_onsets(audio, sr).squeeze()
    ons = ons.cuda()

    if debug:
        plt.figure()
        plt.plot(ons.cpu())
        plt.savefig(SSAR_DIR + "/output/wishlist_rhythmic_audio_onsets.pdf")
        plt.close()

    G = StyleGAN2(SSAR_DIR + "/cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl")
    G = G.eval().cuda()

    @torch.inference_mode()
    def get_inputs():
        ws = G.mapper(torch.randn((2, 512), device="cuda"))
        latents = ws[[0]] * ons.reshape(-1, 1, 1) + ws[[1]] * (1 - ons.reshape(-1, 1, 1))
        latents = gaussian_filter(latents, F)

        noise = gaussian_filter(torch.randn((len(ons), 1, 64, 64), device="cuda"), 3 * F)
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

    metrics = {
        "flow onsets": flow_onsets,
        "abs diff": absolute_difference,
        "average": None,
    }

    n_trials = 30

    xis, dtws, envs = {name: [] for name in metrics}, {name: [] for name in metrics}, {name: [] for name in metrics}
    for i in tqdm(range(n_trials), desc="Computing rhythmic reactivity trials..."):
        latents, noise = get_inputs()
        video = get_video(latents, noise)
        write_video(video, SSAR_DIR + f"/output/wishlist_rhythmic_rand{i}.mp4", fps=SR / 1024, audio_file=rhythmic_file)

        for name, metric in list(metrics.items())[:-1]:
            env = metric(video).squeeze()
            envs[name].append(env)
            xis[name].append(xi(ons.cpu(), env))
            dtws[name].append(dtwd(ons, env))
        avgenvs = torch.mean(torch.stack([torch.stack(envs[name]) for name in list(metrics.keys())[:-1]]), dim=0)
        envs["average"] = [env for env in avgenvs]
        xis["average"] = [xi(ons.cpu(), env) for env in avgenvs]
        dtws["average"] = [dtwd(ons, env) for env in avgenvs]

    joblib.dump((envs, xis, dtws), SSAR_DIR + "/cache/rhythmic_envelopes.pkl", compress=9)

    fig, ax = plt.subplots(len(xis), 1, figsize=(16, 4 * len(xis)))
    for i, ((name, x), d) in enumerate(zip(xis.items(), dtws.values())):
        for env in envs[name]:
            ax[i].plot(env, color=COLORS[i], alpha=1.5 / len(envs[name]), linewidth=0.75)
        ax[i].plot(ons.cpu(), color="black", ls="--", alpha=1, linewidth=0.75)
        ax[i].set_title(f"{name}, xi: {np.mean(x):.2f} +/- {np.std(x):.2f}, dtw: {np.mean(d):.2f} +/- {np.std(d):.2f}")
    plt.tight_layout()
    plt.savefig(SSAR_DIR + "/output/wishlist_rhythmic_comparison.pdf")

    plt.figure(figsize=(16, 9))
    for i, name in enumerate(metrics):
        plt.plot(np.mean([e.numpy() for e in envs[name]], 0), color=COLORS[i], alpha=0.375, label=name)
    plt.plot(ons.numpy(), color="black", ls="--", alpha=0.25, label="audio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SSAR_DIR + "/output/wishlist_rhythmic_comparison_mean.pdf")


def compare_rhythmic_reactivity_metrics(rhythmic_file="/home/hans/datasets/audiovisual/wishlist/spiruen-drums.flac"):
    allenvs, _, _ = joblib.load(SSAR_DIR + "/cache/rhythmic_envelopes.pkl")
    allenvs = {k: torch.stack(v) for k, v in allenvs.items()}

    audio, sr = ta.load(rhythmic_file)
    audio, sr = ta.functional.resample(audio, sr, SR).mean(0).numpy(), SR
    ons = my_audio_onsets(audio, sr).squeeze()

    def permute_percentage_of_frames(envs, strength):
        res = envs.clone()
        num_envs, len_env = res.shape
        num_permute = round(len_env * strength)
        if num_permute > 0:
            for e in range(num_envs):
                to_idxs = np.random.permutation(len_env)[:num_permute]
                from_idxs = np.random.permutation(num_permute)
                res[e, to_idxs] = res[e, to_idxs[from_idxs]]
        return res

    def emphasize(envs, strength, cutoff=0.5):
        return envs * (1 + torch.tanh(strength * (envs - cutoff)))

    transforms = {
        "emphasize": (emphasize, np.linspace(0.01, 5, 100)),
        "power": (lambda envs, strength: envs ** strength, np.linspace(0.01, 5, 100)),
        "scale": (lambda envs, strength: strength * envs, np.linspace(-1, 5, 60)),
        "offset": (lambda envs, strength: torch.roll(envs, strength), np.arange(-200, 200)),
        "permute": (permute_percentage_of_frames, np.linspace(0, 1, 100)),
    }

    dists = {
        "xi": xi,
        "cos": cossim,
        "dtw": dtwd,
        "mse": mse_loss,
    }

    for transname, (transform, strengths) in transforms.items():
        fig, ax = plt.subplots(3, len(dists), figsize=(4 * len(dists), 9))
        for j, (distname, dist) in enumerate(dists.items()):
            for i, (name, envs) in enumerate(allenvs.items()):
                ds = np.array([[dist(ons, env) for env in transform(envs, s)] for s in strengths])
                mu, sig = ds.mean(1), ds.std(1)
                ax[i, j].plot(strengths, mu, color=COLORS[i], linewidth=0.75)
                ax[i, j].fill_between(strengths, mu - sig, mu + sig, color=COLORS[i], alpha=0.3)
                ax[i, 0].set_ylabel(name)
            ax[0, j].set_title(distname)
        plt.suptitle(transname)
        plt.tight_layout()
        plt.savefig(SSAR_DIR + f"/output/wishlist_rhythmic_reactivity_{transname}.pdf")


def rhythmic_reactivity(audio, sr, video, fps):
    if audio.dim() == 2:
        audio = audio.mean(0)

    audio, sr = ta.functional.resample(audio, sr, fps * 1024).numpy(), fps * 1024
    audio = rosa.effects.percussive(audio, margin=8.0)

    def postprocess(x, q=0.025):
        x = gaussian_filter(x, fps / 12, causal=0)
        x = percentile_clip(x, (1 - q) * 100)
        x = torch.clamp(x, torch.quantile(x, 4 * q, dim=0).item(), 1)
        x = gaussian_filter(x, fps / 24)
        return x

    audio_env = rosa.onset.onset_strength(audio, sr, hop_length=1024)
    audio_env = torch.from_numpy(audio_env)
    audio_env = postprocess(audio_env)

    video_env = torch.diff(video, dim=0, append=video[[-1]]).abs().sum(dim=(1, 2, 3))
    video_env = postprocess(video_env)

    similarity = audio_env @ video_env
    return similarity


if __name__ == "__main__":
    generate_rhythmic_onsets(debug=True)
    compare_rhythmic_reactivity_metrics()
