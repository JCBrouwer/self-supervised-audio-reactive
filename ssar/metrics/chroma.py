import enum
import os
import sys

import librosa as rosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio as ta
import torchsort
from torchvision.transforms.functional import resize
from tqdm import tqdm

SSAR_DIR = os.path.abspath(os.path.dirname(__file__)) + "/../../"

sys.path.append(SSAR_DIR)
from ssar.analysis.chatterjee import rank
from ssar.metrics.rhythmic import my_audio_onsets, normalize, percentile_clip
from ssar.supervised.data import gaussian_filter

sys.path.append(SSAR_DIR + "/../maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2
from ops.video import write_video

COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red"]


def my_chromagram(audio, sr):
    harm = rosa.effects.harmonic(audio, margin=8.0)
    chroma = rosa.feature.chroma_cens(harm, sr, hop_length=1024)
    chroma = np.minimum(chroma, rosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine"))
    chroma = torch.from_numpy(chroma).T.float()
    chroma = gaussian_filter(chroma, 2 * sr / 1024 / 24, causal=0)
    chroma = percentile_clip(chroma, 97.5)
    chroma = torch.clamp(chroma, torch.quantile(chroma, 0.1), 1)
    chroma = gaussian_filter(chroma, sr / 1024 / 24)
    return chroma


def chroma_metric_analysis(chroma_file="/home/hans/datasets/audiovisual/wishlist/spiruen-melodic1.flac", fps=24):
    audio, sr = ta.load(chroma_file)
    audio, sr = ta.functional.resample(audio, sr, 1024 * fps).mean(0).numpy(), 1024 * fps

    plt.figure()
    rosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=1024)), ref=np.max),
        sr=sr,
        hop_length=1024,
        x_axis="time",
        y_axis="log",
    )
    plt.savefig(SSAR_DIR + "/output/wishlist_chroma_spectrogram.pdf")
    plt.close()

    chroma = my_chromagram(audio, sr)

    plt.figure()
    rosa.display.specshow(chroma.numpy().T, sr=sr, hop_length=1024, x_axis="time", y_axis="chroma")
    plt.savefig(SSAR_DIR + "/output/wishlist_chroma_chromagram.pdf")
    plt.close()

    chroma /= chroma.sum(1, keepdim=True)
    chroma = chroma.cuda()

    G = StyleGAN2(SSAR_DIR + "/cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl").eval().cuda()

    @torch.inference_mode()
    def get_inputs():
        ws = G.mapper(torch.randn((12, 512), device="cuda"))
        latents = torch.einsum("TC,CNL->TNL", chroma, ws)
        latents = gaussian_filter(latents, fps / 24)

        noise = gaussian_filter(torch.randn((len(chroma), 1, 32, 32), device="cuda"), 3 * fps)
        noise /= noise.std()

        return latents, noise

    @torch.inference_mode()
    def get_video(latents, noise):
        inputs = {"latents": latents, **G.synthesizer.make_noise_pyramid(noise)}
        resize_fn = lambda x: resize(x, size=128, antialias=True)
        video = torch.cat([frames.cpu() for frames in G.render(inputs=inputs, batch_size=32, postprocess_fn=resize_fn)])
        return video

    chroma_normed = chroma / chroma.norm(p=2, dim=1, keepdim=True)
    chroma_ac = normalize(chroma_normed @ chroma_normed.T).cpu()

    drums = ta.functional.resample(*ta.load(chroma_file.replace("melody1", "drums")), 1024 * fps).mean(0).numpy()
    onsets = my_audio_onsets(drums, sr)[:, None]
    onsets_normed = onsets / onsets.norm(p=2)
    onsets_ac = normalize(onsets_normed @ onsets_normed.T)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(chroma_ac)
    ax[0].axis("off")
    ax[0].set_title("Chroma")
    ax[1].imshow(onsets_ac)
    ax[1].axis("off")
    ax[1].set_title("Onsets")
    ax[2].imshow((chroma_ac - onsets_ac).abs())
    ax[2].axis("off")
    ax[2].set_title("Difference")
    plt.tight_layout()
    plt.savefig(SSAR_DIR + "/output/wishlist_chroma_onset_autocorrelations.pdf")
    plt.close()

    num_trials = 30

    cache_file = "cache/chroma_autocorrelations.npz"
    if not os.path.exists(cache_file):
        vhist_acs, diff_acs = [], []
        for i in tqdm(range(num_trials), desc="Gathering chroma-based video autocorrelations..."):
            video = get_video(*get_inputs())
            write_video(video, SSAR_DIR + f"/output/wishlist_chroma_rand{i}.mp4", fps=fps, audio_file=chroma_file)
            video_hists = torch.stack(
                [torch.cat([torch.histc(channel, bins=32) for channel in frame]) for frame in video]
            )
            video_hists = video_hists / video_hists.norm(p=2, dim=1, keepdim=True)
            vhist_ac = normalize(video_hists @ video_hists.T).cpu()
            vhist_acs.append(vhist_ac)
            diff_acs.append(torch.abs(chroma_ac - vhist_ac))
        diff_acs = np.stack(diff_acs)
        vhist_acs = np.stack(vhist_acs)

        np.savez(cache_file, vhist_acs=vhist_acs, diff_acs=diff_acs)
    else:
        with np.load(cache_file) as data:
            vhist_acs = data["vhist_acs"]
            diff_acs = data["diff_acs"]

    acsims = np.array([(chroma_ac * vhist_ac).mean() for vhist_ac in vhist_acs])
    print(acsims)
    # plt.hist(acsims, bins=10)
    # plt.show()

    mean_diff = diff_acs.mean(0)
    std_diff = diff_acs.std(0)
    weighted_diff = normalize(mean_diff) / (normalize(std_diff) + 1e-8)
    weighted_diff = weighted_diff.clip(np.quantile(weighted_diff, 0.001), np.quantile(weighted_diff, 0.999))

    diagonal_diffs = [weighted_diff.diagonal(i) for i in range(1, weighted_diff.shape[1] - 1)]
    diagonal_mean = np.array([np.mean(dd) for dd in diagonal_diffs])
    diagonal_std = np.array([np.std(dd) for dd in diagonal_diffs])

    plt.figure(figsize=(16, 9))
    plt.plot(diagonal_mean)
    plt.fill_between(
        np.arange(len(diagonal_mean)), diagonal_mean - diagonal_std, diagonal_mean + diagonal_std, alpha=0.5
    )
    plt.tight_layout()
    plt.savefig(SSAR_DIR + "/output/wishlist_chroma_autocorrelation_diagonal_diffs.pdf")
    plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    ax[0, 0].imshow(chroma_ac)
    ax[0, 0].axis("off")
    ax[0, 0].set_title("chroma autocorrelation")
    ax[0, 1].imshow(vhist_acs[-1])
    ax[0, 1].axis("off")
    ax[0, 1].set_title("example video histogram autocorrelation")
    ax[0, 2].imshow(diff_acs[-1])
    ax[0, 2].axis("off")
    ax[0, 2].set_title("example difference")
    ax[1, 0].imshow(diff_acs.mean(0))
    ax[1, 0].axis("off")
    ax[1, 0].set_title("mean difference")
    ax[1, 1].imshow(diff_acs.std(0))
    ax[1, 1].axis("off")
    ax[1, 1].set_title("std difference")
    ax[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(SSAR_DIR + "/output/wishlist_chroma_autocorrelations.pdf")
    plt.close()


def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()


def correlation(target, pred, regularization="l2", regularization_strength=0.01, spearman=False):
    if spearman:
        pred = (
            torchsort.soft_rank(
                pred.unsqueeze(0), regularization=regularization, regularization_strength=regularization_strength
            ).squeeze()
            / pred.shape[-1]
        )
    return corrcoef(target, pred)


def chromatic_reactivity(audio, sr, video, fps):
    dev = audio.device
    if round(sr) != round(fps * 1024):
        audio, sr = ta.functional.resample(audio, sr, round(fps * 1024)), round(fps * 1024)
    harm = rosa.effects.harmonic(audio.cpu().numpy(), margin=8.0)
    chroma = rosa.feature.chroma_cens(harm, sr, hop_length=1024)
    chroma = rosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine")
    chroma = torch.from_numpy(chroma).T.float().to(dev)
    chroma = gaussian_filter(chroma, fps / 12, causal=0)
    chroma = percentile_clip(chroma, 97.5)
    chroma = torch.clamp(chroma, torch.quantile(chroma, 0.1), 1)
    chroma = gaussian_filter(chroma, fps / 24)
    chroma = chroma / chroma.norm(p=2, dim=1, keepdim=True)

    vhist = torch.stack([torch.cat([torch.histc(channel, bins=32) for channel in frame]) for frame in video])
    vhist = vhist / vhist.norm(p=2, dim=1, keepdim=True)

    chroma = chroma[: min(len(chroma), len(vhist))]
    vhist = vhist[: min(len(chroma), len(vhist))]

    chroma_ac = chroma @ chroma.T
    vhist_ac = vhist @ vhist.T

    triuy, triux = torch.triu_indices(len(chroma_ac), len(chroma_ac), 1).unbind(0)
    similarity = correlation(chroma_ac[triuy, triux], vhist_ac[triuy, triux])
    return similarity


def compare_chroma_reactivity_metrics(
    chroma_file="/home/hans/datasets/audiovisual/wishlist/spiruen-melodic1.flac", fps=24
):
    audio, sr = ta.load(chroma_file)
    audio, sr = ta.functional.resample(audio, sr, 1024 * fps).mean(0).numpy(), 1024 * fps
    real_chroma = my_chromagram(audio, sr)
    real_chroma /= real_chroma.sum(1, keepdim=True)
    real_chroma = real_chroma.cuda()

    G = StyleGAN2(SSAR_DIR + "/cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl").eval().cuda()

    @torch.inference_mode()
    def get_inputs(chroma):
        ws = G.mapper(torch.randn((12, 512), device="cuda"))
        latents = torch.einsum("TC,CNL->TNL", chroma, ws)
        latents = gaussian_filter(latents, fps / 24)

        noise = gaussian_filter(torch.randn((len(chroma), 1, 32, 32), device="cuda"), 3 * fps)
        noise /= noise.std()

        return latents, noise

    @torch.inference_mode()
    def get_video(latents, noise):
        inputs = {"latents": latents, **G.synthesizer.make_noise_pyramid(noise)}
        resize_fn = lambda x: resize(x, size=128, antialias=True)
        video = torch.cat([frames.cpu() for frames in G.render(inputs=inputs, batch_size=32, postprocess_fn=resize_fn)])
        return video

    n_trials = 30

    def get_chroma_vhist():
        video = get_video(*get_inputs(real_chroma))
        vhist = torch.stack([torch.cat([torch.histc(channel, bins=32) for channel in frame]) for frame in video])
        vhist = vhist / vhist.norm(p=2, dim=1, keepdim=True)
        return vhist

    cache_file = "cache/chroma_vhists.npz"
    if not os.path.exists(cache_file):
        vhists = torch.stack([get_chroma_vhist() for _ in tqdm(range(n_trials))])
        np.savez(cache_file, vhists=vhists.cpu().numpy())
    else:
        with np.load(cache_file) as data:
            vhists = torch.from_numpy(data["vhists"])
    vhists = vhists.cuda()

    def permute_percentage_of_frames(envs, strength):
        res = envs.clone()
        num_envs, len_env, _ = res.shape
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
        "emphasize": (emphasize, np.linspace(0.01, 5, 60)),
        "power": (lambda envs, strength: envs ** strength, np.linspace(0.01, 5, 60)),
        "scale": (lambda envs, strength: strength * envs, np.linspace(-1, 5, 60)),
        "offset": (lambda envs, strength: torch.roll(envs, strength), np.arange(-100, 100)),
        "permute": (permute_percentage_of_frames, np.linspace(0, 1, 60)),
    }

    with tqdm(total=2 * sum([len(s) for _, (_, s) in transforms.items()]) * len(vhists)) as pbar:

        def corr(chroma, vhist, spearman):
            chroma = chroma[: min(len(chroma), len(vhist))]
            vhist = vhist[: min(len(chroma), len(vhist))]

            chroma_ac = chroma @ chroma.T
            vhist_ac = vhist @ vhist.T

            triuy, triux = torch.triu_indices(len(chroma_ac), len(chroma_ac), 1).unbind(0)
            similarity = correlation(chroma_ac[triuy, triux], vhist_ac[triuy, triux], spearman=spearman)

            pbar.update()

            return similarity.cpu().numpy()

        fig, ax = plt.subplots(len(transforms), 2, figsize=(8, 3 * len(transforms)))
        for j, spearman in enumerate([False, True]):
            for i, (transname, (transform, strengths)) in enumerate(transforms.items()):
                ds = np.array(
                    [
                        [corr(real_chroma, transformed_vhist, spearman) for transformed_vhist in transform(vhists, s)]
                        for s in strengths
                    ]
                )
                mu, sig = ds.mean(1), ds.std(1)
                ax[i, j].plot(strengths, mu, color=COLORS[i], linewidth=0.75)
                ax[i, j].fill_between(strengths, mu - sig, mu + sig, color=COLORS[i], alpha=0.3)
                ax[i, j].set_xlabel(transname)
        ax[0, 0].set_title("correlation")
        ax[0, 1].set_title("spearman correlation")
        plt.suptitle("chromatic reactivity sensitivity")
        plt.tight_layout()
        plt.savefig(SSAR_DIR + f"/output/wishlist_chroma_reactivity_sensitivity.pdf")


if __name__ == "__main__":
    # chroma_metric_analysis()
    compare_chroma_reactivity_metrics()
