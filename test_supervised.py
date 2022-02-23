#%%
import argparse
import importlib
import os
import sys
import warnings
from copy import copy, deepcopy
from glob import glob
from pathlib import Path

import matplotlib

# matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from scipy import stats

from train_supervised import Normalize, _audio2video, audio2features, audio2video, device


class FullLengthAudioFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        super().__init__()
        self.files = sum([glob(f"{directory}*.{ext}") for ext in ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]], [])
        from train_supervised import audio2features

        self.a2f = audio2features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        filename, _ = os.path.splitext(file)

        audio, sr = torchaudio.load(file)
        features = self.a2f(audio, sr)

        try:
            latents = joblib.load(f"{filename}.npy").float()
        except:
            try:
                latents = torch.from_numpy(np.load(f"{filename}.npy")).float()
            except:
                latents = torch.zeros(())

        return features, latents


def latent_residual_hist():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    latent_residuals = []
    i, n = 0, 32
    for _, targets in dataloader:
        latent_residuals.append(targets.numpy())
        i += 1
        if i > n:
            break
    latent_residuals = np.concatenate(latent_residuals).flatten()
    plt.figure(figsize=(16, 9))
    plt.hist(
        latent_residuals,
        bins=1000,
        label=f"Histogram (min={latent_residuals.min():.2f}, mean={latent_residuals.mean():.2f}, max={latent_residuals.max():.2f})",
        color="tab:blue",
        alpha=0.5,
        density=True,
    )
    loc, scale = stats.laplace.fit(latent_residuals, loc=0, scale=0.1)
    xs = np.linspace(-2, 2, 1000)
    plt.plot(
        xs,
        stats.laplace.pdf(xs, loc=loc, scale=scale),
        label=rf"PDF of MLE-fit Laplace($\mu$={loc:.2f}, $b$={scale:.2f})",
        color="tab:orange",
        alpha=0.75,
    )
    plt.title("Distribution of latent residuals")
    plt.legend()
    plt.xlim(-2, 2)
    plt.tight_layout()
    plt.savefig(f"plots/latent-residuals-distribution-true-{Path(dataset_cache).stem}.png")


def load_a2l(path):
    # HACK to import all modules that model needs from it's training code cache
    sys.path = [os.path.dirname(path)] + sys.path
    original_main_module = {}
    import train_supervised

    importlib.reload(train_supervised)
    for k in train_supervised.__dict__:
        if not k.startswith("_"):
            try:
                original_main_module[k] = deepcopy(getattr(sys.modules["__main__"], k))
            except:
                pass
            setattr(sys.modules["__main__"], k, getattr(train_supervised, k))

    a2l = joblib.load(path)["a2l"].eval().to(device)

    for k in original_main_module:
        setattr(sys.modules["__main__"], k, original_main_module[k])
    sys.path = sys.path[1:]

    if not hasattr(a2l, "normalize"):
        mean = np.load(f"cache/audio2latent_preprocessed_train_feats_mean.npy")
        std = np.load(f"cache/audio2latent_preprocessed_train_feats_std.npy")
        normalize = Normalize(mean, std).to(device)

        def norm_hook(mod, x):
            return normalize(x[0])

        a2l.register_forward_pre_hook(norm_hook)

    return a2l


#%%
test_audio = "/home/hans/datasets/wavefunk/tau ceti alpha.flac"
audio, sr = torchaudio.load(test_audio)
#%%
features = audio2features(audio, sr).unsqueeze(0).to(device)
#%%
features = features[:, int(22.5 * 24) : int((22.5 + 45) * 24)]
#%%
checkpoint = "runs/Feb22_11-02-45_ubuntu94025backbone:gru:skipTrue_layerwise:conv:6_hidden_size:128_num_layers:8_dropout:0.2_lr:0.0001_wd:0/audio2latent_backbone:gru:skipTrue_layerwise:conv:6_hidden_size:128_num_layers:8_dropout:0.2_lr:0.0001_wd:0_steps01358720_fcd57.8728_b0.0446_val0.0439.pt"
a2l = load_a2l(checkpoint)
#%%
norm_feats = a2l.normalize(features)
#%%
col_widths = [20, 12, 6, 7, 1, 1, 1, 1, 1, 1, 1]
feature_names = [
    "mfcc",
    "chroma",
    "tonnetz",
    "contrast",
    "onsets",
    "onsets_low",
    "onsets_mid",
    "onsets_high",
    "pulse",
    "volume",
    "flatness",
]
assert sum(col_widths) == 52
col_names = [
    *[f"mfcc_{i}" for i in range(20)],
    *[f"chroma_{i}" for i in range(12)],
    *[f"tonnetz_{i}" for i in range(6)],
    *[f"contrast_{i}" for i in range(7)],
    "onsets",
    "onsets_low",
    "onsets_mid",
    "onsets_high",
    "pulse",
    "volume",
    "flatness",
]
for col in range(52):
    print(
        col_names[col],
        f"{norm_feats[:, :, col].min().item():.4f}",
        f"{norm_feats[:, :, col].mean().item():.4f}",
        f"{norm_feats[:, :, col].max().item():.4f}",
    )
#%%
import pandas as pd

#%%
df = pd.DataFrame(norm_feats.squeeze().cpu().numpy(), columns=col_names)
#%%
for col in df.columns:
    plt.plot(df[col].values, label=col)
plt.show()
#%%
fig, ax = plt.subplots(52, 1, figsize=(8, 64))
for c, col in enumerate(df.columns):
    ax[c].plot(df[col])
    ax[c].set_ylabel(col)
plt.tight_layout()
plt.show()
#%%
_audio2video(
    a2l=a2l,
    test_features=features,
    audio_file=test_audio,
    out_file=f"runs/{Path(checkpoint).stem}_{Path(test_audio).stem}_unchanged.mp4",
    stylegan_file="/home/hans/modelzoo/train_checks/neurout2-117.pt",
    offset=22.5,
    duration=45,
    output_size=(1024, 1024),
    seed=123,
)
#%%
from scipy.stats import ttest_ind
from sgw import sgw_gpu

#%%
def perturb_zero(x):
    x[:, c_idx : c_idx + col_width] = 0
    return x


def perturb_invert(x):
    x[:, c_idx : c_idx + col_width] *= -1
    return x


def perturb_random(x):
    x[:, c_idx : c_idx + col_width] = torch.randn_like(x[:, c_idx : c_idx + col_width])
    return x


def perturb_scale(x):
    x[:, c_idx : c_idx + col_width] *= 2
    return x


#%%
with torch.inference_mode():
    c_idx = 0
    trials = 10
    y = a2l(features).squeeze().mean(1)
    x = norm_feats.squeeze().clone()
    xs = torch.stack(sum([list(torch.split(x[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)
    ys = torch.stack(sum([list(torch.split(y[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)

    ref_sgws = [sgw_gpu(xs, ys, device).item() for _ in range(trials)]

    print(
        f"feature".ljust(15),
        "perturbation".ljust(15),
        "grom/wass".ljust(10),
        "std. dev.".ljust(10),
        "t-stat".ljust(10),
        "p-value".ljust(10),
    )
    print(f"all".ljust(15), "none".ljust(15), f"{np.mean(ref_sgws):.4f}".ljust(10), f"{np.std(ref_sgws):.4f}")

    for perturb_name, perturb in [
        ("zeroed", perturb_zero),
        ("inverted", perturb_invert),
        ("random", perturb_random),
        ("2x'd", perturb_scale),
    ]:
        for feat_name, col_width in zip(feature_names, col_widths):
            x = norm_feats.squeeze().clone()
            x = perturb(x)

            xs = torch.stack(sum([list(torch.split(x[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)
            ys = torch.stack(sum([list(torch.split(y[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)

            sgws = [sgw_gpu(xs, ys, device).item() for _ in range(trials)]
            t, p = ttest_ind(ref_sgws, sgws)

            print(
                feat_name.ljust(15),
                perturb_name.ljust(15),
                f"{np.mean(sgws):.4f}".ljust(10),
                f"{np.std(sgws):.4f}".ljust(10),
                f"{t:.4f}".ljust(10),
                f"{p:.4f}",
                "" if p > 0.01 else "*",
            )

            c_idx += col_width

        c_idx, col_width = 0, 53
        x = norm_feats.squeeze().clone()
        x = perturb(x)

        xs = torch.stack(sum([list(torch.split(x[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)
        ys = torch.stack(sum([list(torch.split(y[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)

        sgws = [sgw_gpu(xs, ys, device).item() for _ in range(trials)]
        t, p = ttest_ind(ref_sgws, sgws)

        print(
            f"all".ljust(15),
            perturb_name.ljust(15),
            f"{np.mean(sgws):.4f}".ljust(10),
            f"{np.std(sgws):.4f}".ljust(10),
            f"{t:.4f}".ljust(10),
            f"{p:.4f}",
            "" if p > 0.01 else "*",
        )

#%%
from chatterjee import quadratic_xi

with torch.inference_mode():
    c_idx = 0
    yr = a2l(features).squeeze()
    print(
        f"feature".ljust(15),
        "perturbation".ljust(15),
        "l1/100000".ljust(15),
        "l2/100".ljust(15),
        # "mean correlation".ljust(17),
        # "std. correlation".ljust(17),
    )
    for perturb_name, perturb in [
        ("zeroed", perturb_zero),
        ("inverted", perturb_invert),
        ("random", perturb_random),
        ("2x'd", perturb_scale),
    ]:
        for feat_name, col_width in zip(feature_names, col_widths):
            handle = a2l.normalize.register_forward_hook(lambda m, i, o: perturb(o))
            y = a2l(features).squeeze()
            l1 = torch.norm(yr - y, p=1).item()
            l2 = torch.norm(yr - y, p=2).item()
            # cd = quadratic_xi(yr.mean(1), y.mean(1))
            print(
                feat_name.ljust(15),
                perturb_name.ljust(15),
                f"{l1/100_000:.4f}".ljust(15),
                f"{l2/100:.4f}".ljust(15),
                # f"{np.mean(cd):.4f}".ljust(17),
                # f"{np.std(cd):.4f}".ljust(17),
            )
            handle.remove()
            c_idx += col_width

        c_idx, col_width = 0, 53
        handle = a2l.normalize.register_forward_hook(lambda m, i, o: perturb(o))
        y = a2l(features).squeeze()
        l1 = torch.norm(yr - y, p=1).item()
        l2 = torch.norm(yr - y, p=2).item()
        # cd = quadratic_xi(yr.mean(1).cpu(), y.mean(1).cpu()).item()
        print(
            "all".ljust(15),
            perturb_name.ljust(15),
            f"{l1/100_000:.4f}".ljust(15),
            f"{l2/100:.4f}".ljust(15),
            # f"{np.mean(cd):.4f}".ljust(17),
            # f"{np.std(cd):.4f}".ljust(17),
        )
        handle.remove()
#%%"

if __name__ == "__main__":
    current_a2v = deepcopy(audio2video)

    # fmt:off
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="path to Audio2Latent checkpoint to test")
    parser.add_argument("--audio", help="path to audio file to test with", default="/home/hans/datasets/wavefunk/Ouroboromorphism_49_89.flac")
    parser.add_argument("--stylegan", help="path to StyleGAN model file to test with", default="/home/hans/modelzoo/train_checks/neurout2-117.pt")
    parser.add_argument("--output_size", help="output size for StyleGAN model rendering", nargs=2, default=[512, 512])
    parser.add_argument("--batch_size", help="batch size for StyleGAN model rendering", type=int, default=8)
    parser.add_argument("--offset", help="time in audio to start from in seconds", type=int, default=0)
    parser.add_argument("--duration", help="length in seconds of video to render", type=int, default=40)
    parser.add_argument("--plot", help="output distribution plot instead of only printing laplace scale", action='store_true')
    args = parser.parse_args()
    # fmt:on

    with torch.inference_mode():
        a2l = load_a2l(args.ckpt)

        if args.duration > 0:
            current_a2v(
                a2l=a2l,
                audio_file=args.audio,
                out_file=f"{os.path.splitext(args.ckpt)[0]}_{Path(args.audio).stem}.mp4",
                stylegan_file=args.stylegan,
                output_size=[int(s) for s in args.output_size],
                batch_size=args.batch_size,
                offset=args.offset,
                duration=args.duration,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            latent_residuals = a2l(audio2features(*torchaudio.load(args.audio)).to(device).unsqueeze(0))
        latent_residuals = latent_residuals.flatten().cpu().numpy()
        loc, scale = stats.laplace.fit(latent_residuals, loc=0, scale=0.1)
        print(f"laplace scale: {scale:.4f}")

        out_file = f"plots/latent-residuals-distribution-{Path(args.audio).stem}-{Path(os.path.dirname(args.ckpt)).stem}-{Path(args.ckpt).stem}.png"
        if not os.path.exists(out_file) and args.plot:
            plt.figure(figsize=(16, 9))
            label = f"Histogram (min={latent_residuals.min():.2f}, mean={latent_residuals.mean():.2f}, max={latent_residuals.max():.2f})"
            plt.hist(latent_residuals, bins=1000, label=label, color="tab:blue", alpha=0.5, density=True)
            xs = np.linspace(-2, 2, 1000)
            ys = stats.laplace.pdf(xs, loc=loc, scale=scale)
            label = rf"PDF of MLE-fit Laplace($\mu$={loc:.2f}, $b$={scale:.2f})"
            plt.plot(xs, ys, label=label, color="tab:orange", alpha=0.75)
            plt.title("Distribution of latent residuals")
            plt.legend()
            plt.xlim(-2, 2)
            plt.tight_layout()
            plt.savefig(out_file)
