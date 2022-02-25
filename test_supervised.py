# fmt:off
import argparse
import importlib
import os
import sys
import warnings
from copy import copy, deepcopy
from glob import glob
from pathlib import Path

import joblib
import librosa as rosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from scipy import stats
from scipy.stats import ttest_ind
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import tqdm

from sgw import sgw_gpu
from train_supervised import (Normalize, StyleGAN2Mapper, StyleGAN2Synthesizer,
                              VideoWriter, _audio2video, audio2features,
                              audio2video, device)
# fmt:on

FPS = 24
DUR = 8


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
        mean = np.load(f"cache/audio2latent_{DUR*FPS}frames_train_mean.npy")
        std = np.load(f"cache/audio2latent_{DUR*FPS}frames_train_std.npy")
        normalize = Normalize(mean, std).to(device)

        def norm_hook(mod, x):
            return normalize(x[0])

        a2l.register_forward_pre_hook(norm_hook)

    return a2l


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
    "all",
]


def test_output_sensitivity(checkpoint, plot=False, whole_feature=True):
    a2l = load_a2l(checkpoint)

    test_audio = "/home/hans/datasets/wavefunk/tau ceti alpha.flac"
    audio, sr = torchaudio.load(test_audio)
    features = audio2features(audio, sr, FPS).unsqueeze(0).to(device)

    norm_feats = a2l.normalize(features)

    for col in range(52):
        print(
            col_names[col],
            f"{norm_feats[:, :, col].min().item():.4f}",
            f"{norm_feats[:, :, col].mean().item():.4f}",
            f"{norm_feats[:, :, col].max().item():.4f}",
        )
    print()

    if plot:
        import pandas as pd

        df = pd.DataFrame(norm_feats.squeeze().cpu().numpy(), columns=col_names)

        for col in df.columns:
            plt.plot(df[col].values, label=col)
        plt.savefig("output/a2l_norm_feats_all.pdf")
        plt.close()

        fig, ax = plt.subplots(52, 1, figsize=(8, 64))
        for c, col in enumerate(df.columns):
            ax[c].plot(df[col])
            ax[c].set_ylabel(col)
        plt.tight_layout()
        plt.savefig("output/a2l_norm_feat_by_feat.pdf")
        plt.close()

    def perturb_zero(x):
        print(x.min().item(), x.mean().item(), x.max().item())
        x[..., c_idx : c_idx + col_width] = 0
        print(x.min().item(), x.mean().item(), x.max().item())
        return x

    def perturb_invert(x):
        print(x.min().item(), x.mean().item(), x.max().item())
        x[..., c_idx : c_idx + col_width] *= -1
        print(x.min().item(), x.mean().item(), x.max().item())
        return x

    def perturb_random(x):
        print(x.min().item(), x.mean().item(), x.max().item())
        x[..., c_idx : c_idx + col_width] = torch.randn_like(x[..., c_idx : c_idx + col_width])
        print(x.min().item(), x.mean().item(), x.max().item())
        return x

    def perturb_scale(x):
        print(x.min().item(), x.mean().item(), x.max().item())
        x[..., c_idx : c_idx + col_width] *= 2
        print(x.min().item(), x.mean().item(), x.max().item())
        return x

    if whole_feature:
        col_widths = [20, 12, 6, 7, 1, 1, 1, 1, 1, 1, 1, 53]
        col_idxs = [0, 20, 32, 38, 45, 46, 47, 48, 49, 50, 51, 0]
        global feature_names
    else:
        col_widths = [1] * 52
        col_idxs = list(range(52))
        feature_names = col_names

    for perturb_name, perturb in [
        ("zeroed", perturb_zero),
        ("inverted", perturb_invert),
        ("random", perturb_random),
        ("2x'd", perturb_scale),
    ]:
        for feat_name, c_idx, col_width in zip(feature_names, col_idxs, col_widths):
            if feat_name != "all":
                continue
            handle = a2l.normalize.register_forward_hook(lambda m, i, o: perturb(o))
            _audio2video(
                a2l=a2l,
                test_features=features[:, int(22.5 * 24) : int((22.5 + 45) * 24)].clone(),
                audio_file=test_audio,
                out_file=f"runs/{Path(checkpoint).stem}_{Path(test_audio).stem}_{feat_name}_{perturb_name}.mp4",
                stylegan_file="/home/hans/modelzoo/train_checks/neurout2-117.pt",
                offset=22.5,
                duration=45,
                output_size=(1024, 1024),
                seed=123,
            )
            handle.remove()


@torch.inference_mode()
def feature_sensitivity(whole_feature, checkpoint):
    a2l = load_a2l(checkpoint)

    cache_file = "cache/test_features.npy"
    if not os.path.exists(cache_file):
        features = []
        min_len = 1e10
        for test_audio in tqdm(glob("/home/hans/datasets/wavefunk/*")):
            if rosa.get_duration(filename=test_audio) < 128:
                continue
            audio, sr = torchaudio.load(test_audio)
            feats = audio2features(audio, sr, FPS)
            features.append(feats)
            if len(feats) < min_len:
                min_len = len(feats)
        features = torch.stack([f[:min_len] for f in features])
        np.save(cache_file, features.numpy())
    else:
        features = torch.from_numpy(np.load(cache_file))
    features = features[:, :3072].to(device)

    norm_feats = a2l.normalize(features)

    def perturb_zero(x):
        x[..., c_idx : c_idx + col_width] = 0
        return x

    def perturb_invert(x):
        x[..., c_idx : c_idx + col_width] *= -1
        return x

    def perturb_random(x):
        x[..., c_idx : c_idx + col_width] = torch.randn_like(x[..., c_idx : c_idx + col_width])
        return x

    def perturb_scale(x):
        x[..., c_idx : c_idx + col_width] *= 2
        return x

    if whole_feature:
        col_widths = [20, 12, 6, 7, 1, 1, 1, 1, 1, 1, 1, 53]
        col_idxs = [0, 20, 32, 38, 45, 46, 47, 48, 49, 50, 51, 0]
        global feature_names
    else:
        col_widths = [1] * 52
        col_idxs = list(range(52))
        feature_names = col_names

    trials = 5

    y = a2l(features).mean(2).squeeze()
    b, t, c = y.shape
    y = y.reshape(b * t, c)

    x = norm_feats.squeeze().clone()
    b, t, c = x.shape
    x = x.reshape(b * t, c)

    xs = torch.stack(sum([list(torch.split(x[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)
    ys = torch.stack(sum([list(torch.split(y[start:], 24)[:-1]) for start in range(0, 24, 8)], [])).flatten(1)

    ref_sgws = [sgw_gpu(xs, ys, device).item() for _ in range(trials)]

    print()
    print(
        " | ".join(
            [
                "",
                f"feature".ljust(15),
                "perturbation".ljust(15),
                "grom/wass".ljust(10),
                "std. dev.".ljust(10),
                "t-stat".ljust(10),
                "p-value".ljust(10),
                "*",
                "",
            ]
        )
    )
    print(" | ".join(["", "-" * 15, "-" * 15, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-", ""]))
    print(
        " | ".join(
            [
                "",
                f"all".ljust(15),
                "none".ljust(15),
                f"{np.mean(ref_sgws):.4f}".ljust(10),
                f"{np.std(ref_sgws):.4f}".ljust(10),
                "".ljust(10),
                "".ljust(10),
                " ",
                "",
            ]
        )
    )

    for perturb_name, perturb in [
        ("zeroed", perturb_zero),
        ("inverted", perturb_invert),
        ("random", perturb_random),
        ("2x'd", perturb_scale),
    ]:
        for feat_name, c_idx, col_width in zip(feature_names, col_idxs, col_widths):
            x = norm_feats.squeeze().clone()
            b, t, c = x.shape
            x = x.reshape(b * t, c)
            x = perturb(x)

            xs = torch.stack(sum([list(torch.split(x[start:], 192)[:-1]) for start in range(0, 192, 24)], [])).flatten(
                1
            )
            ys = torch.stack(sum([list(torch.split(y[start:], 192)[:-1]) for start in range(0, 192, 24)], [])).flatten(
                1
            )

            sgws = [sgw_gpu(xs, ys, device).item() for _ in range(trials)]
            t, p = ttest_ind(ref_sgws, sgws)

            print(
                " | ".join(
                    [
                        "",
                        feat_name.ljust(15),
                        perturb_name.ljust(15),
                        f"{np.mean(sgws):.4f}".ljust(10),
                        f"{np.std(sgws):.4f}".ljust(10),
                        f"{t:.4f}".ljust(10),
                        f"{p:.4f}".ljust(10),
                        # " " if abs(np.mean(sgws) - np.mean(ref_sgws)) < 0.03 else "*",
                        " " if p > 0.01 else "*",
                        "",
                    ]
                )
            )

    print()
    print(
        " | ".join(["", f"feature".ljust(15), "perturbation".ljust(15), "l1/100000".ljust(15), "l2/100".ljust(15), ""])
    )
    print(" | ".join(["", "-" * 15, "-" * 15, "-" * 15, "-" * 15, ""]))
    yr = a2l(features).squeeze()
    for perturb_name, perturb in [
        ("zeroed", perturb_zero),
        ("inverted", perturb_invert),
        ("random", perturb_random),
        ("2x'd", perturb_scale),
    ]:
        for feat_name, c_idx, col_width in zip(feature_names, col_idxs, col_widths):
            handle = a2l.normalize.register_forward_hook(lambda m, i, o: perturb(o))
            y = a2l(features).squeeze()
            l1 = torch.norm(yr - y, p=1).item()
            l2 = torch.norm(yr - y, p=2).item()
            print(
                " | ".join(
                    [
                        "",
                        feat_name.ljust(15),
                        perturb_name.ljust(15),
                        f"{l1/100_000:.4f}".ljust(15),
                        f"{l2/100:.4f}".ljust(15),
                        "",
                    ]
                )
            )
            handle.remove()


@torch.inference_mode()
def test_new_spline():
    from scipy.interpolate import splev, splrep

    def spline_loop_latents(y, size):
        y = torch.cat((y, y[[0]]))
        t = torch.linspace(0, 1, len(y)).to(y)
        x = y.permute(1, 0, 2)
        out = NaturalCubicSpline(natural_cubic_spline_coeffs(t, x)).evaluate(torch.linspace(0, 1, size).to(y))
        return out.permute(1, 0, 2)

    def spline_loop_latents_np(latent_selection, loop_len):
        latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])
        x = np.linspace(0, 1, loop_len)
        latents = np.zeros((loop_len, *latent_selection.shape[1:]), dtype=np.float32)
        for lay in range(latent_selection.shape[1]):
            for lat in range(latent_selection.shape[2]):
                tck = splrep(np.linspace(0, 1, latent_selection.shape[0]), latent_selection[:, lay, lat])
                latents[:, lay, lat] = splev(x, tck)
        latents = torch.from_numpy(latents)
        return latents

    checkpoint = "/home/hans/modelzoo/train_checks/neurout2-117.pt"
    output_size = (512, 512)
    mapper = StyleGAN2Mapper(model_file=checkpoint, inference=False).eval().to(device)
    synthesizer = (
        StyleGAN2Synthesizer(
            model_file=checkpoint, inference=False, output_size=output_size, strategy="stretch", layer=0
        )
        .eval()
        .to(device)
    )
    latent_selection = mapper(torch.randn((16, 512), device=device))

    from time import time

    trials = 10

    t = time()
    for _ in range(trials):
        latents = spline_loop_latents_np(latent_selection.cpu().numpy(), 512).to(device)
    print("scipy", (time() - t) / trials * 1000, "ms")

    t = time()
    for _ in range(trials):
        latents = spline_loop_latents(latent_selection, 512)
    print("kidger", (time() - t) / trials * 1000, "ms")

    batch_size = 16
    with VideoWriter(output_file=f"output/test_new_spline.mp4", output_size=output_size) as video:
        for i in tqdm(range(0, len(latents), batch_size), unit_scale=batch_size):
            for frame in synthesizer(latents[i : i + batch_size].to(device)).add(1).div(2):
                video.write(frame.unsqueeze(0))


@torch.inference_mode()
def test_latent_augmenter():
    from train_supervised import LatentAugmenter

    checkpoint = "/home/hans/modelzoo/train_checks/neurout2-117.pt"
    n_patches = 3
    output_size = (512, 512)
    augmenter = LatentAugmenter(checkpoint, n_patches)
    synthesizer = (
        StyleGAN2Synthesizer(
            model_file=checkpoint, inference=False, output_size=output_size, strategy="stretch", layer=0
        )
        .eval()
        .to(device)
    )

    vid_len = 20
    fps = FPS
    vid_frames = vid_len * fps
    batch_size = 16

    for test_audio in tqdm(glob("/home/hans/datasets/wavefunk/*")):
        if rosa.get_duration(filename=test_audio) <= 40 or "finow" in test_audio:
            continue
        audio, sr = torchaudio.load(test_audio)
        feats = audio2features(audio, sr, fps)
        start_idx = np.random.randint(0, len(feats) - vid_frames)
        feats = feats[start_idx : start_idx + vid_frames]
        residuals, offset = augmenter(feats)

        with VideoWriter(
            output_file=f"output/{Path(test_audio).stem}_{start_idx}.mp4",
            output_size=output_size,
            audio_file=test_audio,
            audio_offset=start_idx / fps,
            audio_duration=vid_len,
        ) as video:
            for i in tqdm(range(0, len(residuals), batch_size), unit_scale=batch_size):
                for frame in (
                    synthesizer((residuals[i : i + batch_size] + offset[i : i + batch_size]).to(device)).add(1).div(2)
                ):
                    video.write(frame.unsqueeze(0))


if __name__ == "__main__":
    # test_output_sensitivity(
    #     "runs/Feb24_10-39-54_ubuntu94025backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented/audio2latent_backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented_steps00061440_fcd40.1304_b0.0425_val0.0669.pt",
    #     True,
    #     True,
    # )
    # test_output_sensitivity(
    #     "runs/Feb24_10-39-54_ubuntu94025backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented/audio2latent_backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented_steps00061440_fcd40.1304_b0.0425_val0.0669.pt",
    #     False,
    #     False,
    # )
    # feature_sensitivity(
    #     True,
    #     "runs/Feb24_10-39-54_ubuntu94025backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented/audio2latent_backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented_steps00061440_fcd40.1304_b0.0425_val0.0669.pt",
    # )
    feature_sensitivity(
        False,
        "runs/Feb24_10-39-54_ubuntu94025backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented/audio2latent_backbone:gru:skipFalse_layerwise:conv:3_hidden_size:64_num_layers:8_dropout:0.2_lr:0.001_wd:0_augmented_steps00061440_fcd40.1304_b0.0425_val0.0669.pt",
    )
    exit()

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
            latent_residuals = a2l(audio2features(*torchaudio.load(args.audio), FPS).to(device).unsqueeze(0))
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
