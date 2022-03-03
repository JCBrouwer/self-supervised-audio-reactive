import argparse
import gc
import importlib
import os
import sys
import warnings
from copy import deepcopy
from glob import glob
from math import log2
from pathlib import Path

import joblib
import librosa as rosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy import stats
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, Dataset
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from tqdm import tqdm

from ..analysis.sgw import sgw_gpu
from ..models.audio2latent import Normalize

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from ops.video import VideoWriter

FPS = 24
DUR = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@torch.inference_mode()
def test_output_sensitivity(checkpoint, plot=False, whole_feature=True):
    a2l, a2f = load_a2l(checkpoint)

    test_audio = "/home/hans/datasets/wavefunk/tau ceti alpha.flac"
    audio, sr = torchaudio.load(test_audio)
    features = a2f(audio, sr, FPS).unsqueeze(0).to(device)

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
    a2l, a2f = load_a2l(checkpoint)

    cache_file = "cache/test_features.npy"
    if not os.path.exists(cache_file):
        features = []
        min_len = 1e10
        for test_audio in tqdm(glob("/home/hans/datasets/wavefunk/*")):
            if rosa.get_duration(filename=test_audio) < 128:
                continue
            audio, sr = torchaudio.load(test_audio)
            feats = a2f(audio, sr, FPS)
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
    from .data import audio2features
    from .latent_augmenter import LatentAugmenter

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


@torch.inference_mode()
def latent_residual_hist(latent_residuals):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

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
    plt.savefig(f"plots/latent-residuals-distribution-true.png")


@torch.inference_mode()
def _audio2video(
    a2l,
    test_features,
    audio_file,
    out_file,
    stylegan_file,
    fps,
    output_size=(512, 512),
    batch_size=8,
    offset=0,
    duration=40,
    seed=42,
):
    test_latents = a2l(test_features).squeeze()

    mapper = StyleGAN2Mapper(model_file=stylegan_file, inference=False)
    latent_offset = mapper(torch.from_numpy(np.random.RandomState(seed).randn(1, 512))).to(device)
    del mapper

    synthesizer = StyleGAN2Synthesizer(
        model_file=stylegan_file, inference=False, output_size=output_size, strategy="stretch", layer=0
    )
    synthesizer.eval().to(device)

    with VideoWriter(
        output_file=out_file,
        output_size=output_size,
        fps=fps,
        audio_file=audio_file,
        audio_offset=offset,
        audio_duration=duration,
    ) as video:
        for i in tqdm(range(0, len(test_latents), batch_size), unit_scale=batch_size):
            for frame in synthesizer(latent_offset + test_latents[i : i + batch_size]).add(1).div(2):
                video.write(frame.unsqueeze(0))

    del test_features, test_latents, latent_offset, synthesizer
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def audio2video(
    a2l,
    a2f,
    audio_file,
    out_file,
    stylegan_file,
    fps=24,
    output_size=(512, 512),
    batch_size=8,
    offset=0,
    duration=40,
    onsets_only=False,
):
    try:
        a2l = a2l.eval().to(device)
    except:
        pass
    audio, sr = torchaudio.load(audio_file)
    test_features = a2f(audio, sr, fps, onsets_only=onsets_only)
    test_features = test_features[24 * offset : 24 * (offset + duration)]
    test_features = test_features.unsqueeze(0).to(device)
    _audio2video(
        a2l, test_features, audio_file, out_file, stylegan_file, fps, output_size, batch_size, offset, duration
    )


class AudioFeatures(Dataset):
    def __init__(self, directory, a2f, dur, fps):
        super().__init__()
        self.files = sum([glob(f"{directory}*.{ext}") for ext in ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]], [])
        self.a2f = a2f
        self.L = dur * fps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        audio, sr = torchaudio.load(file)
        features = self.a2f(audio, sr)
        features = torch.stack(  # overlapping slices of DUR seconds
            sum([list(torch.split(features[start:], self.L)[:-1]) for start in range(0, self.L, self.L // 4)], [])
        )
        return features


class ModuleFromFile:
    def __init__(self, path, name) -> None:
        self.path = path
        self.name = name

    def __enter__(self):
        directory = os.path.dirname(self.path)
        with open(f"{directory}/{self.name}.py", "r") as f:
            text = f.read()
        text = text.replace("from context_fid", "# from context_fid").replace("# #", "#")
        with open(f"{directory}/{self.name}.py", "w") as f:
            f.write(text)
        sys.path = [directory] + sys.path
        self.original_main_module = {}
        module = importlib.import_module(self.name)
        importlib.reload(module)
        for k in module.__dict__:
            if not k.startswith("_"):
                try:
                    self.original_main_module[k] = deepcopy(getattr(sys.modules["__main__"], k))
                except:
                    pass
                setattr(sys.modules["__main__"], k, getattr(module, k))
        return module

    def __exit__(self, exc_type, exc_value, exc_tb):
        for k in self.original_main_module:
            setattr(sys.modules["__main__"], k, self.original_main_module[k])
        sys.path = sys.path[1:]


@torch.inference_mode()
def load_a2l(path):
    # HACK to import all modules that model needs from it's training code cache

    dirname = os.path.dirname(path)

    if "PSAGAN" in dirname:
        with ModuleFromFile(path, "ssar") as ssar:
            ckpt = joblib.load(path)
            G = ckpt["G"].eval().to(device)
            n_out = G.layerwise.NO
            out_size = G.layerwise.b2.shape[1] // G.layerwise.NL
            full_len = ckpt["D"].target_length
            a2l = (
                lambda x: F.interpolate(
                    G(x.permute(0, 2, 1)).reshape(x.shape[0], -1, x.shape[1]),
                    scale_factor=2 ** int(log2(full_len) - (3 + G.depth)),
                )
                .permute(0, 2, 1)
                .reshape(x.shape[0], -1, n_out, out_size)
            )
            a2f = ssar.supervised.data.audio2features

    elif "train_supervised.py" in os.listdir(dirname):
        with ModuleFromFile(path, "train_supervised") as module:
            a2l = joblib.load(path)["a2l"].eval().to(device)

            def a2f(x, sr, fps=24):
                try:
                    return module.audio2features(x, sr, fps)
                except:
                    try:
                        return module.audio2features(x, sr)
                    except:
                        raise Exception("wat")

            in_dir = "/home/hans/datasets/audio2latent/"
            try:
                dur = module.DUR
            except:
                dur = DUR
            try:
                fps = module.FPS
            except:
                fps = FPS

        print(a2l.__class__.__name__, hasattr(a2l, "normalize"))

        if not hasattr(a2l, "normalize"):
            if not os.path.exists(f"{dirname}/mean.npy"):
                feats = torch.cat(
                    [f.squeeze() for f in tqdm(DataLoader(AudioFeatures(in_dir, a2f, dur, fps), num_workers=24))]
                )
                mean = torch.mean(feats)
                std = torch.std(feats)
                np.save(f"{dirname}/mean.npy", mean.cpu().numpy())
                np.save(f"{dirname}/std.npy", std.cpu().numpy())
            else:
                mean = torch.from_numpy(np.load(f"{dirname}/mean.npy"))
                std = torch.from_numpy(np.load(f"{dirname}/std.npy"))
            normalize = Normalize(mean, std).to(device)

            def norm_hook(mod, x):
                return normalize(x[0])

            a2l.register_forward_pre_hook(norm_hook)

    else:
        with ModuleFromFile(path, "ssar") as ssar:
            a2l = joblib.load(path)["a2l"].eval().to(device)
            a2f = ssar.supervised.data.audio2features

    return a2l, a2f


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
    parser.add_argument("--duration", help="length in seconds of video to render", type=int, default=0)
    parser.add_argument("--plot", help="output distribution plot instead of only printing laplace scale", action='store_true')
    parser.add_argument("--insense", help="test input sensitivity", action='store_true')
    parser.add_argument("--outsense", help="test output sensitivity", action='store_true')
    parser.add_argument("--whole_feat_sense", help="whether to use whole feature or not", action='store_true')
    parser.add_argument("--lap_scale", help="print laplacian scale", action='store_true')
    args = parser.parse_args()
    # fmt:on

    with torch.inference_mode():
        if args.insense:
            feature_sensitivity(args.whole_feat_sense, args.ckpt)
        if args.outsense:
            test_output_sensitivity(args.ckpt, args.plot, args.whole_feat_sense)

        a2l, a2f = load_a2l(args.ckpt)

        if args.duration > 0:
            current_a2v(
                a2l=a2l,
                a2f=a2f,
                audio_file=args.audio,
                out_file=f"{os.path.splitext(args.ckpt)[0]}_{Path(args.audio).stem}.mp4",
                stylegan_file=args.stylegan,
                output_size=[int(s) for s in args.output_size],
                batch_size=args.batch_size,
                offset=args.offset,
                duration=args.duration,
            )

        if args.lap_scale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                latent_residuals = a2l(a2f(*torchaudio.load(args.audio), FPS).to(device).unsqueeze(0))
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
