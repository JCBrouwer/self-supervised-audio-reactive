# fmt: off
import argparse
import os
from glob import glob
from pathlib import Path

import decord as de
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from tqdm import tqdm

from .audio import (chromagram, drop_strength, mfcc, onsets, pulse, rms,
                    spectral_contrast, spectral_flatness, tonnetz)
from .correlation import op, pwcca, rv2, smi, svcca
from .video import (absdiff, adaptive_freq_rms, directogram, high_freq_rms,
                    hsv_hist, low_freq_rms, mid_freq_rms, rgb_hist,
                    video_flow_onsets, video_spectral_onsets,
                    video_spectrogram, visual_variance)

afns = [chromagram, tonnetz, mfcc, spectral_contrast, spectral_flatness, rms, drop_strength, onsets, pulse]
vfns = [rgb_hist, hsv_hist, video_spectrogram, directogram, low_freq_rms, mid_freq_rms, high_freq_rms, 
        adaptive_freq_rms, absdiff, visual_variance, video_flow_onsets, video_spectral_onsets]

de.bridge.set_bridge("torch")
# fmt: on


def load_audio_video(path, downsample=4, resample_fps=24, enforce_shapes=True):
    v = de.VideoReader(path)
    fps = round(v.get_avg_fps())
    l, (h, w, c) = len(v), v[0].shape
    dur = l / fps
    del v

    sr = round(1024 * fps)
    av = de.AVReader(path, sample_rate=sr, width=w // downsample, height=h // downsample)
    audio, video = av[:]
    del av

    audio = torch.cat(audio, dim=1).float().squeeze().contiguous()
    video = video.permute(0, 3, 1, 2).float().div(255).contiguous()

    if resample_fps and fps != resample_fps:
        audio = resample(audio, sr, 1024 * resample_fps).contiguous()
        video = video.permute(1, 2, 3, 0).reshape(c * h, w, l)
        video = interpolate(video, size=round(dur * resample_fps))
        video = video.reshape(c, h, w, -1).permute(3, 0, 1, 2).contiguous()

    if enforce_shapes:
        l, c, h, w = video.shape
        dur = l / resample_fps
        audio = audio[: round(dur) * 1024 * resample_fps]
        s = min(h, w)
        lh, lw = (h - s) // 2, (w - s) // 2
        uh, uw = h if lh == 0 else -lh, w if lw == 0 else -lw
        video = video[: round(dur) * resample_fps, :, lh:uh, lw:uw]

    return audio, sr, video, fps


class AudioVisualDataset(Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        return filepath, load_audio_video(filepath)


def audiovisual_correlation(afeats, vfeats, cname, correlation_fn, quadratic=False, variation_normalized=False):
    if quadratic:
        res = {}
        for aname, afeat in afeats.items():
            for vname, vfeat in vfeats.items():
                cor = correlation_fn(afeat, vfeat).item()
                if variation_normalized:
                    cor *= afeat.std(0).mean() / (afeat.norm() + 1e-8) + vfeat.std(0).mean() / (vfeat.norm() + 1e-8)
                res[(aname, vname, cname)] = cor
    else:
        afeatvals, vfeatvals = list(afeats.values()), list(vfeats.values())
        af, vf = afeatvals[0], vfeatvals[0]
        for afeat in afeatvals:
            af = torch.cat((af, afeat), dim=1)
        for vfeat in vfeatvals:
            vf = torch.cat((vf, vfeat), dim=1)
        res = correlation_fn(af, vf).item()
        if variation_normalized:
            res *= af.std(0).mean() / (af.norm() + 1e-8) + vf.std(0).mean() / (vf.norm() + 1e-8)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("groups", nargs="+")
    parser.add_argument("--titles", nargs="*", default=[])
    parser.add_argument("-n", type=int, default=None)
    args = parser.parse_args()

    with torch.inference_mode():
        data_dir = args.data_dir
        exp_name = Path(data_dir).stem

        which = slice(args.n) if args.n is not None else slice(None)
        file_groups = {name: sorted(glob(f"{data_dir}/{name}*.mp4"))[which] for name in args.groups}
        for name, files in file_groups.items():
            print(f"# {name} samples:", len(files))
        print()

        print("# audio features:", len(afns))
        print("# video features:", len(vfns))
        print()

        csv_file = f"output/audiovisual_correlations_{exp_name}.csv"
        if not os.path.exists(csv_file):
            results = []
            for group, files in file_groups.items():
                for (filepath,), ((audio,), sr, (video,), _) in tqdm(
                    DataLoader(AudioVisualDataset(files), num_workers=24)
                ):
                    audio, sr, video = audio.cuda(), sr.item(), video.cuda()

                    if not os.path.exists(filepath.replace(".mp4", "_vfeats.npz")):
                        audio, video = audio.cuda(), video.cuda()
                        afeats = {af.__name__: af(audio, sr) for af in afns}
                        vfeats = {vf.__name__: vf(video) for vf in vfns}
                        np.savez_compressed(
                            filepath.replace(".mp4", "_afeats.npz"), **{n: f.cpu().numpy() for n, f in afeats.items()}
                        )
                        np.savez_compressed(
                            filepath.replace(".mp4", "_vfeats.npz"), **{n: f.cpu().numpy() for n, f in vfeats.items()}
                        )
                    else:
                        with np.load(filepath.replace(".mp4", "_afeats.npz")) as arr:
                            afeats = {af.__name__: torch.from_numpy(arr[af.__name__]).cuda() for af in afns}
                        with np.load(filepath.replace(".mp4", "_vfeats.npz")) as arr:
                            vfeats = {vf.__name__: torch.from_numpy(arr[vf.__name__]).cuda() for vf in vfns}

                    row = {"group": group}
                    for cname, correlation in {c.__name__: c for c in [op, pwcca, rv2, smi, svcca]}.items():
                        row = {**row, **audiovisual_correlation(afeats, vfeats, cname, correlation, quadratic=True)}
                        row[("concat", "concat", cname)] = audiovisual_correlation(
                            afeats, vfeats, cname, correlation, quadratic=False
                        )
                    results.append(row)
            df = pd.DataFrame(results)
            print(df)

            df.to_csv(csv_file)
        else:
            df = pd.read_csv(csv_file, index_col=0)

        print(df)

        for group, files in file_groups.items():
            print("\n", group)
            gdf = df[df["group"] == group].reset_index(drop=True)
            highest = gdf.nlargest(n=5, columns=[("concat", "concat", "op")])
            lowest = gdf.nsmallest(n=5, columns=[("concat", "concat", "op")])

            print("best")
            for idx, vals in highest.iterrows():
                print(files[idx], vals[("concat", "concat", "op")])

            print("worst")
            for idx, vals in lowest.iterrows():
                print(files[idx], vals[("concat", "concat", "op")])
        exit()

        grouped = df.groupby("group", sort=False).agg(["mean", "std"])

        stats = []
        for col in grouped.columns[::2]:
            key, _ = col
            af, vf, cn = key
            means = grouped[(key, "mean")]
            stds = grouped[(key, "std")]
            for g, group in enumerate(file_groups):
                group_name = args.titles[g] if len(args.titles) > 0 else group
                stats.append(
                    {
                        "correlation": cn.replace("op", "Orthogonal Procrustes")
                        .replace("rv2", "Adjusted RV Coefficient")
                        .replace("smi", "Matrix Similarity Index")
                        .replace("pwcca", "Projection-weighted Canonical Correlation Analysis"),
                        "audio": af,
                        "video": vf,
                        "group": group_name,
                        "mean": means[group],
                        "std": stds[group],
                    }
                )
        stats = pd.DataFrame(stats)

        stats[(stats["audio"] == "concat") & (stats["correlation"] == "Orthogonal Procrustes")].plot.bar(
            x="group",
            y="mean",
            yerr="std",
            color=plt.get_cmap("YlOrRd")(np.linspace(0, 1, len(file_groups))),
            legend=False,
            xlabel="Group",
            rot=10,
        )
        plt.tight_layout()
        plt.savefig(f"output/{exp_name}_barplotconcatcorr.pdf")
        plt.close()

        stats = stats[stats["correlation"] != "svcca"]
        stats = stats[stats["audio"] != "concat"]
        groups = stats["group"].unique()
        correlations = stats["correlation"].unique()

        def groupbars():
            for pdf, data_fn in [
                ("full", lambda: corrstats.groupby("group", sort=False)[["mean", "std"]].mean()),
                (
                    "chroma",
                    lambda: corrstats[corrstats.audio == "chromagram"]
                    .groupby("group", sort=False)[["mean", "std"]]
                    .mean(),
                ),
                (
                    "onsets",
                    lambda: corrstats[corrstats.audio == "onsets"].groupby("group", sort=False)[["mean", "std"]].mean(),
                ),
            ]:
                fig, ax = plt.subplots(len(correlations) // 2, 2, figsize=(16, 9), sharex=True)
                for c, corr in enumerate(correlations):
                    corrstats = stats[stats["correlation"] == corr]
                    colors = plt.get_cmap("YlOrRd")(np.linspace(0, 1, len(groups)))
                    data_fn().plot.bar(
                        y="mean", yerr="std", ax=ax.flatten()[c], color=colors, legend=False, xlabel="Group", rot=10
                    )
                    ax.flatten()[c].set_title(corr)
                plt.tight_layout()
                plt.savefig(f"output/{exp_name}_{pdf}groupcorrbars mean.pdf")
                plt.close()

        def grouphists():
            for pdf, data_fn in [
                ("full", lambda: groupstats["mean"]),
                ("chroma", lambda: groupstats[groupstats.audio == "chromagram"]["mean"]),
                ("onsets", lambda: groupstats[groupstats.audio == "onsets"]["mean"]),
                ("audio", lambda: groupstats.groupby(["audio"], sort=False)["mean"].mean()),
                ("video", lambda: groupstats.groupby(["video"], sort=False)["mean"].mean()),
            ]:
                fig, ax = plt.subplots(len(correlations) // 2, 2, figsize=(16, 9), sharex=True)
                maxheight = [0 for _ in range(len(correlations))]
                for c, corr in enumerate(correlations):
                    corrstats = stats[stats["correlation"] == corr]
                    for g, group in enumerate(groups):
                        groupstats = corrstats[corrstats["group"] == group]
                        data = data_fn()
                        color = plt.get_cmap("YlOrRd")(np.linspace(0, 1, len(groups))[g])
                        yvals, _, _ = ax.flatten()[c].hist(
                            data,
                            label=group,
                            bins=100,
                            range=(stats["mean"].min(), stats["mean"].max()),
                            alpha=0.666,
                            color=color,
                        )
                        maxheight[c] = max(maxheight[c], yvals.max())
                for c, corr in enumerate(correlations):
                    corrstats = stats[stats["correlation"] == corr]
                    for g, group in enumerate(groups):
                        groupstats = corrstats[corrstats["group"] == group]
                        data = data_fn()
                        color = plt.get_cmap("YlOrRd")(np.linspace(0, 1, len(groups))[g])
                        ax.flatten()[c].vlines(data.mean(), 0, maxheight[c], ls="--", color=color)
                    ax.flatten()[c].set_xlabel(corr)
                plt.tight_layout()
                ax.flatten()[0].legend()
                plt.savefig(f"output/{exp_name}_{pdf}groupcorrhists mean.pdf")
                plt.close()

        def heatmap():
            A = len(afns)
            V = len(vfns)
            hot = plt.get_cmap("hot")

            figs = []
            for c, corr in enumerate(correlations):
                corrstats = stats[stats["correlation"] == corr]
                for g, group in enumerate(groups):
                    groupstats = corrstats[corrstats["group"] == group]

                    g = sns.jointplot(data=groupstats, x="video", y="audio", kind="hist", bins=(V, A))
                    g.ax_marg_y.cla()
                    g.ax_marg_x.cla()
                    sns.heatmap(
                        data=groupstats["mean"].values.reshape(A, V),
                        ax=g.ax_joint,
                        cbar=False,
                        cmap=hot,
                        vmin=0,
                        vmax=stats["mean"].max(),
                    )

                    make_axes_locatable(g.ax_marg_x).append_axes("left", size="10%", pad="20%").axis("off")
                    cax = make_axes_locatable(g.ax_joint).append_axes("left", size="10%", pad="20%")
                    g.fig.colorbar(g.ax_joint.get_children()[1], cax=cax)
                    cax.yaxis.set_ticks_position("left")

                    np.set_printoptions(linewidth=200)
                    audio_feature_marginals = groupstats.groupby(["audio"], sort=False)["mean"].mean()
                    video_feature_marginals = groupstats.groupby(["video"], sort=False)["mean"].mean()
                    g.ax_marg_y.barh(
                        np.arange(0.5, A), audio_feature_marginals.values, color=hot(audio_feature_marginals.values)
                    )
                    g.ax_marg_y.set(xlim=(0, stats["mean"].max()))
                    g.ax_marg_x.bar(
                        np.arange(0.5, V), video_feature_marginals.values, color=hot(video_feature_marginals.values)
                    )
                    g.ax_marg_x.set(ylim=(0, stats["mean"].max()))

                    g.ax_joint.set_xticks(np.arange(0.5, V))
                    g.ax_joint.set_xticklabels(
                        groupstats["video"].unique(), rotation=20, ha="right", rotation_mode="anchor"
                    )
                    g.ax_joint.set_yticks(np.arange(0.5, A))
                    g.ax_joint.set_yticklabels(groupstats["audio"].unique(), rotation=0)

                    # remove ticks between heatmap and histograms
                    g.ax_marg_x.tick_params(axis="x", bottom=False, labelbottom=False)
                    g.ax_marg_y.tick_params(axis="y", left=False, labelleft=False)
                    # # remove ticks showing the heights of the histograms
                    g.ax_marg_x.tick_params(axis="y", left=False, labelleft=False)
                    g.ax_marg_y.tick_params(axis="x", bottom=False, labelbottom=False)

                    g.fig.suptitle(group)
                    g.fig.set_size_inches(16, 9)
                    plt.savefig(f"output/{exp_name}_{corr}_{group} heatmap mean.pdf")
                    plt.close()

        heatmap()
        grouphists()
        groupbars()
