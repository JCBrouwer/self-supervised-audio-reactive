# fmt: off
import argparse
import os
from glob import glob
from pathlib import Path

import decord as de
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torchdatasets as td
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# fmt: on

sns.set_theme(context="paper", style="white", palette="tab10")


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
        l, c, h, w = video.shape
        audio = resample(audio, sr, 1024 * resample_fps).contiguous()
        video = video.permute(1, 2, 3, 0).reshape(c * h, w, l)
        video = interpolate(video, size=round(dur * resample_fps))
        fps = resample_fps
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


class AudioVisualDataset(td.Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        return filepath, load_audio_video(filepath)


def audiovisual_correlation(
    afeats, vfeats, cname, correlation_fn, quadratic=False, variation_normalized=False, mean_diff_normalized=False
):
    if quadratic:
        res = {}
        for aname, afeat in afeats.items():
            for vname, vfeat in vfeats.items():
                cor = correlation_fn(afeat, vfeat).item()
                if mean_diff_normalized:
                    asmooth = torch.max(torch.diff(afeat, dim=0).abs() / afeat.abs().max(dim=0).values)
                    vsmooth = torch.max(torch.diff(vfeat, dim=0).abs() / vfeat.abs().max(dim=0).values)
                    cor *= (asmooth + vsmooth).item()
                if variation_normalized:
                    cor *= (
                        afeat.std(0).mean() / (afeat.norm() + 1e-8) + vfeat.std(0).mean() / (vfeat.norm() + 1e-8)
                    ).item()
                res[(aname, vname, cname)] = cor
    else:
        afeatvals, vfeatvals = list(afeats.values()), list(vfeats.values())
        af, vf = afeatvals[0], vfeatvals[0]
        for afeat in afeatvals:
            af = torch.cat((af, afeat), dim=1)
        for vfeat in vfeatvals:
            vf = torch.cat((vf, vfeat), dim=1)
        res = correlation_fn(af, vf).item()
        if mean_diff_normalized:
            asmooth = torch.max(torch.diff(af, dim=0) / af.max(dim=0).values)
            vsmooth = torch.max(torch.diff(vf, dim=0) / vf.max(dim=0).values)
            res *= (asmooth + vsmooth).item()
        if variation_normalized:
            res *= (af.std(0).mean() / (af.norm() + 1e-8) + vf.std(0).mean() / (vf.norm() + 1e-8)).item()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("groups", nargs="+")
    parser.add_argument("--titles", nargs="*", default=[])
    parser.add_argument("-n", type=int, default=None)
    args = parser.parse_args()

    corr_fns = {c.__name__: c for c in [op, pwcca, rv2, smi, svcca]}
    COLORS = ["forestgreen", "limegreen", "blueviolet", "tab:purple", "tab:pink", "tab:orange"]

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
                        afeats = {af.__name__: af(audio.cpu(), sr).cuda() for af in afns}
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
                    for cname, correlation in corr_fns.items():
                        row = {**row, **audiovisual_correlation(afeats, vfeats, cname, correlation, quadratic=True)}
                        row[("concat", "concat", cname)] = audiovisual_correlation(
                            afeats, vfeats, cname, correlation, quadratic=False
                        )
                    results.append(row)
            df = pd.DataFrame(results)
            df.to_csv(csv_file)
        else:
            df = pd.read_csv(csv_file, index_col=0)
            df = df.rename(columns={col: eval(col) for col in df.columns if "(" in col})
            # df = df.apply(
            #     lambda row: {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in row.items()}, axis=1
            # )

        def extrema():
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

        # extrema()

        grouped = df.groupby("group", sort=False).agg(["median", "std"])

        stats = []
        for col in grouped.columns[::2]:
            af, vf, cn, _ = col
            medians = grouped[(af, vf, cn, "median")]
            stds = grouped[(af, vf, cn, "std")]
            for g, group in enumerate(file_groups):
                group_name = args.titles[g] if len(args.titles) > 0 else group
                stats.append(
                    {
                        "correlation": cn.replace("op", "Orthogonal Procrustes")
                        .replace("rv2", "Adjusted RV Coefficient")
                        .replace("smi", "Matrix Similarity Index")
                        .replace("pwcca", "Projection-weighted CCA"),
                        "audio": af,
                        "video": vf,
                        "group": group_name,
                        "median": medians[group],
                        "std": stds[group],
                    }
                )
        stats = pd.DataFrame(stats)

        def concatbar():
            stats[(stats["audio"] == "concat") & (stats["correlation"] == "Orthogonal Procrustes")].plot.bar(
                x="group",
                y="median",
                yerr="std",
                color=COLORS,
                legend=False,
                xlabel="Group",
                rot=10,
            )
            plt.tight_layout()
            sns.despine()
            plt.savefig(f"output/{exp_name}_barplotconcatcorr.pdf")
            plt.close()

        # concatbar()

        stats = stats[stats["correlation"] != "svcca"]
        stats = stats[stats["audio"] != "concat"]
        groups = stats["group"].unique()
        correlations = stats["correlation"].unique()

        def groupbars():
            for pdf, data_fn in [
                ("full", lambda: corrstats.groupby("group", sort=False)[["median", "std"]].median()),
                (
                    "chroma",
                    lambda: corrstats[corrstats.audio == "chromagram"]
                    .groupby("group", sort=False)[["median", "std"]]
                    .median(),
                ),
                (
                    "onsets",
                    lambda: corrstats[corrstats.audio == "onsets"]
                    .groupby("group", sort=False)[["median", "std"]]
                    .median(),
                ),
            ]:
                fig, ax = plt.subplots(len(correlations) // 2, 2, figsize=(16, 9), sharex=True)
                for c, corr in enumerate(correlations):
                    corrstats = stats[stats["correlation"] == corr]
                    colors = COLORS
                    data_fn().plot.bar(
                        y="median", yerr="std", ax=ax.flatten()[c], color=colors, legend=False, xlabel="Group", rot=10
                    )
                    ax.flatten()[c].set_title(corr)
                plt.tight_layout()
                sns.despine()
                plt.savefig(f"output/{exp_name}_{pdf}groupcorrbars median.pdf")
                plt.close()

        def grouphists():
            for pdf, data_fn in [
                ("quadratic", lambda: cdf[[col for col in cdf.columns if "concat" not in col]]),
                ("concat", lambda: cdf[[col for col in cdf.columns if "concat" in col]]),
                ("chroma", lambda: cdf[[col for col in cdf.columns if "chromagram" in col]]),
                ("onsets", lambda: cdf[[col for col in cdf.columns if "onsets" in col]]),
            ]:
                print("\n", pdf)
                fig, ax = plt.subplots(len(groups), len(corr_fns) - 1, figsize=(16, 9), sharex=True)
                for g, group in enumerate(file_groups):
                    gdf = df[df["group"] == group]
                    for c, corr in enumerate(corr_fns):
                        if corr == "svcca":
                            continue
                        cdf = gdf[[col for col in gdf.columns if corr in col]]
                        data = data_fn()
                        print(group, corr)
                        color = COLORS[g]
                        yvals, _, _ = ax[g, c].hist(
                            data.values.flatten(),
                            bins=100,
                            range=(stats["median"].min(), stats["median"].max()),
                            color=color,
                            density=True,
                        )
                        ax[g, c].vlines(np.median(data.values), 0, yvals.max(), ls="--", color=color)
                        ax[5, c].set_xlabel(
                            corr.replace("op", "Orthogonal Procrustes")
                            .replace("rv2", "Adjusted RV Coefficient")
                            .replace("smi", "Matrix Similarity Index")
                            .replace("pwcca", "Projection-weighted CCA")
                        )
                        ax[g, 0].set_ylabel(args.titles[list(file_groups.keys()).index(group)])
                        ax[g, c].set_yticklabels([])
                plt.tight_layout()
                sns.despine()
                plt.savefig(f"output/{exp_name}_{pdf}groupcorrhists median.pdf")
                plt.close()

        def hists():
            for pdf, data_fn in [
                ("Quadratic", lambda: cdf[[col for col in cdf.columns if "concat" not in col]]),
                ("Concatenated", lambda: cdf[[col for col in cdf.columns if "concat" in col]]),
            ]:

                print("\n", pdf)
                fig, ax = plt.subplots(1, len(corr_fns) - 1, figsize=(16, 4), sharex=True)
                gdf, g = df[df["group"] == "test"], -1
                for c, corr in enumerate(corr_fns):
                    if corr == "svcca":
                        continue
                    cdf = gdf[[col for col in gdf.columns if corr in col]]
                    data = data_fn()
                    print(group, corr)
                    color = COLORS[g]
                    yvals, _, _ = ax[c].hist(
                        data.values.flatten(),
                        bins=100,
                        range=(stats["median"].min(), stats["median"].max()),
                        color=color,
                        density=True,
                    )
                    ax[c].vlines(np.median(data.values), 0, yvals.max(), ls="--", color=color)
                    ax[c].set_xlabel(
                        corr.replace("op", "Orthogonal Procrustes")
                        .replace("rv2", "Adjusted RV Coefficient")
                        .replace("smi", "Matrix Similarity Index")
                        .replace("pwcca", "Projection-weighted CCA")
                    )
                    ax[0].set_ylabel(args.titles[list(file_groups.keys()).index(group)])
                    ax[c].set_yticklabels([])
                plt.suptitle(pdf)
                plt.tight_layout()
                sns.despine()
                plt.savefig(f"output/{exp_name}_{pdf.lower()}_hist_comparison.pdf")
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
                        data=groupstats["median"].values.reshape(A, V),
                        ax=g.ax_joint,
                        cbar=False,
                        cmap=hot,
                        vmin=0,
                        vmax=stats["median"].max(),
                    )

                    make_axes_locatable(g.ax_marg_x).append_axes("left", size="10%", pad="20%").axis("off")
                    cax = make_axes_locatable(g.ax_joint).append_axes("left", size="10%", pad="20%")
                    g.fig.colorbar(g.ax_joint.get_children()[1], cax=cax)
                    cax.yaxis.set_ticks_position("left")

                    np.set_printoptions(linewidth=200)
                    audio_feature_marginals = groupstats.groupby(["audio"], sort=False)["median"].median()
                    video_feature_marginals = groupstats.groupby(["video"], sort=False)["median"].median()
                    g.ax_marg_y.barh(
                        np.arange(0.5, A), audio_feature_marginals.values, color=hot(audio_feature_marginals.values)
                    )
                    g.ax_marg_y.set(xlim=(0, stats["median"].max()))
                    g.ax_marg_x.bar(
                        np.arange(0.5, V), video_feature_marginals.values, color=hot(video_feature_marginals.values)
                    )
                    g.ax_marg_x.set(ylim=(0, stats["median"].max()))

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
                    plt.savefig(f"output/{exp_name}_{corr}_{group} heatmap median.pdf")
                    plt.close()

        if not os.path.exists(csv_file.replace(".csv", "_melted.csv")):
            melted = []
            for col in tqdm(df.columns[1:]):
                af, vf, cn = col
                for idx, row in df.iterrows():
                    group_name = (
                        args.titles[list(file_groups.keys()).index(row.group)] if len(args.titles) > 0 else row.group
                    )
                    melted.append(
                        {
                            "correlation": cn.replace("op", "Orthogonal Procrustes")
                            .replace("rv2", "Adjusted RV Coefficient")
                            .replace("smi", "Matrix Similarity Index")
                            .replace("pwcca", "Projection-weighted CCA"),
                            "val": row[col],
                            "audio": af,
                            "video": vf,
                            "group": group_name,
                        }
                    )
            melted = pd.DataFrame(melted)
            melted.to_csv(csv_file.replace(".csv", "_melted.csv"))
        else:
            melted = pd.read_csv(csv_file.replace(".csv", "_melted.csv"), index_col="Unnamed: 0")
        melted = melted[melted["correlation"] != "svcca"]
        melted = melted[melted["val"] >= 0]

        def full_comparison_barbox():
            sns.set_theme(context="paper", style="white", palette="tab10")

            for pdf, mdf in [
                ("quadratic", melted[melted["audio"] != "concat"]),
                ("chroma", melted[melted["audio"] == "chromagram"]),
                ("onsets", melted[melted["audio"] == "onsets"]),
                ("drop_strength", melted[melted["audio"] == "drop_strength"]),
                ("concat", melted[melted["audio"] == "concat"]),
            ]:
                g = sns.FacetGrid(data=mdf, col="correlation", col_wrap=2, aspect=1.5, sharey=False)
                g.map_dataframe(sns.barplot, x="group", y="val", estimator=np.median, ci=99, palette=COLORS)
                g.set_xlabels("Interpolation Type")
                g.set_xticklabels(rotation=15, ha="right")
                g.set_ylabels("Audio-reactive Correlation")
                g.set_titles(template="{col_name}")
                plt.tight_layout()
                sns.despine()
                plt.savefig(f"output/{exp_name}_{pdf}groupcorrbarplotsimproved.pdf")
                plt.close()

            for pdf, mdf in [
                ("quadratic", melted[melted["audio"] != "concat"]),
                ("chroma", melted[melted["audio"] == "chromagram"]),
                ("onsets", melted[melted["audio"] == "onsets"]),
                ("drop_strength", melted[melted["audio"] == "drop_strength"]),
                ("concat", melted[melted["audio"] == "concat"]),
            ]:
                g = sns.FacetGrid(data=mdf, col="correlation", col_wrap=2, aspect=1.5, sharey=False)
                g.map_dataframe(sns.boxenplot, x="group", y="val", palette=COLORS)
                g.set_xlabels("Interpolation Type")
                g.set_xticklabels(rotation=15, ha="right")
                g.set_ylabels("Audio-reactive Correlation")
                g.set_titles(template="{col_name}")
                plt.tight_layout()
                sns.despine()
                plt.savefig(f"output/{exp_name}_{pdf}groupcorrboxplots.pdf")
                plt.close()

        def barbox():
            g = sns.FacetGrid(
                data=melted[
                    (melted["correlation"] == "Orthogonal Procrustes")
                    & (
                        (melted["audio"] == "chromagram")
                        | (melted["audio"] == "onsets")
                        # | (melted["audio"] == "drop_strength")
                        # | (melted["audio"] == "spectral_flatness")
                    )
                ],
                col="audio",
                sharey=False,
            )
            g.map_dataframe(sns.barplot, x="group", y="val", estimator=np.median, ci=99, palette=COLORS)
            g.set_xlabels("Interpolation Type")
            g.set_xticklabels(rotation=20, ha="right")
            g.set_ylabels("Audio-reactive Correlation")
            g.set_titles(template="{col_name}")
            plt.tight_layout()
            sns.despine()
            plt.savefig(f"output/{exp_name}_audiofeat_comparison.pdf")
            plt.close()

            g = sns.barplot(
                data=melted[(melted["correlation"] == "Orthogonal Procrustes") & ((melted["audio"] != "concat"))],
                x="group",
                y="val",
                estimator=np.median,
                ci=99,
                palette=COLORS,
            )
            g.set_xlabel("Interpolation Type")
            g.set_xticklabels(g.get_xticklabels(), rotation=15, ha="right")
            g.set_ylabel("Audio-reactive Correlation")
            plt.tight_layout()
            sns.despine()
            plt.savefig(f"output/{exp_name}_op_concat_comparison.pdf")
            plt.close()

        # grouphists()
        # groupbars()
        # heatmap()
        # grouphists()
        # full_comparison_barbox()
        barbox()
        hists()
