# fmt: off
import os
from glob import glob
from math import ceil
from pathlib import Path

import decord as de
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision as tv
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from tqdm import tqdm

from .features.audio import (chromagram, drop_strength, mfcc, onsets, pulse,
                             rms, spectral_contrast, spectral_flatness,
                             tonnetz)
from .features.correlation import rv2 as correlation
from .features.video import (absdiff, adaptive_freq_rms, directogram,
                             high_freq_rms, hsv_hist, low_freq_rms,
                             mid_freq_rms, rgb_hist, video_flow_onsets,
                             video_spectral_onsets, video_spectrogram,
                             visual_variance)

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

afns = [chromagram, tonnetz, mfcc, spectral_contrast, spectral_flatness, rms, drop_strength, onsets, pulse]
vfns = [rgb_hist, hsv_hist, video_spectrogram, directogram, low_freq_rms, mid_freq_rms, high_freq_rms, 
        adaptive_freq_rms, absdiff, visual_variance, video_flow_onsets, video_spectral_onsets]

de.bridge.set_bridge("torch")
# fmt: on


@torch.inference_mode()
def rv2s_plot():
    DF = pd.read_csv("output/rv2s.csv")

    splits = ["train", "val", "test"]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
    fig, ax = plt.subplots(3, 3, figsize=(18, 12))
    for i, output in enumerate(["latent", "noise", "envelope"]):
        for j, split in enumerate(splits):
            for k, (sup, dec) in enumerate(
                [
                    ("oldsupervised", "learned"),
                    ("supervised", "learned"),
                    ("supervised", "fixed"),
                    ("selfsupervised", "fixed"),
                ]
            ):
                df = DF[(DF.supervision == sup) & (DF.decoder == dec)]
                x = df[f"iterations"]
                y = df[f"{split}_{output}_rv2"]
                err = df[f"{split}_{output}_rv2_std"]
                if (y < 0).any():
                    continue
                ax[i, j].plot(x, y, color=colors[k], label=fr"{sup} {dec} +/- $\sigma$")
                ax[i, j].fill_between(x, y - err, y + err, alpha=0.25, color=colors[k])
                ax[-1, j].set_xlabel("iterations")
                ax[i, 0].set_ylabel("rv2 loss")
                ax[i, j].set_title(f"{split} {output}")
                ax[i, j].legend()
                ax[i, j].set_xlim(0, 1_000_000)
                ylim = np.nanmax(
                    np.concatenate([DF[f"{s}_{output}_rv2"] + DF[f"{s}_{output}_rv2_std"] for s in splits])
                )
                ax[i, j].set_ylim(0, ylim * 1.1)
    plt.tight_layout()
    plt.savefig("output/rv2s_over_training.pdf")


@torch.inference_mode()
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
        return load_audio_video(self.files[idx])


@torch.inference_mode()
def heatmap(files, name, plot=False, marginals="median"):
    csv_file = f"output/audiovisual_correlations_{name}.csv"
    if not os.path.exists(csv_file):
        results = []

        for (audio,), sr, (video,), fps in tqdm(DataLoader(AudioVisualDataset(files), num_workers=24)):
            audio, sr, video, fps = audio.cuda(), sr.item(), video.cuda(), fps.item()

            afeats = [(af.__name__, af(audio, sr)) for af in afns]
            vfeats = [(vf.__name__, vf(video)) for vf in vfns]

            row = {"group": name}
            for a in range(len(afeats)):
                for v in range(len(vfeats)):
                    aname, afeat = afeats[a]
                    vname, vfeat = vfeats[v]

                    row[f"{aname}_X_{vname}"] = correlation(afeat, vfeat).item()

            results.append(row)

        df = pd.DataFrame(results)

        df.to_csv(csv_file)
    else:
        df = pd.read_csv(csv_file, index_col=0)

    grouped = df.groupby("group").agg(["median", "std"])
    stats = []
    for col in grouped.columns[::2]:
        col_name, _ = col
        af, vf = col_name.split("_X_")
        means = grouped[(col_name, "median")]
        stds = grouped[(col_name, "std")]
        stats.append({"audio": af, "video": vf, "group": name, "median": means[name], "std": stds[name]})
    stats = pd.DataFrame(stats)

    audio_feature_marginals = stats.groupby(["audio"], sort=False)["median"].mean()
    video_feature_marginals = stats.groupby(["video"], sort=False)["median"].mean()

    if plot:
        A = len(afns)
        V = len(vfns)
        hot = plt.get_cmap("hot")

        g = sns.jointplot(data=stats, x="video", y="audio", kind="hist", bins=(V, A))
        g.ax_marg_y.cla()
        g.ax_marg_x.cla()
        sns.heatmap(
            data=stats["median"].values.reshape(A, V),
            ax=g.ax_joint,
            cbar=False,
            cmap=hot,
            vmin=0,
            vmax=0.75,  # TODO better way than fixed value?
        )

        make_axes_locatable(g.ax_marg_x).append_axes("left", size="10%", pad="20%").axis("off")
        cax = make_axes_locatable(g.ax_joint).append_axes("left", size="10%", pad="20%")
        g.fig.colorbar(g.ax_joint.get_children()[1], cax=cax)
        cax.yaxis.set_ticks_position("left")

        np.set_printoptions(linewidth=200)
        g.ax_marg_y.barh(np.arange(0.5, A), audio_feature_marginals.values, color=hot(audio_feature_marginals.values))
        g.ax_marg_y.set(xlim=(0, stats["median"].max()))
        g.ax_marg_x.bar(np.arange(0.5, V), video_feature_marginals.values, color=hot(video_feature_marginals.values))
        g.ax_marg_x.set(ylim=(0, stats["median"].max()))

        g.ax_joint.set_xticks(np.arange(0.5, V))
        g.ax_joint.set_xticklabels(stats["video"].unique(), rotation=20, ha="right", rotation_mode="anchor")
        g.ax_joint.set_yticks(np.arange(0.5, A))
        g.ax_joint.set_yticklabels(stats["audio"].unique(), rotation=0)

        # remove ticks between heatmap and histograms
        g.ax_marg_x.tick_params(axis="x", bottom=False, labelbottom=False)
        g.ax_marg_y.tick_params(axis="y", left=False, labelleft=False)
        # # remove ticks showing the heights of the histograms
        g.ax_marg_x.tick_params(axis="y", left=False, labelleft=False)
        g.ax_marg_y.tick_params(axis="x", bottom=False, labelbottom=False)

        g.fig.suptitle(name)
        g.fig.set_size_inches(16, 9)
        plt.savefig(f"output/{name} heatmap.pdf")
        plt.close()

    return {**audio_feature_marginals.to_dict(), **video_feature_marginals.to_dict()}


def bar_plot(ax, data, xlabels=None, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), list(xlabels))
        ax.set_xticklabels(list(xlabels), rotation=40, ha="right")

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


@torch.inference_mode()
def marginals_bar_plot():
    marg = "max"
    cache_file = f"output/heatmap_marginals_{marg}.csv"
    if not os.path.exists(cache_file):
        rows = []
        for group in groups:
            for split in splits:
                all_files = glob(f"./output/test_vids/{group}*{split}*.mp4")

                grouped = {}
                for file in all_files:
                    steps = Path(file).stem.split("_")[1]
                    if steps in grouped:
                        grouped[steps].append(file)
                    else:
                        grouped[steps] = [file]

                for steps in grouped:
                    print(group, split, steps)
                    marginals = heatmap(grouped[steps], f"{group},{split},{steps}", marginals=marg)
                    rows.append({"group": group, "split": split, "steps": steps, **marginals})
        df = pd.DataFrame(rows)
        df.to_csv(cache_file)
    else:
        df = pd.read_csv(cache_file, index_col=0)

    grouped = df.groupby(["group", "split"]).agg(["max"])

    data = dict(zip([idx[0] + "," + idx[1] for idx in grouped.index], grouped[df.columns[3:]].values))
    data = {k + "," + kk: data[k + "," + kk] for k in groups for kk in splits}
    print(data)

    fig, ax = plt.subplots(figsize=(16, 9))
    bar_plot(
        ax,
        data,
        xlabels=df.columns[3:],
        colors=[
            "lightblue",
            "lightblue",
            "lightblue",
            "tab:blue",
            "tab:blue",
            "tab:blue",
            "lightgreen",
            "lightgreen",
            "lightgreen",
            "tab:green",
            "tab:green",
            "tab:green",
        ],
    )
    plt.savefig("output/heatmap_marginals_barplot.pdf")


@torch.inference_mode()
def autocorrelations(files, name):
    if not os.path.exists(f"output/autocorrelations/{name}_7.pdf"):
        for i, ((audio,), sr, (video,), fps) in enumerate(tqdm(DataLoader(AudioVisualDataset(files), num_workers=24))):
            audio, sr, video, fps = audio.cuda(), sr.item(), video.cuda(), fps.item()

            afeats = [(af.__name__, af(audio, sr)) for af in afns]
            vfeats = [(vf.__name__, vf(video)) for vf in vfns]

            aceil, vceil = ceil(len(afeats) / 3), ceil(len(vfeats) / 3)
            nrows = max(aceil, vceil)
            fig, ax = plt.subplots(nrows, 6, figsize=(18, nrows * 3))
            [x.axis("off") for x in ax.flatten()]
            for a in range(len(afeats)):
                aname, afeat = afeats[a]
                ax[a % aceil, a // aceil].imshow((afeat @ afeat.T).cpu().numpy(), cmap="inferno")
                ax[a % aceil, a // aceil].set_title(aname)
            for v in range(len(vfeats)):
                vname, vfeat = vfeats[v]
                ax[v % vceil, 3 + v // vceil].imshow((vfeat @ vfeat.T).cpu().numpy(), cmap="inferno")
                ax[v % vceil, 3 + v // vceil].set_title(vname)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle("<-- audio | video -->")
            plt.savefig(f"output/autocorrelations/{name}_{i}.pdf")
            plt.close()


def aggregated_autocorrelations():
    files = sum([glob(f"./output/test_vids/{groups[0]}*{split}*.mp4") for split in splits], [])
    fig, ax = plt.subplots(12, 6)
    [x.axis("off") for x in ax.flatten()]
    facs, iacs = [], []
    for i, ((audio,), sr, (_,), _) in enumerate(tqdm(DataLoader(AudioVisualDataset(files), num_workers=24))):
        audio, sr = audio.cuda(), sr.item()
        afeats = [afn(audio, sr) for afn in afns]

        fafeats = torch.cat(afeats, dim=1)
        fac = fafeats @ fafeats.T
        fac = fac - fac.min()
        fac = fac / fac.max()
        facs.append(fac)

        afeats = [af - af.min() for af in afeats]
        afeats = [af / af.max() for af in afeats]
        iac = torch.stack([af @ af.T for af in afeats])
        iac = torch.sum(iac, dim=0)
        iac = iac - iac.min()
        iac = iac / iac.max()
        iacs.append(iac)
    facs = torch.stack(facs)
    iacs = torch.stack(iacs)
    tv.utils.save_image(facs.unsqueeze(1), f"output/full_autocorrelation_audio_features.pdf", nrow=12)
    tv.utils.save_image(iacs.unsqueeze(1), f"output/neghadamard_autocorrelation_audio_features.pdf", nrow=12)


def test_autocorrelations():
    rows = []
    for group in groups:
        for split in splits:
            all_files = glob(f"./output/test_vids/{group}*{split}*.mp4")

            grouped = {}
            for file in all_files:
                steps = Path(file).stem.split("_")[1]
                if steps in grouped:
                    grouped[steps].append(file)
                else:
                    grouped[steps] = [file]

            for steps in grouped:
                autocorrelations(grouped[steps], f"{group},{split},{steps}")


def upper_triangle_feature_autocorrelation_sum(tensor):
    acs = torch.zeros((len(tensor), len(tensor)))
    i = 0
    # mfcc, chroma, tonnetz, contrast, singles
    for section in [20, 12, 6, 7, 10000]:
        acs += (tensor[:, i : i + section] @ tensor[:, i : i + section].T).cpu()
        i += section
    return acs


if __name__ == "__main__":
    groups = ["selfsupervised,fixed", "supervised,fixed", "supervised,learned", "oldsupervised,learned"]
    splits = ["train", "val", "test"]

    with torch.inference_mode():
        for file in sorted(glob("output/longform_test_vids/*.mp4")):

            with np.load(file.replace(".mp4", "_features.npz")) as f:
                features = torch.from_numpy(f[list(f.keys())[0]].squeeze()).cuda()

            with np.load(file.replace(".mp4", "_latnoise.npz")) as f:
                latents = torch.from_numpy(f["residuals"]).squeeze().cuda()
                try:
                    noise1 = torch.from_numpy(f["noise1"]).cuda()
                    noise2 = torch.from_numpy(f["noise2"]).cuda()
                    noise3 = torch.from_numpy(f["noise3"]).cuda()
                    noise4 = torch.from_numpy(f["noise4"]).cuda()
                except:
                    noise1 = torch.zeros_like(noise1).cuda()
                    noise2 = torch.zeros_like(noise2).cuda()
                    noise3 = torch.zeros_like(noise3).cuda()
                    noise4 = torch.zeros_like(noise4).cuda()

            feat_ac = features @ features.T
            lat_ac = latents.flatten(1) @ latents.flatten(1).T
            latabsdiff = absdiff(latents)
            latad_ac = latabsdiff @ latabsdiff.T
            noise1_ac = noise1.flatten(1) @ noise1.flatten(1).T
            noise2_ac = noise2.flatten(1) @ noise2.flatten(1).T
            noise3_ac = noise3.flatten(1) @ noise3.flatten(1).T
            noise4_ac = noise4.flatten(1) @ noise4.flatten(1).T

            fig, ax = plt.subplots(3, 4, figsize=(12, 9))
            [x.axis("off") for x in ax.flatten()]
            ax[0, 0].imshow(feat_ac.cpu().numpy())
            ax[0, 3].bar(
                ["latents", "latabsdiff", "noise1", "noise2", "noise3", "noise4"],
                [
                    correlation(features[: len(latents)], latents.flatten(1)).cpu().numpy(),
                    correlation(features[: len(latabsdiff)], latabsdiff).cpu().numpy(),
                    correlation(features[: len(noise1)], noise1.flatten(1)).cpu().numpy(),
                    correlation(features[: len(noise2)], noise2.flatten(1)).cpu().numpy(),
                    correlation(features[: len(noise3)], noise3.flatten(1)).cpu().numpy(),
                    correlation(features[: len(noise4)], noise4.flatten(1)).cpu().numpy(),
                ],
            )
            ax[0, 3].axis("on")
            ax[1, 0].imshow(lat_ac.cpu().numpy())
            ax[1, 1].imshow(latad_ac.cpu().numpy())
            ax[2, 0].imshow(noise1_ac.cpu().numpy())
            ax[2, 1].imshow(noise2_ac.cpu().numpy())
            ax[2, 2].imshow(noise3_ac.cpu().numpy())
            ax[2, 3].imshow(noise4_ac.cpu().numpy())
            plt.tight_layout()
            plt.savefig(f"output/longform_test_vids/{Path(file).stem}.pdf")
            plt.close()
