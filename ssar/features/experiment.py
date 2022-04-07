# fmt: off
from glob import glob

import decord as de
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from cycler import cycler
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from tqdm import tqdm

from .audio import (chromagram, drop_strength, mfcc, onsets, pulse, rms,
                    spectral_contrast, spectral_flatness, tonnetz)
from .correlation import rv2 as correlation
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
        return load_audio_video(self.files[idx])


if __name__ == "__main__":
    with torch.inference_mode():
        data_dir = "output/maua_correlation_test2/"

        low = sorted(glob(f"{data_dir}/low*"))
        medium = sorted(glob(f"{data_dir}/medium*"))
        test = sorted(glob(f"{data_dir}/test*"))
        hions = sorted(glob(f"{data_dir}/high_o*"))
        hichr = sorted(glob(f"{data_dir}/high_c*"))
        hibot = sorted(glob(f"{data_dir}/high_b*"))

        print("# low samples:", len(low))
        print("# medium samples:", len(medium))
        print("# test samples:", len(test))
        print("# high onset samples:", len(hions))
        print("# high chroma samples:", len(hichr))
        print("# high both samples:", len(hibot))
        print()

        print("# audio features:", len(afns))
        print("# video features:", len(vfns))
        print()

        results = []
        for group, files in [
            ("low", low),
            ("medium", medium),
            ("hions", hions),
            ("hichr", hichr),
            ("test", test),
            ("hibot", hibot),
        ]:
            print(group)
            for (audio,), sr, (video,), fps in tqdm(DataLoader(AudioVisualDataset(files), num_workers=24)):
                audio, sr, video, fps = audio.cuda(), sr.item(), video.cuda(), fps.item()

                afeats = [(af.__name__, af(audio, sr)) for af in afns]
                vfeats = [(vf.__name__, vf(video)) for vf in vfns]

                row = {"group": group}
                for a in range(len(afeats)):
                    for v in range(len(vfeats)):
                        aname, afeat = afeats[a]
                        vname, vfeat = vfeats[v]

                        row[f"{aname}_X_{vname}"] = correlation(afeat, vfeat).item()

                results.append(row)

        df = pd.DataFrame(results)
        print(df)

        df.to_csv("output/audiovisual_correlations2.csv")
        df = pd.read_csv("output/audiovisual_correlations2.csv", index_col=0)
        grouped = df.groupby("group").agg(["median", "std"])

        stats = []
        for col in grouped.columns[::2]:
            name, _ = col
            af, vf = name.split("_X_")
            means = grouped[(name, "median")]
            stds = grouped[(name, "std")]
            for group in ["low", "medium", "test", "hions", "hichr", "hibot"]:
                group_name = (
                    group.replace("low", "Constant speed interpolation (no correlation)")
                    .replace("medium", "Randomly generated 5-patches (medium correlation)")
                    .replace("test", "Test set, manually created patches (high correlation)")
                    .replace("hions", "Pure onset patch (high correlation)")
                    .replace("hichr", "Pure chroma patch (high correlation)")
                    .replace("hibot", "Onset & chroma patch (high correlation)")
                )
                stats.append(
                    {"audio": af, "video": vf, "group": group_name, "median": means[group], "std": stds[group]}
                )
        stats = pd.DataFrame(stats)
        print(stats)

        A = len(afns)
        V = len(vfns)
        hot = plt.get_cmap("hot")

        figs = []
        for group in stats["group"].unique():
            groupstats = stats[stats["group"] == group]

            g = sns.jointplot(data=groupstats, x="video", y="audio", kind="hist", bins=(V, A))
            g.ax_marg_y.cla()
            g.ax_marg_x.cla()
            sns.heatmap(
                data=groupstats["median"].values.reshape(V, A).T,
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

            audio_feature_marginals = groupstats.groupby(["audio"], sort=False)["median"].mean().values
            video_feature_marginals = groupstats.groupby(["video"], sort=False)["median"].mean().values
            g.ax_marg_y.barh(np.arange(0.5, A), audio_feature_marginals, color=hot(audio_feature_marginals))
            g.ax_marg_x.bar(np.arange(0.5, V), video_feature_marginals, color=hot(video_feature_marginals))

            g.ax_joint.set_xticks(np.arange(0.5, V))
            g.ax_joint.set_xticklabels(groupstats["video"].unique(), rotation=20, ha="right", rotation_mode="anchor")
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
            plt.savefig(f"output/{group} heatmap.pdf")
