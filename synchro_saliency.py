import os
import warnings
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time
from typing import List

warnings.catch_warnings()
warnings.simplefilter(action="ignore", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorchvideo
import pytorchvideo.data
import torch
import torchaudio
import torchvision as tv
from npy_append_array import NpyAppendArray
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torch.nn.functional import interpolate
from torch.utils.data import ChainDataset, DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Lambda
from torchvision.transforms.functional import resize
from tqdm import tqdm

from audio_features import chroma, fourier_tempogram, hpcp, mfcc, onsets, rms, tempogram, tonnetz
from models.slowfast import SlowFastExtractor
from models.vggish import VggishExtractor
from sgw import sgw_gpu as sliced_gromov_wasserstein
from visual_beats import normalize, video_onsets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transpose(l):
    return list(map(list, zip(*l)))


@torch.inference_mode()
def preprocess_video(in_dir, out_stem, dur, fps, max_samples=750):
    if not os.path.exists(out_stem + "_files.npy"):

        videos = sum([glob(in_dir + "/*" + ext) for ext in [".mp4", ".avi", ".mkv", ".webm"]], [])
        dataset = ChainDataset(
            pytorchvideo.data.LabeledVideoDataset(
                list(zip(videos, [{} for _ in range(len(videos))])),
                sampler,
                transform=ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(dur * fps),
                            ShortSideScale(size=256),
                            CenterCrop(256),
                            Lambda(lambda x: x / 255.0),
                        ]
                    ),
                ),
                decode_audio=True,
            )
            for sampler in [
                pytorchvideo.data.UniformClipSampler(clip_duration=dur),
                *([pytorchvideo.data.RandomClipSampler(clip_duration=dur)] * 25),
            ]
        )

        slowfast = SlowFastExtractor()  # TODO input normalization correct?
        vggish = VggishExtractor()

        n = 0
        feature_cache_files = None
        for batch in tqdm(
            DataLoader(dataset, batch_size=1, num_workers=min(len(videos), 16)),
            desc="Encoding videos to audio/visual features...",
        ):
            if n > max_samples:
                break

            video = batch["video"].permute(0, 2, 1, 3, 4)
            video_feats = {f"slowfast_layer{f}": feat for f, feat in enumerate(slowfast(video))}
            video_feats["onsets"] = video_onsets(video.squeeze().permute(0, 2, 3, 1))
            video_feats["tempogram"] = tempogram(onsets=video_feats["onsets"])
            video_feats["fourier_tempogram"] = fourier_tempogram(onsets=video_feats["onsets"])[:-1]

            audio = batch["audio"]
            audio = torchaudio.transforms.Resample(round(audio.shape[1] / dur), 16_000)(audio)
            audio_feats = {f"vggish_layer{f}": feat for f, feat in enumerate(vggish(audio))}
            audio = audio.mean(0).numpy()
            audio_feats["onsets"] = interpolate(onsets(audio)[None, None, :, 0], (dur * fps,))
            audio_feats["tempogram"] = resize(tempogram(audio)[None, None], (dur * fps, 384), antialias=True)
            audio_feats["fourier_tempogram"] = resize(
                fourier_tempogram(audio)[None, None], (dur * fps, 193), antialias=True
            )
            audio_feats["chroma"] = resize(chroma(audio)[None, None], (dur * fps, 12), antialias=True)
            audio_feats["hpcp"] = resize(hpcp(audio)[None, None], (dur * fps, 12), antialias=True)
            audio_feats["tonnetz"] = resize(tonnetz(audio)[None, None], (dur * fps, 6), antialias=True)
            audio_feats["mfcc"] = resize(mfcc(audio)[None, None], (dur * fps, 20), antialias=True)
            audio_feats["rms"] = interpolate(rms(audio)[None, None, :, 0], (dur * fps,))

            if feature_cache_files is None:
                feature_cache_files = {
                    **{f"video_{k}": NpyAppendArray(f"{out_stem}_video_{k}.npy") for k in video_feats.keys()},
                    **{f"audio_{k}": NpyAppendArray(f"{out_stem}_audio_{k}.npy") for k in audio_feats.keys()},
                    **{"files": NpyAppendArray(f"{out_stem}_files.npy")},
                }
            for k, feat in video_feats.items():
                feature_cache_files[f"video_{k}"].append(np.ascontiguousarray(feat.squeeze().unsqueeze(0).numpy()))
            for k, feat in audio_feats.items():
                feature_cache_files[f"audio_{k}"].append(np.ascontiguousarray(feat.squeeze().unsqueeze(0).numpy()))
            feature_cache_files["files"].append(np.ascontiguousarray([batch["video_name"][0].ljust(50)[:50]]))

            n += 1
        del feature_cache_files


class AudioVisualFeatures(Dataset):
    def __init__(self, base):
        super().__init__()
        self.video_files = {
            Path(f.replace(base + "_", "")).stem: np.load(f, mmap_mode="r") for f in sorted(glob(base + "_video*.npy"))
        }
        self.audio_files = {
            Path(f.replace(base + "_", "")).stem: np.load(f, mmap_mode="r") for f in sorted(glob(base + "_audio*.npy"))
        }
        self.filenames = np.load(base + "_files.npy", mmap_mode="r")

    def __len__(self):
        return len(list(self.video_files.values())[0])

    def __getitem__(self, index):
        video_features = {k: torch.from_numpy(ff[index].copy()) for k, ff in self.video_files.items()}
        audio_features = {k: torch.from_numpy(ff[index].copy()) for k, ff in self.audio_files.items()}
        return video_features, audio_features, self.filenames[index]


if __name__ == "__main__":
    dur = 4
    fps = 24

    benny_dir = "/home/hans/datasets/audiovisual/trashbenny"
    benny_cache = f"cache/{Path(benny_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(benny_dir, benny_cache, dur, fps)

    maua_dir = "/home/hans/datasets/audiovisual/maua"
    maua_cache = f"cache/{Path(maua_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(maua_dir, maua_cache, dur, fps)

    phony_dir = "/home/hans/datasets/audiovisual/phony"
    phony_cache = f"cache/{Path(phony_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(phony_dir, phony_cache, dur, fps)

    trapnation_dir = "/home/hans/datasets/audiovisual/trapnation"
    trapnation_cache = f"cache/{Path(trapnation_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(trapnation_dir, trapnation_cache, dur, fps)

    invocation_dir = "/home/hans/datasets/audiovisual/invocation"
    invocation_cache = f"cache/{Path(invocation_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(invocation_dir, invocation_cache, dur, fps)

    @torch.jit.script
    def calc_sgws(
        all_vfs: List[torch.Tensor],
        all_afs: List[torch.Tensor],
        device: torch.device,
        n_proj: int = 512,
        return_individual: bool = False,
    ):
        total = torch.zeros(1, device=device)
        overalls = torch.jit.annotate(List[torch.Tensor], [])
        sums = torch.jit.annotate(List[torch.Tensor], [])
        individuals = torch.jit.annotate(List[List[torch.Tensor]], [[] for _ in range(len(all_vfs[0]))])
        for vf in all_vfs:
            for af in all_afs:
                P = torch.randn([max([vf.shape[1], af.shape[1]]), n_proj], device=device)
                overall = sliced_gromov_wasserstein(af.to(device), vf.to(device), device=device, P=P)
                if return_individual:
                    sum = torch.zeros(1, device=device)
                    for i, (a, v) in enumerate(zip(af, vf)):
                        individual = torch.abs(
                            sliced_gromov_wasserstein(a[None].to(device), v[None].to(device), device=device, P=P)
                        )
                        sum += individual
                        individuals[i].append(individual)
                    sums.append(sum)
                overalls.append(overall)
                total += overall
        return total, overalls, sums, individuals

    j = 0
    data, group_data = [], []
    t = time()
    for name, features in [
        ["trashbenny", AudioVisualFeatures(benny_cache)],
        ["maua", AudioVisualFeatures(maua_cache)],
        ["phony", AudioVisualFeatures(phony_cache)],
        ["trapnation", AudioVisualFeatures(trapnation_cache)],
    ]:

        all_vfs, all_afs, all_fns = [], [], []
        for vfs, afs, (file,) in DataLoader(features, shuffle=True):
            all_vfs.append(vfs.values())
            all_afs.append(afs.values())
            all_fns.append(file)

        all_vfs = [normalize(torch.cat(vfs).flatten(1)) for vfs in transpose(all_vfs)]
        all_afs = [normalize(torch.cat(afs).flatten(1)) for afs in transpose(all_afs)]
        vnames = vfs.keys()
        anames = afs.keys()

        total, overalls, sums, individuals = calc_sgws(all_vfs, all_afs, device=device, return_individual=True)

        print(name, total.item())

        group_data.append({"group": name})
        for file in all_fns:
            data.append({"group": name, "file": file.strip()})

        i = 0
        for vn in vnames:
            for an in anames:
                group_data[-1][f"{vn}_{an}"] = overalls[i].item()

                for k, id in enumerate(individuals):
                    data[j + k][f"{vn}_{an}"] = id[i].item()
                i += 1
        j = len(data)

    print(time() - t)

    data = pd.DataFrame(data)
    print(data)
    data.to_csv("sgws.csv")

    group_data = pd.DataFrame(group_data)
    print(group_data)
    group_data.to_csv("group_sgws.csv")

    #

    #

    #

    #

    #

    # in_dir = "/home/hans/datasets/audiovisual/256/"
    # videos = sum([glob(in_dir + "/*" + ext) for ext in [".mp4", ".avi", ".mkv"]], [])
    # dataset = pytorchvideo.data.LabeledVideoDataset(
    #     list(zip(videos, [{} for _ in range(len(videos))])),
    #     pytorchvideo.data.UniformClipSampler(clip_duration=dur),
    #     transform=ApplyTransformToKey(
    #         key="video",
    #         transform=Compose(
    #             [
    #                 UniformTemporalSubsample(dur * fps),
    #                 ShortSideScale(size=256),
    #                 CenterCrop(256),
    #                 Lambda(lambda x: x / 255.0),
    #             ]
    #         ),
    #     ),
    #     decode_audio=True,
    # )

    # slowfast = SlowFastExtractor()  # TODO input normalization correct?
    # vggish = VggishExtractor()

    # num = 400
    # i = 0
    # sgws, vids, auds, names = [], [], [], []
    # with torch.inference_mode():
    #     for shared_batch in tqdm(
    #         DataLoader(dataset, batch_size=1, num_workers=min(len(videos), 16)),
    #         desc="Encoding videos to audio/visual features...",
    #         total=num,
    #     ):
    #         batch = deepcopy(shared_batch)
    #         del shared_batch

    #         video = batch["video"].permute(0, 2, 1, 3, 4)
    #         video_features = slowfast(video)

    #         audio = batch["audio"]
    #         audio = torchaudio.transforms.Resample(round(audio.shape[1] / dur), 16_000)(audio)
    #         audio_features = vggish(audio)

    #         sgw = 0
    #         for vf in video_features:
    #             for af in audio_features:
    #                 sgw += torch.median(
    #                     sliced_gromov_wasserstein(vf.squeeze().to(device), af.squeeze().to(device), device, nproj=500)
    #                 )

    #         names.append(batch["video_name"][0].replace(" - ", "_").replace(" ", "_").lower().split(".")[0][:50])
    #         vids.append(video.cpu())
    #         auds.append(audio.cpu())
    #         sgws.append(sgw.cpu().item())

    #         i += 1
    #         if i > num:
    #             break

    # sgws = np.array(sgws)

    # q1, q3 = np.percentile(sgws, 25), np.percentile(sgws, 75)
    # iqr = q3 - q1
    # print(np.min(sgws), q1, np.median(sgws), np.mean(sgws), q3, np.max(sgws))
    # print("outliers:", np.sort(sgws[(q1 - 1.5 * iqr > sgws) | (sgws > q3 + 1.5 * iqr)]))
    # plt.hist(sgws[sgws < 10], bins=100)
    # plt.savefig("output/sgw_hist.pdf")

    # order = np.argsort(sgws)

    # lower = len(sgws) // 4
    # half = len(sgws) // 2
    # upper = 3 * len(sgws) // 4
    # five = np.arange(5)
    # for idx in [*five, *(lower + five), *(half + five), *(upper + five), *np.flip(-five)]:
    #     tv.io.write_video(
    #         f"output/{sgws[order[idx]]:.4f}_{names[order[idx]]}.mp4",
    #         video_array=vids[order[idx]].squeeze().permute(0, 2, 3, 1).mul(255).int(),
    #         fps=24,
    #         video_codec="h264",
    #         audio_array=auds[order[idx]],
    #         audio_fps=16000,
    #         audio_codec="aac",
    #     )
