import os
from copy import deepcopy
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytorchvideo
import pytorchvideo.data
import torch
import torchaudio
import torchvision as tv
from npy_append_array import NpyAppendArray
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Lambda
from tqdm import tqdm

from models.slowfast import SlowFastExtractor
from models.vggish import VggishExtractor
from SGW.lib.sgw_pytorch import sgw_gpu as sliced_gromov_wasserstein

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transpose(l):
    return list(map(list, zip(*l)))


@torch.inference_mode()
def preprocess_video(in_dir, out_file, duration, fps):
    if not os.path.exists(out_file + "_0.npy"):

        videos = sum([glob(in_dir + "/*" + ext) for ext in [".mp4", ".avi", ".mkv", ".webm"]], [])
        dataset = pytorchvideo.data.LabeledVideoDataset(
            list(zip(videos, [{} for _ in range(len(videos))])),
            pytorchvideo.data.UniformClipSampler(clip_duration=duration),
            transform=ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(duration * fps),
                        ShortSideScale(size=256),
                        CenterCrop(256),
                        Lambda(lambda x: x / 255.0),
                    ]
                ),
            ),
            decode_audio=True,
        )

        slowfast = SlowFastExtractor()  # TODO input normalization correct?
        vggish = VggishExtractor()

        npaas = [NpyAppendArray(f"{out_file}_{i}.npy") for i in range(slowfast.num_layers + vggish.num_layers)]
        for batch in tqdm(
            DataLoader(dataset, batch_size=1, num_workers=min(len(videos), 16)),
            desc="Encoding videos to audio/visual features...",
        ):
            video = batch["video"].permute(0, 2, 1, 3, 4)
            video_features = slowfast(video)

            audio = batch["audio"]
            audio = torchaudio.transforms.Resample(round(audio.shape[1] / duration), 16_000)(audio)
            audio_features = vggish(audio)

            for f, feat in enumerate(video_features + audio_features):
                npaas[f].append(np.ascontiguousarray(feat.numpy()))
        del npaas


class AudioVisualFeatures(Dataset):
    def __init__(self, base):
        super().__init__()
        self.feature_files = [np.load(file, mmap_mode="r") for file in sorted(glob(base + "_*.npy"))]

    def __len__(self):
        return len(self.feature_files[0])

    def __getitem__(self, index):
        features = [torch.from_numpy(ff[index].copy()) for ff in self.feature_files]
        video_features = features[: len(features) // 2]
        audio_features = features[len(features) // 2 :]
        return video_features, audio_features


duration = 4
fps = 24


benny_dir = "/home/hans/datasets/audiovisual/trashbenny"
benny_cache = f"cache/{Path(benny_dir).stem}_features_{duration}sec_{fps}fps"
preprocess_video(benny_dir, benny_cache, duration, fps)

maua_dir = "/home/hans/datasets/audiovisual/maua-short"
maua_cache = f"cache/{Path(maua_dir).stem}_features_{duration}sec_{fps}fps"
preprocess_video(maua_dir, maua_cache, duration, fps)

benny_features = AudioVisualFeatures(benny_cache)
maua_features = AudioVisualFeatures(maua_cache)

for name, features in [["trashbenny", benny_features], ["maua", maua_features]]:
    all_vfs, all_afs = [], []
    for vfs, afs in DataLoader(features, shuffle=True):
        all_vfs.append(vfs)
        all_afs.append(afs)

    all_vfs = [torch.cat(vfs).flatten(1) for vfs in transpose(all_vfs)]
    all_afs = [torch.cat(afs).flatten(1) for afs in transpose(all_afs)]

    sgw = 0
    for vf in all_vfs:
        for af in all_afs:
            sgw += sliced_gromov_wasserstein(af.to(device), vf.to(device), device)

    print(name, sgw.item())

# in_dir = "/home/hans/datasets/audiovisual/256/"
# videos = sum([glob(in_dir + "/*" + ext) for ext in [".mp4", ".avi", ".mkv"]], [])
# dataset = pytorchvideo.data.LabeledVideoDataset(
#     list(zip(videos, [{} for _ in range(len(videos))])),
#     pytorchvideo.data.UniformClipSampler(clip_duration=duration),
#     transform=ApplyTransformToKey(
#         key="video",
#         transform=Compose(
#             [
#                 UniformTemporalSubsample(duration * fps),
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
#         audio = torchaudio.transforms.Resample(round(audio.shape[1] / duration), 16_000)(audio)
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
