import os
from glob import glob
from pathlib import Path

import numpy as np
import pytorchvideo
import pytorchvideo.data
import torch
import torchaudio
from npy_append_array import NpyAppendArray
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Lambda
from tqdm import tqdm

from models.slowfast import SlowFastExtractor
from models.vggish import VggishExtractor
from SGW.lib.sgw_pytorch import sgw_gpu as sliced_gromov_wasserstein

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def preprocess_video(in_dir, out_file, duration, fps):
    if not os.path.exists(out_file + "_0.npy"):

        videos = sum([glob(in_dir + "/*" + ext) for ext in [".mp4", ".avi", ".mkv"]], [])
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
seq_len = duration * fps

benny_dir = "/home/hans/datasets/audiovisual/trashbenny"
benny_cache = f"cache/{Path(benny_dir).stem}_features_{duration}sec_{fps}fps"
preprocess_video(benny_dir, benny_cache, duration, fps)

maua_dir = "/home/hans/datasets/audiovisual/maua-short"
maua_cache = f"cache/{Path(maua_dir).stem}_features_{duration}sec_{fps}fps"
preprocess_video(maua_dir, maua_cache, duration, fps)

benny_features = AudioVisualFeatures(benny_cache)
maua_features = AudioVisualFeatures(maua_cache)

benny_sgw = 0
for bvfs, bafs in DataLoader(benny_features, shuffle=True):
    for bvf in bvfs:
        for baf in bafs:
            benny_sgw += sliced_gromov_wasserstein(
                bvf.squeeze().to(device), baf.squeeze().to(device), device, nproj=500
            ).mean()
print("trashbenny", (benny_sgw / len(benny_features)).item())

maua_sgw = 0
for mvfs, mafs in DataLoader(maua_features, shuffle=True):
    for mvf in mvfs:
        for maf in mafs:
            maua_sgw += sliced_gromov_wasserstein(
                mvf.squeeze().to(device), maf.squeeze().to(device), device, nproj=500
            ).mean()
print("maua", (maua_sgw / len(maua_features)).item())
