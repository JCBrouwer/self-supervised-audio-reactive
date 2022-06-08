import sys
from glob import glob

import decord
import joblib
import numpy as np
import torch
from resize_right import resize
from torch.utils.data import DataLoader, Dataset

from .comparison import DEVICE, SG2, load_video
from .train import STYLEGAN_CKPT

sys.path.append("/home/hans/code/maua/")
from maua.GAN.metrics.compute import compute
from maua.GAN.wrappers.stylegan2 import StyleGAN2


class VideoFrameDataset(Dataset):
    def __init__(self, mp4s) -> None:
        super().__init__()
        videos = []
        lengths = []
        for mp4 in mp4s:
            video = decord.VideoReader(mp4)
            videos.append(video)
            lengths.append(len(video))
        self.videos = videos
        self.lengths = np.cumsum(lengths)
        self.order = np.random.permutation(self.lengths[-1])

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, idx):
        oidx = self.order[idx]
        vidx = np.argmax(self.lengths > oidx)
        voidx = oidx - (self.lengths[vidx] - len(self.videos[vidx]))
        return resize(self.videos[vidx][voidx], out_shape=(224, 224)).squeeze()


class VideoFrameDataset(Dataset):
    def __init__(self, mp4s, downsample=1) -> None:
        super().__init__()
        videos = []
        for mp4 in mp4s:
            video, _ = load_video(mp4, downsample=downsample)
            print(video.min(), video.mean(), video.max())
            videos.append(video)
        self.videos = torch.cat(videos)
        self.order = np.random.permutation(self.videos.shape[0])

    def __len__(self):
        return self.videos.shape[0]

    def __getitem__(self, idx):
        return resize(self.videos[self.order[idx]], out_shape=(224, 224)).squeeze().mul(2).sub(1)


class RandomGeneratedImages(Dataset):
    def __init__(self, G, n_samples, batch_size) -> None:
        super().__init__()
        self.G = G.to(DEVICE)
        self.n_samples = n_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.n_samples // self.batch_size

    def __getitem__(self, idx):
        return resize(self.G(torch.randn((self.batch_size, 512), device=DEVICE)), out_shape=(224, 224)).add(1).div(2)


def train_set_ood():
    global SG2
    del SG2
    model_file = "/home/hans/modelzoo/train_checks/neurout2-006.pt"
    real_samples = DataLoader(
        RandomGeneratedImages(
            StyleGAN2(model_file=model_file, inference=False, output_size=(1024, 1024), strategy="stretch", layer=0)
            .eval()
            .to(DEVICE),
            N,
            B,
        ),
        batch_size=1,
    )
    real_samples.path = model_file

    fake_samples = DataLoader(
        VideoFrameDataset(glob(f"/home/hans/datasets/audiovisual/maua256/*Cancelled*.mp4")), batch_size=B
    )
    fake_iter = iter(fake_samples)
    results = [
        {
            "name": "Train Set",
            **compute(
                real_samples,
                lambda: next(fake_iter),
                n_samples=N,
                extractor="SwAV",
                metrics=["frechet", "kernel", "prdc"],
                batch_size=B,
                device=DEVICE,
                ignore_cache=True,
            ),
        }
    ]
    joblib.dump(results, "output/oodtrainset.pkl")
    print(results[-1])
    exit()


def lucidsonicdreams_ood():
    global SG2
    del SG2
    model_file = "cache/modern art.pkl"
    real_samples = DataLoader(
        RandomGeneratedImages(
            StyleGAN2(model_file=model_file, inference=False, output_size=(512, 512), strategy="stretch", layer=0)
            .eval()
            .to(DEVICE),
            N,
            B,
        ),
        batch_size=1,
    )
    real_samples.path = model_file

    fake_samples = DataLoader(
        VideoFrameDataset(glob(f"/home/hans/datasets/audiovisual/lucid/*.mp4"), downsample=4), batch_size=B
    )
    fake_iter = iter(fake_samples)
    results = [
        {
            "name": "Baseline (LSD)",
            **compute(
                real_samples,
                lambda: next(fake_iter),
                n_samples=N,
                extractor="SwAV",
                metrics=["frechet", "kernel", "prdc"],
                batch_size=B,
                device=DEVICE,
                ignore_cache=True,
            ),
        }
    ]
    joblib.dump(results, "output/oodlucid.pkl")
    print(results[-1])
    exit()


if __name__ == "__main__":
    N = 30000
    B = 1

    # train_set_ood()
    lucidsonicdreams_ood()

    directory = "/home/hans/datasets/audiovisual/density_eval/"
    keys = [
        "random",
        "ssopt",
        "_supervised_learned_decoder_residual:False",
        "_supervised_learned_decoder_residual:True",
        "_supervised_fixed_decoder_residual:False",
        "_supervised_fixed_decoder_residual:True",
        "selfsupervised_learned_decoder_residual:False",
        "selfsupervised_learned_decoder_residual:True",
        "selfsupervised_fixed_decoder_residual:False",
        "selfsupervised_fixed_decoder_residual:True",
    ]

    real_samples = DataLoader(RandomGeneratedImages(SG2, N, B), batch_size=1)
    real_samples.path = STYLEGAN_CKPT

    results = []
    for key in keys:
        fake_samples = DataLoader(VideoFrameDataset(glob(f"{directory}/*{key}*.mp4")), batch_size=B)
        fake_iter = iter(fake_samples)
        results.append(
            {
                "name": key,
                **compute(
                    real_samples,
                    lambda: next(fake_iter),
                    n_samples=N,
                    extractor="SwAV",
                    metrics=["frechet", "kernel", "prdc"],
                    batch_size=B,
                    device="cpu",
                    ignore_cache=True,
                ),
            }
        )
        joblib.dump(results, "output/ood.pkl")
        print(results[-1])
        del fake_samples, fake_iter
