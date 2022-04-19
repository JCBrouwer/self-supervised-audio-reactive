# fmt:off

import io
import os
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio
from ffcv.fields import BytesField, IntField, NDArrayField
from ffcv.fields.decoders import BytesDecoder, IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from npy_append_array import NpyAppendArray as NumpyArray
from ssar.features.audio import (chromagram, drop_strength, harmonic, mfcc,
                                 onset_strength, percussive, plp, rms,
                                 spectral_contrast, spectral_flatness, tonnetz)
from ssar.features.processing import (clamp_lower_percentile,
                                      clamp_peaks_percentile, emphasize,
                                      gaussian_filter, high_pass, low_pass,
                                      mid_pass)
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from tqdm import tqdm

# fmt:on


_FN = [
    *[f"mfcc_{i}" for i in range(20)],
    *[f"chroma_{i}" for i in range(12)],
    *[f"tonnetz_{i}" for i in range(6)],
    *[f"contrast_{i}" for i in range(7)],
    "flatness",
    "onsets",
    "onsets_low",
    "onsets_mid",
    "onsets_high",
    "pulse",
    "harmonic_rms",
    "harmonic_rms_low",
    "harmonic_rms_mid",
    "harmonic_rms_high",
    "long_rms",
    "long_rms_low",
    "long_rms_mid",
    "long_rms_high",
]
VELOCITY = False
if VELOCITY:
    FEATURE_NAMES = _FN + [n + "_velocity" for n in _FN]
else:
    FEATURE_NAMES = _FN


@torch.inference_mode()
def audio2features(audio, sr, fps, clamp=True, smooth=True, emphasis=True):
    if audio.dim() == 2:
        audio = audio.mean(0)
    audio, sr = resample(audio, sr, fps * 1024), fps * 1024

    audio_harm, audio_perc = harmonic(audio), percussive(audio)
    multi_features = [
        gaussian_filter(mfcc(audio, sr), 0.5 * fps),
        gaussian_filter(chromagram(audio_harm, sr), 0.5 * fps),
        gaussian_filter(tonnetz(audio_harm, sr), 0.5 * fps),
        gaussian_filter(spectral_contrast(audio, sr), 0.5 * fps),
    ]
    single_features = [
        gaussian_filter(spectral_flatness(audio, sr), 0.5 * fps),
        onset_strength(audio_perc, sr),
        onset_strength(low_pass(audio_perc, sr), sr),
        onset_strength(mid_pass(audio_perc, sr), sr),
        onset_strength(high_pass(audio_perc, sr), sr),
        plp(audio_perc, sr),
        rms(audio_harm, sr),
        rms(low_pass(audio_harm, sr), sr),
        rms(mid_pass(audio_harm, sr), sr),
        rms(high_pass(audio_harm, sr), sr),
        drop_strength(audio, sr),
        drop_strength(low_pass(audio, sr), sr),
        drop_strength(mid_pass(audio, sr), sr),
        drop_strength(high_pass(audio, sr), sr),
    ]
    features = multi_features + [sf.reshape(-1, 1) for sf in single_features]
    features = torch.cat(features, dim=1)

    if VELOCITY:
        V = torch.diff(gaussian_filter(features, fps), dim=0)
        V = torch.cat((V[[0]], V), dim=0)
        features = torch.cat((features, V), dim=1)

    if clamp:
        P = 2.5  # how hard to clamp
        features = clamp_peaks_percentile(features, 100 - P)
        features = clamp_lower_percentile(features, 4 * P)

    if smooth:
        features = gaussian_filter(features, 0.1 * fps)

    if emphasis:
        features = emphasize(features, strength=2, percentile=75)

    return features


class AudioFeatures(Dataset):
    def __init__(self, directory, a2f, dur, fps):
        super().__init__()
        self.files = sum([glob(f"{directory}*.{ext}") for ext in ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]], [])
        self.a2f = a2f
        self.L = dur * fps
        self.fps = fps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        audio, sr = torchaudio.load(file)
        features = self.a2f(audio, sr, self.fps)
        features = torch.stack(  # overlapping slices of DUR seconds
            sum([list(torch.split(features[start:], self.L)[:-1]) for start in range(0, self.L, self.L // 4)], [])
        )
        return features.cpu()


def to_bytes(x: torch.Tensor) -> np.ndarray:
    bytes = io.BytesIO()
    np.save(bytes, x)
    bytes.seek(0)
    return np.frombuffer(bytes.read(), dtype=np.uint8)


def from_bytes(byte_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.load(io.BytesIO(byte_array.tobytes())))


def load_npy(file):
    try:
        return joblib.load(file).float()
    except:
        return torch.from_numpy(np.load(file)).float()


class FullAudio2LatentFFCVPreprocessor(Dataset):
    def __init__(self, directory, fps, return_bytes=False):
        super().__init__()
        self.fps = fps
        self.return_bytes = return_bytes
        self.files = sum([glob(f"{directory}*.{ext}") for ext in ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]], [])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        filename, _ = os.path.splitext(file)

        audio, sr = torchaudio.load(file)
        features = audio2features(audio, sr, self.fps)

        latents = load_npy(f"{filename}.npy")

        noise4 = load_npy(f"{filename} - Noise 4.npy").float()
        noise8 = load_npy(f"{filename} - Noise 8.npy").float()
        noise16 = load_npy(f"{filename} - Noise 16.npy").float()
        noise32 = load_npy(f"{filename} - Noise 32.npy").float()

        if self.return_bytes:
            return (
                to_bytes(np.array([file])),
                to_bytes(audio),
                sr,
                to_bytes(features),
                to_bytes(latents),
                to_bytes(noise4),
                to_bytes(noise8),
                to_bytes(noise16),
                to_bytes(noise32),
            )
        else:
            return file, audio, sr, features, latents, noise4, noise8, noise16, noise32


class SlicedAudio2LatentFFCVPreprocessor(Dataset):
    def __init__(self, datastem, split="train"):
        super().__init__()
        self.features = np.load(f"{datastem}_{split}_feats.npy", mmap_mode="r")
        self.latents = np.load(f"{datastem}_{split}_lats.npy", mmap_mode="r")
        self.noise4 = np.load(f"{datastem}_{split}_noise4.npy", mmap_mode="r")
        self.noise8 = np.load(f"{datastem}_{split}_noise8.npy", mmap_mode="r")
        self.noise16 = np.load(f"{datastem}_{split}_noise16.npy", mmap_mode="r")
        self.noise32 = np.load(f"{datastem}_{split}_noise32.npy", mmap_mode="r")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        latents = self.latents[index].copy()
        residuals = latents - latents.mean((0, 1))
        return (
            self.features[index].copy().astype(np.float32),
            residuals.astype(np.float32),
            self.noise4[index].copy().astype(np.float32),
            self.noise8[index].copy().astype(np.float32),
            self.noise16[index].copy().astype(np.float32),
            self.noise32[index].copy().astype(np.float32),
        )


def get_full_length_ffcv_dataloaders(input_dir, dur, fps):
    L = dur * fps
    datastem = f"cache/{Path(input_dir).stem}_{L}frames"
    full_beton = f"{datastem}_ffcv_full.beton"
    if not os.path.exists(full_beton):
        print("Preprocessing full length data to FFCV...")
        DatasetWriter(
            full_beton,
            {
                "filename": BytesField(),
                "waveform": BytesField(),
                "sr": IntField(),
                "features": BytesField(),
                "latents": BytesField(),
            },
            num_workers=24,
            page_size=32 * 8388608,
        ).from_indexed_dataset(FullAudio2LatentFFCVPreprocessor(input_dir, fps))
        print("Preprocessing complete!")
    full_loader = Loader(
        full_beton,
        batch_size=1,
        num_workers=24,
        order=OrderOption.QUASI_RANDOM,
        pipelines={
            "filename": [BytesDecoder()],
            "waveform": [BytesDecoder()],
            "sr": [IntDecoder()],
            "features": [BytesDecoder()],
            "latents": [BytesDecoder()],
        },
        os_cache=True,
    )
    return full_loader


def overlapping_slices(tensor, length):
    return torch.stack(
        sum([list(torch.split(tensor[start:], length)[:-1]) for start in range(0, length, length // 4)], [])
    )


# fmt: off
def get_ffcv_dataloaders(input_dir, batch_size, dur, fps):
    L = int(dur * fps)

    datastem = f"cache/{Path(input_dir).stem}_{L}frames"
    train_beton = f"ffcv_{datastem}_ffcv_train.beton"
    val_beton = f"ffcv_{datastem}_ffcv_val.beton"
    mean_file = f"{datastem}_train_mean.npy"
    std_file = f"{datastem}_train_std.npy"

    if not os.path.exists(train_beton):
        full_dataset = FullAudio2LatentFFCVPreprocessor(input_dir, fps)
        full_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=24, prefetch_factor=1)

        train_feat_file, train_lat_file = f"{datastem}_train_feats.npy", f"{datastem}_train_lats.npy"
        val_feat_file, val_lat_file = f"{datastem}_val_feats.npy", f"{datastem}_val_lats.npy"
        train_noise4_file, val_noise4_file = f"{datastem}_train_noise4.npy", f"{datastem}_val_noise4.npy"
        train_noise8_file, val_noise8_file = f"{datastem}_train_noise8.npy", f"{datastem}_val_noise8.npy"
        train_noise16_file, val_noise16_file = f"{datastem}_train_noise16.npy", f"{datastem}_val_noise16.npy"
        train_noise32_file, val_noise32_file = f"{datastem}_train_noise32.npy", f"{datastem}_val_noise32.npy"
        if not os.path.exists(mean_file):
            print("Preprocessing data for FFCV...")
            train_or_val = np.random.RandomState(42).rand(len(full_dataset)) < 0.8
            val_files, train_files = [], []
            val_size, train_size = 0, 0
            with NumpyArray(train_feat_file) as train_feats, NumpyArray(val_feat_file) as val_feats, \
                 NumpyArray(train_lat_file) as train_lats, NumpyArray(val_lat_file) as val_lats, \
                 NumpyArray(train_noise4_file) as train_noise4, NumpyArray(val_noise4_file) as val_noise4, \
                 NumpyArray(train_noise8_file) as train_noise8, NumpyArray(val_noise8_file) as val_noise8, \
                 NumpyArray(train_noise16_file) as train_noise16, NumpyArray(val_noise16_file) as val_noise16, \
                 NumpyArray(train_noise32_file) as train_noise32, NumpyArray(val_noise32_file) as val_noise32:
                for i, ((file,), _, _, features, latents, noise4, noise8, noise16, noise32) in enumerate(tqdm(full_loader, "Chunking...", smoothing=0)):
                    features, latents = features.squeeze(), latents.squeeze()
                    noise4, noise8, noise16, noise32 = noise4.squeeze(0), noise8.squeeze(0), noise16.squeeze(0), noise32.squeeze(0)
                    features = overlapping_slices(features, L)
                    latents = overlapping_slices(latents, L)
                    noise4 = overlapping_slices(noise4, L)
                    noise8 = overlapping_slices(noise8, L)
                    noise16 = overlapping_slices(noise16, L)
                    noise32 = overlapping_slices(noise32, L)
                    feature_dim, (n_ws, latent_dim) = features.shape[-1], latents.shape[2:]
                    (train_files if train_or_val[i] else val_files).append(file)
                    (train_feats if train_or_val[i] else val_feats).append(features.contiguous().numpy())
                    (train_lats if train_or_val[i] else val_lats).append(latents.contiguous().numpy())
                    (train_noise4 if train_or_val[i] else val_noise4).append(noise4.squeeze().contiguous().numpy())
                    (train_noise8 if train_or_val[i] else val_noise8).append(noise8.squeeze().contiguous().numpy())
                    (train_noise16 if train_or_val[i] else val_noise16).append(noise16.squeeze().contiguous().numpy())
                    (train_noise32 if train_or_val[i] else val_noise32).append(noise32.squeeze().contiguous().numpy())
                    if train_or_val[i]:
                        train_size += len(features)
                    else:
                        val_size += len(features)
            with open(f"{datastem}_train_files.txt", "w") as f:
                f.write("\n".join(train_files))
            with open(f"{datastem}_val_files.txt", "w") as f:
                f.write("\n".join(val_files))
            feats_train = np.load(train_feat_file, mmap_mode="r")
            np.save(mean_file, np.mean(feats_train, axis=(0, 1)))
            np.save(std_file, np.std(feats_train, axis=(0, 1)))
            del feats_train
        else:
            train_size, _, feature_dim = np.load(train_feat_file, mmap_mode="r").shape
            val_size, _, n_ws, latent_dim = np.load(val_lat_file, mmap_mode="r").shape

        print("Preprocessing chunks to FFCV...")
        fields = {
            "features": NDArrayField(shape=(L, 59), dtype=np.dtype("float32")),
            "latents": NDArrayField(shape=(L, n_ws, latent_dim), dtype=np.dtype("float32")),
            "noise4": NDArrayField(shape=(L, 4, 4), dtype=np.dtype("float32")),
            "noise8": NDArrayField(shape=(L, 8, 8), dtype=np.dtype("float32")),
            "noise16": NDArrayField(shape=(L, 16, 16), dtype=np.dtype("float32")),
            "noise32": NDArrayField(shape=(L, 32, 32), dtype=np.dtype("float32")),
        }
        DatasetWriter(train_beton, fields, num_workers=24).from_indexed_dataset(SlicedAudio2LatentFFCVPreprocessor(datastem, split="train"))
        DatasetWriter(val_beton, fields, num_workers=24).from_indexed_dataset(SlicedAudio2LatentFFCVPreprocessor(datastem, split="val"))
        print("Preprocessing complete!")

        print()
        print("data summary:")
        print("train feature sequences:", (train_size, L, feature_dim))
        print("train latent sequences:", (train_size, L, n_ws, latent_dim))
        print("val feature sequences:", (val_size, L, feature_dim))
        print("val latent sequences:", (val_size, L, n_ws, latent_dim))
        print(f"train is {int(100 * train_size / (train_size + val_size))}%")
        print()

    class ToDevice(torch.nn.Module):
        def forward(self,x):
            return x.to('cuda' if torch.cuda.is_available() else 'cpu')

    train_mean = np.load(mean_file)
    train_std = np.load(std_file)
    pipelines = {
        "features": [NDArrayDecoder(), ToTensor(), ToDevice()],
        "latents": [NDArrayDecoder(), ToTensor(), ToDevice()],
        "noise4": [NDArrayDecoder(), ToTensor(), ToDevice()],
        "noise8": [NDArrayDecoder(), ToTensor(), ToDevice()],
        "noise16": [NDArrayDecoder(), ToTensor(), ToDevice()],
        "noise32": [NDArrayDecoder(), ToTensor(), ToDevice()]
    }
    train_loader = Loader(
        train_beton, batch_size=batch_size, num_workers=24, order=OrderOption.QUASI_RANDOM, pipelines=pipelines, os_cache=True, drop_last=True
    )
    val_loader = Loader(
        val_beton, batch_size=batch_size, num_workers=24, order=OrderOption.QUASI_RANDOM, pipelines=pipelines, os_cache=True, drop_last=True
    )

    return train_mean, train_std, train_loader, val_loader
