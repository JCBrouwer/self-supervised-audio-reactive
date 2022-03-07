import io
import os
from glob import glob
from pathlib import Path

import joblib
import librosa as rosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from ffcv.fields import BytesField, IntField, NDArrayField
from ffcv.fields.decoders import BytesDecoder, IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from npy_append_array import NpyAppendArray as NumpyArray
from scipy import ndimage as ndi
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def low_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 200, "low", fs=sr, output="sos"), audio)


def mid_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, [200, 2000], "band", fs=sr, output="sos"), audio)


def high_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 2000, "high", fs=sr, output="sos"), audio)


def normalize(x):
    y = x - x.min()
    y = y / (y.max() + 1e-8)
    return y


def gaussian_filter(x, sigma, mode="replicate"):
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]

    radius = min(int(sigma * 4), 3 * len(x))
    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma ** 2 * kernel ** 2)
    kernel[radius + 1 :] *= 0.25  # make kernel less responsive to future information
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    if radius > n_frames:  # prevent padding errors on short sequences
        x = F.pad(x, (n_frames, n_frames), mode=mode)
        print(
            f"WARNING: Gaussian filter radius ({int(sigma * 4)}) is larger than number of frames ({n_frames}).\n\t Filter size has been lowered to ({radius}). You might want to consider lowering sigma ({sigma})."
        )
        x = F.pad(x, (radius - n_frames, radius - n_frames), mode="replicate")
    else:
        x = F.pad(x, (radius, radius), mode=mode)

    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def audio2features(audio, sr, fps, clamp=True, smooth=True, onsets_only=False):
    if audio.dim() == 2:
        audio = audio.mean(0)
    audio = torchaudio.transforms.Resample(sr, 24575)(audio).numpy()
    sr = 24575
    harmonic, percussive = rosa.effects.hpss(audio, margin=8.0)
    if onsets_only:
        multi_features = []
        single_features = [
            rosa.onset.onset_strength(percussive, sr, hop_length=1024),
            rosa.onset.onset_strength(low_pass(percussive, sr), sr, hop_length=1024),
            rosa.onset.onset_strength(mid_pass(percussive, sr), sr, hop_length=1024),
            rosa.onset.onset_strength(high_pass(percussive, sr), sr, hop_length=1024),
        ]
    else:
        # fmt:off
        multi_features = [
            ndi.gaussian_filter(rosa.feature.mfcc(audio, sr, hop_length=1024), [0, 10]),
            ndi.gaussian_filter(rosa.feature.chroma_cens(harmonic, sr, hop_length=1024), [0, 10]),
            ndi.gaussian_filter(rosa.feature.tonnetz(harmonic, sr, hop_length=1024), [0, 10]),
            ndi.gaussian_filter(rosa.feature.spectral_contrast(audio, sr, hop_length=1024), [0, 10]),
        ]
        single_features = [
            ndi.gaussian_filter(rosa.feature.spectral_flatness(audio, hop_length=1024), [0, 10]),
            rosa.onset.onset_strength(percussive, sr, hop_length=1024),
            rosa.onset.onset_strength(low_pass(percussive, sr), sr, hop_length=1024),
            rosa.onset.onset_strength(mid_pass(percussive, sr), sr, hop_length=1024),
            rosa.onset.onset_strength(high_pass(percussive, sr), sr, hop_length=1024),
            rosa.beat.plp(audio, sr, win_length=1024, tempo_min=60, tempo_max=180, hop_length=1024),
            rosa.feature.rms(harmonic, sr, hop_length=1024),
            rosa.feature.rms(low_pass(harmonic, sr), sr, hop_length=1024),
            rosa.feature.rms(mid_pass(harmonic, sr), sr, hop_length=1024),
            rosa.feature.rms(high_pass(harmonic, sr), sr, hop_length=1024),
            sigmoid(normalize(ndi.gaussian_filter(rosa.feature.rms(audio, hop_length=1024), [0, 10])) * 10 - 5),
            sigmoid(normalize(ndi.gaussian_filter(rosa.feature.rms(low_pass(audio, sr), hop_length=1024), [0, 10])) * 10 - 5),
            sigmoid(normalize(ndi.gaussian_filter(rosa.feature.rms(mid_pass(audio, sr), hop_length=1024), [0, 10])) * 10 - 5),
            sigmoid(normalize(ndi.gaussian_filter(rosa.feature.rms(high_pass(audio, sr), hop_length=1024), [0, 10])) * 10 - 5),
        ]
        # fmt:on
    features = [mf.transpose(1, 0) for mf in multi_features] + [sf.reshape(-1, 1) for sf in single_features]
    features = np.concatenate(features, axis=1)
    features = torch.from_numpy(features).float()
    if clamp:
        features = features.clamp(torch.quantile(features, q=0.025, dim=0), torch.quantile(features, q=0.975, dim=0))
    if smooth:
        features = gaussian_filter(features, 2)
    return features


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
    def __init__(self, directory, fps, synthetic, return_bytes=False):
        super().__init__()
        self.fps = fps
        self.return_bytes = return_bytes
        self.files = sum([glob(f"{directory}*.{ext}") for ext in ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]], [])
        self.synthetic = synthetic

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        filename, _ = os.path.splitext(file)

        audio, sr = torchaudio.load(file)
        features = audio2features(audio, sr, self.fps, onsets_only=self.synthetic)

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
def get_ffcv_dataloaders(input_dir, synthetic, batch_size, dur, fps):
    L = int(dur * fps)

    datastem = f"cache/{Path(input_dir).stem}_{L}frames"
    if synthetic:
        datastem += "_synthetic"
    train_beton = f"ffcv_{datastem}_ffcv_train.beton"
    val_beton = f"ffcv_{datastem}_ffcv_val.beton"
    mean_file = f"{datastem}_train_mean.npy"
    std_file = f"{datastem}_train_std.npy"

    if not os.path.exists(train_beton):
        full_dataset = FullAudio2LatentFFCVPreprocessor(input_dir, fps, synthetic)
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
            "features": NDArrayField(shape=(L, 4 if synthetic else 59), dtype=np.dtype("float32")),
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

    train_mean = np.load(mean_file)
    train_std = np.load(std_file)
    pipelines = {
        "features": [NDArrayDecoder(), ToTensor()],
        "latents": [NDArrayDecoder(), ToTensor()],
        "noise4": [NDArrayDecoder(), ToTensor()],
        "noise8": [NDArrayDecoder(), ToTensor()],
        "noise16": [NDArrayDecoder(), ToTensor()],
        "noise32": [NDArrayDecoder(), ToTensor()]
    }
    train_loader = Loader(
        train_beton, batch_size=batch_size, num_workers=24, order=OrderOption.QUASI_RANDOM, pipelines=pipelines, os_cache=True, drop_last=True
    )
    val_loader = Loader(
        val_beton, batch_size=batch_size, num_workers=24, order=OrderOption.QUASI_RANDOM, pipelines=pipelines, os_cache=True, drop_last=True
    )

    return train_mean, train_std, train_loader, val_loader
