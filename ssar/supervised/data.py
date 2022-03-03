import io
import os
from glob import glob
from pathlib import Path

import joblib
import librosa as rosa
import madmom as mm
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
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def low_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 200, "low", fs=sr, output="sos"), audio)


def mid_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, [200, 2000], "band", fs=sr, output="sos"), audio)


def high_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 2000, "high", fs=sr, output="sos"), audio)


def onset_strength(audio, sr):
    audio = rosa.effects.percussive(audio, margin=4.0)

    sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
    sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048)
    stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, circular_shift=True)
    spec = mm.audio.spectrogram.Spectrogram(stft, circular_shift=True)
    filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24)

    spectral_diff = mm.features.onsets.spectral_diff(filt_spec)
    spectral_flux = mm.features.onsets.spectral_flux(filt_spec)
    superflux = mm.features.onsets.superflux(filt_spec)
    complex_flux = mm.features.onsets.complex_flux(filt_spec)
    modified_kullback_leibler = mm.features.onsets.modified_kullback_leibler(filt_spec)

    onset = np.mean(
        [
            spectral_diff / spectral_diff.max(),
            spectral_flux / spectral_flux.max(),
            superflux / superflux.max(),
            complex_flux / complex_flux.max(),
            modified_kullback_leibler / modified_kullback_leibler.max(),
        ],
        axis=0,
    )

    return onset


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


def audio2features(audio, sr, fps, clamp=True, smooth=True, onsets_only=False):
    if audio.dim() == 2:
        audio = audio.mean(0)
    audio = audio.numpy()
    n_frames = int(len(audio) / sr * fps)
    if onsets_only:
        features = [
            onset_strength(audio, sr).reshape(-1, 1),
            onset_strength(low_pass(audio, sr), sr).reshape(-1, 1),
            onset_strength(mid_pass(audio, sr), sr).reshape(-1, 1),
            onset_strength(high_pass(audio, sr), sr).reshape(-1, 1),
        ]
    else:
        features = [
            rosa.feature.mfcc(audio, sr).transpose(1, 0),
            rosa.feature.chroma_cens(rosa.effects.harmonic(audio, margin=4.0), sr).transpose(1, 0),
            rosa.feature.tonnetz(rosa.effects.harmonic(audio, margin=4.0), sr).transpose(1, 0),
            rosa.feature.spectral_contrast(audio, sr).transpose(1, 0),
            onset_strength(audio, sr).reshape(-1, 1),
            onset_strength(low_pass(audio, sr), sr).reshape(-1, 1),
            onset_strength(mid_pass(audio, sr), sr).reshape(-1, 1),
            onset_strength(high_pass(audio, sr), sr).reshape(-1, 1),
            rosa.beat.plp(audio, sr, win_length=1024, tempo_min=60, tempo_max=180).reshape(-1, 1),
            rosa.feature.rms(audio).reshape(-1, 1),
            rosa.feature.spectral_flatness(audio).reshape(-1, 1),
        ]
    features = [signal.resample(f, n_frames) for f in features]
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

        try:
            latents = joblib.load(f"{filename}.npy").float()
        except:
            latents = torch.from_numpy(np.load(f"{filename}.npy")).float()

        if self.return_bytes:
            return to_bytes(np.array([file])), to_bytes(audio), sr, to_bytes(features), to_bytes(latents)
        else:
            return file, audio, sr, features, latents


class SlicedAudio2LatentFFCVPreprocessor(Dataset):
    def __init__(self, datastem, split="train"):
        super().__init__()
        self.features = np.load(f"{datastem}_{split}_feats.npy", mmap_mode="r")
        self.latents = np.load(f"{datastem}_{split}_lats.npy", mmap_mode="r")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index].copy()
        latents = self.latents[index].copy()
        residuals = latents - latents.mean((0, 1))
        return features.astype(np.float32), residuals.astype(np.float32)


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
        if not os.path.exists(mean_file):
            print("Preprocessing data for FFCV...")
            train_or_val = np.random.RandomState(42).rand(len(full_dataset)) < 0.8
            val_files, train_files = [], []
            val_size, train_size = 0, 0
            with NumpyArray(train_feat_file) as train_feats, NumpyArray(train_lat_file) as train_lats, NumpyArray(
                val_feat_file
            ) as val_feats, NumpyArray(val_lat_file) as val_lats:
                for i, ((file,), _, _, features, latents) in enumerate(tqdm(full_loader, "Chunking...", smoothing=0)):
                    features, latents = features.squeeze(), latents.squeeze()
                    features = torch.stack(  # overlapping slices of DUR seconds
                        sum([list(torch.split(features[start:], L)[:-1]) for start in range(0, L, L // 4)], [])
                    )
                    latents = torch.stack(  # overlapping slices of DUR seconds
                        sum([list(torch.split(latents[start:], L)[:-1]) for start in range(0, L, L // 4)], [])
                    )
                    feature_dim, (n_ws, latent_dim) = features.shape[-1], latents.shape[2:]
                    (train_files if train_or_val[i] else val_files).append(file)
                    (train_feats if train_or_val[i] else val_feats).append(features.contiguous().numpy())
                    (train_lats if train_or_val[i] else val_lats).append(latents.contiguous().numpy())
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
        DatasetWriter(
            train_beton,
            {
                "features": NDArrayField(shape=(L, 4 if synthetic else 52), dtype=np.dtype("float32")),
                "latents": NDArrayField(shape=(L, n_ws, latent_dim), dtype=np.dtype("float32")),
            },
            num_workers=24,
        ).from_indexed_dataset(SlicedAudio2LatentFFCVPreprocessor(datastem, split="train"))
        DatasetWriter(
            val_beton,
            {
                "features": NDArrayField(shape=(L, 4 if synthetic else 52), dtype=np.dtype("float32")),
                "latents": NDArrayField(shape=(L, n_ws, latent_dim), dtype=np.dtype("float32")),
            },
            num_workers=24,
        ).from_indexed_dataset(SlicedAudio2LatentFFCVPreprocessor(datastem, split="val"))
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
    train_loader = Loader(
        train_beton,
        batch_size=batch_size,
        num_workers=24,
        order=OrderOption.QUASI_RANDOM,
        pipelines={"features": [NDArrayDecoder(), ToTensor()], "latents": [NDArrayDecoder(), ToTensor()]},
        os_cache=True,
        drop_last=True,
    )
    val_loader = Loader(
        val_beton,
        batch_size=batch_size,
        num_workers=24,
        order=OrderOption.QUASI_RANDOM,
        pipelines={"features": [NDArrayDecoder(), ToTensor()], "latents": [NDArrayDecoder(), ToTensor()]},
        os_cache=True,
        drop_last=True,
    )

    return train_mean, train_std, train_loader, val_loader
