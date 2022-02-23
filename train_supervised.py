# fmt:off

import argparse
from copy import deepcopy
import gc
import os
import shutil
import sys
from functools import partial
from glob import glob
from pathlib import Path

import joblib
import librosa as rosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchdatasets as td
from better_lstm import LSTM
from npy_append_array import NpyAppendArray as NumpyArray
from scipy import signal, stats
from scipy.interpolate import splev, splrep
from torch.nn import (GELU, GRU, AvgPool1d, Conv1d, Dropout, LazyConv1d,
                      LazyConvTranspose1d, LeakyReLU, Linear, Module,
                      ModuleList, Parameter, Sequential, Sigmoid)
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchqrnn import QRNN
from tqdm import tqdm
from x_transformers.x_transformers import AttentionLayers

from context_fid import calculate_fcd

# fmt:on

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from ops.video import VideoWriter

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR = 24575
DUR = 8
FPS = 24


@torch.inference_mode()
def _audio2video(
    a2l,
    test_features,
    audio_file,
    out_file,
    stylegan_file,
    output_size=(512, 512),
    batch_size=8,
    offset=0,
    duration=40,
    seed=42,
):
    test_latents = a2l(test_features).squeeze()

    mapper = StyleGAN2Mapper(model_file=stylegan_file, inference=False)
    latent_offset = mapper(torch.from_numpy(np.random.RandomState(seed).randn(1, 512))).to(device)
    del mapper

    synthesizer = StyleGAN2Synthesizer(
        model_file=stylegan_file, inference=False, output_size=output_size, strategy="stretch", layer=0
    )
    synthesizer.eval().to(device)

    with VideoWriter(
        output_file=out_file,
        output_size=output_size,
        audio_file=audio_file,
        audio_offset=offset,
        audio_duration=duration,
    ) as video:
        for i in tqdm(range(0, len(test_latents), batch_size), unit_scale=batch_size):
            for frame in synthesizer(latent_offset + test_latents[i : i + batch_size]).add(1).div(2):
                video.write(frame.unsqueeze(0))

    del test_features, test_latents, latent_offset, synthesizer
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def audio2video(a2l, audio_file, out_file, stylegan_file, output_size=(512, 512), batch_size=8, offset=0, duration=40):
    a2l = a2l.eval().to(device)
    audio, sr = torchaudio.load(audio_file)
    test_features = audio2features(audio, sr)
    test_features = test_features[24 * offset : 24 * (offset + duration)]
    test_features = test_features.unsqueeze(0).to(device)
    _audio2video(a2l, test_features, audio_file, out_file, stylegan_file, output_size, batch_size, offset, duration)


def low_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 200, "low", fs=sr, output="sos"), audio)


def mid_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, [200, 2000], "band", fs=sr, output="sos"), audio)


def high_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 2000, "high", fs=sr, output="sos"), audio)


def audio2features(audio, sr, clamp=True, smooth=True):
    if audio.dim() == 2:
        audio = audio.mean(0)
    audio = torchaudio.transforms.Resample(sr, SR)(audio).numpy()

    mfcc = rosa.feature.mfcc(y=audio, sr=SR, hop_length=1024).transpose(1, 0)
    chroma = rosa.feature.chroma_cens(y=audio, sr=SR, hop_length=1024).transpose(1, 0)
    tonnetz = rosa.feature.tonnetz(y=audio, sr=SR, hop_length=1024).transpose(1, 0)
    contrast = rosa.feature.spectral_contrast(y=audio, sr=SR, hop_length=1024).transpose(1, 0)
    onsets = rosa.onset.onset_strength(y=audio, sr=SR, hop_length=1024).reshape(-1, 1)
    onsets_low = rosa.onset.onset_strength(y=low_pass(audio, SR), sr=SR, hop_length=1024).reshape(-1, 1)
    onsets_mid = rosa.onset.onset_strength(y=mid_pass(audio, SR), sr=SR, hop_length=1024).reshape(-1, 1)
    onsets_high = rosa.onset.onset_strength(y=high_pass(audio, SR), sr=SR, hop_length=1024).reshape(-1, 1)
    pulse = rosa.beat.plp(audio, SR, hop_length=1024, win_length=1024, tempo_min=60, tempo_max=180).reshape(-1, 1)
    volume = rosa.feature.rms(y=audio, hop_length=1024).reshape(-1, 1)
    flatness = rosa.feature.spectral_flatness(y=audio, hop_length=1024).reshape(-1, 1)

    features = np.concatenate(
        (mfcc, chroma, tonnetz, contrast, onsets, onsets_low, onsets_mid, onsets_high, pulse, volume, flatness), axis=1
    )
    features = torch.from_numpy(features).float()
    if clamp:
        features = features.clamp(torch.quantile(features, q=0.025, dim=0), torch.quantile(features, q=0.975, dim=0))
    if smooth:
        features = gaussian_filter(features, 2)
    return features


class PreprocessAudioLatentDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.files = sum([glob(f"{directory}*.{ext}") for ext in ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]], [])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        filename, _ = os.path.splitext(file)

        audio, sr = torchaudio.load(file)
        features = audio2features(audio, sr)
        features = torch.stack(  # overlapping slices of DUR seconds
            sum(
                [list(torch.split(features[start:], DUR * FPS)[:-1]) for start in range(0, DUR * FPS, DUR // 4 * FPS)],
                [],
            )
        )

        try:
            latents = joblib.load(f"{filename}.npy").float()
        except:
            latents = torch.from_numpy(np.load(f"{filename}.npy")).float()
        latents = torch.stack(
            sum(
                [list(torch.split(latents[start:], DUR * FPS)[:-1]) for start in range(0, DUR * FPS, DUR // 4 * FPS)],
                [],
            )
        )

        return file, features, latents


def collate(batch):
    return tuple(b if isinstance(b[0], str) else torch.cat(b) for b in list(map(list, zip(*batch))))


def filename(file, name):
    return file.replace(".npy", f"_{name}.npy")


def preprocess_audio(in_dir, out_file):
    train_feat_file, train_lat_file = filename(out_file, "train_feats"), filename(out_file, "train_lats")
    val_feat_file, val_lat_file = filename(out_file, "val_feats"), filename(out_file, "val_lats")
    if not os.path.exists(train_lat_file):
        dataset = PreprocessAudioLatentDataset(in_dir)
        train_or_val = np.random.RandomState(42).rand(len(dataset)) < 0.8
        val_files, train_files = [], []
        with NumpyArray(train_feat_file) as train_feats, NumpyArray(train_lat_file) as train_lats, NumpyArray(
            val_feat_file
        ) as val_feats, NumpyArray(val_lat_file) as val_lats:
            for i, ((_file,), _features, _latents) in enumerate(
                tqdm(DataLoader(dataset, num_workers=16, collate_fn=collate), desc="Preprocessing...")
            ):
                file, features, latents = deepcopy(_file), deepcopy(_features), deepcopy(_latents)
                del _file, _features, _latents
                (train_files if train_or_val[i] else val_files).append(file)
                feats = train_feats if train_or_val[i] else val_feats
                lats = train_lats if train_or_val[i] else val_lats
                for feat, lat in zip(features, latents):
                    feats.append(feat.unsqueeze(0).contiguous().numpy())
                    lats.append(lat.unsqueeze(0).contiguous().numpy())

        with open(filename(out_file, "train_files").replace(".npy", ".txt"), "w") as f:
            f.write("\n".join(train_files))
        with open(filename(out_file, "val_files").replace(".npy", ".txt"), "w") as f:
            f.write("\n".join(val_files))

        feats_train = np.load(train_feat_file, mmap_mode="r")
        np.save(filename(out_file, "train_feats_mean"), np.mean(feats_train, axis=(0, 1)))
        np.save(filename(out_file, "train_feats_std"), np.std(feats_train, axis=(0, 1)))

        lats_train = np.load(train_lat_file, mmap_mode="r")
        np.save(filename(out_file, "train_lats_offsets"), np.mean(lats_train, axis=(1, 2)))

        if os.path.exists(val_feat_file):
            feats_val = np.load(val_feat_file, mmap_mode="r")
            np.save(filename(out_file, "val_feats_mean"), np.mean(feats_val, axis=(0, 1)))
            np.save(filename(out_file, "val_feats_std"), np.std(feats_val, axis=(0, 1)))

            lats_val = np.load(val_lat_file, mmap_mode="r")
            np.save(filename(out_file, "val_lats_offsets"), np.mean(lats_val, axis=(1, 2)))


def print_data_summary(train, val):
    print()
    print("data summary:")
    print("train audio feature sequences:", train.features.shape)
    print("train latent sequences:", train.latents.shape)
    print("val audio feature sequences:", val.features.shape)
    print("val latent sequences:", val.latents.shape)
    print("train %:", int(100 * len(train) / (len(train) + len(val))))
    print()


def print_model_summary(model):
    print()
    print("model summary:")
    print("name".ljust(50), "class".ljust(20), "output shape".ljust(40), "num params")
    print("-" * 130)
    global total_params
    total_params = 0
    handles = []
    for name, block in a2l.named_modules():

        def hook(m, i, o, name=name):
            global total_params
            if len(list(m.named_modules())) == 1:
                class_name = m.__class__.__name__
                output_shape = (
                    tuple(tuple(oo.shape) if not isinstance(oo, int) else oo for oo in o)
                    if isinstance(o, tuple)
                    else tuple(o.shape)
                )
                num_params = sum(p.numel() for p in m.parameters())
                total_params += num_params
                print(
                    name.ljust(50),
                    class_name.ljust(20),
                    f"{output_shape}".ljust(40),
                    f"{num_params/ 1000:.2f} K" if num_params > 0 else "0",
                )

        handles.append(block.register_forward_hook(hook))
    a2l(inputs.to(device))
    for handle in handles:
        handle.remove()
    print("-" * 130)
    print("total".ljust(50), f"".ljust(20), f"".ljust(40), f"{total_params/1e6:.2f} M")


class AudioLatentDataset(td.Dataset):
    def __init__(self, file, split="train"):
        super().__init__()
        try:
            self.features = np.load(file.replace(".npy", f"_{split}_feats.npy"), mmap_mode="r")
            self.latents = np.load(file.replace(".npy", f"_{split}_lats.npy"), mmap_mode="r")
            self.offsets = np.load(file.replace(".npy", f"_{split}_lats_offsets.npy"), mmap_mode="r")

            self.mean = np.load(file.replace(".npy", f"_{split}_feats_mean.npy"))
            self.std = np.load(file.replace(".npy", f"_{split}_feats_std.npy"))

        except FileNotFoundError:
            self.features = np.empty((0, 0, 0))
            self.latents = np.empty((0, 0, 0, 0))
            self.offsets = np.empty((0, 0, 0, 0))

            self.mean = np.empty((0, 0, 0))
            self.std = np.empty((0, 0, 0))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        residuals = self.latents[index] - self.offsets[index]
        return torch.from_numpy(self.features[index].copy()), torch.from_numpy(residuals.copy())


import random


def normalize(x):
    y = x - x.min()
    y = y / y.max()
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


def spline_loop_latents(latent_selection, loop_len):
    latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])
    x = np.linspace(0, 1, loop_len)
    latents = np.zeros((loop_len, *latent_selection.shape[1:]))
    for lay in range(latent_selection.shape[1]):
        for lat in range(latent_selection.shape[2]):
            tck = splrep(np.linspace(0, 1, latent_selection.shape[0]), latent_selection[:, lay, lat])
            latents[:, lay, lat] = splev(x, tck)
    latents = torch.from_numpy(latents)
    return latents


class LatentAugmenter:
    def __init__(self, checkpoint, n_patches) -> None:
        model = StyleGAN2Mapper(checkpoint, inference=False).eval()
        self.n_patches = n_patches
        self.ws = model.forward(latent_z=torch.randn((16384, 512)))
        self.feat_idxs = {
            # "mfccs": (0, 20),
            "chroma": (20, 32),
            "tonnetz": (32, 38),
            # "contrast": (38, 45),
            "onsets": (45, 46),
            "onsets_low": (46, 47),
            "onsets_mid": (47, 48),
            "onsets_high": (48, 49),
            # "pulse": (49, 50),
            "volume": (50, 51),
            # "flatness": (51, 52),
        }
        self.feat_keys = list(self.feat_idxs.keys())
        self.single_dim = -5

    @torch.no_grad()
    def __call__(self, feature):
        latent = spline_loop_latents(
            self.ws[np.random.randint(0, self.ws.shape[0], np.random.randint(3, 12))], len(feature)
        )

        for n in range(self.n_patches):
            start, stop = self.feat_idxs[random.choice(self.feat_keys)]

            if np.random.rand() > 0.5:
                lay_start = np.random.randint(0, latent.shape[1] - 6)
                lay_stop = np.random.randint(lay_start, latent.shape[1])
            else:
                lay_start = 0
                lay_stop = self.ws.shape[1]

            if stop - start == 1:
                lat = self.ws[np.random.randint(0, self.ws.shape[0], 1)]
                modulation = normalize(feature[:, start:stop, None])
                latent[:, lay_start:lay_stop] *= 1 - modulation
                latent[:, lay_start:lay_stop] += modulation * lat[:, lay_start:lay_stop]

            else:
                lats = self.ws[np.random.randint(0, self.ws.shape[0], stop - start)]
                modulation = normalize(feature[:, start:stop])
                modulation /= modulation.sum(1, keepdim=True)
                patch_latent = torch.einsum("Atwl,Atwl->twl", modulation.permute(1, 0)[..., None, None], lats[:, None])

                if np.random.rand() > 0.666:
                    inter_start, inter_stop = self.feat_idxs[random.choice(self.feat_keys[self.single_dim :])]
                    intermodulator = normalize(feature[:, inter_start:inter_stop, None])
                    latent[:, lay_start:lay_stop] *= 1 - intermodulator
                    latent[:, lay_start:lay_stop] += intermodulator * patch_latent[:, lay_start:lay_stop]
                else:
                    latent[:, lay_start:lay_stop] = patch_latent[:, lay_start:lay_stop]

        offset = latent.mean(dim=(1, 2), keepdim=True)
        residuals = latent - offset
        return residuals, offset


class AugmentedLatentDataset(td.Dataset):
    def __init__(self, file, split="train", checkpoint="/home/hans/modelzoo/train_checks/neurout2-117.pt", n_patches=3):
        super().__init__()
        self.features = np.load(file.replace(".npy", f"_{split}_feats.npy"), mmap_mode="r")
        self.augmenter = LatentAugmenter(checkpoint, n_patches)

    def __len__(self):
        return len(self.features)

    @torch.no_grad()
    def __getitem__(self, index):
        features = torch.from_numpy(self.features[index].copy())
        residuals, offsets = self.augmenter(features)
        return features.float(), residuals.float()


class Normalize(Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.FloatTensor(mean))
        self.register_buffer("std", torch.FloatTensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std


class MaybePad(Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, x):
        remainder = x.shape[1] % (2 ** self.l)
        if remainder != 0:
            x = F.pad(x, (0, 0, 0, remainder))
        return x


class SwapChannels(Module):
    def forward(self, x):
        return x.transpose(1, 2)  # B, T, C <---> B, C, T


class SwapBatch(Module):
    def forward(self, x):
        return x.transpose(0, 1)  # B, T, C <---> T, B, C


class DummyHiddenState(Module):
    def forward(self, x):
        return x, 0


class UnsqueezeLayerwise(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x.unsqueeze(2).repeat(1, 1, self.n, 1)


class Print(Module):
    def forward(self, x):
        print(x.shape)
        return x


class Audio2Latent(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size,
        hidden_size,
        num_layers,
        n_outputs,
        output_size,
        backbone,
        skip_backbone,
        layerwise,
        n_layerwise,
        dropout,
    ):
        super().__init__()
        self.normalize = Normalize(input_mean, input_std)

        if backbone.lower() == "gru":
            self.backbone = GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
            )
            self.backbone.flatten_parameters()

        elif backbone.lower() == "lstm":
            self.backbone = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropouti=dropout,
                dropoutw=dropout,
                dropouto=dropout,
                batch_first=True,
            )
            self.backbone.flatten_parameters()

        elif backbone.lower() == "conv":

            def ConvBlock(out_c, transpose, k=5, s=2, p=2, op=1):
                return Sequential(
                    (LazyConvTranspose1d(out_c, k, s, p, op) if transpose else LazyConv1d(out_c, k, s, p)),
                    LeakyReLU(0.2),
                    Dropout(dropout),
                )

            multiplier = lambda x: 2 ** min(x, num_layers - x - 1)
            self.backbone = Sequential(
                MaybePad(num_layers // 2),
                SwapChannels(),
                *[ConvBlock(hidden_size * multiplier(n), transpose=n >= num_layers // 2) for n in range(num_layers)],
                SwapChannels(),
                DummyHiddenState(),
            )

        else:
            raise NotImplementedError()

        self.relu = Sequential(LeakyReLU(0.2), Dropout(dropout))

        if skip_backbone:
            skip_size = hidden_size
            self.backbone_skip = Sequential(
                Linear(input_size, hidden_size),
                LeakyReLU(0.2),
                Dropout(dropout),
                Linear(hidden_size, skip_size),
                LeakyReLU(0.2),
                Dropout(dropout),
                AttentionLayer(skip_size, skip_size, 4, 128, dropout),
                LeakyReLU(0.2),
                Dropout(dropout),
            )
        else:
            self.backbone_skip = None

        assert n_outputs % n_layerwise == 0, f"n_outputs must be divisible by n_layerwise! {n_outputs} / {n_layerwise}"
        layerwise_size = hidden_size + (skip_size if skip_backbone else 0)

        if layerwise == "dense":
            self.layerwise = LayerwiseLinear(
                in_channels=layerwise_size,
                out_channels=output_size,
                n_outputs=n_outputs,
                n_layerwise=n_layerwise,
                dropout=dropout,
                act=partial(F.leaky_relu, negative_slope=0.2),
            )

        elif layerwise == "conv":
            self.layerwise = LayerwiseConv(
                in_channels=layerwise_size,
                out_channels=output_size,
                kernel_size=5,
                n_outputs=n_outputs,
                n_layerwise=n_layerwise,
                dropout=dropout,
                act=partial(F.leaky_relu, negative_slope=0.2),
            )

        else:
            raise NotImplementedError()

    def forward(self, x):
        w, _ = self.backbone(self.normalize(x))
        w = w[:, : x.shape[1]]  # remove padding
        wx = self.relu(w) if self.backbone_skip is None else torch.cat((self.relu(w), self.backbone_skip(x)), axis=2)
        w_plus = self.layerwise(wx)
        return w_plus


class LayerwiseLinear(Module):
    def __init__(self, in_channels, out_channels, n_outputs, n_layerwise, dropout, act=F.gelu):
        super().__init__()
        self.dropout = dropout
        self.NO = n_outputs
        self.NL = n_layerwise
        self.w1, self.b1 = self.get_weights_and_bias(n_layerwise, in_channels, in_channels)
        self.w2, self.b2 = self.get_weights_and_bias(1, in_channels, out_channels)
        self.act = act

    def get_weights_and_bias(self, NL, IC, OC):
        w, b = torch.Tensor(NL, IC, OC), torch.Tensor(NL, OC)

        torch.nn.init.kaiming_uniform_(w, a=np.sqrt(5))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(b, -bound, bound)
        return Parameter(w.squeeze()), Parameter(b.squeeze())

    def forward(self, x):  # B,T,IC
        x = x.unsqueeze(2)  # B,T,1,IC
        x = x.tile(1, 1, self.NL, 1)  # B,T,NL,IC
        x = torch.matmul(x.unsqueeze(3), self.w1).squeeze(3) + self.b1  # B,T,NL,IC*2
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        x = torch.matmul(x, self.w2) + self.b2  # B,T,NL,OC
        B, T, _, OC = x.shape
        x = x.unsqueeze(3)  # B,T,NL,1,OC
        x = x.tile(1, 1, 1, self.NO // self.NL, 1)  # B,T,NL,NO//NL,OC
        x = x.reshape(B, T, self.NO, OC)  # B,T,NO,OC
        return x


class LayerwiseConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_outputs, n_layerwise, dropout, act=F.gelu):
        super().__init__()
        self.dropout = dropout
        self.NO = n_outputs
        self.NL = n_layerwise
        self.padding = (kernel_size - 1) // 2
        self.w1, self.b1 = self.get_weights_and_bias(n_layerwise, in_channels, in_channels, kernel_size)
        self.w2, self.b2 = self.get_weights_and_bias(1, in_channels, out_channels, kernel_size)
        self.act = act

    def get_weights_and_bias(self, NL, IC, OC, ks):
        w, b = torch.Tensor(NL * OC, IC, ks), torch.Tensor(1, NL * OC, 1)

        torch.nn.init.kaiming_uniform_(w, a=np.sqrt(5))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(b, -bound, bound)
        return Parameter(w), Parameter(b)

    def forward(self, x):  # B,T,IC
        B, T, IC = x.shape
        x = x.transpose(1, 2)  # B,IC,T
        x = x.unsqueeze(2)  # B,IC,1,T
        x = x.tile(1, 1, self.NL, 1)  # B,NL,IC,T
        x = x.reshape(B, self.NL * IC, T)  # B,NL*IC,T
        x = F.conv1d(x, self.w1, padding=self.padding, groups=self.NL) + self.b1  # B,NL*IC*2,T
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x.reshape(B * self.NL, -1, T)  # B*NL,IC*2,T
        x = F.conv1d(x, self.w2, padding=self.padding) + self.b2  # B*NL,OC,T
        x = x.reshape(B, self.NL, -1, T)  # B,NL,OC,T
        _, _, OC, _ = x.shape
        x = x.unsqueeze(2)  # B,NL,1,OC,T
        x = x.tile(1, 1, self.NO // self.NL, 1, 1)  # B,NL,NO//NL,OC,T
        x = x.reshape(B, self.NO, OC, T)  # B,NO,OC,T
        x = x.permute(0, 3, 1, 2)  # B,T,NO,OC
        return x


class Pool(Module):
    """Average pooling to halve length along 0th (time) axis"""

    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.pool = AvgPool1d(kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.pool(x.permute(1, 2, 0)).permute(2, 0, 1)  # TODO find a way without permutes?


class Unpool(Module):
    """Linear interpolation to double length along 0th (time) axis"""

    def forward(self, y):
        T, B, C = y.shape
        x = torch.linspace(0, 1, T, device=y.device, dtype=y.dtype)
        xq = torch.linspace(0, 1, T * 2, device=y.device, dtype=y.dtype)
        idxr = torch.searchsorted(x, xq, right=False)  # (Tq)
        idxr = torch.clamp(idxr, 1, T - 1)
        idxl = idxr - 1  # (Tq) from [0 to T-2]
        yl = y[:-1]  # (T-1, B, C)
        xl = x[:-1]  # (T-1)
        dy = y[1:] - yl  # (T-1, B, C)
        dx = x[1:] - xl  # (T-1)
        t = (xq - xl[idxl]) / dx[idxl]  # (Tq)
        yq = dy[idxl] * t[:, None, None]  # (Tq, B, C)
        yq += yl[idxl]
        return yq


class ConvTBC(Module):
    """1D convolution over an input of shape (time x batch x channel)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = Parameter(torch.Tensor(kernel_size, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels)

        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, *unused):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)


class EfficientChannelAttention(Module):
    def __init__(self, kernel_size=5):
        super(EfficientChannelAttention, self).__init__()
        self.conv = ConvTBC(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=0, keepdim=True)
        y = self.conv(y.transpose(0, 2)).transpose(0, 2)  # conv over channels
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ConvolutionalBlockAttention(Module):
    def __init__(self, in_channels, out_channels, ratio=8):
        super(ConvolutionalBlockAttention, self).__init__()
        self.linear = Linear(in_channels, out_channels)
        self.fc = Sequential(
            ConvTBC(in_channels, in_channels // ratio, 1, bias=False),
            LeakyReLU(0.2),
            ConvTBC(in_channels // ratio, out_channels, 1, bias=False),
        )
        self.sigmoid = Sigmoid()
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x):
        print(x.shape, self.in_channels, self.out_channels)
        avg_out = self.fc(torch.mean(x, dim=0, keepdim=True))
        max_out = self.fc(torch.max(x, dim=0, keepdim=True).values)
        print(self.linear(x).shape, avg_out.shape, max_out.shape)
        return self.linear(x) * self.sigmoid(avg_out + max_out)


class AttentionLayer(AttentionLayers):
    def __init__(self, in_channels, out_channels, n_head, dim_head, dropout):
        super().__init__(
            dim=in_channels,
            depth=1,
            heads=n_head,
            attn_dim_head=dim_head,
            ff_dim_out=out_channels,
            dropout=dropout,
            rotary_pos_emb=True,
        )


class ContextAndCorrelationLayer(Module):
    def __init__(self, context, correlation, in_channels, out_channels, kernel_size, dropout, additive=False):
        super().__init__()
        self.additive = additive
        out_channels = out_channels if additive else out_channels // 2
        self.hidden_channels = out_channels

        if context == "gru":
            self.context = GRU(in_channels, out_channels)  # TODO dropout?
            self.context.flatten_parameters()
        elif context == "lstm":
            self.context = LSTM(in_channels, out_channels, dropouti=dropout, dropoutw=dropout, dropouto=dropout)
            self.context.flatten_parameters()
        elif context == "qrnn":
            self.context = QRNN(in_channels, out_channels, dropout=dropout)
        elif context == "conv":
            self.context = Sequential(
                ConvTBC(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                Dropout(dropout),
                DummyHiddenState(),
            )
        elif context == "transformer":
            self.context = torch.nn.Sequential(AttentionLayer(in_channels, dropout=dropout), DummyHiddenState())
        else:
            raise NotImplementedError()

        if correlation == "linear":
            self.correlation = Sequential(Linear(in_channels, out_channels), Dropout(dropout))
        elif correlation == "eca":
            self.correlation = Sequential(
                EfficientChannelAttention(kernel_size),
                Linear(in_channels, out_channels),
                Dropout(dropout),
            )
        elif correlation == "cba":
            self.correlation = Sequential(ConvolutionalBlockAttention(in_channels, out_channels), Dropout(dropout))
        else:
            raise NotImplementedError()

    def forward(self, x):
        context, _ = self.context(x)
        correlation = self.correlation(x)
        return context + correlation if self.additive else torch.cat((context, correlation), dim=2)


class Audio2Latent2(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size,
        hidden_size,
        num_layers,
        n_outputs,
        output_size,
        context,
        correlation,
        n_layerwise,
        dropout,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.normalize = Normalize(input_mean, input_std + 1e-8)
        self.maybe_pad = MaybePad(num_layers // 2)
        self.swap_batch = SwapBatch()

        multiplier = lambda x: 2 ** min(x, num_layers - x - 1)
        in_channels = input_size
        layers = []
        for n in range(num_layers):
            out_channels = hidden_size * multiplier(n)
            layers.append(
                Sequential(
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        dropout=dropout,
                    ),
                    LeakyReLU(0.2),
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        dropout=dropout,
                    ),
                    LeakyReLU(0.2),
                )
            )
            in_channels = out_channels
        self.layers = ModuleList(layers)
        self.activation = LeakyReLU(0.2)

        self.pool = Pool(kernel_size=5)
        self.unpool = Unpool()

        self.layerwise = ModuleList(
            [
                Sequential(
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=hidden_size,
                        out_channels=hidden_size * 2,
                        kernel_size=5,
                        dropout=dropout,
                        additive=True,
                    ),
                    LeakyReLU(0.2),
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=hidden_size * 2,
                        out_channels=output_size,
                        kernel_size=5,
                        dropout=dropout,
                        additive=True,
                    ),
                    UnsqueezeLayerwise(n_outputs // n_layerwise),
                )
                for _ in range(n_layerwise)
            ]
        )

    def forward(self, x):
        B, T, C = x.shape
        w = self.maybe_pad(self.normalize(x))
        w = self.swap_batch(w)
        for n, layer in enumerate(self.layers):
            w = layer(w)
            w = self.pool(w) if n < len(self.layers) // 2 else self.unpool(w)
        w = w[:T]  # remove padding
        w_plus = torch.cat([layerwise(w) for layerwise in self.layerwise], axis=2)
        w_plus = self.swap_batch(w_plus)
        return w_plus


class ConvolutionalGatingUnit(Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv = Conv1d(
            channels // 2, channels // 2, kernel_size, padding=(kernel_size - 1) // 2, groups=channels // 2
        )
        self.dense = Linear(channels // 2, channels // 2)

    def forward(self, x, z=None):
        xr, xg = x.chunk(2, dim=2)
        xg = self.conv(xg.transpose(1, 2)).transpose(1, 2)
        xg = self.dense(xg)
        if z is not None:
            xg = xg + z
        return xr * xg


class MLPBlock(Module):
    def __init__(self, channels, kernel_size, mult):
        super().__init__()
        self.dense1 = Linear(channels, channels * mult)
        self.act = GELU()
        self.cgu = ConvolutionalGatingUnit(channels * mult, kernel_size)
        self.dense2 = Linear(channels * mult // 2, channels)

    def forward(self, x, z=None):
        y = self.dense1(x)
        y = self.act(y)
        y = self.cgu(y, z)
        y = self.dense2(y)
        return x + y


class MLP(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        in_channels,
        channels,
        out_channels,
        n_outputs,
        n_layerwise,
        num_layers,
        dropout,
        mult=2,
        kernel_size=15,
    ):
        super().__init__()
        self.normalize = Normalize(input_mean, input_std + 1e-8)
        self.attn = Sequential(
            Linear(in_channels, channels * mult // 2),
            GELU(),
            Dropout(dropout),
            AttentionLayer(channels * mult // 2, channels * mult // 2, n_head=4, dim_head=128, dropout=dropout),
        )
        self.input_dense = Linear(in_channels, channels)
        self.dropout = Dropout(dropout)
        self.blocks = ModuleList([MLPBlock(channels, kernel_size=kernel_size, mult=mult) for _ in range(num_layers)])
        self.layerwise = LayerwiseConv(channels, out_channels, 5, n_outputs, n_layerwise, dropout)

    def forward(self, x):
        x = self.normalize(x)
        z = self.attn(x)
        x = self.input_dense(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, z)
            x = self.dropout(x)
        w = self.layerwise(x)
        return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Audio2Latent options
    parser.add_argument("--backbone", type=str, default=None, choices=["gru", "lstm", "conv"])
    parser.add_argument("--skip_backbone", action="store_true")
    parser.add_argument("--layerwise", type=str, default=None, choices=["dense", "conv"])

    # Audio2Latent2 options
    parser.add_argument("--context", type=str, default=None, choices=["gru", "lstm", "qrnn", "conv", "transformer"])
    parser.add_argument("--correlation", type=str, default=None, choices=["linear", "eca", "cba"])

    # MLP options
    parser.add_argument("--mlp", action="store_true")

    # Shared options
    parser.add_argument("--n_layerwise", type=int, default=3, choices=[1, 2, 3, 6, 9, 18])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training options
    parser.add_argument("--aug_weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    n_frames = int(DUR * FPS)

    in_dir = "/home/hans/datasets/audio2latent/"
    dataset_cache = f"cache/{Path(in_dir).stem}_preprocessed_{n_frames}frames.npy"
    test_audio = "/home/hans/datasets/wavefunk/Ouroboromorphism_49_89.flac"

    batch_size = args.batch_size

    backbone = args.backbone
    skip_backbone = args.skip_backbone
    context = args.context
    correlation = args.correlation
    layerwise = args.layerwise
    n_layerwise = args.n_layerwise
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout

    lr = args.lr
    wd = args.wd
    aug_weight = args.aug_weight

    preprocess_audio(in_dir, dataset_cache)
    train_dataset = AudioLatentDataset(dataset_cache, "train")
    if aug_weight > 0:
        aug_dataset = AugmentedLatentDataset(dataset_cache, "train")
    val_dataset = AudioLatentDataset(dataset_cache, "val")
    print_data_summary(train_dataset, val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    if aug_weight > 0:
        aug_dataloader = DataLoader(aug_dataset, batch_size=batch_size, num_workers=24, shuffle=True, drop_last=True)
    inputs, targets = next(iter(train_dataloader))
    feature_dim = inputs.shape[2]
    n_outputs, output_size = targets.shape[2], targets.shape[3]

    if args.backbone is not None:
        name = "_".join(
            [
                f"backbone:{backbone}:skip{skip_backbone}",
                f"layerwise:{layerwise}:{n_layerwise}",
                f"hidden_size:{hidden_size}",
                f"num_layers:{num_layers}",
                f"dropout:{dropout}",
                f"lr:{lr}",
                f"wd:{wd}",
            ]
        )
        a2l = Audio2Latent(
            input_mean=train_dataset.mean,
            input_std=train_dataset.std,
            backbone=backbone,
            skip_backbone=skip_backbone,
            layerwise=layerwise,
            n_layerwise=n_layerwise,
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_outputs=n_outputs,
            output_size=output_size,
        ).to(device)
    elif args.context is not None:
        name = "_".join(
            [
                f"context:{context}",
                f"correlation:{correlation}",
                f"n_layerwise:{n_layerwise}",
                f"hidden_size:{hidden_size}",
                f"num_layers:{num_layers}",
                f"dropout:{dropout}",
                f"lr:{lr}",
                f"wd:{wd}",
            ]
        )
        a2l = Audio2Latent2(
            context=context,
            correlation=correlation,
            input_mean=train_dataset.mean,
            input_std=train_dataset.std,
            n_layerwise=n_layerwise,
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_outputs=n_outputs,
            output_size=output_size,
        ).to(device)
    else:
        name = "_".join(
            [
                f"mlp",
                f"layerwise:{n_layerwise}",
                f"hidden_size:{hidden_size}",
                f"num_layers:{num_layers}",
                f"dropout:{dropout}",
                f"lr:{lr}",
                f"wd:{wd}",
            ]
        )
        a2l = MLP(
            input_mean=train_dataset.mean,
            input_std=train_dataset.std,
            in_channels=feature_dim,
            channels=hidden_size,
            out_channels=output_size,
            n_outputs=n_outputs,
            n_layerwise=n_layerwise,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    a2l(inputs.to(device))
    print_model_summary(a2l)

    if aug_weight > 0:
        name += "_augmented"

    optimizer = torch.optim.AdamW(a2l.parameters(), lr=lr, weight_decay=wd)

    writer = SummaryWriter(comment=name)
    shutil.copy(__file__, writer.log_dir)

    from time import time

    n_iter = 0
    n_epochs = args.epochs
    video_interval = 20
    eval_interval = 5
    pbar = tqdm(range(n_epochs))
    grad_noise = []
    for epoch in pbar:
        losses, aug_losses = [], []
        a2l.train()

        # t = time()
        if aug_weight > 0:
            dataloader = zip(train_dataloader, aug_dataloader)
        else:
            dataloader = train_dataloader

        for batch in dataloader:
            if aug_weight > 0:
                (inputs, targets), (aug_inputs, aug_targets) = batch
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            # print("data", time() - t)
            # t = time()

            outputs = a2l(inputs)
            loss = F.mse_loss(outputs, targets)

            if aug_weight > 0:
                aug_inputs, aug_targets = aug_inputs.to(device), aug_targets.to(device)
                aug_outputs = a2l(aug_inputs)
                aug_loss = aug_weight * F.mse_loss(aug_outputs, aug_targets)
            else:
                aug_loss = 0

            optimizer.zero_grad()
            (loss + aug_loss).backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), n_iter)
            writer.add_scalar("Loss/augtrain", aug_loss.item(), n_iter)
            losses.append(loss.item())
            aug_losses.append(aug_loss.item())

            n_iter += len(inputs)
        train_loss = np.mean(losses)
        aug_loss = np.mean(aug_losses)

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == n_epochs:
            with torch.inference_mode():
                a2l.eval()
                if len(val_dataset) > 0:
                    val_dataloader = DataLoader(
                        val_dataset, batch_size=batch_size, num_workers=24, shuffle=True, drop_last=True
                    )
                    val_loss, latent_residuals = 0, []
                    for it, (inputs, targets) in enumerate(val_dataloader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = a2l(inputs)
                        latent_residuals.append(np.random.choice(outputs.cpu().numpy().flatten(), 100_000))
                        val_loss += F.mse_loss(outputs, targets)
                    val_loss /= len(val_dataloader)
                    writer.add_scalar("Loss/val", val_loss.item(), n_iter)

                    fcd = calculate_fcd(val_dataloader, a2l)
                    writer.add_scalar("Eval/FCD", fcd.item(), n_iter)

                    try:
                        loc, scale = stats.laplace.fit(np.concatenate(latent_residuals), loc=0, scale=0.1)
                        writer.add_scalar("Eval/laplace_b", scale.item(), n_iter)
                    except Exception as e:
                        pbar.write(f"\nError in Laplace fit:\n{e}\n\n")
                        scale = -1

                else:
                    val_loss, scale = -1, -1

            pbar.write("")
            pbar.write(f"epoch {epoch + 1}")
            pbar.write(f"train_loss: {train_loss:.4f}")
            pbar.write(f"aug_loss  : {aug_loss:.4f}")
            pbar.write(f"val_loss  : {val_loss:.4f}")
            pbar.write(f"laplace_b : {scale:.4f}")
            pbar.write(f"fcd       : {fcd:.4f}")
            pbar.write("")

        if (epoch + 1) % video_interval == 0 or (epoch + 1) == n_epochs:
            checkpoint_name = f"audio2latent_{name}_steps{n_iter:08}_fcd{fcd:.4f}_b{scale:.4f}_val{val_loss:.4f}"
            joblib.dump(
                {"a2l": a2l, "optim": optimizer, "n_iter": n_iter}, f"{writer.log_dir}/{checkpoint_name}.pt", compress=9
            )
            audio2video(
                a2l=a2l,
                audio_file=test_audio,
                out_file=f"{writer.log_dir}/{checkpoint_name}_{Path(test_audio).stem}.mp4",
                stylegan_file="/home/hans/modelzoo/train_checks/neurout2-117.pt",
            )

    writer.close()
