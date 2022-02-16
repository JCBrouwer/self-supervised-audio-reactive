import gc
import os
import shutil
import sys
from glob import glob
from pathlib import Path

import joblib
import librosa as rosa
import numpy as np
import torch
import torchaudio
import torchdatasets as td
from npy_append_array import NpyAppendArray as NumpyArray
from scipy import signal
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from ops.video import VideoWriter

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR = 24575
DUR = 8
FPS = 24


def latent_residual_hist():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    latent_residuals = []
    i, n = 0, 32
    for _, targets in dataloader:
        latent_residuals.append(targets.numpy())
        i += 1
        if i > n:
            break
    latent_residuals = np.concatenate(latent_residuals).flatten()
    plt.figure(figsize=(16, 9))
    plt.hist(
        latent_residuals,
        bins=1000,
        label=f"Histogram (min={latent_residuals.min():.2f}, mean={latent_residuals.mean():.2f}, max={latent_residuals.max():.2f})",
        color="tab:blue",
        alpha=0.5,
        density=True,
    )
    loc, scale = stats.laplace.fit(latent_residuals, loc=0, scale=0.1)
    xs = np.linspace(-2, 2, 1000)
    plt.plot(
        xs,
        stats.laplace.pdf(xs, loc=loc, scale=scale),
        label=rf"PDF of MLE-fit Laplace($\mu$={loc:.2f}, $b$={scale:.2f})",
        color="tab:orange",
        alpha=0.75,
    )
    plt.title("Distribution of latent residuals")
    plt.legend()
    plt.xlim(-2, 2)
    plt.tight_layout()
    plt.savefig("latent-residuals-distribution.png")


@torch.inference_mode()
def audio2video(a2l_file, audio_file, stylegan_file, output_size=(512, 512), batch_size=8, offset=90, duration=60):
    a2l: Audio2Latent = joblib.load(a2l_file)["a2l"].eval().to(device)

    test_features = audio2features(*torchaudio.load(audio_file))[24 * offset : 24 * (offset + duration)].to(device)
    test_latents = a2l(test_features.unsqueeze(0)).squeeze()

    del a2l
    gc.collect()
    torch.cuda.empty_cache()

    mapper = StyleGAN2Mapper(model_file=stylegan_file, inference=False)
    latent_offset = mapper(torch.from_numpy(np.random.RandomState(42).randn(1, 512))).to(device)
    del mapper

    synthesizer = StyleGAN2Synthesizer(
        model_file=stylegan_file, inference=False, output_size=output_size, strategy="stretch", layer=0
    )
    synthesizer.eval().to(device)

    with VideoWriter(
        output_file=os.path.splitext(a2l_file)[0] + ".mp4",
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


def low_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 200, "low", fs=sr, output="sos"), audio)


def mid_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, [200, 2000], "band", fs=sr, output="sos"), audio)


def high_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 2000, "high", fs=sr, output="sos"), audio)


def audio2features(audio, sr):
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
            sum([list(torch.split(features[start:], DUR * FPS)[:-1]) for start in range(0, DUR * FPS, FPS)], [])
        )

        try:
            latents = joblib.load(f"{filename}.npy").float()
        except:
            latents = torch.from_numpy(np.load(f"{filename}.npy")).float()
        latents = torch.stack(
            sum([list(torch.split(latents[start:], DUR * FPS)[:-1]) for start in range(0, DUR * FPS, FPS)], [])
        )

        return features, latents


def collate(batch):
    return tuple(torch.cat(b) for b in list(map(list, zip(*batch))))


def preprocess_audio(in_dir, out_file):
    train_feat_file, train_lat_file = out_file.replace(".npy", "_train_feats.npy"), out_file.replace(
        ".npy", "_train_lats.npy"
    )
    val_feat_file, val_lat_file = out_file.replace(".npy", "_val_feats.npy"), out_file.replace(".npy", "_val_lats.npy")
    if not os.path.exists(train_lat_file):
        dataset = PreprocessAudioLatentDataset(in_dir)
        train_or_val = np.random.RandomState(42).rand(len(dataset)) < 0.8
        with NumpyArray(train_feat_file) as train_feats, NumpyArray(train_lat_file) as train_lats, NumpyArray(
            val_feat_file
        ) as val_feats, NumpyArray(val_lat_file) as val_lats:
            for i, (features, latents) in enumerate(
                tqdm(
                    DataLoader(dataset, batch_size=1, num_workers=16, collate_fn=collate),
                    desc="Preprocessing...",
                )
            ):
                feats = train_feats if train_or_val[i] else val_feats
                lats = train_lats if train_or_val[i] else val_lats
                for feat, lat in zip(features, latents):
                    feats.append(feat.unsqueeze(0).contiguous().numpy())
                    lats.append(lat.unsqueeze(0).contiguous().numpy())

        feats_train = np.load(train_feat_file, mmap_mode="r")
        np.save(out_file.replace(".npy", "_train_feats_mean.npy"), np.mean(feats_train, axis=(0, 1)))
        np.save(out_file.replace(".npy", "_train_feats_std.npy"), np.std(feats_train, axis=(0, 1)))

        lats_train = np.load(train_lat_file, mmap_mode="r")
        np.save(out_file.replace(".npy", "_train_lats_offsets.npy"), np.mean(lats_train, axis=(1, 2)))

        if os.path.exists(val_feat_file):
            feats_val = np.load(val_feat_file, mmap_mode="r")
            np.save(out_file.replace(".npy", "_val_feats_mean.npy"), np.mean(feats_val, axis=(0, 1)))
            np.save(out_file.replace(".npy", "_val_feats_std.npy"), np.std(feats_val, axis=(0, 1)))

            lats_val = np.load(val_lat_file, mmap_mode="r")
            np.save(out_file.replace(".npy", "_val_lats_offsets.npy"), np.mean(lats_val, axis=(1, 2)))


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
    print("name".ljust(25), "shape".ljust(25), "num".ljust(25))
    print("-" * 70)
    total = 0
    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        print(name.ljust(25), f"{tuple(param.shape)}".ljust(25), f"{num}".ljust(25))
    print("-" * 70)
    print("total".ljust(25), f"".ljust(25), f"{total/1e6:.2f} M".ljust(25))
    print()


class AudioLatentDataset(td.Dataset):
    def __init__(self, file, split="train"):
        super().__init__()
        try:
            self.features = np.load(file.replace(".npy", f"_{split}_feats.npy"), mmap_mode="r")
            self.mean = np.load(file.replace(".npy", f"_{split}_feats_mean.npy"), mmap_mode="r")
            self.std = np.load(file.replace(".npy", f"_{split}_feats_std.npy"), mmap_mode="r")
            self.latents = np.load(file.replace(".npy", f"_{split}_lats.npy"), mmap_mode="r")
            self.offsets = np.load(file.replace(".npy", f"_{split}_lats_offsets.npy"), mmap_mode="r")
        except FileNotFoundError:
            self.features = np.empty((0, 0, 0))
            self.mean = np.empty((0, 0, 0))
            self.std = np.empty((0, 0, 0))
            self.latents = np.empty((0, 0, 0, 0))
            self.offsets = np.empty((0, 0, 0, 0))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        features = (features - self.mean) / self.std

        latents = self.latents[index]
        latents = latents - self.offsets[index]
        return torch.from_numpy(features.copy()), torch.from_numpy(latents.copy())


class Audio2Latent(torch.nn.Module):
    def __init__(self, backbone, input_size, hidden_size, num_layers, dropout, n_outputs, output_size) -> None:
        super().__init__()
        self.n_outputs = n_outputs

        if backbone.lower() == "gru":
            self.backbone = torch.nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
            )
            self.backbone.flatten_parameters()
        else:
            raise NotImplementedError()

        self.backbone_skip = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_size // 2, hidden_size // 2),
            torch.nn.LeakyReLU(0.2),
        )

        self.relu = torch.nn.LeakyReLU(0.2)

        dense = {}
        for n in range(n_outputs):
            dense[f"{n:02}_weight1"], dense[f"{n:02}_bias1"] = self.init_weights(
                hidden_size + hidden_size // 2, hidden_size * 2
            )
            dense[f"{n:02}_weight2"], dense[f"{n:02}_bias2"] = self.init_weights(hidden_size * 2, output_size)
        self.dense = torch.nn.ParameterDict(dense)

    def init_weights(self, in_features, out_features):
        weight = torch.empty((in_features, out_features))
        torch.nn.init.kaiming_uniform_(weight, a=np.sqrt(5))

        bias = torch.empty(out_features)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(bias, -bound, bound)

        return torch.nn.Parameter(weight), torch.nn.Parameter(bias)

    def forward(self, x):
        w, _ = self.backbone(x)
        wx = torch.cat((self.relu(w), self.backbone_skip(x)), axis=2)

        w_plus = []
        for n in range(self.n_outputs):
            ww = self.relu(torch.matmul(wx, self.dense[f"{n:02}_weight1"]) + self.dense[f"{n:02}_bias1"])
            www = torch.matmul(ww, self.dense[f"{n:02}_weight2"]) + self.dense[f"{n:02}_bias2"]
            w_plus.append(www)
        w_plus = torch.stack(w_plus, axis=2)

        return w_plus


if __name__ == "__main__":
    in_dir = "/home/hans/datasets/audio2latent/"
    out_file = f"cache/{Path(in_dir).stem}_preprocessed.npy"
    test_audio = "/home/hans/datasets/audio2latent/Frequent @ Red Rocks - Cosmic Tuba Remix.wav"

    n_frames = int(DUR * FPS)
    batch_size = 16

    backbone = "gru"
    hidden_size = 32
    num_layers = 8
    dropout = 0.2

    lr = 1e-4

    preprocess_audio(in_dir, out_file)
    train_dataset = AudioLatentDataset(out_file, "train")
    val_dataset = AudioLatentDataset(out_file, "val")
    print_data_summary(train_dataset, val_dataset)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle=True, drop_last=True)
    inputs, targets = next(iter(dataloader))
    feature_dim = inputs.shape[2]
    n_outputs, output_size = targets.shape[2], targets.shape[3]

    a2l = Audio2Latent(
        backbone=backbone,
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_outputs=n_outputs,
        output_size=output_size,
    ).to(device)
    # a2l = torch.cuda.make_graphed_callables(a2l, (torch.randn_like(inputs.to(device)),))
    print_model_summary(a2l)

    optimizer = torch.optim.Adam(a2l.parameters(), lr=lr)

    writer = SummaryWriter()
    shutil.copy(__file__, writer.log_dir)

    n_iter = 0
    n_epochs = 1000
    video_interval = 20
    eval_interval = 5
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        losses = []
        for inputs, targets in dataloader:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = a2l(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), n_iter)
            losses.append(loss.item())

            n_iter += len(inputs)
        train_loss = np.mean(losses)

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == n_epochs:
            with torch.inference_mode():
                if len(val_dataset) > 0:
                    a2l.eval()
                    val_dataloader = DataLoader(
                        val_dataset, batch_size=batch_size, num_workers=24, shuffle=True, drop_last=True
                    )
                    val_loss = 0
                    for inputs, targets in val_dataloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = a2l(inputs)
                        val_loss += torch.nn.functional.mse_loss(outputs, targets)
                    val_loss /= len(val_dataloader)
                    a2l.train()
                else:
                    val_loss = -1

            pbar.write("")
            pbar.write(f"epoch {epoch+1}")
            pbar.write(f"train_loss: {train_loss:.4f}")
            pbar.write(f"val_loss  : {val_loss:.4f}")
            pbar.write("")

        if (epoch + 1) % video_interval == 0 or (epoch + 1) == n_epochs:
            joblib.dump(
                {"a2l": a2l, "optim": optimizer, "n_iter": n_iter},
                f"{writer.log_dir}/audio2latent_steps{n_iter}_loss{val_loss:.4f}.pt",
                compress=9,
            )
            audio2video(
                a2l_file=f"{writer.log_dir}/audio2latent_steps{n_iter}_loss{val_loss:.4f}.mp4",
                audio_file=test_audio,
                stylegan_file="/home/hans/modelzoo/train_checks/neurout2-117.pt",
            )

    writer.close()
