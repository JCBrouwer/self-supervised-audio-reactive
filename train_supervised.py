import os
import shutil
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


np.set_printoptions(precision=2, suppress=True)

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


def low_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 200, "low", fs=sr, output="sos"), audio)


def mid_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, [200, 2000], "band", fs=sr, output="sos"), audio)


def high_pass(audio, sr):
    return signal.sosfilt(signal.butter(12, 2000, "high", fs=sr, output="sos"), audio)


class PreprocessAudioLatentDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.files = rosa.util.find_files(directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        filename, _ = os.path.splitext(file)

        try:
            latents = joblib.load(f"{filename}.npy").float()
        except:
            latents = torch.from_numpy(np.load(f"{filename}.npy")).float()

        audio, sr = torchaudio.load(file)
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

        features = torch.from_numpy(
            np.concatenate(
                (mfcc, chroma, tonnetz, contrast, onsets, onsets_low, onsets_mid, onsets_high, pulse, volume, flatness),
                axis=1,
            )
        ).float()

        features = torch.stack(
            sum([list(torch.split(features[start:], DUR * FPS)[:-1]) for start in range(0, DUR * FPS, FPS)], [])
        )

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
    if not os.path.exists(val_lat_file):
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

        feats_val = np.load(val_feat_file, mmap_mode="r")
        np.save(out_file.replace(".npy", "_val_feats_mean.npy"), np.mean(feats_val, axis=(0, 1)))
        np.save(out_file.replace(".npy", "_val_feats_std.npy"), np.std(feats_val, axis=(0, 1)))

        lats_val = np.load(val_lat_file, mmap_mode="r")
        np.save(out_file.replace(".npy", "_val_lats_offsets.npy"), np.mean(lats_val, axis=(1, 2)))


class AudioLatentDataset(td.Dataset):
    def __init__(self, file, split="train"):
        super().__init__()
        self.features = np.load(file.replace(".npy", f"_{split}_feats.npy"), mmap_mode="r")
        self.mean = np.load(file.replace(".npy", f"_{split}_feats_mean.npy"), mmap_mode="r")
        self.std = np.load(file.replace(".npy", f"_{split}_feats_std.npy"), mmap_mode="r")
        self.latents = np.load(file.replace(".npy", f"_{split}_lats.npy"), mmap_mode="r")
        self.offsets = np.load(file.replace(".npy", f"_{split}_lats_offsets.npy"), mmap_mode="r")

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
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )

        self.relu = torch.nn.ReLU()

        dense = {}
        for n in range(n_outputs):
            dense[f"{n:02}_weight1"], dense[f"{n:02}_bias1"] = self.init_weights(hidden_size + 32, output_size // 2)
            dense[f"{n:02}_weight2"], dense[f"{n:02}_bias2"] = self.init_weights(output_size // 2, output_size)
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
        wx = torch.cat((w, self.backbone_skip(x)), axis=2)

        w_plus = []
        for n in range(self.n_outputs):
            ww = self.relu(torch.matmul(wx, self.dense[f"{n:02}_weight1"]) + self.dense[f"{n:02}_bias1"])
            www = torch.matmul(ww, self.dense[f"{n:02}_weight2"]) + self.dense[f"{n:02}_bias2"]
            w_plus.append(www)
        w_plus = torch.stack(w_plus, axis=2)

        return w_plus


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dir = "/home/hans/datasets/audio2latent/"
    out_file = f"cache/{Path(in_dir).stem}_preprocessed.npy"

    n_frames = int(DUR * FPS)
    batch_size = 16

    backbone = "gru"
    hidden_size = 128
    num_layers = 8
    dropout = 0.1

    lr = 1e-4

    preprocess_audio(in_dir, out_file)
    train_dataset = AudioLatentDataset(out_file, "train")
    val_dataset = AudioLatentDataset(out_file, "val")

    print()
    print("data summary:")
    print("train audio feature sequences:", train_dataset.features.shape)
    print("train latent sequences:", train_dataset.latents.shape)
    print("val audio feature sequences:", val_dataset.features.shape)
    print("val latent sequences:", val_dataset.latents.shape)
    print("train %:", int(100 * len(train_dataset) / (len(train_dataset) + len(val_dataset))))
    print()

    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, drop_last=True)
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
    a2l = torch.cuda.make_graphed_callables(a2l, (torch.randn_like(inputs.to(device)),))

    print()
    print("model summary:")
    print("name".ljust(25), "shape".ljust(25), "num".ljust(25))
    print("-" * 70)
    total = 0
    for name, param in a2l.named_parameters():
        num = param.numel()
        total += num
        print(name.ljust(25), f"{tuple(param.shape)}".ljust(25), f"{num}".ljust(25))
    print("-" * 70)
    print("total".ljust(25), f"".ljust(25), f"{total/1e6:.2f} M".ljust(25))
    print()

    optimizer = torch.optim.Adam(a2l.parameters(), lr=lr)

    writer = SummaryWriter()
    shutil.copy(__file__, writer.log_dir)
    n_iter = 0
    pbar = tqdm(range(10))
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

        with torch.inference_mode():
            a2l.eval()
            val_loss = 0
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, num_workers=16, shuffle=True, drop_last=True
            )
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = a2l(inputs)
                val_loss += torch.nn.functional.mse_loss(outputs, targets)
            val_loss /= len(val_dataloader)
            a2l.train()

        pbar.write("")
        pbar.write(f"epoch {epoch+1}")
        pbar.write(f"train_loss: {train_loss:.4f}")
        pbar.write(f"val_loss  : {val_loss:.4f}")
        pbar.write("")

    joblib.dump(
        {"a2l": a2l, "optim": optimizer, "n_iter": n_iter},
        f"{writer.log_dir}/checkpoint_steps{n_iter}_loss{val_loss:.4f}.pt",
        compress=9,
    )

    writer.close()
