import os
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import torchextractor as tx
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from losses import PatchNCELoss
from models.reactor import LSTMReactor
from models.stylegan2 import Generator

SR = 24575
DUR = 4
FPS = 24


class PreprocessAudioDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        extensions = [".wav", ".mp3", ".flac"]
        self.files = sum([glob(directory + "/*" + ext) for ext in extensions], [])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio, sr = torchaudio.load(self.files[index])
        audio = audio.mean(0)
        audio = torchaudio.transforms.Resample(sr, SR)(audio)
        batch = torch.split(audio, DUR * SR)
        return torch.stack(batch[:-1]).float()


def collate_fn(batch):
    batch = torch.cat(batch)
    return batch


def preprocess_audio(in_dir, out_file):
    if not os.path.exists(out_file):
        snippets = [
            batch
            for batch in DataLoader(PreprocessAudioDataset(in_dir), batch_size=4, num_workers=4, collate_fn=collate_fn)
        ]
        snippets = torch.cat(snippets).numpy()
        np.save(out_file, snippets)


class AudioFeatureDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.snippets = np.load(file, mmap_mode="r")

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, index):
        audio = self.snippets[index]

        mfcc = torch.from_numpy(librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=19, hop_length=1024)).permute(1, 0)
        mfcc /= torch.norm(mfcc)

        chroma = torch.from_numpy(librosa.feature.chroma_cens(y=audio, sr=SR, hop_length=1024)).permute(1, 0)
        chroma /= torch.norm(chroma)

        onsets = torch.from_numpy(librosa.onset.onset_strength(y=audio, sr=SR, hop_length=1024))[:, None]
        onsets /= torch.norm(onsets)

        return torch.cat((mfcc, chroma, onsets), axis=1).float()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dir = "/home/hans/datasets/wavefunk/"
    out_file = f"cache/{Path(in_dir).stem}_snippets_preprocessed.npy"

    batch_size = 8
    hidden_size = 32
    num_layers = 4
    zoneout = 0.1
    dropout = 0.05
    n_frames = int(DUR * FPS)
    patch_len = 8
    n_patches = 16

    preprocess_audio(in_dir, out_file)
    dataloader = DataLoader(AudioFeatureDataset(out_file), batch_size=batch_size, num_workers=4, shuffle=True)

    generator = Generator(size=1024, style_dim=512, n_mlp=2)
    n_styles = generator.num_layers
    generator.load_state_dict(torch.load("cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pt")["g_ema"])
    generator.requires_grad_(False)
    del generator.style
    generator = generator.eval().to(device)
    layers_of_interest = ["input", "convs.1", "convs.3"]  # , "convs.5"]
    stop_early = int(layers_of_interest[-1].split(".")[-1])
    generator = tx.Extractor(generator, layers_of_interest)

    reactor = (
        LSTMReactor(
            input_size=next(iter(dataloader)).shape[-1],
            hidden_size=hidden_size,
            zoneout=zoneout,
            dropout=dropout,
            n_styles=n_styles,
        )
        .train()
        .to(device)
    )

    optimizer = torch.optim.Adam(reactor.parameters(), lr=1e-3)

    class Reshape(nn.Module):
        def __init__(self, *shape):
            super().__init__()
            self.shape = shape

        def forward(self, x):
            return x.reshape(*self.shape)

    contrastive_heads = nn.ModuleList(
        [
            nn.Sequential(
                Reshape(batch_size * n_frames, -1),
                nn.Linear(n_styles * 512, n_styles * 256),
                nn.LeakyReLU(0.2),
                nn.Linear(n_styles * 256, n_styles * 128),
                nn.LeakyReLU(0.2),
                nn.Linear(n_styles * 128, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 32),
                Reshape(batch_size, n_frames, -1),
            ),
            nn.Sequential(
                Reshape(batch_size * n_frames, -1),
                nn.Linear(32, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, 32),
                nn.LeakyReLU(0.2),
                Reshape(batch_size, n_frames, -1),
            ),
        ]
    ).cuda()

    for audio_features in dataloader:
        print()
        print(audio_features.shape)
        audio_features = audio_features.to(device)
        motion_seed = torch.randn((batch_size, hidden_size), device=device)

        w_seq, inter_l, inter_h = reactor(audio_features, motion_seed)

        patch_idxs = torch.randint(0, n_frames - patch_len, (batch_size, n_patches))
        print(patch_idxs)
        print(patch_idxs.shape)
        print(patch_idxs[np.arange(batch_size)])
        print(patch_idxs[np.arange(batch_size)].shape)
        print(patch_idxs[:, np.arange(n_patches)])
        print(patch_idxs[:, np.arange(n_patches)].shape)
        w_patches = w_seq[patch_idxs : patch_idxs + 8]
        af_patches = audio_features[patch_idxs : patch_idxs + 8]
        print(w_patches.shape)

        loss = PatchNCELoss(contrastive_heads[0](w_patches), contrastive_heads[1](af_patches))

        optimizer.zero_grad()
        print(loss.item())
        loss.backward()
        optimizer.step()

        # features = (
        #     [w_seq.reshape(batch_size, n_frames, -1).cpu()]
        #     + list(inter_l.reshape(num_layers, batch_size, n_frames, -1).cpu().split(1))
        #     + list(inter_h.reshape(num_layers, batch_size, n_frames, -1).cpu().split(1))
        # )

        # inter_G = [[] for _ in range(len(layers_of_interest) + 1)]
        # for w_batch in w_seq.split(1, dim=1):
        #     (rgb, _), feats = generator([w_batch.squeeze()], input_is_latent=True, stop_early=stop_early)
        #     inter_G[0].append(rgb.reshape(batch_size, -1).cpu())
        #     for f, feat in enumerate(feats.values()):
        #         inter_G[f + 1].append(feat.reshape(batch_size, -1).cpu())
        # for i, iG in enumerate(inter_G):
        #     inter_G[i] = torch.stack(iG, dim=1)
        # features += inter_G

        # for feat in features:
        #     print(feat.squeeze().shape)

        exit()
