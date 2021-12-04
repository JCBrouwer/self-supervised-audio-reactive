import os
from glob import glob
from itertools import chain
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
import torchaudio
import torchdatasets as td
import torchextractor as tx
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.patch_contrastive import PatchContrastor, PatchSampler1d, PatchSampler2d
from models.reactor import LSTMReactor
from models.stylegan2 import Generator

SR = 24575
DUR = 8
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


class AudioFeatureDataset(td.Dataset):
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

    batch_size = 16
    hidden_size = 32
    num_layers = 4
    zoneout = 0.1
    dropout = 0.05
    n_frames = int(DUR * FPS)
    patch_len = 32
    n_patches = 16
    use_video = False
    lambda_video = 0.0333

    preprocess_audio(in_dir, out_file)
    dataloader = DataLoader(
        AudioFeatureDataset(out_file).cache(td.cachers.Pickle(Path("./cache/audio_features/"))),
        batch_size=batch_size,
        num_workers=16,
        shuffle=True,
        drop_last=True,
    )

    generator = Generator(size=1024, style_dim=512, n_mlp=2)
    n_styles = generator.num_layers
    generator.load_state_dict(torch.load("cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pt")["g_ema"])
    generator.requires_grad_(False)
    del generator.style
    generator = generator.eval().to(device)
    layers_of_interest = ["input", "convs.1", "convs.3"]
    stop_early = int(layers_of_interest[-1].split(".")[-1])
    generator = tx.Extractor(generator, layers_of_interest)

    reactor = LSTMReactor(
        input_size=next(iter(dataloader)).shape[-1],
        hidden_size=hidden_size,
        zoneout=zoneout,
        dropout=dropout,
        n_styles=n_styles,
    )
    reactor = reactor.train().to(device)

    sequence_sampler = PatchSampler1d(n_patches=n_patches, patch_len=patch_len).to(device)
    sequence_contrastor = PatchContrastor(latent_dim=32).to(device)
    if use_video:
        video_sampler = PatchSampler2d(patch_size=16).to(device)
        video_contrastor = PatchContrastor(latent_dim=32).to(device)

    optimizer = torch.optim.Adam(chain(reactor.parameters(), sequence_contrastor.parameters()), lr=1e-4)

    writer = SummaryWriter()
    n_iter = 0
    losses = []
    for epoch in tqdm(range(10000)):
        for audio_features in dataloader:
            loss_value = 0
            optimizer.zero_grad()

            audio_features = audio_features.to(device)
            motion_seed = torch.randn((batch_size, hidden_size), device=device)

            w_seq, inter_l, inter_h = reactor(audio_features, motion_seed)

            seq_patches, audio_patches = sequence_sampler(
                [w_seq.flatten(2, 3)] + [l for l in inter_l], audio_features
            )  #  + [h for h in inter_h]

            loss = sequence_contrastor(seq_patches, audio_patches)
            loss.backward()
            writer.add_scalar("Loss/train", loss.item(), n_iter)
            losses.append(loss.item())

            if use_video:
                grads, loss_value = torch.empty_like(seq_patches[0]), 0
                for p, (w_patch, audio_patch) in enumerate(zip(seq_patches[0].split(1), audio_patches.split(1))):
                    (rgb, _), feats = generator(
                        [w_patch.reshape(n_patches * patch_len, n_styles, 512)],
                        input_is_latent=True,
                        stop_early=stop_early,
                    )
                    video_patches = [
                        video_sampler(feat.reshape(n_patches, patch_len, *feat.shape[1:]))[None]
                        for feat in list(feats.values()) + [rgb]
                    ]
                    loss = video_contrastor(video_patches, audio_patch) * lambda_video
                    loss_value += loss.item()
                    (grads[p],) = torch.autograd.grad(loss, w_patch)[0]
                writer.add_scalar("Loss/video", loss_value, n_iter)

                w_seq, inter_l, inter_h = reactor(audio_features, motion_seed)
                w_patches, audio_patches = sequence_sampler(w_seq.flatten(2, 3), audio_features)
                w_patches.backward(grads)

            optimizer.step()

            n_iter += len(audio_features)

        if (epoch + 1) % 50 == 0:
            with open("models/reactor.py") as reactor_py, open("models/patch_contrastive.py") as contrastive_py, open(
                "train.py"
            ) as train_py:
                joblib.dump(
                    {
                        "reactor": reactor.state_dict(),
                        "sequence_contrastor": sequence_contrastor.state_dict(),
                        "video_contrastor": video_contrastor.state_dict() if use_video else None,
                        "optim": optimizer,
                        "n_iter": n_iter,
                        "reactor_file": reactor_py.readlines(),
                        "contrastive_file": contrastive_py.readlines(),
                        "train_file": train_py.readlines(),
                    },
                    f"{writer.log_dir}/checkpoint_steps{n_iter}_loss{np.mean(losses[:1000]):.4f}.pt",
                    compress=9,
                )

    writer.close()
