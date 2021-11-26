import os
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SR = 44100
DUR = 4


class PreprocessDataset(Dataset):
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
        return torch.stack(batch[:-1])


def collate_fn(batch):
    batch = torch.cat(batch)
    return batch


def preprocess_audio(in_dir, out_file):
    if not os.path.exists(out_file):
        snippets = [
            batch for batch in DataLoader(PreprocessDataset(in_dir), batch_size=4, num_workers=4, collate_fn=collate_fn)
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

        mfcc = torch.from_numpy(librosa.feature.mfcc(y=audio, sr=SR)).permute(1, 0)
        chroma = torch.from_numpy(librosa.feature.chroma_cens(y=audio, sr=SR)).permute(1, 0)
        onsets = torch.from_numpy(librosa.onset.onset_strength(y=audio, sr=SR))[:, None]

        return torch.cat((mfcc, chroma, onsets), axis=1)


if __name__ == "__main__":
    in_dir = "/home/hans/datasets/wavefunk/"
    out_file = f"{Path(in_dir).stem}_snippets_preprocessed.npy"
    preprocess_audio(in_dir, out_file)

    dataloader = DataLoader(AudioFeatureDataset(out_file), batch_size=8, num_workers=4, shuffle=True)
    for sample in tqdm(dataloader, smoothing=0):
        pass
