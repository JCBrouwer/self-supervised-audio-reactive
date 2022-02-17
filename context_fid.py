import os
from glob import glob
from time import time

import joblib
import numpy as np
import torch
import torchaudio
from sklearn import decomposition
from tqdm import tqdm

from train_supervised import audio2features
from usrlt import CausalCNNEncoderClassifier


class FullLengthAudioFeatureDataset(torch.utils.data.Dataset):
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

        try:
            latents = joblib.load(f"{filename}.npy").float()
        except:
            latents = torch.from_numpy(np.load(f"{filename}.npy")).float()

        return features, latents


if __name__ == "__main__":
    X = np.load("cache/audio2latent_preprocessed_train_lats.npy", mmap_mode="r")
    XO = np.load("cache/audio2latent_preprocessed_train_lats_offsets.npy", mmap_mode="r")
    B, T, N, L = X.shape

    # ipca_batch = 8
    # n_components = 64
    # pca = decomposition.IncrementalPCA(n_components=n_components)
    # index_batches = np.split(np.random.permutation(range(len(X))), len(X) // ipca_batch)
    # for idxs in tqdm(index_batches, unit_scale=ipca_batch):
    #     batch = (X[idxs] - XO[idxs, None, None]).astype(np.float32)
    #     batch = batch.reshape(ipca_batch * T, N * L)
    #     pca.partial_fit(batch)
    # train = []
    # for i in tqdm(range(0, X.shape[0], ipca_batch)):
    #     batch = (X[i : i + ipca_batch] - XO[i : i + ipca_batch]).astype(np.float32)
    #     batch = batch.reshape(ipca_batch * T, N * L)
    #     batch = pca.transform(batch)
    #     batch = batch.reshape(ipca_batch, T, n_components)
    #     batch = batch.transpose(0, 2, 1)
    #     train.append(batch)
    # train = np.concatenate(train)

    encoder = CausalCNNEncoderClassifier(
        compared_length=96,
        nb_random_samples=10,
        batch_size=16,
        nb_steps=200,
        in_channels=N * L,
        channels=128,
        out_channels=64,
        reduced_size=32,
        depth=10,
        cuda=True,
    )
    encoder.fit_encoder(X, XO, verbose=True)
    joblib.dump(encoder, "cache/encoder.pt", compress=9)

    # compared_length=50,
    # nb_random_samples=10,
    # negative_penalty=1,
    # batch_size=1,
    # nb_steps=2000,
    # lr=0.001,
    # penalty=1,
    # early_stopping=None,
    # channels=10,
    # depth=1,
    # reduced_size=10,
    # out_channels=10,
    # kernel_size=4,
    # in_channels=1,
    # cuda=False,
    # gpu=0,
