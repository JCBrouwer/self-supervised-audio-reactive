import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from .supervised.data import FEATURE_NAMES, AudioFeatures, audio2features

np.set_printoptions(precision=3, suppress=True, linewidth=200)

if __name__ == "__main__":
    in_dir = "/home/hans/datasets/wavefunk/"
    dur = 8
    fps = 24

    full_mean_file, full_std_file = Path(in_dir) / "full_mean.npy", Path(in_dir) / "full_std.npy"
    if not os.path.exists(full_mean_file):
        features = torch.cat(
            [f.squeeze() for f in tqdm(DataLoader(AudioFeatures(in_dir, audio2features, dur, fps), num_workers=24))]
        )
        print(features.shape, "\n")

        print("raw")
        for col in range(features.shape[-1]):
            print(
                FEATURE_NAMES[col],
                f"{features[:, :, col].min().item():.4f}",
                f"{features[:, :, col].mean().item():.4f}",
                f"{features[:, :, col].max().item():.4f}",
            )
        print()

        full_mean = features.mean((0, 1))
        full_std = features.std((0, 1))
        np.save(full_mean_file, full_mean)
        np.save(full_std_file, full_std)
        norm_feats = (features - full_mean) / full_std

        print("normalized")
        for col in range(features.shape[-1]):
            print(
                FEATURE_NAMES[col],
                f"{norm_feats[:, :, col].min().item():.4f}",
                f"{norm_feats[:, :, col].mean().item():.4f}",
                f"{norm_feats[:, :, col].max().item():.4f}",
            )
        print()
    else:
        full_mean = np.load(full_mean_file)
        full_std = np.load(full_std_file)

    test_audio = "/home/hans/datasets/wavefunk/naamloos.wav"
    audio, sr = torchaudio.load(test_audio)
    features = audio2features(audio, sr, fps)
    norm_feats = (features - full_mean) / full_std
    feats = norm_feats.squeeze().cpu()

    for name, feat in zip(FEATURE_NAMES, feats.unbind(1)):
        plt.plot(feat.numpy(), alpha=0.1)
    plt.savefig("output/new_and_improved_norm_feats_all.pdf")
    plt.close()

    fig, ax = plt.subplots(features.shape[-1], 1, figsize=(8, feats.shape[1] * 2))
    for c, (name, feat) in enumerate(zip(FEATURE_NAMES, feats.unbind(1))):
        ax[c].plot(feat.numpy())
        ax[c].set_ylabel(name)
    plt.tight_layout()
    plt.savefig("output/new_and_improved_norm_feat_by_feat.pdf")
    plt.close()

    # for name, feat in zip(FEATURE_NAMES, feats.unbind(1)):
    #     plt.plot(feat[15 * fps : 35 : fps].numpy(), alpha=0.1)
    # plt.savefig("output/new_and_improved_norm_feats_all_snippet.pdf")
    # plt.close()

    # fig, ax = plt.subplots(features.shape[-1], 1, figsize=(8, feats.shape[1] * 2))
    # for c, (name, feat) in enumerate(zip(FEATURE_NAMES, feats.unbind(1))):
    #     ax[c].plot(feat[15 * fps : 35 : fps].numpy())
    #     ax[c].set_ylabel(name)
    # plt.tight_layout()
    # plt.savefig("output/new_and_improved_norm_feat_by_feat_snippet.pdf")
    # plt.close()
