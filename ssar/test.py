import importlib
import os
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path
from shutil import copy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .supervised.data import FEATURE_NAMES, AudioFeatures, audio2features
from .supervised.test import _audio2video, load_a2l
from .train import STYLEGAN_CKPT, audio_reactive_loss

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def feature_plots():
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


class NewModuleFromFile:
    def __init__(self, path, name) -> None:
        self.path = path
        self.name = name

    def recursive_import(self, name):
        module = importlib.import_module(name)
        importlib.reload(module)
        for k in module.__dict__:
            if not k.startswith("_"):
                try:
                    self.original_main_module[k] = deepcopy(getattr(sys.modules["__main__"], k))
                except:
                    pass
                setattr(sys.modules["__main__"], k, getattr(module, k))
                setattr(sys.modules[self.name], k, getattr(module, k))
        for k in dir(module):
            if not k.startswith("_"):
                try:
                    self.recursive_import(name + "." + k)
                except:
                    pass
        return module

    def __enter__(self):
        relative_module_list = self.path.split("/ssar")[-1].split("/")[:-1]
        if len(relative_module_list) > 0:
            relative_module_list[0] = "ssar"
        else:
            relative_module_list = ["ssar"]
        directory = os.path.dirname(self.path)
        try:
            if not os.path.exists(f"{directory}/{self.name}.py.bkp"):
                copy(f"{directory}/{self.name}.py", f"{directory}/{self.name}.py.bkp")
            with open(f"{directory}/{self.name}.py", "r") as f:
                text = f.read()
            text = text.replace("from ....", "from " + ".".join(relative_module_list[:-3]) + ".")
            text = text.replace("from ...", "from " + ".".join(relative_module_list[:-2]) + ".")
            text = text.replace("from ..", "from " + ".".join(relative_module_list[:-1]) + ".")
            text = text.replace("from .", "from " + ".".join(relative_module_list) + ".")
            with open(f"{directory}/{self.name}.py", "w") as f:
                f.write(text)
        except Exception as e:
            pass
        sys.path = [directory] + sys.path
        self.original_main_module = {}
        module = self.recursive_import(self.name)
        return module

    def __exit__(self, exc_type, exc_value, exc_tb):
        for k in self.original_main_module:
            setattr(sys.modules["__main__"], k, self.original_main_module[k])
        sys.path = sys.path[1:]
        self.recursive_import(self.name)


def load_model(path):
    """Load checkpoint using it's own saved code"""
    try:
        with NewModuleFromFile(f"{os.path.dirname(path)}/ssar", "ssar") as ssar:
            model = joblib.load(path)["model"].eval()
            a2f = ssar.supervised.data.audio2features
    except:
        model, a2f = load_a2l(path)  # old model
    return model, a2f


class TestFeatures(Dataset):
    def __init__(self, directory, a2f, dur, fps, files=None):
        super().__init__()
        if directory is None:
            assert files is not None, "Must pass either directory or files!"
            self.files = files
        else:
            self.files = sum(
                [glob(f"{directory}*.{ext}") for ext in ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]], []
            )
        self.a2f = a2f
        self.L = dur * fps
        self.fps = fps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        audio, sr = torchaudio.load(file)
        features = self.a2f(audio, sr, self.fps)

        # overlapping slices of DUR seconds
        feature_lists = [list(torch.split(features[start:], self.L)[:-1]) for start in range(0, self.L, self.L // 4)]
        features = torch.stack(sum(feature_lists, []))

        offset_lists = [
            list(self.L * np.arange(len(feature_lists[s])) + start)
            for s, start in enumerate(range(0, self.L, self.L // 4))
        ]
        offsets = torch.tensor(sum(offset_lists, []))

        assert len(offsets) == len(features)
        return features.cpu(), offsets, [file] * len(features)


@torch.inference_mode()
def generate_by_data_split():
    dur, fps = 8, 24
    batch_size = 32

    ss_fixed_dir = "/home/hans/code/selfsupervisedaudioreactive/runs/Apr20_13-26-43_ubuntu94025_selfsupervised_gru_fixed_decoder_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001"
    s_learned_dir = "/home/hans/code/selfsupervisedaudioreactive/runs/Apr19_13-58-58_ubuntu94025_supervised_gru_learned_decoder_n_latent_split:3_hidden_size:16_num_layers:4_dropout:0.0_lr:0.0001"
    s_fixed_dir = "/home/hans/code/selfsupervisedaudioreactive/runs/Apr19_12-53-42_ubuntu94025_supervised_gru_fixed_decoder_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001"
    old_learned_dir = "/home/hans/code/selfsupervisedaudioreactive/runs/earlier/___Feb21_15-59-39_ubuntu94025backbone:gru:skipFalse_layerwise:conv:6_hidden_size:64_num_layers:8_dropout:0.2_lr:0.0001_wd:0"

    with open("cache/audio2latent_192frames_train_files.txt", "r") as f:
        train_files = sorted(f.read().splitlines())
    with open("cache/audio2latent_192frames_val_files.txt", "r") as f:
        val_files = sorted(f.read().splitlines())
    test_files = list(
        filter(
            lambda f: not Path(f).stem.startswith("_"),
            sorted(glob("/home/hans/datasets/wavefunk/*.wav") + glob("/home/hans/datasets/wavefunk/*.flac")),
        )
    )

    print("train:", len(train_files))
    print("val:", len(val_files))
    print("test:", len(test_files))

    step_count = lambda f: int(f.split("steps")[-1].split("_")[0])
    sorted_checkpoints = lambda d: list(sorted(glob(f"{d}/*.pt"), key=step_count))

    for name, checkpoints in (
        ("oldsupervised,learned", sorted_checkpoints(old_learned_dir)),
        ("supervised,learned", sorted_checkpoints(s_learned_dir)),
        ("supervised,fixed", sorted_checkpoints(s_fixed_dir)),
        ("selfsupervised,fixed", sorted_checkpoints(ss_fixed_dir)),
    ):
        if not os.path.exists(f"cache/{name}_test_features.npz"):
            _, a2f = load_model(checkpoints[0])

            def load_features(files):
                all_features = [
                    (f.squeeze(), o.squeeze(), a)
                    for f, o, a in tqdm(DataLoader(TestFeatures(None, a2f, dur, fps, files=files), num_workers=24))
                ]
                features, offsets, audio_files = list(map(list, zip(*all_features)))
                features, offsets, audio_files = torch.cat(features), torch.cat(offsets), sum(audio_files, [])
                return features, offsets, audio_files

            train_features, train_offsets, train_audio_files = load_features(train_files)
            val_features, val_offsets, val_audio_files = load_features(val_files)
            test_features, test_offsets, test_audio_files = load_features(test_files)
            np.savez_compressed(
                f"cache/{name}_test_features.npz",
                train_features=train_features,
                train_offsets=train_offsets,
                train_audio_files=train_audio_files,
                val_features=val_features,
                val_offsets=val_offsets,
                val_audio_files=val_audio_files,
                test_features=test_features,
                test_offsets=test_offsets,
                test_audio_files=test_audio_files,
            )
    with np.load(f"cache/{name}_test_features.npz") as f:
        train_features = torch.from_numpy(f["train_features"])
        val_features = torch.from_numpy(f["val_features"])
        test_features = torch.from_numpy(f["test_features"])

    print(
        "supervision,decoder,iterations,train_latent_rv2,train_latent_rv2_std,train_noise_rv2,train_noise_rv2_std,train_envelope_rv2,train_envelope_rv2_std,val_latent_rv2,val_latent_rv2_std,val_noise_rv2,val_noise_rv2_std,val_envelope_rv2,val_envelope_rv2_std,test_latent_rv2,test_latent_rv2_std,test_noise_rv2,test_noise_rv2_std,test_envelope_rv2,test_envelope_rv2_std"
    )
    vid_idxs = {
        "train": np.random.permutation(len(train_features))[:8],
        "val": np.random.permutation(len(val_features))[:8],
        "test": np.random.permutation(len(test_features))[:8],
    }
    for name, checkpoints in (
        ("selfsupervised,fixed", sorted_checkpoints(ss_fixed_dir)[::3]),
        ("supervised,fixed", sorted_checkpoints(s_fixed_dir)[::3]),
        ("oldsupervised,learned", sorted_checkpoints(old_learned_dir)[::3]),
        ("supervised,learned", sorted_checkpoints(s_learned_dir)[::3]),
    ):
        with np.load(f"cache/{name}_test_features.npz") as f:
            train_features, train_audio_files, train_offsets = (
                torch.from_numpy(f["train_features"]),
                f["train_audio_files"],
                torch.from_numpy(f["train_offsets"]),
            )
            val_features, val_audio_files, val_offsets = (
                torch.from_numpy(f["val_features"]),
                f["val_audio_files"],
                torch.from_numpy(f["val_offsets"]),
            )
            test_features, test_audio_files, test_offsets = (
                torch.from_numpy(f["test_features"]),
                f["test_audio_files"],
                torch.from_numpy(f["test_offsets"]),
            )
        for ckpt in checkpoints:
            model, _ = load_model(ckpt)
            model.cuda()
            iters = int(Path(ckpt).stem.split("steps")[-1].split("_")[0])
            print(f"\n{name}", iters, sep=",")
            for split, features, audio_files, offsets in [
                ("train", train_features, train_audio_files, train_offsets),
                ("val", val_features, val_audio_files, val_offsets),
                ("test", test_features, test_audio_files, test_offsets),
            ]:
                for idx in vid_idxs[split]:
                    _audio2video(
                        a2l=model,
                        features=features[[idx]].cuda(),
                        audio_file=audio_files[idx][0],
                        offset=offsets[idx].item() / fps,
                        duration=dur,
                        out_file=f"output/test_vids/{name}_{iters}_{split}_{idx}_{Path(audio_files[idx][0]).stem}.mp4",
                        stylegan_file=STYLEGAN_CKPT,
                        output_size=(1024, 1024),
                    )

                latrv2s, noirv2s, envrv2s = [], [], []
                for feat in features.split(batch_size):
                    feat = feat.cuda()
                    outputs = model(feat)
                    if isinstance(outputs, tuple):
                        latent_residuals, noise = outputs
                        rv2loss = audio_reactive_loss([n.flatten(2) for n in noise], [feat])
                        noirv2s.append(rv2loss.cpu().numpy())
                    else:
                        latent_residuals = outputs

                    rv2loss = audio_reactive_loss([latent_residuals.flatten(2)], [feat])
                    latrv2s.append(rv2loss.cpu().numpy())

                    try:
                        envelopes = model(feat, return_envelopes=True)
                        rv2loss = audio_reactive_loss([envelopes], [feat])
                        envrv2s.append(rv2loss.cpu().numpy())
                    except:
                        pass
                latrv2s = np.concatenate(latrv2s)
                noirv2s = np.concatenate(noirv2s) if len(noirv2s) > 0 else []
                envrv2s = np.concatenate(envrv2s) if len(envrv2s) > 0 else []
                print(
                    f"{np.mean(latrv2s):.4f}",
                    f"{np.std(latrv2s):.4f}",
                    f"{np.mean(noirv2s):.4f}" if len(noirv2s) > 0 else -1,
                    f"{np.std(noirv2s):.4f}" if len(noirv2s) > 0 else -1,
                    f"{np.mean(envrv2s):.4f}" if len(envrv2s) > 0 else -1,
                    f"{np.std(envrv2s):.4f}" if len(envrv2s) > 0 else -1,
                    sep=",",
                )


@torch.inference_mode()
def generate_longform_vids():
    audio_file = "/home/hans/datasets/wavefunk/naamloos.wav"
    checkpoint_dirs = [
        "/home/hans/code/selfsupervisedaudioreactive/runs/Apr25_15-21-35_ubuntu94025_ssabsdiff_transformer_fixed_decoder_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.1_lr:0.0001",
        "/home/hans/code/selfsupervisedaudioreactive/runs/Apr25_14-21-11_ubuntu94025_ssabsdiff_gru_fixed_decoder_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.1_lr:0.0001",
        "/home/hans/code/selfsupervisedaudioreactive/runs/Apr20_13-26-43_ubuntu94025_selfsupervised_gru_fixed_decoder_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001",
        "/home/hans/code/selfsupervisedaudioreactive/runs/Apr19_13-58-58_ubuntu94025_supervised_gru_learned_decoder_n_latent_split:3_hidden_size:16_num_layers:4_dropout:0.0_lr:0.0001",
        "/home/hans/code/selfsupervisedaudioreactive/runs/Apr19_12-53-42_ubuntu94025_supervised_gru_fixed_decoder_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001",
        "/home/hans/code/selfsupervisedaudioreactive/runs/earlier/___Feb21_15-59-39_ubuntu94025backbone:gru:skipFalse_layerwise:conv:6_hidden_size:64_num_layers:8_dropout:0.2_lr:0.0001_wd:0",
    ]

    step_count = lambda f: int(f.split("steps")[-1].split("_")[0])
    sorted_checkpoints = lambda d: list(sorted(glob(f"{d}/*.pt"), key=step_count))

    checkpoints = [sorted_checkpoints(d) for d in checkpoint_dirs]
    checkpoints = [[cs[round(i)] for i in np.linspace(0, len(cs) - 1, 4)] for cs in checkpoints]

    audio, sr = torchaudio.load(audio_file)

    with tqdm(total=len(checkpoint_dirs) * 4) as pbar:
        for group in checkpoints:

            name = Path(group[0]).stem.split("_steps")[0]

            pbar.write(f"model {name}")
            _, a2f = load_model(group[0])
            features = a2f(audio, sr, fps=24).unsqueeze(0).cuda()

            for ckpt in group:
                iters = int(Path(ckpt).stem.split("steps")[-1].split("_")[0])
                pbar.write(f"steps {iters}")

                filebase = f"output/longform_test_vids/{name}_steps={iters}_{Path(audio_file).stem}"
                np.savez_compressed(f"{filebase}_features.npz", features.cpu().numpy())

                model, _ = load_model(ckpt)
                model.cuda()

                _audio2video(
                    a2l=model,
                    features=features,
                    audio_file=audio_file,
                    offset=0,
                    duration=None,
                    out_file=f"{filebase}.mp4",
                    stylegan_file=STYLEGAN_CKPT,
                    output_size=(1024, 1024),
                    save=f"{filebase}_latnoise.npz",
                )
                pbar.update()


if __name__ == "__main__":
    generate_longform_vids()
