# fmt:off
import os
import sys
from glob import glob
from pathlib import Path
from time import time

import decord
import joblib
import numpy as np
import pandas as pd
import torch
import torchdatasets as td
from resize_right import resize
from torch.utils.data import DataLoader
from torchaudio.functional.functional import resample
from tqdm import tqdm

from .features.audio import (chromagram, drop_strength, mfcc, onsets, pulse,
                             rms, spectral_contrast, spectral_flatness,
                             tonnetz)
from .features.correlation import orthogonal_procrustes_distance
from .features.rosa.segment import laplacian_segmentation_rosa
from .features.video import (absdiff, adaptive_freq_rms, directogram,
                             high_freq_rms, hsv_hist, low_freq_rms,
                             mid_freq_rms, rgb_hist, video_flow_onsets,
                             video_spectral_onsets, video_spectrogram,
                             visual_variance)
from .optimize import FixedLatentNoiseDecoder, HiPPOTimeseries, autocorrelation
from .random.mir import retrieve_music_information
from .random.patch import Patch
from .test import load_model
from .train import STYLEGAN_CKPT, audio_reactive_loss, normalize_gradients

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2

AFNS = [chromagram, tonnetz, mfcc, spectral_contrast, rms, drop_strength, onsets, spectral_flatness, pulse]
VFNS = [rgb_hist, hsv_hist, video_spectrogram, directogram, low_freq_rms, mid_freq_rms, high_freq_rms, 
        adaptive_freq_rms, absdiff, visual_variance, video_flow_onsets, video_spectral_onsets]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS = 24
OUT_SIZE = (1024, 1024)
SG2 = StyleGAN2(
    model_file=STYLEGAN_CKPT, inference=False, output_size=OUT_SIZE, strategy="stretch", layer=0
).eval().to(DEVICE)
M = SG2.mapper
G = SG2.synthesizer
# fmt:on

decord.bridge.set_bridge("torch")


class RandomGenerator:
    def predict(self, audio, sr):
        features, segmentations, tempo = retrieve_music_information(audio, sr)
        patch = Patch(
            features=features,
            segmentations=segmentations,
            tempo=tempo,
            seed=np.random.randint(0, 2 ** 32),
            fps=FPS,
            device=DEVICE,
        )
        z = torch.randn((180, 512), device=DEVICE)
        latent_palette = M(z)
        return patch.forward(latent_palette)


class SupervisedSequenceModel:
    def __init__(self, checkpoint, residual=False):
        self.model, self.a2f = load_model(checkpoint)
        self.model.to(DEVICE)
        try:
            self.model.envelope.backbone.flatten_parameters()
        except:
            pass
        self.residual = residual

    def predict(self, audio, sr):
        latents, noise = self.model(self.a2f(audio.unsqueeze(0), sr, fps=FPS).unsqueeze(0).to(DEVICE))
        if self.residual:
            z = torch.randn((1, 512), device=DEVICE)
            latents = latents + M(z)
        return latents.squeeze(), [n.squeeze() for n in noise]


class SelfSupervisedOptimization:
    def predict(
        self,
        audio,
        sr,
        n_steps=512,
        n_params=2048,
        n_latent_split=3,
        n_latent_groups=3,
        n_latent_per_group=1,
        n_noise=6,
        lr=1e-3,
        use_audio_segmentation_features=True,
        use_env=True,
        feature_weights=None,
    ):
        # extract features from audio
        features = {afn.__name__: afn(audio, sr).to(DEVICE) for afn in AFNS}
        n_frames = features["rms"].shape[0]
        if feature_weights is not None:
            for name, feature in features.items():
                ac = autocorrelation(feature)
                ac -= ac.min()
                ac /= ac.max()
                feature_weights[name] = 1 / ac.mean()
        ks = [2, 4, 6, 8, 12, 16]
        if use_audio_segmentation_features:
            segmentations = laplacian_segmentation_rosa(audio.numpy(), sr, n_frames, ks=ks)
            features[f"rosa_segmentation"] = segmentations.float().to(DEVICE)
            if feature_weights is not None:
                feature_weights[f"rosa_segmentation"] = torch.max(torch.tensor(list(feature_weights.values())))

        # initialize modules
        n_envelopes = n_latent_split * n_latent_groups * n_latent_per_group + 2 * n_noise
        envelopes = HiPPOTimeseries(torch.rand((n_frames, n_envelopes), device=DEVICE), N=n_params).to(DEVICE)
        decoder_latents = M(torch.randn((180, 512), device=DEVICE)).to(DEVICE)
        decoder = FixedLatentNoiseDecoder(decoder_latents, n_latent_split, n_latent_groups, n_latent_per_group)

        # initialize optimization
        optimizer = torch.optim.Adam(envelopes.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr / 100)

        for _ in range(n_steps):
            optimizer.zero_grad()
            envs = envelopes()
            latents, noise = decoder(envs)
            predictions = (
                ([normalize_gradients(envs).flatten(1)] if use_env else [])
                + [normalize_gradients(latents).flatten(1)]
                + [normalize_gradients(n, 1 / len(noise).flatten(1)) for n in noise]
            )

            if feature_weights is not None:
                loss = torch.zeros((), device=DEVICE)
                for p in predictions:
                    for n, f in features.items():
                        loss = loss + feature_weights[n] * orthogonal_procrustes_distance(p, f)

            else:
                preds = torch.stack(predictions, dim=1)
                feats = torch.stack(list(features.values()), dim=1)
                loss = orthogonal_procrustes_distance(preds, feats)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        return decoder(envelopes())


def load_audio(path, fps=24):
    a = decord.AudioReader(path)
    audio = a[:].float().squeeze().contiguous()
    old_sr = round(len(audio) / a.duration())
    new_sr = round(1024 * fps)
    audio = resample(audio, old_sr, new_sr).contiguous()
    return audio, new_sr


class AudioDataset(td.Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        return filepath, load_audio(filepath)


TESTSET = AudioDataset(glob("/home/hans/datasets/audiovisual/maua/*.mp4"))
TESTSET.cache(td.cachers.Pickle(Path("cache/")))
for _ in tqdm(DataLoader(TESTSET, num_workers=8), total=len(TESTSET), desc="Caching test set..."):
    pass


@torch.no_grad()
def evaluate_trained_checkpoint_dirs(dirs):
    try:
        results = joblib.load("output/learned_checkpoint_comparison.pkl")
        jobs_done = len(results)
    except:
        results = []
        jobs_done = 0
    j = 0
    for ckdir in tqdm(dirs):
        backbone, loss, decoder, _, residual, _, _, split, _, hidden, _, layers, dropout, lr = ckdir.split("_")[3:]
        ckpts = sorted(glob(f"{ckdir}/*.pt"))
        ckpts = [ckpts[idx] for idx in np.linspace(0, len(ckpts) - 1, 5).round().astype(int)]
        for ckpt in tqdm(ckpts):
            steps, val = Path(ckpt).stem.split("_")[-2:]
            t = time()
            for (filepath,), ((audio,), sr) in tqdm(DataLoader(TESTSET, num_workers=8)):
                if j < jobs_done:
                    j += 1
                    print(ckdir, ckpt, filepath, "already done, skipping...")
                    continue
                print("load", time() - t)
                t = time()

                latents, noise = SupervisedSequenceModel(ckpt, residual="residual:True" in ckpt).predict(
                    audio, sr.item()
                )
                noise = sum([[noise[0]]] + [[n, n] for n in noise[1:]], [])
                inputs = {"latents": latents, **{f"noise{j}": n.unsqueeze(1) for j, n in enumerate(noise)}}
                print("sequence", time() - t)
                t = time()

                video = torch.cat(list(SG2.render(inputs, postprocess_fn=lambda x: resize(x, out_shape=(128, 128)))))
                print("render", time() - t)
                t = time()

                vfeats = {vf.__name__: vf(video).unsqueeze(0) for vf in VFNS}
                print("vfeats", time() - t)
                t = time()

                if not os.path.exists(filepath.replace(".mp4", "_afeats.npz")):
                    afeats = {af.__name__: af(audio, sr.item()).unsqueeze(0).cuda() for af in AFNS}
                    np.savez_compressed(
                        filepath.replace(".mp4", "_afeats.npz"), **{n: f.cpu().numpy() for n, f in afeats.items()}
                    )
                else:
                    with np.load(filepath.replace(".mp4", "_afeats.npz")) as arr:
                        afeats = {af.__name__: torch.from_numpy(arr[af.__name__]).cuda() for af in AFNS}
                print("afeats", time() - t)
                t = time()

                results.append(
                    dict(
                        filepath=filepath,
                        steps=int(steps.replace("steps", "")),
                        val=float(val.replace("val", "")),
                        backbone=backbone.split(":")[-1],
                        loss=loss.split(":")[-1],
                        decoder=decoder.split(":")[-1],
                        residual=bool(residual.split(":")[-1]),
                        split=int(split.split(":")[-1]),
                        hidden=int(hidden.split(":")[-1]),
                        layers=int(layers.split(":")[-1]),
                        dropout=float(dropout.split(":")[-1]),
                        lr=float(lr.split(":")[-1]),
                        correlation=(1 - audio_reactive_loss(afeats, vfeats)).squeeze().item(),
                        **{
                            an + "|" + vn: (1 - audio_reactive_loss([af], [vf])).squeeze().item()
                            for vn, vf in vfeats.items()
                            for an, af in afeats.items()
                        },
                    )
                )
                print("results", time() - t)
                print(results[-1])
                t = time()
                joblib.dump(results, "output/learned_checkpoint_comparison.pkl")

    results = pd.DataFrame(results)
    results.to_csv("output/learned_checkpoint_comparison.csv")
    return results


@torch.no_grad()
def compare_big_three(seqckpts, n_samples=10):
    results = []
    for model_name, model in [
        *[lambda: SupervisedSequenceModel(ckpt, residual="residual:True" in ckpt) for ckpt in seqckpts],
        RandomGenerator,
        SelfSupervisedOptimization,
    ]:
        for _ in range(n_samples):
            for (filepath,), ((audio,), sr) in tqdm(DataLoader(TESTSET, num_workers=8)):
                latents, noise = model().predict(audio, sr.item())

                noise = sum([[noise[0]]] + [[n, n] for n in noise[1:]], [])
                inputs = {"latents": latents, **{f"noise{j}": n.unsqueeze(1) for j, n in enumerate(noise)}}
                video = torch.cat(list(SG2.render(inputs, postprocess_fn=lambda x: resize(x, out_shape=(256, 256)))))

                vfeats = {vf.__name__: vf(video).unsqueeze(0) for vf in VFNS}

                if not os.path.exists(filepath.replace(".mp4", "_afeats.npz")):
                    audio = audio.cuda()
                    afeats = {af.__name__: af(audio.cpu(), sr.item()).unsqueeze(0).cuda() for af in AFNS}
                    np.savez_compressed(
                        filepath.replace(".mp4", "_afeats.npz"), **{n: f.cpu().numpy() for n, f in afeats.items()}
                    )
                else:
                    with np.load(filepath.replace(".mp4", "_afeats.npz")) as arr:
                        afeats = {af.__name__: torch.from_numpy(arr[af.__name__]).cuda() for af in AFNS}

                results.append(
                    dict(
                        filepath=filepath,
                        model_name=model_name,
                        correlation=(1 - audio_reactive_loss(afeats, vfeats)).squeeze().item(),
                        **{
                            an + "|" + vn: (1 - audio_reactive_loss([af], [vf])).squeeze().item()
                            for (an, af), (vn, vf) in zip(afeats.items(), vfeats.items())
                        },
                    )
                )
                print(results[-1])
    results = pd.DataFrame(results)
    results.to_csv("output/big3_comparison.csv")
    return results


if __name__ == "__main__":
    results = evaluate_trained_checkpoint_dirs(
        [
            "runs/May23_17-01-54_ubuntu94025_gru_selfsupervised_fixed_decoder_residual:False_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001",
            "runs/May23_16-43-01_ubuntu94025_gru_selfsupervised_fixed_decoder_residual:True_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001",
            "runs/May23_16-24-29_ubuntu94025_gru_selfsupervised_learned_decoder_residual:True_n_latent_split:3_hidden_size:16_num_layers:4_dropout:0.0_lr:0.0001",
            "runs/May23_16-05-58_ubuntu94025_gru_selfsupervised_learned_decoder_residual:False_n_latent_split:3_hidden_size:16_num_layers:4_dropout:0.0_lr:0.0001",
            "runs/May23_15-41-49_ubuntu94025_gru_supervised_fixed_decoder_residual:False_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001",
            "runs/May23_15-27-32_ubuntu94025_gru_supervised_fixed_decoder_residual:True_n_latent_split:3_hidden_size:3_num_layers:4_dropout:0.0_lr:0.0001",
            "runs/May23_15-10-44_ubuntu94025_gru_supervised_learned_decoder_residual:True_n_latent_split:3_hidden_size:16_num_layers:4_dropout:0.0_lr:0.0001",
            "runs/May23_14-48-57_ubuntu94025_gru_supervised_learned_decoder_residual:False_n_latent_split:3_hidden_size:16_num_layers:4_dropout:0.0_lr:0.0001",
        ]
    )
    print(results)
