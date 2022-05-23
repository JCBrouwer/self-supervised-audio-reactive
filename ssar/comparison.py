# fmt:off
import sys

import torch
import torchaudio
from torchaudio.functional import resample
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
from .train import STYLEGAN_CKPT, normalize_gradients

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from ops.video import VideoWriter

AFNS = [chromagram, tonnetz, mfcc, spectral_contrast, rms, drop_strength, onsets, spectral_flatness, pulse]
VFNS = [rgb_hist, hsv_hist, video_spectrogram, directogram, low_freq_rms, mid_freq_rms, high_freq_rms, 
        adaptive_freq_rms, absdiff, visual_variance, video_flow_onsets, video_spectral_onsets]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS = 24
OUT_SIZE = (1024, 1024)
M = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False).eval().to(DEVICE)
G = StyleGAN2Synthesizer(
    model_file=STYLEGAN_CKPT, inference=False, output_size=OUT_SIZE, strategy="stretch", layer=0
).eval().to(DEVICE)
# fmt:on


@torch.no_grad()
def latent2mp4(
    audio_file,
    latents,
    noise,
    out_file,
    fps=FPS,
    batch_size=8,
    offset=0,
    duration=None,
):
    start_frame = int(fps * offset)
    end_frame = int(fps * (offset + duration)) if duration is not None else len(latents)

    noise = sum([[noise[0]]] + [[n, n] for n in noise[1:]], [])

    with VideoWriter(
        output_file=out_file,
        output_size=OUT_SIZE,
        fps=fps,
        audio_file=audio_file,
        audio_offset=offset,
        audio_duration=duration,
    ) as video:
        for i in tqdm(range(start_frame, end_frame, batch_size), unit_scale=batch_size):
            for frame in (
                G(
                    latents=latents[i : i + batch_size],
                    **{f"noise{j}": n[i : i + batch_size, None] for j, n in enumerate(noise)},
                )
                .add(1)
                .div(2)
            ):
                video.write(frame.unsqueeze(0))


class RandomGenerator:
    def predict(self, audio, sr, seed=None):
        features, segmentations, tempo = retrieve_music_information(audio, sr)
        patch = Patch(features=features, segmentations=segmentations, tempo=tempo, seed=seed, fps=FPS, device=DEVICE)
        z = torch.randn((180, 512), device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(seed))
        latent_palette = M(z)
        return patch.forward(latent_palette)


class SupervisedSequenceModel:
    def __init__(self, checkpoint, residual=False):
        self.model, self.a2f = load_model(checkpoint)
        self.residual = residual

    def predict(self, audio, sr, seed=None):
        latents, noise = self.model(self.a2f(audio, sr, fps=FPS).unsqueeze(0).to(DEVICE))
        if self.residual:
            z = torch.randn((1, 512), device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(seed))
            latents = latents + M(z)
        return latents, noise


class SelfSupervisedOptimization:
    def predict(
        self,
        audio,
        sr,
        seed=None,
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
        decoder_latents = M(
            torch.randn((180, 512), device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(seed))
        ).to(DEVICE)
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


def audio_reactive_correlation(afeats, vfeats):
    afeatvals, vfeatvals = list(afeats.values()), list(vfeats.values())
    af, vf = afeatvals[0], vfeatvals[0]
    for afeat in afeatvals:
        af = torch.cat((af, afeat), dim=1)
    for vfeat in vfeatvals:
        vf = torch.cat((vf, vfeat), dim=1)
    return 1 - orthogonal_procrustes_distance(af, vf).item()


if __name__ == "__main__":
    audio_file = ""

    audio, sr = torchaudio.load(audio_file)
    audio = audio.mean(0)
    audio = audio[: 40 * sr]
    audio = resample(audio, sr, 1024 * FPS)
    sr = 1024 * FPS

    for (filepath,), ((audio,), sr, (video,), _) in tqdm(DataLoader(AudioVisualDataset(files), num_workers=24)):
        audio, sr, video = audio.cuda(), sr.item(), video.cuda()

        if not os.path.exists(filepath.replace(".mp4", "_vfeats.npz")):
            audio, video = audio.cuda(), video.cuda()
            afeats = {af.__name__: af(audio, sr) for af in AFNS}
            vfeats = {vf.__name__: vf(video) for vf in VFNS}
            np.savez_compressed(
                filepath.replace(".mp4", "_afeats.npz"), **{n: f.cpu().numpy() for n, f in afeats.items()}
            )
            np.savez_compressed(
                filepath.replace(".mp4", "_vfeats.npz"), **{n: f.cpu().numpy() for n, f in vfeats.items()}
            )
        else:
            with np.load(filepath.replace(".mp4", "_afeats.npz")) as arr:
                afeats = {af.__name__: torch.from_numpy(arr[af.__name__]).cuda() for af in afns}
            with np.load(filepath.replace(".mp4", "_vfeats.npz")) as arr:
                vfeats = {vf.__name__: torch.from_numpy(arr[vf.__name__]).cuda() for vf in vfns}

        row = {"group": group}
        for cname, correlation in {c.__name__: c for c in [op, pwcca, rv2, smi, svcca]}.items():
            row = {**row, **audiovisual_correlation(afeats, vfeats, cname, correlation, quadratic=True)}
            row[("concat", "concat", cname)] = audiovisual_correlation(
                afeats, vfeats, cname, correlation, quadratic=False
            )
        results.append(row)
