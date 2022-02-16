import matplotlib

matplotlib.use("Agg")

import random
import sys
from glob import glob
from pathlib import Path
from time import time
from uuid import uuid4

import joblib
import librosa as rosa
import librosa.display as rosa_display
import matplotlib.pyplot as plt
import numpy as np
import torch
from npy_append_array import NpyAppendArray as NpArr
from scipy.signal import resample
from tqdm import trange

# fmt:off
sys.path.append("/home/hans/code/maua/maua")
from audiovisual.audioreactive.audio import (band_pass, harmonic, high_pass,
                                             load_audio, low_pass, percussive,
                                             unmixed)
from audiovisual.audioreactive.features import (chroma, onsets, pitch_track,
                                                pulse, spectral_max, tempo,
                                                tonnetz, volume)
from audiovisual.audioreactive.postprocess import (compress, expand,
                                                   gaussian_filter,
                                                   percentile_clip)
from audiovisual.patches.primitives import (ModulatedLatents, ModulatedNoise,
                                            ModulationSum, PitchTrackLatents,
                                            TempoLoopLatents, TempoLoopNoise,
                                            TonalLatents, TonalNoise)
from GAN.wrappers.stylegan2 import StyleGAN2Mapper, StyleGAN2Synthesizer
from ops.video import VideoWriter

# fmt:on

# just randomly select from my other scripts, rather than full combinatorial

# drop merging difficulty, just use simple random patches
# export the intermediate steps of my own patches as "good" population
# compare the populations with the evaluation metrics


def identity(x, *args, **kwargs):
    return x


class RandomPatchPartial(torch.nn.Module):
    fn = lambda x: None

    def __init__(self, random_kwargs, **kwargs) -> None:
        super().__init__()
        self.kwargs = random_kwargs
        print(self.__class__.__name__, random_kwargs)
        for name, var in kwargs.items():
            setattr(self, name, var)

    @classmethod
    def randomize(cls):
        random_kwargs, regular_kwargs = {}, {}
        for name, var in cls.__dict__.items():
            if isinstance(var, list) and len(var) > 0:
                random_kwargs[name] = random.choice(var)
            else:
                regular_kwargs[name] = var
        return cls(random_kwargs, **regular_kwargs)

    def extra_repr(self):
        return "\n".join(
            ([self.fn.__name__] if self.fn.__name__ != "<lambda>" else [])
            + [
                f"{k}:{v}"
                for k, v in self.__dict__.items()
                if k not in ["fn", "training"] and not k.startswith("_") and not repr(v).startswith("<function")
            ]
        )

    def forward(self, x, **kwargs):
        if self.fn.__name__ == "select_tempo":
            return self.fn(self, x, **kwargs, **self.kwargs)
        return self.fn(x, **kwargs, **self.kwargs)


class AudioSource(RandomPatchPartial):
    fn = identity


class Harmonic(AudioSource):
    fn = harmonic
    margin = [1, 2, 4, 8, 16]


class Percussive(AudioSource):
    fn = percussive
    margin = [1, 2, 4, 8, 16]


class UnmixedVocals(AudioSource):
    def unmixed_vocals(x, sr):
        return unmixed(x, sr, stem="vocals")

    fn = unmixed_vocals


class UnmixedBass(AudioSource):
    def unmixed_bass(x, sr):
        return unmixed(x, sr, stem="bass")

    fn = unmixed_bass


class UnmixedDrums(AudioSource):
    def unmixed_drums(x, sr):
        return unmixed(x, sr, stem="drums")

    fn = unmixed_drums


class UnmixedInstruments(AudioSource):
    def unmixed_instruments(x, sr):
        return unmixed(x, sr, stem="instruments")

    fn = unmixed_instruments


class AudioFilter(RandomPatchPartial):
    fn = identity


class Low(AudioFilter):
    def below_100Hz(x, sr):
        return low_pass(x, sr, fmax=100, db_per_octave=12)

    fn = below_100Hz


class LowMid(AudioFilter):
    def from_100Hz_to_400Hz(x, sr):
        return band_pass(x, sr, fmin=100, fmax=400, db_per_octave=12)

    fn = from_100Hz_to_400Hz


class Mid(AudioFilter):
    def from_400Hz_to_2000Hz(x, sr):
        return band_pass(x, sr, fmin=400, fmax=2000, db_per_octave=12)

    fn = from_400Hz_to_2000Hz


class HighMid(AudioFilter):
    def from_2000Hz_to_5000Hz(x, sr):
        return band_pass(x, sr, fmin=2000, fmax=5000, db_per_octave=12)

    fn = from_2000Hz_to_5000Hz


class High(AudioFilter):
    def above_5000Hz(x, sr):
        return high_pass(x, sr, fmin=5000, db_per_octave=12)

    fn = above_5000Hz


class AudioFeature(RandomPatchPartial):
    pass


class Onsets(AudioFeature):
    type = ["mm", "rosa"]
    prepercussive = [1, 2, 4, 8]
    fn = onsets


class Volume(AudioFeature):
    fn = volume


class Pitch(AudioFeature):
    preharmonic = [1, 2, 4, 8]
    fn = pitch_track


class Tempo(AudioFeature):
    prior = ["uniform", "lognormal"]
    type = ["mm", "rosa"]
    prepercussive = [1, 2, 4, 8]
    tempo_idx = [0, 1, 2, 3]

    def select_tempo(self, x, **kwargs):
        tempo_idx = kwargs["tempo_idx"]
        del kwargs["tempo_idx"]
        return tempo(x, **kwargs)[tempo_idx]

    fn = select_tempo


class Chroma(AudioFeature):
    type = ["cens", "cqt", "stft", "deep", "clp"]
    nearest_neighbor = [True, False]
    preharmonic = [1, 2, 4, 8]
    fn = chroma


class Tonnetz(AudioFeature):
    type = ["cens", "cqt", "stft", "deep", "clp"]
    nearest_neighbor = [True, False]
    preharmonic = [1, 2, 4, 8]
    fn = tonnetz


class Postprocess(RandomPatchPartial):
    fn = identity


class Smooth(Postprocess):
    sigma = [1, 3, 5, 7, 15]
    causal = [1, 0.75, 0.5, 0.25, 0.1, 0]
    fn = gaussian_filter


class Clip(Postprocess):
    percent = [0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
    fn = percentile_clip


class Compress(Postprocess):
    threshold = [0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 0.9]
    ratio = [0.9, 0.75, 0.666, 0.5]
    fn = compress


class Expand(Postprocess):
    threshold = [0.1, 0.25, 0.333, 0.5, 0.666, 0.75, 0.9]
    ratio = [1.5, 2, 3, 4, 8]
    fn = expand


class Target(RandomPatchPartial):
    def forward(self):
        pass


class OnsetLatents(Target):
    is_latent = True
    primitive = ModulatedLatents


class OnsetNoise(Target):
    is_latent = False
    primitive = ModulatedNoise


class VolumeLatents(Target):
    is_latent = True
    primitive = ModulatedLatents


class VolumeNoise(Target):
    is_latent = False
    primitive = ModulatedNoise


class PitchLatents(Target):
    is_latent = True
    primitive = PitchTrackLatents


class TempoLatents(Target):
    is_latent = True
    primitive = TempoLoopLatents
    n_bars = [1, 2, 4, 8, 16, 32]
    type = ["spline", "slerp", "gaussian", "constant"]
    smooth = [1, 2, 4, 8, 16, 32]


class TempoNoise(Target):
    is_latent = False
    primitive = TempoLoopNoise
    n_bars = [1, 2, 4, 8, 16, 32]
    smooth = [1, 2, 4, 8, 16, 32]


class ChromaLatents(Target):
    is_latent = True
    primitive = TonalLatents


class ChromaNoise(Target):
    is_latent = False
    primitive = TonalNoise


class TonnetzLatents(Target):
    is_latent = True
    primitive = TonalLatents


class TonnetzNoise(Target):
    is_latent = False
    primitive = TonalNoise


class RandomPatch(RandomPatchPartial):
    @classmethod
    def randomize(cls):
        which_audio = random.choice(["full", "vocals", "bass", "drums", "instruments", "percussive", "harmonic"])

        if which_audio == "bass":
            which_filtering = "none"
        else:
            which_filtering = random.choice(["none", "low", "low-mid", "mid", "high-mid", "high"])

        which_input = random.choice(["latent", "noise"])

        if which_input == "latent":
            which_feature = random.choice(["volume", "onsets", "pitch_track", "tempo", "chroma", "tonnetz"])
        else:
            which_feature = random.choice(["volume", "onsets", "tempo", "chroma", "tonnetz"])
        if which_audio in ["percussive", "drums"]:
            which_feature = random.choice(["volume", "onsets", "tempo"])

        if which_feature in ["tempo", "pitch_track"]:
            which_postprocess = "none"
        else:
            which_postprocess = random.choice(["none", "smooth", "clip", "compress", "expand"])

        which_layers = random.choice(["full", "low", "mid", "high"])

        return cls(which_audio, which_filtering, which_input, which_feature, which_postprocess, which_layers)

    def __init__(self, which_audio, which_filtering, which_input, which_feature, which_postprocess, which_layers):
        torch.nn.Module.__init__(self)

        if which_audio == "full":
            source = AudioSource.randomize()
        elif which_audio == "harmonic":
            source = Harmonic.randomize()
        elif which_audio == "percussive":
            source = Percussive.randomize()
        elif which_audio == "vocals":
            source = UnmixedVocals.randomize()
        elif which_audio == "bass":
            source = UnmixedBass.randomize()
        elif which_audio == "drums":
            source = UnmixedDrums.randomize()
        elif which_audio == "instruments":
            source = UnmixedInstruments.randomize()

        if which_filtering == "none":
            filter = AudioFilter.randomize()
        elif which_filtering == "low":
            filter = Low.randomize()
        elif which_filtering == "low-mid":
            filter = LowMid.randomize()
        elif which_filtering == "mid":
            filter = Mid.randomize()
        elif which_filtering == "high-mid":
            filter = HighMid.randomize()
        elif which_filtering == "high":
            filter = High.randomize()

        if which_feature == "onsets":
            feature = Onsets.randomize()
            target = OnsetLatents.randomize() if which_input == "latent" else OnsetNoise.randomize()
        elif which_feature == "volume":
            feature = Volume.randomize()
            target = VolumeLatents.randomize() if which_input == "latent" else VolumeNoise.randomize()
        elif which_feature == "pitch_track":
            feature = Pitch.randomize()
            target = PitchLatents.randomize()
        elif which_feature == "tempo":
            feature = Tempo.randomize()
            target = TempoLatents.randomize() if which_input == "latent" else TempoNoise.randomize()
        elif which_feature == "chroma":
            feature = Chroma.randomize()
            target = ChromaLatents.randomize() if which_input == "latent" else ChromaNoise.randomize()
        elif which_feature == "tonnetz":
            feature = Tonnetz.randomize()
            target = TonnetzLatents.randomize() if which_input == "latent" else TonnetzNoise.randomize()

        if which_postprocess == "none":
            postprocess = Postprocess.randomize()
        elif which_postprocess == "smooth":
            postprocess = Smooth.randomize()
        elif which_postprocess == "clip":
            postprocess = Clip.randomize()
        elif which_postprocess == "compress":
            postprocess = Compress.randomize()
        elif which_postprocess == "expand":
            postprocess = Expand.randomize()

        self.source = source
        self.filter = filter
        self.feature = feature
        self.postprocess = postprocess
        self.target = target

        # if which_layers == "full":
        #     layer_fn = partial(identity)
        # elif which_layers == "low":
        #     layer_fn = partial(lambda lats1, lats2: torch.cat((lats2[:, :6], lats1[:, 6:]), dim=1))
        # elif which_layers == "mid":
        #     layer_fn = partial(lambda lats1, lats2: torch.cat((lats1[:, :6], lats2[:, 6:12], lats1[:, 12:]), dim=1))
        # elif which_layers == "high":
        #     layer_fn = partial(lambda lats1, lats2: torch.cat((lats1[:, :12], lats2[:, 12:]), dim=1))
        # self.layer_fn = layer_fn

    def forward(self, audio, sr, dur, fps, audio_file, latent_selection, size):
        print("Input", audio.min(), audio.mean(), audio.max(), audio.shape)
        audio_source = self.source(audio.numpy(), sr=sr)
        print(
            self.source.__class__.__name__,
            audio_source.min(),
            audio_source.mean(),
            audio_source.max(),
            audio_source.shape,
        )
        audio_filtered = self.filter(audio_source, sr=sr)
        print(
            self.filter.__class__.__name__,
            audio_filtered.min(),
            audio_filtered.mean(),
            audio_filtered.max(),
            audio_filtered.shape,
        )
        audio_feature = self.feature(audio_filtered, sr=sr)

        if not isinstance(self.feature, Tempo):
            audio_feature = torch.from_numpy(
                resample(audio_feature, round(dur * fps), axis=np.argmax(audio_feature.shape))
            )
            print(
                self.feature.__class__.__name__,
                audio_feature.min(),
                audio_feature.mean(),
                audio_feature.max(),
                audio_feature.shape,
            )
        else:
            print(self.feature.__class__.__name__, audio_feature)

        audio_postprocessed = self.postprocess(audio_feature)
        if not isinstance(self.feature, Tempo):
            print(
                self.postprocess.__class__.__name__,
                audio_postprocessed.min(),
                audio_postprocessed.mean(),
                audio_postprocessed.max(),
                audio_postprocessed.shape,
            )

        kwargs = self.target.kwargs
        if isinstance(self.feature, Tempo):
            kwargs["fps"] = fps
        print(self.target.__class__.__name__)
        if self.target.is_latent:
            module = self.target.primitive(audio_postprocessed, latent_selection, **kwargs)
        else:
            module = self.target.primitive(audio_postprocessed, size=size, **kwargs)
        return module, audio_feature, audio_postprocessed


if __name__ == "__main__":
    # for _ in range(10):
    # print(RandomPatch.randomize())
    # exit()

    audio_file = random.choice(glob("/home/hans/datasets/wavefunk/*"))
    model_file = random.choice(glob("/home/hans/modelzoo/wavefunk/*/*.pt") + glob("/home/hans/modelzoo/wavefunk/*.pt"))
    checkpoint_name = Path(model_file.replace("/network-snapshot", "")).stem
    checkpoint_name = "_".join(sorted(checkpoint_name.replace("-", "_").split("_"), key=lambda k: random.random())[:7])
    output_file = (
        f"/home/hans/neurout/ssar/random_patch_{Path(audio_file).stem}_{checkpoint_name[:40]}_{str(uuid4())[:8]}.mp4"
    )
    fps = 24
    offset = random.choice([0, 30, 60, 90, 120, 180])
    full_duration = rosa.get_duration(filename=audio_file)
    offset = max(0, min(full_duration - 60, offset))

    audio, sr, dur = load_audio(audio_file, offset=offset, duration=60)
    print("audio_file: ", audio_file)
    print("sr:         ", sr)
    print("model_file: ", model_file)
    print("offset:     ", offset)
    print("duration:   ", dur)
    print("Any NaNs?   ", audio.isnan().any().item())
    print("No Infs?    ", np.isfinite(audio.numpy()).all())

    with torch.inference_mode():
        mapper = StyleGAN2Mapper(model_file, False)

        latents, noises, patches, feats = [], [], [], []
        for i in range(5):
            patch = RandomPatch.randomize()
            print(patch)
            patches.append(patch)

            patch_module, f, p = patch(
                audio=audio,
                sr=sr,
                dur=dur,
                fps=fps,
                audio_file=audio_file,
                latent_selection=mapper(torch.randn(12, 512)),
                size=64,
            )
            feats.append([f, p])

            if not hasattr(patch_module, "modulation"):
                patch_module.modulation = torch.ones((1))
            if "Latent" in patch_module.__class__.__name__:
                latents.append(patch_module)
            else:
                noises.append(patch_module)

        latent = ModulationSum(latents)
        noise = ModulationSum(noises)
        synthesizer = StyleGAN2Synthesizer(model_file, False, (512, 512), "stretch", 0).cuda()
        with VideoWriter(output_file, synthesizer.output_size, fps, audio_file, offset, dur, "slow") as video, NpArr(
            output_file.replace(".mp4", "_noise.npy")
        ) as noise_file, NpArr(output_file.replace(".mp4", "_latent.npy")) as latent_file:
            for _ in trange(round(dur * fps)):
                lat, noi = latent().cuda(), noise().cuda().unsqueeze(0)
                frame = synthesizer(latent_w_plus=lat, **synthesizer.make_noise_pyramid(noi)).add(1).div(2)
                video.write(frame)
                latent_file.append(np.ascontiguousarray(lat.cpu().numpy()))
                noise_file.append(np.ascontiguousarray(noi.cpu().numpy()))

        joblib.dump(patches, output_file.replace(".mp4", f"_patches.pkl"), compress=9)
        with open(output_file.replace(".mp4", f"_patches.txt"), "w") as f:
            for patch in patches:
                f.write(f"{patch}")

        fig, ax = plt.subplots(2, len(feats), figsize=(8 * len(feats), 16))
        for i in range(len(feats)):
            for j in range(len(feats[i])):
                af = feats[i][j]
                if isinstance(af, float):
                    rosa_display.specshow(
                        rosa.feature.tempogram(audio.numpy(), sr), sr=sr, x_axis="time", y_axis="tempo", ax=ax[j, i]
                    )
                    ax[j, i].axhline(af, color="w", linestyle="--", alpha=1, label="Estimated tempo={:g}".format(af))
                    ax[j, i].legend(loc="upper right")
                elif len(af.squeeze().shape) == 1:
                    ax[j, i].plot(af.squeeze().numpy())
                elif af.shape[1] == 6:
                    rosa_display.specshow(
                        np.rollaxis(af.numpy(), 1, 0), sr=sr, x_axis="time", y_axis="tonnetz", ax=ax[j, i]
                    )
                elif af.shape[1] == 12:
                    rosa_display.specshow(
                        np.rollaxis(af.numpy(), 1, 0), sr=sr, x_axis="time", y_axis="chroma", ax=ax[j, i]
                    )
                else:
                    raise Exception(f"what? {af.shape} {patches[i].feature.__class__.__name__}")
                ax[j, i].set(
                    title=patches[i].source.__class__.__name__
                    + " "
                    + patches[i].filter.__class__.__name__
                    + " "
                    + patches[i].target.__class__.__name__
                    + (" " + patches[i].postprocess.__class__.__name__ if j else "")
                )
        plt.tight_layout()
        plt.savefig(output_file.replace(".mp4", ".pdf"))
