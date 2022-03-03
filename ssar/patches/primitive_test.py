import random
from itertools import product

import numpy as np
import torch
from scipy.signal import resample

# fmt:off
from .random_patches import (ChromaLatents, ChromaNoise, OnsetLatents,
                             OnsetNoise, PitchLatents, TempoLatents,
                             TempoNoise, TonnetzLatents, TonnetzNoise,
                             VolumeLatents, VolumeNoise, chroma, load_audio,
                             onsets, pitch_track, tempo, tonnetz, volume)

# fmt:on

audio_file = "/home/hans/datasets/wavefunk/naamloos.wav"
audio, sr, dur = load_audio(audio_file)
audio = audio.numpy()
fps = 60
n_frames = round(fps * dur)

for feature, PatchClass in [
    [chroma, ChromaLatents],
    [onsets, OnsetLatents],
    [pitch_track, PitchLatents],
    [tempo, TempoLatents],
    [tonnetz, TonnetzLatents],
    [volume, VolumeLatents],
]:
    print(f"Testing {PatchClass.__name__}...")

    enum_settings = {k: v for k, v in PatchClass.__dict__.items() if isinstance(v, list) and isinstance(v[0], str)}
    cont_settings = {k: v for k, v in PatchClass.__dict__.items() if isinstance(v, list) and not isinstance(v[0], str)}
    combinations = [
        {**dict(zip(enum_settings.keys(), settings)), **{k: random.choice(v) for k, v in cont_settings.items()}}
        for settings in product(*enum_settings.values())
    ]
    for config in combinations:
        latent_patch = PatchClass(config)

        kwargs = latent_patch.kwargs
        audio_feature = feature(audio, sr)

        if not isinstance(latent_patch, TempoLatents):
            audio_feature = torch.from_numpy(
                resample(audio_feature, round(dur * fps), axis=np.argmax(audio_feature.shape))
            )
        else:
            kwargs["fps"] = fps
            audio_feature = random.choice(audio_feature)

        latent_module = latent_patch.primitive(audio_feature, torch.randn(12, 18, 512), **kwargs)
        shape = tuple(latent_module.forward().shape)
        assert shape == (
            1,
            18,
            512,
        ), f"Output assertion for {feature.__name__}/{PatchClass.__name__} failed! Shape was {shape} with config {config}"

for feature, PatchClass in [
    [chroma, ChromaNoise],
    [onsets, OnsetNoise],
    [tempo, TempoNoise],
    [tonnetz, TonnetzNoise],
    [volume, VolumeNoise],
]:
    print(f"Testing {PatchClass.__name__}...")

    enum_settings = {k: v for k, v in PatchClass.__dict__.items() if isinstance(v, list) and isinstance(v[0], str)}
    cont_settings = {k: v for k, v in PatchClass.__dict__.items() if isinstance(v, list) and not isinstance(v[0], str)}
    combinations = [
        {**dict(zip(enum_settings.keys(), settings)), **{k: random.choice(v) for k, v in cont_settings.items()}}
        for settings in product(*enum_settings.values())
    ]
    for config in combinations:
        noise_patch = PatchClass(config)

        kwargs = noise_patch.kwargs
        audio_feature = feature(audio, sr)

        if not isinstance(noise_patch, TempoNoise):
            audio_feature = torch.from_numpy(
                resample(audio_feature, round(dur * fps), axis=np.argmax(audio_feature.shape))
            )
        else:
            kwargs["fps"] = fps
            audio_feature = random.choice(audio_feature)

        noise_module = noise_patch.primitive(audio_feature, size=128, **kwargs)
        shape = tuple(noise_module.forward().shape)
        assert shape == (
            1,
            1,
            128,
            128,
        ), f"Output assertion for {feature.__name__}/{PatchClass.__name__} failed! Shape was {shape} with config {config}"
