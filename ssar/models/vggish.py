"""
From https://github.com/sangho-vision/acav100m

MIT License

Copyright (c) 2021 Sangho Lee, Jiwan Chung, Youngjae Yu, Gunhee Kim, Thomas Breuel, Gal Chechik, Yale Song

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import resampy
import torch
from einops import rearrange
from torch import nn
from torch.nn.functional import interpolate


class VggishExtractor(nn.Module):
    def __init__(self, sr=16384, fps=24):
        super().__init__()
        self.sr = sr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = 5
        self.vggish = LayerVggish(
            Namespace(
                data=Namespace(cache_dir=Path("./cache")),
                postprocess=False,
                num_layers=self.num_layers,
            ),
            fps=fps,
        ).to(self.device)
        self.preprocess = self.vggish.get_preprocessor()

    def forward(self, x):
        batch = torch.stack([self.preprocess(None, (sample, self.sr))["data"] for sample in x]).to(self.device)
        output = self.vggish(batch)
        return output


class Vggish(nn.Module):
    args = {"postprocess": False}
    output_dims = 128
    model_tag = {"name": "VGGish", "dataset": "YouTube-8M"}

    def __init__(self, args):
        super().__init__()

        torch.hub.set_dir(str(args.data.cache_dir))
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish", verbose=False)
        self.model.postprocess = args.postprocess
        self.model.preprocess = False

    @classmethod
    def download(cls, args):
        torch.hub.set_dir(str(args.data.cache_dir))
        model = torch.hub.load("harritaylor/torchvggish", "vggish")
        return model

    def get_preprocessor(self):
        return preprocess

    def forward(self, data):
        B = data.shape[0]  # BNCHW
        data = rearrange(data, "b n c h w -> (b n) c h w")
        data = self.model.forward(data)
        data = rearrange(data, "(b n) c -> b n c", b=B)
        data = data.mean(dim=1)  # B 128
        return data


class LayerVggish(Vggish):
    args = {"num_layers": 5, "postprocess": False}
    output_dims = [64, 128, 256, 512, 128]

    def __init__(self, args, fps=24):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.fps = fps

    def forward(self, data):
        B, N, _, _, _ = data.shape  # BNCHW
        data = rearrange(data, "b t c h w -> (b t) c h w")
        res = self.model_forward(data)
        res_pooled = []
        for data in res:
            # TODO validate this rearrange
            data = rearrange(data, "(b n) c t -> b (n t) c", b=B)
            if data.shape[1] != self.fps * N:
                data = interpolate(data.permute(0, 2, 1), size=self.fps * N).permute(0, 2, 1)
            res_pooled.append(data)
        return res_pooled

    def model_forward(self, x, fs=None):
        B = x.shape[0]
        model = self.model
        if model.preprocess:
            x = model._preprocess(x, fs)
        x, res = self.vgg_forward(x)
        if model.postprocess:
            x = model._postprocess(x)
            res.append(x.detach().cpu())
        return res

    def vgg_forward(self, inputs):
        model = self.model
        res = self.run_features(inputs)

        res_resized = []
        for feat in res:
            feat = interpolate(feat.mean(-1), size=self.fps)
            res_resized.append(feat.detach().cpu())

        x = res[-1]
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = model.embeddings(x)
        del res
        res_resized.append(x.unsqueeze(2).detach().cpu())
        return x, res_resized

    def run_features(self, x):
        features = self.model.features

        # segment blocks by MaxPool2d
        blocks = [i + 1 for i, m in enumerate(features) if isinstance(m, torch.nn.modules.pooling.MaxPool2d)]
        blocks = zip([0] + blocks[:-1], blocks)
        blocks = [list(range(x[0], x[1])) for x in blocks]
        blocks = [nn.Sequential(*[features[i] for i in block]) for block in blocks]
        res = []
        for block in blocks:
            x = block(x)
            res.append(x)
        return res


def preprocess(visual, audio):
    data, fps = audio
    if torch.is_tensor(data):
        data = data.numpy()
    if data.shape[0] == 0:
        print("To short a video (< 1 min). Skipping the video.")
        preprocessed = None
    else:
        preprocessed = _preprocess(data, fps)

    return {"data": preprocessed, "fps": fps}


def _preprocess(data, sample_rate):
    # Architectural constants.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE = 16384
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    resampled = data
    # Resample to the rate assumed by VGGish.
    if sample_rate != SAMPLE_RATE:
        resampled = resampy.resample(resampled, sample_rate, SAMPLE_RATE)

    def get_log_mel(x):
        return log_mel_spectrogram(
            x,
            audio_sample_rate=SAMPLE_RATE,
            log_offset=LOG_OFFSET,
            window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=NUM_MEL_BINS,
            lower_edge_hertz=MEL_MIN_HZ,
            upper_edge_hertz=MEL_MAX_HZ,
        )

    log_mel = get_log_mel(resampled)
    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(EXAMPLE_HOP_SECONDS * features_sample_rate))

    num_samples = log_mel.shape[0]

    num_frames = int(np.round((num_samples - example_window_length) / example_hop_length))

    shape = (num_frames, example_window_length) + log_mel.shape[1:]
    strides = (log_mel.strides[0] * example_hop_length,) + log_mel.strides
    log_mel_examples = np.lib.stride_tricks.as_strided(log_mel, shape=shape, strides=strides)

    log_mel_examples_tensor = torch.tensor(log_mel_examples, requires_grad=True)[:, None, :, :].float()

    return log_mel_examples_tensor


# License for code below. Code has been auto-formatted and comments removed.

# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


def frame(data, window_length, hop_length):
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))


def stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    frames = frame(signal, window_length, hop_length)
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(
    num_mel_bins=20, num_spectrogram_bins=129, audio_sample_rate=8000, lower_edge_hertz=125.0, upper_edge_hertz=3800.0
):
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" % (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" % (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(
    data, audio_sample_rate=8000, log_offset=0.0, window_length_secs=0.025, hop_length_secs=0.010, **kwargs
):
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = stft_magnitude(
        data, fft_length=fft_length, hop_length=hop_length_samples, window_length=window_length_samples
    )
    mel_spectrogram = np.dot(
        spectrogram,
        spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1], audio_sample_rate=audio_sample_rate, **kwargs
        ),
    )
    return np.log(mel_spectrogram + log_offset)
