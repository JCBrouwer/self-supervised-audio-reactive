# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import shutil
from math import log2, sqrt
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from torch.nn.utils import spectral_norm
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from umap import UMAP
from context_fid import calculate_fcd

from train_supervised import LatentAugmenter, LayerwiseConv, Normalize, audio2video, get_ffcv_dataloaders

from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding


class SelfAttention(nn.Module):
    """Implements a self attention layer with spectral normalisation.

    Attributes:
        n_in_features:
            number of features in the input sequence.
        key_features:
            size of the queries and the keys vectors
        value_features:
            size of the values vectors
        ks_value:
            size of the kernel of the convolutional layer that produces the
            values vectors
        ks_query:
            size of the kernel of the convolutional layer that produces the
            queries vectors
        ks_key:
            size of the kernel of the convolutional layer that produces the
            keys vectors
    """

    def __init__(
        self,
        n_in_features: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        pos_emb_dim: int,
    ):
        super(SelfAttention, self).__init__()
        assert ks_value % 2 == 1, "ks_value should be an odd number"
        assert ks_query % 2 == 1, "ks_query should be an odd number"
        assert ks_key % 2 == 1, "ks_key should be an odd number"

        d_query, p_query = self._set_params(ks_query)

        self.conv_Q = nn.Conv1d(
            in_channels=n_in_features,
            out_channels=key_features,
            kernel_size=ks_query,
            dilation=d_query,
            padding=p_query,
        )

        d_key, p_key = self._set_params(ks_key)
        self.conv_K = nn.Conv1d(
            in_channels=n_in_features,
            out_channels=key_features,
            kernel_size=ks_key,
            dilation=d_key,
            padding=p_key,
        )

        d_value, p_value = self._set_params(ks_value)
        self.conv_V = nn.Conv1d(
            in_channels=n_in_features,
            out_channels=value_features,
            kernel_size=ks_value,
            dilation=d_value,
            padding=p_value,
        )

        self.key_features = key_features

        self.pos_emb = RotaryEmbedding(dim=pos_emb_dim)

    def _set_params(self, kernel_size):
        """Computes dilation and padding parameter given the kernel size

        The dilation and padding parameter are computed such as
        an input sequence to a 1d convolution does not change in length.

        Returns:
            Two integer for dilation and padding.
        """

        if kernel_size % 2 == 1:  # If kernel size is an odd number
            dilation = 1
            padding = int((kernel_size - 1) / 2)
        else:  # If kernel size is an even number
            dilation = 2
            padding = int(kernel_size - 1)

        return dilation, padding

    def forward(self, x):
        """Computes self-attention

        Arguments:
            x: torch.tensor object of shape (batch_size, n_in_features, L_in)

        Returns:
            torch.tensor object of shape (batch_size, value_features, L_in)
        """

        Q = self.conv_Q(x).permute(0, 2, 1)  # Shape (batch_size, key_features, L_in)
        K = self.conv_K(x).permute(0, 2, 1)  # Shape (batch_size, key_features, L_in)
        V = self.conv_V(x).permute(0, 2, 1)  # Shape (batch_size, value_features, L_in)

        freqs = self.pos_emb(torch.arange(x.shape[-1], device=x.device), cache_key=x.shape[-1])
        Q = apply_rotary_emb(freqs, Q)
        K = apply_rotary_emb(freqs, K)

        A = (torch.matmul(Q, K.permute(0, 2, 1)) / sqrt(self.key_features)).softmax(2)  # Shape (batch_size, L_in, L_in)
        H = torch.matmul(A, V).permute(0, 2, 1)  # Shape (batch_size, value_features, L_in)

        return H


class ResidualSelfAttention(nn.Module):
    """Implement self attention module as described in [1].

    It consists of a self attention layer with spectral normalisation,
    followed by 1x1 convolution with spectral normalisation and a
    parametrized skip connection.

    Attributes:
        n_in_features:
            number of features in the input sequence. It is also the number
            of output features.
        key_features:
            size of the queries and the keys vectors
        value_features:
            size of the values vectors
        ks_value:
            size of the kernel of the convolutional layer that produces the
            values vectors
        ks_query:
            size of the kernel of the convolutional layer that produces the
            queries vectors
        ks_key:
            size of the kernel of the convolutional layer that produces the
            keys vectors
        skip_param:
            float value, initial value of the parametrised skip connection.

    [1]: https://arxiv.org/pdf/1805.08318.pdf
    """

    def __init__(
        self,
        n_in_features: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        pos_emb_dim: int,
        skip_param: float = 0.0,
    ):
        super(ResidualSelfAttention, self).__init__()
        self.self_attention = SelfAttention(
            n_in_features=n_in_features,
            key_features=key_features,
            value_features=value_features,
            ks_value=ks_value,
            ks_query=ks_query,
            ks_key=ks_key,
            pos_emb_dim=pos_emb_dim,
        )
        self.conv = nn.Conv1d(
            in_channels=value_features,
            out_channels=n_in_features,
            kernel_size=1,
        )

        self.gamma = nn.Parameter(torch.tensor(skip_param))

    def forward(self, x):
        """Comptues the forward

        Arguments:
            torch.tensor of shape (batch size, n_in_features, length)
        Returns
            torch.tensor of shape (batch size, n_in_features, length)
        """
        y = self.self_attention(x)
        y = self.conv(y)
        return self.gamma * y + x


class ConvResdiualSelfAttention(nn.Module):
    """Implement Convolution with Residual self attention

    It consists of a convolution layer with spectral normalisation
    and LeakyReLU then a residual self attention described above.

    Attributes:
        n_in_features:
            number of features in the input sequence.
        n_out_features:
            number of features in the output sequence.
        ks_conv:
            kernel size of the conv layer
        key_features:
            size of the queries and the keys vectors
        value_features:
            size of the values vectors
        ks_value:
            size of the kernel of the convolutional layer that produces the
            values vectors
        ks_query:
            size of the kernel of the convolutional layer that produces the
            queries vectors
        ks_key:
            size of the kernel of the convolutional layer that produces the
            keys vectors
        skip_param:
            float value, initial value of the parametrised skip connection.

    """

    def __init__(
        self,
        n_in_features: int,
        n_out_features: int,
        ks_conv: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        pos_emb_dim: int,
        skip_param: float = 0.0,
        self_attention: bool = True,
    ):
        super(ConvResdiualSelfAttention, self).__init__()
        dilation, padding = self._set_params(ks_conv)
        self.spectral_conv = spectral_norm(
            nn.Conv1d(
                in_channels=n_in_features,
                out_channels=n_out_features,
                kernel_size=ks_conv,
                dilation=dilation,
                padding=padding,
            )
        )
        self.leakyrelu = nn.LeakyReLU()
        if self_attention:
            self.res_selfattention = ResidualSelfAttention(
                n_in_features=n_out_features,
                key_features=key_features,
                value_features=value_features,
                ks_value=ks_value,
                ks_query=ks_query,
                ks_key=ks_key,
                pos_emb_dim=pos_emb_dim,
                skip_param=skip_param,
            )
        self.self_attention = self_attention

    def _set_params(self, kernel_size):
        """Computes dilation and padding parameter given the kernel size

        The dilation and padding parameter are computed such as
        an input sequence to a 1d convolution does not change in length.

        Returns:
            Two integer for dilation and padding.
        """

        if kernel_size % 2 == 1:  # If kernel size is an odd number
            dilation = 1
            padding = int((kernel_size - 1) / 2)
        else:  # If kernel size is an even number
            dilation = 2
            padding = int(kernel_size - 1)

        return dilation, padding

    def forward(self, x):
        """Computes the forward

        Arguments:
            torch.tensor of shape (batch size, n_in_features, length)
        Returns:
            torch.tensor of shape (batch size, n_out_features, length)
        """
        x = self.spectral_conv(x)
        x = self.leakyrelu(x)
        if self.self_attention:
            x = self.res_selfattention(x)

        return x


class ProgressiveGenerator(nn.Module):
    """Implementation of the progressive generator.

    The generator will take as input a univariate noise vector and time features of length 8. It will then gradually
    generate a time series of length target_len by doubling the size of the input vector at each time.

    Attributes:
        target_len:
            Integer that specifies the output length of the generated time series
        n_features:
            Number of features passed with as input with the noise vector
        ks_conb:
            kernel size of the conv layer before the self attention module
        key_features:
            size of the key vectors in the self attention module
        value_features:
            size of the value and query vectors in the self attention module
        ks_value:
            kernel size of the conv layer computing the value vectors
        ks_query:
            kernel size of the conv layer computing the query vectors
        ks_key:
            kernel size of the conv layer computing the key vectors
    """

    def __init__(
        self,
        input_mean,
        input_std,
        target_len: int,
        n_features: int,
        ks_conv: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        pos_emb_dim: int,
        n_outputs: int,
        n_layerwise: int,
        output_size: int,
        residual_factor: float = 0.0,
        self_attention: bool = True,
        n_channels: int = 32,
    ):
        super(ProgressiveGenerator, self).__init__()
        assert log2(target_len) % 1 == 0, "target len must be an integer that is a power of 2."
        assert target_len >= 8, "target len should be at least of value 8."
        assert 0 <= residual_factor <= 1, "residual factor must be included in [0,1]"

        self.target_len = target_len
        self.n_step = int(log2(target_len)) - 2
        self.n_features = n_features
        self.n_channels = n_channels
        self.residual_factor = residual_factor

        self.normalize = Normalize(input_mean, input_std + 1e-8)

        self.initial_block = ConvResdiualSelfAttention(
            n_in_features=n_features + 1,
            n_out_features=n_channels,
            ks_conv=ks_conv,
            key_features=key_features,
            value_features=value_features,
            ks_value=ks_value,
            ks_query=ks_query,
            ks_key=ks_key,
            pos_emb_dim=pos_emb_dim,
            skip_param=0.0,
            self_attention=self_attention,
        )

        self.block_list = nn.ModuleList([])
        for stage in range(1, self.n_step):
            self.block_list.append(
                ConvResdiualSelfAttention(
                    n_in_features=n_channels + self.n_features,
                    n_out_features=n_channels,
                    ks_conv=ks_conv,
                    key_features=key_features,
                    value_features=value_features,
                    ks_value=ks_value,
                    ks_query=ks_query,
                    ks_key=ks_key,
                    pos_emb_dim=pos_emb_dim,
                    skip_param=0.0,
                    self_attention=self_attention,
                )
            )
        self.skip_block_list = nn.ModuleList([])
        for stage in range(1, self.n_step):
            self.skip_block_list.append(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=1))

        self.layerwise = spectral_norm(
            LayerwiseConv(
                n_channels, output_size, kernel_size=5, n_outputs=n_outputs, n_layerwise=n_layerwise, dropout=0
            ),
            name="w2",
        )

    @staticmethod
    def _get_noise(time_features: torch.Tensor):
        bs = time_features.size(0)
        target_len = time_features.size(2)
        noise = torch.randn((bs, 1, target_len), device=time_features.device)
        noise = torch.cat((time_features, noise), dim=1)
        return noise

    def forward(self, time_features: torch.Tensor, depth: int = None, residual: bool = False):
        """Computes the forward

        Arguments:
            x:
                torch.tensor of shape (batch size, n_features, target_len)
            depth:
                the depth at which the the tensor should flow.

        Returns
            torch.tensor of shape (batch_size, 1, length_out).
            length_out will depend on the current stage we are in. It is included between
            8 and target_len
        """
        if depth is None:
            depth = self.n_step - 1

        x = self._get_noise(self.normalize(time_features))

        assert x.dim() == 3, "input must be three dimensional"
        # assert x.size(2) == self.target_len, "third dimension of input must be equal to target_len"
        assert depth <= self.n_step - 1, "depth is too high"

        reduced_x = F.avg_pool1d(x, kernel_size=2 ** (self.n_step - 1))  # Reduce x to length 8
        y = self.initial_block(reduced_x)
        for idx, l in enumerate(self.block_list[:depth]):
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            previous_y = y
            tf = F.avg_pool1d(x[:, :-1, :], kernel_size=2 ** (self.n_step - 1 - (idx + 1)))  # time features reduced
            y = torch.cat((tf, y), dim=1)
            y = l(y)
            last_idx = idx

        if residual and depth > 0:
            l_skip = self.skip_block_list[last_idx]
            y = self.residual_factor * self.layerwise(y.permute(0, 2, 1)) + (1 - self.residual_factor) * self.layerwise(
                l_skip(previous_y).permute(0, 2, 1)
            )
        else:
            y = self.layerwise(y.permute(0, 2, 1))

        return y


class ProgressiveGeneratorInference(nn.Module):
    def __init__(self, trained_generator: nn.Module):
        super(ProgressiveGeneratorInference, self).__init__()
        self.trained_generator = trained_generator

    @staticmethod
    def _get_noise(time_features: torch.Tensor):
        bs = time_features.size(0)
        target_len = time_features.size(2)
        noise = torch.randn((bs, 1, target_len), device=time_features.device)
        noise = torch.cat((time_features, noise), dim=1)
        return noise

    def forward(self, time_features: torch.Tensor):
        noise = self._get_noise(time_features)
        return self.trained_generator(x=noise)


class ProgressiveDiscriminator(nn.Module):
    """
    Attributes:
        target_len:
            length of the longest time series that can be discriminated
    """

    def __init__(
        self,
        input_mean,
        input_std,
        target_len: int,
        n_features: int,
        ks_conv: int,
        key_features: int,
        value_features: int,
        ks_value: int,
        ks_query: int,
        ks_key: int,
        pos_emb_dim: int,
        n_outputs: int,
        output_size: int,
        residual_factor: float = 0.0,
        self_attention: bool = True,
        n_channels: int = 32,
    ):
        super(ProgressiveDiscriminator, self).__init__()
        assert target_len >= 8, "target length should be at least of value 8"
        assert log2(target_len) % 1 == 0, "input length must be an integer that is a power of 2."
        assert 0 <= residual_factor <= 1, "residual factor must be included in [0,1]"

        self.target_len = target_len
        self.n_step = int(log2(target_len)) - 2  # nb of step to go from 8 to target_len
        self.residual_factor = residual_factor
        self.n_channels = n_channels

        self.normalize = Normalize(input_mean, input_std + 1e-8)

        self.initial_block = nn.Sequential(
            spectral_norm(
                nn.Conv1d(
                    in_channels=n_outputs * output_size + n_features,
                    out_channels=n_channels,
                    kernel_size=1,
                )
            ),
            nn.LeakyReLU(),
        )

        self.block_list = nn.ModuleList([])
        for stage in range(self.n_step - 1, 0, -1):
            self.block_list.append(
                ConvResdiualSelfAttention(
                    n_in_features=n_channels,
                    n_out_features=n_channels,
                    ks_conv=ks_conv,
                    key_features=key_features,
                    value_features=value_features,
                    ks_value=ks_value,
                    ks_query=ks_query,
                    ks_key=ks_key,
                    pos_emb_dim=pos_emb_dim,
                    skip_param=0.0,
                    self_attention=self_attention,
                )
            )

        self.last_block = nn.Sequential(
            ConvResdiualSelfAttention(
                n_in_features=n_channels,
                n_out_features=n_channels,
                ks_conv=ks_conv,
                key_features=key_features,
                value_features=value_features,
                ks_value=ks_value,
                ks_query=ks_query,
                ks_key=ks_key,
                pos_emb_dim=pos_emb_dim,
                skip_param=0.0,
                self_attention=self_attention,
            ),
            spectral_norm(nn.Conv1d(in_channels=n_channels, out_channels=1, kernel_size=1)),
            nn.LeakyReLU(),
        )

        self.fc = spectral_norm(nn.Linear(8, 1))

    def forward(self, x: torch.Tensor, tf: torch.Tensor, depth: int = None, residual: bool = False):
        """Computes the forward pass

        Arguments:
            x:
                tensor of shape (batch size, 18, 512, input_length)
            tf:
                time features of shape (batch_size, n_features, target_len)
            depth:
                the depth at which the the tensor should flow.
        """

        if depth is None:
            depth = self.n_step - 1

        tf = self.normalize(tf)
        x = x.reshape(x.shape[0], -1, x.shape[3])

        assert x.dim() == 3, "input must be three dimensional"
        assert x.size(2) >= 8, "third dimension of input must be greater or equal than 8"
        assert log2(x.size(2)) % 1 == 0, "input length must be an integer that is a power of 2."
        assert tf.size(2) == self.target_len, "length of features should be equal to target len"

        reduce_factor = int(log2(self.target_len)) - int(log2(x.size(2)))
        reduced_tf = F.avg_pool1d(tf, kernel_size=2 ** reduce_factor)

        if residual:
            pre_reduce_tf = F.avg_pool1d(tf, kernel_size=2 ** (reduce_factor + 1))
            pre_x = F.avg_pool1d(x, kernel_size=2)
            pre_x = self.initial_block(torch.cat((pre_reduce_tf, pre_x), dim=1))

        x = torch.cat((reduced_tf, x), dim=1)
        x = self.initial_block(x)

        for idx, l in enumerate(self.block_list[reduce_factor:]):
            x = l(x)
            x = F.avg_pool1d(x, kernel_size=2)
            if idx == 0:
                if residual:
                    x = self.residual_factor * x + (1 - self.residual_factor) * pre_x

        x = self.last_block(x)
        x = self.fc(x.squeeze(1))
        return x


def _residual():
    if n_stage > 0 and len(pretrain_schedule) > 0:
        start_epoch_test = pretrain_schedule[0][0]
        end_epoch_test = pretrain_schedule[0][1]
        if end_epoch_test > epoch > start_epoch_test:
            start_epoch = pretrain_schedule[0][0]
            end_epoch = pretrain_schedule[0][1]
            pretrain_schedule.pop(0)
    try:
        if end_epoch >= epoch >= start_epoch:
            G.residual_factor = D.residual_factor = (epoch - start_epoch) / (end_epoch - start_epoch)
            return True
        else:
            return False
    except Exception:
        return False


def infiniter(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def print_model_summary(G, D):
    global total_params
    total_params = 0
    handles = []
    for name, block in list(G.named_modules()) + list(D.named_modules()):

        def hook(m, i, o, name=name):
            global total_params
            if len(list(m.named_modules())) == 1:
                class_name = m.__class__.__name__
                output_shape = (
                    tuple(tuple(oo.shape) if not isinstance(oo, int) else oo for oo in o)
                    if isinstance(o, tuple)
                    else tuple(o.shape)
                )
                num_params = sum(p.numel() for p in m.parameters())
                total_params += num_params
                print(
                    name.ljust(60),
                    class_name.ljust(20),
                    f"{output_shape}".ljust(40),
                    f"{num_params/ 1000:.2f} K" if num_params > 0 else "0",
                )

        handles.append(block.register_forward_hook(hook))

    print()
    print("G summary:")
    print("name".ljust(60), "class".ljust(20), "output shape".ljust(40), "num params")
    print("-" * 150)
    output = G(inputs.permute(0, 2, 1).to(device))
    print("-" * 150)
    print("total".ljust(60), f"".ljust(20), f"".ljust(40), f"{total_params/1e6:.2f} M")

    print()
    print("D summary:")
    print("name".ljust(60), "class".ljust(20), "output shape".ljust(40), "num params")
    print("-" * 150)
    D(output.permute(0, 2, 3, 1), inputs.permute(0, 2, 1).to(device))
    print("-" * 150)
    print("total".ljust(60), f"".ljust(20), f"".ljust(40), f"{total_params/1e6:.2f} M")

    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_len = 128
    n_channels = 32
    depth = 0
    n_epoch_per_layer = 600
    n_epoch_fade_in_new_layer = 200
    epoch_len = 128
    ks_conv = 3
    key_features = value_features = 32
    pos_emb_dim = 16
    ks_value = ks_query = ks_key = 1
    n_layerwise = 3
    batch_size = 128
    lr = 5e-4
    # aug_weight = 0.5
    synthetic = False

    fps = 24
    dur = target_len / fps

    n_stage = int(log2(target_len)) - 3
    schedule = [n_epoch_per_layer * n for n in range(1, n_stage + 1)]
    pretrain_schedule = [(k, k + n_epoch_fade_in_new_layer) for k in schedule]
    epochs = pretrain_schedule[-1][1] + n_epoch_per_layer

    in_dir = "/home/hans/datasets/audio2latent/"
    dataset_cache = f"cache/{Path(in_dir).stem}_preprocessed_{target_len}frames.npy"
    test_audio = "/home/hans/datasets/wavefunk/Ouroboromorphism_49_89.flac"

    train_mean, train_std, train_dataloader, val_dataloader = get_ffcv_dataloaders(
        in_dir, synthetic, batch_size, dur, fps
    )
    train_mean, train_std = train_mean[None, :, None], train_std[None, :, None]
    valiter = infiniter(val_dataloader)
    trainiter = infiniter(train_dataloader)

    # if aug_weight > 0:
    #     augmenter = LatentAugmenter(
    #         checkpoint="/home/hans/modelzoo/train_checks/neurout2-117.pt", n_patches=3, synthetic=synthetic
    #     )

    inputs, targets = next(trainiter)
    n_features = inputs.shape[2]
    n_outputs, output_size = targets.shape[2], targets.shape[3]

    G = ProgressiveGenerator(
        input_mean=train_mean,
        input_std=train_std,
        target_len=target_len,
        n_features=n_features,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_layerwise=n_layerwise,
        output_size=output_size,
        ks_conv=ks_conv,
        key_features=key_features,
        value_features=value_features,
        ks_value=ks_value,
        ks_query=ks_query,
        ks_key=ks_key,
        pos_emb_dim=pos_emb_dim,
        self_attention=True,
    ).to(device)

    D = ProgressiveDiscriminator(
        input_mean=train_mean,
        input_std=train_std,
        target_len=target_len,
        n_features=n_features,
        n_channels=n_channels,
        n_outputs=n_outputs,
        output_size=output_size,
        ks_conv=ks_conv,
        key_features=key_features,
        value_features=value_features,
        ks_value=ks_value,
        ks_query=ks_query,
        ks_key=ks_key,
        pos_emb_dim=pos_emb_dim,
        self_attention=True,
    ).to(device)

    print_model_summary(G, D)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0, 0.99))
    optimizer_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0, 0.99))

    name = "_".join(["PSAGAN", f"length:{target_len}", f"hidden_size:{n_channels}", f"lr:{lr}"])
    writer = SummaryWriter(comment=name)
    shutil.copy(__file__, writer.log_dir)
    shutil.copy(f"{os.path.dirname(__file__)}/train_supervised.py", writer.log_dir)

    n_iter = 0
    video_interval = 100
    eval_interval = 20
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        for _ in range(epoch_len):

            if n_stage > 0:  # update growth schedule
                update_epoch = schedule[0]
                if epoch > update_epoch:
                    depth += 1
                    n_stage -= 1
                    schedule.pop(0)

            G.train(), D.train()

            # load data
            features, targets = next(trainiter)
            features, targets = features.to(device).permute(0, 2, 1), targets.to(device).permute(0, 2, 3, 1)
            b, n, l, t = targets.size()

            # Generator step

            for p in D.parameters():
                p.requires_grad = False

            generated = G(features, depth=depth, residual=_residual()).permute(0, 2, 3, 1)

            reduce_factor = int(log2(D.target_len)) - int(log2(generated.size(3)))
            targets = F.avg_pool1d(targets.reshape(b, -1, t), kernel_size=2 ** reduce_factor).reshape(b, n, l, -1)

            loss = D(generated, features, depth=depth, residual=_residual()) - 1
            loss_ls = 0.5 * torch.square(loss).mean()

            loss_std = torch.abs(generated.std(dim=(1, 2)) - targets.std(dim=(1, 2))).mean()
            loss_mean = torch.abs(generated.mean(dim=(1, 2)) - targets.mean(dim=(1, 2))).mean()
            loss_moment = loss_std + loss_mean

            loss_g = loss_ls + loss_moment

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            writer.add_scalar("G/loss_ls", loss_ls.item(), n_iter)
            writer.add_scalar("G/loss_moment", loss_moment.item(), n_iter)

            # Discriminator step

            for p in D.parameters():
                p.requires_grad = True

            features, targets = next(trainiter)
            features, targets = features.to(device).permute(0, 2, 1), targets.to(device).permute(0, 2, 3, 1)
            targets = F.avg_pool1d(targets.reshape(b, -1, t), kernel_size=2 ** reduce_factor).reshape(b, n, l, -1)

            with torch.no_grad():
                generated = G(features, depth=depth, residual=_residual()).permute(0, 2, 3, 1)

            preds_fake = D(generated.detach(), features, depth=depth, residual=_residual())
            preds_real = D(targets, features, depth=depth, residual=_residual())
            loss_fake = 0.5 * torch.square(preds_fake).mean()
            loss_real = 0.5 * torch.square(preds_real - 1).mean()
            loss_d = loss_real + loss_fake

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            mean_fake_pred = loss_fake.mean().item()
            mean_real_pred = loss_real.mean().item()
            writer.add_scalar("D/mean_fake_pred", mean_fake_pred, n_iter)
            writer.add_scalar("D/mean_real_pred", mean_real_pred, n_iter)
            writer.add_scalar("D/loss_fake", loss_fake.item(), n_iter)
            writer.add_scalar("D/loss_real", loss_real.item(), n_iter)

            n_iter += len(features)

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs:
            with torch.inference_mode():
                G.eval()

                try:
                    fcd = calculate_fcd(
                        val_dataloader,
                        lambda x: F.interpolate(
                            G(x.permute(0, 2, 1), depth=depth, residual=False)
                            .reshape(x.shape[0], x.shape[1], -1)
                            .permute(0, 2, 1),
                            scale_factor=2 ** reduce_factor,
                        )
                        .permute(0, 2, 1)
                        .reshape(x.shape[0], -1, n_outputs, output_size),
                    )
                    writer.add_scalar("Eval/FCD", fcd.item(), n_iter)
                except Exception as e:
                    pbar.write(f"\nError in FCD:\n{e}\n\n")
                    fcd = -1

                features, targets = next(valiter)
                features, targets = features.to(device).permute(0, 2, 1), targets.to(device).permute(0, 2, 3, 1)
                targets = F.avg_pool1d(targets.reshape(b, -1, t), kernel_size=2 ** reduce_factor).reshape(b, n, l, -1)

                generated = G(features, depth=depth, residual=False).permute(0, 2, 3, 1)

                try:
                    loc, scale = stats.laplace.fit(
                        np.random.choice(generated.cpu().numpy().flatten(), 100_000), loc=0, scale=0.1
                    )
                    writer.add_scalar("Eval/laplace_b", scale.item(), n_iter)
                except Exception as e:
                    pbar.write(f"\nError in Laplace fit:\n{e}\n\n")
                    scale = -1

                generated_stats = (
                    torch.cat((generated.mean((1, 2)), generated.std((1, 2)))).reshape(b, -1).cpu().numpy()
                )
                target_stats = torch.cat((targets.mean((1, 2)), targets.std((1, 2)))).reshape(b, -1).cpu().numpy()
                if generated_stats.shape[1] > 48:
                    generated_stats = PCA(n_components=48).fit_transform(generated_stats)
                    target_stats = PCA(n_components=48).fit_transform(target_stats)
                full_stats = np.concatenate((generated_stats, target_stats), axis=0)
                full_umap = UMAP().fit_transform(full_stats)
                fake_umap = full_umap[:b, :]
                real_umap = full_umap[b:, :]

                plt.plot(fake_umap[:, 0], fake_umap[:, 1], "o", label="fake samples", alpha=0.4)
                plt.plot(real_umap[:, 0], real_umap[:, 1], "o", label="real samples", alpha=0.4)
                plt.legend()
                plt.savefig(f"{writer.log_dir}/umap_{epoch}.pdf")
                plt.close()

            pbar.write("")
            pbar.write(f"epoch {epoch + 1}")
            pbar.write(f"laplace_b : {scale:.4f}")
            pbar.write(f"fake      : {mean_fake_pred:.4f}")
            pbar.write(f"real      : {mean_real_pred:.4f}")
            pbar.write(f"fcd       : {fcd:.4f}")
            pbar.write("")

        if (epoch + 1) % video_interval == 0 or (epoch + 1) == epochs:
            checkpoint_name = f"psagan_{name}_steps{n_iter:08}_fcd{fcd:.4f}_b{scale:.4f}"
            joblib.dump(
                {
                    "n_iter": n_iter,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "G_opt": optimizer_g.state_dict(),
                    "D_opt": optimizer_d.state_dict(),
                },
                f"{writer.log_dir}/{checkpoint_name}.pt",
                compress=9,
            )
            audio2video(
                a2l=lambda x: F.interpolate(
                    G(x.permute(0, 2, 1), depth=depth, residual=False)
                    .reshape(x.shape[0], x.shape[1], -1)
                    .permute(0, 2, 1),
                    scale_factor=2 ** reduce_factor,
                )
                .permute(0, 2, 1)
                .reshape(x.shape[0], -1, n_outputs, output_size),
                audio_file=test_audio,
                out_file=f"{writer.log_dir}/{checkpoint_name}_{Path(test_audio).stem}.mp4",
                stylegan_file="/home/hans/modelzoo/train_checks/neurout2-117.pt",
                onsets_only=synthetic,
            )

    writer.close()
