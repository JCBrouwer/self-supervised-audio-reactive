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

from math import log2, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch.nn.utils import spectral_norm

from .audio2latent import LayerwiseConv, Normalize


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

        Q = self.conv_Q(x).permute(0, 2, 1)  # (batch_size, L_in, key_features)
        K = self.conv_K(x).permute(0, 2, 1)  # (batch_size, L_in, key_features)
        V = self.conv_V(x).permute(0, 2, 1)  # (batch_size, L_in, value_features)

        freqs = self.pos_emb(torch.arange(x.shape[-1], device=x.device), cache_key=x.shape[-1])
        Q = apply_rotary_emb(freqs, Q)
        K = apply_rotary_emb(freqs, K)

        A = (torch.matmul(Q, K.permute(0, 2, 1)) / sqrt(self.key_features)).softmax(2)  # (batch_size, L_in, L_in)
        H = torch.matmul(A, V).permute(0, 2, 1)  # (batch_size, value_features, L_in)

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
        n_epoch_per_layer: int = 1000,
        n_epoch_fade_in_new_layer: int = 200,
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

        self.depth = 0
        self.n_stage = int(log2(target_len)) - 3
        self.schedule = [n_epoch_per_layer * n for n in range(1, self.n_stage + 1)]
        self.pretrain_schedule = [(k, k + n_epoch_fade_in_new_layer) for k in self.schedule]
        self.epoch = 0
        self.epochs = self.pretrain_schedule[-1][1] + n_epoch_per_layer

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

    def update_depth(self, epoch):
        self.epoch = epoch
        if self.n_stage > 0:
            update_epoch = self.schedule[0]
            if epoch > update_epoch:
                self.depth += 1
                self.n_stage -= 1
                self.schedule.pop(0)

    def use_residual(self):
        if self.n_stage > 0 and len(self.pretrain_schedule) > 0:
            start_epoch_test = self.pretrain_schedule[0][0]
            end_epoch_test = self.pretrain_schedule[0][1]
            if end_epoch_test > self.epoch > start_epoch_test:
                start_epoch = self.pretrain_schedule[0][0]
                end_epoch = self.pretrain_schedule[0][1]
                self.pretrain_schedule.pop(0)
        try:
            if end_epoch >= self.epoch >= start_epoch:
                self.residual_factor = (self.epoch - start_epoch) / (end_epoch - start_epoch)
                return True
            else:
                return False
        except Exception:
            return False

    @staticmethod
    def _get_noise(time_features: torch.Tensor):
        bs = time_features.size(0)
        target_len = time_features.size(2)
        noise = torch.randn((bs, 1, target_len), device=time_features.device)
        noise = torch.cat((time_features, noise), dim=1)
        return noise

    def forward(self, time_features: torch.Tensor):
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
        x = self._get_noise(self.normalize(time_features))

        reduced_x = F.avg_pool1d(x, kernel_size=2 ** (self.n_step - 1))  # Reduce x to length 8
        y = self.initial_block(reduced_x)
        for idx, l in enumerate(self.block_list[: self.depth]):
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            previous_y = y
            tf = F.avg_pool1d(x[:, :-1, :], kernel_size=2 ** (self.n_step - 1 - (idx + 1)))  # time features reduced
            y = torch.cat((tf, y), dim=1)
            y = l(y)
            last_idx = idx

        if self.use_residual() and self.depth > 0:
            l_skip = self.skip_block_list[last_idx]
            y = self.residual_factor * self.layerwise(y.permute(0, 2, 1)) + (1 - self.residual_factor) * self.layerwise(
                l_skip(previous_y).permute(0, 2, 1)
            )
        else:
            y = self.layerwise(y.permute(0, 2, 1))

        return y.permute(0, 2, 3, 1)


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
        n_epoch_per_layer: int = 1000,
        n_epoch_fade_in_new_layer: int = 200,
    ):
        super(ProgressiveDiscriminator, self).__init__()
        assert target_len >= 8, "target length should be at least of value 8"
        assert log2(target_len) % 1 == 0, "input length must be an integer that is a power of 2."
        assert 0 <= residual_factor <= 1, "residual factor must be included in [0,1]"

        self.target_len = target_len
        self.n_step = int(log2(target_len)) - 2  # nb of step to go from 8 to target_len
        self.residual_factor = residual_factor
        self.n_channels = n_channels

        self.depth = 0
        self.n_stage = int(log2(target_len)) - 3
        self.schedule = [n_epoch_per_layer * n for n in range(1, self.n_stage + 1)]
        self.pretrain_schedule = [(k, k + n_epoch_fade_in_new_layer) for k in self.schedule]
        self.epoch = 0
        self.epochs = self.pretrain_schedule[-1][1] + n_epoch_per_layer

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

    def update_depth(self, epoch):
        self.epoch = epoch
        if self.n_stage > 0:
            update_epoch = self.schedule[0]
            if epoch > update_epoch:
                self.depth += 1
                self.n_stage -= 1
                self.schedule.pop(0)

    def use_residual(self):
        if self.n_stage > 0 and len(self.pretrain_schedule) > 0:
            start_epoch_test = self.pretrain_schedule[0][0]
            end_epoch_test = self.pretrain_schedule[0][1]
            if end_epoch_test > self.epoch > start_epoch_test:
                start_epoch = self.pretrain_schedule[0][0]
                end_epoch = self.pretrain_schedule[0][1]
                self.pretrain_schedule.pop(0)
        try:
            if end_epoch >= self.epoch >= start_epoch:
                self.residual_factor = (self.epoch - start_epoch) / (end_epoch - start_epoch)
                return True
            else:
                return False
        except Exception:
            return False

    def forward(self, x: torch.Tensor, tf: torch.Tensor):
        """Computes the forward pass

        Arguments:
            x:
                tensor of shape (batch size, 18, 512, input_length)
            tf:
                time features of shape (batch_size, n_features, target_len)
            depth:
                the depth at which the the tensor should flow.
        """

        tf = self.normalize(tf)
        x = x.reshape(x.shape[0], -1, x.shape[3])

        reduce_factor = int(log2(self.target_len)) - int(log2(x.size(2)))
        reduced_tf = F.avg_pool1d(tf, kernel_size=2 ** reduce_factor)

        residual = self.use_residual()
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
