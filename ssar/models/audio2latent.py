from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from better_lstm import LSTM
from torch.nn import GRU, Dropout, LazyConv1d, LazyConvTranspose1d, LeakyReLU, Linear, Module, Parameter, Sequential

from x_transformers.x_transformers import AttentionLayers


class AttentionLayer(AttentionLayers):
    def __init__(self, in_channels, out_channels, n_head, dim_head, dropout):
        super().__init__(
            dim=in_channels, depth=1, heads=n_head, attn_dim_head=dim_head, ff_dim_out=out_channels, dropout=dropout
        )


class Normalize(Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.FloatTensor(mean))
        self.register_buffer("std", torch.FloatTensor(std))

    def forward(self, x):
        return (x - self.mean) / self.std


class MaybePad(Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, x):
        remainder = x.shape[1] % (2 ** self.l)
        if remainder != 0:
            x = F.pad(x, (0, 0, 0, remainder))
        return x


class SwapChannels(Module):
    def forward(self, x):
        return x.transpose(1, 2)  # B, T, C <---> B, C, T


class SwapBatch(Module):
    def forward(self, x):
        return x.transpose(0, 1)  # B, T, C <---> T, B, C


class DummyHiddenState(Module):
    def forward(self, x):
        return x, 0


class UnsqueezeLayerwise(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x.unsqueeze(2).repeat(1, 1, self.n, 1)


class Print(Module):
    def forward(self, x):
        print(x.shape)
        return x


class LayerwiseLinear(Module):
    def __init__(self, in_channels, out_channels, n_outputs, n_layerwise, dropout, act=F.gelu):
        super().__init__()
        self.dropout = dropout
        self.NO = n_outputs
        self.NL = n_layerwise
        self.w1, self.b1 = self.get_weights_and_bias(n_layerwise, in_channels, in_channels)
        self.w2, self.b2 = self.get_weights_and_bias(1, in_channels, out_channels)
        self.act = act

    def get_weights_and_bias(self, NL, IC, OC):
        w, b = torch.Tensor(NL, IC, OC), torch.Tensor(NL, OC)

        torch.nn.init.kaiming_uniform_(w, a=np.sqrt(5))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(b, -bound, bound)

        return Parameter(w.squeeze()), Parameter(b.squeeze())

    def forward(self, x):  # B,T,IC
        x = x.unsqueeze(2)  # B,T,1,IC
        x = x.tile(1, 1, self.NL, 1)  # B,T,NL,IC
        x = torch.matmul(x.unsqueeze(3), self.w1).squeeze(3) + self.b1  # B,T,NL,IC*2
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        x = torch.matmul(x, self.w2) + self.b2  # B,T,NL,OC
        B, T, _, OC = x.shape
        x = x.unsqueeze(3)  # B,T,NL,1,OC
        x = x.tile(1, 1, 1, self.NO // self.NL, 1)  # B,T,NL,NO//NL,OC
        x = x.reshape(B, T, self.NO, OC)  # B,T,NO,OC
        return x


class LayerwiseConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_outputs, n_layerwise, dropout, act=F.gelu):
        super().__init__()
        self.dropout = dropout
        self.NO = n_outputs
        self.NL = n_layerwise
        self.padding = (kernel_size - 1) // 2
        self.w1, self.b1 = self.get_weights_and_bias(n_layerwise, in_channels, in_channels, kernel_size)
        self.w2, self.b2 = self.get_weights_and_bias(1, in_channels, out_channels, kernel_size)
        self.act = act

    def get_weights_and_bias(self, NL, IC, OC, ks):
        w, b = torch.Tensor(NL * OC, IC, ks), torch.Tensor(1, NL * OC, 1)

        torch.nn.init.kaiming_uniform_(w, a=np.sqrt(5))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(b, -bound, bound)

        return Parameter(w), Parameter(b)

    def forward(self, x):  # B,T,IC
        B, T, IC = x.shape
        x = x.transpose(1, 2)  # B,IC,T
        x = x.unsqueeze(2)  # B,IC,1,T
        x = x.tile(1, 1, self.NL, 1)  # B,NL,IC,T
        x = x.reshape(B, self.NL * IC, T)  # B,NL*IC,T
        x = F.conv1d(x, self.w1, padding=self.padding, groups=self.NL) + self.b1  # B,NL*IC*2,T
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x.reshape(B * self.NL, -1, T)  # B*NL,IC*2,T
        x = F.conv1d(x, self.w2, padding=self.padding) + self.b2  # B*NL,OC,T
        x = x.reshape(B, self.NL, -1, T)  # B,NL,OC,T
        _, _, OC, _ = x.shape
        x = x.unsqueeze(2)  # B,NL,1,OC,T
        x = x.tile(1, 1, self.NO // self.NL, 1, 1)  # B,NL,NO//NL,OC,T
        x = x.reshape(B, self.NO, OC, T)  # B,NO,OC,T
        x = x.permute(0, 3, 1, 2)  # B,T,NO,OC
        return x


class Audio2Latent(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size,
        hidden_size,
        num_layers,
        n_outputs,
        output_size,
        backbone,
        skip_backbone,
        layerwise,
        n_layerwise,
        dropout,
    ):
        super().__init__()
        self.normalize = Normalize(input_mean, input_std)

        if backbone.lower() == "gru":
            self.backbone = GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
            )
            self.backbone.flatten_parameters()

        elif backbone.lower() == "lstm":
            self.backbone = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropouti=dropout,
                dropoutw=dropout,
                dropouto=dropout,
                batch_first=True,
            )
            self.backbone.flatten_parameters()

        elif backbone.lower() == "conv":

            def ConvBlock(out_c, transpose, k=5, s=2, p=2, op=1):
                return Sequential(
                    (LazyConvTranspose1d(out_c, k, s, p, op) if transpose else LazyConv1d(out_c, k, s, p)),
                    LeakyReLU(0.2),
                    Dropout(dropout),
                )

            multiplier = lambda x: 2 ** min(x, num_layers - x - 1)
            self.backbone = Sequential(
                MaybePad(num_layers // 2),
                SwapChannels(),
                *[ConvBlock(hidden_size * multiplier(n), transpose=n >= num_layers // 2) for n in range(num_layers)],
                SwapChannels(),
                DummyHiddenState(),
            )

        else:
            raise NotImplementedError()

        self.relu = Sequential(LeakyReLU(0.2), Dropout(dropout))

        if skip_backbone:
            skip_size = hidden_size
            self.backbone_skip = Sequential(
                Linear(input_size, hidden_size),
                LeakyReLU(0.2),
                Dropout(dropout),
                Linear(hidden_size, skip_size),
                LeakyReLU(0.2),
                Dropout(dropout),
                AttentionLayer(skip_size, skip_size, 4, 128, dropout),
                LeakyReLU(0.2),
                Dropout(dropout),
            )
        else:
            self.backbone_skip = None

        assert n_outputs % n_layerwise == 0, f"n_outputs must be divisible by n_layerwise! {n_outputs} / {n_layerwise}"
        layerwise_size = hidden_size + (skip_size if skip_backbone else 0)

        if layerwise == "dense":
            self.layerwise = LayerwiseLinear(
                in_channels=layerwise_size,
                out_channels=output_size,
                n_outputs=n_outputs,
                n_layerwise=n_layerwise,
                dropout=dropout,
                act=partial(F.leaky_relu, negative_slope=0.2),
            )

        elif layerwise == "conv":
            self.layerwise = LayerwiseConv(
                in_channels=layerwise_size,
                out_channels=output_size,
                kernel_size=5,
                n_outputs=n_outputs,
                n_layerwise=n_layerwise,
                dropout=dropout,
                act=partial(F.leaky_relu, negative_slope=0.2),
            )

        else:
            raise NotImplementedError()

    def forward(self, x):
        w, _ = self.backbone(self.normalize(x))
        w = w[:, : x.shape[1]]  # remove padding
        wx = self.relu(w) if self.backbone_skip is None else torch.cat((self.relu(w), self.backbone_skip(x)), axis=2)
        w_plus = self.layerwise(wx)
        return w_plus
