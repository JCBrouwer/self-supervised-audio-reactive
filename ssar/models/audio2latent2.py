import numpy as np
import torch
from better_lstm import LSTM
from torch.nn import GRU, AvgPool1d, Dropout, LeakyReLU, Linear, Module, ModuleList, Parameter, Sequential, Sigmoid
from torchqrnn import QRNN

from .audio2latent import DummyHiddenState, MaybePad, Normalize, SwapBatch, UnsqueezeLayerwise, AttentionLayer


class Pool(Module):
    """Average pooling to halve length along 0th (time) axis"""

    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.pool = AvgPool1d(kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.pool(x.permute(1, 2, 0)).permute(2, 0, 1)  # TODO find a way without permutes?


class Unpool(Module):
    """Linear interpolation to double length along 0th (time) axis"""

    def forward(self, y):
        T, B, C = y.shape
        x = torch.linspace(0, 1, T, device=y.device, dtype=y.dtype)
        xq = torch.linspace(0, 1, T * 2, device=y.device, dtype=y.dtype)
        idxr = torch.searchsorted(x, xq, right=False)  # (Tq)
        idxr = torch.clamp(idxr, 1, T - 1)
        idxl = idxr - 1  # (Tq) from [0 to T-2]
        yl = y[:-1]  # (T-1, B, C)
        xl = x[:-1]  # (T-1)
        dy = y[1:] - yl  # (T-1, B, C)
        dx = x[1:] - xl  # (T-1)
        t = (xq - xl[idxl]) / dx[idxl]  # (Tq)
        yq = dy[idxl] * t[:, None, None]  # (Tq, B, C)
        yq += yl[idxl]
        return yq


class ConvTBC(Module):
    """1D convolution over an input of shape (time x batch x channel)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = Parameter(torch.Tensor(kernel_size, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels)

        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, *unused):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)


class EfficientChannelAttention(Module):
    def __init__(self, kernel_size=5):
        super(EfficientChannelAttention, self).__init__()
        self.conv = ConvTBC(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=0, keepdim=True)
        y = self.conv(y.transpose(0, 2)).transpose(0, 2)  # conv over channels
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ConvolutionalBlockAttention(Module):
    def __init__(self, in_channels, out_channels, ratio=8):
        super(ConvolutionalBlockAttention, self).__init__()
        self.linear = Linear(in_channels, out_channels)
        self.fc = Sequential(
            ConvTBC(in_channels, in_channels // ratio, 1, bias=False),
            LeakyReLU(0.2),
            ConvTBC(in_channels // ratio, out_channels, 1, bias=False),
        )
        self.sigmoid = Sigmoid()
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x):
        avg_out = self.fc(torch.mean(x, dim=0, keepdim=True))
        max_out = self.fc(torch.max(x, dim=0, keepdim=True).values)
        return self.linear(x) * self.sigmoid(avg_out + max_out)




class ContextAndCorrelationLayer(Module):
    def __init__(self, context, correlation, in_channels, out_channels, kernel_size, dropout, additive=False):
        super().__init__()
        self.additive = additive
        out_channels = out_channels if additive else out_channels // 2
        self.hidden_channels = out_channels

        if context == "gru":
            self.context = GRU(in_channels, out_channels)  # TODO dropout?
            self.context.flatten_parameters()
        elif context == "lstm":
            self.context = LSTM(in_channels, out_channels, dropouti=dropout, dropoutw=dropout, dropouto=dropout)
            self.context.flatten_parameters()
        elif context == "qrnn":
            self.context = QRNN(in_channels, out_channels, dropout=dropout)
        elif context == "conv":
            self.context = Sequential(
                ConvTBC(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                Dropout(dropout),
                DummyHiddenState(),
            )
        elif context == "transformer":
            self.context = torch.nn.Sequential(AttentionLayer(in_channels, dropout=dropout), DummyHiddenState())
        else:
            raise NotImplementedError()

        if correlation == "linear":
            self.correlation = Sequential(Linear(in_channels, out_channels), Dropout(dropout))
        elif correlation == "eca":
            self.correlation = Sequential(
                EfficientChannelAttention(kernel_size),
                Linear(in_channels, out_channels),
                Dropout(dropout),
            )
        elif correlation == "cba":
            self.correlation = Sequential(ConvolutionalBlockAttention(in_channels, out_channels), Dropout(dropout))
        else:
            raise NotImplementedError()

    def forward(self, x):
        context, _ = self.context(x)
        correlation = self.correlation(x)
        return context + correlation if self.additive else torch.cat((context, correlation), dim=2)


class Audio2Latent2(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size,
        hidden_size,
        num_layers,
        n_outputs,
        output_size,
        context,
        correlation,
        n_layerwise,
        dropout,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.normalize = Normalize(input_mean, input_std + 1e-8)
        self.maybe_pad = MaybePad(num_layers // 2)
        self.swap_batch = SwapBatch()

        multiplier = lambda x: 2 ** min(x, num_layers - x - 1)
        in_channels = input_size
        layers = []
        for n in range(num_layers):
            out_channels = hidden_size * multiplier(n)
            layers.append(
                Sequential(
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        dropout=dropout,
                    ),
                    LeakyReLU(0.2),
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        dropout=dropout,
                    ),
                    LeakyReLU(0.2),
                )
            )
            in_channels = out_channels
        self.layers = ModuleList(layers)
        self.activation = LeakyReLU(0.2)

        self.pool = Pool(kernel_size=5)
        self.unpool = Unpool()

        self.layerwise = ModuleList(
            [
                Sequential(
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=hidden_size,
                        out_channels=hidden_size * 2,
                        kernel_size=5,
                        dropout=dropout,
                        additive=True,
                    ),
                    LeakyReLU(0.2),
                    ContextAndCorrelationLayer(
                        context=context,
                        correlation=correlation,
                        in_channels=hidden_size * 2,
                        out_channels=output_size,
                        kernel_size=5,
                        dropout=dropout,
                        additive=True,
                    ),
                    UnsqueezeLayerwise(n_outputs // n_layerwise),
                )
                for _ in range(n_layerwise)
            ]
        )

    def forward(self, x):
        B, T, C = x.shape
        w = self.maybe_pad(self.normalize(x))
        w = self.swap_batch(w)
        skips = []
        for n, layer in enumerate(self.layers):
            w = layer(w)
            if n < len(self.layers) // 2:
                w = self.pool(w)
                skips.append(w)
            else:
                w = self.unpool(w)
                w += skips.pop()
        w = w[:T]  # remove padding
        w_plus = torch.cat([layerwise(w) for layerwise in self.layerwise], axis=2)
        w_plus = self.swap_batch(w_plus)
        return w_plus
