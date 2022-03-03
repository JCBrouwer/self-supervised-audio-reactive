import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import GELU, Conv1d, ConvTranspose1d, Identity, Linear, Module, ModuleList, Parameter, Sequential

from .audio2latent import LayerwiseConv, Normalize


class DropPath(Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerNorm(Module):
    def __init__(self, normalized_shape, data_format="channels_first", eps=1e-6):
        super().__init__()
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
            return x


class ConvNeXtBlock(Module):
    def __init__(self, dim, drop_path=0.0, gamma_eps=1e-6):
        super().__init__()
        self.dwconv = Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, data_format="channels_last")
        self.pwconv1 = Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = GELU()
        self.pwconv2 = Linear(4 * dim, dim)
        self.gamma = Parameter(gamma_eps * torch.ones((dim)), requires_grad=True) if gamma_eps > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size=52,
        hidden_size=64,
        output_size=512,
        n_outputs=18,
        n_layerwise=3,
        depths=[3, 3, 6, 3],
        cbase=16,
        drop_path_rate=0.2,
        gamma_eps=1e-6,
    ):
        super().__init__()
        self.normalize = Normalize(input_mean, input_std + 1e-8)

        dims = cbase * np.array([1, 2, 4, 8])

        self.downsample_layers, self.upsample_layers = ModuleList(), ModuleList()
        self.downsample_layers.append(
            Sequential(Conv1d(input_size, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0]))
        )
        for i in range(3):
            self.downsample_layers.append(Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.upsample_layers.append(ConvTranspose1d(dims[3 - i], dims[3 - i - 1], kernel_size=2, stride=2))
        self.upsample_layers.append(
            Sequential(ConvTranspose1d(dims[0], hidden_size, kernel_size=4, stride=4), LayerNorm(hidden_size))
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        c = 0
        self.down_stages, self.up_stages = ModuleList(), ModuleList()
        for i in range(4):
            self.down_stages.append(
                Sequential(
                    *[ConvNeXtBlock(dims[i], drop_rates[c + j], gamma_eps) for j in range(depths[i])],
                    LayerNorm(dims[i]),
                )
            )
            self.up_stages.append(
                Sequential(
                    *[ConvNeXtBlock(dims[3 - i], drop_rates[c + j], gamma_eps) for j in reversed(range(depths[3 - i]))],
                    LayerNorm(dims[3 - i]),
                )
            )
            c += depths[i]

        self.norm = LayerNorm(hidden_size, data_format="channels_last")
        self.layerwise = LayerwiseConv(
            hidden_size,
            output_size,
            kernel_size=5,
            n_outputs=n_outputs,
            n_layerwise=n_layerwise,
            dropout=drop_path_rate,
        )

    def forward(self, x):
        x = self.normalize(x)
        x = x.permute(0, 2, 1)
        skips = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.down_stages[i](x)
            if i < 3:  # don't need last x
                skips.append(x)
        for i in range(4):
            x = self.up_stages[i](x)
            x = self.upsample_layers[i](x)
            if i < 3:  # don't need last x
                x += skips.pop()
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.layerwise(x)
        return x
