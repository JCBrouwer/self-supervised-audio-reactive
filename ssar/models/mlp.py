from torch.nn import GELU, Conv1d, Dropout, Linear, Module, ModuleList, Sequential

from .audio2latent import AttentionLayer, LayerwiseConv, Normalize


class ConvolutionalGatingUnit(Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.conv = Conv1d(
            channels // 2, channels // 2, kernel_size, padding=(kernel_size - 1) // 2, groups=channels // 2
        )
        self.dense = Linear(channels // 2, channels // 2)

    def forward(self, x, z=None):
        xr, xg = x.chunk(2, dim=2)
        xg = self.conv(xg.transpose(1, 2)).transpose(1, 2)
        xg = self.dense(xg)
        if z is not None:
            xg = xg + z
        return xr * xg


class MLPBlock(Module):
    def __init__(self, channels, kernel_size, mult):
        super().__init__()
        self.dense1 = Linear(channels, channels * mult)
        self.act = GELU()
        self.cgu = ConvolutionalGatingUnit(channels * mult, kernel_size)
        self.dense2 = Linear(channels * mult // 2, channels)

    def forward(self, x, z=None):
        y = self.dense1(x)
        y = self.act(y)
        y = self.cgu(y, z)
        y = self.dense2(y)
        return x + y


class MLP(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        in_channels,
        channels,
        out_channels,
        n_outputs,
        n_layerwise,
        num_layers,
        dropout,
        mult=2,
        kernel_size=15,
    ):
        super().__init__()
        self.normalize = Normalize(input_mean, input_std + 1e-8)
        self.attn = Sequential(
            Linear(in_channels, channels * mult // 2),
            GELU(),
            Dropout(dropout),
            AttentionLayer(channels * mult // 2, channels * mult // 2, n_head=4, dim_head=128, dropout=dropout),
        )
        self.input_dense = Linear(in_channels, channels)
        self.dropout = Dropout(dropout)
        self.blocks = ModuleList([MLPBlock(channels, kernel_size=kernel_size, mult=mult) for _ in range(num_layers)])
        self.layerwise = LayerwiseConv(channels, out_channels, 5, n_outputs, n_layerwise, dropout)

    def forward(self, x):
        x = self.normalize(x)
        z = self.attn(x)
        x = self.input_dense(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, z)
            x = self.dropout(x)
        w = self.layerwise(x)
        return w


class MLPSeq2Seq(Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_layers,
        dropout,
        mult=2,
        kernel_size=15,
    ):
        super().__init__()
        self.attn = Sequential(
            Linear(in_channels, channels * mult // 2),
            GELU(),
            Dropout(dropout),
            AttentionLayer(
                channels * mult // 2, channels * mult // 2, n_head=4, dim_head=channels // 4, dropout=dropout
            ),
        )
        self.input_dense = Linear(in_channels, channels)
        self.dropout = Dropout(dropout)
        self.blocks = ModuleList([MLPBlock(channels, kernel_size=kernel_size, mult=mult) for _ in range(num_layers)])

    def forward(self, x):
        z = self.attn(x)
        x = self.input_dense(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, z)
            x = self.dropout(x)
        return x
