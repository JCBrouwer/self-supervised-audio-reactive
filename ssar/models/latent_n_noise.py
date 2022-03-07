import torch
from torch.nn import GELU, Conv3d, LeakyReLU, Linear, Module, ModuleList, Sequential, Upsample
from x_transformers import Encoder

from .audio2latent import Normalize, UnsqueezeLayerwise


class GLU(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.act = GELU()
        self.proj = Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class Reshape(Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        b, t, _ = x.shape
        return x.reshape(b, -1, t, *self.shape)


class Reactor(Module):
    def __init__(
        self,
        input_mean,
        input_std,
        dim_in,
        dim,
        n_hid_latents,
        n_out_latents,
        latent_dim,
        n_layers=8,
        n_head=8,
        dropout=0.2,
    ):
        super().__init__()
        self.normalize = Normalize(input_mean, input_std)
        self.encoder = Sequential(
            GLU(dim_in, dim),
            Encoder(
                dim=dim,
                depth=n_layers,
                heads=n_head,
                attn_dim_head=dim // 2,
                dropout=dropout,
                alibi_pos_bias=True,
                alibi_num_heads=4,
                ff_glu=True,
            ),
        )

        self.latent_outsamplers = ModuleList(
            [
                Sequential(
                    Encoder(
                        dim=dim,
                        depth=2,
                        heads=n_head,
                        attn_dim_head=dim // 2,
                        dropout=dropout,
                        alibi_pos_bias=True,
                        alibi_num_heads=4,
                        ff_glu=True,
                    ),
                    Linear(dim, latent_dim),
                    LeakyReLU(0.2),
                    UnsqueezeLayerwise(n_out_latents // n_hid_latents),
                )
                for n in range(n_hid_latents)
            ]
        )

        self.noise_prep = Sequential(GLU(dim, dim * 4), Reshape(2, 2), Conv3d(dim, dim, 3, 1, 1), GELU())
        self.noise_upsamplers = ModuleList(
            [
                Sequential(
                    Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
                    Conv3d(dim, dim, 3, 1, 1),
                    GELU(),
                ),
                Sequential(
                    Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
                    Conv3d(dim, dim, 3, 1, 1),
                    GELU(),
                ),
                Sequential(
                    Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
                    Conv3d(dim, dim, 3, 1, 1),
                    GELU(),
                ),
                Sequential(
                    Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
                    Conv3d(dim, dim, 3, 1, 1),
                    GELU(),
                ),
            ]
        )
        self.noise_outsamplers = ModuleList(
            [Conv3d(dim, 1, 3, 1, 1), Conv3d(dim, 1, 3, 1, 1), Conv3d(dim, 1, 3, 1, 1), Conv3d(dim, 1, 3, 1, 1)]
        )

    def forward(self, features):
        hidden = self.encoder(self.normalize(features))

        latents = torch.cat([outsampler(hidden) for outsampler in self.latent_outsamplers], dim=2)

        noise = []
        hidden_noise = self.noise_prep(hidden)
        for upsampler, outsampler in zip(self.noise_upsamplers, self.noise_outsamplers):
            hidden_noise = upsampler(hidden_noise)
            noise.append(outsampler(hidden_noise).squeeze())

        return [latents] + noise


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    time_length = 192
    n_features = 59
    inputs = torch.randn(batch_size, time_length, n_features)
    input_mean = inputs.mean((0, 1))
    input_std = inputs.std((0, 1))

    model = Reactor(
        input_mean, input_std, dim_in=n_features, dim=32, n_hid_latents=3, n_out_latents=18, latent_dim=512
    ).to(device)

    def print_model_summary(model):
        global total_params
        total_params = 0
        handles = []
        for name, block in model.named_modules():

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
                        name.ljust(50),
                        class_name.ljust(30),
                        f"{output_shape}".ljust(40),
                        f"{num_params/ 1000:.2f} K" if num_params > 0 else "0",
                    )

            handles.append(block.register_forward_hook(hook))

        print()
        print("model summary:")
        print("name".ljust(50), "class".ljust(30), "output shape".ljust(40), "num params")
        print("-" * 140)
        model(inputs.to(device))
        print("-" * 140)
        print("total".ljust(50), f"".ljust(30), f"".ljust(40), f"{total_params/1e6:.2f} M")

        for handle in handles:
            handle.remove()

    print_model_summary(model)
    print([o.shape for o in model(inputs.to(device))])
