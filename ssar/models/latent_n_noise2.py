import numpy as np
import torch
from torch.nn import GELU, GRU, Dropout, Linear, Module, Parameter, Sequential
from torch.nn.functional import dropout, gelu

from ..features.processing import gaussian_filter
from .audio2latent import LayerwiseLinear, Normalize
from .sashimi.sashimi import Sashimi


class EnvelopeReactor(torch.nn.Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size,
        hidden_size=64,
        output_size=None,
        num_layers=8,
        backbone="sashimi",
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.normalize = Normalize(input_mean, input_std)

        self.encode = Sequential(Linear(input_size, hidden_size), GELU())

        if backbone.lower() == "gru":
            self.backbone = GRU(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
            self.backbone.flatten_parameters()
        else:
            self.backbone = Sashimi(hidden_size, num_layers, dropout=dropout)

        self.decode = Sequential(GELU(), Linear(hidden_size, hidden_size if output_size is None else output_size))

    def forward(self, x):
        h = self.normalize(x)
        h = self.encode(h)
        init_state = torch.randn((self.num_layers, x.shape[0], self.hidden_size), device=x.device, dtype=x.dtype)
        h, _ = self.backbone(h, init_state)
        y = self.decode(h)
        return y


class Noise(Module):
    def __init__(self, in_channels, n_outputs, dropout, act=gelu):
        super().__init__()
        self.dropout = dropout
        self.NO = n_outputs
        self.w1, self.b1 = self.get_weights_and_bias(n_outputs, in_channels, in_channels // 2)
        self.w2, self.b2 = self.get_weights_and_bias(1, in_channels // 2, 2)
        self.act = act

    def get_weights_and_bias(self, NL, IC, OC):
        w, b = torch.Tensor(NL, IC, OC), torch.Tensor(NL, OC)

        torch.nn.init.kaiming_uniform_(w, a=np.sqrt(5))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(b, -bound, bound)
        return Parameter(w.squeeze()), Parameter(b.squeeze())

    def forward(self, x):
        x = x.unsqueeze(2)  # B,T,1,IC
        x = x.tile(1, 1, self.NO, 1)  # B,T,NO,IC

        h = torch.matmul(x.unsqueeze(3), self.w1).squeeze(3) + self.b1  # B,T,NO,IC//2
        h = dropout(self.act(h), self.dropout, self.training)

        mu_sigs = torch.matmul(h, self.w2) + self.b2  # B,T,NO,2

        B, T, _, _ = x.shape

        noise = []
        for i, mu_sig in enumerate(mu_sigs.unbind(2)):
            mu, sig = mu_sig.unbind(-1)
            size = 2 ** (i + 2)
            mu = mu[..., None, None].expand(B, T, size, size)
            sig = sig[..., None, None].expand(B, T, size, size)
            n = mu + sig * gaussian_filter(torch.randn_like(mu).transpose(1, 0), 5).transpose(1, 0)
            noise.append(n)

        return noise


class LearnedLatentNoiseDecoder(torch.nn.Module):
    def __init__(self, latents, hidden_size=64, n_latent_split=3, n_noise=4, dropout=0.0):
        super().__init__()
        self.act = Sequential(GELU(), Dropout(dropout))
        self.layerwise = LayerwiseLinear(
            in_channels=hidden_size,
            out_channels=latents.shape[2],
            n_outputs=latents.shape[1],
            n_layerwise=n_latent_split,
            dropout=dropout,
        )
        self.noisewise = Noise(in_channels=hidden_size, n_outputs=n_noise, dropout=dropout)

    def forward(self, x):
        latents = self.layerwise(x)
        noise = self.noisewise(x)
        return latents, noise


class FixedLatentNoiseDecoder(torch.nn.Module):
    def __init__(self, latents, hidden_size=12, n_latent_split=3, n_noise=4):
        super().__init__()

        self.S = n_latent_split
        self.H = hidden_size
        assert (
            len(latents) == self.S * self.H
        ), f"Number of latent vectors supplied does not equal n_latent_split * hidden_size ({self.S * self.H})"
        self.latents = latents
        self.W = self.latents.shape[1] // self.S

    def forward(self, x):
        latents = []
        for i in range(self.S):

            env = x[..., i * self.H : (i + 1) * self.H]
            env = env / env.sum(dim=-1, keepdim=True)

            with torch.no_grad():
                lat = self.latents[i * self.H : (i + 1) * self.H, i * self.W : (i + 1) * self.W]

            latents.append(torch.einsum("BTH,HWL->BTWL", env, lat))

        latents = torch.cat(latents, dim=2)
        latents = latents - latents.mean(dim=1, keepdim=True)  # latents --> residuals

        noise_envs = x[..., self.S * self.H :]
        noise = []
        B, T, _ = x.shape
        for i in range(noise_envs.shape[-1] // 2):

            mu, sig = noise_envs[..., 2 * i : 2 * (i + 1)].unbind(-1)

            size = 2 ** (i + 2)
            mu = mu[..., None, None].expand(B, T, size, size)
            sig = sig[..., None, None].expand(B, T, size, size)

            with torch.no_grad():
                n = gaussian_filter(torch.randn_like(mu).transpose(1, 0), 5).transpose(1, 0)

            n = mu + sig * n

            noise.append(n)

        return latents, noise


class LatentNoiseReactor(torch.nn.Module):
    def __init__(
        self,
        input_mean,
        input_std,
        input_size,
        latents,
        # envelope
        num_layers=8,
        backbone="sashimi",
        hidden_size=64,
        # decoder
        decoder="fixed",  # or "learned"
        n_latent_split=3,
        n_noise=4,
        dropout=0.0,
    ):
        super().__init__()

        if decoder == "fixed":
            self.decoder = FixedLatentNoiseDecoder(latents, hidden_size, n_latent_split, n_noise)
            n_envelopes = hidden_size * n_latent_split + 2 * n_noise
        elif decoder == "learned":
            self.decoder = LearnedLatentNoiseDecoder(latents, hidden_size, n_latent_split, n_noise, dropout)
            n_envelopes = hidden_size

        self.envolope = EnvelopeReactor(
            input_mean=input_mean,
            input_std=input_std,
            input_size=input_size,
            hidden_size=n_envelopes,
            num_layers=num_layers,
            backbone=backbone,
            dropout=dropout,
        )

    def forward(self, x, return_envelopes=False):
        envelopes = self.envolope(x)
        if return_envelopes:
            return envelopes
        latents, noise = self.decoder(envelopes)
        return latents, noise
