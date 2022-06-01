import random
import sys

import numpy as np
import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from .data import normalize

sys.path.append("/home/hans/code/maua/")
from maua.GAN.wrappers.stylegan2 import StyleGAN2Mapper


def spline_loop_latents(y, size):
    y = torch.cat((y, y[[0]]))
    t_in = torch.linspace(0, 1, len(y)).to(y)
    t_out = torch.linspace(0, 1, size).to(y)
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


class LatentAugmenter:
    def __init__(self, checkpoint, n_patches) -> None:
        model = StyleGAN2Mapper(checkpoint, inference=False).eval()
        self.n_patches = n_patches
        self.num = 16384
        self.ws = model.forward(latent_z=torch.randn((self.num, 512)))
        self.nw = self.ws.shape[1]
        self.feat_idxs = {
            # "mfccs": (0, 20),
            "chroma": (20, 32),
            "chroma": (20, 32),
            "chroma": (20, 32),
            "chroma": (20, 32),
            "tonnetz": (32, 38),
            "tonnetz": (32, 38),
            "tonnetz": (32, 38),
            "tonnetz": (32, 38),
            # "contrast": (38, 45),
            # "flatness": (45, 46),
            "onsets": (46, 47),
            "onsets_low": (47, 48),
            "onsets_mid": (48, 49),
            "onsets_high": (49, 50),
            # "pulse": (50, 51),
            "volume": (51, 52),
            "volume_low": (52, 53),
            "volume_mid": (53, 54),
            "volume_high": (54, 55),
            "volume_long": (55, 56),
            "volume_low_long": (56, 57),
            "volume_mid_long": (57, 58),
            "volume_high_long": (58, 59),
        }
        self.feat_keys = list(self.feat_idxs.keys())
        self.single_dim = -12

    def __call__(self, features):
        residuals, offsets = [], []
        for feature in features:
            residual, offset = self.random_patch(feature)
            residuals.append(residual)
            offsets.append(offset)
        return torch.stack(residuals), torch.stack(offsets)

    @torch.no_grad()
    def random_patch(self, feature):
        latent = spline_loop_latents(
            self.ws[np.random.randint(0, self.num, np.random.randint(3, 12))].to(feature), len(feature)
        )

        for _ in range(self.n_patches):
            start, stop = self.feat_idxs[random.choice(self.feat_keys)]

            if np.random.rand() > 0.5:
                lay_start = np.random.randint(0, self.nw - 6)
                lay_stop = np.random.randint(lay_start, self.nw)
            else:
                lay_start = 0
                lay_stop = self.nw

            if stop - start == 1:
                lat = self.ws[np.random.randint(0, self.num, 1)].to(feature)
                modulation = normalize(feature[:, start:stop, None])
                latent[:, lay_start:lay_stop] *= 1 - modulation
                latent[:, lay_start:lay_stop] += modulation * lat[:, lay_start:lay_stop]

            else:
                lats = self.ws[np.random.randint(0, self.num, stop - start)].to(feature)
                modulation = normalize(feature[:, start:stop])
                modulation /= modulation.sum(1, keepdim=True) + 1e-8
                patch_latent = torch.einsum("Atwl,Atwl->twl", modulation.permute(1, 0)[..., None, None], lats[:, None])

                if np.random.rand() > 0.666:
                    inter_start, inter_stop = self.feat_idxs[random.choice(self.feat_keys[self.single_dim :])]
                    intermodulator = normalize(feature[:, inter_start:inter_stop, None])
                    latent[:, lay_start:lay_stop] *= 1 - intermodulator
                    latent[:, lay_start:lay_stop] += intermodulator * patch_latent[:, lay_start:lay_stop]
                else:
                    latent[:, lay_start:lay_stop] = patch_latent[:, lay_start:lay_stop]

        offset = latent.mean(dim=(0, 1), keepdim=True)
        residuals = latent - offset
        return residuals.detach(), offset.detach()
