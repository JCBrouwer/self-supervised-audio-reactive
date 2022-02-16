import argparse
import importlib
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio

from train_supervised import audio2features, audio2video, device

# fmt:off
parser = argparse.ArgumentParser()
parser.add_argument("ckpt", help="path to Audio2Latent checkpoint to test")
parser.add_argument("audio", help="path to audio file to test with")
parser.add_argument("--stylegan", help="path to StyleGAN model file to test with", default="/home/hans/modelzoo/train_checks/neurout2-117.pt")
parser.add_argument("--output_size", help="output size for StyleGAN model rendering", nargs=2, default=[512, 512])
parser.add_argument("--batch_size", help="batch size for StyleGAN model rendering", type=int, default=8)
parser.add_argument("--offset", help="time in audio to start from in seconds", type=int, default=0)
parser.add_argument("--duration", help="length in seconds of video to render", type=int, default=40)
args = parser.parse_args()
# fmt:on

if __name__ == "__main__":
    with torch.inference_mode():
        sys.path = [os.path.dirname(args.ckpt)] + sys.path
        import train_supervised

        importlib.reload(train_supervised)

        from train_supervised import *

        if args.duration > 0:
            audio2video(
                args.ckpt,
                args.audio,
                args.stylegan,
                output_size=[int(s) for s in args.output_size],
                batch_size=args.batch_size,
                offset=args.offset,
                duration=args.duration,
            )
        else:
            out_file = f"plots/latent-residuals-distribution-{Path(args.audio).stem}-{Path(os.path.dirname(args.ckpt)).stem}-{Path(args.ckpt).stem}.png"
            if not os.path.exists(out_file):
                a2l = joblib.load(args.ckpt)["a2l"].eval().to(device)

                # for block in a2l.backbone.modules():
                #     block.register_forward_hook(lambda m, i, o: print(o.shape))

                latent_residuals = (
                    a2l(audio2features(*torchaudio.load(args.audio)).to(device).unsqueeze(0)).flatten().cpu().numpy()
                )

                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from scipy import stats

                plt.figure(figsize=(16, 9))
                plt.hist(
                    latent_residuals,
                    bins=1000,
                    label=f"Histogram (min={latent_residuals.min():.2f}, mean={latent_residuals.mean():.2f}, max={latent_residuals.max():.2f})",
                    color="tab:blue",
                    alpha=0.5,
                    density=True,
                )
                loc, scale = stats.laplace.fit(latent_residuals, loc=0, scale=0.1)
                xs = np.linspace(-2, 2, 1000)
                plt.plot(
                    xs,
                    stats.laplace.pdf(xs, loc=loc, scale=scale),
                    label=rf"PDF of MLE-fit Laplace($\mu$={loc:.2f}, $b$={scale:.2f})",
                    color="tab:orange",
                    alpha=0.75,
                )
                plt.title("Distribution of latent residuals")
                plt.legend()
                plt.xlim(-2, 2)
                plt.tight_layout()
                plt.savefig(out_file)
