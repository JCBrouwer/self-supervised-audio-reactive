import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .features.correlation import rv2
from .models.latent_n_noise2 import LatentNoiseReactor
from .supervised.data import audio2features, get_ffcv_dataloaders
from .supervised.test import audio2video

sys.path.append("/home/hans/code/maua/maua/")
from GAN.wrappers.stylegan2 import StyleGAN2Mapper

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STYLEGAN_CKPT = "cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl"


def get_output_shape(o):
    if isinstance(o, (tuple, list)):
        return tuple(get_output_shape(oo) for oo in o)
    elif isinstance(o, torch.Tensor):
        return tuple(o.shape)


@torch.no_grad()
def print_model_summary(model):
    global total_params
    total_params = 0
    handles = []
    for name, block in model.named_modules():

        def hook(m, i, o, name=name):
            global total_params
            if len(list(m.named_modules())) == 1:
                class_name = m.__class__.__name__
                output_shape = get_output_shape(o)
                num_params = sum(p.numel() for p in m.parameters())
                total_params += num_params
                print(
                    name.ljust(70),
                    class_name.ljust(30),
                    f"{output_shape}".ljust(40),
                    f"{num_params/ 1000:.2f} K" if num_params > 0 else "0",
                )

        handles.append(block.register_forward_hook(hook))

    print()
    print("model summary:")
    print("name".ljust(70), "class".ljust(30), "output shape".ljust(40), "num params")
    print("-" * 140)
    model(inputs.to(device))
    for handle in handles:
        handle.remove()
    print("-" * 140)
    print("total".ljust(70), f"".ljust(30), f"".ljust(40), f"{total_params/1e6:.2f} M")


@torch.no_grad()
def validate():

    val_loss, latent_residuals = 0, []
    for _, (inputs, latents, n4, n8, n16, n32) in enumerate(val_dataloader):
        latents, noise = model(inputs)
        latent_residuals.append(np.random.choice(latents.cpu().numpy().flatten(), 100_000))
        val_loss += sum([F.mse_loss(o, t) for o, t in zip([latents] + noise, [latents, n4, n8, n16, n32])])
    val_loss /= len(val_dataloader)
    writer.add_scalar("Loss/val", val_loss.item(), it)

    try:
        loc, scale = stats.laplace.fit(np.concatenate(latent_residuals), loc=0, scale=0.1)
        writer.add_scalar("Eval/laplace_b", scale.item(), it)
    except Exception as e:
        progress.write(f"\nError in Laplace fit:\n{e}\n\n")
        scale = -1

    return val_loss, scale


@torch.jit.script
def rv2_loss(predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros(len(predictions[0]), device=predictions[0].device)
    for b in range(len(predictions)):
        for p in predictions[b]:
            for t in targets[b]:
                loss[b] = loss[b] + 1 - rv2(p, t)
    return loss


def infiniter(data_loader):
    while True:
        for batch in data_loader:
            yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model options
    parser.add_argument("--decoder", type=str, default="learned", choices=["learned", "fixed"])
    parser.add_argument("--backbone", type=str, default="sashimi", choices=["sashimi", "gru"])
    parser.add_argument("--n_latent_split", type=int, default=3, choices=[1, 2, 3, 6, 9, 18])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)

    # Loss options
    parser.add_argument("--loss", type=str, default="supervised", choices=["supervised", "selfsupervised"])

    # Training options
    parser.add_argument("--n_examples", type=int, default=1_024_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=10_240)
    parser.add_argument("--ckpt_every", type=int, default=5 * 10_240)

    args = parser.parse_args()

    dur, fps = args.duration, args.fps
    batch_size = args.batch_size

    n_frames = dur * fps
    in_dir = "/home/hans/datasets/audio2latent/"
    dataset_cache = f"cache/{Path(in_dir).stem}_preprocessed_{n_frames}frames.npy"
    test_audio = "/home/hans/datasets/wavefunk/naamloos.wav"

    train_mean, train_std, train_dataloader, val_dataloader = get_ffcv_dataloaders(in_dir, batch_size, dur, fps)
    train_iter = infiniter(train_dataloader)

    inputs, latents, n4, n8, n16, n32 = next(iter(train_dataloader))
    feature_dim = inputs.shape[2]
    n_outputs, output_size = latents.shape[2], latents.shape[3]

    mapper = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False)
    latents = mapper(torch.from_numpy(np.random.RandomState(42).randn(args.n_latent_split * args.hidden_size, 512))).to(
        device
    )
    del mapper

    name = "_".join(
        [
            args.loss,
            args.decoder,
            "decoder",
            f"n_latent_split:{args.n_latent_split}",
            f"hidden_size:{args.hidden_size}",
            f"num_layers:{args.num_layers}",
            f"dropout:{args.dropout}",
            f"lr:{args.lr}",
        ]
    )
    model = LatentNoiseReactor(
        train_mean,
        train_std,
        feature_dim,
        latents,
        num_layers=args.num_layers,
        backbone=args.backbone,
        hidden_size=args.hidden_size,
        decoder=args.decoder,
        n_latent_split=args.n_latent_split,
        n_noise=4,
        dropout=args.dropout,
    ).to(device)
    print_model_summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(comment=name)
    shutil.copytree(os.path.dirname(__file__), writer.log_dir + "/ssar")

    losses = []
    with tqdm(range(0, args.n_examples, batch_size)) as progress:
        for it in progress:
            if args.loss == "supervised":
                inputs, latents, n4, n8, n16, n32 = next(train_iter)

                latents, noise = model(inputs)
                loss = sum([F.mse_loss(o, t) for o, t in zip([latents] + noise, [latents, n4, n8, n16, n32])])
                print(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/supervised", loss.item(), it)
                losses.append(loss.item())

            elif args.loss == "selfsupervised":
                inputs, _, _, _, _, _ = next(train_iter)
                inputs = inputs.to(device)

                latents, noise = model(inputs)
                loss = rv2_loss([latents] + noise, inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/selfsupervised", loss.item(), it)
                losses.append(loss.item())

            if it % args.eval_every == 0 and it != 0:
                model.eval()
                val_loss, scale = validate()
                progress.write("")
                progress.write(f"iter {it}")
                progress.write(f"train_loss: {np.mean(losses):.4f}")
                progress.write(f"val_loss  : {val_loss:.4f}")
                progress.write(f"laplace_b : {scale:.4f}")
                progress.write("")
                losses = []
                model.train()

            if it % args.ckpt_every == 0 and it != 0:
                model.eval()
                checkpoint_name = f"audio2latent_{name}_steps{it:08}_b{scale:.4f}_val{val_loss:.4f}"
                joblib.dump(
                    {"model": model, "optim": optimizer, "n_iter": it},
                    f"{writer.log_dir}/{checkpoint_name}.pt",
                    compress=9,
                )
                audio2video(
                    a2l=model,
                    a2f=audio2features,
                    audio_file=test_audio,
                    out_file=f"{writer.log_dir}/{checkpoint_name}_{Path(test_audio).stem}.mp4",
                    stylegan_file=STYLEGAN_CKPT,
                    output_size=(1024, 1024),
                )
                model.train()

    writer.close()
