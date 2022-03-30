import argparse
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.audio2latent import Audio2Latent
from ..models.audio2latent2 import Audio2Latent2
from ..models.convnext import ConvNeXt
from ..models.mlp import MLP
from .context_fid import calculate_fcd
from .data import audio2features, get_ffcv_dataloaders
from .latent_augmenter import LatentAugmenter
from .test import audio2video

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_model_summary(model):
    print()
    print("model summary:")
    print("name".ljust(50), "class".ljust(20), "output shape".ljust(40), "num params")
    print("-" * 130)
    global total_params
    total_params = 0
    handles = []
    for name, block in a2l.named_modules():

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
                    class_name.ljust(20),
                    f"{output_shape}".ljust(40),
                    f"{num_params/ 1000:.2f} K" if num_params > 0 else "0",
                )

        handles.append(block.register_forward_hook(hook))
    a2l(inputs.to(device))
    for handle in handles:
        handle.remove()
    print("-" * 130)
    print("total".ljust(50), f"".ljust(20), f"".ljust(40), f"{total_params/1e6:.2f} M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Audio2Latent options
    parser.add_argument("--backbone", type=str, default=None, choices=["gru", "lstm", "conv"])
    parser.add_argument("--skip_backbone", action="store_true")
    parser.add_argument("--layerwise", type=str, default=None, choices=["dense", "conv"])

    # Audio2Latent2 options
    parser.add_argument("--context", type=str, default=None, choices=["gru", "lstm", "qrnn", "conv", "transformer"])
    parser.add_argument("--correlation", type=str, default=None, choices=["linear", "eca", "cba"])

    # Other architectures
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--convnext", action="store_true")

    # Shared options
    parser.add_argument("--n_layerwise", type=int, default=6, choices=[1, 2, 3, 6, 9, 18])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)

    # Training options
    parser.add_argument("--aug_weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    dur, fps = args.duration, args.fps
    n_frames = dur * fps

    in_dir = "/home/hans/datasets/audio2latent/"
    dataset_cache = f"cache/{Path(in_dir).stem}_preprocessed_{n_frames}frames.npy"
    test_audio = "/home/hans/datasets/wavefunk/Ouroboromorphism_49_89.flac"

    batch_size = args.batch_size

    backbone = args.backbone
    skip_backbone = args.skip_backbone
    context = args.context
    correlation = args.correlation
    layerwise = args.layerwise
    n_layerwise = args.n_layerwise
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout

    lr = args.lr
    wd = args.wd
    aug_weight = args.aug_weight

    train_mean, train_std, train_dataloader, val_dataloader = get_ffcv_dataloaders(
        in_dir, batch_size, dur, fps
    )

    if aug_weight > 0:
        augmenter = LatentAugmenter(
            checkpoint="/home/hans/modelzoo/train_checks/neurout2-117.pt", n_patches=3
        )

    inputs, targets = next(iter(train_dataloader))
    feature_dim = inputs.shape[2]
    n_outputs, output_size = targets.shape[2], targets.shape[3]

    if args.backbone is not None:
        name = "_".join(
            [
                f"backbone:{backbone}:skip{skip_backbone}",
                f"layerwise:{layerwise}:{n_layerwise}",
                f"hidden_size:{hidden_size}",
                f"num_layers:{num_layers}",
                f"dropout:{dropout}",
                f"lr:{lr}",
                f"wd:{wd}",
            ]
        )
        a2l = Audio2Latent(
            input_mean=train_mean,
            input_std=train_std,
            backbone=backbone,
            skip_backbone=skip_backbone,
            layerwise=layerwise,
            n_layerwise=n_layerwise,
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_outputs=n_outputs,
            output_size=output_size,
        ).to(device)
    elif args.context is not None:
        name = "_".join(
            [
                f"context:{context}",
                f"correlation:{correlation}",
                f"n_layerwise:{n_layerwise}",
                f"hidden_size:{hidden_size}",
                f"num_layers:{num_layers}",
                f"dropout:{dropout}",
                f"lr:{lr}",
                f"wd:{wd}",
            ]
        )
        a2l = Audio2Latent2(
            context=context,
            correlation=correlation,
            input_mean=train_mean,
            input_std=train_std,
            n_layerwise=n_layerwise,
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_outputs=n_outputs,
            output_size=output_size,
            dropout=dropout,
        ).to(device)
    elif args.mlp:
        name = "_".join(
            [
                f"mlp",
                f"layerwise:{n_layerwise}",
                f"hidden_size:{hidden_size}",
                f"num_layers:{num_layers}",
                f"dropout:{dropout}",
                f"lr:{lr}",
                f"wd:{wd}",
            ]
        )
        a2l = MLP(
            input_mean=train_mean,
            input_std=train_std,
            in_channels=feature_dim,
            channels=hidden_size,
            out_channels=output_size,
            n_outputs=n_outputs,
            n_layerwise=n_layerwise,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    elif args.convnext:
        name = "_".join(
            [
                f"convnext",
                f"layerwise:conv:{n_layerwise}",
                f"hidden_size:{hidden_size}",
                f"dropout:{dropout}",
                f"lr:{lr}",
            ]
        )
        a2l = ConvNeXt(
            input_mean=train_mean,
            input_std=train_std,
            input_size=feature_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            n_outputs=n_outputs,
            n_layerwise=n_layerwise,
            drop_path_rate=dropout,
        ).to(device)
    else:
        raise NotImplementedError()
    a2l(inputs.to(device))
    print_model_summary(a2l)

    if aug_weight > 0:
        name += "_augmented"

    optimizer = torch.optim.AdamW(a2l.parameters(), lr=lr, weight_decay=wd)

    writer = SummaryWriter(comment=name)
    shutil.copytree(os.path.dirname(__file__) + "/../../ssar", writer.log_dir + "/ssar")

    # from time import time

    n_iter = 0
    n_epochs = args.epochs
    video_interval = 20
    eval_interval = 5
    pbar = tqdm(range(n_epochs))
    grad_noise = []
    for epoch in pbar:
        losses, aug_losses = [], []
        a2l.train()

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if aug_weight > 0:
                (inputs, aug_inputs), (targets, _) = inputs.chunk(2), targets.chunk(2)

            outputs = a2l(inputs)
            loss = F.mse_loss(outputs, targets)

            if aug_weight > 0:
                aug_inputs = aug_inputs.to(device)
                aug_outputs = a2l(aug_inputs)
                aug_targets, _ = augmenter(aug_inputs)
                aug_loss = aug_weight * F.mse_loss(aug_outputs, aug_targets)
            else:
                aug_loss = torch.zeros(())

            optimizer.zero_grad()
            (loss + aug_loss).backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), n_iter)
            writer.add_scalar("Loss/augtrain", aug_loss.item(), n_iter)
            losses.append(loss.item())
            aug_losses.append(aug_loss.item())

            n_iter += len(inputs)

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == n_epochs:
            with torch.inference_mode():
                a2l.eval()

                val_loss, latent_residuals = 0, []
                for it, (inputs, targets) in enumerate(val_dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = a2l(inputs)
                    latent_residuals.append(np.random.choice(outputs.cpu().numpy().flatten(), 100_000))
                    val_loss += F.mse_loss(outputs, targets)
                val_loss /= len(val_dataloader)
                writer.add_scalar("Loss/val", val_loss.item(), n_iter)

                try:
                    fcd = calculate_fcd(val_dataloader, a2l)
                    writer.add_scalar("Eval/FCD", fcd.item(), n_iter)
                except Exception as e:
                    pbar.write(f"\nError in FCD:\n{e}\n\n")
                    fcd = -1

                try:
                    loc, scale = stats.laplace.fit(np.concatenate(latent_residuals), loc=0, scale=0.1)
                    writer.add_scalar("Eval/laplace_b", scale.item(), n_iter)
                except Exception as e:
                    pbar.write(f"\nError in Laplace fit:\n{e}\n\n")
                    scale = -1

            pbar.write("")
            pbar.write(f"epoch {epoch + 1}")
            pbar.write(f"train_loss: {np.mean(losses):.4f}")
            pbar.write(f"aug_loss  : {np.mean(aug_losses):.4f}")
            pbar.write(f"val_loss  : {val_loss:.4f}")
            pbar.write(f"laplace_b : {scale:.4f}")
            pbar.write(f"fcd       : {fcd:.4f}")
            pbar.write("")

        if (epoch + 1) % video_interval == 0 or (epoch + 1) == n_epochs:
            checkpoint_name = f"audio2latent_{name}_steps{n_iter:08}_fcd{fcd:.4f}_b{scale:.4f}_val{val_loss:.4f}"
            joblib.dump(
                {"a2l": a2l, "optim": optimizer, "n_iter": n_iter}, f"{writer.log_dir}/{checkpoint_name}.pt", compress=9
            )
            audio2video(
                a2l=a2l,
                a2f=audio2features,
                audio_file=test_audio,
                out_file=f"{writer.log_dir}/{checkpoint_name}_{Path(test_audio).stem}.mp4",
                stylegan_file="/home/hans/modelzoo/train_checks/neurout2-117.pt",
            )

    writer.close()
