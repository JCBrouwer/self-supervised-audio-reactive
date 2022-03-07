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

from ..models.latent_n_noise import Reactor
from .context_fid import calculate_fcd
from .data import audio2features, get_ffcv_dataloaders
from .test import audio2video

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_model_summary(model):
    print()
    print("model summary:")
    print("name".ljust(50), "class".ljust(30), "output shape".ljust(40), "num params")
    print("-" * 140)
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
    model(inputs.to(device))
    for handle in handles:
        handle.remove()
    print("-" * 140)
    print("total".ljust(50), f"".ljust(30), f"".ljust(40), f"{total_params/1e6:.2f} M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Shared options
    parser.add_argument("--n_layerwise", type=int, default=3, choices=[1, 2, 3, 6, 9, 18])
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)

    # Training options
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    dur, fps = args.duration, args.fps
    batch_size = args.batch_size
    n_layerwise = args.n_layerwise
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    lr = args.lr

    n_frames = dur * fps
    in_dir = "/home/hans/datasets/audio2latent/"
    dataset_cache = f"cache/{Path(in_dir).stem}_preprocessed_{n_frames}frames.npy"
    test_audio = "/home/hans/datasets/wavefunk/Ouroboromorphism_49_89.flac"

    train_mean, train_std, train_dataloader, val_dataloader = get_ffcv_dataloaders(in_dir, False, batch_size, dur, fps)

    inputs, latents, noise4, noise8, noise16, noise32 = next(iter(train_dataloader))
    feature_dim = inputs.shape[2]
    n_outputs, output_size = latents.shape[2], latents.shape[3]

    name = "_".join(
        [
            "reactor",
            f"layerwise:{n_layerwise}",
            f"hidden_size:{hidden_size}",
            f"num_layers:{num_layers}",
            f"dropout:{dropout}",
            f"lr:{lr}",
        ]
    )
    a2l = Reactor(
        input_mean=train_mean,
        input_std=train_std,
        dim_in=feature_dim,
        dim=hidden_size,
        n_hid_latents=n_layerwise,
        n_out_latents=n_outputs,
        latent_dim=output_size,
    ).to(device)
    print_model_summary(a2l)

    optimizer = torch.optim.Adam(a2l.parameters(), lr=lr)

    writer = SummaryWriter(comment=name)
    shutil.copytree(os.path.dirname(__file__) + "/../../ssar", writer.log_dir + "/ssar")

    n_iter = 0
    n_epochs = args.epochs
    video_interval = 100
    eval_interval = 20
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        losses = []
        a2l.train()

        for inputs, latents, noise4, noise8, noise16, noise32 in train_dataloader:
            inputs, targets = inputs.to(device), [t.to(device) for t in [latents, noise4, noise8, noise16, noise32]]

            outputs = a2l(inputs)
            loss = sum([F.mse_loss(o, t) for o, t in zip(outputs, targets)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), n_iter)
            losses.append(loss.item())

            n_iter += len(inputs)

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == n_epochs:
            with torch.inference_mode():
                a2l.eval()

                val_loss, latent_residuals = 0, []
                for it, (inputs, latents, noise4, noise8, noise16, noise32) in enumerate(val_dataloader):
                    inputs = inputs.to(device)
                    targets = [t.to(device) for t in [latents, noise4, noise8, noise16, noise32]]
                    outputs = a2l(inputs)
                    latent_residuals.append(np.random.choice(outputs[0].cpu().numpy().flatten(), 100_000))
                    val_loss += sum([F.mse_loss(o, t) for o, t in zip(outputs, targets)])
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
