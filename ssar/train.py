import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

import joblib
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from functorch import vmap
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .features.video import absdiff
from .models.latent_n_noise2 import LatentNoiseReactor
from .supervised.data import audio2features, get_ffcv_dataloaders
from .supervised.test import audio2video

sys.path.append("/home/hans/code/maua/")
from maua.GAN.wrappers.stylegan2 import StyleGAN2Mapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STYLEGAN_CKPT = "cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pkl"


batch_abs_diff = vmap(absdiff)


def get_output_shape(o):
    if isinstance(o, (tuple, list)):
        return tuple(get_output_shape(oo) for oo in o)
    elif isinstance(o, torch.Tensor):
        return tuple(o.shape)


@torch.no_grad()
def print_model_summary(model, example_inputs):
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
    model(example_inputs.to(device))
    for handle in handles:
        handle.remove()
    print("-" * 140)
    print("total".ljust(70), f"".ljust(30), f"".ljust(40), f"{total_params/1e3:.2f} K")


def torchrand(pop_size, num_samples, device: str):
    vec = torch.unique((torch.rand(num_samples, device=device) * pop_size).floor().long())
    while vec.shape[0] != num_samples:
        vec = torch.unique(
            torch.cat(
                [
                    vec,
                    (torch.rand(num_samples - vec.shape[0], device=device) * pop_size).floor().long(),
                ]
            )
        )
    return vec.view(-1)


@torch.no_grad()
def validate():
    val_loss, latent_residuals = 0, []
    for _, (inputs, latents, n4, n8, n16, n32) in enumerate(val_dataloader):
        latents, noise = model(inputs)
        flatlats = latents.flatten()
        flatlats = flatlats[torchrand(flatlats.numel(), 100_000, flatlats.device)]
        latent_residuals.append(flatlats.cpu().numpy())
        val_loss += sum([F.mse_loss(o, t) for o, t in zip([latents] + noise, [latents, n4, n8, n16, n32])])
    val_loss /= len(val_dataloader)
    writer.add_scalar("Loss/val", val_loss.item(), it)

    # try:
    #     loc, scale = stats.laplace.fit(np.concatenate(latent_residuals), loc=0, scale=0.1)
    #     writer.add_scalar("Eval/laplace_b", scale.item(), it)
    # except Exception as e:
    #     progress.write(f"\nError in Laplace fit:\n{e}\n\n")
    #     scale = -1

    envelopes = model.forward(inputs, return_envelopes=True)
    n_env = envelopes[0].shape[-1]
    most_correlated = torch.topk(
        torch.tensor(
            [1 - orthogonal_procrustes_distance(inp.unsqueeze(-1), envelopes[0]) for inp in inputs[0].unbind(-1)]
        ),
        k=n_env,
    ).indices
    _, ax = plt.subplots(n_env + 1, 2, figsize=(8, 4 * (n_env + 1)))
    [x.axis("off") for x in ax.flatten()]

    def sum_ac(tensors):
        feats = [af - af.min() for af in tensors]
        feats = [af / af.max() for af in feats]
        ac = torch.stack([af[:, None] @ af[:, None].T for af in feats])
        ac = torch.sum(ac, dim=0)
        ac = ac - ac.min()
        ac = ac / ac.max()
        return ac

    iac = sum_ac(inputs[0].unbind(-1))
    ax[0, 0].imshow(iac.detach().cpu().numpy())
    ax[0, 0].set_title("sum of normalized input envelopes")

    iac = sum_ac(envelopes[0].unbind(-1))
    ax[0, 1].imshow(iac.detach().cpu().numpy())
    ax[0, 1].set_title("sum of normalized generated envelopes")

    for e, env in enumerate(envelopes[0].unbind(-1)):
        ienv = inputs[0, :, most_correlated[e]]
        ax[e + 1, 0].imshow((ienv[:, None] @ ienv[:, None].T).detach().cpu().numpy())
        ax[e + 1, 1].imshow((env[:, None] @ env[:, None].T).detach().cpu().numpy())
    ax[1, 0].set_title("most correlated input envelopes")
    ax[1, 1].set_title("generated intermediate envelopes")

    plt.tight_layout()
    plt.savefig(f"{writer.log_dir}/envelopes_{it}.pdf")

    return val_loss  # , scale


def infiniter(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def orthogonal_procrustes_distance(x, y):
    x = x - x.mean(dim=0, keepdim=True)
    x = x / torch.linalg.norm(x, ord="fro")
    y = y - y.mean(dim=0, keepdim=True)
    y = y / torch.linalg.norm(y, ord="fro")
    return 1 - torch.linalg.norm(x.t() @ y, ord="nuc")


def audio_reactive_loss(afeats, vfeats):
    if isinstance(afeats, dict):
        afeats, vfeats = list(afeats.values()), list(vfeats.values())
    return torch.stack(
        [
            orthogonal_procrustes_distance(a, v)
            for a, v in zip(
                torch.cat([af.flatten(2) for af in afeats], dim=2),
                torch.cat([vf.flatten(2) for vf in vfeats], dim=2),
            )
        ]
    )


class NormalizeGradients(torch.autograd.Function):
    """Normalize gradient scale in the backward pass"""

    @staticmethod
    def forward(self, input_tensor, strength=1):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = self.strength * grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input, None


normalize_gradients = NormalizeGradients.apply

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model options
    parser.add_argument("--decoder", type=str, default="learned", choices=["learned", "fixed"])
    parser.add_argument(
        "--backbone", type=str, default="gru", choices=["sashimi", "gru", "lstm", "transformer", "conv", "mlp"]
    )
    parser.add_argument("--n_latent_split", type=int, default=3, choices=[1, 2, 3, 6, 9, 18])
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)

    # Loss options
    parser.add_argument("--loss", type=str, default="supervised", choices=["supervised", "selfsupervised", "ssabsdiff"])
    parser.add_argument("--residual", action="store_true")

    # Training options
    parser.add_argument("--n_examples", type=int, default=128_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=10_240)
    parser.add_argument("--ckpt_every", type=int, default=10_240)

    args = parser.parse_args()

    dur, fps = args.duration, args.fps
    batch_size = args.batch_size

    n_frames = dur * fps
    in_dir = "/home/hans/datasets/audio2latent/"
    dataset_cache = f"cache/{Path(in_dir).stem}_preprocessed_{n_frames}frames.npy"
    test_audio = "/home/hans/datasets/wavefunk/naamloos.wav"

    train_mean, train_std, train_dataloader, val_dataloader = get_ffcv_dataloaders(in_dir, batch_size, dur, fps)
    train_iter = infiniter(train_dataloader)

    example_inputs, example_latents, _, _, _, _ = next(iter(train_dataloader))
    feature_dim = example_inputs.shape[2]
    n_outputs, output_size = example_latents.shape[2], example_latents.shape[3]

    mapper = StyleGAN2Mapper(model_file=STYLEGAN_CKPT, inference=False)
    decoder_latents = mapper(
        torch.from_numpy(np.random.RandomState(42).randn(args.n_latent_split * args.hidden_size, 512))
    ).to(device)
    del mapper

    model = LatentNoiseReactor(
        train_mean,
        train_std,
        feature_dim,
        decoder_latents,
        residual=args.residual,
        num_layers=args.num_layers,
        backbone=args.backbone,
        hidden_size=args.hidden_size,
        decoder=args.decoder,
        n_latent_split=args.n_latent_split,
        n_noise=4,
        dropout=args.dropout,
    ).to(device)
    model.train()
    print_model_summary(model, example_inputs)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    name = "_".join(
        [
            "",
            args.backbone,
            args.loss,
            args.decoder,
            "decoder",
            f"residual:{args.residual}",
            f"n_latent_split:{args.n_latent_split}",
            f"hidden_size:{args.hidden_size}",
            f"num_layers:{args.num_layers}",
            f"dropout:{args.dropout}",
            f"lr:{args.lr}",
        ]
    )
    writer = SummaryWriter(comment=name)
    shutil.copytree(os.path.dirname(__file__), writer.log_dir + "/ssar")

    losses = []
    with tqdm(range(0, args.n_examples, batch_size)) as progress:
        for it in progress:
            if args.loss == "supervised":
                inputs, latents, n4, n8, n16, n32 = next(train_iter)
                if args.residual:
                    latents -= latents.mean(dim=1, keepdim=True)

                pred_lats, pred_noise = model(inputs)
                loss = sum([F.mse_loss(o, t) for o, t in zip([pred_lats] + pred_noise, [latents, n4, n8, n16, n32])])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/supervised", loss.item(), it)
                losses.append(loss.item())

            elif args.loss == "selfsupervised":
                inputs, _, _, _, _, _ = next(train_iter)
                inputs = inputs.to(device)

                latents, noise = model(inputs)
                predictions = [latents] + noise
                # predictions = [normalize_gradients(p) for p in predictions]  # equalize gradient contributions
                loss = audio_reactive_loss(predictions, [inputs]).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/selfsupervised", loss.item(), it)
                losses.append(loss.item())

            elif args.loss == "ssabsdiff":
                inputs, _, _, _, _, _ = next(train_iter)
                inputs = inputs.to(device)

                latents, noise = model(inputs)
                predictions = [latents] + noise
                # predictions = [normalize_gradients(p) for p in predictions]  # equalize gradient contributions
                predictions = [batch_abs_diff(p) for p in predictions]
                loss = audio_reactive_loss(predictions, [inputs]).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/selfsupervised", loss.item(), it)
                losses.append(loss.item())

            if it % args.eval_every == 0:  # and it != 0:
                model.eval()
                val_loss = validate()
                progress.write("")
                progress.write(f"iter {it}")
                progress.write(f"train_loss: {np.mean(losses):.4f}")
                progress.write(f"val_loss  : {val_loss:.4f}")
                # progress.write(f"laplace_b : {scale:.4f}")
                progress.write("")
                losses = []
                model.train()

            if it % args.ckpt_every == 0:  # and it != 0:
                model.eval()
                checkpoint_name = f"audio2latent_{name}_steps{it:08}_val{val_loss:.4f}"
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
                    seed=None,
                    residual=args.residual,
                )
                model.train()

    checkpoint_name = f"audio2latent_{name}_steps{it:08}_val{val_loss:.4f}"
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
        seed=None,
        residual=args.residual,
    )

    writer.close()
