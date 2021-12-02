import os
from glob import glob
from pathlib import Path

import numpy as np
import pytorchvideo
import pytorchvideo.data
import torch
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, ShortSideScale, UniformTemporalSubsample
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
from torchvision.transforms import CenterCrop, Compose, Lambda, Resize
from tqdm import tqdm

from models.pixel2style2pixel import GradualStyleEncoder
from models.stylevideogan import StyleVideoDiscriminator, StyleVideoGenerator


@torch.inference_mode()
def preprocess_video(in_dir, out_file, duration, fps):
    if not os.path.exists(out_file):
        videos = sum([glob(in_dir + "/*" + ext) for ext in [".mp4", ".avi", ".mkv"]], [])
        dataset = pytorchvideo.data.LabeledVideoDataset(
            list(zip(videos, [{} for _ in range(len(videos))])),
            pytorchvideo.data.UniformClipSampler(clip_duration=duration),
            transform=ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(fps),
                        Lambda(lambda x: x / 255.0),
                        Normalize(0.5, 0.5),
                        ShortSideScale(size=1024),
                        CenterCrop(1024),
                        Resize(256),
                    ]
                ),
            ),
            decode_audio=False,
        )

        encoder = GradualStyleEncoder(num_layers=50, input_nc=3, n_styles=18, mode="ir_se").to(device)
        ckpt = torch.load("../pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt")
        latent_avg = ckpt["latent_avg"][None].to(device)
        ckpt = {k.replace("encoder.", ""): v for k, v in ckpt["state_dict"].items() if "encoder" in k}
        encoder.load_state_dict(ckpt)

        snippets = []
        for batch in tqdm(
            DataLoader(dataset, batch_size=1, num_workers=min(len(videos), 16)),
            desc="Encoding videos to latent sequences...",
        ):
            video = batch["video"].squeeze().permute(1, 0, 2, 3).to(device)
            latents = encoder(video) + latent_avg
            snippets.append(latents)
        snippets = torch.stack(snippets).cpu().numpy()
        np.save(out_file, snippets)


class LatentSequenceDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.sequences = np.load(file, mmap_mode="r")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index].copy()


def infiniter(data_loader):
    while True:
        for batch in data_loader:
            yield batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    in_dir = "/home/hans/datasets/groves/videos/17_21_35_aligned"
    duration = 1
    fps = 24
    seq_len = duration * fps
    batch_size = 16
    n_styles = 18
    latent_dim = 32
    nsteps = 100_000
    learning_rate = 1e-4
    b1 = 0.5
    b2 = 0.999
    critic_iter = 1
    lambda_gp = 50
    lambda_gap = 100

    cache_file = f"cache/{Path(in_dir).stem}_latents_preprocessed_{duration}sec_{fps}fps.npy"
    preprocess_video(in_dir, cache_file, duration, fps)

    G = StyleVideoGenerator(n_styles=n_styles, latent_dim=latent_dim).to(device)
    D = StyleVideoDiscriminator(seq_len=seq_len, n_styles=n_styles, latent_dim=latent_dim).to(device)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    G.apply(weights_init)
    D.apply(weights_init)

    G_ema = ExponentialMovingAverage(G.parameters(), decay=0.995)

    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(b1, b2))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(b1, b2))

    dataloader = infiniter(
        DataLoader(
            LatentSequenceDataset(cache_file), batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True
        )
    )

    is_real = torch.tensor(1.0, device=device)
    is_fake = torch.tensor(-1.0, device=device)

    with tqdm(range(nsteps)) as pbar:
        for step in pbar:
            for p in D.parameters():
                p.requires_grad = True
            for p in G.parameters():
                p.requires_grad = False

            for d_iter in range(critic_iter):
                D.zero_grad()

                real = next(dataloader).to(device)
                d_loss_real = D(real).mean()
                d_loss_real.backward(is_real)

                z = torch.randn((batch_size, seq_len, latent_dim), device=device)
                fake = G(z)
                d_loss_fake = D(fake).mean()
                d_loss_fake.backward(is_fake)

                eta = torch.FloatTensor(batch_size, 1, 1, 1).to(device).uniform_(0, 1)
                interpolated = eta * real + ((1 - eta) * fake)
                interpolated.requires_grad_(True)
                prob_interpolated = D(interpolated)
                gradients = torch.autograd.grad(
                    outputs=prob_interpolated,
                    inputs=interpolated,
                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
                gradient_penalty.backward()

                d_optimizer.step()

            for p in D.parameters():
                p.requires_grad = False
            for p in G.parameters():
                p.requires_grad = True

            G.zero_grad()
            z = torch.randn((batch_size, seq_len, latent_dim), device=device)
            z.requires_grad_()
            with torch.backends.cudnn.flags(enabled=False):  # Double backwards is not supported for CuDNN RNNs
                fake = G(z)
            g_loss = D(fake).mean()
            g_loss.backward(is_real, retain_graph=True)

            timestep_diff = G.l[-1] - G.l[0]
            G.update_gap_buffers(timestep_diff)
            l_var = G.l_sq - G.l_mu ** 2
            timestep_dist = (timestep_diff - G.l_mu) / l_var
            d = torch.sqrt(torch.sum(timestep_dist ** 2))

            gradients = torch.autograd.grad(
                outputs=d,
                inputs=z,
                create_graph=True,
                retain_graph=True,
            )[0]
            gradients = gradients.permute(1, 0, 2)  # N, L, latent_dim

            phi = torch.arctan(torch.norm(gradients[:, 1:], dim=(1, 2)) / torch.norm(gradients[:, [0]], dim=(1, 2)))

            loss_gap = (torch.minimum(torch.zeros_like(phi), phi - torch.pi / 4) ** 2).mean() * lambda_gap
            loss_gap.backward()

            g_optimizer.step()
            G_ema.update()

            pbar.write(
                f"d_loss_fake: {d_loss_fake.item():.4f}".ljust(30)
                + f"d_loss_real: {d_loss_real.item():.4f}".ljust(30)
                + f"gradient_penalty: {gradient_penalty.item():.4f}".ljust(30)
                + f"g_loss: {g_loss.item():.4f}".ljust(20)
                + f"dist: {d.mean().item():.4f}".ljust(20)
                + f"G.l_mu: {G.l_mu.mean().item():.4f}".ljust(20)
                + f"G.l_var: {l_var.mean().item():.4f}".ljust(20)
                + f"loss_gap: {loss_gap.item():.4f}".ljust(20)
            )
