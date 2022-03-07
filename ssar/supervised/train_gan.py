# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import matplotlib

matplotlib.use("Agg")
import os
import shutil
from math import log2
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from umap import UMAP

from ..models.psagan import ProgressiveDiscriminator, ProgressiveGenerator
from .context_fid import calculate_fcd
from .data import audio2features, get_ffcv_dataloaders
from .latent_augmenter import LatentAugmenter
from .test import audio2video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infiniter(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def print_model_summary(G, D):
    global total_params
    total_params = 0
    handles = []
    for name, block in list(G.named_modules()) + list(D.named_modules()):

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
                    name.ljust(60),
                    class_name.ljust(20),
                    f"{output_shape}".ljust(40),
                    f"{num_params/ 1000:.2f} K" if num_params > 0 else "0",
                )

        handles.append(block.register_forward_hook(hook))

    print()
    print("G summary:")
    print("name".ljust(60), "class".ljust(20), "output shape".ljust(40), "num params")
    print("-" * 150)
    G.depth = G.n_stage
    output = G(inputs.permute(0, 2, 1).to(device))
    G.depth = 0
    print("-" * 150)
    print("total".ljust(60), f"".ljust(20), f"".ljust(40), f"{total_params/1e6:.2f} M")

    print()
    print("D summary:")
    print("name".ljust(60), "class".ljust(20), "output shape".ljust(40), "num params")
    print("-" * 150)
    D.depth = D.n_stage
    D(output, inputs.permute(0, 2, 1).to(device))
    D.depth = 0
    print("-" * 150)
    print("total".ljust(60), f"".ljust(20), f"".ljust(40), f"{total_params/1e6:.2f} M")

    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    target_len = 128
    n_channels = 32
    n_epoch_per_layer = 600
    n_epoch_fade_in_new_layer = 200
    epoch_len = 128
    ks_conv = 3
    key_features = value_features = 32
    pos_emb_dim = 16
    ks_value = ks_query = ks_key = 1
    n_layerwise = 3
    batch_size = 128
    lr = 5e-4
    # aug_weight = 0.5
    synthetic = False

    fps = 24
    dur = target_len / fps

    in_dir = "/home/hans/datasets/audio2latent/"
    dataset_cache = f"cache/{Path(in_dir).stem}_preprocessed_{target_len}frames.npy"
    test_audio = "/home/hans/datasets/wavefunk/Ouroboromorphism_49_89.flac"

    train_mean, train_std, train_dataloader, val_dataloader = get_ffcv_dataloaders(
        in_dir, synthetic, batch_size, dur, fps
    )
    train_mean, train_std = train_mean[None, :, None], train_std[None, :, None]
    valiter = infiniter(val_dataloader)
    trainiter = infiniter(train_dataloader)

    # if aug_weight > 0:
    #     augmenter = LatentAugmenter(
    #         checkpoint="/home/hans/modelzoo/train_checks/neurout2-117.pt", n_patches=3, synthetic=synthetic
    #     )

    inputs, targets = next(trainiter)
    n_features = inputs.shape[2]
    n_outputs, output_size = targets.shape[2], targets.shape[3]

    G = ProgressiveGenerator(
        input_mean=train_mean,
        input_std=train_std,
        target_len=target_len,
        n_features=n_features,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_layerwise=n_layerwise,
        output_size=output_size,
        ks_conv=ks_conv,
        key_features=key_features,
        value_features=value_features,
        ks_value=ks_value,
        ks_query=ks_query,
        ks_key=ks_key,
        pos_emb_dim=pos_emb_dim,
        self_attention=True,
        n_epoch_per_layer=n_epoch_per_layer,
        n_epoch_fade_in_new_layer=n_epoch_fade_in_new_layer,
    ).to(device)

    D = ProgressiveDiscriminator(
        input_mean=train_mean,
        input_std=train_std,
        target_len=target_len,
        n_features=n_features,
        n_channels=n_channels,
        n_outputs=n_outputs,
        output_size=output_size,
        ks_conv=ks_conv,
        key_features=key_features,
        value_features=value_features,
        ks_value=ks_value,
        ks_query=ks_query,
        ks_key=ks_key,
        pos_emb_dim=pos_emb_dim,
        self_attention=True,
        n_epoch_per_layer=n_epoch_per_layer,
        n_epoch_fade_in_new_layer=n_epoch_fade_in_new_layer,
    ).to(device)
    n_epochs = D.epochs

    print_model_summary(G, D)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0, 0.99))

    name = "_".join(["PSAGAN", f"length:{target_len}", f"hidden_size:{n_channels}", f"lr:{lr}"])
    writer = SummaryWriter(comment=name)
    shutil.copytree(os.path.dirname(__file__) + "/../../ssar", writer.log_dir + "/ssar")

    n_iter = 0
    video_interval = 100
    eval_interval = 20
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:

        G.update_depth(epoch), D.update_depth(epoch)
        scale_factor = 2 ** (int(log2(D.target_len)) - (3 + G.depth))
        G.train(), D.train()

        for _ in range(epoch_len):
            # load data
            features, targets = next(trainiter)
            features, targets = features.to(device).permute(0, 2, 1), targets.to(device).permute(0, 2, 3, 1)
            b, n, l, t = targets.size()
            targets = F.avg_pool1d(targets.reshape(b, -1, t), kernel_size=scale_factor).reshape(b, n, l, -1)

            # Generator step

            for p in D.parameters():
                p.requires_grad = False

            generated = G(features)

            loss = D(generated, features) - 1
            loss_ls = 0.5 * torch.square(loss).mean()

            loss_std = torch.abs(generated.std(dim=(1, 2)) - targets.std(dim=(1, 2))).mean()
            loss_mean = torch.abs(generated.mean(dim=(1, 2)) - targets.mean(dim=(1, 2))).mean()
            loss_moment = loss_std + loss_mean

            loss_G = loss_ls + loss_moment

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            writer.add_scalar("G/loss_ls", loss_ls.item(), n_iter)
            writer.add_scalar("G/loss_moment", loss_moment.item(), n_iter)

            # Discriminator step

            for p in D.parameters():
                p.requires_grad = True

            features, targets = next(trainiter)
            features, targets = features.to(device).permute(0, 2, 1), targets.to(device).permute(0, 2, 3, 1)
            targets = F.avg_pool1d(targets.reshape(b, -1, t), kernel_size=scale_factor).reshape(b, n, l, -1)

            with torch.no_grad():
                generated = G(features)

            preds_fake = D(generated.detach(), features)
            preds_real = D(targets, features)
            loss_fake = 0.5 * torch.square(preds_fake).mean()
            loss_real = 0.5 * torch.square(preds_real - 1).mean()
            loss_D = loss_real + loss_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            mean_fake_pred = loss_fake.mean().item()
            mean_real_pred = loss_real.mean().item()
            writer.add_scalar("D/mean_fake_pred", mean_fake_pred, n_iter)
            writer.add_scalar("D/mean_real_pred", mean_real_pred, n_iter)
            writer.add_scalar("D/loss_fake", loss_fake.item(), n_iter)
            writer.add_scalar("D/loss_real", loss_real.item(), n_iter)

            n_iter += len(features)

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == n_epochs:
            with torch.inference_mode():
                G.eval()

                try:
                    fcd = calculate_fcd(
                        val_dataloader,
                        lambda x: F.interpolate(
                            G(x.permute(0, 2, 1)).reshape(x.shape[0], -1, x.shape[1]), scale_factor=scale_factor
                        )
                        .permute(0, 2, 1)
                        .reshape(x.shape[0], -1, n_outputs, output_size),
                    )
                    writer.add_scalar("Eval/FCD", fcd.item(), n_iter)
                except Exception as e:
                    pbar.write(f"\nError in FCD:\n{e}\n\n")
                    fcd = -1

                features, targets = next(valiter)
                features, targets = features.to(device).permute(0, 2, 1), targets.to(device).permute(0, 2, 3, 1)
                targets = F.avg_pool1d(targets.reshape(b, -1, t), kernel_size=scale_factor).reshape(b, n, l, -1)

                generated = G(features)

                try:
                    loc, scale = stats.laplace.fit(
                        np.random.choice(generated.cpu().numpy().flatten(), 100_000), loc=0, scale=0.1
                    )
                    writer.add_scalar("Eval/laplace_b", scale.item(), n_iter)
                except Exception as e:
                    pbar.write(f"\nError in Laplace fit:\n{e}\n\n")
                    scale = -1

                generated_stats = (
                    torch.cat((generated.mean((1, 2)), generated.std((1, 2)))).reshape(b, -1).cpu().numpy()
                )
                target_stats = torch.cat((targets.mean((1, 2)), targets.std((1, 2)))).reshape(b, -1).cpu().numpy()
                if generated_stats.shape[1] > 48:
                    generated_stats = PCA(n_components=48).fit_transform(generated_stats)
                    target_stats = PCA(n_components=48).fit_transform(target_stats)
                full_stats = np.concatenate((generated_stats, target_stats), axis=0)
                full_umap = UMAP().fit_transform(full_stats)
                fake_umap = full_umap[:b, :]
                real_umap = full_umap[b:, :]

                plt.plot(fake_umap[:, 0], fake_umap[:, 1], "o", label="fake samples", alpha=0.4)
                plt.plot(real_umap[:, 0], real_umap[:, 1], "o", label="real samples", alpha=0.4)
                plt.legend()
                plt.savefig(f"{writer.log_dir}/umap_{epoch}.pdf")
                plt.close()

            pbar.write("")
            pbar.write(f"epoch {epoch + 1}")
            pbar.write(f"laplace_b : {scale:.4f}")
            pbar.write(f"fake      : {mean_fake_pred:.4f}")
            pbar.write(f"real      : {mean_real_pred:.4f}")
            pbar.write(f"fcd       : {fcd:.4f}")
            pbar.write("")

        if (epoch + 1) % video_interval == 0 or (epoch + 1) == n_epochs:
            checkpoint_name = f"psagan_{name}_steps{n_iter:08}_fcd{fcd:.4f}_b{scale:.4f}"
            joblib.dump(
                {"n_iter": n_iter, "G": G, "D": D, "opt_G": opt_G, "opt_D": opt_D},
                f"{writer.log_dir}/{checkpoint_name}.pt",
                compress=9,
            )
            audio2video(
                a2l=lambda x: F.interpolate(
                    G(x.permute(0, 2, 1)).reshape(x.shape[0], -1, x.shape[1]), scale_factor=scale_factor
                )
                .permute(0, 2, 1)
                .reshape(x.shape[0], -1, n_outputs, output_size),
                a2f=audio2features,
                audio_file=test_audio,
                out_file=f"{writer.log_dir}/{checkpoint_name}_{Path(test_audio).stem}.mp4",
                stylegan_file="/home/hans/modelzoo/train_checks/neurout2-117.pt",
                onsets_only=synthetic,
            )

    writer.close()
