import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DF = pd.read_csv("output/rv2s.csv")

splits = ["train", "val", "test"]
colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
fig, ax = plt.subplots(3, 3, figsize=(18, 12))
for i, output in enumerate(["latent", "noise", "envelope"]):
    for j, split in enumerate(splits):
        for k, (sup, dec) in enumerate(
            [
                ("oldsupervised", "learned"),
                ("supervised", "learned"),
                ("supervised", "fixed"),
                ("selfsupervised", "fixed"),
            ]
        ):
            df = DF[(DF.supervision == sup) & (DF.decoder == dec)]
            x = df[f"iterations"]
            y = df[f"{split}_{output}_rv2"]
            err = df[f"{split}_{output}_rv2_std"]
            if (y < 0).any():
                continue
            ax[i, j].plot(x, y, color=colors[k], label=fr"{sup} {dec} +/- $\sigma$")
            ax[i, j].fill_between(x, y - err, y + err, alpha=0.25, color=colors[k])
            ax[-1, j].set_xlabel("iterations")
            ax[i, 0].set_ylabel("rv2 loss")
            ax[i, j].set_title(f"{split} {output}")
            ax[i, j].legend()
            ax[i, j].set_xlim(0, 1_000_000)
            ylim = np.nanmax(np.concatenate([DF[f"{s}_{output}_rv2"] + DF[f"{s}_{output}_rv2_std"] for s in splits]))
            ax[i, j].set_ylim(0, ylim * 1.1)
plt.tight_layout()
plt.savefig("output/rv2s_over_training.pdf")
