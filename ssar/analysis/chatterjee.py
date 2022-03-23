import numpy as np
import torch


def rank_ordinal(a):
    arr = torch.flatten(a)
    sorter = torch.argsort(arr)
    inv = torch.empty(sorter.numel(), dtype=torch.long, device=a.device)
    inv[sorter] = torch.arange(sorter.numel(), dtype=torch.long, device=a.device)
    return inv + 1


def rank_max(a):
    arr = torch.flatten(a)
    sorter = torch.argsort(arr)
    inv = torch.empty(sorter.numel(), dtype=torch.long, device=a.device)
    inv[sorter] = torch.arange(sorter.numel(), dtype=torch.long, device=a.device)
    arr = arr[sorter]
    obs = torch.cat((torch.ones((1)).to(arr), (arr[1:] != arr[:-1]))).to(torch.long)
    dense = torch.cumsum(obs, 0)[inv]
    count = torch.cat((torch.nonzero(obs).squeeze(), torch.tensor([len(obs)]).to(obs)))
    return count[dense]


def rank(x):
    len_x = len(x)
    randomized_indices = torch.randperm(len_x, device=x.device)
    randomized = x[randomized_indices]
    rankdata = rank_ordinal(randomized)
    randomized_indices_order = torch.argsort(randomized_indices)
    unrandomized_indices = torch.arange(len_x, device=x.device)[randomized_indices_order]
    return rankdata[unrandomized_indices]


from tqdm import tqdm


def quadratic_xi(x, y):
    xis = []
    for xcol in tqdm(range(x.shape[1])):
        for ycol in range(y.shape[1]):
            xis.append(xi(x[:, xcol], y[:, ycol]))
    return xis


def xi(x, y):
    """Based on https://github.com/czbiohub/xicor

    MIT License

    Copyright (c) 2020 Chan Zuckerberg Biohub

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    n = len(x)

    y_rank_max = rank_max(y) / n

    x_ordered = torch.argsort(rank_ordinal(x))
    x_rank_max_ordered = y_rank_max[x_ordered]

    mean_absolute = torch.mean(torch.abs(x_rank_max_ordered[:-1] - x_rank_max_ordered[1:])) * (n - 1) / (2 * n)

    g = rank_max(-y) / n
    inverse_g_mean = torch.mean(g * (1 - g))

    return 1 - mean_absolute / inverse_g_mean


import rpy2.rinterface_lib.callbacks

rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: x
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: x

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, isinstalled

if not isinstalled("FOCI"):
    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    utils.install_packages("FOCI")
foci_r = importr("FOCI")
numpy2ri.activate()


def mean_conditional_dependence(x, y):
    results = foci_r.foci(y.detach().cpu().numpy(), x.detach().cpu().numpy(), num_features=x.shape[1], stop=False)
    return np.mean([c[0] for c in results[0]])


def conditional_dependence(x, y):
    results = foci_r.foci(y.detach().cpu().numpy(), x.detach().cpu().numpy(), num_features=x.shape[1], stop=False)
    columns = [c[0] for c in results[0]]
    order = np.argsort(columns)
    return results[1][order]


if __name__ == "__main__":

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    x = torch.linspace(-5, 5, 100)
    y = torch.linspace(-5, 5, 100)

    n0 = lambda: torch.zeros_like(y)
    n1 = lambda: torch.from_numpy(np.random.uniform(0, 1, 100))
    n2 = lambda: torch.from_numpy(np.random.uniform(0, 3, 100))
    n3 = lambda: torch.from_numpy(np.random.uniform(0, 5, 100))

    fig, ax = plt.subplots(3, 4, figsize=(12, 8))
    for i, fn in enumerate([lambda y: y, lambda y: torch.sin(y), lambda y: y ** 2]):
        for j, n in enumerate([n0, n1, n2, n3]):

            yy = fn(y) + n()

            xi_score = xi(x, yy)
            foci_score = conditional_dependence(x[:, None], yy)[0]

            ax[i, j].plot(x, yy, ".")
            ax[i, j].set_title(f"xi={xi_score:.3f}, foci={foci_score:.3f}")
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.savefig("output/chatterjee.png")
    plt.close()
