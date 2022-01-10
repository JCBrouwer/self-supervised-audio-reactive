import numpy as np


def rank_ordinal(a):
    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    return inv + 1


def rank_max(a):
    arr = np.ravel(np.asarray(a))
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return count[dense]


def rank(x):
    len_x = len(x)
    randomized_indices = np.random.choice(np.arange(len_x), len_x, replace=False)
    randomized = x[randomized_indices]
    rankdata = rank_ordinal(randomized)
    randomized_indices_order = np.argsort(randomized_indices)
    unrandomized_indices = np.arange(len_x)[randomized_indices_order]
    return rankdata[unrandomized_indices]


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

    x_ordered = np.argsort(rank_ordinal(x))
    x_rank_max_ordered = y_rank_max[x_ordered]

    mean_absolute = np.mean(np.abs(x_rank_max_ordered[:-1] - x_rank_max_ordered[1:])) * (n - 1) / (2 * n)

    g = rank_max(-y) / n
    inverse_g_mean = np.mean(g * (1 - g))

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


def foci(x, y):
    results = foci_r.foci(y, x, num_features=x.shape[1], stop=False)
    columns = [c[0] for c in results[0]]
    order = np.argsort(columns)
    return results[1][order]


if __name__ == "__main__":

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    n0 = lambda: np.zeros_like(y)
    n1 = lambda: np.random.uniform(0, 1, 100)
    n2 = lambda: np.random.uniform(0, 3, 100)
    n3 = lambda: np.random.uniform(0, 5, 100)

    fig, ax = plt.subplots(3, 4, figsize=(12, 8))
    for i, fn in enumerate([lambda y: y, lambda y: np.sin(y), lambda y: y ** 2]):
        for j, n in enumerate([n0, n1, n2, n3]):

            yy = fn(y) + n()

            xi_score = xi(x, yy)
            foci_score = foci(x[:, None], yy)[0]

            ax[i, j].plot(x, yy, ".")
            ax[i, j].set_title(f"xi={xi_score:.3f}, foci={foci_score:.3f}")
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.savefig("output/chatterjee.png")
    plt.close()
