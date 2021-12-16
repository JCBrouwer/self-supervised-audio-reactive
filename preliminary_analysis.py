#%%
# %pylab inline
# %%
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import statsmodels.api as sm
import torch
import torch.nn as nn
import tslearn as tsl
from dtaidistance import clustering, dtw, dtw_ndim
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.multivariate.image import JointRecurrencePlot
from pyts.multivariate.transformation import MultivariateTransformer
from scipy import stats
from sklearn.decomposition import FastICA
from torch.utils.data import DataLoader
from tsai.all import *
from tslearn.clustering import TimeSeriesKMeans

from sgw import sgw_gpu as sliced_gromov_wasserstein
from synchro_saliency import AudioVisualFeatures

my_setup()
gpu = torch.device("cuda")


#%%

dur = 4
fps = 24

benny_cache = f"cache/trashbenny_features_{dur}sec_{fps}fps"
maua_cache = f"cache/maua_features_{dur}sec_{fps}fps"
phony_cache = f"cache/phony_features_{dur}sec_{fps}fps"
trapnation_cache = f"cache/trapnation_features_{dur}sec_{fps}fps"
invocation_cache = f"cache/invocation_features_{dur}sec_{fps}fps"

all_features = []
for name, features in [
    ["trashbenny", AudioVisualFeatures(benny_cache)],
    ["maua", AudioVisualFeatures(maua_cache)],
    ["phony", AudioVisualFeatures(phony_cache)],
    ["trapnation", AudioVisualFeatures(trapnation_cache)],
    ["invocation", AudioVisualFeatures(invocation_cache)],
]:
    all_vfs, all_afs, all_fns = [], [], []
    for vfs, afs, file in DataLoader(features, shuffle=True):
        all_features.append(
            {
                "file": file.strip(),
                "group": name,
                **{k: vf.squeeze().numpy() for k, vf in vfs.items()},
                **{k: af.squeeze().numpy() for k, af in afs.items()},
            }
        )
all_features = pd.DataFrame(all_features)

#%%

audio_feats = np.stack(all_features["audio_vggish_layer0"])
np.random.shuffle(audio_feats)

video_feats = np.stack(all_features["video_slowfast_layer0"])
np.random.shuffle(video_feats)

audiovideo_feats = np.concatenate(
    [np.stack(all_features["audio_vggish_layer0"]), np.stack(all_features["video_slowfast_layer0"])], axis=2
)
np.random.shuffle(audiovideo_feats)

audio_tempogram = np.stack(all_features["audio_tempogram"])
video_tempogram = np.stack(all_features["video_tempogram"])
video_onsets = np.stack(all_features["video_onsets"])
audio_onsets = np.stack(all_features["audio_onsets"])

#%%
def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def covariance(A, B):
    A = A - A.mean((1, 2)).view(-1, 1, 1)
    B = B - B.mean((1, 2)).view(-1, 1, 1)
    C = torch.bmm(A, B.transpose(2, 1))
    return C


def correlation(A, B):
    C = torch.bmm(A, B.transpose(2, 1))
    return C


def PCA_svd(X, k: int):
    m, t, p = X.size()
    X = X.reshape(m * t, p)
    n, p = X.shape
    X_center = X - X.mean(1).view(n, 1)
    u, s, v = torch.svd(X_center)
    components = v[:k].T
    X = X @ components
    return X.reshape(m, t, k)


def PCA_eig(X, k: int, scale: bool = False):
    m, t, p = X.size()
    X = X.reshape(m * t, p)
    n, p = X.size()
    X_center = X - X.mean(1).view(n, 1)
    covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center)
    if scale:
        covariance = torch.mm(torch.diag(torch.sqrt(1 / torch.diag(covariance))), covariance)
    _, eigenvectors = torch.linalg.eigh(covariance)  # cov is symmetric => eigh OK
    components = eigenvectors[:, :k]
    X = X @ components
    return X.reshape(m, t, k)


def distance_correlation(A, B):
    return 1 - torch.bmm(A, B).diagonal(0, -1, -2).sum(-1) / (torch.norm(A, dim=(1, 2)) * torch.norm(B, dim=(1, 2)))


def distance_wasserstein(A, B):
    B12 = symsqrt(B)
    C = symsqrt(torch.bmm(torch.bmm(B12, A), B12))
    return torch.sqrt((A + B - 2 * C).diagonal(0, -1, -2).sum(-1))


def distance_riemann(A, B):
    return torch.sqrt((torch.log(torch.lobpcg(A, B=B, k=16)) ** 2).sum())


def distance_kullback(A, B):
    dim = A.shape[0]
    logdet = torch.logdet(B) - torch.logdet(A)
    print(logdet)
    kl = torch.bmm(torch.linalg.pinv(B), A).diagonal(0, -1, -2).sum(-1) - dim + logdet
    return 0.5 * kl


def distance_condition(A, B):  # TODO fix ill-conditioned eigvalsh, maybe pinv(symsqrt) related?
    invroot = torch.linalg.pinv(symsqrt(A))
    sigma_star = torch.bmm(invroot, torch.bmm(B, invroot))
    eigvals = torch.linalg.eigvals(sigma_star).real
    print(eigvals)
    print(torch.log(eigvals[:, 0]))
    print(torch.log(eigvals[:, -1]))
    results = torch.log(eigvals[:, 0]) - torch.log(eigvals[:, -1])
    return results


af = torch.from_numpy(audio_feats).cuda()
vf = torch.from_numpy(video_feats).cuda()

aacf = covariance(af, af)
vacf = covariance(vf, vf)

ccf = covariance(PCA_eig(af, 32), PCA_eig(vf, 32))

print(torch.median(distance_correlation(aacf, vacf)))
print(torch.median(distance_correlation(aacf, ccf)))
print(torch.median(distance_correlation(vacf, ccf)))

print(torch.median(distance_wasserstein(aacf, vacf)))
print(torch.median(distance_wasserstein(aacf, ccf)))
print(torch.median(distance_wasserstein(vacf, ccf)))


# def make_positive_definite(C):
#     # ensure positive definite
#     D, V = torch.linalg.eigh(C)
#     D = torch.maximum(D, torch.zeros_like(D))
#     BB = torch.matmul(V, torch.matmul(D[:, None, :], V).squeeze()[..., None]).squeeze()
#     T = 1 / torch.sqrt(torch.diag_embed(BB, 0, -1, -2))
#     TT = torch.bmm(T, T.transpose(2, 1))
#     C = BB[..., None] * TT
#     return C
# def symmetric(a, rtol=1e-05, atol=1e-08):
#     return torch.allclose(a, a.T, rtol=rtol, atol=atol)
# def positive_definite(a):
#     return torch.all(torch.linalg.eigvals(a) > 0)
# for a, v in zip(aacf, vacf):
#     assert positive_definite(a)
#     assert positive_definite(v)
#     assert symmetric(a)
#     assert symmetric(v)
# torch.linalg.cholesky(aacf)
# torch.linalg.cholesky(vacf)

# print(torch.median(distance_kullback(aacf, vacf)))
# print(torch.median(distance_kullback(aacf, ccf)))
# print(torch.median(distance_kullback(vacf, ccf)))

# print(torch.median(distance_condition(aacf, vacf)))
# print(torch.median(distance_condition(aacf, ccf)))
# print(torch.median(distance_condition(vacf, ccf)))

# print(torch.median(distance_riemann(aacf, vacf)))
# print(torch.median(distance_riemann(aacf, ccf)))
# print(torch.median(distance_riemann(vacf, ccf)))

#%%
class MutualInformation(nn.Module):
    def __init__(self, sigma=0.4, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()

        self.sigma = 2 * sigma ** 2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device=device).float(), requires_grad=False)

    def marginal_pdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def joint_pdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf

    def get_mutual_information(self, input1, input2):
        # Torch tensors for images between (0, 1)
        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W = input1.shape
        assert input1.shape == input2.shape

        x1 = input1.view(B, H * W, C)
        x2 = input2.view(B, H * W, C)

        pdf_x1, kernel_values1 = self.marginal_pdf(x1)
        pdf_x2, kernel_values2 = self.marginal_pdf(x2)
        pdf_x1x2 = self.joint_pdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        return mutual_information

    def forward(self, input1, input2):
        """
        input1: B, C, H, W
        input2: B, C, H, W
        return: scalar
        """
        return self.get_mutual_information(input1, input2)


mutual_information = MutualInformation()

#%%
def compare_av_feats(df, vid_cols, aud_cols, misaligned=False):
    sgws, pca_corrs, pca_wasss = [], [], []
    for vid_col in vid_cols:
        for aud_col in aud_cols:
            if misaligned:
                idxs = np.random.permutation(n_samples)
            else:
                idxs = np.arange(n_samples)
            vf = torch.from_numpy(np.stack(df[vid_col].values[idxs])).to(gpu)
            af = torch.from_numpy(np.stack(df[aud_col].values)).to(gpu)
            if vf.dim() < 3:
                vf = vf.unsqueeze(-1)
            if af.dim() < 3:
                af = af.unsqueeze(-1)
            vcov, acov = covariance(vf, vf), covariance(af, af)

            sgws.append(sliced_gromov_wasserstein(vf.flatten(1), af.flatten(1), gpu))
            pca_corrs.append(distance_correlation(vcov, acov).mean())
            pca_wasss.append(distance_wasserstein(vcov, acov).mean())
    return sum(sgws), sum(pca_corrs), sum(pca_wasss)


#%%
slowfast_layers = [
    "video_slowfast_layer0",
    "video_slowfast_layer1",
    "video_slowfast_layer2",
    "video_slowfast_layer3",
    "video_slowfast_layer4",
]
vggish_layers = [
    "audio_vggish_layer0",
    "audio_vggish_layer1",
    "audio_vggish_layer2",
    "audio_vggish_layer3",
    "audio_vggish_layer4",
]
n_samples = 100

#%%
aligned = []
for group, group_df in all_features.groupby("group"):
    idxs = np.random.permutation(len(group_df))[:n_samples]
    group_df = group_df.take(idxs)

    acav_sgw, acav_pca_corr, acav_pca_wass = compare_av_feats(group_df, slowfast_layers, vggish_layers)
    chroma_sgw, chroma_pca_corr, chroma_pca_wass = compare_av_feats(group_df, slowfast_layers, ["audio_chroma"])
    aud_ons_sgw, aud_ons_pca_corr, aud_ons_pca_wass = compare_av_feats(group_df, slowfast_layers, ["audio_onsets"])
    vid_ons_sgw, vid_ons_pca_corr, vid_ons_pca_wass = compare_av_feats(group_df, ["video_onsets"], vggish_layers)

    dtw_dists = [
        dtw.distance_fast(vo.astype(np.float64), ao.astype(np.float64))
        for ao, vo in zip(group_df["video_onsets"], group_df["audio_onsets"])
    ]
    dtw_min, dtw_med, dtw_mean, dtw_max = (
        np.min(dtw_dists),
        np.median(dtw_dists),
        np.mean(dtw_dists),
        np.max(dtw_dists),
    )

    vft = torch.from_numpy(np.stack(group_df["video_fourier_tempogram"].values)).unsqueeze(1).to(gpu)
    vt = torch.from_numpy(np.stack(group_df["video_tempogram"].values)).unsqueeze(1).to(gpu)
    aft = torch.from_numpy(np.stack(group_df["audio_fourier_tempogram"].values)).unsqueeze(1).to(gpu)
    at = torch.from_numpy(np.stack(group_df["audio_tempogram"].values)).unsqueeze(1).to(gpu)
    avmi = mutual_information(vt, at).sum() + mutual_information(vft, aft).sum()

    aligned.append(
        {
            "group": group,
            "acav_sgw": acav_sgw.item(),
            "acav_pca_corr": acav_pca_corr.item(),
            "acav_pca_wass": acav_pca_wass.item(),
            "chroma_sgw": chroma_sgw.item(),
            "chroma_pca_corr": chroma_pca_corr.item(),
            "chroma_pca_wass": chroma_pca_wass.item(),
            "aud_ons_sgw": aud_ons_sgw.item(),
            "aud_ons_pca_corr": aud_ons_pca_corr.item(),
            "aud_ons_pca_wass": aud_ons_pca_wass.item(),
            "vid_ons_sgw": vid_ons_sgw.item(),
            "vid_ons_pca_corr": vid_ons_pca_corr.item(),
            "vid_ons_pca_wass": vid_ons_pca_wass.item(),
            "dtw_min": dtw_min,
            "dtw_med": dtw_med,
            "dtw_mean": dtw_mean,
            "dtw_max": dtw_max,
            "avmi": avmi.item(),
        }
    )

aligned = pd.DataFrame(aligned)
aligned.set_index("group")
aligned

#%%

misaligned = []
for group, group_df in all_features.groupby("group"):
    idxs = np.random.permutation(len(group_df))[:n_samples]
    group_df = group_df.take(idxs)

    acav_sgw, acav_pca_corr, acav_pca_wass = compare_av_feats(group_df, slowfast_layers, vggish_layers, misaligned=True)
    chroma_sgw, chroma_pca_corr, chroma_pca_wass = compare_av_feats(
        group_df, slowfast_layers, ["audio_chroma"], misaligned=True
    )
    aud_ons_sgw, aud_ons_pca_corr, aud_ons_pca_wass = compare_av_feats(
        group_df, slowfast_layers, ["audio_onsets"], misaligned=True
    )
    vid_ons_sgw, vid_ons_pca_corr, vid_ons_pca_wass = compare_av_feats(
        group_df, ["video_onsets"], vggish_layers, misaligned=True
    )

    dtw_dists = [
        dtw.distance_fast(vo.astype(np.float64), ao.astype(np.float64))
        for ao, vo in zip(group_df["video_onsets"].values[np.random.permutation(n_samples)], group_df["audio_onsets"])
    ]
    dtw_min, dtw_med, dtw_mean, dtw_max = (
        np.min(dtw_dists),
        np.median(dtw_dists),
        np.mean(dtw_dists),
        np.max(dtw_dists),
    )

    vft = (
        torch.from_numpy(np.stack(group_df["video_fourier_tempogram"].values[np.random.permutation(n_samples)]))
        .unsqueeze(1)
        .to(gpu)
    )
    vt = (
        torch.from_numpy(np.stack(group_df["video_tempogram"].values[np.random.permutation(n_samples)]))
        .unsqueeze(1)
        .to(gpu)
    )
    aft = torch.from_numpy(np.stack(group_df["audio_fourier_tempogram"].values)).unsqueeze(1).to(gpu)
    at = torch.from_numpy(np.stack(group_df["audio_tempogram"].values)).unsqueeze(1).to(gpu)

    avmi = mutual_information(vt, at).sum() + mutual_information(vft, aft).sum()

    misaligned.append(
        {
            "group": group,
            "acav_sgw": acav_sgw.item(),
            "acav_pca_corr": acav_pca_corr.item(),
            "acav_pca_wass": acav_pca_wass.item(),
            "chroma_sgw": chroma_sgw.item(),
            "chroma_pca_corr": chroma_pca_corr.item(),
            "chroma_pca_wass": chroma_pca_wass.item(),
            "aud_ons_sgw": aud_ons_sgw.item(),
            "aud_ons_pca_corr": aud_ons_pca_corr.item(),
            "aud_ons_pca_wass": aud_ons_pca_wass.item(),
            "vid_ons_sgw": vid_ons_sgw.item(),
            "vid_ons_pca_corr": vid_ons_pca_corr.item(),
            "vid_ons_pca_wass": vid_ons_pca_wass.item(),
            "dtw_min": dtw_min,
            "dtw_med": dtw_med,
            "dtw_mean": dtw_mean,
            "dtw_max": dtw_max,
            "avmi": avmi.item(),
        }
    )

misaligned = pd.DataFrame(misaligned)
misaligned.set_index("group")
misaligned


#%%
aligned.to_csv("aligned.csv")
misaligned.to_csv("misaligned.csv")

#%%


exit(0)


#


#


#


#


#


#%%

X_gaf = MultivariateTransformer(GramianAngularField(), flatten=False).fit_transform(audio_feats)
X_jrp = JointRecurrencePlot(threshold="point", percentage=50).fit_transform(audio_feats)
print(audio_feats.shape, X_gaf.shape, X_jrp.shape)

X_gaf = MultivariateTransformer(GramianAngularField(), flatten=False).fit_transform(video_feats)
X_jrp = JointRecurrencePlot(threshold="point", percentage=50).fit_transform(video_feats)
print(video_feats.shape, X_gaf.shape, X_jrp.shape)

X_gaf = MultivariateTransformer(GramianAngularField(), flatten=False).fit_transform(audiovideo_feats)
X_jrp = JointRecurrencePlot(threshold="point", percentage=50).fit_transform(audiovideo_feats)
print(audiovideo_feats.shape, X_gaf.shape, X_jrp.shape)

# X_mtf = MultivariateTransformer(MarkovTransitionField(), flatten=False).fit_transform(audio_feats)
# X_rp = MultivariateTransformer(RecurrencePlot(), flatten=False).fit_transform(audio_feats)
# print(audio_feats.shape, X_mtf.shape, X_rp.shape)

# %%
# Plot the Gramian angular fields
fig = plt.figure(figsize=(16, 16))

grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1, share_all=True, cbar_mode="single")
for i, ax in enumerate(grid):
    im = ax.imshow(X_gaf[0, i], cmap="rainbow", origin="lower", vmin=-1.0, vmax=1.0)
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
plt.colorbar(im, cax=grid.cbar_axes[0])
ax.cax.toggle_label(True)

fig.suptitle("Gramian angular summation fields")
plt.tight_layout()
plt.show()

#%%
mrf = MiniRocketFeatures(audio_feats.shape[1], audio_feats.shape[2]).to(default_device())
audio_minirocket = get_minirocket_features(audio_feats, mrf, chunksize=1024, to_np=False).squeeze()
video_minirocket = get_minirocket_features(video_feats, mrf, chunksize=1024, to_np=False).squeeze()
audiovideo_minirocket = get_minirocket_features(audiovideo_feats, mrf, chunksize=1024, to_np=False).squeeze()
print(audio_minirocket.shape, video_minirocket.shape, audiovideo_minirocket.shape)

#%%

print(
    sliced_gromov_wasserstein(
        torch.from_numpy(audio_feats).flatten(1).cuda(), torch.from_numpy(video_feats).flatten(1).cuda(), gpu
    )
)
print(
    sliced_gromov_wasserstein(
        torch.from_numpy(audio_feats).flatten(1).cuda(), torch.from_numpy(audiovideo_feats).flatten(1).cuda(), gpu
    )
)
print(
    sliced_gromov_wasserstein(
        torch.from_numpy(video_feats).flatten(1).cuda(), torch.from_numpy(audiovideo_feats).flatten(1).cuda(), gpu
    )
)

print(sliced_gromov_wasserstein(audio_minirocket, video_minirocket, gpu))
print(sliced_gromov_wasserstein(audio_minirocket, audiovideo_minirocket, gpu))
print(sliced_gromov_wasserstein(video_minirocket, audiovideo_minirocket, gpu))

#%%

distances = [
    dtw.distance_fast(ao.astype(np.float64), vo.astype(np.float64)) for ao, vo in zip(audio_onsets, video_onsets)
]
all_features["dtw"] = distances

#%%

all_features.groupby("group")["dtw"].agg(["min", "median", "mean", "max", "count"])

#%%
amodel = clustering.LinkageTree(partial(dtw_ndim.distance_matrix_fast, ndim=3), {})
aclust = amodel.fit(audio_feats.astype(np.float64))
vmodel = clustering.LinkageTree(partial(dtw_ndim.distance_matrix_fast, ndim=3), {})
vclust = vmodel.fit(video_feats.astype(np.float64))
avmodel = clustering.LinkageTree(partial(dtw_ndim.distance_matrix_fast, ndim=3), {})
avclust = avmodel.fit(audiovideo_feats.astype(np.float64))


#%%
fig, ax = plt.subplots(3, 4, figsize=(16, 9))
for i, (name, X_train) in enumerate([("audio", audio_feats), ("video", video_feats), ("audiovideo", audiovideo_feats)]):
    X_train = tsl.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(X_train)

    print("Soft-DTW k-means", name, X_train.shape)
    sdtw_km = TimeSeriesKMeans(n_clusters=4, metric="softdtw", metric_params={"gamma": 0.01}, verbose=True, n_jobs=24)
    sdtw_km.fit(X_train)
    y_pred = sdtw_km.predict(X_train)

    print("cluster centers:", sdtw_km.cluster_centers_.shape)
    print("inertia:", sdtw_km.inertia_)
    print("cluster sizes:")
    for yi in range(4):
        print(yi, X_train[y_pred == yi].shape)
        for xx in X_train[y_pred == yi]:
            ax[i, yi].plot(xx.mean(1), "k-", alpha=0.2)
        ax[i, yi].plot(sdtw_km.cluster_centers_[yi].mean(1), "r-")
        ax[i, yi].xlim(0, X_train.shape[1])
        ax[i, yi].ylim(-4, 4)
        ax[i, yi].text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=ax[i, yi].transAxes)

    ax[0, 0].title("Soft-DTW $k$-means")
    plt.tight_layout()
    plt.show()

#%%
fig, ax = plt.subplots(3, 1, figsize=(12, 24))

tempogram = audio_tempogram[0].T
librosa.display.specshow(tempogram, x_axis="time", y_axis="tempo", cmap="magma", ax=ax[0])
librosa.display.specshow(video_tempogram[0].T, x_axis="time", y_axis="tempo", cmap="magma", ax=ax[1])

freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=512, sr=16_00)[1:]

tempos = tempogram.mean(1)[1:]
ax[2].plot(freqs, tempos)

n_peaks = 4
peaks = signal.find_peaks(tempos)[0]
top_peaks = peaks[np.argsort(tempos[peaks])[-n_peaks:]]
ax[2].vlines(freqs[top_peaks], ymin=np.zeros(n_peaks), ymax=np.ones(n_peaks), ls="--")
plt.tight_layout()

print("BPMs:", freqs[top_peaks])
print("Lags:", 60 / freqs[top_peaks])

# %%
freqs = librosa.tempo_frequencies(dur * fps, hop_length=512, sr=16_00)[1:]

for i, (at, vt) in enumerate(zip(audio_tempogram, video_tempogram)):
    video_tempo = librosa.beat.tempo(onset_envelope=video_onsets[i])[0]
    audio_tempo = librosa.beat.tempo(onset_envelope=audio_onsets[i])[0]

    n_peaks = 2

    audio_tempos = at.T.mean(1)[1:]
    audio_peaks = signal.find_peaks(audio_tempos)[0]
    audio_peaks = audio_peaks[np.argsort(audio_tempos[audio_peaks])[-n_peaks:]]

    video_tempos = vt.T.mean(1)[1:]
    video_peaks = signal.find_peaks(video_tempos)[0]
    video_peaks = video_peaks[np.argsort(video_tempos[video_peaks])[-n_peaks:]]

    bpms = [video_tempo, audio_tempo, *freqs[audio_peaks], *freqs[video_peaks]]
    print(bpms)

# %%

# Create a dynamic factor model
mod_dfm = sm.tsa.DynamicFactorMQ(audio_feats, k_factors=4, factor_order=2)
# Note that mod_dfm is an instance of the DynamicFactorMQ class

# Fit the model via maximum likelihood, using the EM algorithm
res_dfm = mod_dfm.fit()
# Note that res_dfm is an instance of the DynamicFactorMQResults class

# Show the summary of results
print(res_dfm.summary())

# Show a plot of the r^2 values from regressions of
# individual estimated factors on endogenous variables.
fig_dfm = res_dfm.plot_coefficients_of_determination()

# sm.tsa.filters.filtertools.miso_lfilter()

#%%
compressed = FastICA(n_components=8).fit_transform(audio_feats)
compressed

#%%
data = pd.read_csv("sgws.csv", index_col=0)
data

group_data = pd.read_csv("group_sgws.csv", index_col=0).set_index("group")
group_data = np.log(group_data)
group_data


def plot_kdes(D):
    X = np.linspace(1.3 * D.values.min(), 0, 200)
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange"]
    for i, (group, row) in enumerate(D.iterrows()):

        kernel = stats.gaussian_kde(row)
        Z = kernel(X)
        plt.hist(row.values, color=colors[i], bins=20, alpha=0.1, density=True)
        plt.plot(X, Z, color=colors[i], alpha=0.5)
        plt.vlines(X[np.argmax(Z)], ymin=0, ymax=Z.max(), color=colors[i], label=group, alpha=0.5)
    plt.legend()
    plt.tight_layout()


plot_kdes(group_data)
plot_kdes(np.log(data.groupby("group").agg("median")))
data.groupby("group").agg("count")
