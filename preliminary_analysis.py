#%%
# %pylab inline
# %%
import dtaidistance as dtai
import librosa.display
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import pyts
import scipy.signal as signal
import statsmodels.api as sm
import tslearn as tsl
from scipy import stats
from sklearn.decomposition import FastICA
from torch.utils.data import DataLoader

from synchro_saliency import AudioVisualFeatures

#%%
data = pd.read_csv("sgws.csv", index_col=0)
data
# %%
group_data = pd.read_csv("group_sgws.csv", index_col=0).set_index("group")
group_data = np.log(group_data)
group_data

# %%
def plot_kdes(D):
    X = np.linspace(1.3 * D.values.min(), 0, 200)
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    for i, (group, row) in enumerate(D.iterrows()):

        kernel = stats.gaussian_kde(row)
        Z = kernel(X)
        plt.hist(row.values, color=colors[i], bins=20, alpha=0.1, density=True)
        plt.plot(X, Z, color=colors[i], alpha=0.5)
        plt.vlines(X[np.argmax(Z)], ymin=0, ymax=Z.max(), color=colors[i], label=group, alpha=0.5)
    plt.legend()
    plt.tight_layout()


# %%
plot_kdes(group_data)
# %%
plot_kdes(np.log(data.groupby("group").agg("median")))
#%%
data.groupby("group").agg("count")

#%%

dur = 4
fps = 24

benny_cache = f"cache/audio_features/trashbenny_features_{dur}sec_{fps}fps"
maua_cache = f"cache/audio_features/maua-short_features_{dur}sec_{fps}fps"
phony_cache = f"cache/audio_features/phony_features_{dur}sec_{fps}fps"
trapnation_cache = f"cache/audio_features/trapnation_features_{dur}sec_{fps}fps"
# invocation_cache = f"cache/audio_features/invocation_features_{dur}sec_{fps}fps"

all_features = []
for name, features in [
    ["trashbenny", AudioVisualFeatures(benny_cache)],
    ["maua", AudioVisualFeatures(maua_cache)],
    ["phony", AudioVisualFeatures(phony_cache)],
    ["trapnation", AudioVisualFeatures(trapnation_cache)],
]:
    all_vfs, all_afs, all_fns = [], [], []
    for vfs, afs, (file,) in DataLoader(features, shuffle=True):
        all_features.append(
            {
                "file": file.strip(),
                "group": name,
                **{k: vf.squeeze().numpy() for k, vf in vfs.items()},
                **{k: af.squeeze().numpy() for k, af in afs.items()},
            }
        )
all_features = pd.DataFrame(all_features)
all_features


#%%

seed = 42

audio_feats = np.stack(all_features["audio_vggish_layer0"])
numpy.random.shuffle(audio_feats)

video_feats = np.stack(all_features["video_slowfast_layer0"])
numpy.random.shuffle(video_feats)

audiovideo_feats = np.concatenate(
    [np.stack(all_features["audio_vggish_layer0"]), np.stack(all_features["video_slowfast_layer0"])], axis=2
)
numpy.random.shuffle(audiovideo_feats)


for X_train in [audio_feats, video_feats, audiovideo_feats]:
    X_train = tsl.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(X_train)
    sz = X_train.shape[1]

    # Euclidean k-means
    print("Euclidean k-means")
    km = tsl.clustering.TimeSeriesKMeans(n_clusters=4, verbose=True, random_state=seed)
    y_pred = km.fit_predict(X_train)

    plt.figure()
    for yi in range(4):
        plt.subplot(4, 4, yi + 1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.mean(1), "k-", alpha=0.2)
        plt.plot(km.cluster_centers_[yi].mean(1), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("Euclidean $k$-means")

    # DBA-k-means
    print("DBA k-means")
    dba_km = tsl.clustering.TimeSeriesKMeans(
        n_clusters=4, n_init=2, metric="dtw", verbose=True, max_iter_barycenter=10, random_state=seed, n_jobs=12
    )
    y_pred = dba_km.fit_predict(X_train)
    y_pred = dba_km.predict(X_train)

    for yi in range(4):
        plt.subplot(4, 4, 5 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.mean(1), "k-", alpha=0.2)
        plt.plot(dba_km.cluster_centers_[yi].mean(1), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("DBA $k$-means")

    # Soft-DTW-k-means
    print("Soft-DTW k-means")
    sdtw_km = tsl.clustering.TimeSeriesKMeans(
        n_clusters=4, metric="softdtw", metric_params={"gamma": 0.01}, verbose=True, random_state=seed, n_jobs=12
    )
    y_pred = sdtw_km.fit_predict(X_train)
    y_pred = sdtw_km.predict(X_train)

    for yi in range(4):
        plt.subplot(4, 4, 11 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.mean(1), "k-", alpha=0.2)
        plt.plot(sdtw_km.cluster_centers_[yi].mean(1), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("Soft-DTW $k$-means")

    plt.tight_layout()
    plt.show()

#%%
audio_tempogram = np.stack(all_features["audio_tempogram"])
video_tempogram = np.stack(all_features["video_tempogram"])
video_onsets = np.stack(all_features["video_onsets"])
audio_onsets = np.stack(all_features["audio_onsets"])

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

#%%

print(video_onsets.shape)
for vo, ao in zip(video_onsets, audio_onsets):
    sm.tsa.acf(vo)
    sm.tsa.acovf(vo)
    sm.tsa.acf(ao)
    sm.tsa.acovf(ao)
    sm.tsa.ccf(vo, ao)
    sm.tsa.ccovf(vo, ao)

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

#%%
audio_feats = np.stack(all_features["audio_vggish_layer0"])
np.random.shuffle(audio_feats)

video_feats = np.stack(all_features["video_slowfast_layer0"])
np.random.shuffle(video_feats)

audiovideo_feats = np.concatenate(
    [np.stack(all_features["audio_vggish_layer0"]), np.stack(all_features["video_slowfast_layer0"])], axis=2
)
np.random.shuffle(audiovideo_feats)

sm.tsa.filters.filtertools.miso_lfilter()


#%%
compressed = FastICA(n_components=8).fit_transform(audio_feats)
compressed

# %%
from pyts.multivariate.image import JointRecurrencePlot

# Recurrence plot transformation
jrp = JointRecurrencePlot(threshold="point", percentage=50)
X_jrp = jrp.fit_transform(audio_feats)

# Show the results for the first time series
plt.figure(figsize=(5, 5))
plt.imshow(X_jrp[0], cmap="binary", origin="lower")
plt.title("Joint Recurrence Plot", fontsize=18)
plt.tight_layout()
plt.show()

#%%

all_onsets = np.concatenate((audio_onsets, video_onsets)).astype(np.float64)
dtai.dtw.distance_matrix_fast(
    all_onsets, block=((0, len(audio_onsets)), (len(audio_onsets), len(audio_onsets) + len(video_onsets)))
)

#%%
from dtaidistance import clustering

amodel = clustering.LinkageTree(dtai.dtw.distance_matrix_fast, {})
aclust = amodel.fit(audio_feats.astype(np.float64))
vmodel = clustering.LinkageTree(dtai.dtw.distance_matrix_fast, {})
vclust = vmodel.fit(video_feats.astype(np.float64))
# %%
