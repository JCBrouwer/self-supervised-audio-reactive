import gc
import os
import warnings
from copy import deepcopy
from glob import glob
from pathlib import Path

warnings.catch_warnings()
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import torch
import torchaudio
import torchvision as tv
from npy_append_array import NpyAppendArray
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import center_crop, resize
from tqdm import tqdm

from ..models.slowfast import SlowFastExtractor
from ..models.vggish import VggishExtractor
from .audio_features import chroma, fourier_tempogram, hpcp, mfcc, onsets, rms, tempogram, tonnetz
from .visual_beats import video_onsets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset = ChainDataset(
#     pytorchvideo.data.LabeledVideoDataset(
#         list(zip(videos, [{} for _ in range(len(videos))])),
#         sampler,
#         transform=ApplyTransformToKey(
#             key="video",
#             transform=Compose(
#                 [
#                     UniformTemporalSubsample(dur * fps),
#                     ShortSideScale(size=256),
#                     CenterCrop(256),
#                     Lambda(lambda x: x / 255.0),
#                 ]
#             ),
#         ),
#         decode_audio=True,
#     )
#     for sampler in [pytorchvideo.data.RandomClipSampler(clip_duration=dur) for _ in range(25)]
#     + [pytorchvideo.data.UniformClipSampler(clip_duration=dur)]
# )


def transpose(l):
    return list(map(list, zip(*l)))


class VideoDataset(Dataset):
    def __init__(self, in_dir):
        super().__init__()
        self.filenames = sum([glob(in_dir + "/*" + ext) for ext in [".mp4", ".avi", ".mkv", ".webm"]], [])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        file = self.filenames[index]
        video, audio, info = tv.io.read_video(file)
        video = center_crop(resize(video.permute(0, 3, 1, 2), 256, antialias=True), 256).permute(0, 2, 3, 1)
        video = video[: fps * (len(video) // fps)]
        n_secs = len(video) // fps
        audio_pad = torch.zeros(max(n_secs * info["audio_fps"], audio.shape[1]))
        audio_pad[: audio.shape[1]] = audio.mean(0)
        audio = torchaudio.transforms.Resample(info["audio_fps"], 16384)(audio_pad)
        audio = audio[: 16384 * (len(audio) // 16384)]
        return video, audio, file


@torch.inference_mode()
def preprocess_video(in_dir, out_stem, dur, fps):
    if not os.path.exists(out_stem + "_files.npy"):

        slowfast = SlowFastExtractor()  # TODO input normalization correct?
        vggish = VggishExtractor()  # TODO rearrange correct?

        feature_cache_files = None
        for batch in tqdm(
            DataLoader(VideoDataset(in_dir), batch_size=1, num_workers=1, shuffle=True),
            desc="Encoding videos to audio/visual features...",
        ):
            try:
                batch_local = deepcopy(batch)
                del batch
                video, audio, (file,) = batch_local

                video_feats = {f"slowfast_layer{f}": feat for f, feat in enumerate(slowfast(video.squeeze()))}
                video_feats["onsets"] = video_onsets(video.squeeze())
                del video
                gc.collect()
                onslen = len(video_feats["onsets"])
                video_feats["tempogram"] = tempogram(onsets=video_feats["onsets"])[:onslen]
                video_feats["fourier_tempogram"] = fourier_tempogram(onsets=video_feats["onsets"])[:onslen]

                for k in video_feats:
                    video_feats[k] = torch.stack(torch.split(video_feats[k].squeeze(), fps))
                chunks = len(video_feats[k])

                audio_feats = {f"vggish_layer{f}": feat for f, feat in enumerate(vggish(audio))}
                audio = audio.squeeze().numpy()
                audio_feats["onsets"] = onsets(audio)
                onslen = len(audio_feats["onsets"])
                audio_feats["tempogram"] = tempogram(audio)[:onslen]
                audio_feats["fourier_tempogram"] = fourier_tempogram(audio)[:onslen]
                audio_feats["chroma"] = chroma(audio)[:onslen]
                audio_feats["tonnetz"] = tonnetz(audio)[:onslen]
                audio_feats["mfcc"] = mfcc(audio)[:onslen]
                audio_feats["rms"] = rms(audio)[:onslen]
                audio_feats["hpcp"] = hpcp(audio)
                audio_feats["hpcp"] = resize(
                    audio_feats["hpcp"][None, None], (onslen, audio_feats["hpcp"].shape[1]), antialias=True
                ).squeeze()
                del audio
                gc.collect()

                for k in audio_feats:
                    audio_feats[k] = torch.stack(torch.chunk(audio_feats[k].squeeze(), chunks))
                    if audio_feats[k].dim() == 3:
                        audio_feats[k] = resize(
                            audio_feats[k][:, None], (fps, audio_feats[k].shape[2]), antialias=True
                        ).squeeze()
                    elif audio_feats[k].dim() == 2:
                        audio_feats[k] = interpolate(audio_feats[k][:, None], (fps,)).squeeze()
                    else:
                        raise NotImplementedError

                if feature_cache_files is None:
                    feature_cache_files = {
                        **{f"video_{k}": NpyAppendArray(f"{out_stem}_video_{k}.npy") for k in video_feats.keys()},
                        **{f"audio_{k}": NpyAppendArray(f"{out_stem}_audio_{k}.npy") for k in audio_feats.keys()},
                        **{"files": NpyAppendArray(f"{out_stem}_files.npy")},
                    }
                for k, feat in video_feats.items():
                    feature_cache_files[f"video_{k}"].append(np.ascontiguousarray(feat.numpy()))
                for k, feat in audio_feats.items():
                    feature_cache_files[f"audio_{k}"].append(np.ascontiguousarray(feat.numpy()))
                feature_cache_files["files"].append(np.ascontiguousarray([file[0].ljust(50)[:50]]))
            except Exception as e:
                print("\n\n\n\nERROR:Preprocessing failed for file:", file, "\n" + str(e) + "\n\n\n")
        del feature_cache_files


class AudioVisualFeatures(Dataset):
    def __init__(self, base):
        super().__init__()
        self.video_files = {
            Path(f.replace(base + "_", "")).stem: np.load(f, mmap_mode="r") for f in sorted(glob(base + "_video*.npy"))
        }
        self.audio_files = {
            Path(f.replace(base + "_", "")).stem: np.load(f, mmap_mode="r") for f in sorted(glob(base + "_audio*.npy"))
        }
        self.filenames = np.load(base + "_files.npy", mmap_mode="r")

    def __len__(self):
        return len(list(self.video_files.values())[0])

    def __getitem__(self, index):
        video_features = {k: torch.from_numpy(ff[index].copy()) for k, ff in self.video_files.items()}
        audio_features = {k: torch.from_numpy(ff[index].copy()) for k, ff in self.audio_files.items()}
        return video_features, audio_features, self.filenames[index]


if __name__ == "__main__":
    dur = 4
    fps = 24

    benny_dir = "/home/hans/datasets/audiovisual/trashbenny"
    benny_cache = f"cache/{Path(benny_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(benny_dir, benny_cache, dur, fps)

    maua_dir = "/home/hans/datasets/audiovisual/maua256"
    maua_cache = f"cache/{Path(maua_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(maua_dir, maua_cache, dur, fps)

    phony_dir = "/home/hans/datasets/audiovisual/phony"
    phony_cache = f"cache/{Path(phony_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(phony_dir, phony_cache, dur, fps)

    trapnation_dir = "/home/hans/datasets/audiovisual/trapnation"
    trapnation_cache = f"cache/{Path(trapnation_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(trapnation_dir, trapnation_cache, dur, fps)

    invocation_dir = "/home/hans/datasets/audiovisual/invocation"
    invocation_cache = f"cache/{Path(invocation_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(invocation_dir, invocation_cache, dur, fps)

    varied_dir = "/home/hans/datasets/audiovisual/varied"
    varied_cache = f"cache/{Path(varied_dir).stem}_features_{dur}sec_{fps}fps"
    preprocess_video(varied_dir, varied_cache, dur, fps)
