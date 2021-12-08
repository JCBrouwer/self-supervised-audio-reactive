import gc
import sys
from pathlib import Path

import joblib
import librosa
import torch
import torchaudio
import torchvision as tv
from tqdm import tqdm

from models.reactor import LSTMReactor
from models.stylegan2 import Generator

SR = 24575
DUR = 8
FPS = 24


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    hidden_size = 32
    num_layers = 4
    zoneout = 0.1
    dropout = 0.05
    n_frames = int(DUR * FPS)

    checkpoint = sys.argv[1]
    input_file = sys.argv[2]

    with torch.inference_mode():
        audio_original, sr = torchaudio.load(input_file)
        audio = audio_original.mean(0)
        audio = torchaudio.transforms.Resample(sr, SR)(audio)

        mfcc = torch.from_numpy(librosa.feature.mfcc(y=audio.numpy(), sr=SR, n_mfcc=19, hop_length=1024)).permute(1, 0)
        mfcc /= torch.norm(mfcc)

        chroma = torch.from_numpy(librosa.feature.chroma_cens(y=audio.numpy(), sr=SR, hop_length=1024)).permute(1, 0)
        chroma /= torch.norm(chroma)

        onsets = torch.from_numpy(librosa.onset.onset_strength(y=audio.numpy(), sr=SR, hop_length=1024))[:, None]
        onsets /= torch.norm(onsets)

        input_features = torch.cat((mfcc, chroma, onsets), axis=1).float()

        generator = Generator(size=1024, style_dim=512, n_mlp=2).eval()
        generator.load_state_dict(torch.load("cache/RobertAGonzalves-MachineRay2-AbstractArt-188.pt")["g_ema"])
        generator.requires_grad_(False)

        reactor = LSTMReactor(
            input_size=input_features.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_styles=generator.num_layers,
            dropout=dropout,
            zoneout=zoneout,
        )
        reactor = reactor.eval().to(device)
        state_dict = joblib.load(checkpoint)["reactor"]
        reactor.load_state_dict(state_dict)

        audio_features = input_features.unsqueeze(0).to(device)
        motion_seed = torch.randn((1, hidden_size), device=device)

        w_seq, _, _ = reactor(audio_features, motion_seed)
        w_seq = w_seq.squeeze().cpu()
        w_seq = torch.cat([w_seq, w_seq[:, [-1]]], axis=1)
        # w_seq /= torch.norm(w_seq, 2, dim=2)[..., None]  # HACK w_seq distribution isn't anything like what G expects

        del audio_features, reactor, motion_seed
        gc.collect()
        torch.cuda.empty_cache()

        generator = generator.to(device)

        frames = []
        for i in tqdm(range(0, 150 * batch_size, batch_size)):
            frame, _ = generator([w_seq[i : i + batch_size].to(device)], input_is_latent=True, randomize_noise=False)
            frames.append(frame.cpu())
        frames = torch.cat(frames).permute(0, 2, 3, 1)
        frames = frames.add(1).div(2).clamp(0, 1).mul(255)

        tv.io.write_video(
            f"output/{Path(checkpoint).stem}_{Path(input_file).stem}.mp4",
            video_array=frames.to(torch.uint8),
            fps=FPS,
            video_codec="h264",
            audio_array=audio_original[:, : round(len(frames) / FPS * sr)],
            audio_fps=sr,
            audio_codec="aac",
        )
