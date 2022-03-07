import math
import os
import random
import sys
import uuid
import warnings

import librosa as rosa
import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

sys.path.append("/home/hans/code/maua-stylegan2/")
import audioreactive as ar
from generate_audiovisual import generate

track_info = [
    {
        "id": 0,
        "title": "Halogenix - Vex",
        "start": 0,
        "end": 2 * 60 + 36.279,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____darkblu_latents.npy",
        "vibe": "liquid",
        "uuid": "5e5045",
    },
    {
        "id": 1,
        "title": "Air. K & Cephei & Minos - No Promises",
        "start": 2 * 60 + 47.442,
        "end": 4 * 60 + 16.744,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____tropical_latents.npy",
        "vibe": "liquid",
        "uuid": "4fbca2",
    },
    {
        "id": 2,
        "title": "Alix Perez & Spectrasoul - The Need",
        "start": 4 * 60 + 16.744,
        "end": 5 * 60 + 46.047,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____licht2_latents.npy",
        "vibe": "liquid",
        "uuid": "130865",
    },
    {
        "id": 3,
        "title": "GLXY & Steo - Conclusions",
        "start": 6 * 60 + 8.372,
        "end": 7 * 60 + 37.674,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____lichtblauw_latents.npy",
        "vibe": "liquid",
        "uuid": "47510e",
    },
    {
        "id": 4,
        "title": "Culprate - No Words (Dexcell Remix)",
        "start": 8 * 60,
        "end": 9 * 60 + 29.302,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____light2_latents.npy",
        "vibe": "liquid",
        "uuid": "79a20b",
    },
    {
        "id": 5,
        "title": "Halogenix - The Night ft. SOLAH",
        "start": 9 * 60 + 34.884,
        "end": 11 * 60 + 4.554,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____darkblu2_latents.npy",
        "vibe": "liquid",
        "uuid": "8999d2",
    },
    {
        "id": 6,
        "title": "RefraQ - Fold (feat. Chalky)",
        "start": 11 * 60 + 29.051,
        "end": 12 * 60 + 40.769,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____blupurp_latents.npy",
        "vibe": "liquid",
        "uuid": "7f6558",
    },
    {
        "id": 7,
        "title": "Halogenix - Independent",
        "start": 13 * 60 + 3.554,
        "end": 14 * 60 + 55.182,
        "latent": "workspace/independent-latents.npy",
        "vibe": "tech",
        "uuid": "4cf734",
    },
    {
        "id": 8,
        "title": "Alix Perez & Spectrasoul - So Close",
        "start": 15 * 60 + 17.507,
        "end": 16 * 60 + 46.810,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____lichtblauw_latents.npy",
        "vibe": "liquid",
        "uuid": None,
    },
    {
        "id": 9,
        "title": "QZB & Amoss - Tigra",
        "start": 17 * 60 + 9.135,
        "end": 18 * 60 + 38.437,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____darkblu2_latents.npy",
        "vibe": "liquid",
        "uuid": "a6e748",
    },
    {
        "id": 10,
        "title": "Monty & Visages - Black Shield",
        "start": 19 * 60 + 0.763,
        "end": 20 * 60 + 52.391,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____darkblu_latents.npy",
        "vibe": "tech",
        "uuid": "1536a2",
    },
    {
        "id": 11,
        "title": "Synergy - Intervals",
        "start": 21 * 60 + 14.717,
        "end": 21 * 60 + 59.368,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____blauwrood_latents.npy",
        "vibe": "tech",
        "uuid": "25456d",
    },
    {
        "id": 12,
        "title": "GROUND - Amnesia",
        "start": 22 * 60 + 34.321,
        "end": 24 * 60 + 13.321,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker_latents.npy",
        "vibe": "tech",
        "uuid": "a9d150",
    },
    {
        "id": 13,
        "title": "QZB & Emperor - Nixon",
        "start": 24 * 60 + 24.484,
        "end": 25 * 60 + 18.903,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker2_latents.npy",
        "vibe": "tech",
        "uuid": "b2bf27",
    },
    {
        "id": 14,
        "title": "DLR & Break - Hit the Target",
        "start": 25 * 60 + 31.461,
        "end": 27 * 60 + 23.089,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker3_latents.npy",
        "vibe": "tech",
        "uuid": "d3eb33",
    },
    {
        "id": 15,
        "title": "QZB - Elektron Dub",
        "start": 27 * 60 + 23.089,
        "end": 28 * 60 + 30.115,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____dark_latents.npy",
        "vibe": "tech",
        "uuid": "995073",
    },
    {
        "id": 16,
        "title": "Fre4knc - Rotor",
        "start": 28 * 60 + 52.391,
        "end": 30 * 60 + 19.251,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker_latents.npy",
        "vibe": "tech",
        "uuid": "29c35b",
    },
    {
        "id": 17,
        "title": "Data 3 - Tyrant",
        "start": 30 * 60 + 44.019,
        "end": 32 * 60 + 13.321,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker2_latents.npy",
        "vibe": "tech",
        "uuid": "308fd0",
    },
    {
        "id": 18,
        "title": "GROUND - Sleep Talking",
        "start": 32 * 60 + 46.810,
        "end": 34 * 60 + 16.112,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker3_latents.npy",
        "vibe": "tech",
        "uuid": "29fa55",
    },
    {
        "id": 19,
        "title": "Molecular - NRG",
        "start": 34 * 60 + 27.275,
        "end": 35 * 60 + 56.577,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____dark_latents.npy",
        "vibe": "tech",
        "uuid": "307e08",
    },
    {
        "id": 20,
        "title": "Grey Code - Masque",
        "start": 36 * 60 + 18.903,
        "end": 37 * 60 + 31.461,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____neuro-rood2_latents.npy",
        "vibe": "neuro",
        "uuid": "66075c",
    },
    {
        "id": 21,
        "title": "Ewol & Espired - Malfunction VIP",
        "start": 37 * 60 + 53.786,
        "end": 38 * 60 + 49.6,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____darkblu2_latents.npy",
        "vibe": "neuro",
        "uuid": None,
    },
    {
        "id": 22,
        "title": "Audio - Collision",
        "start": 39 * 60 + 11.926,
        "end": 41 * 60 + 37.042,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____neuro-green_latents.npy",
        "vibe": "neuro",
        "uuid": "e7b3f3",
    },
    {
        "id": 23,
        "title": "Noisia - Tentacles",
        "start": 42 * 60 + 21.344,
        "end": 43 * 60 + 6.344,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____tentacles_latents.npy",
        "vibe": "neuro",
        "uuid": "29322c",
    },
    {
        "id": 24,
        "title": "InsideInfo & Mefjus - Mythos",
        "start": 43 * 60 + 28.67,
        "end": 44 * 60 + 57.972,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____neuro-blauw_latents.npy",
        "vibe": "neuro",
        "uuid": "87d4ad",
    },
    {
        "id": 25,
        "title": "Barely Alive x Ewol & Espired - Fireflies",
        "start": 45 * 60 + 31.461,
        "end": 46 * 60 + 38.437,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____neuro-donker_latents.npy",
        "vibe": "neuro",
        "uuid": "547d5a",
    },
    {
        "id": 26,
        "title": "Culprate - Fester",
        "start": 47 * 60 + 0.763,
        "end": 48 * 60 + 30.065,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____darkblu_latents.npy",
        "vibe": "neuro",
        "uuid": "532f82",
    },
    {
        "id": 27,
        "title": "Moody Good - Super Lager",
        "start": 49 * 60 + 17.703,
        "end": 50 * 60 + 30.065,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____neuro-rood_latents.npy",
        "vibe": "neuro",
        "uuid": "d12413",
    },
    {
        "id": 28,
        "title": "Disprove - Damage",
        "start": 51 * 60 + 3.554,
        "end": 52 * 60 + 55.182,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____neuro-dc_latents.npy",
        "vibe": "neuro",
        "uuid": "40f7c0",
    },
    {
        "id": 29,
        "title": "Parallax - Plunged",
        "start": 53 * 60 + 17.507,
        "end": 54 * 60 + 24.484,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker2_latents.npy",
        "vibe": "neuro",
        "uuid": "5aa1cb",
    },
    {
        "id": 30,
        "title": "Abis & Signal & Tasha Baxter - The Wall",
        "start": 54 * 60 + 46.81,
        "end": 57 * 60 + 11.926,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____donker3_latents.npy",
        "vibe": "neuro",
        "uuid": "354c07",
    },
    {
        "id": 31,
        "title": "Apashe feat. Geoffroy - Distance (Buunshin Remix)",
        "start": 57 * 60 + 45.414,
        "end": 60 * 60 + 24.484,
        "latent": "workspace/alaeset-GPU1-files-1024-013000____neuro-bw_latents.npy",
        "vibe": "neuro",
        "uuid": "b6c1ad",
    },
    {
        "id": "dummy",
        "title": "dummy",
        "start": 60 * 60 + 24.484,
        "end": 60 * 60 + 24.484,
        "latent": "dummy",
        "vibe": "dummy",
        "uuid": "",
    },
]


track_id_start = 0  # int(os.environ["START"])
track_id_end = 31  # int(os.environ["END"])

uuid = str(uuid.uuid4())[:6]
if track_id_start == track_id_end:
    output_file = f"workspace/donderslag-{track_info[track_id_start]['title'].replace(' ', '').lower()}-{uuid}.mp4"
else:
    output_file = f"workspace/donderslag-{track_info[track_id_start]['title'].replace(' ','').lower()}-tot-{track_info[track_id_end]['title'].replace(' ','').lower()}-{uuid}.mp4"

FPS = 24
OVERRIDE = {
    "offset": track_info[track_id_start]["start"],
    "duration": track_info[track_id_end]["end"] - track_info[track_id_start]["start"],
    "fps": FPS,
    "output_file": output_file,
}

vid_start = track_info[track_id_start]["start"]
vid_end = track_info[track_id_end]["end"]
vid_dur = vid_end - vid_start
vid_frames = int(vid_dur * FPS)

liquid_end = 21 * 60 + 59.368
tech_start = 22 * 60 + 34.321
tech_end = 35 * 60 + 56.577
neuro_start = 36 * 60 + 18.903

# interstitial = ar.load_latents("workspace/alaeset-GPU1-files-1024-013000____donker_latents.npy")
# interstitial = torch.cat([interstitial[:-6], interstitial[-3:]])


def get_latents(selection, args):
    all_latents = []
    for track, next_track in zip(track_info, track_info[1:]):
        title, start, end, latent, vibe = track["title"], track["start"], track["end"], track["latent"], track["vibe"]
        print("latents", title)
        start_time = max(start, vid_start)
        end_time = min(next_track["start"], vid_end)
        start_frame = int((start_time * FPS))
        end_frame = int((end_time * FPS))
        trans = int((end_time - track["end"]) * FPS)
        if end_frame <= int(vid_start * FPS) or start_frame >= int(vid_end * FPS):
            continue
        track_frames = end_frame - start_frame
        start_frame -= int(vid_start * FPS)
        end_frame -= int(vid_start * FPS)

        bpm = 160 if "RefraQ" in title else 172
        num_bars = (end - start) * (172 / 60) / 4
        phrase_frames = int((16 * 4 * (60 / 172) * FPS))
        num_phrases = int(math.ceil(1.01 * num_bars / 16))

        # print(
        #     f"{title.ljust(40)}   frames: {start_frame} -> {end_frame}   duration: {end_time - start_time:.2f} sec   bars: {num_bars:.1f} in {num_phrases} phrases   transition: {trans} frames"
        # )
        # write_wav(
        #     f"/home/hans/datasets/audio2latent/Wavefunk @ Donderslag II - {title}.wav",
        #     args.sr,
        #     args.audio[int(start_time * args.sr) : int(end * args.sr)],
        # )

        if not os.path.exists(f"workspace/{title.replace(' ','').lower()}_ba_onsets.npy"):
            drums, args.sr = rosa.load(
                args.audio_file.replace(".wav", "-drums.wav"),
                offset=start_time,
                duration=min(next_track["start"], vid_end) - start_time,
            )
            bass, args.sr = rosa.load(
                args.audio_file.replace(".wav", "-bass.wav"),
                offset=start_time,
                duration=min(next_track["start"], vid_end) - start_time,
            )
            if vibe == "liquid":
                # low_onsets = ar.onsets(drums, args.sr, track_frames, fmax=150, clip=95, smooth=2, power=1)
                # high_onsets = ar.onsets(drums, args.sr, track_frames, fmin=150, clip=93, smooth=2, power=1)
                bas_onsets = ar.rms(bass, args.sr, track_frames, smooth=6, clip=97, power=1)
                drum_onsets = 1.5 * ar.onsets(drums, args.sr, track_frames, fmin=150, clip=94, smooth=3, power=1)

                # kick_penalty = 2 * ar.expand(ar.percentile_clip(low_onsets, 90), 0.25, 10)
                # high_onsets -= kick_penalty
                # high_onsets = np.maximum(high_onsets, np.zeros_like(high_onsets))
                # high_onsets = ar.percentile_clip(high_onsets, 95)
            elif vibe == "neuro":
                # low_onsets = ar.onsets(drums, args.sr, track_frames, fmax=150, clip=95, smooth=1)
                # high_onsets = ar.onsets(drums, args.sr, track_frames, fmin=150, clip=95, smooth=1)
                bas_onsets = ar.rms(bass, args.sr, track_frames, smooth=2, clip=95, power=2)
                drum_onsets = 2 * ar.onsets(drums, args.sr, track_frames, fmin=150, clip=90, smooth=3, power=1)
            else:
                bas_onsets = ar.rms(bass, args.sr, track_frames, smooth=2, clip=95, power=1)
                bas_onsets = torch.cat([bas_onsets[int(FPS / 2) :], bas_onsets[: int(FPS / 2)]])
                drum_onsets = 2 * ar.onsets(drums, args.sr, track_frames, fmin=150, clip=90, smooth=3, power=1)
            if "Amnesia" in track["title"]:
                bas_onsets *= 0.666
            if "Nixon" in track["title"]:
                bas_onsets *= 0.75
            if "NRG" in track["title"]:
                bas_onsets = ar.percentile_clip(bas_onsets, 93)

            # import scipy.io.wavfile
            # import scipy.signal as signal

            # scipy.io.wavfile.write(
            #     f"workspace/{title.replace(' ','').lower()}-hi.wav",
            #     args.sr,
            #     drums * signal.resample(high_onsets, len(drums)),
            # )
            # scipy.io.wavfile.write(
            #     f"workspace/{title.replace(' ','').lower()}-lo.wav",
            #     args.sr,
            #     drums * signal.resample(low_onsets, len(drums)),
            # )

            # signals = [drum_onsets, bas_onsets]
            # plt.figure(figsize=(16, 4 * len(signals)))
            # for sbplt, y in enumerate(signals):
            #     try:
            #         signal = signal.cpu().numpy()
            #     except:
            #         pass
            #     plt.subplot(len(signals), 1, sbplt + 1)
            #     plt.plot(y.squeeze())
            # plt.tight_layout()
            # plt.savefig(f"workspace/{title.replace(' ','').lower()}.pdf")
            # plt.close()

            # track_info[track["id"]]["lo_onsets"] = low_onsets.clone()
            # track_info[track["id"]]["hi_onsets"] = high_onsets.clone()

            np.save(f"workspace/{title.replace(' ','').lower()}_ba_onsets.npy", bas_onsets.numpy())
            np.save(f"workspace/{title.replace(' ','').lower()}_dr_onsets.npy", drum_onsets.numpy())
        else:
            bas_onsets = torch.from_numpy(np.load(f"workspace/{title.replace(' ','').lower()}_ba_onsets.npy")).float()
            drum_onsets = torch.from_numpy(np.load(f"workspace/{title.replace(' ','').lower()}_dr_onsets.npy")).float()

        track_info[track["id"]]["ba_onsets"] = bas_onsets.clone()
        track_info[track["id"]]["dr_onsets"] = drum_onsets.clone()

        if not os.path.exists(f"workspace/{title.replace(' ','').lower()}_latents.npy"):
            # if track["uuid"] is None:
            #     track_selection = ar.load_latents(latent)
            #     if args.shuffle_latents:
            #         random_indices = random.sample(range(len(track_selection)), len(track_selection))
            #         track_selection = track_selection[random_indices]
            #     np.save(
            #         f"workspace/{title.replace(' ','').lower()}_chroma_latents_{uuid}.npy",
            #         track_selection[:12],
            #     )
            # else:
            #     track_selection = ar.load_latents(
            #         f"workspace/{title.replace(' ','').lower()}_chroma_latents_{track['uuid']}.npy"
            #     )
            track_selection = ar.generate_latents(n_latents=12, ckpt=args.ckpt, G_res=1024, noconst=True)

            chroma = ar.chroma(
                args.audio[int(start_time * args.sr) : int(end * args.sr)], args.sr, track_frames - trans
            )
            chroma_latents = ar.chroma_weight_latents(chroma, track_selection[:12])
            latents = ar.gaussian_filter(chroma_latents, 4)

            # lo_onsets = low_onsets[: len(latents), None, None]
            # hi_onsets = high_onsets[: len(latents), None, None]
            ba_onsets = bas_onsets[: len(latents), None, None]
            dr_onsets = drum_onsets[: len(latents), None, None]

            for ons in ["dr", "ba"]:
                # select random latent per 16 bars
                phrase = []
                if True:  # track["uuid"] is None:
                    indices = random.sample(range(len(track_selection)), num_phrases)
                    phrase_selection = track_selection[indices]
                    np.save(f"workspace/{title.replace(' ','').lower()}_{ons}_ons_latents_{uuid}.npy", phrase_selection)
                else:
                    phrase_selection = ar.load_latents(
                        f"workspace/{title.replace(' ','').lower()}_{ons}_ons_latents_{track['uuid']}.npy"
                        if not ons == "dr"
                        else f"workspace/{title.replace(' ','').lower()}_hi_ons_latents_{track['uuid']}.npy"
                    )
                for i in range(num_phrases):
                    phrase += [phrase_selection[[i]]] * phrase_frames
                phrase = torch.cat(phrase[: len(latents)])
                phrase = ar.gaussian_filter(phrase, 8)
                latents = eval(f"{ons}_onsets") * phrase + (1 - eval(f"{ons}_onsets")) * latents

            np.save(f"/home/hans/datasets/audio2latent/Wavefunk @ Donderslag II - {title}.npy", latents.numpy())
        else:
            latents = torch.from_numpy(np.load(f"workspace/{title.replace(' ','').lower()}_latents.npy")).float()
        all_latents.append(latents)

        # if trans > 0:
        #     if not os.path.exists(f"workspace/{title.replace(' ','').lower()}_transition_latents.npy"):
        #         num_latents = max(4, min(int(trans / 24 / 4), 12))  # ~ latent per 4 second
        #         transition_latents = ar.spline_loops(interstitial[:num_latents], trans, 1, loop=False)
        #         color_latents = torch.cat(
        #             [
        #                 track_selection[: int(math.floor(num_latents / 2))],
        #                 ar.load_latents(next_track["latent"])[: int(math.ceil(num_latents / 2))],
        #             ]
        #         )
        #         transition_colors = ar.spline_loops(color_latents, trans, 1, loop=False)
        #         colayr = 12
        #         transition_latents[colayr:] = transition_colors[colayr:]

        #         np.save(f"workspace/{title.replace(' ','').lower()}_transition_latents.npy", transition_latents.numpy())
        #     else:
        #         transition_latents = torch.from_numpy(
        #             np.load(f"workspace/{title.replace(' ','').lower()}_transition_latents.npy")
        #         ).float()
        #     all_latents.append(transition_latents)
        print(f"latent length: {len(torch.cat(all_latents))}")

    all_latents = torch.cat(all_latents).float()
    all_latents = ar.gaussian_filter(all_latents, 3, causal=0.2)

    return all_latents[2:vid_frames]


def get_noise(height, width, scale, num_scales, args):
    if width > 32:
        return None

    all_noise = []
    for track, next_track in zip(track_info, track_info[1:]):
        title, start, end, latent, vibe = track["title"], track["start"], track["end"], track["latent"], track["vibe"]
        print("noise", title)
        start_time = max(start, vid_start)
        end_time = min(next_track["start"], vid_end)
        start_frame = int((start_time * FPS))
        end_frame = int((end_time * FPS))
        trans = int((end_time - track["end"]) * FPS)
        if end_frame <= int(vid_start * FPS) or start_frame >= int(vid_end * FPS):
            continue
        track_frames = end_frame - start_frame
        start_frame -= int(vid_start * FPS)
        end_frame -= int(vid_start * FPS)

        noise_file = f"/home/hans/datasets/audio2latent/Wavefunk @ Donderslag II - {title} - Noise {width}.npy"
        if not os.path.exists(noise_file):
            # lo_onsets = track_info[track["id"]]["lo_onsets"][:, None, None, None]
            # hi_onsets = track_info[track["id"]]["hi_onsets"][:, None, None, None]
            ba_onsets = track_info[track["id"]]["ba_onsets"][:, None, None, None]
            dr_onsets = track_info[track["id"]]["dr_onsets"][:, None, None, None]
            if vibe == "liquid":
                n = [20, 1, 96]
                noise_factor = 0.8
            elif vibe == "tech":
                n = [10, 1, 64]
                noise_factor = 0.9
            else:
                n = [5, 1, 64]
                noise_factor = 1

            noise_noisy = (
                9 * ar.gaussian_filter(torch.randn((track_frames, 1, height, width), device="cuda"), n[0]).cpu()
            )
            noise_noiser = (
                4 * ar.gaussian_filter(torch.randn((track_frames, 1, height, width), device="cuda"), n[1]).cpu()
            )
            noise = 27 * ar.gaussian_filter(torch.randn((track_frames, 1, height, width), device="cuda"), n[2]).cpu()

            if scale < 9:
                noise = ba_onsets * noise_noisy + (1 - ba_onsets) * noise
            if scale > 2:
                noise = dr_onsets * noise_noiser + (1 - dr_onsets / 2) * noise

            noise_norm = noise / ar.gaussian_filter(noise.std(axis=(1, 2, 3)), 12)[:, None, None, None]
            noise = 1 / 2 * noise + 1 / 2 * noise_norm
            noise *= noise_factor

            np.save(noise_file, ar.gaussian_filter(noise, 3, causal=0.2).numpy())
        else:
            noise = torch.from_numpy(np.load(noise_file)).float()

        all_noise.append(noise)

    all_noise = torch.cat(all_noise)

    # for track, next_track in zip(track_info[1:], track_info[2:]):
    #     title, start, end, latent, vibe = track["title"], track["start"], track["end"], track["latent"], track["vibe"]
    #     start_time = max(start, vid_start)
    #     end_time = min(next_track["start"], vid_end)
    #     start_frame = int((start_time * FPS))
    #     end_frame = int((end_time * FPS))
    #     trans = int((end_time - track["end"]) * FPS)
    #     if end_frame <= int(vid_start * FPS) or start_frame >= int(vid_end * FPS):
    #         continue
    #     track_frames = end_frame - start_frame
    #     start_frame -= int(vid_start * FPS)
    #     end_frame -= int(vid_start * FPS)

    #     all_noise[start_frame - FPS : start_frame + FPS] = ar.gaussian_filter(
    #         all_noise[start_frame - 3 * FPS : start_frame + 3 * FPS], 5, causal=0.2
    #     )[2 * args.fps : -2 * args.fps]

    return all_noise[2:vid_frames]


def get_bends(args):
    class Bigger0thLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise = torch.cuda.FloatTensor([])

        def forward(self, x):
            if self.noise.shape[0] != x.shape[0]:
                self.noise = torch.cat([torch.normal(x.mean(), x.std(), size=(1, x.shape[1], 4, 8))] * x.shape[0])
            return self.noise.to(x.device)

    bends = [{"layer": 0, "transform": Bigger0thLayer()}]

    return bends


def get_truncation(args):
    trunc = []
    for track, next_track in zip(track_info, track_info[1:]):
        start, end, vibe = track["start"], track["end"], track["vibe"]
        start_time = max(start, vid_start)
        end_time = min(next_track["start"], vid_end)
        start_frame = int((start_time * FPS))
        end_frame = int((end_time * FPS))
        if end_frame <= int(vid_start * FPS) or start_frame >= int(vid_end * FPS):
            continue
        track_frames = end_frame - start_frame

        if vibe == "liquid":
            val = 1
        elif vibe == "tech":
            val = 1
        elif vibe == "neuro":
            val = 1.25

        trunc += [val] * track_frames

    trunc = torch.as_tensor(trunc).float()
    trunc = ar.gaussian_filter(trunc, 12)
    return trunc[2:vid_frames]


if __name__ == "__main__":
    generate(
        "/home/hans/modelzoo/maua-sg2/alaeset-GPU1-files-1024-013000___.pt",
        "/home/hans/datasets/donderslag/WAVEFUNK - TVGF DONDERSLAG II WORLDWIDE MIX.flac",
        initialize=None,
        get_latents=get_latents,
        get_noise=get_noise,
        get_bends=get_bends,
        get_rewrites=None,
        get_truncation=get_truncation,
        output_dir="./workspace",
        audioreactive_file="audioreactive/examples/default.py",
        offset=0,
        duration=-1,
        latent_file=None,
        shuffle_latents=False,
        G_res=1024,
        out_size=1024,
        fps=30,
        latent_count=12,
        batch=8,
        dataparallel=False,
        truncation=1.0,
        stylegan1=False,
        noconst=False,
        latent_dim=512,
        n_mlp=8,
        channel_multiplier=2,
        randomize_noise=False,
        ffmpeg_preset="slow",
        base_res_factor=1,
        output_file=None,
        args=None,
    )
