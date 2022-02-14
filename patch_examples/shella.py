import random
import sys
import uuid

import librosa as rosa
import numpy as np
import torch as th
from render_ddp import render_ddp

import audioreactive as ar
from audioreactive import gaussian_filter, spline_loops

sys.path.append("nvsg2a")
from nvsg2a import dnnlib, legacy

# import nvsg2a.torch_utils.persistence
# def remove_shape_asserts(meta):
#     meta.module_src = meta.module_src.replace("misc.assert_shape", "# misc.assert_shape")
#     return meta
# nvsg2a.torch_utils.persistence.import_hook(remove_shape_asserts)

th.set_grad_enabled(False)
th.backends.cudnn.benchmark = True


def initialize(args):
    drums, drsr = rosa.load(args.audio_file.replace(".wav", "/drums.wav"))
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, clip=94, smooth=1)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets = ar.compress(args.drum_onsets, 0.5, 0.5)
    args.drum_onsets[: int(len(args.drum_onsets) / 3.1)] *= 0.666
    args.drum_onsets = ar.gaussian_filter(args.drum_onsets, 2, causal=0)

    # bass, basr = rosa.load(args.audio_file.replace(".wav", "/bass.wav"))
    # args.bass_onsets = ar.rms(bass, basr, args.n_frames, smooth=1, clip=99, power=1.75)
    # args.bass_onsets = ar.normalize(args.bass_onsets)

    focus, basr = rosa.load(args.audio_file.replace(".wav", "/other.wav"))
    args.focus_onsets = ar.onsets(focus, basr, args.n_frames, smooth=2, clip=95, power=1)
    args.focus_onsets = ar.compress(args.focus_onsets, 0.5, 0.5)
    args.focus_onsets = ar.gaussian_filter(args.focus_onsets, 1)
    args.focus_chroma = ar.chroma(focus, basr, args.n_frames)

    ar.plot_signals([args.drum_onsets, args.focus_onsets])
    # ar.plot_spectra([args.focus_chroma], chroma=True)
    # exit(0)
    return args


def get_latents(selection, args):
    focus_selection = selection
    focus_latents = ar.chroma_weight_latents(args.focus_chroma, focus_selection[:12])

    base_latents = ar.spline_loops(selection[[0, 1, 0, 2, 3, 4, 3, 5, 7, 8, 9, 10]], args.n_frames, 1)
    loop_latents = ar.spline_loops(selection[random.sample(range(len(selection)), len(selection))], args.n_frames, 12)

    # bass_onsets = args.bass_onsets[:, None, None]
    focus_onsets = args.focus_onsets[:, None, None]

    latents = focus_onsets * focus_latents + (1 - focus_onsets) * base_latents
    # latents = 0.25 * bass_onsets * loop_latents + 0.75 * (1 - bass_onsets) * latents
    latents = ar.gaussian_filter(latents.float(), 3, causal=0.2)

    layer = 12
    latents[:, layer:] = base_latents[:, layer:]

    return latents


def get_noise(height, width, scale, num_scales, args):
    if height > 128:
        return None

    # bass_onsets = args.bass_onsets[:, None, None, None]
    drum_onsets = args.drum_onsets[:, None, None, None]

    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 15).cpu()
    noise_noiser = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 2).cpu()
    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 64).cpu()

    # noise = bass_onsets * noise_noisy + (1 - bass_onsets) * noise
    if width > 16:
        noise = drum_onsets * noise_noiser + (1 - drum_onsets) * noise

    noise /= noise.std() * (1 + np.random.rand())

    return noise


if __name__ == "__main__":
    args = Store()
    args.audio_file = "/home/hans/datasets/shella.wav"
    args.audio, args.sr = rosa.load(args.audio_file)

    duration = rosa.get_duration(filename=args.audio_file)
    num_frames = int(round(duration * 24))
    args.n_frames = num_frames

    # args.focus = ar.onsets(args.audio, args.sr, args.n_frames, fmin=330, fmax=500, smooth=5, clip=90, power=1)

    drums, drsr = rosa.load(args.audio_file.replace(".wav", "/drums.wav"))
    # args.lo_onsets = ar.onsets(drums, drsr, args.n_frames, fmax=150, clip=96, smooth=3)
    args.drum_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=500, clip=96, smooth=3)

    bass, basr = rosa.load(args.audio_file.replace(".wav", "/bass.wav"))
    args.bass_onsets = ar.rms(bass, basr, args.n_frames, smooth=4, clip=99, power=1)

    # ar.plot_signals([args.focus, args.drum_onsets, args.bass_onsets])

    device = th.device("cuda")
    output_size = "1024x1024"

    network_pkl = np.random.choice(
        [
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000314.pkl",
            # "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000221.pkl",
            "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000105.pkl",
            "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000424.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000393.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000240.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000135.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000086.pkl",
            # "/home/hans/modelzoo/00026-XXL-mirror-mirrory-aydao-resumecustom/network-snapshot-000564.pkl",
            # "/home/hans/modelzoo/00021-XXL-mirror-mirrory-aydao-resumecustom/network-snapshot-000241.pkl",
            # "/home/hans/modelzoo/00019-lyreca-mirror-mirrory-auto2-resumeffhq1024/network-snapshot-000288.pkl",
            "/home/hans/modelzoo/00006-stylegan2-lyreca-2gpu-config-f/network-snapshot-000019.pkl",
            # "/home/hans/modelzoo/00006-stylegan2-lyreca-2gpu-config-f/network-snapshot-000086.pkl",
            "/home/hans/modelzoo/00007-stylegan2-lyreca-2gpu-config-f/network-snapshot-000049.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000314.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000013.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000019.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000172.pkl",
            # "/home/hans/modelzoo/00031-naomo-mirror-wav-gamma10.4858-resumeffhq1024/network-snapshot-000320.pkl",
            # "/home/hans/modelzoo/00031-naomo-mirror-wav-gamma10.4858-resumeffhq1024/network-snapshot-000320.pkl",
            # "/home/hans/modelzoo/00038-naomo-mirror-wav-resumeffhq1024/network-snapshot-000140.pkl",
            # "/home/hans/modelzoo/00033-naomo-mirror-wav-gamma0.1-resumeffhq1024/network-snapshot-000120.pkl",
            # "/home/hans/modelzoo/00045-naomo-mirror-wav-gamma500-resumeffhq1024/network-snapshot-000240.pkl",
        ]
    )
    # net = network_pkl.split("/")[-2].split("-")[1]
    net = "neurout_665"
    if net == "stylegan2":
        net = network_pkl.split("/")[-2].split("-")[2]

    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)

    if output_size == "1920x1080":
        double_width = th.nn.ReflectionPad2d((2, 2, 0, 0))
        a_lil_noise = th.randn(size=(1, 1, 4, 8), device="cuda")

        def fullHDpls(self, input, output):
            return double_width(output) + a_lil_noise

        G.synthesis.b4.conv1.register_forward_hook(fullHDpls)

    lats = G.mapping(th.randn(size=(12, 512), device="cuda"), c=None).cpu()
    lats = get_latents(lats, args)

    layer = 8
    style = G.mapping(th.randn(size=(1, 512), device="cuda"), c=None)
    lats[:, layer:] = style[:, layer:]
    lats = gaussian_filter(lats, 2)

    nois = []
    for scale in range(5, 8 * 2 + 2):
        nois.append(
            get_noise(
                height=2 ** int(scale / 2),
                width=(2 if not scale == 5 and output_size == "1920x1080" else 1) * 2 ** int(scale / 2),
                args=args,
            )
        )

    del G.mapping, G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels, G.num_ws
    render_ddp(
        synthesis=G.synthesis,
        latents=lats,
        noise=nois,
        batch_size=6,
        duration=duration,
        output_size=output_size,
        output_file=f"/home/hans/neurout/tvgf_{net}-{str(uuid.uuid4())[:8]}-promo-filmpje.mp4",
        audio_file=args.audio_file,
    )
