from .random_patches import *


def render():
    # synthesizer = StyleGAN2Synthesizer(model_file, False, (512, 512), "stretch", 0).cuda()
    # with VideoWriter(output_file, synthesizer.output_size, fps, audio_file, offset, dur, "slow") as video, NpArr(
    #     output_file.replace(".mp4", "_latent.npy")
    # ) as latent_file:
    #     for _ in trange(round(dur * fps)):
    #         lat = latent().cuda()
    #         frame = synthesizer(latents=lat).add(1).div(2)
    #         video.write(frame)
    #         latent_file.append(np.ascontiguousarray(lat.cpu().numpy()))

    # joblib.dump(patches, output_file.replace(".mp4", f"_patch.pkl"), compress=9)
    # with open(output_file.replace(".mp4", f"_patch.txt"), "w") as f:
    #     for patch in patches:
    #         f.write(f"{patch}")

    fig, ax = plt.subplots(2, len(feats), figsize=(8 * len(feats), 16))
    for i in range(len(feats)):
        for j in range(len(feats[i])):
            af = feats[i][j]
            if isinstance(af, float):
                rosa_display.specshow(
                    rosa.feature.tempogram(audio.numpy(), sr),
                    sr=sr,
                    x_axis="time",
                    y_axis="tempo",
                    ax=ax.flatten()[i * len(feats) + j],
                )
                ax.flatten()[i * len(feats) + j].axhline(
                    af, color="w", linestyle="--", alpha=1, label="Estimated tempo={:g}".format(af)
                )
                ax.flatten()[i * len(feats) + j].legend(loc="upper right")
            elif len(af.squeeze().shape) == 1:
                ax.flatten()[i * len(feats) + j].plot(af.squeeze().numpy())
            elif af.shape[1] == 6:
                rosa_display.specshow(
                    np.rollaxis(af.numpy(), 1, 0),
                    sr=sr,
                    x_axis="time",
                    y_axis="tonnetz",
                    ax=ax.flatten()[i * len(feats) + j],
                )
            elif af.shape[1] == 12:
                rosa_display.specshow(
                    np.rollaxis(af.numpy(), 1, 0),
                    sr=sr,
                    x_axis="time",
                    y_axis="chroma",
                    ax=ax.flatten()[i * len(feats) + j],
                )
            else:
                raise Exception(f"what? {af.shape} {patches[i].feature.__class__.__name__}")
            ax.flatten()[i * len(feats) + j].set(
                title=patches[i].source.__class__.__name__
                + " "
                + patches[i].filter.__class__.__name__
                + " "
                + patches[i].target.__class__.__name__
                + (" " + patches[i].postprocess.__class__.__name__ if j else "")
            )
    plt.tight_layout()
    plt.savefig(output_file.replace(".mp4", ".pdf"))


if __name__ == "__main__":

    audio_file = random.choice(glob("/home/hans/datasets/wavefunk/*"))
    model_file = "/home/hans/modelzoo/wavefunk/cyphept-1024-010000.pt"  # random.choice(glob("/home/hans/modelzoo/wavefunk/*/*.pt") + glob("/home/hans/modelzoo/wavefunk/*.pt"))
    checkpoint_name = Path(model_file.replace("/network-snapshot", "")).stem
    checkpoint_name = "_".join(sorted(checkpoint_name.replace("-", "_").split("_"), key=lambda k: random.random())[:7])
    fps = 24
    dur = 30
    offset = random.choice([0, 30, 60, 90, 120, 180])
    full_duration = rosa.get_duration(filename=audio_file)
    offset = max(0, min(full_duration - dur, offset))

    audio, sr, dur = load_audio(audio_file, offset=offset, duration=dur)

    with torch.inference_mode():

        patches = []
        for pat in range(10):
            which_audio = random.choice(["full", "drums", "percussive"])
            which_filtering = random.choice(["none", "none", "none", "low", "low-mid", "mid", "high-mid", "high"])
            which_input = "latent"
            which_feature = "onsets"
            which_postprocess = random.choice(["none", "smooth", "clip", "compress", "expand"])
            which_layers = "not-implemented"
            print(which_audio, which_filtering, which_postprocess)
            patches.append(
                RandomPatch(which_audio, which_filtering, which_input, which_feature, which_postprocess, which_layers)
            )

        mapper = StyleGAN2Mapper(model_file, False)
        # for l in range(5):
        latent_selection = mapper(torch.randn(12, 512))
        for p, patch in enumerate(patches):
            output_file = f"/home/hans/neurout/ssar/drum_patches/random_patch_{Path(audio_file).stem}_{checkpoint_name[:40]}_{p}.mp4"
            latent, f, p = patch(
                audio=audio,
                sr=sr,
                dur=dur,
                fps=fps,
                audio_file=audio_file,
                latent_selection=latent_selection,
                size=64,
            )
            feats = [[f, p]]
            render()
