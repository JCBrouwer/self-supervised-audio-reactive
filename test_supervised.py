import argparse

from train_supervised import audio2video, Audio2Latent

parser = argparse.ArgumentParser()
parser.add_argument("ckpt", help="path to Audio2Latent checkpoint to test")
parser.add_argument("audio", help="path to audio file to test with")
parser.add_argument("stylegan", help="path to StyleGAN model file to test with")
parser.add_argument("--output_size", help="output size for StyleGAN model rendering", nargs=2, default=[512, 512])
parser.add_argument("--batch_size", help="batch size for StyleGAN model rendering", type=int, default=8)
parser.add_argument("--offset", help="time in audio to start from in seconds", type=int, default=90)
parser.add_argument("--duration", help="length in seconds of video to render", type=int, default=60)
args = parser.parse_args()

if __name__ == "__main__":
    audio2video(
        args.ckpt,
        args.audio,
        args.stylegan,
        output_size=[int(s) for s in args.output_size],
        batch_size=args.batch_size,
        offset=args.offset,
        duration=args.duration,
    )
