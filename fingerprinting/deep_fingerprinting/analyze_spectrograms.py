import argparse
import os

from tqdm import tqdm

from utils.utils import (analyze_mel_spectrograms, crawl_directory, create_dir,
                         is_dir, is_dir_empty, is_file, stereo_to_mono)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/songs",
        help="Input directory of audio files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="spectograms/",
        help="Output directory of spectograms",
    )

    args = parser.parse_args()
    return args


def main(directory: str, output: str) -> None:
    tree = crawl_directory(directory)

    for audio_file in tqdm(tree):
        dst = os.path.join(output, audio_file.split(os.sep)[-1].replace(".wav", ""))
        create_dir(dst)
        analyze_mel_spectrograms(audio_file, dst)


if __name__ == "__main__":
    args = parse_args()

    # if not is_dir(args.input):
    #     raise FileNotFoundError(f"{args.input} is not a directory")
    # if not is_dir(args.output):
    #     raise FileNotFoundError(f"{args.input} is not a directory")

    main(args.input, args.output)

    # if not is_dir(args.input):
    #     raise FileNotFoundError(f"{args.input} is not a directory")
    # if not is_dir(args.output):
    #     raise FileNotFoundError(f"{args.input} is not a directory")

    main(args.input, args.output)
