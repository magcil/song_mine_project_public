import os
import subprocess
import sys

from utils.utils import blockPrint, enablePrint


def youtube_dl(playlist: str, output: str):
    result = subprocess.Popen(
        [
            "youtube-dl",
            "-x",
            "--audio-format",
            "wav",
            "--postprocessor-args",
            "-ar 8000 -ac 1",
            "-o",
            f"{output}/%(title)s.%(ext)s",
            "--download-archive",
            "downloaded.txt",
            "--rm-cache-dir",
            "--yes-playlist",
            playlist,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    _, err = result.communicate()
    if len(err) > 0:
        print(f"Error: {err}")
        raise Exception(err)


def spot_dl(playlist: str, output: str):
    retries = 0
    while retries < 3:
        result = subprocess.Popen(
            [
                "spotdl",
                "download",
                playlist,
                "--output",
                f"{output}/{{genre}}_{{title}}.{{output-ext}}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, err = result.communicate()

        if len(err) > 0:
            print(f"Error: {err}")
            retries += 1
            print(f"Retrying {retries} times Downloading {playlist} to {output}")
        else:
            break


def downsample_to_wav(filename: str, output) -> None:
    dst = os.path.join(output, os.path.splitext(os.path.basename(filename))[0] + ".wav")

    result = subprocess.Popen(
        [
            "ffmpeg",
            "-n",
            "-i",
            filename,
            "-ar",
            "8000",
            "-ac",
            "1",
            dst,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, err = result.communicate()

    # if len(err) > 0:
    #     print(f"Error: {err}")
