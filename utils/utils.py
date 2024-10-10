import os
import sys
import subprocess
import contextlib
import wave


def is_dir(directory: str) -> bool:
    return os.path.isdir(directory)


def is_file(filename: str) -> bool:
    return os.path.isfile(path=filename)


def create_dir(directory: str) -> bool:
    try:
        return os.mkdir(directory)
    except FileExistsError:
        print(f"{directory} already exists")
        return False


def crawl_directory(directory: str, extension: str = None) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            if extension is not None:
                if _file.endswith(extension):
                    tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))
    return tree


def create_dir(directory: str) -> bool:
    try:
        return os.mkdir(directory)
    except FileExistsError:
        print(f"{directory} already exists")
        return False


def blockPrint():
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__


def get_wav_duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
