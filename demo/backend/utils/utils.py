import os

import librosa
import numpy as np

SEGMENT_SIZE = 8000
SR = 8000
HOP_SIZE = SR // 2


def crawl_directory(directory: str) -> list:
    if not os.path.is_dir(directory):
        raise FileNotFoundError

    subdirs = [folder[0] for folder in os.walk(directory)]
    tree = []
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            tree.append(os.path.join(subdir, _file))
    return tree


def extract_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 8000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 256,
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # convert to dB for log-power mel-spectrograms
    return librosa.power_to_db(S, ref=np.max)


def prepare_fingerprints(song: str, create_index=True):
    xb = []
    idxs = []
    features = np.load(song)
    for f in features:
        if create_index:
            xb.append(f)
        idxs.append(song.removesuffix(".npy"))
    if create_index:
        return np.stack(xb), np.array(idxs)
    else:
        return np.array(idxs)


def get_indices(
    y: np.ndarray,
    sr: int = SR,
    segment_size: int = SEGMENT_SIZE,
    hop_size: int = HOP_SIZE,
) -> np.ndarray:
    """Find the starting points of each segment for given segment & hop size.

    Args:
        y (np.ndarray): The audio signal.
        sr (int): The sampling rate of each audio sample
        segment_size (int): The segment size, if equal to sr then 1sec.
        hop_size (int): The hop size.

    Returns:
        np.ndarray: The starting points of each segment
    """
    if y.size > sr:
        last = (y.size // sr) * sr
        indices = list(range(0, last, hop_size))
        if hop_size < sr:
            _ = indices.pop(-1)
        return indices
    else:
        return None
