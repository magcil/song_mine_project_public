import linecache
import os
import sys
from functools import wraps
from time import time
import random
from typing import Tuple, Optional, Dict
import shutil
from collections import Counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, TimeMask, SpecFrequencyMask
import soundfile as sf


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def timeit(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} took {end - start} seconds")

    return wrapper


def is_dir(directory: str) -> bool:
    return os.path.isdir(directory)


def is_dir_empty(directory: str) -> bool:
    return len(os.listdir(directory)) == 0


def create_dir(directory: str) -> bool:
    try:
        return os.makedirs(directory)
    except FileExistsError:
        print(f"{directory} already exists")
        return False


def is_file(filename: str) -> bool:
    return os.path.isfile(filename)


def crawl_directory(directory: str) -> list:
    if not is_dir(directory):
        raise FileNotFoundError

    subdirs = [folder[0] for folder in os.walk(directory)]
    tree = []
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            tree.append(os.path.join(subdir, _file))
    return tree


def stereo_to_mono(signal):
    """
    Input signal (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal


def analyze_mel_spectrograms(audio: str, dst=None, n_fft=1024, hop_length=256) -> np.ndarray:
    y, sr = librosa.load(audio, sr=8000)
    chunks = int(y.shape[0] / sr)

    for i in range(chunks):
        chunk = y[i * sr:(i + 1) * sr]

        S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length)
        # Convert to decibels
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(5.12, 0.32), dpi=100)  # output image size 512 x 32 pixels
        img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis="off", y_axis="off", ax=ax)

        output = os.path.join(dst, f"mel_spectrogram_{i}.png")
        fig.savefig(output, bbox_inches=None, pad_inches=0, transparent=True)
        plt.close(fig)


def extract_mel_spectrogram(
    signal: np.ndarray, sr: int = 8000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 256
) -> np.ndarray:

    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # convert to dB for log-power mel-spectrograms
    return librosa.power_to_db(S, ref=np.max)


def energy_in_db(signal: np.ndarray) -> float:
    """Return the energy of the input signal in dB.
    
    Args:
        signal (np.ndarray): The input signal.

    Returns:
        float: The energy in dB.
    """
    return 20 * np.log10(np.sum(signal**2))


def time_offset_modulation(signal: np.ndarray, time_index: int, sr: int = 8000, max_offset: float = 0.25) -> np.ndarray:
    """Given an audio segment of signal returns the signal result with a time offset of +- max_offset ms.
    
    Args:
        signal (np.ndarray): The original signal.
        time_index (int): The starting point (i.e. second) of the audio segment.
        max_offset (float): The maximum offset time difference from the original audio segment.

    Return:
        np.ndarray: The signal corresponding to offset of the original audio segment.
    """

    offset = random.choice([random.uniform(-max_offset, -0.1),
                            random.uniform(0.1, max_offset)]) if time_index else random.uniform(0.1, max_offset)
    offset_samples = int(offset * sr)
    start = time_index * sr + offset_samples

    return signal[start:start + sr]


def overlap_noise(signal: np.ndarray, noise_path: str, snr: float) -> np.ndarray:
    """Adds noise of fixed SNR to the input signal.
    
    Args:
        signal (np.ndarray): The input signal.
        noise_path (str): The path of the wav file with noise.
        snr (float): Signal to noise ratio.

    Returns:
        np.ndarray: The noisy signal.
    """

    back_noise, sr = librosa.load(noise_path, sr=8000)

    if signal.size < back_noise.size:
        # crop noise to the length of signal
        offset = np.random.randint(low=0, high=back_noise.size - signal.size)
        back_noise = back_noise[offset:offset + signal.size]
    else:
        back_noise = np.resize(back_noise, signal.size)

    # compute energies:
    energy_s = np.mean(signal**2)
    energy_n = np.mean(back_noise**2)

    # balance signal energies:
    energy_ratio = energy_s / (energy_n + 0.000000001)
    signal /= np.sqrt(energy_ratio)

    back_noise = back_noise / np.sqrt(snr)

    noisy_signal = signal + back_noise

    return noisy_signal / np.max((np.abs(noisy_signal.max()), np.abs(noisy_signal.min())))


def split_to_train_val_sets(y: np.ndarray,
                            train_ratio: float = 0.8,
                            rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray]:
    """Given a numpy array of unique elements returns two arrays splitted randomly according to train_ratio.
    
    Args:
        y (np.ndarray): A numpy array consiting of unique elements.
        train_ratio (float): The ratio to split the array
        rng: A numpy random Generator (if given)
    Returns:
        Tuple[np.ndarray]: A tuple of (arr1, arr2) where arr1 contains the train_ratio of elements of y.
    """

    rng = rng if rng else np.random.default_rng()

    train_size = int(train_ratio * y.size)
    train_indices = rng.choice(y.size, size=train_size, replace=False)

    train_set = y[train_indices]
    test_set = y[np.isin(np.arange(y.size), train_indices, assume_unique=True, invert=True)]

    return train_set, test_set


def audio_augmentation_chain(
    signal: np.ndarray, time_index: int, noise_path: str, ir_path: str, rng: np.random.Generator, sr: int = 8000
):
    """Given the original clean audio applies a series of audio-augmentation to signal[time_index*sr: (time_index+1)*sr]
    
    Args:
        signal (np.ndarray): The original clean audio.
        time_index (int): The start of the segment corresponding to 1 sec of the original signal.
        noise_path (str): The path containing the wav files corresponding to the noise examples.
        ir_path (str): The path containing the impulse responses to apply.
        rng (np.random.Generator): A np.random.Generator object (required to apply time-offset augmentation)
        sr (int): The sampling rate of the signal.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple corresponding to (Spectrogram_original, Spectrogram_augmented).
    """

    augmentation_chain = Compose(
        [
            AddBackgroundNoise(sounds_path=noise_path, min_snr_in_db=0., max_snr_in_db=10., p=1.),
            ApplyImpulseResponse(ir_path=ir_path, p=1.),
        ]
    )
    # Get the corresponding segment
    y = signal[time_index * sr:(time_index + 1) * sr]

    # Offset probability
    if rng.random() > 0.75:
        offset_signal = time_offset_modulation(signal=signal, time_index=time_index)
        augmented_signal = augmentation_chain(offset_signal, sample_rate=8000)
    else:
        augmented_signal = augmentation_chain(y, sample_rate=8000)

    # Clean signal & augmented segments
    S1 = extract_mel_spectrogram(y)
    S2 = extract_mel_spectrogram(augmented_signal)

    return S1, S2


def clear_static_dataset(train_path: str, val_path: str, train_out_path: str, val_out_path: str):
    """Move all corrupted npy files from train/val path to train/val out_paths
	
	Args:
	    train_path (str): The path containing the npy files corresponding to the training set.
	    val_path (str): The path containing the npy files corresponding to the val set.
	    train_out_path (str): Path to move the corrupted npy files from train set.
	    val_out_path (str): Path to move the corrupted npy files from val set.
	"""
    train_list = os.listdir(train_path)
    val_list = os.listdir(val_path)
    corrupted = 0

    for file in train_list:
        file = os.path.join(train_path, file)
        try:
            np.load(file, allow_pickle=True)
        except Exception as err:
            shutil.move(file, train_out_path)
            corrupted += 1
    print(f"Training corrupted samples: {corrupted}|{len(train_list)}")

    corrupted = 0
    for file in val_list:
        file = os.path.join(val_path, file)
        try:
            np.load(file, allow_pickle=True)
        except Exception as err:
            shutil.move(file, val_out_path)
            corrupted += 1
    print(f"Val corrupted samples: {corrupted}|{len(val_list)}")


def find_songs_intervals(csv_file):
    d = {}
    start = 0
    for file in csv_file:
        y, sr = librosa.load(file, sr=8000)
        dur = y.size / sr
        d[os.path.basename(file)] = {'start': start, 'end': start + dur}
        start += dur
    return d


def prepare_database(database: str, create_index=True):
    xb = []
    idxs = []
    data = os.listdir(database)
    for arr in data:
        features = np.load(os.path.join(database, arr))
        for f in features:
            if create_index:
                xb.append(f)
            idxs.append(arr.removesuffix('.npy'))
    if create_index:
        return np.stack(xb), np.array(idxs)
    else:
        return np.array(idxs)


def create_augmented_sample(src: str, ir: str, bnoise: str, dst: str = None):
    """Apply a series of audio augmentations on each 1 sec segment of audio's input and saves the result
    
    Args:
        src (str): The wav file's source path
        dst (str): The destination path to save the output wav files
        ir (str): The path to the impulse responses
        bnoise (str): The path to the background noises
    """
    y, sr = librosa.load(src, sr=8000)
    song_dur = int(y.size / sr)
    augmented = []
    if not dst:
        dst = os.getcwd()

    augmentation_chain = Compose(
        [
            AddBackgroundNoise(sounds_path=bnoise, min_snr_in_db=0., max_snr_in_db=10., p=1.),
            ApplyImpulseResponse(ir_path=ir, p=1.),
            TimeMask(min_band_part=0.06, max_band_part=0.08, fade=True, p=0.1)
        ]
    )
    dst = os.path.join(dst, 'Augmented_' + os.path.basename(src))
    for i in range(song_dur):
        z = y[i * sr:(i + 1) * sr]
        augmented.append(augmentation_chain(z, sample_rate=sr))

    augmented = np.concatenate(augmented)
    sf.write(file=dst, data=augmented, samplerate=sr, subtype='FLOAT')

def query_sequence_search(D, I):
    compensations = []
    for i, idx in enumerate(I):
        compensations.append([(x - i) for x in idx])
    candidates = np.unique(compensations)
    scores = []
    D_flat = D.flatten()
    I_flat = I.flatten()
    for c in candidates:
        idxs = np.where((c <= I_flat) & (I_flat <= c + len(D)))[0]
        scores.append(np.sum(D_flat[idxs]))
    return candidates[np.argmax(scores)], round(max(scores), 4)

def cutout_spec_augment_mask(rng: np.random.Generator = None):

    H, W = 256, 32
    H_max, W_max = H // 2, int(0.9 * W)
    mask = np.ones((1, H, W), dtype=np.float32)

    rng = rng if rng else np.random.default_rng()
    H_start, dH = rng.integers(low=0, high=H_max, size=2)
    W_start = rng.integers(low=0, high=W_max, size=1).item()
    dW = rng.integers(low=0, high=int(0.1 * W), size=1).item()
    
    mask[:, H_start:H_start + dH, W_start:W_start + dW] = 0

    return mask

def search_index(idx: int, sorted_arr: np.ndarray):
    candidate_indices = np.where(sorted_arr <= idx)[0]
    return sorted_arr[candidate_indices].max()

def majority_vote_search(d: Dict, I: np.ndarray, sorted_array: np.ndarray):
    preds = []
    I_flat = I.flatten()
    preds = [d[str(search_index(idx, sorted_array))] for idx in I_flat]
    c = Counter(preds)
    return c.most_common()[0][0]

def get_winner(d: Dict, I: np.ndarray, D: np.ndarray, sorted_array: np.ndarray):
    preds = []
    I_flat = I.flatten()
    D_flat = D.flatten()
    preds = np.array([d[str(search_index(idx, sorted_array))] for idx in I_flat])
    c = Counter(preds)
    winner = c.most_common()[0][0]
    idxs = np.where(preds == winner)[0]
    # num_matches = c.most_common()[0][1]
    
    D_shape = D.shape[0] * D.shape[1]

    return winner, (1 / D_shape) * D_flat[idxs].sum()

def exists_in_db(pos_des, neg_des, score):
    log_pos_prob = pos_des.score_samples(np.array(score).reshape(1, 1))[0]
    log_neg_prob = neg_des.score_samples(np.array(score).reshape(1, 1))[0]
    
    return True if log_pos_prob >= log_neg_prob else False