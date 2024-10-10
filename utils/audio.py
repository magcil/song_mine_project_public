import numpy as np
from pyAudioAnalysis import audioBasicIO


def read_audio_file(filename: str) -> tuple[int, np.ndarray]:
    rate, signal = audioBasicIO.read_audio_file(filename)
    signal = audioBasicIO.stereo_to_mono(signal)  # ws edw einai int16!
    signal = np.double(signal)
    signal = signal / (2.0**15)  # gia na paei sto -1 1 range
    return rate, signal


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    return signal / signal.max()
