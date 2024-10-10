import os
import sys
from typing import List, Dict

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(os.path.dirname(current_file_path))

sys.path.insert(0, parent_dir_path)
from utils.utils import energy_in_db, audio_augmentation_chain, extract_mel_spectrogram

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from numpy.random import default_rng
from tqdm import tqdm

SEED = 42


class StaticAudioDataset(Dataset):
    """Create Static Dataset"""

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = os.listdir(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.data[idx])
        x_org, x_aug = np.load(img_path, allow_pickle=True)
        return torch.unsqueeze(torch.tensor(x_org, dtype=torch.float32),
                               0), torch.unsqueeze(torch.tensor(x_aug, dtype=torch.float32), 0)


class DynamicAudioDataset(Dataset):
    """Create Dynamic Dataset"""

    def __init__(self, data_path, noise_path, ir_path):
        self.data_path = data_path
        self.noise_path = noise_path
        self.ir_path = ir_path
        self.data = set(os.listdir(data_path))
        self.rng = default_rng(SEED)
        self.time_indices_dict = {}
        self.get_energy_index()

    def get_energy_index(self):
        '''
        Keeps only segments where energy > 0.
        Returns a dictionary where the keys are the paths to the audio files 
        and the values are a random time index for each audio file.
        '''
        to_keep = []
        for wav in self.data:
            indices = []
            full_wav_path = os.path.abspath(os.path.join(self.data_path, wav))

            try:
                signal, sr = librosa.load(full_wav_path, sr=8000)
            except Exception as err:
                log_info = f"Error occured on: {os.path.basename(full_wav_path)}."
                print(log_info)
                print(f"Exception: {err}")
                print(f'Removed filename: {wav}')
            else:
                max_time_index = int(signal.size / sr) - 1
                if max_time_index:
                    for time_index in range(0, max_time_index):
                        energy = energy_in_db(signal[time_index * sr:(time_index + 1) * sr])
                        if energy > 0:
                            indices.append(time_index)
                        else:
                            continue

                    if len(indices) > 0:
                        # keep all the random indices for each song (time_indices_dict)
                        self.time_indices_dict[full_wav_path] = indices
                        to_keep.append(wav)
                    else:
                        print(f'File {wav} has no segments that have higher energy than zero')
                        print(f'Removed filename: {wav}')
                else:
                    print(f'File: {wav} has duration less than 1 sec. Skipping...')

        self.data = [os.path.abspath(os.path.join(self.data_path, wav)) for wav in to_keep]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        song_path = self.data[idx]
        time_index = self.rng.choice(self.time_indices_dict[song_path])
        signal, sr = librosa.load(song_path, sr=8000)

        x_org, x_aug = audio_augmentation_chain(signal, time_index, self.noise_path, self.ir_path, self.rng)

        return torch.from_numpy(x_org).expand(1, *x_org.shape), torch.from_numpy(x_aug).expand(1, *x_aug.shape)


class GtzanDataset(Dataset):

    def __init__(self, data: List[str], class_mapping: Dict[str, int]):
        self.data = data
        self.class_mapping = class_mapping
        self.to_keep = []
        self.create_dataset()

    def create_dataset(self):
        for song in tqdm(self.data, desc='Preparing training dataset.'):
            try:
                y, sr = librosa.load(song, sr=8000)
            except Exception as e:
                print(f'Cannot open :{os.path.basename(song)}. Skipping...')
            else:
                song_dur = int(y.size // sr)
                self.to_keep += [(song, i) for i in range(song_dur)]

    def __len__(self):
        return len(self.to_keep)

    def __getitem__(self, idx):
        song, i = self.to_keep[idx]
        y, sr = librosa.load(song, sr=8000)
        S = extract_mel_spectrogram(y[i * sr:(i + 1) * sr])
        genre = os.path.basename(os.path.dirname(song))

        return torch.from_numpy(S.reshape(1, *S.shape)), torch.tensor(self.class_mapping[genre], dtype=torch.int64)
