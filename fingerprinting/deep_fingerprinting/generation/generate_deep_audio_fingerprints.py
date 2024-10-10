import json
import os
import argparse

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import librosa
from tqdm import tqdm
import torch

from utils.utils import crawl_directory, extract_mel_spectrogram
from models.neural_fingerprinter import Neural_Fingerprinter
from config import HOP_SIZE, SEGMENT_SIZE, SR


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Generate deep audio fingerprints',
        description='Using a pretrained model extract the latent representations for each audio segment',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_dirs', nargs='+', required=True, help='The directories containing the wav files.')
    parser.add_argument('-o', '--output_path', required=True, help="The folder's output path to store the fingerprints")
    parser.add_argument('-m', '--model', required=True, help='The model to be used for feature extraction.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='The batch size to be used for inference.')

    return parser.parse_args()


class FileDataset(Dataset):
    def __init__(self, file, sr, hop_size):
        self.y, self.F = librosa.load(file, sr=sr)
        self.H = hop_size
        self.dur = self.y.size // self.F

        # Extract spectrograms
        self._get_spectrograms()

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return torch.from_numpy(self.spectrograms[idx])

    def _get_spectrograms(self):
        self.spectrograms = []
        J = int(np.floor((self.y.size - self.F) / self.H)) + 1
        for j in range(J):
            S = extract_mel_spectrogram(signal=self.y[j * self.H:j * self.H + self.F])
            self.spectrograms.append(S.reshape(1, *S.shape))


if __name__ == '__main__':

    # parse args
    args = parse_args()

    input_dirs = args.input_dirs
    output_dir = args.output_path
    batch_size = args.batch_size
    pt_file = args.model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Neural_Fingerprinter().to(device)
    model.load_state_dict(torch.load(pt_file))
    print(f'Running on {device}')

    # Check if dir exists
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"dir {output_dir} does not exist, please create it and rerun")

    all_songs = []
    for dir in input_dirs:
        all_songs += crawl_directory(dir)
    print(f'All songs: {len(all_songs)}')

    # Discard already fingerprinted songs
    to_discard = [os.path.basename(song).removesuffix('.npy') + '.wav' for song in crawl_directory(output_dir)]
    all_songs = [song for song in all_songs if os.path.basename(song) not in to_discard]
    print(f'Songs to fingerprint: {len(all_songs)} | Discarded: {len(to_discard)}')

    model.eval()
    fails = 0
    totals = len(all_songs)
    p_bar = tqdm(all_songs, desc='Extracting deep audio fingerprints', total=totals)
    with torch.no_grad():
        for file in p_bar:
            file_dset = FileDataset(file=file, sr=SR, hop_size=HOP_SIZE)
            if file_dset.dur < 1:
                print(f'Song: {os.path.basename(file)} has duration less than 1 sec. Skipping...')
                fails += 1
                continue
            file_dloader = DataLoader(file_dset, batch_size=batch_size, shuffle=False)
            fingerprints = []

            for X in file_dloader:
                X = model(X.to(device))
                fingerprints.append(X.cpu().numpy())
            try:
                fingerprints = np.vstack(fingerprints)
                np.save(file=os.path.join(output_dir,
                                          os.path.basename(file).removesuffix('.wav') + '.npy'),
                        arr=fingerprints)
            except Exception as e:
                print(f'Failed to save {os.path.basename(file)} | Error: {e}')
                fails += 1
                continue

    print(f'Totals: {totals}\nFails: {fails}')
