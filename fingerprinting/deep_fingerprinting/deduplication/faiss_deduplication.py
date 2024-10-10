import argparse
import os
import sys
from collections import Counter
from typing import Dict
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import crawl_directory

import numpy as np
import faiss
from tqdm import tqdm
import pandas as pd

F = 8000
H = 4000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--fingerprints',
                        required=True,
                        nargs='+',
                        help='Path to fingerprints for deduplication.')
    parser.add_argument('-t',
                        '--threshold',
                        type=float,
                        default=0.9,
                        help='Threshold to be considered already in database.')
    parser.add_argument('-d', '--duration', type=int, default=5, help='Query duration.')
    parser.add_argument('-l', '--limit', type=int, default=0, help='Number to use a non exhaustive index.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Control verbosity.')

    return parser.parse_args()


class FaissDB():
    def __init__(self, index_str: str = "Flat", d: int = 128, k: int = 1):

        self.index = faiss.index_factory(d, index_str, faiss.METRIC_INNER_PRODUCT)
        self.json = {}
        self.k = k
        self.d = d
        self.sorted_arr = np.array([], dtype=np.int32)

    def update(self, fingerprints: np.ndarray, song_name: str):
        idx = self.index.ntotal
        self.index.add(fingerprints)
        self.json[idx] = song_name
        self.sorted_arr = np.append(self.sorted_arr, idx)

    def is_empty(self):
        return self.index.ntotal == 0

    def total_fingerprints(self):
        return self.index.ntotal

    def songs_in_db(self):
        return list(self.json.values())

    def total_songs(self):
        return len(self.json)

    def search(self, query: np.ndarray):

        D, I = self.index.search(query, self.k)
        winner, score = self.get_winner(D, I)

        return winner, score

    def get_winner(self, D: np.ndarray, I: np.ndarray):
        preds = []
        I_flat = I.flatten()
        D_flat = D.flatten()
        preds = np.array([self.json[self.search_index(idx)] for idx in I_flat])
        c = Counter(preds)
        winner = c.most_common()[0][0]
        idxs = np.where(preds == winner)[0]
        # num_matches = c.most_common()[0][1]

        D_shape = D.shape[0] * D.shape[1]

        return winner, (1 / D_shape) * D_flat[idxs].sum()

    def search_index(self, idx: int):
        candidate_indices = np.where(self.sorted_arr <= idx)[0]
        return self.sorted_arr[candidate_indices].max()

    def train_index(self, items):
        self.index.train(items)

    def insert_from_path(self, fingerprints):
        num_fingerprints = 0
        self.json = {}
        self.sorted_arr = np.array([], dtype=np.int32)
        for f in fingerprints:
            x = np.load(f)
            self.sorted_arr = np.append(self.sorted_arr, num_fingerprints)
            self.json[num_fingerprints] = os.path.basename(f).removesuffix('.npy') + '.wav'
            num_fingerprints += x.shape[0]
        items = np.zeros(shape=(num_fingerprints, self.d), dtype=np.float32)
        idx = 0
        for f in fingerprints:
            x = np.load(f)
            items[idx:idx + x.shape[0]] = x
            idx += x.shape[0]
        self.index.add(items)


if __name__ == '__main__':

    # Parser arguments
    args = parse_args()
    all_fingerprints = []
    for dir in args.fingerprints:
        all_fingerprints += crawl_directory(dir)

    print(f'Total fingerprints: {len(all_fingerprints)}')
    threshold = args.threshold
    verbose = args.verbose
    dur = args.duration
    num_lim = args.limit

    index = faiss.IndexFlatIP(128)
    json = {}

    faiss_db = FaissDB(index_str='Flat', d=128, k=1)
    logs = []
    deleted = 0
    progress_bar = tqdm(all_fingerprints, postfix={"Deleted": f"{deleted}"})

    for j, f in enumerate(progress_bar):
        fingerprints = np.load(f)
        wav_file = os.path.basename(f.removesuffix('.npy') + '.wav')

        if faiss_db.is_empty():
            faiss_db.update(fingerprints, wav_file)
        else:
            total_fingerprints = fingerprints.shape[0]
            total_dur = int(total_fingerprints * (H / F)) + 1
            if total_dur // 2 + dur > total_dur:
                if verbose:
                    print(f'Song: {wav_file} has not enough duration. Skipping...')
                    continue
            else:
                start_idx = int((F / H) * (total_dur // 2))
                queries = fingerprints[start_idx:start_idx + 2 * dur - 1]
                winner, score = faiss_db.search(queries)

                if score > threshold:
                    out_str = f'Removed: {wav_file} | Match: {winner} | Score: {score:.3f}'
                    if verbose:
                        print(out_str)
                    logs.append(out_str)
                    deleted += 1
                else:
                    faiss_db.update(fingerprints, wav_file)

        # Continue with a non exhaustive index
        if (j + 1) == num_lim:

            all_songs = set(faiss_db.songs_in_db())
            print(f'Creating a non exhaustive index. Total songs in db: {faiss_db.total_songs()}')

            faiss_db = FaissDB(index_str="IVF50,Flat", d=128, k=1)
            new_fingerprints = []
            for f in all_fingerprints:
                if os.path.basename(f).removesuffix('.npy') + '.wav' in all_songs:
                    new_fingerprints.append(f)

            print(f'Training index...')
            faiss_db.train_index(np.vstack([np.load(f) for f in new_fingerprints]))
            print(f'Ok!')
            print('Inserting fingerprints...')
            faiss_db.insert_from_path(new_fingerprints)
            print('Ok!')

        progress_bar.set_postfix({"Deleted": f"{deleted}"})

    with open("removed_songs.txt", "w") as f:
        logs = [l + "\n" for l in logs]
        f.writelines(logs)

    all_songs = faiss_db.songs_in_db()
    with open("songs_in_db.txt", "w") as f:
        all_songs = [s + '\n' for s in all_songs]
        f.writelines(all_songs)
