import json
import logging
import os
import pathlib
import sys
from collections import Counter
from hashlib import sha1
from typing import Dict

import faiss
import numpy as np

SR = 8000
SEGMENT_SIZE = SR
HOP_SIZE = SR // 2  # 0.5 sec
D = 128


class FaissMind:
    def __init__(self, index_path=None, songs_path=None, k: int = 1, nprobe: int = 5):
        self.index_path = index_path
        self.songs_path = songs_path
        self.k = k
        self.nprobe = nprobe
        self.read_faiss_index()
        self.read_songs()

    def read_faiss_index(self):
        if self.index_path is None:
            raise FileNotFoundError
        self.faiss_index = faiss.read_index(self.index_path)
        self.faiss_index.nprobe = self.nprobe

    def read_songs(self):
        if self.songs_path is None:
            raise FileNotFoundError
        self.songs = json.load(open(self.songs_path))
        self.sorted_array = np.array(sorted(map(int, self.songs.keys())))

    def faiss_search(self, query):
        D, I = self.faiss_index.search(query, self.k)
        return D, I

    def faiss_stats(self):
        return self.faiss_index.ntotal, self.faiss_index.d, self.faiss_index.is_trained

    def get_total_songs(self):
        return len(self.songs)

    def get_total_fingerprints(self):
        return self.faiss_index.ntotal

    def add_song(self):
        pass

    def get_all_tracks(self, skip=0, limit=100):
        total_tracks = list(self.songs.values())

        return total_tracks[skip : limit + skip], len(total_tracks)

    def get_query_results(self, query):
        D, I = self.faiss_search(query)
        return self.get_winner(self.songs, D, I)

    def get_song_by_id(self, song_id):
        for key, value in self.songs.items():
            if value == song_id:
                return key

    def terminate(self):
        self.connection.close()
        print("Connection closed")

    def __exit__(self):
        self.terminate(self)

    def search_index(self, idx: int):
        candidate_indices = np.where(self.sorted_array <= idx)[0]
        return self.sorted_array[candidate_indices].max()

    def get_winner(self, d: Dict, D: np.ndarray, I: np.ndarray):
        preds = []
        I_flat = I.flatten()
        D_flat = D.flatten()
        preds = np.array([d[str(self.search_index(idx))] for idx in I_flat])
        c = Counter(preds)
        winner, counts = c.most_common()[0]
        idxs = np.where(preds == winner)[0]
        D_shape = D.shape[0] * D.shape[1]
        score = (1 / D_shape) * D_flat[idxs].sum()

        return winner, score

    def query_sequence_search(self, D: np.ndarray, I: np.ndarray):
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

    def predict_query_offset(self, query: np.ndarray):
        D, I = self.faiss_search(query)

        idx, score = self.query_sequence_search(D, I)
        start_idx = self.search_index(idx=idx)
        winner = self.songs[str(start_idx)]

        # Offset (sec) from start
        offset = (idx - start_idx) * HOP_SIZE / SR

        return winner, score, offset

    @staticmethod
    def hash_file(file_path: str, block_size: int = 2**20) -> str:
        s = sha1()
        with open(file_path, "rb") as f:
            while True:
                buf = f.read(block_size)
                if not buf:
                    break
                s.update(buf)
        return s.hexdigest().upper()

    @staticmethod
    def hash_data(data: bytes, block_size: int = 2**20) -> str:
        s = sha1()

        s.update(data)
        return s.hexdigest().upper()

    @staticmethod
    def prepare_fingerprints(fingerprints):
        xb = []
        for f in fingerprints:
            xb.append(f)
        return xb
