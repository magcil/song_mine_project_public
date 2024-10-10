import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import faiss
import numpy as np
import librosa
from torch.utils.data import DataLoader

from config import SR, SEGMENT_SIZE, HOP_SIZE
from models.neural_fingerprinter import Neural_Fingerprinter
from utils.utils import extract_mel_spectrogram, get_winner, search_index, crawl_directory
from tqdm import tqdm
from generation.generate_deep_audio_fingerprints import FileDataset


def parse_args():
    parser = argparse.ArgumentParser(prog='InsertSongs',
                                     description='Insert songs to a faiss index and check for duplicates.')
    parser.add_argument('-i',
                        '--input_dirs',
                        nargs='+',
                        required=True,
                        help='Directories containing the wav or npy files to insert.')
    parser.add_argument('-pt', '--pt_file', required=True, help='The .pt file of the model.')
    parser.add_argument('-fs', '--faiss', required=True, help='The faiss index.')
    parser.add_argument('-js', '--json_file', required=True, help='The json file.')
    parser.add_argument('-d', '--duration', default=10, type=int, help='Query duration for duplicates.')
    parser.add_argument('-th', '--threshold', required=True, type=float, help='Threshold for duplicates.')
    parser.add_argument('-bs', '--batch_size', type=int, required=True, default=32, help='The batch size.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    input_dirs = args.input_dirs
    dur = args.duration
    threshold = args.threshold
    batch_size = args.batch_size

    index = faiss.read_index(args.faiss)
    with open(args.json_file, 'r') as f:
        json_correspondence = json.load(f)

    sorted_array = np.sort([int(k) for k in json_correspondence.keys()])
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Neural_Fingerprinter().to(device)
    model.load_state_dict(torch.load(args.pt_file, map_location=torch.device(device=device)))

    all_files = []
    for dir in input_dirs:
        all_files += crawl_directory(dir)
    next_index = index.ntotal
    start_total, inserted, duplicates, fails = index.ntotal, 0, 0, 0
    with torch.no_grad():
        for file in tqdm(all_files, desc='Processing files', total=len(all_files)):
            y, sr = librosa.load(file, sr=SR)
            file_duration = y.size // sr

            # Check duration
            if (file_duration // 2) + dur > file_duration:
                print(f"{os.path.basename(file)} not enough duration. Skipping...")
                fails += 1
                continue
            # Check for duplicate
            y_seg = y[(file_duration // 2) * sr:((file_duration // 2) + dur) * sr]
            J = int(np.floor((y_seg.size - sr) / HOP_SIZE)) + 1
            xq = np.stack(
                [extract_mel_spectrogram(y_seg[j * HOP_SIZE:j * HOP_SIZE + SR]).reshape(1, 256, 32) for j in range(J)])
            out = model(torch.from_numpy(xq).to(device))
            D, I = index.search(out.cpu().numpy(), 1)
            winner, score = get_winner(d=json_correspondence, I=I, D=D, sorted_array=sorted_array)

            if score >= threshold:
                print(f"{os.path.basename(file)} already in database. Skipping...")
                duplicates += 1
                continue
            # If not in database, insert
            file_dset = FileDataset(file=file, sr=SR, hop_size=HOP_SIZE)
            file_dloader = DataLoader(file_dset, batch_size=batch_size, shuffle=False)
            fingerprints = []
            for X in file_dloader:
                X = model(X.to(device))
                fingerprints.append(X.cpu().numpy())
            fingerprints = np.vstack(fingerprints)
            index.add(fingerprints)
            json_correspondence[next_index] = os.path.basename(file).removesuffix('.wav')
            sorted_array = np.append(sorted_array, next_index)
            next_index += fingerprints.shape[0]
            inserted += 1
    print(f"Initial songs: {start_total} | Total: {index.ntotal} | Inserted: {inserted} | Duplicates: {duplicates}",
          sep="")
    print(f" | Fails: {fails}")

    fs_path = args.faiss.removesuffix(".index") + "_updated.index"
    js_path = args.json_file.removesuffix(".json") + "_updated.json"
    faiss.write_index(fs_path)
    with open(js_path, "w") as f:
        json.dump(f)
