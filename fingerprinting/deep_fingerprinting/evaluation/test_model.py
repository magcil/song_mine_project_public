import argparse
import os
from collections import Counter
import json
import sys
import pickle

import faiss
import torch
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from models.neural_fingerprinter import Neural_Fingerprinter

HOP_LENGTH = 4000
F = 8000
NOT_IN_DB = 'Not In Database'


def parse_args():

    parser = argparse.ArgumentParser(
        description='Test a Deep Audio Fingerprinter on recorded data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True, help='The path to input recordings to be predicted.')
    parser.add_argument('-wf', '--wav_files', required=True, help='The path to the ground truth wav files.')
    parser.add_argument('-d', '--duration', type=int, default=10, help='The segment size of each query (sec).')
    parser.add_argument(
        '-l',
        '--list_of_test_files',
        required=True,
        help='The path to the csv file (or txt) containing the ground truth labels.'
    )
    parser.add_argument('-m', '--model', required=True, help='The path containing the model.')
    parser.add_argument('-id', '--index', help='A faiss index for the database.')
    parser.add_argument('-js', '--json_file', required=True, help='The json file with the correspondence.')
    parser.add_argument(
        '-s',
        '--search',
        choices=['sequence_search', 'majority_vote'],
        default='majority_vote',
        help='The search method to use.'
    )
    parser.add_argument('-k', '--k_neighbors', type=int, help='The number of nearest neighbors to retrieve.', default=4)
    parser.add_argument('-pb', '--probes', default=5, type=int, help='The number of probes.')
    parser.add_argument('-td', '--threshold_densities', help='The pk file with threshold densities.')
    parser.add_argument('-dv', '--device', default='cpu', help='The device to be used for inference (cpu or cuda).')
    return parser.parse_args()


def print_results(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
    print(
        f"\nAccuracy score: {acc*100:.2f}%\nPrecision score: {precision*100:.2f}%" +
        f"\nRecall score: {recall*100:.2f}%\n"
    )


if __name__ == '__main__':

    args = parse_args()
    input_wavs = args.input
    dur = args.duration
    csv_file = args.list_of_test_files
    model_pt = args.model
    wav_files_pt = args.wav_files
    k = args.k_neighbors
    probes = args.probes
    json_file = args.json_file
    search = args.search
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f'Running on ', device)

    with open(csv_file, 'r') as f:
        test_songs = [os.path.join(wav_files_pt, fname).strip() for fname in f.readlines()]

    with open(json_file, 'r') as f:
        correspondence = json.load(f)

    if args.threshold_densities:
        with open(args.threshold_densities, 'rb') as f:
            densities = pickle.load(f)
            pos_des = densities['tp_density']
            neg_des = densities['tn_density']
            threshold_preds = []
    sorted_array = np.sort(np.array(list(map(int, correspondence.keys()))))

    index = faiss.read_index(args.index)
    index.nprobe = probes

    songs_intervals = utils.find_songs_intervals(test_songs)
    model = Neural_Fingerprinter().to(device)
    if device == 'cuda':
        model.load_state_dict(torch.load(model_pt))
    else:
        model.load_state_dict(torch.load(model_pt, map_location=torch.device('cpu')))

    # Warming up GPU, if on GPU
    if device == 'cuda':
        print('Warming up GPU...', end='')
        x = torch.randn((1, 1, 256, 32), device=device)
        for _ in range(10):
            _ = model(x)
        print('Ok!')

    recording, sr = librosa.load(input_wavs, sr=8000)
    preds, ground_truth, scores = [], [], []
    query_times, inference_times = [], []
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        pass
    progress_bar = tqdm(test_songs, desc='Processing songs.')

    model.eval()
    with torch.no_grad():
        for song in progress_bar:

            label = os.path.basename(song)
            start, end = songs_intervals[label]['start'], songs_intervals[label]['end']
            iters = int((end - start) / dur)
            q, r = divmod(start * sr, sr)
            start = int(q * sr) + int(r)

            for iter in range(iters):

                y_slice = recording[start + iter * dur * F:start + (iter + 1) * dur * F]

                # Inference
                J = int(np.floor((y_slice.size - F) / HOP_LENGTH)) + 1
                if device == 'cuda':
                    starter.record()
                else:
                    tic = time.perf_counter()
                batch = np.stack(
                    [
                        utils.extract_mel_spectrogram(y_slice[j * HOP_LENGTH:j * HOP_LENGTH + F]).reshape(1, 256, 32)
                        for j in range(J)
                    ]
                )
                out = model(torch.from_numpy(batch).to(device))
                if device == 'cuda':
                    ender.record()
                    torch.cuda.synchronize()
                    inference_times.append(starter.elapsed_time(ender))
                else:
                    inference_times.append(1000 * (time.perf_counter() - tic))

                # Search
                out = out.cpu().numpy()
                tic = time.perf_counter()
                D, I = index.search(out, k)

                if search == 'sequence_search':
                    indx, score = utils.query_sequence_search(D, I)
                    indx = utils.search_index(indx, sorted_array)
                    preds.append(correspondence[str(indx)])
                    scores.append(score)
                elif search == 'majority_vote':
                    pred, score = utils.get_winner(correspondence, I, D, sorted_array)
                    # Maximum Likelihood
                    if args.threshold_densities:
                        if utils.exists_in_db(pos_des, neg_des, score):
                            threshold_preds.append(pred)
                        else:
                            threshold_preds.append(NOT_IN_DB)

                    preds.append(pred)
                    scores.append(score)
                query_times.append(time.perf_counter() - tic)
                ground_truth.append(label.removesuffix('.wav'))

    print(f'Results (No threshold)')
    print_results(ground_truth, preds)
    if args.threshold_densities:
        print(f'Results with threshold')
        print_results(ground_truth, threshold_preds)
    print(f'Mean inference time (on {device}): {sum(inference_times) / len(inference_times):.2f} ms.')
    print(f'Mean query time: {1000*sum(query_times) / len(query_times):.2f} ms.')
    print(f'Total Avg. time per query: {(1000*sum(query_times) + sum(inference_times)) / len(query_times):.2f} ms.')

    if args.threshold_densities:
        df = pd.DataFrame(
            {
                'Ground Truth': ground_truth,
                'Predictions (No threshold)': preds,
                'Predictions (Threshold)': threshold_preds,
                'Scores': scores
            }
        )
    else:
        df = pd.DataFrame({'Ground Truth': ground_truth, 'Predictions': preds, 'Scores': scores})
    df.to_csv(f"results_query_{dur}_secs.csv")