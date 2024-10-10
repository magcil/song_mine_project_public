import argparse
import os
import time
from collections import Counter
import datetime
import time
import json
import sys
import pickle

import torch
import pyaudio
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import extract_mel_spectrogram, query_sequence_search, search_index, get_winner, exists_in_db
from models.neural_fingerprinter import Neural_Fingerprinter

F = 8000
FMB = 4000
H = 4000
NOT_IN_DB = 'Not in Database'


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--index', required=True, help='The faiss index to load.')
    parser.add_argument(
        '-j',
        '--json_file',
        required=True,
        help='The json file containing the correspondence between filenames and indices.'
    )
    parser.add_argument('-m', '--model', required=True, help='The path to the model.')
    parser.add_argument('-d', '--duration', type=int, default=10, help='The recording duration in seconds.')
    parser.add_argument(
        '-s',
        '--search',
        choices=['sequence_search', 'majority_vote'],
        default='majority_vote',
        help='The search type to use for predictions.'
    )
    parser.add_argument('-pb', '--probes', type=int, default=5, help='The number of probes.')
    parser.add_argument('-kn', '--neighbors', type=int, default=4, help='Number of nearest neighbors to search.')
    parser.add_argument('-td', '--threshold_densities', help='The pkl file of threshold densities.')
    parser.add_argument('-t', '--threshold', type=float, help='The threshold to use to check if songs exists in db.')
    parser.add_argument('-dv', '--device', default='cpu', help='Whether to run on cuda or cpu.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    # Get args
    index_path = args.index
    dur = args.duration
    probes = args.probes
    k = args.neighbors
    model_pt = args.model
    json_file = args.json_file
    device = args.device
    search = args.search
    threshold = args.threshold

    with open(json_file, 'r') as f:
        names = json.load(f)
    if args.threshold_densities:
        with open(args.threshold_densities, 'rb') as f:
            densities = pickle.load(f)
            pos_des = densities['tp_density']
            neg_des = densities['tn_density']

    sorted_array = np.sort(np.array(list(map(int, names.keys()))))
    torch.cuda.empty_cache()
    index = faiss.read_index(index_path)
    print(f'Total items on index: {index.ntotal}\nIndex train status: {index.is_trained}.')
    index.nprobe = probes

    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    print(f"Running on {device}")
    model = Neural_Fingerprinter().to(device)

    if device == 'cuda':
        model.load_state_dict(torch.load(model_pt))
    else:
        model.load_state_dict(torch.load(model_pt, map_location=torch.device('cpu')))

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=F, frames_per_buffer=FMB, input=True)

    print(f'Recording starts...')

    if device == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    try:
        model.eval()
        with torch.no_grad():
            while True:
                # Get buffer
                frames = []
                for i in range(0, int((F / FMB) * dur)):
                    data = stream.read(FMB)
                    frames.append(data)
                aggregated_buf = np.frombuffer(b"".join(frames), dtype=np.float32)

                # Inference
                if device == "cuda":
                    starter.record()
                elif device == "cpu":
                    inference_start = time.perf_counter()

                J = int(np.floor((aggregated_buf.size - F) / H)) + 1
                batch = np.stack(
                    [extract_mel_spectrogram(aggregated_buf[j * H:j * H + F]).reshape(1, 256, 32) for j in range(J)]
                )
                out = model(torch.from_numpy(batch).to(device))

                if device == "cuda":
                    ender.record()
                    torch.cuda.synchronize()
                    inference_time = starter.elapsed_time(ender)
                elif device == "cpu":
                    inference_end = time.perf_counter()
                    inference_time = (inference_end - inference_start) * 1000  # miliseconds

                # Search
                tic = time.perf_counter()
                D, I = index.search(out.cpu().numpy(), k)
                if search == 'sequence_search':
                    idx, score = query_sequence_search(D, I)
                    idx = search_index(idx, sorted_array)
                    id = names[idx]
                elif search == 'majority_vote':
                    id, score = get_winner(names, I, D, sorted_array)
                    if args.threshold_densities and not exists_in_db(pos_des, neg_des, score):
                        id = NOT_IN_DB
                    elif threshold and score <= threshold:
                        id = NOT_IN_DB
                query_time = time.perf_counter() - tic
                now = datetime.datetime.now()
                print(
                    f"{now}: {8*'*'} {id} {8*'*'}. " + f"Inference time: {inference_time:.2f} ms." +
                    f" Query Time: {1000*query_time:.3f} ms. Score: {score}"
                )

    except KeyboardInterrupt:
        print('Stopped!')
        torch.cuda.empty_cache()
        pass