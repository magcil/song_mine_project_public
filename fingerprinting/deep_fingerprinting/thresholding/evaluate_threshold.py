import argparse
import pickle
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from models.neural_fingerprinter import Neural_Fingerprinter

import torch
import faiss
import numpy as np
from tqdm import tqdm
import librosa
from sklearn.metrics import accuracy_score, precision_score, recall_score
from audiomentations import AddBackgroundNoise

MIN_SNR_DB = 0
MAX_SNR_DB = 6


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--index', required=True, help='The faiss index.')
    parser.add_argument('-j', '--json', required=True, help='Json with filenames.')
    parser.add_argument('-d', '--duration', type=int, default=10, help='Query duration (sec).')
    parser.add_argument('-ps', '--positive_samples', required=True, help='Path to positive samples.')
    parser.add_argument('-ns', '--negative_samples', required=True, help='Path to negative samples.')
    parser.add_argument('-m', '--model', required=True, help='Path to model weights.')
    parser.add_argument('-th', '--threshold_densities', required=True, help='The pkl file with densities.')
    parser.add_argument('-bs', '--background_noise', required=True, help='Path to background noises.')
    parser.add_argument('-nps', '--number_pos_samples', type=int, help='The number of positive samples to use.')
    parser.add_argument('-nns', '--number_negative_samples', type=int, help='The number of negative samples to use.')

    return parser.parse_args()


def evaluate_threshold_on_samples(
    samples, dur, index, model, labels, sorted_array, pos_des, neg_des, b_noises, category
):

    y_true, y_pred = [], []
    true_song, pred_song = [], []
    distances = []
    total_trues, total_preds = [], []
    p_bar = tqdm(samples, desc=category)
    b_noise = AddBackgroundNoise(b_noises, min_snr_in_db=MIN_SNR_DB, max_snr_in_db=MAX_SNR_DB, p=1.)

    model.eval()
    with torch.no_grad():
        for sample in p_bar:
            y, sr = librosa.load(sample, sr=8000)
            total_dur = y.size // sr
            segments = total_dur // dur
            step = 2 * dur - 1
            idxs = np.arange(0, segments * dur, 0.5)
            for seg in range(segments):
                batch = []
                for i in idxs[seg * step:(seg + 1) * step]:
                    s = utils.extract_mel_spectrogram(b_noise(y[int(i * sr):int((i + 1) * sr)], sample_rate=sr))
                    s = s.reshape(1, *s.shape)
                    batch.append(s)
                batch = np.stack(batch)
                out = model(torch.from_numpy(batch).to(device))
                D, I = index.search(out.cpu().numpy(), 4)
                winner, distance = utils.get_winner(labels, I, D, sorted_array)

                distances.append(distance)
                if utils.exists_in_db(pos_des, neg_des, distance):
                    y_pred.append(1)
                    true_song.append(os.path.basename(sample).removesuffix('.wav'))
                    pred_song.append(winner)
                    total_preds.append(winner)
                else:
                    y_pred.append(0)
                    total_preds.append('not_in_db')
                if category == 'positive':
                    y_true.append(1)
                    total_trues.append(winner)
                elif category == 'negative':
                    y_true.append(0)
                    total_trues.append('not_in_db')

    return y_true, y_pred, true_song, pred_song, distances, total_trues, total_preds


if __name__ == '__main__':

    args = parse_args()
    dur = args.duration

    with open(args.threshold_densities, 'rb') as f:
        d = pickle.load(f)
        pos_des = d['tp_density']
        neg_des = d['tn_density']

    with open(args.json, 'r') as f:
        labels = json.load(f)

    sorted_array = np.array(sorted(map(int, labels.keys())))
    index = faiss.read_index(args.index)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Neural_Fingerprinter().to(device)
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))

    if args.number_pos_samples:
        positive_samples = np.random.choice(
            utils.crawl_directory(args.positive_samples), size=args.number_pos_samples, replace=False
        )
    else:
        positive_samples = utils.crawl_directory(args.positive_samples)

    if args.number_negative_samples:
        negative_samples = np.random.choice(
            utils.crawl_directory(args.negative_samples), size=args.number_negative_samples, replace=False
        )
    else:
        negative_samples = utils.crawl_directory(args.negative_samples)

    y_true, y_pred = [], []
    true_song, pred_song = [], []
    total_trues, total_preds = [], []

    y_true_pos, y_pred_pos, true_song_pos, pred_song_pos, pos_distances,\
          pos_total_trues, pos_total_preds = evaluate_threshold_on_samples(
        samples=positive_samples,
        dur=dur,
        index=index,
        labels=labels,
        sorted_array=sorted_array,
        category='positive',
        model=model,
        pos_des=pos_des,
        neg_des=neg_des,
        b_noises=args.background_noise
    )

    y_true += y_true_pos
    y_pred += y_pred_pos
    true_song += true_song_pos
    pred_song += pred_song_pos
    total_trues += pos_total_trues
    total_preds += pos_total_preds

    y_true_neg, y_pred_neg, true_song_neg, pred_song_neg, neg_distances,\
         neg_total_trues, neg_total_preds = evaluate_threshold_on_samples(
        samples=negative_samples,
        dur=dur,
        index=index,
        labels=labels,
        sorted_array=sorted_array,
        model=model,
        category='negative',
        pos_des=pos_des,
        neg_des=neg_des,
        b_noises=args.background_noise
    )

    y_true += y_true_neg
    y_pred += y_pred_neg
    true_song += true_song_neg
    pred_song += pred_song_neg
    total_trues += neg_total_trues
    total_preds += neg_total_preds

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
    print(
        f"Threshold Evaluation:" + f"\nAccuracy score: {acc*100:.2f}%\nPrecision score: {precision*100:.2f}%" +
        f"\nRecall score: {recall*100:.2f}%"
    )

    acc = accuracy_score(true_song, pred_song)
    precision = precision_score(true_song, pred_song, average='macro', labels=list(set(true_song)), zero_division=0)
    recall = recall_score(true_song, pred_song, average='macro', labels=list(set(true_song)), zero_division=0)
    print(
        f"\n\nModel Evaluation:" + f"\nAccuracy score: {acc*100:.2f}%\nPrecision score: {precision*100:.2f}%" +
        f"\nRecall score: {recall*100:.2f}%"
    )

    acc = accuracy_score(total_trues, total_preds)
    precision = precision_score(
        total_trues, total_preds, average='macro', labels=list(set(total_trues)), zero_division=0
    )
    recall = recall_score(total_trues, total_preds, average='macro', labels=list(set(total_trues)), zero_division=0)
    print(
        f"\n\nOverall System Evaluation:" + f"\nAccuracy score: {acc*100:.2f}%\nPrecision score: {precision*100:.2f}%" +
        f"\nRecall score: {recall*100:.2f}%"
    )

    print(f'\nStatistics for positives samples\nMean value: {np.mean(pos_distances)}\nStd: {np.std(pos_distances)}\n')
    print(f'\nStatistics for negative samples\nMean value: {np.mean(neg_distances)}\nStd: {np.std(neg_distances)}')