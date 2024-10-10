import os
import argparse
import json
from audiomentations import (
    AddBackgroundNoise,
    ApplyImpulseResponse,
    Compose,
    TimeMask,
)
import librosa
import numpy as np
import soundfile as sf
from collections import Counter
from sklearn.neighbors import KernelDensity

from tqdm import tqdm
import torch
import faiss
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from models.neural_fingerprinter import Neural_Fingerprinter

STEP = 0.5

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ps", "--positive_samples", type=str, required=True, help="The path folder of the .wav that exist in db."
    )
    parser.add_argument(
        "-ns", "--negative_samples", type=str, required=True, help="The path folder of the .wav that do not exist in db."
    )
    parser.add_argument(
        "-j", "--json", type=str, required=True, help="The json file with the wav names and their corresponding index."
    )
    parser.add_argument(
        "-id", "--index_file", type=str, required=True, help="The path to the file with the faiss indices."
    )
    parser.add_argument(
        "-bn", "--background_noise", type=str, required=True, help="The path to the folder with the background noises."
    )
    parser.add_argument(
        "-ir", "--impulse_response", type=str, required=True, help="The path to the folder with the impules responses."
    )
    parser.add_argument("-m", "--model_120", type=str, required=True, help="The path of the model")
    parser.add_argument(
        "-pn", "--positive_negative", choices=["positive", "negative"], default="positive", help="Inference on positive or negative samples."
    )
    parser.add_argument("-s", "--segment_size", type=int, default=10, help="The segment size of the query in seconds.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    json_file = args.json
    index_file = args.index_file
    positive_samples = args.positive_samples
    negative_samples = args.negative_samples
    background_noise = args.background_noise
    impulse_response = args.impulse_response
    model_120 = args.model_120
    positive_negative = args.positive_negative
    segment_size = args.segment_size

    is_positive = False
    is_negative = False
    if positive_negative == "positive":
        is_positive = True
        output_positive = "augmented_positive/"
        if not os.path.exists(output_positive):
            os.makedirs(output_positive)
        output_dir = output_positive
        samples = positive_samples
        TP = []
        # FP=[]
    elif positive_negative == "negative":
        is_negative = True
        output_negative = "augmented_negative/"
        if not os.path.exists(output_negative):
            os.makedirs(output_negative)
        output_dir = output_negative
        samples = negative_samples
        TN = []
        # FN=[]

    print(output_dir)

    # create positive/negative samples with background noise
    if utils.is_dir_empty(output_dir):
        min_snr = 0.
        max_snr = 6.0

        transform = Compose(
            [
                AddBackgroundNoise(sounds_path=background_noise, min_snr_in_db=min_snr, max_snr_in_db=max_snr, p=1.),
                ApplyImpulseResponse(ir_path=impulse_response, p=1.)
            ]
        )

        progress = tqdm(os.listdir(samples), desc=f"Create augmented samples for {output_dir}.")
        for filename in tqdm(progress):
            if filename.endswith('.wav'):
                file_path = os.path.join(samples, filename)

                signal, sr = librosa.load(file_path, sr=8000, mono=True)
                augmented = transform(samples=signal, sample_rate=sr)

                rms_signal = np.sqrt(np.mean(signal**2))
                rms_noise = np.sqrt(np.mean((augmented - signal)**2))
                snr = 20 * np.log10(rms_signal / rms_noise)

                augmented_filename = f"augment_{filename}"
                augmented_filepath = os.path.join(output_dir, augmented_filename)

                print(f"{augmented_filepath} (SNR: {snr:.2f} dB)")
                sf.write(augmented_filepath, augmented, 8000, subtype='FLOAT')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Neural_Fingerprinter().to(device)
    model.load_state_dict(torch.load(model_120, map_location=torch.device(device)))

    with open(json_file, 'rb') as f:
        labels = json.load(f)

    indices = faiss.read_index(index_file)
    sorted_ind_labels = np.array(sorted(map(int, labels.keys())))

    print(f"Songs in DB: {len(labels)}")
    print(f"Fingerprints in DB: {indices.ntotal}")

    progress = tqdm(os.listdir(output_dir), desc=f"Inference on {output_dir} . . .")
    for song_name in progress:
        full_song_path = os.path.join(output_dir, song_name)
        y, sr = librosa.load(full_song_path, sr=8000)

        song_dur = librosa.get_duration(y=y, sr=sr)
        song_dur = int(song_dur)
        if song_dur >= segment_size:
            # Trim duration to be divisible by segment_size
            song_dur = int(song_dur / segment_size) * segment_size
            n_samples = int(song_dur * sr)
            y = y[:n_samples]
            song_dur = int(y.size / sr)
        else:
            continue

        print(f"\nSONG {song_name} DURATION DIVISIBLE BY SEGMENT SIZE ({segment_size}): {song_dur}")

        samples_per_segment = int(segment_size * sr)
        for i in range(0, y.size, samples_per_segment):

            seg = y[i:i + samples_per_segment]
            seg_dur = int(samples_per_segment / sr)

            xq = []
            t = np.arange(start=0, stop=seg_dur, step=STEP)
            t = t[:-1]

            model.eval()
            with torch.no_grad():
                for i in t:
                    s = utils.extract_mel_spectrogram(seg[int(i * sr):int((i + 1) * sr)])
                    s = s.reshape((1, *s.shape))
                    xq.append(s)
                xq = np.stack(xq)
                out = model(torch.from_numpy(xq).to(device))

            D, I = indices.search(out.cpu().numpy(), 4)

            song_names = []
            for row in I:
                song_row = []
                for col in row:
                    idx = utils.search_index(col, sorted_ind_labels)
                    song = labels[str(idx)]
                    song_row.append(song)
                song_names.append(song_row)
            song_arr = np.array(song_names)

            winner, conf_dist = utils.get_winner(labels, I, D, sorted_ind_labels)
            print(f"conf_dist {conf_dist}")
            tmp_name = song_name.replace("augment_", "")
            tmp_name = tmp_name.replace(".wav", "")

            if is_positive:
                if winner.lower().strip() == tmp_name.lower().strip():
                    print(f"[Found] {winner} / {tmp_name}")
                    TP.append(conf_dist)

            if is_negative:
                if winner.lower().strip() != tmp_name.lower().strip():
                    print(f"[NOT Found] {winner} / {tmp_name}")
                    TN.append(conf_dist)

    if is_positive:
        with open("true_positives.txt", "w") as f:
            for distance in TP:
                f.write(str(distance) + "\n")

    if is_negative:
        with open("true_negatives.txt", "w") as f:
            for distance in TN:
                f.write(str(distance) + "\n")
