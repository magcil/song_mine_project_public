import argparse
import json
import os
import pathlib
from time import perf_counter

import librosa
import numpy as np
import soundfile as sf
from audiomentations import (AddBackgroundNoise, ApplyImpulseResponse, Compose,
                             SpecFrequencyMask, TimeMask)
from config import ENERGY_THRESHOLD, SEED
from numpy.random import default_rng
from tqdm import tqdm

from utils import utils

CURR_PATH = pathlib.Path(__file__).parent



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Generate training dataset",
        description="Generating tuples of (original, augmented) low-power mel spectrograms "
        + "corresponding to 1 sec of audio samples from the original wav files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input_folder",
        help="The path of the input folder containing the wav files.",
    )
    parser.add_argument(
        "-n",
        "--number_segments",
        type=int,
        default=5,
        help="The total number of segments to extract from each wav file separately.",
    )
    parser.add_argument(
        "-tr",
        "--train_ratio",
        type=float,
        default=0.8,
        help="The size of train set, remaining corresponds to val set",
    )
    parser.add_argument(
        "-ir", "--impulse_response", help="The path containing the impulse responses."
    )
    parser.add_argument(
        "-bn",
        "--background_noise",
        help="The path containing the background noisy files.",
    )
    parser.add_argument(
        "-o", "--output_path", help="The path to save the outputs of this script."
    )

    return parser.parse_args()


def generate_set(data_set: np.ndarray, desc: str, is_training : bool = True):
    sample_indices = list(rng.choice(data_set.size, size=2, replace=False))
    progress_bar = tqdm(data_set, desc=f"Preparing {desc} set...")
    json_data = {}

    for j, wav_file in enumerate(progress_bar):

        try:
            signal, sr = librosa.load(wav_file, sr=8000)
            total_segs = int(signal.size / sr)
            filenames = []

            # Discard segments with energy <= ENERGY_THRESHOLD
            energies = np.array(
                [
                    utils.energy_in_db(signal[i * sr : (i + 1) * sr])
                    for i in range(total_segs)
                ]
            )
            seg_indices = rng.choice(
                np.where(energies > ENERGY_THRESHOLD)[0], size=num_segs, replace=False
            )

            if seg_indices.any():
                for idx in seg_indices:
                    idx = idx.item()
                    # Get the corresponding segment
                    y = signal[idx * sr : (idx + 1) * sr]

                    # Initialize augmentation chain
                    if is_training:
                        ir_path = rng.choice(train_impulse_responses, size=1).item()
                        noise_path = rng.choice(train_background_noises, size=1).item()
                    else:
                        ir_path = rng.choice(val_impulse_responses, size=1).item()
                        noise_path = rng.choice(val_background_noises, size=1).item()

                    augmentation_chain = Compose(
                        [
                            AddBackgroundNoise(
                                sounds_path=noise_path,
                                min_snr_in_db=0.0,
                                max_snr_in_db=20.0,
                                p=1.0,
                            ),
                            ApplyImpulseResponse(ir_path=ir_path, p=1.0),
                            TimeMask(
                                min_band_part=0.06, max_band_part=0.08, fade=True, p=0.1
                            ),
                        ]
                    )

                    # Offset probability
                    if rng.random() > 0.75:
                        offset_signal = utils.time_offset_modulation(
                            signal=signal, time_index=idx
                        )
                        augmented_signal = augmentation_chain(
                            offset_signal, sample_rate=8000
                        )
                    else:
                        augmented_signal = augmentation_chain(y, sample_rate=8000)

                    S1 = utils.extract_mel_spectrogram(y)
                    S2 = SpecFrequencyMask(p=0.33)(
                        utils.extract_mel_spectrogram(augmented_signal)
                    )

                    filename = f"{os.path.basename(wav_file.rstrip('.wav'))}_{idx}.wav"
                    if is_training:
                        np.save(os.path.join(train_path, filename.rstrip(".wav")), (S1, S2))
                    else:
                        np.save(os.path.join(val_path, filename.rstrip(".wav")), (S1, S2))
                    filenames.append(filename)

                    if j in sample_indices:
                        sf.write(
                            os.path.join(samples_path, "Original", filename),
                            y,
                            8000,
                            subtype="FLOAT",
                        )
                        sf.write(
                            os.path.join(samples_path, "Augmented", filename),
                            augmented_signal,
                            8000,
                            subtype="FLOAT",
                        )

                json_data[os.path.basename(wav_file)] = filenames

            else:
                log_info = (
                    f"File: {wav_file} does not contain enough energy. Skipping..."
                )
                print(log_info)

        except Exception as err:
            log_info = (
                f"Error occured on: {os.path.basename(wav_file)}."
                + f" Saved files: {filenames}."
            )
            print(log_info)

    return json_data


if __name__ == "__main__":

    args = parse_args()

    wav_files = np.array(utils.crawl_directory(args.input_folder))
    num_segs = args.number_segments
    output_path = args.output_path if args.output_path else CURR_PATH
    train_ratio = args.train_ratio
    impulse_responses = np.array(utils.crawl_directory(args.impulse_response))
    background_noises = np.array(utils.crawl_directory(args.background_noise))

    if output_path is not CURR_PATH:
        try:
            os.mkdir(output_path)
        except (FileExistsError, FileNotFoundError) as err:
            print(f"Exception occured: {err}")
            raise err

    rng = default_rng(SEED)
    train_set, val_set = utils.split_to_train_val_sets(wav_files, train_ratio, rng)

    log_info = f"Total files: {len(wav_files)}\nTrain files: {train_set.size}\nTest files: {val_set.size}"
    print(log_info)
    print(f"Input folder: {args.input_folder}")
    print(f"Train ratio: {train_ratio}")

    train_impulse_responses, val_impulse_responses = utils.split_to_train_val_sets(
        impulse_responses, train_ratio, rng
    )
    print(f"Loading impulse responses from: {args.impulse_response}")
    print(
        f"Train impulse responses: {train_impulse_responses.size}"
        + f"\nTest impulse responses: {val_impulse_responses.size}"
    )

    train_background_noises, val_background_noises = utils.split_to_train_val_sets(
        background_noises, train_ratio, rng
    )
    print(f"Loading backroung noises from: {args.background_noise}")
    print(
        f"Train background noises: {train_background_noises.size}"
        + f"\nTest background noises: {val_background_noises.size}"
    )
    
    dataset_path = os.path.join(output_path, f"Dataset_{wav_files.size}")
    try:
        os.mkdir(dataset_path)
    except (FileExistsError, FileNotFoundError) as err:
        print(f"Exception occured: {err}")
        raise err

    log_info = f"Created folder: Dataset_{wav_files.size} in {output_path}"
    print(log_info)

    # Train set
    train_path = os.path.join(dataset_path, "train_set")
    try:
        os.mkdir(train_path)
    except (FileExistsError, FileNotFoundError) as err:
        print(f"Exception occured: {err}")
        raise err

    samples_path = os.path.join(dataset_path, "Samples")
    try:
        os.mkdir(samples_path)
        os.mkdir(os.path.join(samples_path, "Original"))
        os.mkdir(os.path.join(samples_path, "Augmented"))
    except (FileExistsError, FileNotFoundError) as err:
        print(f"Exception occured: {err}")
        raise err

    tic = perf_counter()
    json_tr_data = generate_set(train_set, desc="training", )
    log_info = f"Training dataset completed. Run time: {(perf_counter()-tic):.3f}"
    print(log_info)

    with open(os.path.join(output_path, "train_data.json"), "w") as f:
        json.dump(json_tr_data, f)
        print(f"Successfully saved train_data.json in {output_path}")

    # Val set
    val_path = os.path.join(dataset_path, "val_set")
    try:
        os.mkdir(val_path)
    except (FileExistsError, FileNotFoundError) as err:
        print(f"Exception occured: {err}")
        raise 

    tic = perf_counter()
    json_val_data = generate_set(val_set, desc="validation", is_training=False)
    log_info = f"Validation dataset completed. Run time: {(perf_counter()-tic):.3f}"
    print(log_info)

    with open(os.path.join(output_path, "val_data.json"), "w") as f:
        json.dump(json_val_data, f)
        print(f"Successfully saved val_data.json in {output_path}")
