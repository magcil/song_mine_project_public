import argparse
import math
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
from callbacks.callback import EarlyStopping
from config import SEED
from datasets.datasets import DynamicAudioDataset, StaticAudioDataset
from loss.loss import NTxent_Loss, NTxent_Loss_2
from models.neural_fingerprinter import Neural_Fingerprinter
from numpy.random import default_rng
from on_line_training import create_sets
from optim.lamb import Lamb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.colors import Colors
from utils.torch_utils import Collate_Fn
from utils.utils import crawl_directory


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        prog="Trainer.py",
        description="The training loop for the CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-ep", "--epochs", type=int, default=200, help="The number of epochs.")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("-m", "--model_name", type=str, help="The model name to save the pkt file.")
    parser.add_argument("-o", "--output_path", type=str, help="The path to save the model.")
    parser.add_argument('-tl', help='Transfer learning.')

    parser.add_argument(
        "--optim",
        type=str,
        choices=["Adam", "Lamb"],
        default="Adam",
        help="Choose the optimizer to use (Adam or Lamb).",
    )

    subparser = parser.add_subparsers(
        help="The type of dataset to be used. Static for static," + "dynamic for online augmentations.",
        dest="dataset",
        required=True,
    )

    static = subparser.add_parser("static", help="Flag to use the static dataset")
    static.add_argument(
        "-tp",
        "--train_path",
        type=str,
        required=True,
        help="The path of the training samples.",
    )
    static.add_argument(
        "-vp",
        "--val_path",
        type=str,
        required=True,
        help="The path of the validation samples.",
    )

    dynamic_online = subparser.add_parser(
        "dynamic_online", help="Flag to the training schedule with online augmentations."
    )

    dynamic_online.add_argument("-p", "--path", required=True, help="The path containing the raw wav files.")
    dynamic_online.add_argument(
        "-ir",
        "--impulse_responses",
        required=True,
        help="The path containing the impulse responses.",
    )
    dynamic_online.add_argument(
        "-bn",
        "--background_noise",
        required=True,
        help="The path containing the background noises",
    )
    dynamic_online.add_argument(
        "-tr",
        "--train_ratio",
        type=float,
        default=0.8,
        help="The size of train set, remaining corresponds to val set",
    )

    dynamic = subparser.add_parser(
        "dynamic", help="Flag to the training schedule with online augmentations (train/val split done)."
    )

    dynamic.add_argument(
        "-tp",
        "--train_path",
        required=True,
        help="The path containing the train_set wav files.",
    )
    dynamic.add_argument(
        "-tr",
        "--train_impulse_responses",
        required=True,
        help="The path containing the training impulse responses.",
    )
    dynamic.add_argument(
        "-tb",
        "--train_background_noise",
        required=True,
        help="The path containing the training background noises",
    )

    dynamic.add_argument(
        "-vp",
        "--val_path",
        required=True,
        help="The path containing the val_set wav files.",
    )
    dynamic.add_argument(
        "-vr",
        "--val_impulse_responses",
        required=True,
        help="The path containing the validation impulse responses.",
    )
    dynamic.add_argument(
        "-vb",
        "--val_background_noise",
        required=True,
        help="The path containing the validation background noises",
    )

    return parser.parse_args()


def training_loop(train_dset, val_dset, epochs, batch_size, model_name=None, output_path=None, optim="Adam", tl=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    N = batch_size // 2
    model = Neural_Fingerprinter().to(device)
    num_workers = multiprocessing.cpu_count()
    if tl is not None:
        model.load_state_dict(torch.load(tl))
        print(f'Loaded weights from {tl}')
    loss_fn = NTxent_Loss_2(N, N, device=device)
    train_dloader = DataLoader(
        train_dset,
        batch_size=N,
        shuffle=True,
        collate_fn=Collate_Fn(rng=np.random.default_rng(SEED)),
        num_workers=num_workers,
        drop_last=True
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=N,
        shuffle=False,
        collate_fn=Collate_Fn(rng=np.random.default_rng(SEED)),
        num_workers=num_workers,
        drop_last=True
    )
    if optim == "Adam":
        optim = Adam(model.parameters(), lr=1e-4 * batch_size / 640)
    elif optim == "Lamb":
        optim = Lamb(model.parameters(), lr=1e-4 * batch_size / 640)
    else:
        raise ValueError(f"Invalid optimizer specified: {optim}")

    lr_scheduler = CosineAnnealingLR(optimizer=optim, T_max=100, eta_min=1e-7)

    ear_stopping = EarlyStopping(patience=25, verbose=True, path=os.path.join(output_path, model_name + ".pt"))

    train_loss, val_loss = 0.0, 0.0

    _padding = len(str(epochs + 1))

    for epoch in range(1, epochs + 1):

        model.train()
        i = 0

        with tqdm(train_dloader, unit="batch", leave=False, desc="Training set") as tbatch:
            for i, (x_org, x_aug) in enumerate(tbatch, 1):
                # Forward pass
                X = torch.cat((x_org, x_aug), dim=0).to(device)
                X = model(X)
                x_org, x_aug = torch.split(X, N, 0)
                loss = loss_fn(x_org, x_aug)
                train_loss += loss.item()

                # Backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()
            train_loss /= len(train_dloader)
            lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            with tqdm(val_dloader, unit="batch", leave=False, desc="Validation set") as vbatch:
                for x_org, x_aug in vbatch:
                    # Forward pass
                    X = torch.cat((x_org, x_aug), dim=0).to(device)
                    X = model(X)
                    x_org, x_aug = torch.split(X, N, 0)
                    loss = loss_fn(x_org, x_aug)
                    val_loss += loss.item()
        val_loss /= len(val_dloader)

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")

        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0


if __name__ == "__main__":

    args = parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.model_name if args.model_name else f"model_{batch_size}_{epochs}"
    output_path = args.output_path if args.output_path else os.getcwd()
    tl = args.tl

    if args.dataset == "static":

        train_path, val_path = args.train_path, args.val_path
        train_dset = StaticAudioDataset(train_path)
        val_dset = StaticAudioDataset(val_path)

        training_loop(train_dset, val_dset, epochs, batch_size, model_name, output_path, optim=args.optim)
        sys.exit(0)

    elif args.dataset == "dynamic_online":
        # if splitting wav files into training and validation set is not done,
        # split them and save them in them into "online_data"
        wav_files = np.array(crawl_directory(args.path))
        background_noises = np.array(crawl_directory(args.background_noise))
        impulse_responses = np.array(crawl_directory(args.impulse_responses))

        rng = default_rng(SEED)

        train_set, train_impulse_responses, train_background_noises, \
            val_set, val_impulse_responses, val_background_noises = create_sets(wav_files,
             background_noises, impulse_responses, args.train_ratio, rng)

    elif args.dataset == "dynamic":
        train_set = args.train_path
        val_set = args.val_path
        train_impulse_responses = args.train_impulse_responses
        val_impulse_responses = args.val_impulse_responses
        train_background_noises = args.train_background_noise
        val_background_noises = args.val_background_noise

    train_dset = DynamicAudioDataset(train_set, train_background_noises, train_impulse_responses)
    val_dset = DynamicAudioDataset(val_set, val_background_noises, val_impulse_responses)

    training_loop(train_dset, val_dset, epochs, batch_size, model_name, output_path, optim=args.optim, tl=tl)