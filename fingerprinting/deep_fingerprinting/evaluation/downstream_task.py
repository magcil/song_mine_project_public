import argparse
import os
import glob
import sys

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.nn.functional import softmax
from sklearn.metrics import classification_report

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(os.path.dirname(current_file_path))

sys.path.insert(0, parent_dir_path)
from utils import utils
from callbacks.callback import EarlyStopping
from models.neural_fingerprinter import LinearClassifier, Neural_Fingerprinter
from datasets.datasets import GtzanDataset

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SEED = 42


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_folder',
        required=True,
        help='The path to the folder containing the wav files of the downstream task separated in classes.'
    )
    parser.add_argument('-m', '--model', help='The path of the .pt file to load the weights.')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='The number of epochs to train.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='The batch size.')

    return parser.parse_args()


def split_dataset(input_folder, train_ratio, val_ratio, test_ratio, rng):
    train_set, val_set, test_set = [], [], []
    tree = os.walk(input_folder)
    _, genres, _ = next(tree)

    for genre in genres:
        genre_songs = np.array(utils.crawl_directory(os.path.join(input_folder, genre)))
        train, val = utils.split_to_train_val_sets(genre_songs, train_ratio=train_ratio, rng=rng)
        adjusted_test_ratio = test_ratio / (test_ratio + val_ratio)
        test, val = utils.split_to_train_val_sets(val, train_ratio=adjusted_test_ratio, rng=rng)
        print(f'Genre: {genre}. Train: {train.size}. Val: {val.size}. Test: {test.size}')
        train_set += list(train)
        val_set += list(val)
        test_set += list(test)

    return genres, train_set, val_set, test_set


def train_linear_classifier(model, train_set, val_set, class_mapping, epochs, batch_size):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss().to(device)

    train_dloader = DataLoader(GtzanDataset(train_set, class_mapping), batch_size=batch_size, shuffle=True)
    val_dloader = DataLoader(GtzanDataset(val_set, class_mapping), batch_size=batch_size, shuffle=False)
    ear_stopping = EarlyStopping(patience=5, path='classifier.pt', verbose=True)
    optim = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer=optim, T_max=100, eta_min=1e-7)

    train_loss, val_loss = 0., 0.
    _padding = len(str(epochs + 1))

    for epoch in range(1, epochs + 1):

        model.train()
        with tqdm(train_dloader, unit='batch', leave=False, desc='Training set.') as tbatch:
            for x, y in tbatch:
                # Forward
                out = model(x.to(device))
                loss = loss_fn(out, y.to(device))
                train_loss += loss.item()

                # Backward
                optim.zero_grad()
                loss.backward()
                optim.step()
            train_loss /= len(train_dloader)
            lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            with tqdm(val_dloader, unit='batch', leave=False, desc='Validation set.') as vbatch:
                for x, y in vbatch:
                    out = model(x.to(device))
                    loss = loss_fn(out, y.to(device))
                    val_loss += loss.item()
            val_loss /= len(val_dloader)

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")
        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0


def song_level_inference(src, model, class_mapping, batch_size):
    loader = DataLoader(GtzanDataset([src], class_mapping), batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    inv_class_mapping = {i: genre for (genre, i) in class_mapping.items()}
    pred_labels, pred_probs = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = softmax(logits.cpu(), dim=1)
            p, l = probs.max(dim=1)
            pred_labels += list(l.numpy())
            pred_probs += list(p.numpy())
    df = pd.DataFrame({'preds': pred_labels, 'probs': pred_probs})
    df = df.groupby(['preds']).agg({
        'preds': 'size',
        'probs': 'mean'
    }).rename(columns={
        'preds': 'count'
    }).reset_index().sort_values(by=['count', 'probs'], ascending=False)
    id, counts, p = df.iloc[0]
    true_genre = os.path.basename(os.path.dirname(src))
    print(f'True genre: {true_genre}. Predicted: {inv_class_mapping[int(id)]}. Prob: {p:.3f}')

    return true_genre, inv_class_mapping[int(id)]


def evaluate_linear_classifier(model, test_set, class_mapping, batch_size):

    y_true, y_preds = [], []
    for src in test_set:
        label, pred = song_level_inference(src, model, class_mapping, batch_size)
        y_true.append(label)
        y_preds.append(pred)

    print(classification_report(y_true=y_true, y_pred=y_preds))


if __name__ == '__main__':

    args = parse_args()
    input_folder = args.input_folder
    epochs = args.epochs
    batch_size = args.batch_size
    rng = np.random.default_rng(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model:
        model = Neural_Fingerprinter()
        model.load_state_dict(torch.load(args.model))
        classifier = LinearClassifier()
        classifier.encoder = model.encoder
        del model
        classifier = classifier.to(device)
        # Freeze all layers of the encoder
        for p in classifier.encoder.parameters():
            p.requires_grad = False
    else:
        classifier = LinearClassifier().to(device)

    genres, train_set, val_set, test_set = split_dataset(input_folder, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, rng)

    class_mapping = {genre: i for (i, genre) in enumerate(genres)}
    print(class_mapping)

    # Train model
    train_linear_classifier(classifier, train_set, val_set, class_mapping, epochs, batch_size)

    # Test model
    classifier = LinearClassifier()
    classifier.load_state_dict(torch.load('classifier.pt'))
    evaluate_linear_classifier(classifier, test_set, class_mapping, batch_size)
