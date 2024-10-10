import argparse
import glob
import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
from config import FAIS_M, FAISS_D, FAISS_NLIST
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_fingerprints",
        nargs="+",
        required=True,
        help="The directories to the deep audio fingerprints.",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="The name to use for storing faiss index and json.",
    )

    parser.add_argument(
        "-m",
        type=int,
        default=FAIS_M,
        required=False,
        help="The number of subquantizers to use.",
    )
    parser.add_argument(
        "-d",
        type=int,
        default=FAISS_D,
        required=False,
        help="The number of dimensions to use.",
    )
    parser.add_argument(
        "-nlist",
        type=int,
        default=FAISS_NLIST,
        required=False,
        help="The number of cells to use.",
    )
    parser.add_argument('-l', '--list', help='List of songs to place to the database in txt format.')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    fingerprints = args.input_fingerprints
    name = args.name
    txt = args.list

    nlist = args.nlist
    m = args.m
    d = args.d
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    # index = faiss.index_factory(d, "OPQ64,IVF256,PQ64")
    items = []
    for f in fingerprints:
        items += list(glob.glob(os.path.join(f, "*.npy")))
    print(f"Before: {len(items)}", end="")
    if txt:
        with open(txt, 'r') as f:
            songs_to_insert = [x.rstrip('\n').removesuffix('.wav') for x in f.readlines()]
        to_keep = []
        items = [item for item in items if os.path.basename(item).removesuffix(".npy") in songs_to_insert]
    print(f"| After: {len(items)}")

    xb = []
    names = {}
    idx = 0
    tic = time.perf_counter()
    progress_bar = tqdm(items, desc="Calculating index dimensions...", leave=False)
    num_vectors = 0
    for item in progress_bar:
        x = np.load(item)
        num_vectors += x.shape[0]

    xb = np.zeros(shape=(num_vectors, d), dtype=np.float32)
    progress_bar = tqdm(items, desc="Filling database...", leave=False)
    idx = 0
    for item in progress_bar:
        x = np.load(item)
        num_vectors = x.shape[0]
        xb[idx:idx + num_vectors, :] = x
        names[idx] = os.path.basename(item).removesuffix(".npy")
        idx += num_vectors
    print(f"Training index... This may take a while.")
    index.train(xb)
    print(f"Train finished!")
    index.add(xb)
    print(f"Total items on index: {index.ntotal}")

    print(f"Saving index and json...")
    faiss.write_index(index, name + ".index")
    with open(name + ".json", "w") as f:
        json.dump(names, f)
    print("ok!")
    tac = time.perf_counter()
    total = tac - tic
    m, s = divmod(total, 60)
    print(f"Total time: {m} minutes and {s:.2f} seoncds")
