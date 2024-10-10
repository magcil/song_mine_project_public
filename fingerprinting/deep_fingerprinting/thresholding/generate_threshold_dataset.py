import argparse
import sys
import os
import json
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

import faiss
import numpy as np
from tqdm import tqdm


POSITIVE_SAMPLES = 200
NEGATIVE_SAMPLES = 100


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-ps', '--positive_samples', required=True, help='Path with true positive wav files.')
    parser.add_argument('-ns', '--negative_samples', required=True, help='Path with true negative wav files.')
    parser.add_argument('-fps', '--fingerprints', required=True, help='The fingerprints of the model.')
    parser.add_argument('-nl', '--nlist', type=int, default=50, help='Number of lists in IVF.')
    parser.add_argument('-sb', '--subquantizers', type=int, default=32, help='Number of subquantizers in PQ.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    ps = utils.crawl_directory(args.positive_samples)
    ns = utils.crawl_directory(args.negative_samples)
    fps_path = args.fingerprints
    fps = utils.crawl_directory(fps_path)
    nlist = args.nlist
    m = args.subquantizers
    
    data_path = os.path.join(os.getcwd(), 'threshold_dataset')
    tic = time.perf_counter()

    try:
        os.mkdir(data_path)
        os.mkdir(os.path.join(data_path, 'positive_samples'))
        os.mkdir(os.path.join(data_path, 'negative_samples'))
    except Exception as e:
        raise ValueError(e)
    
    positive_wavs = np.random.choice(ps, size=POSITIVE_SAMPLES, replace=False)
    negative_wavs = np.random.choice(ns, size=NEGATIVE_SAMPLES, replace=False)

    # Copy positives
    for wav_to_move in positive_wavs:
        try:
            shutil.copy(src=wav_to_move, dst=os.path.join(data_path, 'positive_samples'))
        except Exception as e:
            print(f'Failed to copy: {os.path.basename(wav_to_move)}. Skipping...')
    
    # Copy negatives
    for wav_to_move in negative_wavs:
        try:
            shutil.copy(src=wav_to_move, dst=os.path.join(data_path, 'negative_samples'))
        except Exception as e:
            print(f'Failed to copy: {os.path.basename(wav_to_move)}. Skipping...')

    # Generate json & faiss index / harcoded index for now
    d = 128
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    fps = set([os.path.basename(fp).removesuffix('.npy') for fp in fps])
    ps = set([os.path.basename(tp).removesuffix('.wav') for tp in ps])
    fps_to_keep = fps.intersection(ps)

    num_fingerprints = 0
    p_bar = tqdm(fps_to_keep, desc='Calculating number of fingerprints in db.', leave=False)
    for fp in p_bar:
        x = np.load(os.path.join(fps_path, fp + '.npy'))
        num_fingerprints += x.shape[0]
    print(f'Number of total fingerprints in db: {num_fingerprints}.')

    xb = np.zeros(shape=(num_fingerprints, d), dtype=np.float32)

    # Create index
    p_bar = tqdm(fps_to_keep, desc='Filling index...', leave=False)
    idx = 0
    names = {}
    for fp in p_bar:
        x = np.load(os.path.join(fps_path, fp + '.npy'))
        num_fingerprints = x.shape[0]
        xb[idx: idx + num_fingerprints, :] = x
        names[idx] = os.path.basename(fp)
        idx += num_fingerprints
    
    print(f'Training index.... This may take a while.')
    index.train(xb)
    print(f'Train finished!')
    index.add(xb)

    print(f'Saving index and json...')
    faiss.write_index(index, 'threshold_index.index')
    with open('threshold_json.json', 'w') as f:
        json.dump(names, f)
    print('ok!')

    tac = time.perf_counter()
    total = tac - tic
    m, s = divmod(total, 60)
    print(f'Total time: {m} minutes and {s:.2f} seoncds')




    

    


    
    