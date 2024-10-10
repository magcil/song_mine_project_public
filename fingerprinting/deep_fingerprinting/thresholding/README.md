# Threshold Estimation for song existance in database

This folder contains all the necessary scripts for threshold estimation and evaluation:

1. `generate_threshold_dataset.py`
2. `estimate_threshold.py`
3. `evaluate_threshold.py`

## Generate the dataset to use fot the threshold estimation

The script below is designed to generate a threshold dataset by selecting a specified number of 200 positive (samples that do exist in the database) 
and 100 negative samples (that do not exist in the database) from the provided directories. 
It generates a JSON file along with a FAISS index for all the wav files that exist in the database.

The script accepts the following command line arguments:

- `ps` or `--positive_samples`: Path to the directory containing true positive WAV files.
- `ns` or `--negative_samples`: Path to the directory containing true negative WAV files.
- `fps` or `--fingerprints`: Path to the directory containing the fingerprints of the model.

Example usage: 
```
python3 generate_threshold_dataset.py -ps positive_samples -ns negative samples -fps fingerprints
```

## Estimate threshold

The script below is used for the estimation of the threshold. It takes a set of positive and negative samples, 
applies augmentation techniques (adding background noise and impulse response), and then matches the samples against a the database using a pre-trained deep audio fingerprinting model. 

The script requires the following arguments:

- `ps`, `--positive_samples`: The path folder of the 200 positive samples in .wav format that exist in the database.
- `ns`, `--negative_samples`: The path folder of the 100 negative samples in .wav format that do not exist in the database.
- `j`, `--json`: The JSON file containing the WAV names and their corresponding indices in the database.
- `id`, `--index_file`: The path to the file with the Faiss indices.
- `bn`, `--background_noise`: The path to the folder containing the (test) background noises.
- `ir`, `--impulse_response`: The path to the folder containing the (test) impulse responses.
- `m`, `--model_120`: The path to the trained model.
- `pn`, `--positive_negative` (optional): Inference on positive or negative samples. Default is positive.
- `s`, `--segment_size` (optional): The segment size of the query in seconds. Default is 10.

Example usage:

```
python3 estimate_threshold.py -ps positive_samples -ns negative_samples -j json_file -id index_file -bn background_noises -ir impulse_responses -pn positive -s 10
```
> The `-pn` flag is used to calculate the confidence either on positive or negative samples.

The script outputs evaluation metrics based on the inference:

- For positive samples, a file named _true_positives.txt_ is created, which contains the confidence distances for true positive matches.
- For negative samples, a file named _true_negatives.txt_ is created, which contains the confidence distances for true negative matches.

## Densities pkl

The code below performs kernel density estimation on the data provided from the .txt files, using Gaussian kernels and Scott's rule for bandwidth selection. 
The resulting density estimations are then saved in a dictionary and serialized using the pickle module, with the output stored in a file named _densities.pkl_.

```
from sklearn.neighbors import KernelDensity
import numpy as np
import pickle

with open('true_negatives.txt', 'r') as f:
    true_negatives = [float(x) for x in f.readlines()]

with open('true_positives.txt', 'r') as f:
    true_positives = [float(x) for x in f.readlines()]

kde_true_negatives = KernelDensity(kernel='gaussian', bandwidth='scott').fit(np.array(true_negatives).reshape(len(true_negatives), 1))
kde_true_positives = KernelDensity(kernel='gaussian', bandwidth='scott').fit(np.array(true_positives).reshape(len(true_positives), 1))

d = {}
d['tp_density'] = kde_true_positives
d['tn_density'] = kde_true_negatives

with open('densities.pkl', 'wb') as f:
    pickle.dump(d, f)
```

## Evaluate threshold

This script is used to evaluate the estimated threshold for the pre-trained model and the given dataset. 

- `id`, `--index`: The faiss index.
- `j`, `--json`: Json file with filenames.
- `d`, `--duration`: Query duration in seconds (default: 10).
- `ps`, `--positive_samples`: Path to positive samples.
- `ns`, `--negative_samples`: Path to negative samples.
- `m`, `--model`: Path to the pre-trained model.
- `th`, `--threshold_densities`: The pkl file with the densities.
- `bs`, `--background_noise`: Path to the background noises.
- `nps`, `--number_pos_samples`: The number of positive samples to use (optional).
- `nns`, `--number_negative_samples`: The number of negative samples to use (optional).

Example usage:

```
python evaluate_threshold.py -id faiss.index -j filenames.json -d 10 -ps positive_samples -ns negative_samples -m model.pth -th threshold_densities.pkl -bs background_noises
```

The script prints the evaluation results, including accuracy, precision, and recall scores. It also prints 
the mean and standard deviation for distances calculated for positive and negative samples.



