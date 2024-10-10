<h1> Deep Fingerprinting </h1>

This folder contains an alternative approach to audio fingerprinting by utilizing deep learning approaches. The main references for this approach are the following papers:

<ol>
  <li><i><a href="https://arxiv.org/pdf/2010.11910.pdf">Neural Audio Fingerprint for high-specific Audio Retrieval
    Based On Contrastive Learning</a></i>
</li>
  <li><i><a href="https://arxiv.org/pdf/2210.08624.pdf">Attention-Based Audio Embeddings For Query-By-Example</a></i></li>
  <li><i><a href="https://sci-hub.se/10.1145/3380828">SAMAF: Sequence-to-sequence Autoencoder Model
for Audio Fingerprinting</a></i></li>
</ol>

<h2>Training</h2>

There are 2 methods of training. The first one is the _static_ method, in which we create the augmented dataset in before hand and use it to train our model, and the second one is the _online_ approach that applies the audio augmentations during the training in an online fashion.

To run the scripts you'll need to download the wav files corresponding to the _impulse responses_ and the _background noises_ from <a href="https://drive.google.com/drive/folders/11_wSBttlNK31SaV47_id5DFtmJhveEJn">here</a>. The impulse responses will be used to add reverbation and the background noises to add noise on the original samples.

<h3>1. Static Method</h3>
<h4>1.1 Generation of the augmented dataset</h4>

In this paragraph we describe how the generation of the training dataset works. The resulting dataset consists of pairs with cleaned-original sounds and augmented sound resulting from various transformations on the original one (e.g. adding background noise, reverbation, time offset and masking on time & spectral domain). For the purposes of the dataset generation a script named `generate_training_dataset.py` has been designed. Below we explain its main functionalities and we give information on its usage.

The input arguments are the following:

<ol type="1">
  <li> <code> -i</code> or <code>--input_folder</code>: The path of the input folder containing the wav files.</li>
  <li><code> -n </code> or <code> --number_segments </code>: The total number of segments corresponding to 1sec to extract from each wav file (default: 5).</li>
  <li> <code> -tr</code> or <code> --train_ratio</code>: The size of the training set, remaining corresponds to validation set (default 0.8). </li>
  <li><code> -ir</code> or <code> --impulse_response</code>: The path containing the impulse responses (default: None).</li>
  <li><code> -bn</code> or <code> --background_noise </code>: The path containing the background noisy files (default: None). </li>
  <li> <code> -o </code> or <code> --output_path </code>: The path to save the outputs of this script (default: None).</li>
</ol>

Assuming the working directory is something like `~/projects/song_mine_project/`, atypical run of the script (with the default arguments) would be:

```bash
python RnD/deep_fingerprinting/generate_training_dataset.py -i <path_to_wav_files> -bn <path_background_noise> -ir <path_to_impulse_responses>
```

For example assuming everything is placed on `~/projects/song_mine_project/data/` the previous command becomes

```bash
python RnD/deep_fingerprinting/generate_training_dataset.py -i data/folder_of_wav_files -bn data/background_noise -ir data/impulse_responses/
```

The outputs of the script will be placed in the specified `output_path` by the user. The output should be a folder named `Dataset_x`, where `x` is the total number of wav files processed. This folder will contain a subfolder named `train_set` containing the pairs of `(x_reg, x_aug)`, where each `x_reg, x_aug` is a numpy array of size $256 \times 32$ corresponding to the original log-power mel-spectrograms and the augmented log-power mel-spectrograms, respectively for the original audio segment of length 1 sec and the augmented one. A folder named `val_set` with similar specifications will contain the numpy arrays for the validation set. In addition, a folder `Samples` will be created containing some samples for the original audio segments and their corresponding augmented ones for inspection reasons. Finally, two `json` files (`train_data.json`, `val_data.json`) will be created containing as keys the wav files and as values a list showing on which part of the song (i.e. which second) the samples were obtained. For example, if the wav filename is `02  Spice Girls - Say You'll Be There.wav` then a value of `02  Spice Girls - Say You'll Be There_5.wav` means that the audio segment of 1 seconds corresponds to the segment between the 5th and 6th second of the original song.

<h4>1.2 Training loop</h4>

In this paragraph we describe how to train the CNN Neural Fingerprinter using `trainer.py`. The script has the following 5 optional arguments:

<ol type="1">
  <li> <code> -ep</code> or <code>--epochs</code>: The number of total epochs to train the model (if no early stopping is encountered) (default 200).</li>
  <li><code> -bs </code> or <code> --batch_size </code>: The batch size. For small GPU's 32 is recommended. Ideal values are 120, 320, 640 (default 32).</li>
  <li> <code> -m</code> or <code> --model_name</code>: The filename to save the model's parameters in a pt. file. </li>
  <li> <code>--optim</code>: Choose an optimizer between Adam and Lamb (default Adam).
  <li><code> -o</code> or <code> --output_path</code>: The path to save the model's parameters. (default cwd)</li>
</ol>

Furthermore, one positional argument is required with options `static` or `dynamic`. `static` corresponds to training the model on a predetermined dataset and `dynamic` to training the model with online augmentations. Given the positional argument `static` two more arguments are required:

<ol type="1">
  <li> <code> -tp</code> or <code>--train_path</code>: The path containing the training dataset (i.e. the npy files) </li>
  <li><code> -vp</code> or <code> --val_path </code>: The path containng the validation dataset (i.e. the npy files) </li>
</ol>

Assuming you are on `~/projects/song_mine_project/RnD/deep_fingerprinting` an example use case would be:

```bash
python trainer.py -ep 200 -bs 120 -m <model_name> -o <output_path> static -tp <path_to_training_data> -vp <path_to_val_data>
```

<h3>2. Online method</h3>

To perform online training, the augmentations will be done on-the-fly, using the `trainer.py` script. There are 2 ways to run the script. The first one is to run it by using the flag `dynamic_online` to perform the split of the `wav_files`, the `impulse_responses` and the `background_noises`. The 5 optional arguments described in `1.2` (_epochs_, _batch_size_, _model_name_, _optim_, _output_path_) can be used.
Example:

```bash
python3 RnD/deep_fingerprinting/trainer.py -ep 200 -m model dynamic_online -p data/rnd_songs -ir data/impulse_responses/ -bn data/background_noise/
```

- `-p`: The path containing the wav files
- `-ir`: The path containing the impulse responses
- `-bn`: The path containing the background noises
- `-tr`: The train ratio (default 0.8)

This script will perform a train/test split of the wav files, the impulse responses and the background noises and will create a folder named `online_data`. In this folder, 2 subfolders are created:

- `train` : It contains 3 folders `train_set`,`train_impulse_responses` and `train_background_noises`
- `val`: It contains 3 folders `val_set`, `val_impulse_responses` and `val_background_noises`

The wav files are divided into these folders based on the train ratio.

The second way to run the script is by using the flag `dynamic`. After splitting the wav files into the folders described above, you can use the flag `dynamic` with the paths of these folders to train the model (avoiding the process of splitting the files). Example:

```bash
python3 RnD/deep_fingerprinting/trainer.py dynamic -tp online_data/train/train_set -tr online_data/train/train_impulse_responses -tb online_data/train/train_background_noises -vp online_data/val/val_set -vr online_data/val/val_impulse_responses -vb online_data/val/val_background_noises
```

where

- `-tp`: The path containing the wav files of the training set
- `-tr`: The path containing the wav files of the impulse responses for the training
- `-tb`: The path containing the wav files of the background noises for the training
- `-vp`: The path containing the wav files of the validation set
- `-vr`: The path containing the wav files of the impulse responses for the validation
- `-vb`: The path containing the wav files of the background noises for the validation process

<h2>Create the database using faiss</h2>

In this paragraph we describe how to run the script `create_faiss_indexes.py` in order to create an index from the representations of the neural fingerprinter. The MySQL analog of the dejavu-approach is an object called _index_. We use an open source library called <a href="https://github.com/facebookresearch/faiss">Faiss</a> for the creation of the index as well as to handle the similarity searches on the index. For the moment, we use the <a href="https://en.wikipedia.org/wiki/Inverted_index#:~:text=In%20computer%20science%2C%20an%20inverted,index%2C%20which%20maps%20from%20documents">Inverted File Index</a> (IVF) with Product Quantization (PQ) that handles the compression of the vector representations obtained from the model. The script `create_faiss_indexes.py` has the following arguments:

- `-i`: The path to the deep audio fingerprints.
- `-n`: The name to use for storing the faiss index and json.
- `-m`: (Optional) The number of subquantizers to use (default: 32).
- `-d`: (Optional) The number of dimensions to use (default: 128).
- `-nlist`: (Optional) The number of cells to use (default: 256).

A typical execution would be

```bash
python RnD/deep_fingerprinting/create_faiss_indexes.py -i <path_to_fingerprintings> -n <name_of_files>
```
