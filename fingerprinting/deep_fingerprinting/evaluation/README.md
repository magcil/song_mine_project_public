<h1> Evaluation scripts of the deep fingerprinting approach </h1>

This folder contains all the scripts that are used to evaluate the neural fingerprinter. For this purpose, three scripts have been created

1. `test_model.py`
2. `online_inference.py`
3. `downstream_task.py`

**Note** In all scripts that require any input from the user provide the full absolute path for the arguments.

<h2> Test on a record file </h2>

The first script `test_model.py` tests the model on a recorded file. The arguments are the following:

- `-i`: The path to input recordings to be predicted.
- `-wf`: The path to the ground truth wav files.
- `-d`: The segment size of each query (in secs).
- `-l`: The path to the csv file (or txt) containing the ground truth labels.
- `-m`: The containing the model.
- `-id`: A faiss index for the database.
- `-js`: The json file with the correspondence.
- `-s`: The search method to use (sequence_search/majority_vote).
- `-k`: The number of nearest neighbors to retrieve.
- `-pb`: The number of probes.
- `-td`: The pk file with threshold densities.
- `-dv`: The device to be used for inference (cpu or cuda).

This test is used to test the model on the recordings `low_t.wav`, `mid_t.wav` and `high_t.wav`. The argument `-i` corresponds to any of these three recordings. The argument `-wf` corresponds to the original wav files of the songs played in any of these three recordings. This argument is required in order to extract the time duration of the songs in order to properly align the model's predictions with the ground truth labels. The argument `-d` is the size of the query measured in seconds. The argument `-l` is a txt/csv file with names of the songs played in the same order as shown in the txt/csv file. An example of this file for the three aforementioned recordings is shown below

```
055-Coldplay-Viva La Vida.wav
1970-036 Creedence Clearwater Revival - Lookin' Out My Back Door.wav
Brad Paisley - We Danced.wav
083. Sam Smith - How Do You Sleep_.wav
The Four Seasons - Spring - Allegro.wav
1990-002 Roxette - It Must Have Been Love.wav
1991-043 Firehouse - Love Of A Lifetime.wav
16 - I_ve got you under my skin.wav
2007-029 Pink - U And Ur Hand.wav
015-Rihanna-Rude Boy.wav
1960-018 Brenda Lee - Sweet Nothin's.wav
1988-048 Chicago - I Don't Wanna Live Without Your Love.wav
437. Ice Cube - A Bird In The Hand.wav
011- Alexia - Summer is Crazy.wav
004. Billie Eilish - bad guy.wav
```

<h2> Online inference </h2>

The script `online_inference.py` tests the model in a real scenario where music is played through a source, captured by the microphone and processed by the model to yield the desired result by employing a similarity search on a faiss index. The arguments of the script are the following:

- `-id`: The faiss index to use.
- `-j`: The json file containing the correspondence between filnames and vector representations on faiss index.
- `-d`: The query duration (in secs.)
- `-s`: The search type to be used (either majority_vote or sequence_search).
- `-pb`: The number of probes.
- `-kn`: Number of nearest neighbors to retrieve for each query segment.
- `-td`: The pkl file of threshold densities.
- `-t`: The threshold to us to check if a song exists or not in db.
- `-dv`: Whether to run on cuda or cpu.

**Note** To execute the above scripts use absolute paths for the arguments.