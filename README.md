# CS229 - Concept Discovery
This project's code is based on the code of [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://github.com/google-deepmind/searchless_chess) by DeepMind. The code is modified and is shared under the `searchless_chess` directory. There are two main modifications:
1. ``Activation Recording``: The provided model is modified to log the activations at each layer for a given chess board FEN string and the candidate move. The activations of the best move at a given board state are saved to be used for the concept discovery.
2. ``Concept Discovery``: The activations and the corresponding concept labels for a board state are used to train a classifier to predict the concept labels. The classifier takes the activations as its input and predicts the concept labels.
3. ``Data Preprocessing``: The data preprocessing scripts (.ipynb files) are provided, which were used to preprocess the datasets for the activation recording and the concept discovery.
## Activation Recording
### Requirements
To record the activations of the model, the following steps are required (which are the slight modifications to the ones provided in the original repository (refer to the README file under the `searchless_chess` directory)):
1. Clone the repository:
```bash
git clone
```
1. Navigate to the `searchless_chess` directory:
```bash
cd searchless_chess
```
1. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
1. Download the model checkpoints:
```bash
cd checkpoints
bash ./download.sh
cd ..
```
### Running the Activation Recording
Run the activation recording script for a given dataset. The possible datasets are listed under the `data/concept_data` directory with the names `lichess_puzzles_openings.csv`, `stockfish_boolean_concepts_primary.csv`, `sts_all_concepts.csv`. Here are the commands to run the activation recording script for these datasets (you should replace the `<Absolute Path>` with the absolute path of the repository):
#### ``lichess_puzzles_openings`` Dataset
```bash
npy_dir='<Absolute Path>/searchless_chess/data/concept_data/npys/opening/'
rm -rf $npy_dir
mkdir $npy_dir
cd src
python3 activation_recorder.py \
    --checkpoint_dir <Absolute Path>/searchless_chess/checkpoints/270M \
    --csv <Absolute Path>/CS229/searchless_chess/data/concept_data/lichess_puzzles_openings.csv \
    --npy_dir $npy_dir \
    --csv_suffix _activations \
    --batch_size 32 \
    --position_key FEN \
    --num_per_label 300 \
    --last_cols_for_concept 7 \
    --log_all_sequence True \
    --save_step_count 1
cd ..
```
#### ``stockfish_boolean_concepts_primary`` Dataset
```bash
npy_dir='<Absolute Path>/searchless_chess/data/concept_data/npys/stockfish/'
rm -rf $npy_dir
mkdir $npy_dir
cd src
python3 activation_recorder.py \
    --checkpoint_dir <Absolute Path>/searchless_chess/checkpoints/270M \
    --csv <Absolute Path>/CS229/searchless_chess/data/concept_data/stockfish_boolean_concepts_primary.csv \
    --npy_dir $npy_dir \
    --csv_suffix _activations \
    --batch_size 32 \
    --position_key FEN \
    --num_per_label 300 \
    --num_per_anti_label 300 \
    --last_cols_for_concept 10 \
    --log_all_sequence True \
    --save_step_count 1
cd ..
```
#### ``sts_all_concepts`` Dataset
```bash
npy_dir='<Absolute Path>/searchless_chess/data/concept_data/npys/sts_all/'
rm -rf $npy_dir
mkdir $npy_dir
cd src
python3 activation_recorder.py \
    --checkpoint_dir <Absolute Path>/searchless_chess/checkpoints/270M \
    --csv <Absolute Path>/searchless_chess/data/concept_data/sts_all_concepts.csv \
    --npy_dir $npy_dir \
    --csv_suffix _activations \
    --batch_size 32 \
    --log_all_sequence True \
    --save_step_count 1 \
    --last_cols_for_concept 15 \
    --position_key Position
cd ..
```
### Generated Output
Under the `concept_data` directory, this generates the following:
- ``*_activations.csv``: The metadata file that contains the board state FEN string, best move at the board, concept labels and the pointer to the recorded activations. Examples can be found under this directory.
- ``npys/*/``: The directory that contains the activations for each board state. The activations are saved in the `.npy` format.
### Running Input Token Recording
Run the recording of the input tokens for a given dataset. This is for the concept discovery experiments where no learnable transformations are applied to the input tokens by the model, to assess the usefulness of the model representations for the concept modeling. Likewise, it could be run for the abovementioned datasets as follows:
#### ``lichess_puzzles_openings`` Dataset
```bash
npy_dir='<Absolute Path>/searchless_chess/data/concept_data/npys/opening_input_tokens/'
rm -rf $npy_dir
mkdir $npy_dir
cd src
python3 activation_recorder.py \
    --checkpoint_dir /content/drive/Shareddrives/CS229/searchless_chess/checkpoints/270M \
    --csv /content/drive/Shareddrives/CS229/searchless_chess/data/concept_data/lichess_puzzles_openings.csv \
    --npy_dir $npy_dir \
    --csv_suffix  _input_tokens \
    --batch_size 32 \
    --position_key FEN \
    --num_per_label 300 \
    --last_cols_for_concept 7 \
    --log_all_sequence True \
    --save_step_count 1 \
    --log_only_input True
cd ..
```
#### ``stockfish_boolean_concepts_primary`` Dataset
```bash
npy_dir='<Absolute Path>/searchless_chess/data/concept_data/npys/stockfish_input_tokens/'
rm -rf $npy_dir
mkdir $npy_dir
cd src
python3 activation_recorder.py \
    --checkpoint_dir <Absolute Path>/searchless_chess/checkpoints/270M \
    --csv <Absolute Path>/searchless_chess/data/concept_data/stockfish_boolean_concepts_primary.csv \
    --npy_dir $npy_dir \
    --csv_suffix  _input_tokens \
    --batch_size 32 \
    --position_key FEN \
    --num_per_label 300 \
    --num_per_anti_label 300 \
    --last_cols_for_concept 10 \
    --log_all_sequence True \
    --save_step_count 1 \
    --log_only_input True
cd ..
```
#### ``sts_all_concepts`` Dataset
```bash
npy_dir='<Absolute Path>/searchless_chess/data/concept_data/npys/sts_all_input_tokens/'
rm -rf $npy_dir
mkdir $npy_dir
cd src
python3 activation_recorder.py \
    --checkpoint_dir <Absolute Path>/searchless_chess/checkpoints/270M \
    --csv <Absolute Path>/searchless_chess/data/concept_data/sts_all_concepts.csv \
    --npy_dir $npy_dir \
    --csv_suffix _input_tokens \
    --batch_size 32 \
    --log_all_sequence True \
    --save_step_count 1 \
    --log_only_input True \
    --last_cols_for_concept 15 \
    --position_key Position
cd ..
```
### Generated Output
Under the `concept_data` directory, this generates the following:
- ``*_input_tokens.csv``: The metadata file that contains the board state FEN string, best move at the board, concept labels and the pointer to the recorded input tokens. Examples can be found under this directory.
- ``npys/*/``: The directory that contains the input tokens for each board state. The input tokens are saved in the `.npy` format.

## Concept Discovery
### Requirements
To train the classifier for the concept discovery, the following steps are required (independent of the activation recording):
1. Clone the repository:
```bash
git clone
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
### Running the Concept Discovery
Run the concept discovery script `concept_discovery.py` with the following arguments:
- `--csv_file_path`: The path to the metadata file generated by the activation recording script.
- `--concept_name`: The name of the concept to be discovered.
- `--layer_idx`: The layer index of the model (to which the activations belong) to be used for the concept discovery.
- `--seq_type`: Used to distinguish if the activations or the input tokens are used for the concept discovery. The possible values are `activations` and `input`.
- `--model_idx`: The classifier model to be trained for the concept discovery. THe available models, indexed from 0 to 2, are: `Logistic Regression`, `Min Concept Vector`, and `All Sequence NN`. 

Here are some example scripts:
```bash
python3 concept_discovery.py \
    --csv_file_path <Absolute Path>/searchless_chess/data/concept_data/lichess_puzzles_openings_activations.csv \
    --concept_name sicilian \
    --layer_idx 2 \
    --seq_type activations \
    --model_idx 0
```
```bash
python3 concept_discovery.py \
    --csv_file_path <Absolute Path>/searchless_chess/data/concept_data/stockfish_boolean_concepts_primary_activations.csv \
    --concept_name white_rook_trapped \
    --layer_idx 15 \
    --seq_type input \
    --model_idx 2
```

### Generated Output
The script outputs the achieved accuracy and the loss of the classifier across 3 times of 6-fold cross-validation run for 100 epochs. 

## Data Preprocessing
The data preprocessing scripts are provided in the form of Jupyter notebooks. The notebooks are used to preprocess the datasets `lichess_puzzles_openings`, `stockfish_boolean_concepts_primary`, and `sts_all_concepts`, and generate the `.csv` files that are used for the activation recording and the concept discovery. The notebooks per dataset are provided under the `data_preprocessing` directory. 