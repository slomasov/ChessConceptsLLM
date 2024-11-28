import os
import sys
sys.path.append('../..')

# Standard libraries
import functools

# JAX and Numpy
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn
import numpy as np
import scipy.special

# Haiku
import haiku as hk

# Chess library
import chess

# searchless_chess package imports
from searchless_chess.src import constants, tokenizer, utils, training_utils
from searchless_chess.src.engines import engine
from searchless_chess.src.engines.neural_engines import NeuralEngine
from searchless_chess.src.transformer import (
    TransformerConfig,
    PositionalEncodings,
    shift_right,
    embed_sequences,
    layer_norm,
    _attention_block,
    _mlp_block,
)

import argparse
import pandas as pd

from tqdm import tqdm

def transformer_decoder_with_intermediate(
    targets: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
    """Returns the transformer decoder output, shape [B, T, V].

    Follows the LLaMa architecture:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py
    Main changes to the original Transformer decoder:
    - Using gating in the MLP block, with SwiGLU activation function.
    - Using normalization before the attention and MLP blocks.

    Args:
        targets: The integer target values, shape [B, T].
        config: The config to use for the transformer.
    """
    # Right shift the targets to get the inputs (the first token is now a 0).
    inputs = shift_right(targets)
    
    # Embeds the inputs and adds positional encodings.
    embeddings = embed_sequences(inputs, config)
    h = embeddings  # [B, T, D]
    
    # List to store the outputs of each layer
    layer_outputs = []

    for _ in range(config.num_layers):
        # record the output of each layer
        layer_outputs.append(jnp.expand_dims(h, axis=1))  # [B, 1, T, D]
        
        attention_input = layer_norm(h)
        attention = _attention_block(attention_input, config)
        h += attention

        mlp_input = layer_norm(h)
        mlp_output = _mlp_block(mlp_input, config)
        h += mlp_output

    if config.apply_post_ln:
        h = layer_norm(h)
    layer_outputs.append(jnp.expand_dims(h, axis=1))  # [B, 1, T, D]
    logits = hk.Linear(config.output_size)(h)
    
    layer_outputs = jnp.concatenate(layer_outputs, axis=1)
    assert len(layer_outputs.shape) == 4, f"Expected 4D output, got {layer_outputs.shape}"

    probs = jnn.log_softmax(logits, axis=-1)
    return layer_outputs, probs


def build_transformer_predictor_with_intermediate(
    config: TransformerConfig,
) -> constants.Predictor:
    """Returns a transformer predictor."""
    model = hk.transform(functools.partial(transformer_decoder_with_intermediate, config=config))
    return constants.Predictor(initial_params=model.init, predict=model.apply)


def _update_scores_with_repetitions(
    board: chess.Board,
    scores: np.ndarray,
) -> None:
    """Updates the win-probabilities for a board given possible repetitions."""
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    for i, move in enumerate(sorted_legal_moves):
        board.push(move)
        # If the move results in a draw, associate 50% win prob to it.
        if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
            scores[i] = 0.5
        board.pop()


class ActionValueDebugEngine(NeuralEngine):
    """Neural engine using a function P(r | s, a)."""

    def analyse(self, board: chess.Board):
        """Returns buckets log-probs for each action, and FEN."""
        # Tokenize the legal actions.
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        legal_actions = [utils.MOVE_TO_ACTION[x.uci()] for x in sorted_legal_moves]
        legal_actions = np.array(legal_actions, dtype=np.int32)
        legal_actions = np.expand_dims(legal_actions, axis=-1)
        # Tokenize the return buckets.
        dummy_return_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)
        # Tokenize the board.
        tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
        sequences = np.stack([tokenized_fen] * len(legal_actions))
        # Create the sequences.
        sequences = np.concatenate(
            [sequences, legal_actions, dummy_return_buckets],
            axis=1,
        )  # [(M)oves x (S)equence Length]
        layer_outputs, log_probs = self.predict_fn(sequences)  # [M x L x S x E], [M x S x V]
        
        if not self.log_all_sequence:
            layer_outputs = layer_outputs[:, :, -1],  # [M x L x E]
        
        return {
            'layer_outputs': layer_outputs,  # [M x L (x ?) x E]
            'log_probs': log_probs[:, -1],  # [M x V]
            'fen': board.fen(),
        }
        
    def play(self, board: chess.Board):
        analysis = self.analyse(board)
        return_buckets_log_probs = self.analyse(board)['log_probs']
        return_buckets_probs = np.exp(return_buckets_log_probs)
        win_probs = np.inner(return_buckets_probs, self._return_buckets_values)
        _update_scores_with_repetitions(board, win_probs)
        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        if self.temperature is not None:
            probs = scipy.special.softmax(win_probs / self.temperature, axis=-1)
            best_index = self._rng.choice(np.arange(len(sorted_legal_moves)), p=probs)
        else:
            best_index = np.argmax(win_probs)
        return sorted_legal_moves[best_index], analysis['layer_outputs'][best_index]  # .., [L (x ?) x E]


def my_wrap_predict_fn(
    predictor: constants.Predictor,
    params: hk.Params,
    batch_size: int = 32,
):
    """Returns a simple prediction function from a predictor and parameters.

    Args:
        predictor: Used to predict outputs.
        params: Neural network parameters.
        batch_size: How many sequences to pass to the predictor at once.
    """
    jitted_predict_fn = jax.jit(predictor.predict)

    def fixed_predict_fn(sequences: np.ndarray) -> np.ndarray:
        """Wrapper around the predictor `predict` function."""
        assert sequences.shape[0] == batch_size
        return jitted_predict_fn(
            params=params,
            targets=sequences,
            rng=None,
        )

    def predict_fn(sequences: np.ndarray) -> np.ndarray:
        """Wrapper to collate batches of sequences of fixed size."""
        # sequences: [M x S]
        remainder = -len(sequences) % batch_size
        padded = np.pad(sequences, ((0, remainder), (0, 0)))  # [(M + R) x S]
        sequences_split = np.split(padded, len(padded) // batch_size)  # [(M + R) / B x B x S]
        all_layer_outputs, all_probs = [], []
        for sub_sequences in sequences_split:
            # sub_sequences: [B x S]
            layer_outputs, probs = fixed_predict_fn(sub_sequences)  # layer_outputs: [B x L x S x E], probs: [B x S x V]
            all_layer_outputs.append(layer_outputs)
            all_probs.append(probs)
        layer_outputs = np.concatenate(all_layer_outputs, axis=0)  # [(M + R) x L x S x E]
        probs = np.concatenate(all_probs, axis=0)  # [(M + R) x S x V]
        assert len(probs) == len(padded)
        assert len(layer_outputs) == len(padded)
        return layer_outputs[: len(sequences)], probs[: len(sequences)]  # [M x L x S x E], [M x S x V]

    return predict_fn


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='CSV file to read FENs from')
    # Model
    parser.add_argument('--checkpoint_dir', type=str, default='/home/mhamza/searchless_chess/checkpoints/270M', help='Directory to load the model from')
    parser.add_argument('--step', type=int, default=6400000, help='Step to load the model from')
    parser.add_argument('--policy', type=str, default='action_value', help='Policy to use')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of layers in the transformer')
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Dimension of the embeddings')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the transformer')
    parser.add_argument('--num_return_buckets', type=int, default=128, help='Number of return buckets')

    parser.add_argument('--num_data_points', type=int, default=-1, help="How many datapoints to be utilized while labeling the dataset")
    parser.add_argument("--num_per_label", type=int, default=-1, help="Number of data points to label per label")
    parser.add_argument('--position_key', type=str, default='Position', help="Key for the position column in the CSV file")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for the predictor")
    parser.add_argument('--last_cols_for_concept', type=int, default=-1, help="Number of last columns to consider for the concept")
    parser.add_argument('--log_all_sequence', type=str, default='False', help="Log all the sequence each layer or not")

    args = parser.parse_args()
    output_size = args.num_return_buckets

    predictor_config = TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=output_size,
        pos_encodings=PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor = build_transformer_predictor_with_intermediate(config=predictor_config)

    _, return_buckets_values = utils.get_uniform_buckets_edges_values(
        args.num_return_buckets
    )

    params = training_utils.load_parameters(
        checkpoint_dir=args.checkpoint_dir,
        params=predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        ),
        step=args.step,
    )

    if args.policy == 'action_value':
        play_engine = ActionValueDebugEngine(
            return_buckets_values=return_buckets_values,
            predict_fn=my_wrap_predict_fn(
                predictor=predictor,
                params=params,
                batch_size=args.batch_size,
            ), 
        )
        play_engine.log_all_sequence = bool(args.log_all_sequence)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Load and process the DataFrame
    df = pd.read_csv(args.csv)
    if args.num_data_points != -1:
        # shuffle the rows
        df = df.sample(frac=1)
        df = df.head(args.num_data_points)

    all_outputs = []

    if args.num_per_label != -1:
        # traverse all the last last_cols_for_concept columns, and for each of them, retrieve samples of count num_per_label (retrieve if that column is set to 1)
        df_subsets = []
        for i in range(len(df) - 1, len(df) - args.last_cols_for_concept - 1, -1):
            # now retrieve all the rows where the ith column is set to 1 (make sure to stringify 1 so that it matches the string in the csv)
            df_subset = df[df.iloc[:, i].astype(str) == '1']
            # assert the size is not less than num_per_label
            assert len(df_subset) >= args.num_per_label
            # now sample num_per_label rows from this subset
            df_subset = df_subset.sample(n=args.num_per_label)
            df_subsets.append(df_subset)
        df = pd.concat(df_subsets)

    # Process each position and track progress with tqdm
    for i in tqdm(range(len(df)), desc="Processing Positions", position=0, leave=True):
        board = chess.Board(df[args.position_key][i])
        outputs = play_engine.play(board)
        all_outputs.append(outputs)

    # Prepare the DataFrame by adding 'Move' and 'Layer Outputs' columns
    df['Move'] = [x[0].uci() for x in all_outputs]

    # Concatenate all layer output arrays and store starting indices
    layer_outputs = [x[1] for x in all_outputs]
    layer_outputs_stacked = np.stack(layer_outputs, axis=0)  # Combine all arrays
    start_indices = np.arange(0, len(layer_outputs))

    # Define the output path based on args.csv with '_layer_outputs.npy' suffix
    layer_output_file = args.csv.replace('.csv', '_layer_outputs.npy')
    np.save(layer_output_file, layer_outputs_stacked)

    # Add start indices to the DataFrame and metadata for array file path
    df['Layer Outputs Index'] = start_indices  # Stores the start index of each row's array
    df['Layer Outputs File'] = layer_output_file  # Adds file path as a column for reference

    # Save the modified DataFrame as CSV
    df.to_csv(args.csv.replace('.csv', '_with_activation.csv'), index=False)

if __name__ == '__main__':
    __main__()