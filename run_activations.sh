#!/usr/bin/env bash
set -euo pipefail

ABSOLUTE_PATH="$(pwd)"

echo "Launching activation-recording jobs …"

###############################################################################
# lichess_puzzles_openings  →  GPU 0
###############################################################################
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
  npy_dir=\"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/npys/opening/\"
  rm -rf \"\$npy_dir\" && mkdir -p \"\$npy_dir\"

  cd searchless_chess/src
  python3 activation_recorder.py \
    --checkpoint_dir \"${ABSOLUTE_PATH}/searchless_chess/checkpoints/270M\" \
    --csv \"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/lichess_puzzles_openings.csv\" \
    --npy_dir \"\$npy_dir\" \
    --csv_suffix _input_tokens \
    --batch_size 32 \
    --position_key FEN \
    --num_per_label 300 \
    --last_cols_for_concept 7 \
    --log_all_sequence True \
    --save_step_count 1 \
    --log_only_input True \
  > \"${ABSOLUTE_PATH}/lichess_puzzles_openings.log\" 2>&1
" &


###############################################################################
# stockfish_boolean_concepts_primary  →  GPU 1
###############################################################################
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
  npy_dir=\"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/npys/stockfish/\"
  rm -rf \"\$npy_dir\" && mkdir -p \"\$npy_dir\"

  cd searchless_chess/src
  python3 activation_recorder.py \
    --checkpoint_dir \"${ABSOLUTE_PATH}/searchless_chess/checkpoints/270M\" \
    --csv \"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/stockfish_boolean_concepts_primary.csv\" \
    --npy_dir \"\$npy_dir\" \
    --csv_suffix _activations \
    --batch_size 32 \
    --position_key FEN \
    --num_per_label 300 \
    --num_per_anti_label 300 \
    --last_cols_for_concept 10 \
    --log_all_sequence True \
    --save_step_count 1 \
  > \"${ABSOLUTE_PATH}/stockfish_boolean_concepts_primary.log\" 2>&1
" &


###############################################################################
# sts_all_concepts  →  GPU 2
###############################################################################
CUDA_VISIBLE_DEVICES=2 nohup bash -c "
  npy_dir=\"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/npys/sts_all/\"
  rm -rf \"\$npy_dir\" && mkdir -p \"\$npy_dir\"

  cd searchless_chess/src
  python3 activation_recorder.py \
    --checkpoint_dir \"${ABSOLUTE_PATH}/searchless_chess/checkpoints/270M\" \
    --csv \"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/sts_all_concepts.csv\" \
    --npy_dir \"\$npy_dir\" \
    --csv_suffix _activations \
    --batch_size 32 \
    --log_all_sequence True \
    --save_step_count 1 \
    --last_cols_for_concept 15 \
    --position_key Position \
  > \"${ABSOLUTE_PATH}/sts_all_concepts.log\" 2>&1
" &


###############################################################################
# fischer_random  →  GPU 3
###############################################################################
CUDA_VISIBLE_DEVICES=3 nohup bash -c "
  npy_dir=\"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/npys/fischer_random/\"
  rm -rf \"\$npy_dir\" && mkdir -p \"\$npy_dir\"

  cd searchless_chess/src
  python3 activation_recorder.py \
    --checkpoint_dir \"${ABSOLUTE_PATH}/searchless_chess/checkpoints/270M\" \
    --csv \"${ABSOLUTE_PATH}/searchless_chess/data/concept_data/fischer_random_fen.csv\" \
    --npy_dir \"\$npy_dir\" \
    --csv_suffix _activations \
    --batch_size 32 \
    --log_all_sequence True \
    --save_step_count 1 \
    --last_cols_for_concept 1 \
    --position_key FEN \
  > \"${ABSOLUTE_PATH}/fischer_random.log\" 2>&1
" &

echo "Activation jobs are now running in the background (check the *.log files)."
