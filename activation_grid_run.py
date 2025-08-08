#!/usr/bin/env python3
import itertools, os, subprocess, time
from pathlib import Path

# ─────────────────────────── PARAMS ──────────────────────────── #
ABS = Path.cwd()  # assume repo root == current dir

TRAIN_CSVS_1 = [
    (ABS / "searchless_chess/data/concept_data/fischer_random_fen_activations.csv", "fischer_random"),
    (None, None),
]

TRAIN_CSVS_2 = [
    (ABS / "searchless_chess/data/concept_data/sts_all_concepts_activations.csv", "sts"),
    (None, None),
]

TEST_CSVS = [
    (None, None),
    (ABS / "searchless_chess/data/concept_data/fischer_random_fen_activations.csv", "fischer_random"),
    (ABS / "searchless_chess/data/concept_data/sts_all_concepts_activations.csv", "sts"),
]

#Concepts: Open Files and Diagonals,Knight Outposts,Advancement of f/g/h pawns,Advancement of a/b/c Pawns,Center Control,Pawn Play in the Center
CONCEPTS   = ["Open Files and Diagonals", "Knight Outposts",
             "Advancement of f/g/h pawns", "Advancement of a/b/c Pawns",
             "Center Control", "Pawn Play in the Center"]
LAYER_IDS  = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
SEQ_TYPES  = ["activations"]
MODEL_IDXS = [0,1,2]           # 0=LR, 1=MinConceptVector, 2=AllSeqNN

GPUS             = [3,4,5,6]
MAX_JOBS_PER_GPU = 5
#NUM_ITS          = 3       # forwarded to v2 script
# ──────────────────────────────────────────────────────────────── #

SCRIPT = ABS / "concept_discovery_v2.py"
LOGDIR = ABS / "concept_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)


def build_cmd(gpu, train_pairs, test_pairs,
              concept, layer, seq_type, model_idx):
    """Return (command_list, log_path)."""

    # ── flatten to unique paths & collect names ───────────────────
    train_paths, train_names = [], []
    for p, n in train_pairs:
        if p is not None and str(p) not in train_paths:
            train_paths.append(str(p))
            train_names.append(n or Path(p).stem)

    test_paths, test_names = [], []
    for p, n in test_pairs:
        if p is not None:
            test_paths.append(str(p))
            test_names.append(n or Path(p).stem)

    # ── build log filename ────────────────────────────────────────
    datasets_tag = f"train-{'+' .join(train_names)}"
    if test_names:
        datasets_tag += f"__test-{'+' .join(test_names)}"

    log_name = (
        f"{datasets_tag}__{concept}_L{layer}_{seq_type}_M{model_idx}_gpu{gpu}.log"
    )
    log_path = LOGDIR / log_name

    # ── command list ──────────────────────────────────────────────
    cmd = ["nohup", "python3", str(SCRIPT),
           "--train_csv_paths", *train_paths,
           "--concept_name", concept,
           "--seq_type", seq_type,
           "--model_idx", str(model_idx),
           "--redundant_train_holdout_frac", "0.5",
           #"--num_its", str(NUM_ITS)
    ]
    if test_paths:
        cmd += ["--test_csv_paths", *test_paths]
    if layer is not None:
        cmd += ["--layer_idx", str(layer)]

    return cmd, log_path


def launch_job(cmd, log_path, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_f,
                            stderr=subprocess.STDOUT, env=env)


def main():
    param_grid = itertools.product(
        TRAIN_CSVS_1, TRAIN_CSVS_2,
        CONCEPTS, LAYER_IDS, SEQ_TYPES, MODEL_IDXS
    )

    gpu_slots = {g: [] for g in GPUS}

    for pair1, pair2, concept, layer, seq_type, model_idx in param_grid:
        # build final training/test pairs lists after None-filtering
        train_pairs = [pair for pair in (pair1, pair2) if pair[0] is not None]
        if not train_pairs:       # both None → skip
            continue
        test_pairs = [pair for pair in TEST_CSVS if pair[0] is not None]

        # wait until some GPU has a free slot
        while True:
            for gpu, procs in gpu_slots.items():
                gpu_slots[gpu] = [p for p in procs if p.poll() is None]
                if len(gpu_slots[gpu]) < MAX_JOBS_PER_GPU:
                    cmd, log_path = build_cmd(
                        gpu, train_pairs, test_pairs,
                        concept, layer, seq_type, model_idx
                    )
                    proc = launch_job(cmd, log_path, gpu)
                    gpu_slots[gpu].append(proc)
                    print(f"[GPU {gpu}] launched {' '.join(cmd)}")
                    break
            else:
                time.sleep(10)
                continue  # GPUs still busy
            break         # job launched → proceed

    # wait for all jobs to finish
    for procs in gpu_slots.values():
        for p in procs:
            p.wait()

    print("All jobs finished.")


if __name__ == "__main__":
    main()
