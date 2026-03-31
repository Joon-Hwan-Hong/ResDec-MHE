#!/bin/bash
# Launch 5-fold cross-attention fusion training across 2 GPUs.
# Runs 2 folds in parallel (one per GPU), then the remaining fold.

set -e

export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

CONFIG="configs/production_5fold_rank3_xattn.yaml"
SPLITS="outputs/splits.json"
PRECOMPUTED="data/precomputed/rosmap"
LOG_DIR="outputs/logs/xattn_fusion"

mkdir -p "$LOG_DIR"

run_fold() {
    local FOLD=$1
    local GPU=$2
    echo "$(date): Starting fold $FOLD on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/training/train.py \
        --config "$CONFIG" \
        --fold "$FOLD" \
        --splits-path "$SPLITS" \
        --precomputed-dir "$PRECOMPUTED" \
        > "$LOG_DIR/xattn_fusion_fold${FOLD}.log" 2>&1
    echo "$(date): Fold $FOLD complete (exit code: $?)"
}

# Batch 1: folds 0,1 on GPU 0,1
run_fold 0 0 &
run_fold 1 1 &
wait
echo "$(date): Batch 1 (folds 0,1) complete"

# Batch 2: folds 2,3 on GPU 0,1
run_fold 2 0 &
run_fold 3 1 &
wait
echo "$(date): Batch 2 (folds 2,3) complete"

# Batch 3: fold 4 on GPU 0
run_fold 4 0 &
wait
echo "$(date): Batch 3 (fold 4) complete"

echo "$(date): All 5 folds complete"
