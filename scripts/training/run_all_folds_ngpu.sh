#!/usr/bin/env bash
# Run K-fold CV across N GPUs in two modes:
#
#   DDP mode (default):      folds run sequentially, each fold uses all N GPUs via DDP
#   Parallel-folds mode:     folds run concurrently (N at a time), 1 GPU per fold
#
# Usage:
#   bash scripts/training/run_all_folds_ngpu.sh [OPTIONS] [-- OVERRIDES...]
#
# Options:
#   --n-gpus N            Number of GPUs (default: auto-detect)
#   --n-folds N           Number of folds (default: 5)
#   --config PATH         Config file (default: configs/default.yaml)
#   --splits-path PATH    Splits JSON (default: outputs/splits.json)
#   --precomputed-dir P   Precomputed features dir (default: data/precomputed/rosmap/)
#   --parallel-folds      Run folds concurrently, 1 GPU each (instead of DDP)
#   --no-tensorboard      Don't auto-launch TensorBoard
#   --tb-port PORT        TensorBoard port (default: 6006)
#
# Overrides (after --):
#   Any OmegaConf dotlist overrides passed to train.py, e.g.:
#     -- training.max_epochs=100 training.strategy=auto
#
# Examples:
#   # DDP: 5 folds sequentially, 2 GPUs per fold
#   bash scripts/training/run_all_folds_ngpu.sh --n-gpus 2 --n-folds 5
#
#   # Parallel: 5 folds, 2 at a time on 2 GPUs, custom config
#   bash scripts/training/run_all_folds_ngpu.sh --parallel-folds --config outputs/best_config.yaml \
#       -- training.max_epochs=100
#
#   # Session-persistent (survives logout):
#   setsid nohup bash scripts/training/run_all_folds_ngpu.sh --parallel-folds \
#       > outputs/logs/5fold.log 2>&1 &
set -euo pipefail

# Reproducibility: PYTHONHASHSEED must be set before Python starts.
# Controls hash randomization for dicts/sets — affects iteration order in
# data pipelines and config serialization.
export PYTHONHASHSEED=42

# --- Defaults ---
N_GPUS=""
N_FOLDS=5
CONFIG="configs/default.yaml"
SPLITS="outputs/splits.json"
PRECOMPUTED="data/precomputed/rosmap/"
PARALLEL=false
USE_TB=true
TB_PORT=6006
OVERRIDES=()

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-gpus)       N_GPUS="$2"; shift 2 ;;
        --n-folds)      N_FOLDS="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --splits-path)  SPLITS="$2"; shift 2 ;;
        --precomputed-dir) PRECOMPUTED="$2"; shift 2 ;;
        --parallel-folds) PARALLEL=true; shift ;;
        --no-tensorboard) USE_TB=false; shift ;;
        --tb-port)      TB_PORT="$2"; shift 2 ;;
        --)             shift; OVERRIDES=("$@"); break ;;
        *)              echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Auto-detect GPUs if not specified
if [ -z "$N_GPUS" ]; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
if [ "$N_GPUS" -lt 1 ]; then
    echo "Error: no GPUs detected" >&2; exit 1
fi

LOGDIR="outputs/logs/fold_runs"
mkdir -p "$LOGDIR"

# --- TensorBoard ---
TB_PID=""
if $USE_TB; then
    uv run tensorboard --logdir outputs/ --bind_all --port "$TB_PORT" &>/dev/null &
    TB_PID=$!
    echo "  TensorBoard: http://localhost:$TB_PORT (pid $TB_PID)"
fi

# Track all child PIDs for cleanup on exit
CHILD_PIDS=()
cleanup_children() {
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    [ -n "$TB_PID" ] && kill "$TB_PID" 2>/dev/null || true
}
trap cleanup_children EXIT

# --- Common train.py arguments ---
COMMON_ARGS=(
    --config "$CONFIG"
    --splits-path "$SPLITS"
    --precomputed-dir "$PRECOMPUTED"
)

run_fold() {
    local fold=$1
    local gpu=$2       # only used in parallel mode
    local logfile="$LOGDIR/fold_${fold}.log"

    if $PARALLEL; then
        echo "[$(date '+%H:%M:%S')] Fold $fold -> GPU $gpu (log: $logfile)"
        CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/training/train.py \
            "${COMMON_ARGS[@]}" --fold "$fold" \
            training.devices=1 training.strategy=auto \
            "${OVERRIDES[@]}" \
            > "$logfile" 2>&1
    else
        echo "[$(date '+%H:%M:%S')] Fold $fold -> $N_GPUS GPU(s) DDP (log: $logfile)"
        uv run python scripts/training/train.py \
            "${COMMON_ARGS[@]}" --fold "$fold" \
            training.devices="$N_GPUS" \
            "${OVERRIDES[@]}" \
            2>&1 | tee "$logfile"
    fi
}

if $PARALLEL; then
    echo "=== ${N_FOLDS}-fold CV, parallel folds ($N_GPUS at a time) at $(date) ==="
else
    echo "=== ${N_FOLDS}-fold CV, $N_GPUS GPU(s) per fold (DDP) at $(date) ==="
fi

FAIL=0

if $PARALLEL; then
    # Launch folds in batches of N_GPUS
    fold=0
    while [ "$fold" -lt "$N_FOLDS" ]; do
        PIDS=()
        FOLD_FOR_PID=()
        batch_end=$((fold + N_GPUS))
        if [ "$batch_end" -gt "$N_FOLDS" ]; then
            batch_end=$N_FOLDS
        fi

        echo "--- Batch: folds $fold..$((batch_end - 1)) ---"
        gpu=0
        for f in $(seq "$fold" $((batch_end - 1))); do
            run_fold "$f" "$gpu" &
            PIDS+=($!)
            FOLD_FOR_PID+=("$f")
            CHILD_PIDS+=($!)
            gpu=$((gpu + 1))
        done

        # Wait for this batch; track failures with fold identity
        for i in "${!PIDS[@]}"; do
            if ! wait "${PIDS[$i]}"; then
                echo "!!! Fold ${FOLD_FOR_PID[$i]} FAILED (pid ${PIDS[$i]}) !!!" >&2
                FAIL=1
            fi
        done

        fold=$batch_end
    done
else
    # Sequential DDP
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        run_fold "$fold" 0 || { echo "!!! Fold $fold FAILED !!!" >&2; FAIL=1; continue; }
    done
fi

if [ "$FAIL" -ne 0 ]; then
    echo "=== Some folds failed. Check logs in $LOGDIR/ ===" >&2; exit 1
fi

echo "=== All $N_FOLDS folds completed at $(date) ==="
