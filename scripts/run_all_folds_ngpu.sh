#!/usr/bin/env bash
# Run K CV folds using all available GPUs per fold (DDP).
# Folds run sequentially; each fold uses all N GPUs via DDP.
#
# Usage: bash scripts/run_all_folds_ngpu.sh [N_GPUS] [N_FOLDS]
#   N_GPUS  - number of GPUs to use per fold (default: auto-detect all)
#   N_FOLDS - number of folds (default: 5)
#
# TensorBoard: auto-launches on port 6006 (--no-tensorboard to disable)
#   Open http://localhost:6006 in your browser
set -euo pipefail

# --- Parse flags ---
USE_TB=true
TB_PORT=6006
for arg in "$@"; do
    case "$arg" in
        --no-tensorboard) USE_TB=false; shift ;;
        --tb-port=*) TB_PORT="${arg#*=}"; shift ;;
    esac
done

N_GPUS=${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_FOLDS=${2:-5}

if [ "$N_GPUS" -lt 1 ]; then
    echo "Error: no GPUs detected" >&2; exit 1
fi

LOGDIR="outputs/logs/fold_runs"
mkdir -p "$LOGDIR"

COMMON_ARGS=(
    --config configs/default.yaml
    --splits-path outputs/splits.json
    --precomputed-dir data/precomputed/rosmap/
    training.devices="$N_GPUS"
)

# --- TensorBoard ---
# Event files land under outputs/<exp_hash>/logs/tensorboard/, so scan from outputs/.
TB_PID=""
if $USE_TB; then
    uv run tensorboard --logdir outputs/ --bind_all --port "$TB_PORT" &>/dev/null &
    TB_PID=$!
    echo "  TensorBoard: http://localhost:$TB_PORT (pid $TB_PID)"
fi
trap '[ -n "$TB_PID" ] && kill $TB_PID 2>/dev/null' EXIT

# --- Run folds sequentially, each using all GPUs via DDP ---
echo "=== $N_FOLDS-fold CV, $N_GPUS GPU(s) per fold (DDP) at $(date) ==="

FAIL=0
for fold in $(seq 0 $((N_FOLDS - 1))); do
    logfile="$LOGDIR/fold_${fold}.log"
    echo "  Fold $fold -> $N_GPUS GPU(s) DDP (log: $logfile)"

    uv run python scripts/train.py \
        "${COMMON_ARGS[@]}" --fold "$fold" \
        2>&1 | tee "$logfile" || {
        echo "!!! Fold $fold FAILED !!!" >&2; FAIL=1
        continue
    }
done

if [ "$FAIL" -ne 0 ]; then
    echo "=== Some folds failed. Check logs in $LOGDIR/ ===" >&2; exit 1
fi

echo "=== All $N_FOLDS folds completed at $(date) ==="
