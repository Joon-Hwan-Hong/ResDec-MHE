#!/usr/bin/env bash
# Run production training across multiple configs and folds.
# Distributes folds across available GPUs in parallel.
#
# Usage:
#   bash scripts/training/run_sensitivity.sh configs/production_5fold_rank*.yaml
#   bash scripts/training/run_sensitivity.sh --n-gpus 1 --n-folds 3 configs/my_config.yaml
#   setsid nohup bash scripts/training/run_sensitivity.sh configs/production_5fold_rank*.yaml \
#       > outputs/logs/sensitivity.log 2>&1 &
set -euo pipefail
export PYTHONHASHSEED=42

# ── Defaults (overridable via flags) ─────────────────────────────────────────
N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_FOLDS=5
SPLITS="outputs/splits.json"
PRECOMPUTED="data/precomputed/"
LOGDIR="outputs/logs/sensitivity"

# ── Parse flags ──────────────────────────────────────────────────────────────
CONFIGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-gpus)    N_GPUS="$2";      shift 2 ;;
        --n-folds)   N_FOLDS="$2";     shift 2 ;;
        --splits)    SPLITS="$2";      shift 2 ;;
        --precomputed) PRECOMPUTED="$2"; shift 2 ;;
        --logdir)    LOGDIR="$2";      shift 2 ;;
        -*)          echo "Unknown flag: $1"; exit 1 ;;
        *)           CONFIGS+=("$1");  shift ;;
    esac
done

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Usage: $0 [--n-gpus N] [--n-folds N] config1.yaml [config2.yaml ...]"
    exit 1
fi

mkdir -p "$LOGDIR"

TOTAL=0
FAILED=0
FAIL_LIST=""
N_CONFIGS=${#CONFIGS[@]}
TOTAL_RUNS=$((N_CONFIGS * N_FOLDS))

echo "=== Production Runs: $N_CONFIGS configs × $N_FOLDS folds = $TOTAL_RUNS runs ==="
echo "GPUs: $N_GPUS, Splits: $SPLITS"
echo "Started at $(date)"
echo ""

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    echo "--- Config: $name ---"

    # Run folds in batches of N_GPUS
    fold=0
    while [ "$fold" -lt "$N_FOLDS" ]; do
        PIDS=()
        FOLD_MAP=()
        for gpu in $(seq 0 $((N_GPUS - 1))); do
            if [ "$fold" -ge "$N_FOLDS" ]; then
                break
            fi
            logfile="$LOGDIR/${name}_fold${fold}.log"
            echo "[$(date '+%H:%M:%S')] $name fold $fold -> GPU $gpu"

            CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/training/train.py \
                --config "$config" \
                --splits-path "$SPLITS" \
                --precomputed-dir "$PRECOMPUTED" \
                --fold "$fold" \
                training.devices=1 \
                training.strategy=auto \
                > "$logfile" 2>&1 &
            PIDS+=($!)
            FOLD_MAP+=("$name:fold${fold}:pid=$!")
            TOTAL=$((TOTAL + 1))
            fold=$((fold + 1))
        done

        # Wait for this batch to finish
        for i in "${!PIDS[@]}"; do
            if ! wait "${PIDS[$i]}"; then
                FAILED=$((FAILED + 1))
                FAIL_LIST="$FAIL_LIST ${FOLD_MAP[$i]}"
                echo "  FAILED: ${FOLD_MAP[$i]}"
            fi
        done
    done
    echo ""
done

echo "=== Complete at $(date) ==="
echo "Total: $TOTAL runs, Failed: $FAILED"
if [ "$FAILED" -gt 0 ]; then
    echo "Failed runs: $FAIL_LIST"
    exit 1
fi
