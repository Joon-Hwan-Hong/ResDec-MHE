#!/usr/bin/env bash
# Run repeated stratified K-fold CV with different random seeds.
# Measures how much fold assignment affects reported metrics at small sample sizes.
#
# Usage:
#   bash scripts/training/run_fold_variance.sh --config configs/rank03.yaml --seeds 42,43,44,45,46
#   bash scripts/training/run_fold_variance.sh --config configs/rank03.yaml --seeds 42,43,44 --n-gpus 1
#   setsid nohup bash scripts/training/run_fold_variance.sh --config configs/rank03.yaml \
#       --seeds 42,43,44,45,46 > outputs/logs/fold_variance.log 2>&1 &
set -euo pipefail
export PYTHONHASHSEED=42

# ── Defaults ──────────────────────────────────────────────────────────────────
N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
N_FOLDS=5
SPLITS_TEMPLATE="outputs/splits_seed{SEED}.json"
PRECOMPUTED="data/precomputed/"
BASE_LOGDIR="outputs/logs"

# ── Parse flags ───────────────────────────────────────────────────────────────
CONFIG=""
SEEDS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --seeds)        SEEDS="$2";        shift 2 ;;
        --n-gpus)       N_GPUS="$2";       shift 2 ;;
        --n-folds)      N_FOLDS="$2";      shift 2 ;;
        --precomputed)  PRECOMPUTED="$2";  shift 2 ;;
        --logdir)       BASE_LOGDIR="$2";  shift 2 ;;
        -*)             echo "Unknown flag: $1"; exit 1 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ] || [ -z "$SEEDS" ]; then
    echo "Usage: $0 --config <config.yaml> --seeds <42,43,44,...> [--n-gpus N] [--n-folds N]"
    exit 1
fi

# Parse comma-separated seeds
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
N_SEEDS=${#SEED_ARRAY[@]}
TOTAL=$((N_SEEDS * N_FOLDS))
CONFIG_NAME=$(basename "$CONFIG" .yaml)

echo "=== Fold Variance Study: $N_SEEDS seeds × $N_FOLDS folds = $TOTAL runs ==="
echo "Config: $CONFIG"
echo "Seeds: ${SEEDS}"
echo "GPUs: $N_GPUS"
echo "Started at $(date)"
echo ""

FAILED=0

for SEED in "${SEED_ARRAY[@]}"; do
    SPLITS="outputs/splits_seed${SEED}.json"
    LOGDIR="${BASE_LOGDIR}/sensitivity_seed${SEED}"
    mkdir -p "$LOGDIR"

    # Create splits file if it doesn't exist
    if [ ! -f "$SPLITS" ]; then
        echo "Creating splits for seed $SEED..."
        uv run python scripts/data/create_splits.py \
            --config configs/default.yaml \
            --output "$SPLITS" \
            --test-frac 0.0 \
            --seed "$SEED" \
            --precomputed-dir "$PRECOMPUTED" 2>/dev/null
    fi

    echo "--- Seed $SEED ---"

    fold=0
    while [ "$fold" -lt "$N_FOLDS" ]; do
        PIDS=()
        FOLD_MAP=()
        for gpu in $(seq 0 $((N_GPUS - 1))); do
            if [ "$fold" -ge "$N_FOLDS" ]; then break; fi
            logfile="$LOGDIR/${CONFIG_NAME}_fold${fold}.log"
            echo "[$(date '+%H:%M:%S')] seed=$SEED fold=$fold -> GPU $gpu"

            CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/training/train.py \
                --config "$CONFIG" \
                --splits-path "$SPLITS" \
                --precomputed-dir "$PRECOMPUTED" \
                --fold "$fold" \
                training.devices=1 \
                training.strategy=auto \
                > "$logfile" 2>&1 &
            PIDS+=($!)
            FOLD_MAP+=("seed${SEED}:fold${fold}:pid=$!")
            fold=$((fold + 1))
        done

        for i in "${!PIDS[@]}"; do
            if ! wait "${PIDS[$i]}"; then
                FAILED=$((FAILED + 1))
                echo "  FAILED: ${FOLD_MAP[$i]}"
            fi
        done
    done
    echo ""
done

echo "=== Complete at $(date) ==="
echo "Total: $TOTAL runs, Failed: $FAILED"
if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
