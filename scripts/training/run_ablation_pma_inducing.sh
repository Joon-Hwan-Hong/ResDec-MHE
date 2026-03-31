#!/usr/bin/env bash
# 3×3 ablation: PMA seeds {1, 2, 4} × inducing points {32, 64, 128}
# All conditions: no OGM-GE, no temp anneal, det=True, benchmark=False
# max_cells_per_type=2500
#
# Usage:
#   bash scripts/training/run_ablation_pma_inducing.sh [--fold N] [--n-gpus N]
#   setsid nohup bash scripts/training/run_ablation_pma_inducing.sh > outputs/logs/ablation_pma_inducing.log 2>&1 &
set -euo pipefail

FOLD=${1:-0}
N_GPUS=${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}
CONFIG="configs/production_5fold.yaml"
SPLITS="outputs/splits.json"
PRECOMPUTED="data/precomputed/rosmap/"
LOGDIR="outputs/logs/ablation_pma_inducing"
mkdir -p "$LOGDIR"

# Common: 30 epochs, no OGM, no anneal (warmup=999), det=True, 2500 max cells
COMMON=(
    --config "$CONFIG"
    --splits-path "$SPLITS"
    --precomputed-dir "$PRECOMPUTED"
    --fold "$FOLD"
    training.max_epochs=30
    training.temperature_annealing.warmup_epochs=999
    training.kl_annealing.warmup_epochs=2
    training.early_stopping.min_epochs=17
    training.early_stopping.patience=15
    training.devices=1
    training.strategy=auto
    training.gradient_modulation.enabled=false
    reproducibility.deterministic=true
    reproducibility.benchmark=false
    data.cell_sampling.max_cells_per_type=2500
)

run_condition() {
    local name=$1
    local gpu=$2
    local pma=$3
    local inducing=$4
    local logfile="$LOGDIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] $name (pma=$pma, ind=$inducing) -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/training/train.py \
        "${COMMON[@]}" \
        experiment.run_name="ablation_${name}" \
        model.set_transformer.n_pma_seeds="$pma" \
        model.set_transformer.n_inducing_points="$inducing" \
        > "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] $name DONE"
}

echo "=== 3×3 PMA × Inducing ablation on fold $FOLD at $(date) ==="
echo "  max_cells_per_type=2500, no OGM, no anneal, det=True"
echo "  GPUs: $N_GPUS"
echo ""

# 9 conditions in batches of N_GPUS
# PMA: 1, 2, 4   Inducing: 32, 64, 128

# --- Batch 1 ---
echo "--- Batch 1 ---"
run_condition "pma1_ind32"  0  1  32 &
run_condition "pma1_ind64"  1  1  64 &
wait
echo ""

# --- Batch 2 ---
echo "--- Batch 2 ---"
run_condition "pma1_ind128" 0  1 128 &
run_condition "pma2_ind32"  1  2  32 &
wait
echo ""

# --- Batch 3 ---
echo "--- Batch 3 ---"
run_condition "pma2_ind64"  0  2  64 &
run_condition "pma2_ind128" 1  2 128 &
wait
echo ""

# --- Batch 4 ---
echo "--- Batch 4 ---"
run_condition "pma4_ind32"  0  4  32 &
run_condition "pma4_ind64"  1  4  64 &
wait
echo ""

# --- Batch 5 ---
echo "--- Batch 5 ---"
run_condition "pma4_ind128" 0  4 128
echo ""

echo "=== Ablation complete at $(date) ==="
echo "Logs: $LOGDIR/"
