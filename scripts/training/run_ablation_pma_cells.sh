#!/usr/bin/env bash
# Ablation: PMA seeds {1, 2, 4} × max_cells {2500, 5000}
# All: ind=64, no OGM, no anneal, det=True, benchmark=False
#
# Usage:
#   setsid nohup bash scripts/training/run_ablation_pma_cells.sh > outputs/logs/ablation_pma_cells.log 2>&1 &
set -euo pipefail

FOLD=${1:-0}
N_GPUS=${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}
CONFIG="configs/production_5fold.yaml"
SPLITS="outputs/splits.json"
PRECOMPUTED="data/precomputed/rosmap/"
LOGDIR="outputs/logs/ablation_pma_cells"
mkdir -p "$LOGDIR"

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
    model.set_transformer.n_inducing_points=64
)

run_condition() {
    local name=$1
    local gpu=$2
    local pma=$3
    local max_cells=$4
    local logfile="$LOGDIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] $name (pma=$pma, cells=$max_cells) -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/training/train.py \
        "${COMMON[@]}" \
        experiment.run_name="ablation_${name}" \
        model.set_transformer.n_pma_seeds="$pma" \
        data.cell_sampling.max_cells_per_type="$max_cells" \
        > "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] $name DONE"
}

echo "=== PMA × max_cells ablation on fold $FOLD at $(date) ==="
echo "  ind=64, no OGM, no anneal, det=True"
echo ""

# --- Batch 1: PMA=1 ---
echo "--- Batch 1: pma1 ---"
run_condition "pma1_2500c" 0 1 2500 &
run_condition "pma1_5000c" 1 1 5000 &
wait
echo ""

# --- Batch 2: PMA=2 ---
echo "--- Batch 2: pma2 ---"
run_condition "pma2_2500c" 0 2 2500 &
run_condition "pma2_5000c" 1 2 5000 &
wait
echo ""

# --- Batch 3: PMA=4 ---
echo "--- Batch 3: pma4 ---"
run_condition "pma4_2500c" 0 4 2500 &
run_condition "pma4_5000c" 1 4 5000 &
wait
echo ""

echo "=== Ablation complete at $(date) ==="
echo "Logs: $LOGDIR/"
