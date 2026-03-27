#!/usr/bin/env bash
# 2×2 ablation: (OGM-GE vs no OGM-GE) × (temp anneal vs no temp anneal)
# Plus baseline condition matching HPO2 exactly.
# All on fold 0, 30 epochs, compressed annealing schedule.
#
# Usage:
#   bash scripts/run_ablation_2x2.sh [--fold N] [--n-gpus N]
#   setsid nohup bash scripts/run_ablation_2x2.sh > outputs/logs/ablation_2x2.log 2>&1 &
set -euo pipefail

FOLD=${1:-0}
N_GPUS=${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}
CONFIG="configs/production_5fold.yaml"
SPLITS="outputs/splits.json"
PRECOMPUTED="data/precomputed/rosmap/"
LOGDIR="outputs/logs/ablation_2x2"
mkdir -p "$LOGDIR"

# Common overrides: 30 epochs, compressed annealing, single GPU
COMMON=(
    --config "$CONFIG"
    --splits-path "$SPLITS"
    --precomputed-dir "$PRECOMPUTED"
    --fold "$FOLD"
    training.max_epochs=30
    training.temperature_annealing.warmup_epochs=2
    training.temperature_annealing.anneal_epochs=15
    training.kl_annealing.warmup_epochs=2
    training.early_stopping.min_epochs=17
    training.early_stopping.patience=15
    training.devices=1
    training.strategy=auto
)

run_condition() {
    local name=$1
    local gpu=$2
    shift 2
    local overrides=("$@")
    local logfile="$LOGDIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] $name -> GPU $gpu (log: $logfile)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/train.py \
        "${COMMON[@]}" \
        experiment.run_name="ablation_${name}" \
        "${overrides[@]}" \
        > "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] $name DONE"
}

echo "=== 2×2 Ablation + baseline on fold $FOLD at $(date) ==="
echo "  Config: $CONFIG"
echo "  GPUs: $N_GPUS"
echo ""

# Condition 0: Baseline — exact HPO2 conditions (no OGM-GE, det=False, benchmark=True)
# Condition 1: No OGM-GE + temp anneal + det=True (isolate deterministic effect)
# Condition 2: OGM-GE + temp anneal + det=True (isolate OGM-GE effect)
# Condition 3: No OGM-GE + no temp anneal + det=True
# Condition 4: OGM-GE + no temp anneal + det=True

# --- Batch 1: Baseline + Condition 1 ---
echo "--- Batch 1: baseline + no_ogm_anneal_detTrue ---"
run_condition "baseline_hpo2" 0 \
    training.gradient_modulation.enabled=false \
    reproducibility.deterministic=false \
    reproducibility.benchmark=true &
PID0=$!

run_condition "no_ogm_anneal_detTrue" 1 \
    training.gradient_modulation.enabled=false \
    reproducibility.deterministic=true \
    reproducibility.benchmark=false &
PID1=$!
wait $PID0 $PID1
echo ""

# --- Batch 2: Condition 2 + Condition 3 ---
echo "--- Batch 2: ogm_anneal_detTrue + no_ogm_no_anneal_detTrue ---"
run_condition "ogm_anneal_detTrue" 0 \
    training.gradient_modulation.enabled=true \
    reproducibility.deterministic=true \
    reproducibility.benchmark=false &
PID2=$!

run_condition "no_ogm_no_anneal_detTrue" 1 \
    training.gradient_modulation.enabled=false \
    training.temperature_annealing.warmup_epochs=999 \
    reproducibility.deterministic=true \
    reproducibility.benchmark=false &
PID3=$!
wait $PID2 $PID3
echo ""

# --- Batch 3: Condition 4 ---
echo "--- Batch 3: ogm_no_anneal_detTrue ---"
run_condition "ogm_no_anneal_detTrue" 0 \
    training.gradient_modulation.enabled=true \
    training.temperature_annealing.warmup_epochs=999 \
    reproducibility.deterministic=true \
    reproducibility.benchmark=false
echo ""

echo "=== Ablation complete at $(date) ==="
echo "Logs: $LOGDIR/"
