#!/usr/bin/env bash
# Seed-variation driver: runs the canonical 5-fold ResDec-MHE pipeline once
# per seed. Each seed gets its own output directory and log; folds within a
# seed parallelize across both GPUs via run_5fold_parallel.sh.
#
# Env overrides (all optional):
#   SEEDS         space-separated list of seeds (default: "67 21 2000 426")
#   OUT_BASE      parent dir for per-seed output dirs (default: outputs/canonical)
#   CONFIG        phase YAML (passed through to run_5fold_parallel.sh)
#   N_GPUS        number of GPUs to use per seed (passed through)
#   GPU_LIST      comma-separated GPU list (passed through)
#
# Usage:
#   bash scripts/resdec_mhe/training/run_seed_variation.sh
#   SEEDS="42 67 21 2000 426" bash scripts/resdec_mhe/training/run_seed_variation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

SEEDS="${SEEDS:-67 21 2000 426}"
OUT_BASE="${OUT_BASE:-outputs/canonical}"
LOG_DIR="${OUT_BASE}/seed_variation_logs"

mkdir -p "$LOG_DIR"

cd "$ROOT"

echo "=== seed-variation run ==="
echo "ROOT     = $ROOT"
echo "SEEDS    = $SEEDS"
echo "OUT_BASE = $OUT_BASE"
echo "LOG_DIR  = $LOG_DIR"
date

for SEED in $SEEDS; do
    OUT="${OUT_BASE}/p5_canonical_seed${SEED}"
    LOG="${LOG_DIR}/seed${SEED}.log"
    echo ""
    echo "=== seed ${SEED} → ${OUT} (log: ${LOG}) ==="
    date
    OUTROOT="$OUT" SEED="$SEED" RUN_REINFER=1 \
        bash "$SCRIPT_DIR/run_5fold_parallel.sh" > "$LOG" 2>&1
    echo "=== seed ${SEED} done ==="
    date
done

echo ""
echo "=== ALL SEEDS DONE ==="
date
