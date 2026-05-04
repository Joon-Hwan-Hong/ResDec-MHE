#!/usr/bin/env bash
# Phase 5: Variant A counterfactuals (Wachter Mode-A literal). Two target modes
# (relative + absolute), 10 resilient + 10 vulnerable subjects each. ~2 hr wall
# on 1 GPU. Reuses canonical run_counterfactuals.py with variant config +
# variant residual CSV (computed first by build_variant_residual_csv.py).
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DEVICE="${DEVICE:-cuda:0}"
VARIANT_NAME="${VARIANT_NAME:-gpath_only}"

VARIANT_CONFIG="$ROOT/configs/resdec_mhe/cogn_residual/${VARIANT_NAME}.yaml"
CANONICAL_DIR="$ROOT/outputs/canonical/cogn_residual/${VARIANT_NAME}/p5_seed42"
RESIDUAL_CACHE_DIR="$ROOT/outputs/canonical/cogn_residual/${VARIANT_NAME}/cache"
INTERP_OUT="$ROOT/outputs/canonical/cogn_residual/${VARIANT_NAME}/interpretability"
RESIDUAL_CSV="$INTERP_OUT/variant_residual_per_subject.csv"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
LOG="$INTERP_OUT/phase5_cf.log"
mkdir -p "$INTERP_OUT" "$SENTINEL_DIR"

export PYTHONPATH="$ROOT"
cd "$ROOT"

echo "[$(date -Iseconds)] === Phase 5: Variant ${VARIANT_NAME} CF (Wachter Mode-A) ===" | tee -a "$LOG"

# 1. Build variant residual CSV (per-subject residualized cognition from val folds)
if [ ! -f "$RESIDUAL_CSV" ]; then
    echo "[$(date -Iseconds)] building $RESIDUAL_CSV" | tee -a "$LOG"
    if ! uv run python scripts/resdec_mhe/cogn_residual/build_variant_residual_csv.py \
        --residual-cache-dir "$RESIDUAL_CACHE_DIR" \
        --splits-path "$ROOT/outputs/splits.json" \
        --out-csv "$RESIDUAL_CSV" \
        >> "$LOG" 2>&1; then
        rc=$?
        echo "[$(date -Iseconds)] === Phase 5 residual-CSV build FAILED rc=$rc ===" | tee -a "$LOG"
        touch "$SENTINEL_DIR/PHASE5_FAIL_BUILD_${rc}"
        exit "$rc"
    fi
fi

# 2. Run CF in both target modes (relative, absolute), fold 0
N_FAILED=0
for MODE in relative absolute; do
    OUT_DIR="$INTERP_OUT/counterfactuals_${MODE}"
    mkdir -p "$OUT_DIR"
    echo "[$(date -Iseconds)] running CF mode=${MODE} fold=0" | tee -a "$LOG"
    if CF_CONFIG="$VARIANT_CONFIG" \
        CF_FOLD=0 \
        CF_CANONICAL_DIR="$CANONICAL_DIR" \
        CF_SPLITS_PATH="$ROOT/outputs/splits.json" \
        CF_RESIDUAL_CSV="$RESIDUAL_CSV" \
        CF_DEVICE="$DEVICE" \
        CF_OUT_DIR="$OUT_DIR" \
        uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
        --target-mode "$MODE" \
        >> "$LOG" 2>&1; then
        echo "[$(date -Iseconds)] CF mode=${MODE} done" | tee -a "$LOG"
    else
        rc=$?
        echo "[$(date -Iseconds)] CF mode=${MODE} FAILED rc=$rc" | tee -a "$LOG"
        N_FAILED=$((N_FAILED + 1))
    fi
done

if [ "$N_FAILED" -eq 0 ]; then
    echo "[$(date -Iseconds)] === Phase 5 done ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE5_DONE"
else
    echo "[$(date -Iseconds)] === Phase 5 FAILED (${N_FAILED} mode runs) ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE5_FAIL_${N_FAILED}"
    exit 1
fi
