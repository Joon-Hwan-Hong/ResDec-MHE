#!/usr/bin/env bash
# Phase 7: Variant A learning curve 5-seed × 4 sub-Ns under stacked base.
# Tests whether the residualized-cognition R^2 holds across N regimes.
# Per (seed, N) cycle: subsample metadata -> residualize -> top-k -> TabPFN
# cache -> RF cache -> stacked cache -> 5-fold train -> aggregate. ~10-13 hr
# wall total (5 seeds * 4 Ns * ~30 min each).
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VARIANT_NAME="${VARIANT_NAME:-gpath_only}"
SEEDS_ARG="${SEEDS:-42 67 21 2000 426}"
N_VALUES_ARG="${N_VALUES:-100 200 300 400}"

OUT_BASE="$ROOT/outputs/canonical/cogn_residual/${VARIANT_NAME}/learning_curve_k5"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
LOG="$OUT_BASE/phase7_master.log"
mkdir -p "$OUT_BASE" "$SENTINEL_DIR"

export PYTHONPATH="$ROOT"
cd "$ROOT"

echo "[$(date -Iseconds)] === Phase 7: Variant ${VARIANT_NAME} learning curve ===" | tee -a "$LOG"
echo "[$(date -Iseconds)] SEEDS=${SEEDS_ARG} N_VALUES=${N_VALUES_ARG}" | tee -a "$LOG"

if uv run python scripts/resdec_mhe/cogn_residual/run_learning_curve_cogn_residual.py \
    --variant-name "$VARIANT_NAME" \
    --rng-seeds $SEEDS_ARG \
    --N-values $N_VALUES_ARG \
    --output-base "$OUT_BASE" \
    >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] === Phase 7 done ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE7_DONE"
else
    rc=$?
    echo "[$(date -Iseconds)] === Phase 7 FAILED rc=$rc ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE7_FAIL_${rc}"
    exit "$rc"
fi
