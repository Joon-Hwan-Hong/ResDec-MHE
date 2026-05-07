#!/usr/bin/env bash
# Phase 2: Variant B (multi_axis) full attribution suite under stacked base.
# Replaces the prior thin run (Captum IG + LOCO only). Adds: GradientSHAP,
# SmoothGrad, 5 attention methods (AttnLRP/GMAR/GAF AF/GF/AGF), CCC attention,
# SAE 31-CT causal patching. ~30-45 min on 1 GPU; uses GPU 0 by default.
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONPATH="$ROOT"
cd "$ROOT"

LOG_DIR="$ROOT/outputs/canonical/cogn_residual/multi_axis/interpretability_logs"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

LOG="$LOG_DIR/phase2_full_attribution.log"
echo "[$(date -Iseconds)] === Phase 2: Variant B full attribution (stacked base) ===" | tee -a "$LOG"

if uv run python scripts/resdec_mhe/cogn_residual/run_attribution_for_cogn_residual.py \
    --variant-name multi_axis --device "$DEVICE" \
    >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] === Phase 2 done ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE2_DONE"
else
    rc=$?
    echo "[$(date -Iseconds)] === Phase 2 FAILED rc=$rc ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE2_FAIL_${rc}"
    exit "$rc"
fi
