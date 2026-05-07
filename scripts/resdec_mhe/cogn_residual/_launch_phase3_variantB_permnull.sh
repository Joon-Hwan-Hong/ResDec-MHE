#!/usr/bin/env bash
# Phase 3: Variant B (multi_axis) perm null N=20 under stacked base.
# Wraps _launch_permnull_cogn_residual_shard.sh. ~3 hr wall on 2 GPUs.
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
N_PERMS="${N_PERMS:-20}"
VARIANT_CONFIG="$ROOT/configs/resdec_mhe/cogn_residual/multi_axis.yaml"
RESIDUAL_CACHE_DIR="$ROOT/outputs/canonical/cogn_residual/multi_axis/cache"
OUT_BASE="$ROOT/outputs/canonical/cogn_residual/multi_axis/permutation_test_n${N_PERMS}_stacked"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
mkdir -p "$OUT_BASE" "$SENTINEL_DIR"

export PYTHONPATH="$ROOT"
cd "$ROOT"

LOG="$OUT_BASE/phase3_master.log"
echo "[$(date -Iseconds)] === Phase 3: Variant B perm null N=${N_PERMS} stacked ===" | tee -a "$LOG"

if N_PERMS="$N_PERMS" \
    VARIANT_CONFIG="$VARIANT_CONFIG" \
    RESIDUAL_CACHE_DIR="$RESIDUAL_CACHE_DIR" \
    OUT_BASE="$OUT_BASE" \
    bash "$ROOT/scripts/resdec_mhe/cogn_residual/_launch_permnull_cogn_residual_shard.sh" \
    >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] === Phase 3 shards done; aggregating ===" | tee -a "$LOG"
    if uv run python scripts/resdec_mhe/cogn_residual/aggregate_permnull_cogn_residual.py \
        --shards-dir "$OUT_BASE" \
        --canonical-summary "$ROOT/outputs/canonical/cogn_residual/multi_axis/p5_seed42/best_vs_tabpfn_summary.json" \
        --out-json "$OUT_BASE/permutation_summary.json" \
        >> "$LOG" 2>&1; then
        echo "[$(date -Iseconds)] === Phase 3 done ===" | tee -a "$LOG"
        touch "$SENTINEL_DIR/PHASE3_DONE"
    else
        rc=$?
        echo "[$(date -Iseconds)] === Phase 3 aggregation FAILED rc=$rc ===" | tee -a "$LOG"
        touch "$SENTINEL_DIR/PHASE3_FAIL_AGG_${rc}"
        exit "$rc"
    fi
else
    rc=$?
    echo "[$(date -Iseconds)] === Phase 3 sharded run FAILED rc=$rc ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE3_FAIL_${rc}"
    exit "$rc"
fi
