#!/usr/bin/env bash
# Phase 4: Variant A (gpath_only) perm null N=50 under stacked base. Matches
# canonical N=50 headline (p-floor 1/51 ≈ 0.0196 vs N=20's 1/21 ≈ 0.0476).
# ~6-8 hr wall on 2 GPUs.
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
N_PERMS="${N_PERMS:-50}"
VARIANT_CONFIG="$ROOT/configs/resdec_mhe/cogn_residual/gpath_only.yaml"
RESIDUAL_CACHE_DIR="$ROOT/outputs/canonical/cogn_residual/gpath_only/cache"
OUT_BASE="$ROOT/outputs/canonical/cogn_residual/gpath_only/permutation_test_n${N_PERMS}_stacked"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
mkdir -p "$OUT_BASE" "$SENTINEL_DIR"

export PYTHONPATH="$ROOT"
cd "$ROOT"

LOG="$OUT_BASE/phase4_master.log"
echo "[$(date -Iseconds)] === Phase 4: Variant A perm null N=${N_PERMS} stacked ===" | tee -a "$LOG"

if N_PERMS="$N_PERMS" \
    VARIANT_CONFIG="$VARIANT_CONFIG" \
    RESIDUAL_CACHE_DIR="$RESIDUAL_CACHE_DIR" \
    OUT_BASE="$OUT_BASE" \
    bash "$ROOT/scripts/resdec_mhe/cogn_residual/_launch_permnull_cogn_residual_shard.sh" \
    >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] === Phase 4 shards done; aggregating ===" | tee -a "$LOG"
    if uv run python scripts/resdec_mhe/cogn_residual/aggregate_permnull_cogn_residual.py \
        --shards-dir "$OUT_BASE" \
        --canonical-summary "$ROOT/outputs/canonical/cogn_residual/gpath_only/p5_seed42/best_vs_tabpfn_summary.json" \
        --out-json "$OUT_BASE/permutation_summary.json" \
        >> "$LOG" 2>&1; then
        echo "[$(date -Iseconds)] === Phase 4 done ===" | tee -a "$LOG"
        touch "$SENTINEL_DIR/PHASE4_DONE"
    else
        rc=$?
        echo "[$(date -Iseconds)] === Phase 4 aggregation FAILED rc=$rc ===" | tee -a "$LOG"
        touch "$SENTINEL_DIR/PHASE4_FAIL_AGG_${rc}"
        exit "$rc"
    fi
else
    rc=$?
    echo "[$(date -Iseconds)] === Phase 4 sharded run FAILED rc=$rc ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE4_FAIL_${rc}"
    exit "$rc"
fi
