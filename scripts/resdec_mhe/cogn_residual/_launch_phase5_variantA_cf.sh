#!/usr/bin/env bash
# Phase 5: Variant A counterfactuals (Wachter Mode-A literal). Two target modes
# (relative + absolute), 10 resilient + 10 vulnerable subjects each. Both modes
# dispatched in PARALLEL on cuda:0 (relative) and cuda:1 (absolute) and the
# script waits for both to complete before emitting the sentinel.
# IDEMPOTENT: if a mode's counterfactuals_fold0.json already exists with valid
# JSON, that mode is skipped (the run is treated as already done).
# Variant CF empirically takes ~2.5 hr per mode (vs canonical 9.1 hr) due to
# residualized-target convergence; both modes in parallel ≈ 2.5 hr total wall.
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VARIANT_NAME="${VARIANT_NAME:-gpath_only}"
RELATIVE_DEVICE="${RELATIVE_DEVICE:-cuda:0}"
ABSOLUTE_DEVICE="${ABSOLUTE_DEVICE:-cuda:1}"

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

echo "[$(date -Iseconds)] === Phase 5: Variant ${VARIANT_NAME} CF (Wachter Mode-A, parallel) ===" | tee -a "$LOG"

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

# Helper: launch one CF mode on the assigned device IF its output JSON does not
# already exist (idempotent re-runs after partial failure / interrupted run).
# Writes the mode-specific PID to the named global var when launched, or "" when
# skipped. Caller is responsible for `wait`ing on the PID.
launch_cf_mode() {
    local mode="$1"; local device="$2"; local pid_var="$3"
    local out_dir="$INTERP_OUT/counterfactuals_${mode}"
    local out_json="$out_dir/counterfactuals_fold0.json"
    mkdir -p "$out_dir"
    if [ -f "$out_json" ] && [ -s "$out_json" ] && \
       uv run python -c "import json,sys; json.load(open('$out_json'))" 2>/dev/null; then
        echo "[$(date -Iseconds)] SKIP CF mode=${mode} (valid JSON already at $out_json)" | tee -a "$LOG"
        eval "$pid_var=''"
        return 0
    fi
    echo "[$(date -Iseconds)] launching CF mode=${mode} on ${device}" | tee -a "$LOG"
    CF_CONFIG="$VARIANT_CONFIG" \
        CF_FOLD=0 \
        CF_CANONICAL_DIR="$CANONICAL_DIR" \
        CF_SPLITS_PATH="$ROOT/outputs/splits.json" \
        CF_RESIDUAL_CSV="$RESIDUAL_CSV" \
        CF_DEVICE="$device" \
        CF_OUT_DIR="$out_dir" \
        CF_METADATA_PATH="$ROOT/data/metadata_ROSMAP" \
        CF_PRECOMPUTED_DIR="$ROOT/data/precomputed" \
        uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
        --target-mode "$mode" \
        >> "$LOG" 2>&1 &
    eval "$pid_var=$!"
}

PID_REL=""
PID_ABS=""
launch_cf_mode "relative" "$RELATIVE_DEVICE" PID_REL
launch_cf_mode "absolute" "$ABSOLUTE_DEVICE" PID_ABS

N_FAILED=0
if [ -n "$PID_REL" ]; then
    if wait "$PID_REL"; then
        echo "[$(date -Iseconds)] CF mode=relative done" | tee -a "$LOG"
    else
        rc=$?
        echo "[$(date -Iseconds)] CF mode=relative FAILED rc=$rc" | tee -a "$LOG"
        N_FAILED=$((N_FAILED + 1))
    fi
fi
if [ -n "$PID_ABS" ]; then
    if wait "$PID_ABS"; then
        echo "[$(date -Iseconds)] CF mode=absolute done" | tee -a "$LOG"
    else
        rc=$?
        echo "[$(date -Iseconds)] CF mode=absolute FAILED rc=$rc" | tee -a "$LOG"
        N_FAILED=$((N_FAILED + 1))
    fi
fi

if [ "$N_FAILED" -eq 0 ]; then
    echo "[$(date -Iseconds)] === Phase 5 done ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE5_DONE"
else
    echo "[$(date -Iseconds)] === Phase 5 FAILED (${N_FAILED} mode runs) ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE5_FAIL_${N_FAILED}"
    exit 1
fi
