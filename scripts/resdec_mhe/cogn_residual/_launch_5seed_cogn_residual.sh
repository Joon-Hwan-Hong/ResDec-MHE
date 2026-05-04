#!/usr/bin/env bash
# 5-seed cross-replicate for Variant A under stacked + TabPFN-only residual bases.
# Tests whether the +0.031 R^2 gain from stacking the residual base holds across
# multiple training seeds (seed 42 already done in EXP-051; this adds 4 more).
#
# Per (seed, base) combination: 5 folds via run_5fold_parallel.sh on 2 GPUs.
# Total: 4 seeds * 2 bases * ~1 hr = ~8 hr wall. RUN_REINFER=1 includes the
# best-checkpoint reinfer step so each (seed, base) writes the same artifact
# layout as seed 42.
#
# Env overrides:
#   SEEDS         space-separated list (default: "67 21 2000 426")
#   BASES         space-separated list of "stacked" / "tabpfn_base" (default: both)
#   ROOT          worktree root (default: auto-derived from script path)
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This driver runs ~8 hr and must be in tmux to survive SSH disconnect." >&2
    echo "  tmux new -s cogn_residual_5seed 'bash $0'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SEEDS="${SEEDS:-67 21 2000 426}"
BASES="${BASES:-stacked tabpfn_base}"
METADATA_PATH="${METADATA_PATH:-data/metadata_ROSMAP}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-data/precomputed}"

export PYTHONPATH="$ROOT"

LOG_DIR="$ROOT/outputs/canonical/cogn_residual/gpath_only/seed_variation_logs"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
mkdir -p "$LOG_DIR" "$SENTINEL_DIR"

cd "$ROOT"

MASTER="$LOG_DIR/master.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$MASTER"; }

log "=== 5-seed cross-replicate (Variant A) ==="
log "ROOT     = $ROOT"
log "SEEDS    = $SEEDS"
log "BASES    = $BASES"
log "LOG_DIR  = $LOG_DIR"

N_FAILED=0

for BASE in $BASES; do
    if [ "$BASE" = "stacked" ]; then
        CFG="configs/resdec_mhe/cogn_residual/gpath_only.yaml"
        OUT_PREFIX="p5_seed"
    elif [ "$BASE" = "tabpfn_base" ]; then
        CFG="configs/resdec_mhe/cogn_residual/gpath_only_tabpfn_base.yaml"
        OUT_PREFIX="p5_tabpfn_base_seed"
    else
        log "ERROR: unknown BASE '$BASE'; expected 'stacked' or 'tabpfn_base'"
        exit 1
    fi
    for SEED in $SEEDS; do
        OUT="outputs/canonical/cogn_residual/gpath_only/${OUT_PREFIX}${SEED}"
        LOG="$LOG_DIR/${BASE}_seed${SEED}.log"
        if [ -f "$OUT/fold4/summary.json" ] && [ -f "$OUT/fold0/val_predictions_best.npz" ]; then
            log "SKIP ${BASE} seed ${SEED} (all 5 folds + reinfer present at $OUT)"
            continue
        fi
        log "=== ${BASE} seed ${SEED} -> ${OUT} (log: ${LOG}) ==="
        if CONFIG="$CFG" OUTROOT="$OUT" SEED="$SEED" RUN_REINFER=1 \
            METADATA_PATH="$METADATA_PATH" PRECOMPUTED_DIR="$PRECOMPUTED_DIR" \
            bash "$ROOT/scripts/resdec_mhe/training/run_5fold_parallel.sh" > "$LOG" 2>&1; then
            log "OK   ${BASE} seed ${SEED}"
        else
            rc=$?
            log "FAIL ${BASE} seed ${SEED} (rc=$rc, see $LOG)"
            N_FAILED=$((N_FAILED + 1))
        fi
    done
done

log "=== Phase 1 complete (failed=$N_FAILED) ==="
if [ "$N_FAILED" -gt 0 ]; then
    touch "$SENTINEL_DIR/PHASE1_FAIL_${N_FAILED}"
    exit 1
fi
touch "$SENTINEL_DIR/PHASE1_DONE"
