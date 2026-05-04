#!/usr/bin/env bash
# Chain orchestrator for Phases 2-7 of the cogn-residual experiment chain.
# Polls for the upstream phase's sentinel before launching the next.
# Designed to run in a separate tmux session from Phase 1; safe to launch
# while Phase 1 is still running.
#
# Phases (order):
#   2. Variant B full attribution suite (stacked base)
#   3. Variant B perm null N=20 stacked
#   4. Variant A perm null N=50 stacked
#   5. Variant A counterfactuals (Wachter Mode-A) [stub — pending Phase 5 launcher]
#   6. Variant A distributional (Wasserstein-1 + raw CMI) [stub]
#   7. Variant A learning curve 5-seed × 4 sub-Ns [stub]
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
mkdir -p "$SENTINEL_DIR"

export PYTHONPATH="$ROOT"
cd "$ROOT"

LOG_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_logs"
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/chain_master.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$MASTER"; }

wait_for_sentinel() {
    local phase="$1"
    local sentinel_done="$SENTINEL_DIR/${phase}_DONE"
    local sentinel_glob="$SENTINEL_DIR/${phase}_FAIL_*"
    log "waiting for $phase sentinel (check every 60s)"
    until [ -f "$sentinel_done" ] || compgen -G "$sentinel_glob" > /dev/null; do
        sleep 60
    done
    if [ -f "$sentinel_done" ]; then
        log "$phase DONE"
        return 0
    else
        log "$phase FAILED — chain stopping"
        return 1
    fi
}

run_phase() {
    local phase="$1"; local script="$2"
    log "=== launching $phase: $script ==="
    if bash "$ROOT/scripts/resdec_mhe/cogn_residual/$script"; then
        log "$phase script returned 0"
    else
        rc=$?
        log "$phase script FAILED rc=$rc"
        return "$rc"
    fi
}

log "=== Chain orchestrator started ==="
log "ROOT = $ROOT"

# ── PHASE 1 (Variant A 5-seed × 2 bases) ── waits on its own sentinel
wait_for_sentinel "PHASE1" || exit 1

# ── PHASE 2 (Variant B full attribution) ──
run_phase "PHASE2" "_launch_phase2_variantB_attribution.sh" || exit 1

# ── PHASE 3 (Variant B perm null N=20 stacked) ──
run_phase "PHASE3" "_launch_phase3_variantB_permnull.sh" || exit 1

# ── PHASE 4 (Variant A perm null N=50 stacked) ──
run_phase "PHASE4" "_launch_phase4_variantA_permnull_n50.sh" || exit 1

# ── PHASE 5 (Variant A counterfactuals: relative + absolute, fold 0) ──
run_phase "PHASE5" "_launch_phase5_variantA_cf.sh" || exit 1

# ── PHASE 6 (Variant A distributional: Wasserstein-1 + raw CMI) ──
run_phase "PHASE6" "_launch_phase6_variantA_distrib.sh" || exit 1

# ── PHASE 7 (Variant A learning curve 5-seed × 4 sub-Ns) ──
# Pending — needs new orchestrator that re-fits per-fold OLS residualization on
# subsampled training subjects (existing canonical run_learning_curve.py uses
# raw cogn_global). Will be filled in a follow-up commit; the chain stops here
# and emits CHAIN_PHASES_1_TO_6_DONE.
if [ -f "$ROOT/scripts/resdec_mhe/cogn_residual/_launch_phase7_variantA_learning_curve.sh" ]; then
    run_phase "PHASE7" "_launch_phase7_variantA_learning_curve.sh" || exit 1
    touch "$SENTINEL_DIR/CHAIN_PHASES_1_TO_7_DONE"
    log "=== Chain Phases 1-7 done ==="
else
    log "=== Phase 7 launcher absent; chain stops at Phase 6 ==="
    touch "$SENTINEL_DIR/CHAIN_PHASES_1_TO_6_DONE"
    log "=== Chain Phases 1-6 done ==="
fi
