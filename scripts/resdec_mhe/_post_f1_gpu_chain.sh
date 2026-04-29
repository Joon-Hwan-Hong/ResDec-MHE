#!/usr/bin/env bash
# Post-F1 GPU experiment chain
# Sequential: A (perm null inference-only) → B (F1 trajectory rerun) → D (F1 multi-fold) → E (SAE smaller-m)
# Failure-resilient: any single step's failure is logged and the chain continues
# Auto-updates MASTER-INFO via headless claude --resume after each step
# Tail the log file for human-readable progress: tail -F outputs/post_f1_chain.log
#
# Run in tmux for SIGHUP safety (already done by the controller):
#   tmux new-session -d -s f1_post_chain bash scripts/resdec_mhe/_post_f1_gpu_chain.sh

set -uo pipefail   # NO -e — per-step failures must NOT abort the chain

WORKTREE=/host/milan/tank/Joon/proj_ml_snrna/.worktrees/refinement-two
cd "$WORKTREE"

SESSION_ID="d9dde607-47b4-4b4d-a57d-05f65d980bef"
LOG="$WORKTREE/outputs/post_f1_chain.log"
mkdir -p "$(dirname "$LOG")"

F1_DONE_MARKER="$WORKTREE/outputs/canonical/interpretability/counterfactuals_optimized_absolute_delta0p3/counterfactuals_fold0.json"

# === Logging helpers ===
log()     { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
section() { echo "" | tee -a "$LOG"; echo "[$(date -Iseconds)] ==== $* ====" | tee -a "$LOG"; }

run_claude() {
    local prompt="$1"
    log "  >>> headless claude --resume $SESSION_ID dispatching post-step prompt"
    claude --resume "$SESSION_ID" --dangerously-skip-permissions --print -p "$prompt" 2>&1 \
        | tee -a "$LOG" || log "  WARN: headless claude failed (continuing chain)"
    log "  <<< headless claude exit"
}

log "==================================================="
log "Post-F1 GPU experiment chain START"
log "Worktree:    $WORKTREE"
log "Session id:  $SESSION_ID"
log "Log file:    $LOG"
log "==================================================="

# === Step 1: wait for F1 grid completion (with heartbeats) ===
section "Step 1: waiting for F1 grid (Phase 2) completion"
log "Marker file: $F1_DONE_MARKER"
WAIT_T0=$(date +%s)
LAST_HB=$WAIT_T0
HB_INTERVAL=300
until [ -f "$F1_DONE_MARKER" ]; do
    NOW=$(date +%s)
    if [ $((NOW - LAST_HB)) -ge $HB_INTERVAL ]; then
        log "  heartbeat: still waiting for F1 grid (elapsed $((NOW - WAIT_T0))s)"
        LAST_HB=$NOW
    fi
    sleep 60
done
SIZE=$(stat -c%s "$F1_DONE_MARKER" 2>/dev/null || echo "?")
log "F1 grid complete; marker file size ${SIZE} bytes"

# === Step A: N=100 inference-only perm null ===
section "Step A: N=100 inference-only permutation null"
A_OUT="$WORKTREE/outputs/canonical/permutation_test_n100"
mkdir -p "$A_OUT"
A_T0=$(date +%s)
{
    uv run python scripts/resdec_mhe/training/run_permutation_test_inference_only.py \
        --n-perms 100 \
        --canonical-dir outputs/canonical/p5_canonical_seed42 \
        --output "$A_OUT/permutation_summary.json" \
        2>&1 | tee -a "$LOG"
} || log "  ERROR: Step A failed (continuing)"
A_ELAPSED=$(($(date +%s) - A_T0))
if [ -f "$A_OUT/permutation_summary.json" ]; then
    A_SIZE=$(stat -c%s "$A_OUT/permutation_summary.json")
    log "Step A DONE; elapsed ${A_ELAPSED}s; output ${A_SIZE} bytes at $A_OUT/permutation_summary.json"
else
    log "Step A FAILED to produce output JSON; elapsed ${A_ELAPSED}s"
fi
run_claude "Step A of post-F1 chain done: N=100 inference-only permutation null. Read outputs/canonical/permutation_test_n100/permutation_summary.json. Update MASTER-INFO §4 + §27 with the new N=100 numbers (canonical R², null mean, null std, z under null, p-floor, n_perms_ge_canonical). Commit (note: docs/ is gitignored, MASTER-INFO is local-only). Reply 'A processed' and exit."

# === Step B: F1 trajectory rerun (δ=0.5, both modes, record_trajectory=True) ===
section "Step B: F1 trajectory rerun (δ=0.5, both modes, record_trajectory=True)"
B_OUT_REL="$WORKTREE/outputs/canonical/interpretability/counterfactuals_trajectory_relative_delta0p5"
B_OUT_ABS="$WORKTREE/outputs/canonical/interpretability/counterfactuals_trajectory_absolute_delta0p5"
mkdir -p "$B_OUT_REL" "$B_OUT_ABS"
B_T0=$(date +%s)
{
    log "  GPU 0: relative δ=0.5 (trajectory ON) → $B_OUT_REL"
    log "  GPU 1: absolute δ=0.5 (trajectory ON) → $B_OUT_ABS"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
        --target-mode relative --target-delta 0.5 --tol 1e-2 --lambda-max 1e4 --max-steps 1000 \
        --record-trajectory --out-dir "$B_OUT_REL" --device cuda:0 \
        > "$B_OUT_REL/run.log" 2>&1 &
    PID0=$!
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
        --target-mode absolute --target-delta 0.5 --tol 1e-2 --lambda-max 1e4 --max-steps 1000 \
        --record-trajectory --out-dir "$B_OUT_ABS" --device cuda:0 \
        > "$B_OUT_ABS/run.log" 2>&1 &
    PID1=$!
    wait "$PID0"; RC0=$?
    wait "$PID1"; RC1=$?
    log "  B GPU 0 rc: $RC0; GPU 1 rc: $RC1"
} || log "  ERROR: Step B launch wrapper failed (continuing)"
B_ELAPSED=$(($(date +%s) - B_T0))
log "Step B DONE-ish; elapsed ${B_ELAPSED}s ($((B_ELAPSED/60)) min)"
[ -f "$B_OUT_REL/counterfactuals_fold0.json" ] && log "  B rel JSON: $(stat -c%s "$B_OUT_REL/counterfactuals_fold0.json") bytes"
[ -f "$B_OUT_ABS/counterfactuals_fold0.json" ] && log "  B abs JSON: $(stat -c%s "$B_OUT_ABS/counterfactuals_fold0.json") bytes"
run_claude "Step B of post-F1 chain done: F1 trajectory rerun at δ=0.5, both target modes, with record_trajectory=True. Each per-subject result in outputs/canonical/interpretability/counterfactuals_trajectory_{relative,absolute}_delta0p5/counterfactuals_fold0.json now has a 'trajectory' field = list of (lambda, residual_at_end) tuples. Build a per-subject convergence-curve figure (loss vs lambda-doubling-index OR loss vs cumulative-step, color-coded by regime, all 20 subjects on one panel). Save to outputs/canonical/interpretability/figures/lab_meeting/fig_F1_loss_landscape.{png,pdf} via a NEW small orchestrator at scripts/resdec_mhe/interpretability/make_loss_landscape_figure.py. Update MASTER-INFO §8.1.1 with the convergence-trajectory mechanism finding. Commit. Reply 'B processed' and exit."

# === Step D: F1 multi-fold replication (folds 1-4 × 2 deltas × 2 modes) ===
section "Step D: F1 multi-fold replication (folds 1-4 × 2 deltas × 2 modes)"
D_T0=$(date +%s)
for FOLD in 1 2 3 4; do
    section "  Step D fold=$FOLD"
    for DELTA in 0.5 0.3; do
        DSTR="${DELTA/./p}"
        D_OUT_REL="$WORKTREE/outputs/canonical/interpretability/counterfactuals_optimized_relative_delta${DSTR}_fold${FOLD}"
        D_OUT_ABS="$WORKTREE/outputs/canonical/interpretability/counterfactuals_optimized_absolute_delta${DSTR}_fold${FOLD}"
        mkdir -p "$D_OUT_REL" "$D_OUT_ABS"
        log "  D fold=$FOLD δ=$DELTA: launching 2-GPU pair"
        D_PAIR_T0=$(date +%s)
        {
            CUDA_VISIBLE_DEVICES=0 uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
                --fold "$FOLD" --target-mode relative --target-delta "$DELTA" \
                --tol 1e-2 --lambda-max 1e4 --max-steps 1000 \
                --out-dir "$D_OUT_REL" --device cuda:0 \
                > "$D_OUT_REL/run.log" 2>&1 &
            PID0=$!
            CUDA_VISIBLE_DEVICES=1 uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
                --fold "$FOLD" --target-mode absolute --target-delta "$DELTA" \
                --tol 1e-2 --lambda-max 1e4 --max-steps 1000 \
                --out-dir "$D_OUT_ABS" --device cuda:0 \
                > "$D_OUT_ABS/run.log" 2>&1 &
            PID1=$!
            wait "$PID0"; RC0=$?
            wait "$PID1"; RC1=$?
            D_PAIR_ELAPSED=$(($(date +%s) - D_PAIR_T0))
            log "  D fold=$FOLD δ=$DELTA pair done; rc=($RC0,$RC1); elapsed ${D_PAIR_ELAPSED}s"
        } || log "  ERROR: D fold=$FOLD δ=$DELTA failed (continuing)"
    done
done
D_ELAPSED=$(($(date +%s) - D_T0))
log "Step D DONE; total elapsed ${D_ELAPSED}s ($((D_ELAPSED/3600)) hr)"
run_claude "Step D of post-F1 chain done: F1 multi-fold replication (folds 1-4 × 2 deltas × 2 modes; fold 0 from Phase 1+2 already in §8.1.1). Aggregate per-regime asymmetry across all 5 folds (fold 0 + folds 1-4 from this step). For each fold: per-regime success rates (should be 10/10 at relaxed tol; verify), mean+std step counts per regime, top-3 perturbation CTs per regime. Build a per-fold step-count box-and-whisker by regime. Update MASTER-INFO §8.1.1 with the multi-fold replication; turn the fold-0 finding into a 5-fold robustness statement. Commit. Reply 'D processed' and exit."

# === Step E: SAE smaller-m / smaller-K stability sweep ===
section "Step E: SAE smaller-m / smaller-K stability sweep (180 configs)"
E_T0=$(date +%s)
{
    OUT_ROOT="outputs/canonical/sae/stability_smaller_m" \
        bash scripts/resdec_mhe/run_sae_sweep_smaller_m.sh 2>&1 | tee -a "$LOG"
} || log "  ERROR: Step E failed (continuing)"
E_ELAPSED=$(($(date +%s) - E_T0))
log "Step E DONE; elapsed ${E_ELAPSED}s ($((E_ELAPSED/3600)) hr)"
run_claude "Step E of post-F1 chain done: SAE smaller-m / smaller-K stability sweep (m∈{4,8,16} × K∈{4,8,16,32,64} × 3 seeds × 2 archs × 2 layers = 180 configs). Aggregate cross-seed stability (Paulo-Belrose 0.7 cosine threshold) per (m, K, arch, layer). Test whether the canonical-config 0% stability finding (§31.10) was an over-completeness artifact: does smaller m yield > 0% stability? Update MASTER-INFO §31.10 with the stability-vs-overcompleteness sweep result. If 0% persists at small m → §31.11 distributed-representation claim strengthens (not an artifact). If > 0% recovered → caveat the §31.11 claim. Commit. Reply 'E processed' and exit."

# === Final summary ===
section "Chain final summary"
TOTAL_ELAPSED=$(($(date +%s) - WAIT_T0))
log "Total chain elapsed (including F1 wait): ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED/3600)) hr $((TOTAL_ELAPSED%3600/60)) min)"
run_claude "Post-F1 GPU chain complete. Read outputs/post_f1_chain.log to see per-step status (DONE/FAILED). Aggregate the chain's overall outcome: which steps succeeded, key new findings (perm null tighter p-floor with N=100, F1 trajectory convergence-asymmetry mechanism, multi-fold replication, SAE stability across overcompleteness). Add a NEW §33 'Post-F1 GPU chain results (2026-04-29)' section to MASTER-INFO with full Run-provenance template (per §8.1.1 standard) summarizing chain deliverables + linking to the per-step output paths. Commit. Reply 'Chain summary done' and exit."

log "==================================================="
log "All chain steps attempted"
log "End: $(date -Iseconds)"
log "==================================================="
