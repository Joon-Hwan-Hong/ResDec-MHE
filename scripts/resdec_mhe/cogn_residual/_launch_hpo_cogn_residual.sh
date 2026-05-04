#!/usr/bin/env bash
# Launch Optuna HPO on the residualized cognition target across 2 GPUs.
# Each worker pulls trials from the same SQLite-backed study. Per-trial
# 5-fold (folds 0..4) with MedianPruner; 7-axis search space defined in
# run_hpo_cogn_residual.py.
#
# Required env:
#   OUT_BASE       output dir (e.g. outputs/canonical/cogn_residual/gpath_only/hpo_wide)
# Optional env:
#   N_TRIALS_TOTAL total trials across both workers (default: 60)
#   STUDY_NAME     default: cogn_residual_gpath_only_hpo
#                  override per-run (e.g. ..._wide, ..._v2) so different sweeps
#                  don't collide on the same SQLite schema
#   BASE_CONFIG    default: configs/resdec_mhe/cogn_residual/gpath_only.yaml
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This launcher should be run inside tmux to survive SSH disconnect." >&2
    echo "  tmux new -s cogn_residual_hpo 'bash $0'" >&2
    exit 1
fi

: "${OUT_BASE:?OUT_BASE required}"
N_TRIALS_TOTAL="${N_TRIALS_TOTAL:-60}"
STUDY_NAME="${STUDY_NAME:-cogn_residual_gpath_only_hpo}"
BASE_CONFIG="${BASE_CONFIG:-configs/resdec_mhe/cogn_residual/gpath_only.yaml}"

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WT"

# Force PYTHONPATH to worktree root so all child Python invocations (this
# launcher's `uv run python` calls AND their nested subprocess.run train.py
# calls) resolve `import src.*` to this worktree, not the parent repo. The
# parent repo (master) lacks the variant residualization injection and would
# silently train on raw cogn_global. See feedback_no_silent_plan_drops.md
# (audit-discipline rule) and the recurrence note added to
# feedback_no_transient_codes_in_commits.md after the 2026-05-04 incident.
export PYTHONPATH="${PYTHONPATH:-$WT}"

mkdir -p "$OUT_BASE/work_a" "$OUT_BASE/work_b"
DB_PATH="$WT/$OUT_BASE/study.db"
STORAGE="sqlite:///$DB_PATH"
LOG="$OUT_BASE/master.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Split trials evenly between workers
HALF=$(( (N_TRIALS_TOTAL + 1) / 2 ))
OTHER=$(( N_TRIALS_TOTAL - HALF ))

# Pre-create the SQLite study schema BEFORE forking the two workers. Without
# this, both workers race on `optuna.create_study(load_if_exists=True)` and
# whichever loses gets `sqlite3.OperationalError: table studies already
# exists` and dies at startup (cost: Worker B silently dead for 4h on
# 2026-05-04). Pre-creating means every subsequent worker just does a
# load_if_exists hit on an already-populated DB — no race possible.
log "==================================================="
log "Cogn-residual HPO launch: study=$STUDY_NAME, total trials=$N_TRIALS_TOTAL"
log "Worker A (GPU 0): up to $HALF trials  →  $OUT_BASE/work_a/"
log "Worker B (GPU 1): up to $OTHER trials →  $OUT_BASE/work_b/"
log "Storage: $STORAGE"
log "Base config: $BASE_CONFIG"
log "Search axes: lr [1e-4, 1e-2], wd [1e-9, 1e-3], n_stages {1,2,3,4},"
log "             aux [0, 5], n_heads {2,4,8,16}, batch_size {16,24,32,48},"
log "             gradient_clip_val [0.3, 2.0]"
log "Per-trial: 5-fold (folds 0..4), MedianPruner reports running mean"
log "==================================================="
log "Pre-creating SQLite study to eliminate worker startup race…"
uv run python -c "
import optuna
optuna.create_study(study_name='$STUDY_NAME', storage='$STORAGE',
                    load_if_exists=True, direction='maximize')
print('study schema ready')
" 2>&1 | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/resdec_mhe/cogn_residual/run_hpo_cogn_residual.py \
    --study-name "$STUDY_NAME" --storage "$STORAGE" \
    --base-config "$BASE_CONFIG" \
    --n-trials "$HALF" \
    --work-dir "$OUT_BASE/work_a" \
    --seed 42 \
    > "$OUT_BASE/worker_a.stdout" 2>&1 &
PID_A=$!
log "Worker A PID: $PID_A"

CUDA_VISIBLE_DEVICES=1 \
  uv run python scripts/resdec_mhe/cogn_residual/run_hpo_cogn_residual.py \
    --study-name "$STUDY_NAME" --storage "$STORAGE" \
    --base-config "$BASE_CONFIG" \
    --n-trials "$OTHER" \
    --work-dir "$OUT_BASE/work_b" \
    --seed 43 \
    > "$OUT_BASE/worker_b.stdout" 2>&1 &
PID_B=$!
log "Worker B PID: $PID_B"

wait "$PID_A"; RC_A=$?
log "Worker A finished, rc=$RC_A"
wait "$PID_B"; RC_B=$?
log "Worker B finished, rc=$RC_B"

log "==================================================="
if [ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ]; then
    log "Both workers succeeded"
    log "HPO_DONE"
else
    log "One or more workers failed (rc_a=$RC_A, rc_b=$RC_B)"
    log "HPO_FAIL_A=$RC_A HPO_FAIL_B=$RC_B"
    exit 1
fi
