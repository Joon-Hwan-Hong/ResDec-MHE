#!/usr/bin/env bash
# Launch variant Optuna HPO across 2 GPUs (one worker per GPU, both pulling
# trials from the same SQLite-backed study).
#
# Required env:
#   OUT_BASE       output dir (e.g. outputs/canonical/variants/gpath_only/hpo)
# Optional env:
#   N_TRIALS_TOTAL total trials across both workers (default: 30)
#   STUDY_NAME     default: variant_gpath_only_hpo
#   BASE_CONFIG    default: configs/resdec_mhe/variants/gpath_only.yaml
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This launcher should be run inside tmux to survive SSH disconnect." >&2
    echo "  tmux new -s variant_hpo 'bash $0'" >&2
    exit 1
fi

: "${OUT_BASE:?OUT_BASE required}"
N_TRIALS_TOTAL="${N_TRIALS_TOTAL:-30}"
STUDY_NAME="${STUDY_NAME:-variant_gpath_only_hpo}"
BASE_CONFIG="${BASE_CONFIG:-configs/resdec_mhe/variants/gpath_only.yaml}"

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WT"

mkdir -p "$OUT_BASE/work_a" "$OUT_BASE/work_b"
DB_PATH="$WT/$OUT_BASE/study.db"
STORAGE="sqlite:///$DB_PATH"
LOG="$OUT_BASE/master.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Split trials evenly between workers
HALF=$(( (N_TRIALS_TOTAL + 1) / 2 ))
OTHER=$(( N_TRIALS_TOTAL - HALF ))

log "==================================================="
log "Variant HPO launch: study=$STUDY_NAME, total trials=$N_TRIALS_TOTAL"
log "Worker A (GPU 0): up to $HALF trials  →  $OUT_BASE/work_a/"
log "Worker B (GPU 1): up to $OTHER trials →  $OUT_BASE/work_b/"
log "Storage: $STORAGE"
log "Base config: $BASE_CONFIG"
log "==================================================="

CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/resdec_mhe/variants/run_hpo_variant.py \
    --study-name "$STUDY_NAME" --storage "$STORAGE" \
    --base-config "$BASE_CONFIG" \
    --n-trials "$HALF" \
    --work-dir "$OUT_BASE/work_a" \
    --seed 42 \
    > "$OUT_BASE/worker_a.stdout" 2>&1 &
PID_A=$!
log "Worker A PID: $PID_A"

CUDA_VISIBLE_DEVICES=1 \
  uv run python scripts/resdec_mhe/variants/run_hpo_variant.py \
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
else
    log "One or more workers failed (rc_a=$RC_A, rc_b=$RC_B)"
    exit 1
fi
