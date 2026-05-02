#!/usr/bin/env bash
# Perm-shard parallel launcher for N=50 full-pipeline permutation null.
# Shard A (GPU 0) runs perms 0-24; Shard B (GPU 1) runs perms 25-49.
# Each shard runs 5 folds sequentially within each perm on its single GPU.
# Expected wall: ~16.25 hr (vs ~19.4 hr for fold-shard within-perm strategy).
set -uo pipefail   # NO -e — continue on per-perm failures

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This launcher runs ~16+ hr and must be in tmux to survive SSH disconnect." >&2
    echo "  tmux new -d -s permnull_n50_shard 'bash $0'" >&2
    exit 1
fi

WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$WT"

# Allow env override so re-runs (e.g. N=50 v2) write to a distinct directory
# rather than colliding with the original.
OUT_BASE="${OUT_BASE:-outputs/canonical/permutation_test_n50_full}"
mkdir -p "$OUT_BASE/shard_a" "$OUT_BASE/shard_b"

LOG="$WT/$OUT_BASE/master.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "==================================================="
log "N=50 full-pipeline permutation null — perm-shard launch"
log "Shard A (GPU 0): perms 0-24 → $OUT_BASE/shard_a/"
log "Shard B (GPU 1): perms 25-49 → $OUT_BASE/shard_b/"
log "==================================================="

# Note on CUDA visibility stacking: parent sets CUDA_VISIBLE_DEVICES=N to
# pin a physical GPU; the child sees that as logical GPU 0, so we pass
# --gpus 0 (logical index relative to parent's mask). Result: Shard B truly
# runs on physical GPU 1 even though the inner script sees logical GPU 0.

# Shard A: GPU 0, perms 0-24
log "Launching Shard A (GPU 0, perms 0-24)"
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/resdec_mhe/training/run_permutation_test.py \
    --num-perms 25 --start-perm 0 \
    --output-base "$OUT_BASE/shard_a" \
    --gpus 0 \
    > "$OUT_BASE/shard_a.stdout" 2>&1 &
PID_A=$!
log "Shard A PID: $PID_A"

# Shard B: GPU 1, perms 25-49
log "Launching Shard B (GPU 1, perms 25-49)"
CUDA_VISIBLE_DEVICES=1 \
  uv run python scripts/resdec_mhe/training/run_permutation_test.py \
    --num-perms 25 --start-perm 25 \
    --output-base "$OUT_BASE/shard_b" \
    --gpus 0 \
    > "$OUT_BASE/shard_b.stdout" 2>&1 &
PID_B=$!
log "Shard B PID: $PID_B"

# Wait for both
log "Waiting for both shards to complete..."
wait "$PID_A"
RC_A=$?
log "Shard A finished, rc=$RC_A"

wait "$PID_B"
RC_B=$?
log "Shard B finished, rc=$RC_B"

log "==================================================="
if [ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ]; then
    log "Both shards succeeded — invoking aggregator"
    uv run python scripts/resdec_mhe/training/aggregate_permnull_n50_shards.py 2>&1 | tee -a "$LOG"
    log "Aggregation done"
else
    log "One or more shards failed (rc_a=$RC_A, rc_b=$RC_B); aggregation skipped"
    log "Run manually after fixing failed perms:"
    log "  uv run python scripts/resdec_mhe/training/aggregate_permnull_n50_shards.py"
fi
log "==================================================="
