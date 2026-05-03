#!/usr/bin/env bash
# Variant perm-null sharded launcher.
# Splits N permutations across 2 GPUs (shard A: perms [0, ceil(N/2)),
# shard B: perms [ceil(N/2), N)). Each shard runs sequentially within
# its perms, with all 5 folds on its single pinned GPU.
#
# Required env vars:
#   N_PERMS              total permutation count (e.g. 20)
#   VARIANT_CONFIG       configs/resdec_mhe/variants/<variant>.yaml
#   RESIDUAL_CACHE_DIR   variant residual cache (residual_target_fold{0..4}.npz)
#   OUT_BASE             output dir (e.g. outputs/canonical/variants/<variant>/permutation_test_n20)
#
# Optional env vars:
#   METADATA_PATH        default data/metadata_ROSMAP
#   PRECOMPUTED_DIR      default data/precomputed
#   SPLITS_PATH          default outputs/splits.json
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This launcher runs ~hours and must be in tmux to survive SSH disconnect." >&2
    echo "  tmux new -s permnull_variant 'bash $0'" >&2
    exit 1
fi

: "${N_PERMS:?N_PERMS required (e.g. 20)}"
: "${VARIANT_CONFIG:?VARIANT_CONFIG required}"
: "${RESIDUAL_CACHE_DIR:?RESIDUAL_CACHE_DIR required}"
: "${OUT_BASE:?OUT_BASE required}"

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$WT"

METADATA_PATH="${METADATA_PATH:-data/metadata_ROSMAP}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-data/precomputed}"
SPLITS_PATH="${SPLITS_PATH:-outputs/splits.json}"

mkdir -p "$OUT_BASE/shard_a" "$OUT_BASE/shard_b"
LOG="$OUT_BASE/master.log"
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

SHARD_A_N=$(( (N_PERMS + 1) / 2 ))
SHARD_B_N=$(( N_PERMS - SHARD_A_N ))
SHARD_A_START=0
SHARD_B_START=$SHARD_A_N

log "==================================================="
log "Variant perm-null sharded launch"
log "Total perms: $N_PERMS"
log "Shard A (GPU 0, pinned via CUDA_VISIBLE_DEVICES=0): perms [$SHARD_A_START, $SHARD_A_START + $SHARD_A_N) -> $OUT_BASE/shard_a/"
log "Shard B (GPU 1, pinned via CUDA_VISIBLE_DEVICES=1): perms [$SHARD_B_START, $SHARD_B_START + $SHARD_B_N) -> $OUT_BASE/shard_b/"
log "Variant config: $VARIANT_CONFIG"
log "Residual cache: $RESIDUAL_CACHE_DIR"
log "==================================================="

# Per feedback_cuda_visible_devices_subprocess.md: parent CUDA_VISIBLE_DEVICES=N
# pins each shard to one physical GPU; the inner script must inherit that mask
# (run_permutation_test_variant.py passes --gpus 0 which becomes physical-N).

log "Launching Shard A on GPU 0"
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/resdec_mhe/variants/run_permutation_test_variant.py \
    --num-perms "$SHARD_A_N" --start-perm "$SHARD_A_START" \
    --output-base "$OUT_BASE/shard_a" \
    --variant-config "$VARIANT_CONFIG" \
    --residual-cache-dir "$RESIDUAL_CACHE_DIR" \
    --metadata-path "$METADATA_PATH" \
    --precomputed-dir "$PRECOMPUTED_DIR" \
    --splits-path "$SPLITS_PATH" \
    --gpus 0 \
    > "$OUT_BASE/shard_a.stdout" 2>&1 &
PID_A=$!
log "Shard A PID: $PID_A"

if [ "$SHARD_B_N" -gt 0 ]; then
    log "Launching Shard B on GPU 1"
    CUDA_VISIBLE_DEVICES=1 \
      uv run python scripts/resdec_mhe/variants/run_permutation_test_variant.py \
        --num-perms "$SHARD_B_N" --start-perm "$SHARD_B_START" \
        --output-base "$OUT_BASE/shard_b" \
        --variant-config "$VARIANT_CONFIG" \
        --residual-cache-dir "$RESIDUAL_CACHE_DIR" \
        --metadata-path "$METADATA_PATH" \
        --precomputed-dir "$PRECOMPUTED_DIR" \
        --splits-path "$SPLITS_PATH" \
        --gpus 0 \
        > "$OUT_BASE/shard_b.stdout" 2>&1 &
    PID_B=$!
    log "Shard B PID: $PID_B"
else
    log "Shard B skipped (N_PERMS too small)"
    PID_B=""
fi

log "Waiting for shards..."
wait "$PID_A"
RC_A=$?
log "Shard A finished, rc=$RC_A"
if [ -n "$PID_B" ]; then
    wait "$PID_B"
    RC_B=$?
    log "Shard B finished, rc=$RC_B"
else
    RC_B=0
fi

log "==================================================="
if [ "$RC_A" -eq 0 ] && [ "$RC_B" -eq 0 ]; then
    log "Both shards succeeded"
else
    log "One or more shards failed (rc_a=$RC_A, rc_b=$RC_B)"
    exit 1
fi
log "==================================================="
