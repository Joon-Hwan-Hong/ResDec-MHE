#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Post-sweep job queue. Runs strictly after the entropy_reg sweep finishes.
# Each job is FAULT-TOLERANT: if one fails, the next still starts. Per-job
# logs go to /tmp/post_sweep_<name>.log; queue-state log goes to /tmp/post_sweep_queue.log.
#
# Order (sequential, single-job-per-GPU enforced via tools/check_gpu_free.sh):
#   1. λ=0.01 fold 3 re-run (recovers OOM-failed cell from sweep, ~5 min)
#   2. LMO full 5-fold (joint zero-out, ~45 min)
#   3. GradientSHAP + SmoothGrad full 5-fold (~2-3 hr)
#
# Each job claims GPU 0 once it's free (< 2 GB used by other procs).
#
# Invocation (run inside its own tmux session for SIGHUP safety):
#   tmux new-session -d -s post_sweep_queue -c <worktree-root> \\
#       "bash scripts/resdec_mhe/run_post_sweep_queue.sh 2>&1 | tee /tmp/post_sweep_queue.log"
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
# NOTE: deliberately NO `set -e` so a failed job does not abort the queue.
# Per-job exit codes are handled explicitly via if-blocks.

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$WORKTREE_ROOT"

PYTHONPATH=. export PYTHONPATH

GATE_GPU="${GATE_GPU:-0}"
MAX_USED_MB="${MAX_USED_MB:-2000}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-60}"

log_event() {
    printf '[%s] [post_sweep_queue] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_tmux_done() {
    local session="$1"
    log_event "Waiting for tmux session '$session' to terminate..."
    local n=0
    while tmux has-session -t "$session" 2>/dev/null; do
        sleep 30
        n=$((n + 1))
        if (( n % 4 == 0 )); then
            log_event "  still running; tmux '$session' alive"
        fi
    done
    log_event "tmux session '$session' is gone — proceeding."
}

wait_until_gpu_free() {
    local gpu="$1"
    local max_mb="$2"
    log_event "Waiting for GPU $gpu to drop below ${max_mb} MB used..."
    local n=0
    until bash tools/check_gpu_free.sh "$gpu" "$max_mb" >/dev/null 2>&1; do
        sleep 30
        n=$((n + 1))
        if (( n % 4 == 0 )); then
            local used
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | head -1)
            log_event "  still waiting; GPU $gpu = ${used} MB"
        fi
    done
    log_event "GPU $gpu is free (< ${max_mb} MB used)."
}

run_job() {
    local name="$1"
    local cmd="$2"
    local log="/tmp/post_sweep_${name}.log"
    log_event "→ START: $name (log: $log)"
    local t0
    t0=$(date +%s)
    bash -c "$cmd" >"$log" 2>&1
    local ec=$?
    local t1
    t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if (( ec == 0 )); then
        log_event "  ✓ DONE: $name (exit=0, elapsed=${elapsed}s)"
    else
        log_event "  ✗ FAILED: $name (exit=$ec, elapsed=${elapsed}s, see $log)"
    fi
    return "$ec"
}

# Wait for the upstream sweep tmux session to fully terminate, NOT just for
# GPU memory to drop. The memory-only gate slips through during the brief gap
# between sweep fold launches and triggers OOM (verified during the first
# queue invocation where lambda_0p01_fold3_rerun OOM'd against a
# 44-GB-resident sweep training process). Use SWEEP_TMUX="" to skip.
SWEEP_TMUX="${SWEEP_TMUX:-entropy_reg_sweep}"
if [[ -n "$SWEEP_TMUX" ]]; then
    wait_for_tmux_done "$SWEEP_TMUX"
fi

# ─── Job 1: λ=0.01 fold 3 re-run ────────────────────────────────────────────
wait_until_gpu_free "$GATE_GPU" "$MAX_USED_MB"
run_job "lambda_0p01_fold3_rerun" \
    "CUDA_VISIBLE_DEVICES=$GATE_GPU uv run python scripts/resdec_mhe/training/train.py \
        --config configs/resdec_mhe/entropy_reg.yaml \
        --fold 3 --reg-weight 0.01 \
        --output-dir outputs/canonical/p5_entropy_reg/lambda_0p01" \
    || log_event "  (continuing despite failure)"

# ─── Job 2: LMO full 5-fold ─────────────────────────────────────────────────
wait_until_gpu_free "$GATE_GPU" "$MAX_USED_MB"
run_job "lmo_5fold" \
    "CUDA_VISIBLE_DEVICES=$GATE_GPU uv run python scripts/resdec_mhe/interpretability/run_lmo_zero_out.py" \
    || log_event "  (continuing despite failure)"

# ─── Job 3: GradientSHAP + SmoothGrad full 5-fold ───────────────────────────
wait_until_gpu_free "$GATE_GPU" "$MAX_USED_MB"
run_job "gradshap_smoothgrad_5fold" \
    "CUDA_VISIBLE_DEVICES=$GATE_GPU uv run python scripts/resdec_mhe/interpretability/gradient_shap_smoothgrad_attribution.py \
        --pred-root outputs/canonical/p5_canonical_seed42 \
        --out-dir outputs/canonical/interpretability/captum_robustness \
        --n-steps 50 --internal-batch-size 4 \
        --gs-n-samples 20 --gs-n-baselines 5 --sg-n-samples 10" \
    || log_event "  (continuing despite failure)"

log_event "Queue complete. Per-job logs in /tmp/post_sweep_*.log"
