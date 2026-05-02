#!/usr/bin/env bash
# F1 optimized-code 4-config 2-GPU grid:
#   target_mode ∈ {relative, absolute} × target_delta ∈ {0.5, 0.3}
#
# Phase 1 (parallel):  GPU 0 → relative δ=0.5  |  GPU 1 → absolute δ=0.5
# Phase 2 (parallel):  GPU 0 → relative δ=0.3  |  GPU 1 → absolute δ=0.3
#
# Per-config wall: 30-90 min on optimized code (P1.1 + P1.2 + P1.3 + P1.4 + P1.6).
# Total wall: 1-3 hr on 2 GPUs × 2 sequential phases.
#
# Run inside tmux for SIGHUP safety (memory rule feedback_long_runs_need_tmux):
#   tmux new-session -d -s f1_opt_grid \
#     bash scripts/resdec_mhe/interpretability/_launch_f1_optimized_grid.sh
set -euo pipefail

# Tmux preflight: 1-3 hr total wall (per docstring above) is well above the
# 30-min SIGHUP-safety threshold. Force tmux usage; override with
# FORCE_NO_TMUX=1 for short smoke tests.
if [ -z "${TMUX:-}" ] && [ -z "${FORCE_NO_TMUX:-}" ]; then
    echo "ERROR: 1-3 hr run; must be in tmux." >&2
    echo "       Run inside tmux, or set FORCE_NO_TMUX=1 to override." >&2
    exit 1
fi

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$WORKTREE_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/canonical/interpretability}"
TOL="${F1_TOL:-1e-2}"
LAMBDA_MAX="${F1_LAMBDA_MAX:-1e4}"
MAX_STEPS="${F1_MAX_STEPS:-1000}"

run_config() {
    local mode="$1"
    local delta="$2"
    local gpu="$3"
    local tag
    tag="$(printf '%s_delta%s' "$mode" "$delta" | tr '.' 'p')"
    local out_dir="$OUT_ROOT/counterfactuals_optimized_${tag}"
    local log="$out_dir/run.log"
    mkdir -p "$out_dir"
    echo "[f1_opt_grid] $(date -Iseconds) | GPU $gpu | mode=$mode delta=$delta → $log"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
            --target-mode "$mode" \
            --target-delta "$delta" \
            --tol "$TOL" \
            --lambda-max "$LAMBDA_MAX" \
            --max-steps "$MAX_STEPS" \
            --out-dir "$out_dir" \
            --device "cuda:0" \
            > "$log" 2>&1
    echo "[f1_opt_grid] $(date -Iseconds) | GPU $gpu | mode=$mode delta=$delta DONE"
}

echo "[f1_opt_grid] start $(date -Iseconds) | OUT_ROOT=$OUT_ROOT TOL=$TOL LAMBDA_MAX=$LAMBDA_MAX MAX_STEPS=$MAX_STEPS"

# Phase 1: δ=0.5 across both modes
# `wait || { exit 1; }` is intentional: if either GPU fails phase 1, we
# abort before launching phase 2. The earlier `wait || echo ...` form
# swallowed the non-zero exit and silently launched phase 2 on top of a
# broken phase 1.
run_config relative 0.5 0 &
PID0_5_REL=$!
run_config absolute 0.5 1 &
PID0_5_ABS=$!
wait "$PID0_5_REL" || { echo "[f1_opt_grid] GPU0 rel-0.5 failed (pid $PID0_5_REL); aborting phase 2"; exit 1; }
wait "$PID0_5_ABS" || { echo "[f1_opt_grid] GPU1 abs-0.5 failed (pid $PID0_5_ABS); aborting phase 2"; exit 1; }
echo "[f1_opt_grid] $(date -Iseconds) | Phase 1 (δ=0.5) complete"

# Phase 2: δ=0.3 across both modes
run_config relative 0.3 0 &
PID0_3_REL=$!
run_config absolute 0.3 1 &
PID0_3_ABS=$!
wait "$PID0_3_REL" || { echo "[f1_opt_grid] GPU0 rel-0.3 failed (pid $PID0_3_REL)"; exit 1; }
wait "$PID0_3_ABS" || { echo "[f1_opt_grid] GPU1 abs-0.3 failed (pid $PID0_3_ABS)"; exit 1; }
echo "[f1_opt_grid] $(date -Iseconds) | Phase 2 (δ=0.3) complete"

echo "[f1_opt_grid] ALL CONFIGS COMPLETE $(date -Iseconds)"
