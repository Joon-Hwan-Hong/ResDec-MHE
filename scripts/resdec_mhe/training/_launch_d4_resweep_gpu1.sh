#!/usr/bin/env bash
# d4 entropy-reg re-sweep — single-GPU sequential variant for GPU 1
# (GPU 0 reserved for the running F1 vulnerable CF deep-dive).
#
# Naming: the ``_launch`` prefix follows the codebase convention that this
# is a MANUAL entry-point, not auto-invoked by another driver. Single-GPU
# sequential is intentional (B-D43): GPU 0 is being used by F1 deep-dive,
# so we cannot parallelize across both GPUs.
#
# 5λ × 5 fold = 25 cells × ~25 min/cell ≈ 10.4 hr wall on one RTX 6000 Ada.
# Output: outputs/canonical/p5_entropy_reg_d4/lambda_<λ>/fold<N>/summary.json
#
# Run inside tmux for SIGHUP safety:
#   tmux new-session -d -s d4_resweep_gpu1 \
#     bash scripts/resdec_mhe/training/_launch_d4_resweep_gpu1.sh
set -euo pipefail

# tmux preflight (feedback_long_runs_need_tmux.md): 25 cells × ~25 min/cell ≈
# 10.4 hr wall — 20× the 30-min threshold. Bare SSH loses the run on
# disconnect.
if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This sweep runs ~10 hr and must be in tmux to survive SSH disconnect." >&2
    echo "  tmux new -s d4_resweep_gpu1 'bash $0'" >&2
    exit 1
fi

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$WORKTREE_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/canonical/p5_entropy_reg_d4}"
LAMBDAS=(0 0.001 0.01 0.1 1.0)
FOLDS=(0 1 2 3 4)

mkdir -p "$OUT_ROOT/logs"
echo "[d4_resweep_gpu1] start $(date -Iseconds) | OUT_ROOT=$OUT_ROOT"

for lam in "${LAMBDAS[@]}"; do
    lam_dir="$(printf 'lambda_%s' "$lam" | tr '.' 'p')"
    out_dir="$OUT_ROOT/$lam_dir"
    mkdir -p "$out_dir"
    for fold in "${FOLDS[@]}"; do
        log="$OUT_ROOT/logs/${lam_dir}_fold${fold}.log"
        echo "[d4_resweep_gpu1] $(date -Iseconds) | λ=$lam fold=$fold → $log"
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \
            uv run python scripts/resdec_mhe/training/train.py \
                --config configs/resdec_mhe/entropy_reg.yaml \
                --fold "$fold" \
                --reg-weight "$lam" \
                --output-dir "$out_dir" \
                > "$log" 2>&1
        echo "[d4_resweep_gpu1] $(date -Iseconds) | λ=$lam fold=$fold DONE"
    done
    echo "[d4_resweep_gpu1] λ=$lam complete"
done
echo "[d4_resweep_gpu1] sweep complete $(date -Iseconds)"
