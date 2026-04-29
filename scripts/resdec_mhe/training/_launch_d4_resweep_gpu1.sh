#!/usr/bin/env bash
# d4 entropy-reg re-sweep — single-GPU sequential variant for GPU 1
# (GPU 0 reserved for the running F1 vulnerable CF deep-dive).
#
# 5λ × 5 fold = 25 cells × ~25 min/cell ≈ 10.4 hr wall on one RTX 6000 Ada.
# Output: outputs/canonical/p5_entropy_reg_d4/lambda_<λ>/fold<N>/summary.json
#
# Run inside tmux for SIGHUP safety:
#   tmux new-session -d -s d4_resweep_gpu1 \
#     bash scripts/resdec_mhe/training/_launch_d4_resweep_gpu1.sh
set -euo pipefail
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
