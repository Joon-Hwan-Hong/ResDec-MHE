#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Entropy-bonus attention regularization sweep driver.
#
# Sweeps λ ∈ {0, 1e-3, 1e-2, 1e-1, 1.0} × fold ∈ {0..4} = 25 (λ, fold) cells.
# Splits across two GPUs: even folds → GPU 0, odd folds → GPU 1.
# Each cell ≈ 25 min wall on one RTX 6000 Ada → total ~5 hr 2-GPU.
#
# Run inside tmux for SIGHUP safety:
#   tmux new -s entropy_reg_sweep
#   bash scripts/resdec_mhe/training/run_entropy_reg_sweep.sh
#
# Smoke (1 fold × 1 λ × 1 epoch, ~2 min) for plumbing verification:
#   SMOKE=1 bash scripts/resdec_mhe/training/run_entropy_reg_sweep.sh
#
# Output goes to outputs/canonical/p5_entropy_reg/lambda_<λ>/fold<N>/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$WORKTREE_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/canonical/p5_entropy_reg}"
LAMBDAS=(0 0.001 0.01 0.1 1.0)
FOLDS=(0 1 2 3 4)

if [[ "${SMOKE:-0}" == "1" ]]; then
    LAMBDAS=(0.01)
    FOLDS=(0)
    EXTRA_FLAGS="--max-epochs 1"
    echo "[entropy_reg_sweep] SMOKE mode: λ=0.01, fold=0, max-epochs=1"
else
    EXTRA_FLAGS=""
fi

mkdir -p "$OUT_ROOT/logs"

declare -a PIDS_GPU0 PIDS_GPU1

launch_one() {
    local lam="$1"
    local fold="$2"
    local gpu="$3"
    local lam_dir
    lam_dir="$(printf 'lambda_%s' "$lam" | tr '.' 'p')"
    local out_dir="$OUT_ROOT/$lam_dir"
    local log="$OUT_ROOT/logs/${lam_dir}_fold${fold}.log"
    mkdir -p "$out_dir"
    echo "[entropy_reg_sweep] GPU $gpu | λ=$lam fold=$fold → $log"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$gpu" \
        uv run python scripts/resdec_mhe/training/train.py \
            --config configs/resdec_mhe/entropy_reg.yaml \
            --fold "$fold" \
            --reg-weight "$lam" \
            --output-dir "$out_dir" \
            $EXTRA_FLAGS \
            > "$log" 2>&1 &
    if [[ "$gpu" == "0" ]]; then
        PIDS_GPU0+=("$!")
    else
        PIDS_GPU1+=("$!")
    fi
}

wait_gpu() {
    local gpu="$1"
    if [[ "$gpu" == "0" ]]; then
        if (( ${#PIDS_GPU0[@]} > 0 )); then
            for pid in "${PIDS_GPU0[@]}"; do
                wait "$pid" || echo "[entropy_reg_sweep] GPU0 pid $pid failed"
            done
        fi
        PIDS_GPU0=()
    else
        if (( ${#PIDS_GPU1[@]} > 0 )); then
            for pid in "${PIDS_GPU1[@]}"; do
                wait "$pid" || echo "[entropy_reg_sweep] GPU1 pid $pid failed"
            done
        fi
        PIDS_GPU1=()
    fi
}

for lam in "${LAMBDAS[@]}"; do
    # Process folds in pairs so at most ONE job runs per GPU at any time.
    # Pairs: (0,1), (2,3), (4,) — last fold in odd-count case waits alone.
    n="${#FOLDS[@]}"
    for (( i=0; i<n; i+=2 )); do
        f0="${FOLDS[$i]}"
        launch_one "$lam" "$f0" "0"
        if (( i+1 < n )); then
            f1="${FOLDS[$((i+1))]}"
            launch_one "$lam" "$f1" "1"
        fi
        # Wait for this pair to finish before launching the next.
        wait_gpu 0
        wait_gpu 1
    done
    echo "[entropy_reg_sweep] λ=$lam complete"
done

echo "[entropy_reg_sweep] Sweep complete. Results under $OUT_ROOT/"
