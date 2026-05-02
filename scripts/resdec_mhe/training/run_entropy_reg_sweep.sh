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

# tmux preflight (feedback_long_runs_need_tmux.md): full sweep is ~5 hr
# 2-GPU. Smoke mode (SMOKE=1) is ~2 min and is exempt below.
if [[ "${SMOKE:-0}" != "1" ]] && [ -z "${TMUX:-}" ]; then
    echo "ERROR: This sweep runs ~5 hr and must be in tmux to survive SSH disconnect." >&2
    echo "  tmux new -s entropy_reg_sweep 'bash $0'" >&2
    echo "  (or set SMOKE=1 for a 2-min plumbing check that bypasses this guard)" >&2
    exit 1
fi

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$WORKTREE_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/canonical/p5_entropy_reg}"
LAMBDAS=(0 0.001 0.01 0.1 1.0)
FOLDS=(0 1 2 3 4)

# EXTRA_FLAGS as bash array (B-ER3: was a string subjected to word-splitting;
# any value containing spaces would have broken). Empty array expands to
# nothing in the call site below.
declare -a EXTRA_FLAGS=()

if [[ "${SMOKE:-0}" == "1" ]]; then
    LAMBDAS=(0.01)
    FOLDS=(0)
    EXTRA_FLAGS=(--max-epochs 1)
    echo "[entropy_reg_sweep] SMOKE mode: λ=0.01, fold=0, max-epochs=1"
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
            "${EXTRA_FLAGS[@]}" \
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
    # B-ER2: With 5 folds the fifth runs solo on GPU 0 while GPU 1 sits idle
    # (~25 min lost per λ × 5 λs ≈ 2hr). Acceptable for this one-shot 25-cell
    # sweep — a true worker-pool refactor would centralise this with the
    # similar pattern in run_5fold_parallel.sh; deferred until reused.
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
