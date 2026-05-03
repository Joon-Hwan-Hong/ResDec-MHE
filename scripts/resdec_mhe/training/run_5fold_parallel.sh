#!/usr/bin/env bash
# Parallel 5-fold ResDec-MHE training driver using BOTH GPUs.
# Dispatches folds across available GPUs with a batch-of-N_GPUS pattern
# (mirrors scripts/training/run_sensitivity.sh). Waits for each batch to
# complete before dispatching the next.
#
# Env overrides (all optional):
#   CONFIG          phase YAML (default: configs/resdec_mhe/canonical.yaml)
#   OUTROOT         output directory (default: outputs/canonical/p5_phase2_residual)
#   MAX_EPOCHS      override cfg.training.max_epochs (default: unset → config wins)
#   SEED            override cfg.experiment.seed (default: unset → config wins; e.g. 43)
#   METADATA_PATH   override cfg.data.metadata_path (e.g. data/metadata_ROSMAP)
#   PRECOMPUTED_DIR override cfg.data.precomputed_dir (e.g. data/precomputed)
#   RUN_REINFER     1|0 auto-run reinfer_best_ckpt after training (default: 1)
#   N_GPUS          number of GPUs to use (default: all visible)
#   GPU_LIST        comma-separated GPU list, e.g. "0,1"
#
# Usage:
#   bash scripts/resdec_mhe/training/run_5fold_parallel.sh
#   MAX_EPOCHS=20 bash scripts/resdec_mhe/training/run_5fold_parallel.sh
#   RUN_REINFER=0 bash scripts/resdec_mhe/training/run_5fold_parallel.sh
set -euo pipefail

# tmux preflight (feedback_long_runs_need_tmux.md): 5-fold ResDec-MHE on 2 GPUs
# is ~1-2 hr; with RUN_REINFER=1 it climbs to 1.5-2.5 hr — past the 30-min
# threshold. SSH disconnect without tmux kills the run.
if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This driver runs 1-2.5 hr and must be in tmux to survive SSH disconnect." >&2
    echo "  tmux new -s resdec_5fold 'bash $0'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIG="${CONFIG:-configs/resdec_mhe/canonical.yaml}"
OUTROOT="${OUTROOT:-outputs/canonical/p5_phase2_residual}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
SEED="${SEED:-}"
METADATA_PATH="${METADATA_PATH:-}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-}"
RUN_REINFER="${RUN_REINFER:-1}"
FOLDS=(0 1 2 3 4)

export PYTHONPATH="${PYTHONPATH:-$ROOT}"

# Which GPUs to dispatch folds to. Comma-separated in cycle order.
if [[ -n "${GPU_LIST:-}" ]]; then
    IFS=',' read -ra GPUS <<< "$GPU_LIST"
else
    N=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
    GPUS=()
    for ((i = 0; i < N; i++)); do GPUS+=("$i"); done
fi
N_GPUS_EFF=${#GPUS[@]}
echo "Using GPUs: ${GPUS[*]} (N=$N_GPUS_EFF)"

mkdir -p "$OUTROOT"

# Track per-fold failures so the outer script can propagate non-zero exit if
# any fold failed. Without this counter the summary heredoc would run on
# degenerate input and the script would return 0.
N_FAILED=0

# Walk folds in batches of N_GPUS_EFF
idx=0
while (( idx < ${#FOLDS[@]} )); do
    PIDS=()
    DESCR=()
    for ((g = 0; g < N_GPUS_EFF; g++)); do
        if (( idx >= ${#FOLDS[@]} )); then break; fi
        fold=${FOLDS[$idx]}
        gpu=${GPUS[$g]}
        out="$OUTROOT/fold${fold}"
        mkdir -p "$out"
        # Skip-on-success resume: if the per-fold summary.json already exists,
        # the fold completed successfully on a prior run. Mirrors the same
        # idempotency contract in run_diff_test_5fold.sh:14-19.
        summary="$out/summary.json"
        if [[ -f "$summary" ]]; then
            echo "[$(date '+%H:%M:%S')] fold $fold already done — skipping"
            idx=$((idx + 1))
            continue
        fi
        echo "[$(date '+%H:%M:%S')] fold $fold -> GPU $gpu"
        # Per feedback_cuda_visible_devices_subprocess.md: when this script
        # runs in single-GPU pinned-shard mode (parent shell sets
        # CUDA_VISIBLE_DEVICES=N to assign a physical GPU and we see N_GPUS_EFF=1),
        # do NOT prepend CUDA_VISIBLE_DEVICES=$gpu — the literal "0" override
        # is ABSOLUTE in subprocess env and would force the child onto physical
        # GPU 0 even when the parent pinned us to physical GPU 1, fighting any
        # parallel shard. Inherit parent's mask in single-GPU mode; only
        # override when the parent has multiple GPUs and we need to assign one.
        if (( N_GPUS_EFF > 1 )); then
            CMD=(CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/resdec_mhe/training/train.py
                 --config "$CONFIG"
                 --fold "$fold"
                 --output-dir "$OUTROOT")
        else
            CMD=(uv run python scripts/resdec_mhe/training/train.py
                 --config "$CONFIG"
                 --fold "$fold"
                 --output-dir "$OUTROOT")
        fi
        if [[ -n "$MAX_EPOCHS" ]]; then
            CMD+=(--max-epochs "$MAX_EPOCHS")
        fi
        if [[ -n "$SEED" ]]; then
            CMD+=(--seed "$SEED")
        fi
        if [[ -n "$METADATA_PATH" ]]; then
            CMD+=(--metadata-path "$METADATA_PATH")
        fi
        if [[ -n "$PRECOMPUTED_DIR" ]]; then
            CMD+=(--precomputed-dir "$PRECOMPUTED_DIR")
        fi
        env "${CMD[@]}" > "$out/fold${fold}_train.log" 2>&1 &
        PIDS+=($!)
        DESCR+=("fold${fold}:gpu${gpu}:pid=$!")
        idx=$((idx + 1))
    done

    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "[$(date '+%H:%M:%S')] done ${DESCR[$i]}"
        else
            echo "[$(date '+%H:%M:%S')] FAILED ${DESCR[$i]}"
            N_FAILED=$((N_FAILED + 1))
        fi
    done
done

echo ""
if [[ "$RUN_REINFER" == "1" ]]; then
    echo "=== Running reinfer_best_ckpt across all folds for val_predictions_best.npz ==="
    # Propagate GPU selection if the caller set one; reinfer driver reads the
    # same env var. Default: reinfer uses whatever GPUs it sees.
    CONFIG="$CONFIG" OUTROOT="$OUTROOT" \
        bash scripts/resdec_mhe/training/run_reinfer_parallel.sh || \
        echo "WARN: reinfer driver returned non-zero; check per-fold *_reinfer.log"
    echo ""
fi

echo "=== All 5 folds attempted. Summarizing... ==="
# Pass OUTROOT via env so the heredoc stays quoted (no $-substitution surprises).
OUTROOT="$OUTROOT" uv run python - <<'PY'
import os  # only for os.environ — path ops use pathlib below
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

root = Path(os.environ["OUTROOT"])
results = []
for f in range(5):
    preds_path = root / f"fold{f}/val_predictions_final.npz"
    if not preds_path.exists():
        print(f"  fold {f}: no predictions npz (check log at fold{f}/fold{f}_train.log)")
        continue
    d = np.load(preds_path, allow_pickle=True)
    p = d["predictions"]; t = d["targets"]
    r2 = r2_score(t, p)
    mae = mean_absolute_error(t, p)
    rmse = float(np.sqrt(mean_squared_error(t, p)))
    pr = float(pearsonr(p, t).statistic)
    sr = float(spearmanr(p, t).correlation)
    results.append({"fold": f, "r2": r2, "mae": mae, "rmse": rmse,
                    "pearson_r": pr, "spearman_rho": sr, "n_val": len(t)})
    print(f"  fold {f}: R²={r2:+.4f} MAE={mae:.4f} RMSE={rmse:.4f} r={pr:+.4f} ρ={sr:+.4f} (n={len(t)})")

if len(results) >= 2:
    for key, label in [("r2", "R²"), ("mae", "MAE"), ("rmse", "RMSE"),
                       ("pearson_r", "Pearson r"), ("spearman_rho", "Spearman ρ")]:
        vals = [r[key] for r in results]
        print(f"  {label:10s}: {np.mean(vals):+.4f} ± {np.std(vals):.4f}")
PY

# Propagate per-fold failure count as the script's exit code so callers
# (run_seed_variation.sh + CI invocations) see non-zero on partial failure.
if (( N_FAILED > 0 )); then
    echo "$N_FAILED fold(s) failed during training" >&2
    exit 1
fi
