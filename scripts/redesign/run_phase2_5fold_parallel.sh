#!/usr/bin/env bash
# Parallel 5-fold Phase 2 driver using BOTH GPUs.
# Dispatches folds across available GPUs with a batch-of-N_GPUS pattern
# (mirrors scripts/training/run_sensitivity.sh). Waits for each batch to
# complete before dispatching the next.
#
# Env overrides (all optional):
#   CONFIG        phase YAML (default: configs/redesign/p5_phase2_residual.yaml)
#   OUTROOT       output directory (default: outputs/redesign/p5_phase2_residual)
#   MAX_EPOCHS    override cfg.training.max_epochs (default: unset → config wins)
#   RUN_REINFER   1|0 auto-run reinfer_best_ckpt after training (default: 1)
#   N_GPUS        number of GPUs to use (default: all visible)
#   GPU_LIST      comma-separated GPU list, e.g. "0,1"
#
# Usage:
#   bash scripts/redesign/run_phase2_5fold_parallel.sh
#   MAX_EPOCHS=20 bash scripts/redesign/run_phase2_5fold_parallel.sh
#   RUN_REINFER=0 bash scripts/redesign/run_phase2_5fold_parallel.sh
set -euo pipefail

ROOT="/host/milan/tank/Joon/proj_ml_snrna/.worktrees/redesign-resdec-h3"
CONFIG="${CONFIG:-configs/redesign/p5_phase2_residual.yaml}"
OUTROOT="${OUTROOT:-outputs/redesign/p5_phase2_residual}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
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
        echo "[$(date '+%H:%M:%S')] fold $fold -> GPU $gpu"
        CMD=(CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/redesign/train_resdec.py
             --config "$CONFIG"
             --fold "$fold"
             --output-dir "$OUTROOT")
        if [[ -n "$MAX_EPOCHS" ]]; then
            CMD+=(--max-epochs "$MAX_EPOCHS")
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
        fi
    done
done

echo ""
if [[ "$RUN_REINFER" == "1" ]]; then
    echo "=== Running reinfer_best_ckpt across all folds for val_predictions_best.npz ==="
    # Propagate GPU selection if the caller set one; reinfer driver reads the
    # same env var. Default: reinfer uses whatever GPUs it sees.
    CONFIG="$CONFIG" OUTROOT="$OUTROOT" \
        bash scripts/redesign/run_reinfer_parallel.sh || \
        echo "WARN: reinfer driver returned non-zero; check per-fold *_reinfer.log"
    echo ""
fi

echo "=== All 5 folds attempted. Summarizing... ==="
uv run python -c "
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

root = Path('$OUTROOT')
results = []
for f in range(5):
    preds_path = root / f'fold{f}/val_predictions_final.npz'
    if not preds_path.exists():
        print(f'  fold {f}: no predictions npz (check log at fold{f}/fold{f}_train.log)')
        continue
    d = np.load(preds_path, allow_pickle=True)
    p = d['predictions']; t = d['targets']
    r2 = r2_score(t, p)
    mae = mean_absolute_error(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    pr = pearsonr(p, t).statistic
    sr = spearmanr(p, t).correlation
    results.append({'fold': f, 'r2': r2, 'mae': mae, 'rmse': rmse,
                    'pearson_r': pr, 'spearman_rho': sr, 'n_val': len(t)})
    print(f'  fold {f}: R²={r2:+.4f} MAE={mae:.4f} RMSE={rmse:.4f} r={pr:+.4f} ρ={sr:+.4f} (n={len(t)})')

if len(results) >= 2:
    for key, label in [('r2','R²'), ('mae','MAE'), ('rmse','RMSE'),
                       ('pearson_r','Pearson r'), ('spearman_rho','Spearman ρ')]:
        vals = [r[key] for r in results]
        print(f'  {label:10s}: {np.mean(vals):+.4f} ± {np.std(vals):.4f}')
"
