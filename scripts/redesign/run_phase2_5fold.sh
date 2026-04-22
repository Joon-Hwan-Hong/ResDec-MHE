#!/usr/bin/env bash
# Run ResDec-H3 Phase 2 (TabPFN residual integration) across all 5 outer folds
# sequentially on a single GPU. Each fold trains for 60 epochs at bs=24 (live
# encoder + head joint training).
#
# Writes per-fold outputs to outputs/redesign/p5_phase2_residual/fold{0..4}/
# with the val_predictions_final.npz + summary.json containing full metrics
# (R², MAE, RMSE, Pearson r, Spearman ρ).
#
# Usage: PYTHONPATH=<worktree-root> CUDA_VISIBLE_DEVICES=0 bash scripts/redesign/run_phase2_5fold.sh
set -euo pipefail

ROOT="/host/milan/tank/Joon/proj_ml_snrna/.worktrees/redesign-resdec-h3"
CONFIG="configs/redesign/p5_phase2_residual.yaml"
OUTROOT="outputs/redesign/p5_phase2_residual"

export PYTHONPATH="${PYTHONPATH:-$ROOT}"

for fold in 0 1 2 3 4; do
  out="$OUTROOT/fold${fold}"
  mkdir -p "$out"
  echo "=== $(date '+%H:%M:%S') fold $fold ==="
  uv run python scripts/redesign/train_resdec.py \
      --config "$CONFIG" \
      --fold "$fold" \
      --max-epochs 60 \
      --output-dir "$OUTROOT" \
      > "$out/fold${fold}_train.log" 2>&1
  echo "=== $(date '+%H:%M:%S') fold $fold done ==="
done

echo ""
echo "=== All 5 folds complete. Summarizing... ==="
uv run python -c "
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

root = Path('$OUTROOT')
results = []
for f in range(5):
    summary_path = root / f'fold{f}/summary.json'
    preds_path = root / f'fold{f}/val_predictions_final.npz'
    if not summary_path.exists() or not preds_path.exists():
        print(f'  fold {f}: missing outputs')
        continue
    s = json.loads(summary_path.read_text())
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

if results:
    r2s = [r['r2'] for r in results]
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    prs = [r['pearson_r'] for r in results]
    srs = [r['spearman_rho'] for r in results]
    print()
    print(f'Mean ± std over {len(results)} folds:')
    print(f'  R²:       {np.mean(r2s):+.4f} ± {np.std(r2s):.4f}')
    print(f'  MAE:      {np.mean(maes):.4f} ± {np.std(maes):.4f}')
    print(f'  RMSE:     {np.mean(rmses):.4f} ± {np.std(rmses):.4f}')
    print(f'  Pearson:  {np.mean(prs):+.4f} ± {np.std(prs):.4f}')
    print(f'  Spearman: {np.mean(srs):+.4f} ± {np.std(srs):.4f}')
"
