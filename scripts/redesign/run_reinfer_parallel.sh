#!/usr/bin/env bash
# Parallel 5-fold re-inference of best-by-val/r2 checkpoints across both GPUs.
# Per fold: loads max-R² best-*.ckpt, runs validate(), dumps val_predictions_best.npz
# + best_summary.json into outputs/redesign/p5_phase2_residual/fold{N}/.
#
# Usage:
#   bash scripts/redesign/run_reinfer_parallel.sh
#   N_GPUS=2 bash scripts/redesign/run_reinfer_parallel.sh
#   GPU_LIST="0,1" bash scripts/redesign/run_reinfer_parallel.sh
set -euo pipefail

ROOT="/host/milan/tank/Joon/proj_ml_snrna/.worktrees/redesign-resdec-h3"
CONFIG="configs/redesign/p5_phase2_residual.yaml"
OUTROOT="outputs/redesign/p5_phase2_residual"
FOLDS=(0 1 2 3 4)

export PYTHONPATH="${PYTHONPATH:-$ROOT}"

if [[ -n "${GPU_LIST:-}" ]]; then
    IFS=',' read -ra GPUS <<< "$GPU_LIST"
else
    N=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
    GPUS=()
    for ((i = 0; i < N; i++)); do GPUS+=("$i"); done
fi
N_GPUS_EFF=${#GPUS[@]}
echo "Using GPUs: ${GPUS[*]} (N=$N_GPUS_EFF)"

idx=0
while (( idx < ${#FOLDS[@]} )); do
    PIDS=()
    DESCR=()
    for ((g = 0; g < N_GPUS_EFF; g++)); do
        if (( idx >= ${#FOLDS[@]} )); then break; fi
        fold=${FOLDS[$idx]}
        gpu=${GPUS[$g]}
        out="$OUTROOT/fold${fold}"
        echo "[$(date '+%H:%M:%S')] fold $fold -> GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/redesign/reinfer_best_ckpt.py \
            --config "$CONFIG" \
            --fold "$fold" \
            --output-dir "$OUTROOT" \
            > "$out/fold${fold}_reinfer.log" 2>&1 &
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
echo "=== Reinfer done. Summarizing best-epoch metrics + TabPFN comparison... ==="
uv run python - <<'PY'
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

root = Path("outputs/redesign/p5_phase2_residual")
tabpfn_dir = Path("data/redesign")

rows = []
for f in range(5):
    ours = root / f"fold{f}/val_predictions_best.npz"
    if not ours.exists():
        print(f"fold {f}: val_predictions_best.npz MISSING")
        continue
    d = np.load(ours, allow_pickle=True)
    our_ids = d["subject_ids"].astype(str)
    our_p = d["predictions"].astype(np.float32)
    our_t = d["targets"].astype(np.float32)

    tab_ge = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
    tab_en = tabpfn_dir / f"tabpfn_outer_fold{f}_A+C+E+P+R.npz"

    def _metrics(y_true, y_pred):
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "pearson_r": float(pearsonr(y_pred, y_true).statistic),
            "spearman_rho": float(spearmanr(y_pred, y_true).correlation),
        }

    ours_m = _metrics(our_t, our_p)
    tab_ge_m = None
    tab_en_m = None
    if tab_ge.exists():
        t = np.load(tab_ge, allow_pickle=True)
        ids = t["val_subject_ids"].astype(str)
        y_tab = t["y_tabpfn"].astype(np.float32)
        y_true = t["y_true"].astype(np.float32)
        idx = {s: i for i, s in enumerate(ids)}
        pick = np.array([idx[s] for s in our_ids])
        tab_ge_m = _metrics(y_true[pick], y_tab[pick])
    if tab_en.exists():
        t = np.load(tab_en, allow_pickle=True)
        ids = t["val_subject_ids"].astype(str)
        y_tab = t["y_tabpfn"].astype(np.float32)
        y_true = t["y_true"].astype(np.float32)
        idx = {s: i for i, s in enumerate(ids)}
        pick = np.array([idx[s] for s in our_ids])
        tab_en_m = _metrics(y_true[pick], y_tab[pick])

    rows.append({
        "fold": f, "n": len(our_ids),
        "ours": ours_m, "tab_ge": tab_ge_m, "tab_en": tab_en_m,
    })
    parts = [f"fold {f}", f"n={len(our_ids)}"]
    parts.append(f"ours R²={ours_m['r2']:+.4f} MAE={ours_m['mae']:.4f} "
                 f"RMSE={ours_m['rmse']:.4f} r={ours_m['pearson_r']:+.4f} "
                 f"ρ={ours_m['spearman_rho']:+.4f}")
    if tab_ge_m is not None:
        parts.append(f"tab_ge R²={tab_ge_m['r2']:+.4f}")
    if tab_en_m is not None:
        parts.append(f"tab_en R²={tab_en_m['r2']:+.4f}")
    print("  " + " | ".join(parts))

if len(rows) >= 2:
    for group in ("ours", "tab_ge", "tab_en"):
        have = [r[group] for r in rows if r[group] is not None]
        if not have:
            continue
        agg = {k: np.array([m[k] for m in have]) for k in have[0]}
        print(f"\n[{group}] across {len(have)} folds:")
        for k, v in agg.items():
            print(f"    {k:12s}: {v.mean():+.4f} ± {v.std():.4f}")

summary = {
    "per_fold": rows,
    "threshold_note": "Phase 3 gate: mean R² > TabPFN-enriched (0.4145) + full metric comparison",
}
out = Path("outputs/redesign/p5_phase2_residual/best_vs_tabpfn_summary.json")
out.write_text(json.dumps(summary, indent=2, default=float))
print(f"\nWrote {out}")
PY
