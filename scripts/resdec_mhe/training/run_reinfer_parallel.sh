#!/usr/bin/env bash
# Parallel 5-fold re-inference of best-by-val/r2 checkpoints across both GPUs.
# Per fold: loads max-R² best-*.ckpt, runs validate(), dumps val_predictions_best.npz
# + best_summary.json into $OUTROOT/fold{N}/.
#
# Env overrides (all optional):
#   CONFIG          phase YAML (default: configs/resdec_mhe/canonical.yaml)
#   OUTROOT         output dir to read ckpts from + write best.npz to
#                   (default: outputs/canonical/p5_phase2_residual)
#   TABPFN_DIR      directory holding tabpfn_outer_fold{N}*.npz files for
#                   the comparison summary (default: data/canonical)
#   METADATA_PATH   override cfg.data.metadata_path (e.g. data/metadata_ROSMAP)
#   PRECOMPUTED_DIR override cfg.data.precomputed_dir (e.g. data/precomputed)
#   TABPFN_OOF_DIR  override cfg.data.tabpfn_oof_dir (variant TabPFN cache)
#   TABPFN_OUTER_DIR override cfg.data.tabpfn_outer_dir
#   N_GPUS          number of GPUs (default: all visible)
#   GPU_LIST        comma-separated GPU list, e.g. "0,1"
#
# Usage:
#   bash scripts/resdec_mhe/training/run_reinfer_parallel.sh
#   OUTROOT=outputs/canonical/<your_run> bash scripts/resdec_mhe/training/run_reinfer_parallel.sh
set -euo pipefail

# Resolve repo root from this script's location (scripts/resdec_mhe/training/<script>.sh
# -> repo root is three levels up). Allows env override for unusual layouts.
# Fixes B-RP1: prior literal /host/.../redesign-resdec-h3 referenced a
# DELETED worktree (verified 2026-04-23 per memory file).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIG="${CONFIG:-configs/resdec_mhe/canonical.yaml}"
OUTROOT="${OUTROOT:-outputs/canonical/p5_phase2_residual}"
TABPFN_DIR="${TABPFN_DIR:-data/canonical}"
METADATA_PATH="${METADATA_PATH:-}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-}"
TABPFN_OOF_DIR="${TABPFN_OOF_DIR:-}"
TABPFN_OUTER_DIR="${TABPFN_OUTER_DIR:-}"
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
        EXTRA_ARGS=()
        if [[ -n "$METADATA_PATH" ]]; then
            EXTRA_ARGS+=(--metadata-path "$METADATA_PATH")
        fi
        if [[ -n "$PRECOMPUTED_DIR" ]]; then
            EXTRA_ARGS+=(--precomputed-dir "$PRECOMPUTED_DIR")
        fi
        if [[ -n "$TABPFN_OOF_DIR" ]]; then
            EXTRA_ARGS+=(--tabpfn-oof-dir "$TABPFN_OOF_DIR")
        fi
        if [[ -n "$TABPFN_OUTER_DIR" ]]; then
            EXTRA_ARGS+=(--tabpfn-outer-dir "$TABPFN_OUTER_DIR")
        fi
        CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/resdec_mhe/training/reinfer_best_ckpt.py \
            --config "$CONFIG" \
            --fold "$fold" \
            --output-dir "$OUTROOT" \
            "${EXTRA_ARGS[@]}" \
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
# Pass paths via env so the heredoc stays quoted (no $-substitution surprises).
OUTROOT="$OUTROOT" TABPFN_DIR="$TABPFN_DIR" uv run python - <<'PY'
import json
import os  # only for os.environ — path ops use pathlib below
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

root = Path(os.environ["OUTROOT"])
tabpfn_dir = Path(os.environ["TABPFN_DIR"])

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
    "threshold_note": "Paper threshold: mean R² > TabPFN-enriched (~0.4145) + full metric comparison",
    "outroot": str(root),
    "tabpfn_dir": str(tabpfn_dir),
}
out = root / "best_vs_tabpfn_summary.json"
out.write_text(json.dumps(summary, indent=2, default=float))
print(f"\nWrote {out}")
PY
