"""Classical baselines on a variant's residualized cognition target.

Trains 6 baselines per fold using the per-fold residualized target:
  - Ridge       (sklearn, alpha=1.0; canonical match)
  - ElasticNet  (sklearn, alpha=1.0, l1_ratio=0.5; canonical match)
  - RandomForest (sklearn, n_estimators=100, max_depth=16, seed=42; canonical match)
  - SVR         (sklearn, C=1.0, kernel='rbf'; canonical match)
  - XGBoost     (CPU hist tree, max_depth=6, lr=0.1, num_boost=100; canonical match)
  - Clinical-only (sklearn LinReg on APOE-ε4 count + age_death + msex + educ + braaksc)

Writes per-fold + cross-fold-mean R² per model to <out-dir>/variant_baselines.json.
CPU-only so it can run in parallel with GPU-bound jobs. Uses canonical top-k
features per fold (same as variant TabPFN cache, isolates target effect).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.baselines.rf_defaults import RF_KWARGS  # noqa: E402

from src.data.feature_loaders import load_flat_features  # noqa: E402
from src.data.splits import load_splits  # noqa: E402


def _load_residual_target_for_fold(
    cache_dir: Path, fold: int,
) -> dict[str, float]:
    """Return {sid: residual_target_float} for a fold (NaN-skipped)."""
    npz = np.load(cache_dir / f"residual_target_fold{fold}.npz", allow_pickle=True)
    sids = npz["subject_ids"].tolist()
    vals = npz["target"].astype(float)
    return {s: float(v) for s, v in zip(sids, vals) if np.isfinite(v)}


def _apoe_e4_count(genotype: pd.Series) -> pd.Series:
    """ε4 allele count from APOE genotype encoding (22/23/24/33/34/44)."""
    mapping = {22: 0, 23: 0, 24: 1, 33: 0, 34: 1, 44: 2}
    return genotype.map(mapping)


def _fit_predict(model, X_tr, y_tr, X_va) -> np.ndarray:
    model.fit(X_tr, y_tr)
    return model.predict(X_va)


def _fit_xgb(X_tr, y_tr, X_va) -> np.ndarray:
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va)
    params = {
        "max_depth": 6, "learning_rate": 0.1,
        "tree_method": "hist", "device": "cpu",
        "objective": "reg:squarederror", "verbosity": 0,
    }
    bst = xgb.train(params, dtr, num_boost_round=100)
    return np.asarray(bst.predict(dva))


def _clinical_baseline_one_fold(
    md_clin: pd.DataFrame, train_ids, val_ids, target_map,
) -> dict:
    train_rows = md_clin.loc[
        md_clin.index.intersection(train_ids)
    ].dropna()
    val_rows = md_clin.loc[
        md_clin.index.intersection(val_ids)
    ].dropna()
    train_kept = train_rows.index.tolist()
    val_kept = val_rows.index.tolist()
    if len(train_kept) < 30 or len(val_kept) < 5:
        return {"r2": float("nan"),
                "n_train": len(train_kept), "n_val": len(val_kept)}
    y_tr = np.array([target_map[s] for s in train_kept])
    y_va = np.array([target_map[s] for s in val_kept])
    reg = LinearRegression().fit(train_rows.values, y_tr)
    pred = reg.predict(val_rows.values)
    return {"r2": float(r2_score(y_va, pred)),
            "n_train": len(train_kept), "n_val": len(val_kept)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant-name", required=True)
    p.add_argument("--residual-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits-path", type=Path,
                   default=_ROOT / "outputs/splits.json")
    p.add_argument("--precomputed-dir", type=Path,
                   default=_ROOT / "data/precomputed")
    p.add_argument("--metadata-csv", type=Path,
                   default=_ROOT / "data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", type=Path,
                   default=_ROOT / "data/canonical")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--skip-svr", action="store_true",
                   help="Skip SVR (slowest; ~30s per fold).")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(str(args.splits_path))
    metadata = pd.read_csv(args.metadata_csv)
    md_indexed = metadata.set_index("ROSMAP_IndividualID").copy()
    md_indexed["apoe_e4_count"] = _apoe_e4_count(md_indexed["apoe_genotype"])
    md_clin = md_indexed[["age_death", "msex", "educ", "braaksc", "apoe_e4_count"]]

    all_ids = sorted(
        {sid for f in splits["folds"] for sid in f["train"] + f["val"]}
    )
    print(f"loading flat features for {len(all_ids)} subjects...", flush=True)
    feat_map = load_flat_features(args.precomputed_dir, all_ids)
    sid_order = sorted(feat_map.keys())
    sid_to_row = {s: i for i, s in enumerate(sid_order)}
    feat_mat = np.stack([feat_map[s] for s in sid_order]).astype(np.float32)
    print(f"  feature matrix: {feat_mat.shape}", flush=True)

    model_factories = {
        "ridge": lambda: Ridge(alpha=1.0, random_state=42),
        "elastic_net": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5,
                                          random_state=42, max_iter=10000),
        "random_forest": lambda: RandomForestRegressor(**RF_KWARGS),
    }
    if not args.skip_svr:
        model_factories["svr"] = lambda: SVR(C=1.0, kernel="rbf")

    results: dict = {"variant_name": args.variant_name,
                     "per_fold": [], "models": {}}

    for fold in range(len(splits["folds"])):
        print(f"\n=== fold {fold} ===", flush=True)
        target_map = _load_residual_target_for_fold(args.residual_cache_dir, fold)
        train_ids_raw = splits["folds"][fold]["train"]
        val_ids_raw = splits["folds"][fold]["val"]
        train_ids = [s for s in train_ids_raw if s in sid_to_row and s in target_map]
        val_ids = [s for s in val_ids_raw if s in sid_to_row and s in target_map]

        X_train_full = feat_mat[[sid_to_row[s] for s in train_ids]]
        X_val_full = feat_mat[[sid_to_row[s] for s in val_ids]]
        y_train = np.array([target_map[s] for s in train_ids], dtype=np.float64)
        y_val = np.array([target_map[s] for s in val_ids], dtype=np.float64)

        top_k_path = args.top_k_dir / f"top_{args.top_k}_features_fold{fold}.json"
        top_k_idx = json.loads(top_k_path.read_text())["indices"]
        X_train = X_train_full[:, top_k_idx]
        X_val = X_val_full[:, top_k_idx]

        # Standardize for linear/SVR models (XGBoost / RF use raw).
        sc = StandardScaler().fit(X_train)
        X_train_s = sc.transform(X_train)
        X_val_s = sc.transform(X_val)

        fold_row: dict = {
            "fold": fold,
            "n_train": int(len(train_ids)),
            "n_val": int(len(val_ids)),
        }

        for mname, factory in model_factories.items():
            t0 = time.time()
            X_tr = X_train_s if mname in ("ridge", "elastic_net", "svr") else X_train
            X_va = X_val_s if mname in ("ridge", "elastic_net", "svr") else X_val
            pred = _fit_predict(factory(), X_tr, y_train, X_va)
            r2 = float(r2_score(y_val, pred))
            fold_row[f"{mname}_r2"] = r2
            print(f"  {mname:13s} R²={r2:+.4f}  ({time.time() - t0:.1f}s)", flush=True)

        t0 = time.time()
        pred = _fit_xgb(X_train, y_train, X_val)
        r2 = float(r2_score(y_val, pred))
        fold_row["xgboost_r2"] = r2
        print(f"  xgboost       R²={r2:+.4f}  ({time.time() - t0:.1f}s)", flush=True)

        t0 = time.time()
        clin_res = _clinical_baseline_one_fold(md_clin, train_ids, val_ids, target_map)
        fold_row["clinical_r2"] = clin_res["r2"]
        print(f"  clinical      R²={clin_res['r2']:+.4f} (n_train={clin_res['n_train']}, n_val={clin_res['n_val']})  ({time.time() - t0:.1f}s)", flush=True)

        results["per_fold"].append(fold_row)

    model_keys = list(model_factories.keys()) + ["xgboost", "clinical"]
    for model in model_keys:
        vals = [r[f"{model}_r2"] for r in results["per_fold"]
                if not np.isnan(r[f"{model}_r2"])]
        if len(vals) >= 2:
            results["models"][model] = {
                "mean_r2": statistics.fmean(vals),
                "std_r2": statistics.stdev(vals),
                "n_folds": len(vals),
            }
        else:
            results["models"][model] = {
                "mean_r2": float("nan"), "std_r2": float("nan"),
                "n_folds": len(vals),
            }

    out = args.out_dir / "variant_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    for model, agg in results["models"].items():
        print(
            f"  {model:14s}  {agg['mean_r2']:+.4f} ± {agg['std_r2']:.4f}  ({agg['n_folds']} folds)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
