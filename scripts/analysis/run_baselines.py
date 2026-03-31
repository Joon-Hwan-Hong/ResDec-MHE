"""
GPU-accelerated classical ML baselines for cognitive resilience prediction.

Runs 5 models (Ridge, ElasticNet, SVR, RandomForest, XGBoost) on 3 feature
sets (cell-type proportions, pseudobulk, all-combined) using the same 5-fold
CV splits as the deep model.  All computation on GPU via cuML / XGBoost CUDA.

Usage:
    uv run python scripts/analysis/run_baselines.py \
        --precomputed-dir data/precomputed/rosmap/ \
        --splits-path outputs/splits.json \
        --metadata-path data/metadata_ROSMAP/ \
        --output outputs/baseline_results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Ensure CUDA NVRTC library is findable by CuPy (needed for SVD solver, etc.)
_nvrtc_dir = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
if _nvrtc_dir.exists():
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        str(_nvrtc_dir) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )

import cupy as cp
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from cuml import ElasticNet, Ridge
from cuml.ensemble import RandomForestRegressor
from cuml.metrics import r2_score as cuml_r2_score
from cuml.preprocessing import StandardScaler
from cuml.svm import SVR
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[_FlushHandler()],
)
logger = logging.getLogger(__name__)

# ── Feature set names ────────────────────────────────────────────────────────
FEATURE_SETS = ["C", "A", "A+C+E"]

# ── Number of CCC edge types ────────────────────────────────────────────────
N_CCC_TYPES = 5

# ── Number of cell types ────────────────────────────────────────────────────
N_CELL_TYPES = 31


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features_c(pt_data: dict) -> np.ndarray:
    """Cell-type proportions [31].  cell_counts normalised to sum to 1."""
    cell_counts = pt_data["cell_counts"].float()
    total = cell_counts.sum()
    if total > 0:
        proportions = cell_counts / total
    else:
        proportions = torch.zeros_like(cell_counts)
    return proportions.numpy()


def extract_features_a(pt_data: dict) -> np.ndarray:
    """Flattened pseudobulk [31 * 4797 = 148_607]."""
    return pt_data["pseudobulk"].numpy().flatten()


def extract_features_e(pt_data: dict) -> np.ndarray:
    """CCC graph summary features [~18].

    Per-type (5 types):
        - edge count
        - mean edge attribute
        - std edge attribute
    Global node-degree:
        - mean, std, max
    """
    edge_index = pt_data["ccc_edge_index"]   # [2, n_edges]
    edge_type = pt_data["ccc_edge_type"]     # [n_edges]
    edge_attr = pt_data["ccc_edge_attr"]     # [n_edges, edge_dim]

    n_edges = edge_index.shape[1]

    # Per-type statistics
    counts = np.zeros(N_CCC_TYPES, dtype=np.float32)
    mean_attrs = np.zeros(N_CCC_TYPES, dtype=np.float32)
    std_attrs = np.zeros(N_CCC_TYPES, dtype=np.float32)

    for t in range(N_CCC_TYPES):
        mask = edge_type == t
        c = mask.sum().item()
        counts[t] = c
        if c > 0:
            attrs_t = edge_attr[mask]            # [c, edge_dim]
            mean_attrs[t] = attrs_t.mean().item()
            std_attrs[t] = attrs_t.std().item() if c > 1 else 0.0

    # Node degree statistics (based on source node = row 0 of edge_index)
    degrees = torch.zeros(N_CELL_TYPES, dtype=torch.float32)
    if n_edges > 0:
        src_nodes = edge_index[0]
        for i in range(n_edges):
            degrees[src_nodes[i]] += 1

    degree_mean = degrees.mean().item()
    degree_std = degrees.std().item()
    degree_max = degrees.max().item()

    return np.concatenate([
        counts,       # 5
        mean_attrs,   # 5
        std_attrs,    # 5
        np.array([degree_mean, degree_std, degree_max], dtype=np.float32),  # 3
    ])  # total: 18


def load_all_features(
    precomputed_dir: Path,
    subject_ids: list[str],
    metadata_df: pd.DataFrame,
    subject_column: str,
) -> dict[str, np.ndarray]:
    """Load features and target for all subjects.

    Returns dict with keys: 'C', 'A', 'E', 'target', 'subject_ids' (filtered).
    Subjects missing a .pt file or cogn_global are dropped with a warning.
    """
    meta_lookup = metadata_df.set_index(subject_column)["cogn_global"].to_dict()

    features_c = []
    features_a = []
    features_e = []
    targets = []
    valid_ids = []

    for sid in subject_ids:
        pt_path = precomputed_dir / f"{sid}.pt"
        if not pt_path.exists():
            logger.warning("Missing .pt file for %s — skipping", sid)
            continue
        if sid not in meta_lookup or pd.isna(meta_lookup[sid]):
            logger.warning("Missing cogn_global for %s — skipping", sid)
            continue

        pt_data = torch.load(pt_path, weights_only=False)
        features_c.append(extract_features_c(pt_data))
        features_a.append(extract_features_a(pt_data))
        features_e.append(extract_features_e(pt_data))
        targets.append(meta_lookup[sid])
        valid_ids.append(sid)

    n = len(valid_ids)
    logger.info("Loaded features for %d / %d subjects", n, len(subject_ids))

    return {
        "C": np.stack(features_c),   # [n, 31]
        "A": np.stack(features_a),   # [n, 148607]
        "E": np.stack(features_e),   # [n, 18]
        "target": np.array(targets, dtype=np.float32),
        "subject_ids": valid_ids,
    }


def build_feature_matrix(
    data: dict[str, np.ndarray],
    feature_set: str,
) -> np.ndarray:
    """Concatenate feature arrays according to the feature set name."""
    if feature_set == "C":
        return data["C"]
    elif feature_set == "A":
        return data["A"]
    elif feature_set == "A+C+E":
        return np.concatenate([data["A"], data["C"], data["E"]], axis=1)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


# ── Model factory ────────────────────────────────────────────────────────────

def make_model(name: str, n_features: int = 0):
    """Return an unfitted model instance with fixed defaults."""
    if name == "Ridge":
        solver = "svd" if n_features > 500 else "eig"
        return Ridge(alpha=1.0, solver=solver)
    elif name == "ElasticNet":
        return ElasticNet(alpha=1.0, l1_ratio=0.5)
    elif name == "SVR":
        return SVR(C=1.0, kernel="rbf")
    elif name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            random_state=42,
        )
    elif name == "XGBoost":
        return None
    else:
        raise ValueError(f"Unknown model: {name}")


# ── HP grids for CV tuning ────────────────────────────────────────────────

def _get_param_grid(name: str, n_features: int) -> tuple:
    """Return (estimator, param_grid) for GridSearchCV.

    Uses cuML estimators on GPU. XGBoost and PLS handled separately.
    """
    from cuml.model_selection import GridSearchCV as cuGridSearchCV

    if name == "Ridge":
        solver = "svd" if n_features > 500 else "eig"
        estimator = Ridge(solver=solver)
        param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    elif name == "ElasticNet":
        estimator = ElasticNet()
        param_grid = {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    elif name == "SVR":
        estimator = SVR(kernel="rbf")
        param_grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", 0.01, 0.1],
        }
    elif name == "RandomForest":
        estimator = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [8, 16],
        }
    else:
        raise ValueError(f"No CV grid for model: {name}")

    return estimator, param_grid


def fit_predict_cv(
    model_name: str,
    X_train: cp.ndarray,
    y_train: cp.ndarray,
    X_val: cp.ndarray,
    n_features: int,
    cv_folds: int = 5,
) -> tuple[np.ndarray, dict]:
    """Fit model with inner GridSearchCV on training data, predict on val.

    Returns (predictions, best_params).
    """
    from cuml.model_selection import GridSearchCV as cuGridSearchCV
    from sklearn.metrics import make_scorer

    estimator, param_grid = _get_param_grid(model_name, n_features)
    gs = cuGridSearchCV(
        estimator, param_grid,
        cv=cv_folds, scoring=make_scorer(cuml_r2_score), refit=True,
    )
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    preds = gs.predict(X_val)
    if hasattr(preds, "get"):
        preds = preds.get()

    return np.asarray(preds), best_params


MODEL_NAMES = ["Ridge", "ElasticNet", "RandomForest", "XGBoost", "PLS"]


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics (all on CPU numpy)."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_rho": float(spearmanr(y_true, y_pred)[0]),
    }


# ── Training / evaluation ────────────────────────────────────────────────────

def fit_predict_cuml(
    model,
    X_train: cp.ndarray,
    y_train: cp.ndarray,
    X_val: cp.ndarray,
) -> np.ndarray:
    """Fit a cuML model and return val predictions as numpy."""
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    # cuML returns cupy array or cudf series; ensure numpy
    if hasattr(preds, "get"):
        return preds.get()
    return np.asarray(preds)


def fit_predict_xgboost(
    X_train: cp.ndarray,
    y_train: cp.ndarray,
    X_val: cp.ndarray,
) -> np.ndarray:
    """Fit XGBoost with GPU and return val predictions as numpy."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val)
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "device": "cuda",
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "verbosity": 0,
    }
    bst = xgb.train(params, dtrain, num_boost_round=100)
    preds = bst.predict(dval)
    return np.asarray(preds)


def run_fold(
    model_name: str,
    feature_set: str,
    fold_idx: int,
    data: dict[str, np.ndarray],
    train_ids: list[str],
    val_ids: list[str],
    cv_tune: bool = False,
) -> dict:
    """Run a single (model, feature_set, fold) experiment.

    If cv_tune=True, uses inner GridSearchCV to select HPs on training data.
    Returns a dict with model, feature_set, fold, and all metric values.
    """
    # Map subject IDs to row indices
    sid_to_idx = {sid: i for i, sid in enumerate(data["subject_ids"])}
    train_idx = [sid_to_idx[s] for s in train_ids if s in sid_to_idx]
    val_idx = [sid_to_idx[s] for s in val_ids if s in sid_to_idx]

    if len(train_idx) == 0 or len(val_idx) == 0:
        logger.warning(
            "Empty train/val for %s/%s/fold%d — skipping", model_name, feature_set, fold_idx
        )
        return None

    # Build feature matrix for this feature set
    X_all = build_feature_matrix(data, feature_set)
    y_all = data["target"]

    X_train_np = X_all[train_idx].astype(np.float32)
    X_val_np = X_all[val_idx].astype(np.float32)
    y_train_np = y_all[train_idx].astype(np.float32)
    y_val_np = y_all[val_idx].astype(np.float32)

    # Transfer to GPU (cupy)
    X_train_gpu = cp.asarray(X_train_np)
    X_val_gpu = cp.asarray(X_val_np)
    y_train_gpu = cp.asarray(y_train_np)

    # StandardScaler on GPU (fit on train only)
    scaler = StandardScaler()
    X_train_gpu = scaler.fit_transform(X_train_gpu)
    X_val_gpu = scaler.transform(X_val_gpu)

    # PCA reduction for ElasticNet on high-dim features (148K coordinate descent is infeasible)
    PCA_DIM_THRESHOLD = 1000
    PCA_N_COMPONENTS = 100
    pca_applied = False
    if model_name == "ElasticNet" and X_train_gpu.shape[1] > PCA_DIM_THRESHOLD:
        from sklearn.decomposition import PCA as skPCA
        # Use sklearn PCA on CPU — cuML PCA OOMs on 148K features (needs 88GB GPU)
        logger.info("  PCA %d -> %d for ElasticNet (CPU sklearn)",
                    X_all.shape[1], PCA_N_COMPONENTS)
        pca_reducer = skPCA(n_components=PCA_N_COMPONENTS, random_state=42)
        X_train_reduced = pca_reducer.fit_transform(X_train_np)
        X_val_reduced = pca_reducer.transform(X_val_np)
        X_train_gpu = cp.asarray(X_train_reduced.astype(np.float32))
        X_val_gpu = cp.asarray(X_val_reduced.astype(np.float32))
        # Re-scale after PCA
        scaler2 = StandardScaler()
        X_train_gpu = scaler2.fit_transform(X_train_gpu)
        X_val_gpu = scaler2.transform(X_val_gpu)
        pca_applied = True

    # Fit and predict
    best_params = None
    n_feat = X_train_gpu.shape[1]

    if model_name == "PLS":
        # PLS runs on CPU (sklearn) — designed for p >> n, perfect for 148k features
        scaler_cpu = SklearnStandardScaler()
        X_tr_cpu = scaler_cpu.fit_transform(X_train_np)
        X_va_cpu = scaler_cpu.transform(X_val_np)
        if cv_tune:
            from sklearn.model_selection import GridSearchCV as skGridSearchCV
            n_max = min(20, X_tr_cpu.shape[0], X_tr_cpu.shape[1])
            pls_gs = skGridSearchCV(
                PLSRegression(),
                {"n_components": list(range(2, n_max + 1, 2))},
                cv=5, scoring="r2", refit=True,
            )
            pls_gs.fit(X_tr_cpu, y_train_np)
            y_pred = pls_gs.predict(X_va_cpu).ravel()
            best_params = pls_gs.best_params_
        else:
            n_comp = min(10, X_tr_cpu.shape[0], X_tr_cpu.shape[1])
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr_cpu, y_train_np)
            y_pred = pls.predict(X_va_cpu).ravel()
    elif model_name == "XGBoost":
        if cv_tune:
            # XGBoost CV via sklearn-style grid search on CPU (safe for all feature dims)
            from sklearn.model_selection import GridSearchCV as skGridSearchCV
            xgb_est = xgb.XGBRegressor(
                tree_method="hist",
                device="cuda" if n_feat <= 50000 else "cpu",
                objective="reg:squarederror",
                verbosity=0,
                random_state=42,
            )
            xgb_grid = {
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [100, 200],
            }
            gs = skGridSearchCV(xgb_est, xgb_grid, cv=5, scoring="r2", refit=True)
            gs.fit(X_train_np, y_train_np)
            y_pred = gs.predict(X_val_np)
            best_params = gs.best_params_
        else:
            if n_feat > 50000:
                dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
                dval = xgb.DMatrix(X_val_np)
                params = {
                    "max_depth": 6, "learning_rate": 0.1,
                    "tree_method": "hist", "objective": "reg:squarederror", "verbosity": 0,
                }
                bst = xgb.train(params, dtrain, num_boost_round=100)
                y_pred = bst.predict(dval)
            else:
                y_pred = fit_predict_xgboost(X_train_gpu, y_train_gpu, X_val_gpu)
    elif cv_tune and model_name in ("Ridge", "ElasticNet", "SVR", "RandomForest"):
        y_pred, best_params = fit_predict_cv(
            model_name, X_train_gpu, y_train_gpu, X_val_gpu, n_feat,
        )
    else:
        model = make_model(model_name, n_features=n_feat)
        y_pred = fit_predict_cuml(model, X_train_gpu, y_train_gpu, X_val_gpu)

    # Free GPU memory for next model
    del X_train_gpu, X_val_gpu, y_train_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Metrics (CPU numpy)
    metrics = compute_metrics(y_val_np, y_pred)

    result = {
        "model": model_name,
        "feature_set": feature_set,
        "fold": fold_idx,
        **metrics,
    }
    if best_params is not None:
        result["best_params"] = str(best_params)
        logger.info("  Best params: %s", best_params)
    return result


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(results_df: pd.DataFrame) -> None:
    """Print mean +/- std across folds for each (model, feature_set)."""
    metric_cols = ["r2", "mae", "rmse", "pearson_r", "spearman_rho"]
    grouped = results_df.groupby(["model", "feature_set"])[metric_cols]

    summary_parts = []
    for (model, fset), grp in grouped:
        row = {"model": model, "feature_set": fset}
        for col in metric_cols:
            mean = grp[col].mean()
            std = grp[col].std()
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[col] = f"{mean:.4f} +/- {std:.4f}"
        summary_parts.append(row)

    summary_df = pd.DataFrame(summary_parts)

    print("\n" + "=" * 100)
    print("BASELINE RESULTS — mean +/- std across 5 folds")
    print("=" * 100)

    for fset in FEATURE_SETS:
        subset = summary_df[summary_df["feature_set"] == fset]
        if subset.empty:
            continue
        print(f"\n--- Feature set: {fset} ---")
        display_cols = ["model"] + metric_cols
        print(subset[display_cols].to_string(index=False))

    print("\n" + "=" * 100 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated classical ML baselines for cognitive resilience prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--precomputed-dir",
        type=Path,
        required=True,
        help="Directory with per-subject .pt files (e.g. data/precomputed/rosmap/)",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        required=True,
        help="Path to splits.json with 5-fold CV definitions",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Directory containing metadata.csv with cogn_global column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baseline_results.csv"),
        help="Output CSV path (default: outputs/baseline_results.csv)",
    )
    parser.add_argument(
        "--subject-column",
        type=str,
        default="ROSMAP_IndividualID",
        help="Column in metadata.csv containing subject IDs (default: ROSMAP_IndividualID)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=FEATURE_SETS,
        choices=FEATURE_SETS,
        help="Feature sets to evaluate (default: all)",
    )
    parser.add_argument(
        "--cv-tune",
        action="store_true",
        help="Enable inner CV hyperparameter tuning for all models (cuML GridSearchCV on GPU)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Load splits ──────────────────────────────────────────────────────
    logger.info("Loading splits from %s", args.splits_path)
    with open(args.splits_path) as f:
        splits = json.load(f)

    n_folds = len(splits["folds"])
    pool_ids = splits["train_val_pool"]
    logger.info("Train/val pool: %d subjects, %d folds", len(pool_ids), n_folds)

    # ── Load metadata ────────────────────────────────────────────────────
    metadata_csv = args.metadata_path / "metadata.csv"
    logger.info("Loading metadata from %s", metadata_csv)
    metadata_df = pd.read_csv(metadata_csv)

    # ── Load all features once (numpy) ───────────────────────────────────
    logger.info("Loading features from %s", args.precomputed_dir)
    data = load_all_features(
        args.precomputed_dir,
        pool_ids,
        metadata_df,
        args.subject_column,
    )

    dim_c = data["C"].shape[1]
    dim_a = data["A"].shape[1]
    dim_e = data["E"].shape[1]
    logger.info(
        "Feature dimensions — C: %d, A: %d, E: %d, A+C+E: %d",
        dim_c, dim_a, dim_e, dim_a + dim_c + dim_e,
    )

    # ── Run experiments ──────────────────────────────────────────────────
    results = []
    total = len(args.models) * len(args.feature_sets) * n_folds
    done = 0

    for model_name in args.models:
        for feature_set in args.feature_sets:
            for fold_idx in range(n_folds):
                done += 1
                train_ids = splits["folds"][fold_idx]["train"]
                val_ids = splits["folds"][fold_idx]["val"]

                logger.info(
                    "[%d/%d] %s / %s / fold %d  (train=%d, val=%d)",
                    done, total, model_name, feature_set, fold_idx,
                    len(train_ids), len(val_ids),
                )

                t0 = time.time()
                result = run_fold(
                    model_name, feature_set, fold_idx,
                    data, train_ids, val_ids,
                    cv_tune=args.cv_tune,
                )
                elapsed = time.time() - t0

                if result is not None:
                    results.append(result)
                    logger.info(
                        "  -> R2=%.4f  MAE=%.4f  (%.1fs)",
                        result["r2"], result["mae"], elapsed,
                    )

    # ── Save results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    logger.info("Saved %d rows to %s", len(results_df), args.output)

    # ── Print summary ────────────────────────────────────────────────────
    print_summary(results_df)


if __name__ == "__main__":
    main()
