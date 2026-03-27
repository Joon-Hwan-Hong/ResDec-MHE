"""
MixMIL baseline for ROSMAP cognitive resilience regression.

Runs MixMIL (GLMM + attention MIL) with Gaussian likelihood on 30-dim
scVI embeddings, using our exact 5-fold cross-validation splits.
Warm-start from OLS on mean embeddings (author's simulation_normal.ipynb pattern).

Usage:
    baselines/mixmil/.venv/bin/python baselines/mixmil/run_rosmap.py \
        --data-h5ad baselines/shared/mixmil_input.h5ad \
        --splits outputs/splits.json \
        --results-dir outputs/baselines/mixmil \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# MixMIL imports — add the vendored repo to sys.path
# ---------------------------------------------------------------------------
_REPO_DIR = str(Path(__file__).resolve().parent / "repo")
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from mixmil import MixMIL  # noqa: E402
from mixmil.data import normalize_feats  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_device(obj, device):
    """Recursively move tensors / modules / lists / dicts to *device*."""
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_device(x, device) for x in obj]
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device)
    return obj


def load_data(h5ad_path: str):
    """Load the shared h5ad and group cells by subject.

    Returns
    -------
    sample_ids : list[str]
        Ordered list of unique sample IDs.
    Xs : list[torch.Tensor]
        Per-subject cell embedding tensors, each ``[n_cells_i, 30]``.
    Y : torch.Tensor
        Phenotype per subject, shape ``[N, 1]``.
    F : torch.Tensor
        Intercept-only fixed effects, shape ``[N, 1]``.
    """
    print(f"Loading data from {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} features")

    # Ensure dense matrix
    X_np = np.asarray(adata.X, dtype=np.float32)

    # Group by sample_id, preserving order of first appearance
    obs = adata.obs
    sample_ids_all = obs["sample_id"].values
    _, first_idx = np.unique(sample_ids_all, return_index=True)
    sample_ids = sample_ids_all[np.sort(first_idx)].tolist()

    # Build per-subject tensors and phenotype vector
    Xs = []
    phenotypes = []
    sid_to_rows = obs.groupby("sample_id", sort=False).indices
    for sid in sample_ids:
        rows = sid_to_rows[sid]
        Xs.append(torch.tensor(X_np[rows], dtype=torch.float32))
        phenotypes.append(obs["phenotype"].iloc[rows[0]])

    Y = torch.tensor(phenotypes, dtype=torch.float32).reshape(-1, 1)
    F = torch.ones((len(sample_ids), 1), dtype=torch.float32)

    print(f"  {len(sample_ids)} subjects, Q={Xs[0].shape[1]}")
    return sample_ids, Xs, Y, F


def ols_warmstart(model: MixMIL, Xs_train, F_train, Y_train, Q: int):
    """Warm-start model parameters from OLS on mean embeddings.

    Follows the exact pattern from the author's simulation_normal.ipynb.
    """
    X_m = np.stack([x.cpu().numpy().mean(0) for x in Xs_train])   # [N_train, Q]
    F_X_m = np.concatenate([F_train.cpu().numpy(), X_m], axis=1)  # [N_train, Q+1]
    ols = LinearRegression(fit_intercept=False).fit(F_X_m, Y_train.cpu().numpy())

    sd = model.state_dict()
    sd["alpha"] = torch.tensor(ols.coef_.ravel()[0], dtype=torch.float32).reshape(1, 1)
    beta = ols.coef_.ravel()[1:]
    u = X_m.dot(beta)
    denom = np.sqrt((beta ** 2).mean())
    if denom < 1e-12 or u.std() < 1e-12:
        print("  WARNING: OLS warm-start skipped (degenerate beta), using default init")
        return model
    beta = u.std() * beta / denom
    sd["posterior.mu"] = torch.cat(
        [sd["posterior.mu"][:, :Q], torch.tensor(beta, dtype=torch.float32).reshape(1, -1)],
        dim=1,
    )
    model.load_state_dict(sd)
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R^2, MAE, Pearson r, and Spearman rho."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true.ravel(), y_pred.ravel())[0]),
        "spearman_rho": float(spearmanr(y_true.ravel(), y_pred.ravel())[0]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MixMIL baseline — ROSMAP regression")
    parser.add_argument("--data-h5ad", required=True, help="Path to mixmil_input.h5ad")
    parser.add_argument("--splits", required=True, help="Path to splits.json")
    parser.add_argument("--results-dir", default="outputs/baselines/mixmil", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--n-epochs", type=int, default=2000, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # ---- Load data --------------------------------------------------------
    sample_ids, Xs, Y, F = load_data(args.data_h5ad)
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    Q = Xs[0].shape[1]  # 30

    # ---- Load splits ------------------------------------------------------
    with open(args.splits) as f:
        splits = json.load(f)

    fold_results = []
    variance_rows = []

    for fold_idx, fold in enumerate(splits["folds"]):
        fold_num = fold_idx + 1
        fold_dir = results_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Fold {fold_num}")
        print(f"{'='*60}")

        train_ids = fold["train"]
        test_ids = fold["val"]  # our fold "val" = test set for MixMIL
        train_idxs = [sid_to_idx[s] for s in train_ids]
        test_idxs = [sid_to_idx[s] for s in test_ids]

        # Subset data (CPU tensors for normalize_feats)
        Xs_train_raw = [Xs[i] for i in train_idxs]
        Xs_test_raw = [Xs[i] for i in test_idxs]
        F_train = F[train_idxs]
        F_test = F[test_idxs]
        Y_train = Y[train_idxs]
        Y_test = Y[test_idxs]

        print(f"  Train: {len(train_idxs)} subjects, Test: {len(test_idxs)} subjects")

        # ---- Normalize features (MixMIL's normalize_feats) ---------------
        X_dict = {"train": Xs_train_raw, "test": Xs_test_raw}
        X_dict = normalize_feats(X_dict, norm_factor="std_sqrt")
        Xs_train = X_dict["train"]
        Xs_test = X_dict["test"]

        # ---- Build and warm-start model ----------------------------------
        model = MixMIL(Q=Q, K=1, P=1, likelihood="normal")
        model = ols_warmstart(model, Xs_train, F_train, Y_train, Q)
        print(f"  OLS warm-start applied")

        # ---- Move to device and train ------------------------------------
        model = model.to(device)
        Xs_train = to_device(Xs_train, device)
        F_train = F_train.to(device)
        Y_train = Y_train.to(device)

        print(f"  Training for {args.n_epochs} epochs on {device} ...")
        t0 = time.time()
        history = model.train(
            Xs_train, F_train, Y_train,
            n_epochs=args.n_epochs, batch_size=64, lr=args.lr,
        )
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s")

        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(fold_dir / "history.csv", index=False)

        # ---- Predict on test set (CPU) --------------------------------------
        model.cpu()
        Y_test_np = Y_test.cpu().numpy().ravel()

        # Xs_test and F_test are already on CPU (never moved to GPU)
        u_pred = model.predict(Xs_test)  # [N_test, P]
        y_pred = (F_test.mm(model.alpha) + u_pred).detach().numpy().ravel()

        metrics = compute_metrics(Y_test_np, y_pred)
        metrics["fold"] = fold_num
        metrics["train_time_s"] = round(elapsed, 1)
        fold_results.append(metrics)
        print(f"  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
              f"r={metrics['pearson_r']:.4f}  rho={metrics['spearman_rho']:.4f}")

        # ---- Save predictions --------------------------------------------
        pred_df = pd.DataFrame({
            "sample_id": test_ids,
            "y_true": Y_test_np,
            "y_pred": y_pred,
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        # ---- Save model state dict ---------------------------------------
        torch.save(model.state_dict(), fold_dir / "model.pt")

        # ---- Extract and save instance weights ---------------------------
        w, _w = model.get_weights(Xs_test)
        # w and _w are lists of tensors (one per test subject)
        with open(fold_dir / "instance_weights.pkl", "wb") as f:
            pickle.dump({"post_softmax": w, "pre_softmax": _w}, f)

        # ---- Variance components -----------------------------------------
        sigma_u = torch.exp(model.log_sigma_u).item()
        sigma_z = torch.exp(model.log_sigma_z).item()
        sigma_obs = torch.exp(model.log_scale).item()
        variance_rows.append({
            "fold": fold_num,
            "sigma_u": sigma_u,
            "sigma_z": sigma_z,
            "sigma_obs": sigma_obs,
        })
        print(f"  Variance: sigma_u={sigma_u:.4f}, sigma_z={sigma_z:.4f}, sigma_obs={sigma_obs:.4f}")

        # Free GPU memory
        del model, Xs_train, F_train, Y_train
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Aggregate results -----------------------------------------------
    all_folds_df = pd.DataFrame(fold_results)
    all_folds_df.to_csv(results_dir / "AllFolds_MixMIL_ROSMAP.csv", index=False)

    metric_cols = ["r2", "mae", "pearson_r", "spearman_rho"]
    summary_rows = []
    for col in metric_cols:
        vals = all_folds_df[col]
        summary_rows.append({
            "metric": col,
            "mean": vals.mean(),
            "std": vals.std(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "Summary_MixMIL_ROSMAP.csv", index=False)

    # Variance components table
    var_df = pd.DataFrame(variance_rows)
    var_df.to_csv(results_dir / "variance_components.csv", index=False)

    print(f"\n{'='*60}")
    print("  Summary across folds")
    print(f"{'='*60}")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:15s}  {row['mean']:.4f} +/- {row['std']:.4f}")
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
