"""Shared utilities for MIL regression baselines (ABMIL, Set Transformer, etc.).

Provides data loading, normalization, training loop, and evaluation — the same
infrastructure used by all simple MIL baselines for fair comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def load_data(h5ad_path: str, pickle_cache: str | None = None):
    """Load cell embeddings or expression and group by subject.

    Handles three input modes:
    - pickle_cache (fastest): scPhase's pre-processed cache with per-subject dense arrays
    - dense h5ad: scVI embeddings (small, dense .X)
    - sparse h5ad: raw gene expression (large, sparse .X)

    Returns (sample_ids, Xs, Y) where Xs is a list of [n_cells_i, Q] tensors.
    """
    if pickle_cache and Path(pickle_cache).exists():
        import pickle
        print(f"Loading from pickle cache: {pickle_cache} ...")
        with open(pickle_cache, "rb") as f:
            data = pickle.load(f)
        sample_ids = data["sample_ids"]
        Xs = [torch.tensor(arr, dtype=torch.float32) for arr in data["data_list_dense"]]
        Y = torch.tensor(np.array(data["labels"]), dtype=torch.float32)
        print(f"  {len(sample_ids)} subjects, Q={Xs[0].shape[1]} (from pickle)")
        return sample_ids, Xs, Y

    import scipy.sparse

    print(f"Loading data from {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)

    obs = adata.obs
    sample_ids_all = obs["sample_id"].values
    _, first_idx = np.unique(sample_ids_all, return_index=True)
    sample_ids = sample_ids_all[np.sort(first_idx)].tolist()

    is_sparse = scipy.sparse.issparse(adata.X)
    print(f"  {adata.n_obs} cells, {adata.n_vars} features, sparse={is_sparse}")

    Xs = []
    phenotypes = []
    sid_to_rows = obs.groupby("sample_id", sort=False, observed=True).indices
    for sid in sample_ids:
        rows = sid_to_rows[sid]
        chunk = adata.X[rows]
        if is_sparse:
            chunk = chunk.toarray()
        Xs.append(torch.tensor(np.asarray(chunk, dtype=np.float32)))
        phenotypes.append(obs["phenotype"].iloc[rows[0]])

    Y = torch.tensor(phenotypes, dtype=torch.float32)
    print(f"  {len(sample_ids)} subjects, Q={Xs[0].shape[1]}")
    return sample_ids, Xs, Y


def normalize_bags(Xs_train, Xs_test):
    """Normalize embeddings by train-set cell statistics (mean/std)."""
    all_train = torch.cat(Xs_train, dim=0)
    mean = all_train.mean(0, keepdim=True)
    std = all_train.std(0, keepdim=True).clamp(min=1e-8)
    Xs_train = [(x - mean) / std for x in Xs_train]
    Xs_test = [(x - mean) / std for x in Xs_test]
    return Xs_train, Xs_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true.ravel(), y_pred.ravel())[0]),
        "spearman_rho": float(spearmanr(y_true.ravel(), y_pred.ravel())[0]),
    }


def train_one_fold(
    model: torch.nn.Module,
    Xs_train: list[torch.Tensor],
    Y_train: torch.Tensor,
    Xs_val: list[torch.Tensor],
    Y_val: torch.Tensor,
    device: torch.device,
    n_epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> tuple[torch.nn.Module, list[dict]]:
    """Train model with early stopping on val MSE. Returns (model, history)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mse = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        perm = torch.randperm(len(Xs_train))
        for i in perm:
            x = Xs_train[i].to(device)
            y = Y_train[i].to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(Xs_train)

        model.eval()
        val_preds = []
        with torch.no_grad():
            for x in Xs_val:
                val_preds.append(model(x.to(device)).cpu().item())
        val_preds = np.array(val_preds)
        val_true = Y_val.numpy()
        val_mse = float(np.mean((val_preds - val_true) ** 2))
        val_r2 = float(r2_score(val_true, val_preds))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse, "val_r2": val_r2})

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def run_5fold(
    model_name: str,
    model_cls,
    model_kwargs: dict,
    sample_ids: list[str],
    Xs: list[torch.Tensor],
    Y: torch.Tensor,
    splits: dict,
    results_dir: Path,
    device: torch.device,
    n_epochs: int,
    lr: float,
    patience: int,
    seed: int,
) -> pd.DataFrame:
    """Run 5-fold CV for one model. Returns DataFrame of per-fold metrics."""
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fold_idx, fold in enumerate(splits["folds"]):
        fold_num = fold_idx + 1
        fold_dir = results_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_idxs = [sid_to_idx[s] for s in fold["train"]]
        test_idxs = [sid_to_idx[s] for s in fold["val"]]

        Xs_train = [Xs[i] for i in train_idxs]
        Xs_test = [Xs[i] for i in test_idxs]
        Y_train = Y[train_idxs]
        Y_test = Y[test_idxs]

        Xs_train, Xs_test = normalize_bags(Xs_train, Xs_test)

        # Internal val split for early stopping (15% of train)
        n_val = max(1, int(len(Xs_train) * 0.15))
        torch.manual_seed(seed + fold_idx)
        perm = torch.randperm(len(Xs_train))
        val_idxs = perm[:n_val]
        trn_idxs = perm[n_val:]
        Xs_trn = [Xs_train[i] for i in trn_idxs]
        Xs_val = [Xs_train[i] for i in val_idxs]
        Y_trn = Y_train[trn_idxs]
        Y_val = Y_train[val_idxs]

        print(f"\n  {model_name} Fold {fold_num}: train={len(Xs_trn)}, val={len(Xs_val)}, test={len(Xs_test)}")

        torch.manual_seed(seed + fold_idx)
        model = model_cls(**model_kwargs)
        t0 = time.time()
        model, history = train_one_fold(model, Xs_trn, Y_trn, Xs_val, Y_val, device, n_epochs, lr, patience)
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s (stopped at epoch {len(history)})")

        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

        model.eval()
        model.cpu()
        preds = []
        with torch.no_grad():
            for x in Xs_test:
                preds.append(model(x).item())
        y_pred = np.array(preds)
        y_true = Y_test.numpy()

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold_num
        metrics["train_time_s"] = round(elapsed, 1)
        metrics["best_epoch"] = len(history)
        all_results.append(metrics)
        print(f"  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
              f"r={metrics['pearson_r']:.4f}  rho={metrics['spearman_rho']:.4f}")

        pd.DataFrame({
            "sample_id": fold["val"],
            "y_true": y_true,
            "y_pred": y_pred,
        }).to_csv(fold_dir / "predictions.csv", index=False)
        torch.save(model.state_dict(), fold_dir / "model.pt")

    all_df = pd.DataFrame(all_results)
    all_df.insert(0, "model", model_name)
    all_df.to_csv(results_dir / f"AllFolds_{model_name}.csv", index=False)

    print(f"\n  {model_name} Summary:")
    for col in ["r2", "mae", "pearson_r", "spearman_rho"]:
        print(f"    {col:15s}  {all_df[col].mean():.4f} +/- {all_df[col].std():.4f}")

    return all_df
