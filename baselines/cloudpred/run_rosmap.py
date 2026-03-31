"""
CloudPred baseline for ROSMAP cognitive resilience regression.

Implements CloudPred (Toloşi & Bock 2023) with Gaussian mixture density features
and a polynomial classifier head, adapted for continuous regression on snRNA-seq
cell_data tensors.

Optimized vs. reference repo:
  - Vectorized Mixture: all K Gaussians computed in a single batched matmul
    instead of a Python loop. Numerically equivalent.
  - GPU training: model + data on GPU for fine-tuning.
  - Per-fold data loading: avoids holding all 516 subjects' raw 4796-dim data
    in memory simultaneously (~30 GB saving).

Algorithm is faithful to cloudpred/cloudpred.py:train():
  Phase 1: GMM init + warmup polynomial head on precomputed mixture features
  Phase 2: End-to-end stochastic SGD fine-tuning (one subject per step)

Usage:
    baselines/cloudpred/.venv/bin/python baselines/cloudpred/run_rosmap.py \\
        --data-dir data/precomputed/ \\
        --splits outputs/splits.json \\
        --metadata-dir data/metadata_ROSMAP/ \\
        --results-dir outputs/baselines/cloudpred \\
        --device cuda:1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

# Shared project utilities
_SHARED_DIR = str(Path(__file__).resolve().parent.parent / "shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from pt_data_utils import (  # noqa: E402
    compute_metrics,
    load_metadata,
    load_splits,
    load_subject_pt,
    save_results,
)


# ---------------------------------------------------------------------------
# Vectorized CloudPred modules (numerically equivalent to repo, no Python loops)
# ---------------------------------------------------------------------------

class VectorizedMixture(nn.Module):
    """Mixture of diagonal Gaussians — fully vectorized.

    Equivalent to cloudpred.cloudpred.Mixture but computes all K components
    in a single batched operation instead of a Python for-loop.
    """

    def __init__(self, mus: torch.Tensor, invvars: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            mus: [K, D] — Gaussian means (from GMM)
            invvars: [K, D] — inverse variances (1 / covariance diagonal)
            weights: [K] — mixture weights
        """
        super().__init__()
        self.mus = nn.Parameter(mus)               # [K, D]
        self.invvars = nn.Parameter(invvars)         # [K, D]
        self.weights = nn.Parameter(weights.unsqueeze(1))  # [K, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mixture density features for a set of cells.

        Args:
            x: [N, D] cell features (PCA-reduced)

        Returns:
            [K] — normalized mean density per component (same as repo Mixture.forward)
        """
        K, D = self.mus.shape
        invvar = torch.abs(self.invvars).clamp(1e-5)  # [K, D]

        # Log-density of each cell under each Gaussian: [K, N]
        # Equivalent to: for each k, -0.5*(D*log(2π) - sum(log(invvar_k)) + sum((mu_k - x)^2 * invvar_k))
        log_det = torch.sum(torch.log(invvar), dim=1, keepdim=True)  # [K, 1]
        diff = self.mus.unsqueeze(1) - x.unsqueeze(0)  # [K, N, D]
        sq_maha = torch.sum(diff ** 2 * invvar.unsqueeze(1), dim=2)  # [K, N]
        logp = -0.5 * (D * math.log(2 * math.pi) - log_det + sq_maha)  # [K, N]

        # Softmax-style normalization (same as repo: exp(logp - shift) * weights / sum)
        shift, _ = torch.max(logp, dim=0)  # [N]
        p = torch.exp(logp - shift) * self.weights  # [K, N]
        p = p / torch.sum(p, dim=0)  # [K, N] — normalized

        return torch.mean(p, dim=1)  # [K] — mean across cells


class VectorizedDensityClassifier(nn.Module):
    """DensityClassifier with vectorized Mixture.

    Drop-in replacement for cloudpred.cloudpred.DensityClassifier.
    """

    def __init__(self, mixture: VectorizedMixture, n_centers: int, states: int = 2):
        super().__init__()
        self.mixture = mixture
        self.pl = PolynomialLayer(n_centers, states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] cell features
        Returns:
            [1, states] — regression output
        """
        d = self.mixture(x).unsqueeze(0)  # [1, K]
        return self.pl(d)  # [1, states]


class PolynomialLayer(nn.Module):
    """Exact copy of cloudpred.cloudpred.PolynomialLayer."""

    def __init__(self, centers: int, states: int = 2):
        super().__init__()
        self.polynomial = nn.ModuleList([Polynomial(centers) for _ in range(states - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.zeros(x.shape[0], 1, device=x.device)]
            + [p(x).unsqueeze_(1) for p in self.polynomial],
            dim=1,
        )


class Polynomial(nn.Module):
    """Exact copy of cloudpred.cloudpred.Polynomial."""

    def __init__(self, centers: int = 1, degree: int = 2):
        super().__init__()
        self.centers = centers
        self.degree = degree
        self.a = nn.Parameter(torch.zeros(degree, centers))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(
                sum(self.a[i, :] * (x ** (i + 1)) for i in range(self.degree)),
                dim=1,
            )
            + self.c
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_fold_subjects(
    data_dir: Path,
    subject_ids: list[str],
    targets: dict[str, float],
) -> list[tuple[np.ndarray, float, str]]:
    """Load cell_data for fold subjects. Returns (cell_matrix, target, sid) tuples."""
    subjects = []
    for sid in subject_ids:
        if sid not in targets:
            continue
        pt_data = load_subject_pt(data_dir, sid)
        cell_data = pt_data["cell_data"]
        if isinstance(cell_data, torch.Tensor):
            cell_data = cell_data.numpy()
        subjects.append((cell_data.astype(np.float32), targets[sid], sid))
    return subjects


class TorchPCA:
    """GPU-accelerated PCA via torch.pca_lowrank.

    Uses randomized SVD on GPU — much faster than sklearn for large matrices.
    API mirrors sklearn PCA (fit/transform) for drop-in use.
    """

    def __init__(self, n_components: int = 10, device: torch.device = torch.device("cpu")):
        self.n_components = n_components
        self.device = device
        self.mean_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None  # [n_components, n_features]

    def fit(self, X: np.ndarray) -> "TorchPCA":
        """Fit PCA on [n_samples, n_features] array."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.mean_ = X_t.mean(dim=0)
        X_t -= self.mean_  # center in-place to avoid 2x memory
        # pca_lowrank returns (U, S, V) where V is [n_features, n_components]
        _, _, V = torch.pca_lowrank(X_t, q=self.n_components, center=False, niter=2)
        self.components_ = V.T  # [n_components, n_features]
        del X_t
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project [n_samples, n_features] -> [n_samples, n_components]."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_reduced = (X_t - self.mean_) @ self.components_.T
        result = X_reduced.cpu().numpy()
        del X_t, X_reduced
        return result


def fit_pca(
    train_subjects: list[tuple[np.ndarray, float, str]],
    n_components: int = 10,
    device: torch.device = torch.device("cpu"),
) -> TorchPCA:
    """Fit PCA on stacked training cells using GPU-accelerated torch.pca_lowrank."""
    all_cells = np.concatenate([s[0] for s in train_subjects], axis=0)
    print(f"  PCA: fitting on {all_cells.shape[0]:,} training cells -> {n_components} dims "
          f"(device={device})", flush=True)
    pca = TorchPCA(n_components=n_components, device=device)
    pca.fit(all_cells)
    del all_cells
    return pca


def pca_transform(
    subjects: list[tuple[np.ndarray, float, str]],
    pca: TorchPCA,
) -> list[tuple[np.ndarray, float, str]]:
    """Apply PCA transform to each subject's cell matrix."""
    return [(pca.transform(cells).astype(np.float32), target, sid)
            for cells, target, sid in subjects]


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(
    train_subjects: list[tuple[np.ndarray, float, str]],
    n_centers: int = 10,
    device: torch.device = torch.device("cpu"),
) -> VectorizedDensityClassifier:
    """Build VectorizedDensityClassifier with GMM-initialized mixture.

    Faithful to cloudpred/cloudpred.py:train() lines 11-24.
    """
    all_cells = np.concatenate([s[0] for s in train_subjects], axis=0)
    dim = all_cells.shape[1]
    print(f"  GMM: fitting {n_centers} centers on {all_cells.shape[0]:,} cells, dim={dim}",
          flush=True)

    gm = GaussianMixture(n_components=n_centers, covariance_type="diag", random_state=42)
    gm.fit(all_cells)
    del all_cells

    # Build vectorized mixture from GMM parameters
    mus = torch.tensor(gm.means_, dtype=torch.float32)            # [K, D]
    invvars = torch.tensor(1.0 / gm.covariances_, dtype=torch.float32)  # [K, D]
    weights = torch.tensor(gm.weights_, dtype=torch.float32)      # [K]

    mixture = VectorizedMixture(mus, invvars, weights)
    classifier = VectorizedDensityClassifier(mixture, n_centers, states=2)
    return classifier.to(device)


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def warmup_polynomial_head(
    classifier: VectorizedDensityClassifier,
    train_subjects: list[tuple[np.ndarray, float, str]],
    val_subjects: list[tuple[np.ndarray, float, str]],
    n_steps: int = 1000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> VectorizedDensityClassifier:
    """Phase 1: Train polynomial head only on precomputed mixture features.

    Matches cloudpred/cloudpred.py:train() lines 26-68.
    """
    # Precompute mixture features on GPU (detached)
    with torch.no_grad():
        X_train = torch.stack([
            classifier.mixture(torch.tensor(cells, dtype=torch.float32, device=device))
            for cells, _, _ in train_subjects
        ])  # [N_train, K]
        y_train = torch.tensor(
            [t for _, t, _ in train_subjects], dtype=torch.float32, device=device
        )

        X_val = torch.stack([
            classifier.mixture(torch.tensor(cells, dtype=torch.float32, device=device))
            for cells, _, _ in val_subjects
        ])
        y_val = torch.tensor(
            [t for _, t, _ in val_subjects], dtype=torch.float32, device=device
        )

    optimizer = torch.optim.SGD(classifier.pl.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(classifier.pl.state_dict())

    for step in range(n_steps):
        z = classifier.pl(X_train)[:, 1]
        loss = criterion(z, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_loss = criterion(classifier.pl(X_val)[:, 1], y_val)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(classifier.pl.state_dict())
        if step % 200 == 0:
            print(f"    Warmup step {step}: train={loss.item():.6f} val={val_loss.item():.6f}",
                  flush=True)

    classifier.pl.load_state_dict(best_state)
    print(f"    Warmup done. Best val_loss={best_loss.item():.6f}", flush=True)
    return classifier


def finetune_end_to_end(
    classifier: VectorizedDensityClassifier,
    train_subjects: list[tuple[np.ndarray, float, str]],
    val_subjects: list[tuple[np.ndarray, float, str]],
    n_iterations: int = 1000,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> VectorizedDensityClassifier:
    """Phase 2: End-to-end stochastic SGD fine-tuning on GPU.

    Faithful to cloudpred.utils.train_classifier() with stochastic=True:
    each iteration loops over all training subjects, doing per-subject
    forward/backward/step. Validated every iteration, best model kept.
    """
    # Pre-convert to GPU tensors (PCA-reduced, so small: ~3k cells × 10 dims each)
    train_tensors = [
        (torch.tensor(cells, dtype=torch.float32, device=device),
         torch.tensor([target], dtype=torch.float32, device=device))
        for cells, target, _ in train_subjects
    ]
    val_tensors = [
        (torch.tensor(cells, dtype=torch.float32, device=device),
         torch.tensor([target], dtype=torch.float32, device=device))
        for cells, target, _ in val_subjects
    ]

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(classifier.state_dict())

    print(f"    Fine-tuning: {n_iterations} iters × {len(train_tensors)} subjects, "
          f"lr={lr}, device={device}", flush=True)

    for iteration in range(n_iterations):
        # --- Train ---
        classifier.train()
        train_loss_sum = 0.0
        for x, y in train_tensors:
            z = classifier(x)  # [1, 2]
            loss = criterion(z[:, 1], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        # --- Validate ---
        classifier.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_tensors:
                z = classifier(x)
                val_loss_sum += criterion(z[:, 1], y).item()

        val_loss = val_loss_sum / len(val_tensors)
        if val_loss <= best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(classifier.state_dict())

        if iteration % 100 == 0 or iteration == n_iterations - 1:
            train_loss = train_loss_sum / len(train_tensors)
            print(f"    Iter {iteration:4d}: train={train_loss:.6f} val={val_loss:.6f}",
                  flush=True)

    classifier.load_state_dict(best_state)
    print(f"    Fine-tune done. Best val_loss={best_loss:.6f}", flush=True)
    return classifier


def predict(
    classifier: VectorizedDensityClassifier,
    subjects: list[tuple[np.ndarray, float, str]],
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Generate predictions."""
    classifier.eval()
    preds = []
    with torch.no_grad():
        for cells, _, _ in subjects:
            x = torch.tensor(cells, dtype=torch.float32, device=device)
            z = classifier(x)
            preds.append(z[:, 1].item())
    return np.array(preds, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CloudPred baseline — ROSMAP regression")
    parser.add_argument("--data-dir", required=True, help="Directory with precomputed .pt files")
    parser.add_argument("--splits", required=True, help="Path to splits.json")
    parser.add_argument("--metadata-dir", required=True, help="Directory with metadata.csv")
    parser.add_argument("--results-dir", default="outputs/baselines/cloudpred", help="Output dir")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--n-centers", type=int, default=10, help="GMM centers")
    parser.add_argument("--n-pca-dims", type=int, default=10, help="PCA dimensions")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Phase 1 warmup steps")
    parser.add_argument("--warmup-lr", type=float, default=1e-3, help="Phase 1 learning rate")
    parser.add_argument("--finetune-steps", type=int, default=1000, help="Phase 2 iterations")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="Phase 2 learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- Reproducibility --------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # ---- Save config ------------------------------------------------------
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Load metadata and splits -----------------------------------------
    targets = load_metadata(args.metadata_dir)
    splits = load_splits(args.splits)
    print(f"Loaded {len(targets)} subjects with cogn_global targets", flush=True)
    print(f"Running {len(splits['folds'])} folds\n", flush=True)

    data_dir = Path(args.data_dir)

    # ---- 5-fold CV --------------------------------------------------------
    fold_results = []

    for fold_idx, fold in enumerate(splits["folds"]):
        fold_num = fold_idx + 1
        fold_dir = results_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"{'='*60}", flush=True)
        print(f"  Fold {fold_num}", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()

        # ---- Load this fold's subjects (not all 516) ---------------------
        print(f"  Loading fold subjects...", flush=True)
        train_subjects = load_fold_subjects(data_dir, fold["train"], targets)
        val_subjects = load_fold_subjects(data_dir, fold["val"], targets)
        print(f"  Train: {len(train_subjects)} subjects, Val: {len(val_subjects)} subjects",
              flush=True)

        # ---- PCA ---------------------------------------------------------
        pca = fit_pca(train_subjects, n_components=args.n_pca_dims, device=device)
        train_pca = pca_transform(train_subjects, pca)
        val_pca = pca_transform(val_subjects, pca)
        # Free raw 4796-dim data
        del train_subjects, val_subjects
        t_load = time.time() - t0
        print(f"  Data loading + PCA: {t_load:.1f}s", flush=True)

        # ---- Build model with GMM init -----------------------------------
        classifier = build_model(train_pca, n_centers=args.n_centers, device=device)

        # ---- Phase 1: warmup polynomial head -----------------------------
        print("  Phase 1: Warmup (polynomial head only)", flush=True)
        classifier = warmup_polynomial_head(
            classifier, train_pca, val_pca,
            n_steps=args.warmup_steps, lr=args.warmup_lr, device=device,
        )

        # ---- Phase 2: end-to-end fine-tuning -----------------------------
        print("  Phase 2: End-to-end fine-tuning", flush=True)
        classifier = finetune_end_to_end(
            classifier, train_pca, val_pca,
            n_iterations=args.finetune_steps, lr=args.finetune_lr, device=device,
        )
        elapsed = time.time() - t0

        # ---- Predict on validation set -----------------------------------
        y_true = np.array([t for _, t, _ in val_pca])
        y_pred = predict(classifier, val_pca, device=device)

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold_num
        metrics["train_time_s"] = round(elapsed, 1)
        fold_results.append(metrics)
        print(f"  Fold {fold_num} done in {elapsed:.1f}s — "
              f"R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
              f"r={metrics['pearson_r']:.4f}  rho={metrics['spearman_rho']:.4f}",
              flush=True)

        # ---- Save per-fold outputs ---------------------------------------
        val_sids = [s for s in fold["val"] if s in targets]
        np.savez(
            fold_dir / "predictions.npz",
            sample_ids=np.array(val_sids),
            y_true=y_true,
            y_pred=y_pred,
        )
        torch.save(classifier.state_dict(), fold_dir / "model.pt")

        # ---- Cleanup -----------------------------------------------------
        del classifier, pca, train_pca, val_pca
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(flush=True)

    # ---- Save aggregated results -----------------------------------------
    save_results(fold_results, results_dir, "CloudPred")


if __name__ == "__main__":
    main()
