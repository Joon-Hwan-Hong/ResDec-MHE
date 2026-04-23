"""
CloudPred baseline — per-cell-type variant for ROSMAP cognitive resilience.

Instead of treating all cells as one unstructured bag, this variant:
  1. Splits cells by cell type using cell_offsets (31 types)
  2. Fits a separate GMM per active cell type
  3. Concatenates per-type density features → polynomial regression

This gives CloudPred awareness of cell type structure, making it a fairer
comparison to the main multi-view model, which processes cell types separately.

Usage:
    baselines/cloudpred/.venv/bin/python baselines/cloudpred/run_rosmap_pertype.py \\
        --data-dir data/precomputed/ \\
        --splits outputs/splits.json \\
        --metadata-dir data/metadata_ROSMAP/ \\
        --results-dir outputs/baselines/cloudpred_pertype \\
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

N_CELL_TYPES = 31
MIN_TRAIN_CELLS = 500  # min total training cells to fit GMM for a type


# ---------------------------------------------------------------------------
# Vectorized CloudPred modules (same as run_rosmap.py)
# ---------------------------------------------------------------------------

class BatchedPerTypeMixture(nn.Module):
    """All T per-type mixtures computed in a single batched GPU kernel.

    Instead of looping over T types with separate mixture.forward() calls
    (24 GPU kernel launches per subject), this stacks all type parameters
    into [T, K, D] tensors and processes them in one fused operation.
    """

    def __init__(self, all_mus: torch.Tensor, all_invvars: torch.Tensor,
                 all_weights: torch.Tensor):
        """
        Args:
            all_mus: [T, K, D] — stacked Gaussian means per type
            all_invvars: [T, K, D] — stacked inverse variances
            all_weights: [T, K] — stacked mixture weights
        """
        super().__init__()
        self.all_mus = nn.Parameter(all_mus)           # [T, K, D]
        self.all_invvars = nn.Parameter(all_invvars)     # [T, K, D]
        self.all_weights = nn.Parameter(all_weights.unsqueeze(-1))  # [T, K, 1]

    def forward(self, cells_padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute per-type density features in one batched op.

        Args:
            cells_padded: [T, max_cells, D] — padded cells per type
            mask: [T, max_cells] — True for real cells, False for padding

        Returns:
            [T * K] — flattened density features for all types
        """
        T, K, D = self.all_mus.shape
        invvar = torch.abs(self.all_invvars).clamp(1e-5)  # [T, K, D]

        # Log-density: [T, K, max_cells]
        log_det = torch.sum(torch.log(invvar), dim=2, keepdim=True)  # [T, K, 1]
        # [T, K, max_cells, D] = [T, K, 1, D] - [T, 1, max_cells, D]
        diff = self.all_mus.unsqueeze(2) - cells_padded.unsqueeze(1)
        sq_maha = torch.sum(diff ** 2 * invvar.unsqueeze(2), dim=3)  # [T, K, max_cells]
        logp = -0.5 * (D * math.log(2 * math.pi) - log_det + sq_maha)  # [T, K, max_cells]

        # Softmax normalization across K components: [T, K, max_cells]
        shift, _ = torch.max(logp, dim=1, keepdim=True)  # [T, 1, max_cells]
        p = torch.exp(logp - shift) * self.all_weights  # [T, K, max_cells]
        p = p / torch.sum(p, dim=1, keepdim=True).clamp(min=1e-10)  # [T, K, max_cells]

        # Masked mean across cells: [T, K]
        mask_exp = mask.unsqueeze(1).float()  # [T, 1, max_cells]
        p = p * mask_exp  # zero out padding
        counts = mask.sum(dim=1, keepdim=True).unsqueeze(1).clamp(min=1).float()  # [T, 1, 1]
        features = p.sum(dim=2) / counts.squeeze(-1)  # [T, K]

        return features.reshape(-1)  # [T * K]


class PolynomialLayer(nn.Module):
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
# Per-cell-type classifier
# ---------------------------------------------------------------------------

class PerTypeDensityClassifier(nn.Module):
    """CloudPred with batched per-type GMM.

    All T type-specific mixtures are stacked into one BatchedPerTypeMixture.
    Forward pass: one batched GPU kernel for all types → polynomial regression.
    """

    def __init__(
        self,
        mixture: BatchedPerTypeMixture,
        total_features: int,
        states: int = 2,
    ):
        super().__init__()
        self.mixture = mixture
        self.total_features = total_features
        self.pl = PolynomialLayer(total_features, states)

    def compute_features(
        self,
        cells_padded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute features via batched mixture.

        Args:
            cells_padded: [T, max_cells, D]
            mask: [T, max_cells]

        Returns:
            [1, total_features]
        """
        return self.mixture(cells_padded, mask).unsqueeze(0)

    def forward(
        self,
        cells_padded: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward: batched mixture → polynomial → prediction."""
        feat = self.compute_features(cells_padded, mask)
        return self.pl(feat)  # [1, states]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_fold_subjects_with_types(
    data_dir: Path,
    subject_ids: list[str],
    targets: dict[str, float],
) -> list[dict]:
    """Load cell_data + cell_offsets for each subject.

    Returns list of dicts with keys: cell_data, cell_offsets, target, sid.
    """
    subjects = []
    for sid in subject_ids:
        if sid not in targets:
            continue
        pt_data = load_subject_pt(data_dir, sid)
        cell_data = pt_data["cell_data"]
        if isinstance(cell_data, torch.Tensor):
            cell_data = cell_data.numpy()
        offsets = pt_data["cell_offsets"]
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.numpy()
        subjects.append({
            "cell_data": cell_data.astype(np.float32),
            "cell_offsets": offsets,
            "target": targets[sid],
            "sid": sid,
        })
    return subjects


def split_cells_by_type(
    cell_data: np.ndarray,
    cell_offsets: np.ndarray,
) -> dict[int, np.ndarray]:
    """Split [total_cells, D] into per-type arrays using offsets."""
    result = {}
    for t in range(len(cell_offsets) - 1):
        start, end = int(cell_offsets[t]), int(cell_offsets[t + 1])
        if end > start:
            result[t] = cell_data[start:end]
    return result


class TorchPCA:
    """GPU-accelerated PCA via torch.pca_lowrank."""

    def __init__(self, n_components: int = 10, device: torch.device = torch.device("cpu")):
        self.n_components = n_components
        self.device = device
        self.mean_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None

    def fit(self, X: np.ndarray) -> "TorchPCA":
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.mean_ = X_t.mean(dim=0)
        X_t -= self.mean_  # center in-place to avoid 2x memory
        _, _, V = torch.pca_lowrank(X_t, q=self.n_components, center=False, niter=2)
        self.components_ = V.T
        del X_t
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_reduced = (X_t - self.mean_) @ self.components_.T
        result = X_reduced.cpu().numpy()
        del X_t, X_reduced
        return result


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def determine_active_types(
    train_subjects: list[dict],
    min_cells: int = MIN_TRAIN_CELLS,
) -> list[int]:
    """Find cell types with enough total training cells to fit a GMM."""
    total_counts = np.zeros(N_CELL_TYPES, dtype=np.int64)
    for subj in train_subjects:
        offsets = subj["cell_offsets"]
        for t in range(N_CELL_TYPES):
            total_counts[t] += int(offsets[t + 1]) - int(offsets[t])

    active = [t for t in range(N_CELL_TYPES) if total_counts[t] >= min_cells]
    return active, total_counts


def build_pertype_model(
    train_subjects: list[dict],
    pca: TorchPCA,
    active_types: list[int],
    k_per_type: int = 3,
    device: torch.device = torch.device("cpu"),
) -> PerTypeDensityClassifier:
    """Build PerTypeDensityClassifier with batched per-type GMMs."""
    T = len(active_types)
    D = pca.n_components

    all_mus = torch.zeros(T, k_per_type, D)
    all_invvars = torch.ones(T, k_per_type, D)
    all_weights = torch.zeros(T, k_per_type)

    for i, t in enumerate(active_types):
        # Collect all cells of this type from training subjects
        type_cells = []
        for subj in train_subjects:
            offsets = subj["cell_offsets"]
            start, end = int(offsets[t]), int(offsets[t + 1])
            if end > start:
                type_cells.append(subj["cell_data"][start:end])

        all_type_cells = np.concatenate(type_cells, axis=0)
        all_type_cells_pca = pca.transform(all_type_cells)
        del all_type_cells

        n_cells = all_type_cells_pca.shape[0]
        actual_k = min(k_per_type, max(1, n_cells // 10))

        gm = GaussianMixture(
            n_components=actual_k, covariance_type="diag", random_state=42,
        )
        gm.fit(all_type_cells_pca)
        del all_type_cells_pca

        # Fill into stacked tensors (remaining slots stay at zero-weight defaults)
        all_mus[i, :actual_k] = torch.tensor(gm.means_, dtype=torch.float32)
        all_invvars[i, :actual_k] = torch.tensor(1.0 / gm.covariances_, dtype=torch.float32)
        all_weights[i, :actual_k] = torch.tensor(gm.weights_, dtype=torch.float32)

    mixture = BatchedPerTypeMixture(all_mus, all_invvars, all_weights)
    total_features = T * k_per_type
    classifier = PerTypeDensityClassifier(mixture, total_features, states=2)
    return classifier.to(device)


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def prepare_subject_tensors(
    subjects: list[dict],
    pca: TorchPCA,
    active_types: list[int],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert subjects to pre-padded GPU tensors for batched mixture.

    Returns list of (cells_padded [T, max_cells, D], mask [T, max_cells], y [1]).
    Each subject's types are padded to that subject's max cell count.
    """
    T = len(active_types)
    D = pca.n_components
    result = []

    for subj in subjects:
        offsets = subj["cell_offsets"]
        # PCA-transform per-type cells and find max_cells for this subject
        type_cells = []
        for t in active_types:
            start, end = int(offsets[t]), int(offsets[t + 1])
            if end > start:
                raw = subj["cell_data"][start:end]
                type_cells.append(pca.transform(raw).astype(np.float32))
            else:
                type_cells.append(None)

        max_cells = max((c.shape[0] for c in type_cells if c is not None), default=1)

        # Build padded tensor and mask
        cells_padded = torch.zeros(T, max_cells, D, device=device)
        mask = torch.zeros(T, max_cells, dtype=torch.bool, device=device)

        for i, cells in enumerate(type_cells):
            if cells is not None:
                n = cells.shape[0]
                cells_padded[i, :n] = torch.tensor(cells, device=device)
                mask[i, :n] = True

        y = torch.tensor([subj["target"]], dtype=torch.float32, device=device)
        result.append((cells_padded, mask, y))

    return result


def warmup_polynomial_head(
    classifier: PerTypeDensityClassifier,
    train_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_steps: int = 1000,
    lr: float = 1e-3,
) -> PerTypeDensityClassifier:
    """Phase 1: Precompute per-type density features, train polynomial head only."""
    # Precompute features (detached)
    with torch.no_grad():
        X_train = torch.cat([
            classifier.compute_features(cp, m) for cp, m, _ in train_tensors
        ])  # [N_train, total_features]
        y_train = torch.cat([y for _, _, y in train_tensors])

        X_val = torch.cat([
            classifier.compute_features(cp, m) for cp, m, _ in val_tensors
        ])
        y_val = torch.cat([y for _, _, y in val_tensors])

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
    classifier: PerTypeDensityClassifier,
    train_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_iterations: int = 1000,
    lr: float = 1e-4,
) -> PerTypeDensityClassifier:
    """Phase 2: End-to-end stochastic SGD fine-tuning (batched per-type)."""
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(classifier.state_dict())

    n_train = len(train_tensors)
    n_val = len(val_tensors)

    print(f"    Fine-tuning: {n_iterations} iters x {n_train} subjects (batched)",
          flush=True)

    for iteration in range(n_iterations):
        classifier.train()
        train_loss_sum = 0.0
        for cells_padded, mask, y in train_tensors:
            z = classifier(cells_padded, mask)
            loss = criterion(z[:, 1], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        classifier.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for cells_padded, mask, y in val_tensors:
                z = classifier(cells_padded, mask)
                val_loss_sum += criterion(z[:, 1], y).item()

        val_loss = val_loss_sum / n_val
        if val_loss <= best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(classifier.state_dict())

        if iteration % 100 == 0 or iteration == n_iterations - 1:
            print(f"    Iter {iteration:4d}: train={train_loss_sum / n_train:.6f} "
                  f"val={val_loss:.6f}", flush=True)

    classifier.load_state_dict(best_state)
    print(f"    Fine-tune done. Best val_loss={best_loss:.6f}", flush=True)
    return classifier


def predict(
    classifier: PerTypeDensityClassifier,
    tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> np.ndarray:
    classifier.eval()
    preds = []
    with torch.no_grad():
        for cells_padded, mask, _ in tensors:
            z = classifier(cells_padded, mask)
            preds.append(z[:, 1].item())
    return np.array(preds, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CloudPred baseline (per-cell-type) — ROSMAP regression")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--results-dir", default="outputs/baselines/cloudpred_pertype")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k-per-type", type=int, default=3, help="GMM centers per cell type")
    parser.add_argument("--n-pca-dims", type=int, default=10, help="PCA dimensions")
    parser.add_argument("--min-train-cells", type=int, default=MIN_TRAIN_CELLS,
                        help="Min total training cells to activate a type")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-steps", type=int, default=1000)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    targets = load_metadata(args.metadata_dir)
    splits = load_splits(args.splits)
    print(f"Loaded {len(targets)} subjects with cogn_global targets", flush=True)
    print(f"Running {len(splits['folds'])} folds\n", flush=True)

    data_dir = Path(args.data_dir)
    fold_results = []

    for fold_idx, fold in enumerate(splits["folds"]):
        fold_num = fold_idx + 1
        fold_dir = results_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"{'='*60}", flush=True)
        print(f"  Fold {fold_num}", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()

        # ---- Load subjects with cell type info ----------------------------
        print("  Loading fold subjects...", flush=True)
        train_subjects = load_fold_subjects_with_types(data_dir, fold["train"], targets)
        val_subjects = load_fold_subjects_with_types(data_dir, fold["val"], targets)
        print(f"  Train: {len(train_subjects)}, Val: {len(val_subjects)}", flush=True)

        # ---- Determine active cell types ----------------------------------
        active_types, total_counts = determine_active_types(
            train_subjects, min_cells=args.min_train_cells,
        )
        print(f"  Active cell types: {len(active_types)}/{N_CELL_TYPES} "
              f"(min {args.min_train_cells} training cells)", flush=True)
        # Load type names for display
        pt0 = load_subject_pt(data_dir, fold["train"][0])
        type_names = pt0.get("cell_type_order", [f"type_{i}" for i in range(N_CELL_TYPES)])
        for t in active_types:
            print(f"    [{t:2d}] {type_names[t]:<45s} {total_counts[t]:>8,d} cells", flush=True)

        # ---- Global PCA ---------------------------------------------------
        print("  Fitting global PCA...", flush=True)
        all_cells = np.concatenate([s["cell_data"] for s in train_subjects], axis=0)
        print(f"  PCA: {all_cells.shape[0]:,} cells x {all_cells.shape[1]} genes -> "
              f"{args.n_pca_dims} dims", flush=True)
        pca = TorchPCA(n_components=args.n_pca_dims, device=device)
        pca.fit(all_cells)
        del all_cells
        t_load = time.time() - t0
        print(f"  Data loading + PCA: {t_load:.1f}s", flush=True)

        # ---- Build per-type model -----------------------------------------
        print(f"  Building per-type GMM model (K={args.k_per_type} per type)...", flush=True)
        classifier = build_pertype_model(
            train_subjects, pca, active_types,
            k_per_type=args.k_per_type, device=device,
        )
        total_features = len(active_types) * args.k_per_type
        print(f"  Total density features: {len(active_types)} types x {args.k_per_type} = "
              f"{total_features}", flush=True)

        # ---- Prepare GPU tensors ------------------------------------------
        print("  Preparing GPU tensors...", flush=True)
        train_tensors = prepare_subject_tensors(
            train_subjects, pca, active_types, device,
        )
        val_tensors = prepare_subject_tensors(
            val_subjects, pca, active_types, device,
        )
        # Free raw data
        del train_subjects, val_subjects

        # ---- Phase 1: warmup polynomial head ------------------------------
        print("  Phase 1: Warmup (polynomial head only)", flush=True)
        classifier = warmup_polynomial_head(
            classifier, train_tensors, val_tensors,
            n_steps=args.warmup_steps, lr=args.warmup_lr,
        )

        # ---- Phase 2: end-to-end fine-tuning ------------------------------
        print("  Phase 2: End-to-end fine-tuning", flush=True)
        classifier = finetune_end_to_end(
            classifier, train_tensors, val_tensors,
            n_iterations=args.finetune_steps, lr=args.finetune_lr,
        )
        elapsed = time.time() - t0

        # ---- Predict on validation ----------------------------------------
        y_true = np.array([y.item() for _, _, y in val_tensors])
        y_pred = predict(classifier, val_tensors)

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold_num
        metrics["train_time_s"] = round(elapsed, 1)
        metrics["n_active_types"] = len(active_types)
        metrics["total_features"] = total_features
        fold_results.append(metrics)
        print(f"  Fold {fold_num} done in {elapsed:.1f}s — "
              f"R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
              f"r={metrics['pearson_r']:.4f}  rho={metrics['spearman_rho']:.4f}",
              flush=True)

        # ---- Save outputs -------------------------------------------------
        val_sids = [s for s in fold["val"] if s in targets]
        np.savez(
            fold_dir / "predictions.npz",
            sample_ids=np.array(val_sids),
            y_true=y_true,
            y_pred=y_pred,
        )
        torch.save(classifier.state_dict(), fold_dir / "model.pt")

        del classifier, pca, train_tensors, val_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(flush=True)

    save_results(fold_results, results_dir, "CloudPred (per-type)")


if __name__ == "__main__":
    main()
