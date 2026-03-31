"""GPIO (Graph Perceiver IO) baseline for ROSMAP cognitive resilience regression.

Self-contained implementation of GPIO (Shirzad et al. 2023): Random Walk
Positional Encoding (RWPE) on cell-communication graphs + Perceiver IO core.
No PyG dependency -- uses only torch, numpy, scipy.

Architecture:
  1. RWPE: Build adjacency from ccc_edge_index, row-normalize to transition
     matrix T, compute diag(T^k) for k=1..16 -> [31, 16] per subject
  2. Input projection: cat(pseudobulk [31, 4796], RWPE [31, 16]) -> Linear -> [31, d_model]
  3. Perceiver IO: learnable latent array [32, d_model], cross-attention from
     latents to graph nodes, 4 self-attention blocks
  4. Graph readout: mean pool latents -> LayerNorm -> Linear -> scalar

Training: Adam lr=1e-4, MSE loss, 100 epochs, early stopping patience=15,
batch_size=32.

Usage:
    baselines/gpio/.venv/bin/python baselines/gpio/run_rosmap.py \\
        --data-dir data/precomputed/ \\
        --splits outputs/splits.json \\
        --metadata-dir data/metadata_ROSMAP/ \\
        --results-dir outputs/baselines/gpio \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Random Walk Positional Encoding (RWPE)
# ---------------------------------------------------------------------------

def compute_rwpe(edge_index: torch.Tensor, n_nodes: int, pe_dim: int = 16) -> torch.Tensor:
    """Compute RWPE from edge_index, faithfully matching GPIO GraphPE.py.

    1. Build sparse adjacency A from edge_index
    2. Row-normalize: T = D^{-1} A (transition matrix)
    3. Compute diag(T^k) for k = 1..pe_dim

    Args:
        edge_index: [2, n_edges] LongTensor
        n_nodes: number of nodes in the graph
        pe_dim: number of random walk steps (default 16)

    Returns:
        PE: [n_nodes, pe_dim] FloatTensor
    """
    if edge_index.shape[1] == 0:
        # No edges: PE is all zeros
        return torch.zeros(n_nodes, pe_dim, dtype=torch.float32)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    # Build sparse adjacency -- matches GraphPE.py line 18
    A = sp.coo_matrix(
        (np.ones(len(src), dtype=np.float32), (src, dst)),
        shape=(n_nodes, n_nodes),
    )

    # Row-normalize: T = D^{-1} A -- matches GraphPE.py lines 22-25
    rowsum = np.array(A.sum(1)).clip(1)
    r_inv = np.power(rowsum, -1.0).flatten()
    Dinv = sp.diags(r_inv)
    # Note: GPIO repo computes RW = A * Dinv (column-normalize).
    # This matches their code exactly: RW = A * Dinv, then diag(RW^k).
    RW = A * Dinv

    # Iterate and collect diagonals -- matches GraphPE.py lines 32-37
    M = RW
    PE = [torch.from_numpy(M.diagonal().copy()).float()]
    M_power = M
    for _ in range(pe_dim - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal().copy()).float())
    PE = torch.stack(PE, dim=-1)  # [n_nodes, pe_dim]

    return PE


# ---------------------------------------------------------------------------
# Perceiver IO components (self-contained, no einops dependency)
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    """GEGLU activation (Shazeer 2020) -- matches GPIO model.py."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """FFN with GEGLU -- matches GPIO model.py."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention without einops -- functionally equivalent to
    GPIO model.py Attention class."""

    def __init__(self, query_dim: int, context_dim: int | None = None,
                 heads: int = 8, dim_head: int = 64):
        super().__init__()
        context_dim = context_dim if context_dim is not None else query_dim
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, N, D], context: [B, M, C] or None (self-attention)."""
        B, N, _ = x.shape
        h = self.heads

        if context is None:
            context = x

        q = self.to_q(x)                           # [B, N, inner]
        k, v = self.to_kv(context).chunk(2, dim=-1)  # [B, M, inner] each

        # Reshape to multi-head: [B, heads, seq, dim_head]
        q = q.view(B, N, h, self.dim_head).transpose(1, 2)
        M = k.shape[1]
        k = k.view(B, M, h, self.dim_head).transpose(1, 2)
        v = v.view(B, M, h, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, N, M]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B, h, N, dim_head]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, inner]
        return self.to_out(out)


class PreNorm(nn.Module):
    """Pre-LayerNorm wrapper -- matches GPIO model.py."""

    def __init__(self, dim: int, fn: nn.Module, context_dim: int | None = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        if self.norm_context is not None and "context" in kwargs:
            kwargs["context"] = self.norm_context(kwargs["context"])
        return self.fn(x, **kwargs)


# ---------------------------------------------------------------------------
# GPIORegressor: Full model
# ---------------------------------------------------------------------------

class GPIORegressor(nn.Module):
    """Graph Perceiver IO for regression on cell-communication graphs.

    Architecture:
      1. Input projection: cat(pseudobulk, RWPE) -> Linear -> d_model
      2. Perceiver IO:
         - Learnable latent array [num_latents, d_model]
         - Cross-attention: latents attend to projected graph nodes
         - Self-attention blocks (depth layers)
      3. Readout: mean pool latents -> LayerNorm -> Linear -> 1
    """

    def __init__(
        self,
        input_dim: int = 4796,
        pe_dim: int = 16,
        d_model: int = 128,
        num_latents: int = 32,
        depth: int = 4,
        n_heads: int = 4,
        cross_heads: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pe_dim = pe_dim

        # Input projection: pseudobulk + RWPE -> d_model
        self.input_proj = nn.Linear(input_dim + pe_dim, d_model)

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))

        # Cross-attention: latents attend to graph nodes (single layer)
        dim_head = d_model // n_heads
        self.cross_attn = PreNorm(
            d_model,
            MultiHeadAttention(d_model, context_dim=d_model, heads=cross_heads, dim_head=dim_head),
            context_dim=d_model,
        )
        self.cross_ff = PreNorm(d_model, FeedForward(d_model))

        # Self-attention blocks
        self.self_attn_layers = nn.ModuleList()
        for _ in range(depth):
            self.self_attn_layers.append(nn.ModuleList([
                PreNorm(d_model, MultiHeadAttention(d_model, heads=n_heads, dim_head=dim_head)),
                PreNorm(d_model, FeedForward(d_model)),
            ]))

        # Readout head
        self.readout_norm = nn.LayerNorm(d_model)
        self.readout_head = nn.Linear(d_model, 1)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [B, n_nodes, input_dim + pe_dim] -- pre-concatenated
                pseudobulk + RWPE features.

        Returns:
            predictions: [B] scalar predictions
        """
        B = node_features.shape[0]

        # Project input
        x = self.input_proj(node_features)  # [B, n_nodes, d_model]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, num_latents, d_model]

        # Cross-attention: latents attend to graph nodes
        latents = self.cross_attn(latents, context=x) + latents
        latents = self.cross_ff(latents) + latents

        # Self-attention blocks
        for self_attn, self_ff in self.self_attn_layers:
            latents = self_attn(latents) + latents
            latents = self_ff(latents) + latents

        # Readout: mean pool latents -> LayerNorm -> linear -> scalar
        pooled = latents.mean(dim=1)  # [B, d_model]
        pooled = self.readout_norm(pooled)
        return self.readout_head(pooled).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_subjects(
    data_dir: Path,
    subject_ids: list[str],
    targets: dict[str, float],
    pe_dim: int = 16,
    n_nodes: int = 31,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load all subjects, compute RWPE, and return stacked tensors.

    For each subject:
      - Load pseudobulk [31, 4796] and ccc_edge_index [2, n_edges]
      - Compute RWPE [31, pe_dim]
      - Concatenate -> [31, 4796 + pe_dim]

    Returns:
        X: [N, 31, 4796 + pe_dim] FloatTensor
        Y: [N] FloatTensor (cogn_global targets)
        valid_sids: list of subject IDs that were loaded
    """
    features_list = []
    targets_list = []
    valid_sids = []

    for sid in subject_ids:
        if sid not in targets:
            continue
        pt_data = load_subject_pt(data_dir, sid)

        pseudobulk = pt_data["pseudobulk"]
        if isinstance(pseudobulk, np.ndarray):
            pseudobulk = torch.from_numpy(pseudobulk)
        pseudobulk = pseudobulk.float()  # [31, 4796]

        edge_index = pt_data["ccc_edge_index"]
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        edge_index = edge_index.long()

        # Compute RWPE
        pe = compute_rwpe(edge_index, n_nodes=n_nodes, pe_dim=pe_dim)  # [31, pe_dim]

        # Concatenate pseudobulk + PE
        node_feat = torch.cat([pseudobulk, pe], dim=-1)  # [31, 4796 + pe_dim]
        features_list.append(node_feat)
        targets_list.append(targets[sid])
        valid_sids.append(sid)

    X = torch.stack(features_list, dim=0)  # [N, 31, 4796 + pe_dim]
    Y = torch.tensor(targets_list, dtype=torch.float32)  # [N]
    return X, Y, valid_sids


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_fold(
    model: GPIORegressor,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
) -> tuple[GPIORegressor, dict]:
    """Train GPIO model for one fold with early stopping.

    Returns:
        model: best model (loaded from checkpoint)
        info: dict with best_epoch, best_val_loss, train_losses, val_losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Move data to device
    X_train_dev = X_train.to(device)
    Y_train_dev = Y_train.to(device)
    X_val_dev = X_val.to(device)
    Y_val_dev = Y_val.to(device)

    n_train = X_train_dev.shape[0]
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    # For reproducible shuffling
    rng = torch.Generator()
    rng.manual_seed(seed)

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        # ---- Train -----------------------------------------------------------
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_dev[idx]
            yb = Y_train_dev[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # ---- Validate --------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_dev)
            val_loss = criterion(val_pred, Y_val_dev).item()
        val_losses.append(val_loss)

        # ---- Early stopping --------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  best={best_val_loss:.6f}@{best_epoch}")

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return model, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPIO baseline — ROSMAP regression")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with precomputed .pt files")
    parser.add_argument("--splits", required=True,
                        help="Path to splits.json")
    parser.add_argument("--metadata-dir", required=True,
                        help="Directory with metadata.csv")
    parser.add_argument("--results-dir", default="outputs/baselines/gpio",
                        help="Output directory")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--num-latents", type=int, default=32,
                        help="Number of Perceiver latents")
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of self-attention blocks")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads in self-attention")
    parser.add_argument("--cross-heads", type=int, default=1,
                        help="Number of attention heads in cross-attention")
    parser.add_argument("--pe-dim", type=int, default=16,
                        help="RWPE positional encoding dimension")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Reproducibility -----------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    print(f"GPIO baseline — device={device}, seed={args.seed}")
    print(f"  d_model={args.d_model}, num_latents={args.num_latents}, "
          f"depth={args.depth}, n_heads={args.n_heads}, pe_dim={args.pe_dim}")

    # ---- Save config ---------------------------------------------------------
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Load metadata and splits --------------------------------------------
    targets = load_metadata(args.metadata_dir)
    splits = load_splits(args.splits)
    print(f"Loaded {len(targets)} subjects with cogn_global targets")
    print(f"Running {len(splits['folds'])} folds\n")

    # ---- Collect all subject IDs across folds --------------------------------
    all_sids = set()
    for fold in splits["folds"]:
        all_sids.update(fold["train"])
        all_sids.update(fold["val"])

    # ---- Pre-load all subjects with RWPE -------------------------------------
    print("Loading subjects and computing RWPE...")
    data_dir = Path(args.data_dir)
    X_all, Y_all, valid_sids = load_all_subjects(
        data_dir, sorted(all_sids), targets,
        pe_dim=args.pe_dim, n_nodes=31,
    )
    # Build sid -> index map for fast fold slicing
    sid_to_idx = {sid: i for i, sid in enumerate(valid_sids)}
    print(f"  Loaded {len(valid_sids)} subjects, "
          f"features shape: {X_all.shape}\n")

    # ---- 5-fold CV -----------------------------------------------------------
    fold_results = []
    input_dim = 4796  # pseudobulk feature dim

    for fold_idx, fold in enumerate(splits["folds"]):
        fold_num = fold_idx + 1
        fold_dir = results_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'=' * 60}")
        print(f"  Fold {fold_num}")
        print(f"{'=' * 60}")

        # Get indices for this fold
        train_idx = [sid_to_idx[s] for s in fold["train"] if s in sid_to_idx]
        val_idx = [sid_to_idx[s] for s in fold["val"] if s in sid_to_idx]
        val_sids = [s for s in fold["val"] if s in sid_to_idx]

        X_train = X_all[train_idx]
        Y_train = Y_all[train_idx]
        X_val = X_all[val_idx]
        Y_val = Y_all[val_idx]

        print(f"  Train: {len(train_idx)} subjects, Val: {len(val_idx)} subjects")

        # ---- Build model -----------------------------------------------------
        # Re-seed for each fold so model init is reproducible
        torch.manual_seed(args.seed + fold_idx)

        model = GPIORegressor(
            input_dim=input_dim,
            pe_dim=args.pe_dim,
            d_model=args.d_model,
            num_latents=args.num_latents,
            depth=args.depth,
            n_heads=args.n_heads,
            cross_heads=args.cross_heads,
        )
        n_params = sum(p.numel() for p in model.parameters())
        if fold_num == 1:
            print(f"  Model parameters: {n_params:,}")

        # ---- Train -----------------------------------------------------------
        t0 = time.time()
        model, train_info = train_one_fold(
            model, X_train, Y_train, X_val, Y_val,
            device=device,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed + fold_idx,
        )
        elapsed = time.time() - t0

        # ---- Predict on validation set ---------------------------------------
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val.to(device)).cpu().numpy()
        y_true = Y_val.numpy()

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold_num
        metrics["train_time_s"] = round(elapsed, 1)
        metrics["best_epoch"] = train_info["best_epoch"]
        metrics["best_val_loss"] = round(train_info["best_val_loss"], 6)
        fold_results.append(metrics)

        print(f"  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
              f"r={metrics['pearson_r']:.4f}  rho={metrics['spearman_rho']:.4f}  "
              f"({elapsed:.1f}s, best@{train_info['best_epoch']})")

        # ---- Save per-fold predictions ---------------------------------------
        np.savez(
            fold_dir / "predictions.npz",
            sample_ids=np.array(val_sids),
            y_true=y_true,
            y_pred=y_pred,
        )

        # ---- Save model state dict -------------------------------------------
        torch.save(model.state_dict(), fold_dir / "model.pt")

        # ---- Cleanup ---------------------------------------------------------
        del model
        torch.cuda.empty_cache()
        print()

    # ---- Save aggregated results ---------------------------------------------
    save_results(fold_results, results_dir, "GPIO")


if __name__ == "__main__":
    main()
