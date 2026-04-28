"""
Perceiver IO baseline for ROSMAP cognitive resilience regression.

Self-contained Perceiver IO implementation for multi-modal input:
  - Pseudobulk [B, 31, 4796]: 31 cell-type tokens
  - CCC summary [B, 18]: 1 aggregated token
Per-modality linear projections to d_model, cross-attention encoder,
4 self-attention blocks, decoder cross-attention to scalar output.

Usage:
    baselines/perceiver_io/.venv/bin/python baselines/perceiver_io/run_rosmap.py \
        --data-dir data/precomputed/ \
        --splits outputs/splits.json \
        --metadata-dir data/metadata_ROSMAP/ \
        --results-dir outputs/baselines/perceiver_io \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Shared project utilities
_SHARED_DIR = str(Path(__file__).resolve().parent.parent / "shared")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

from pt_data_utils import (
    compute_metrics,
    extract_ccc_summary,
    load_metadata,
    load_splits,
    load_subject_pt,
    save_results,
)


# ---------------------------------------------------------------------------
# Perceiver IO model
# ---------------------------------------------------------------------------

class PerceiverIORegressor(nn.Module):
    """Perceiver IO for multi-modal regression.

    Input tokens:
      - 31 cell-type tokens from pseudobulk [B, 31, 4796] projected to d_model
      - 1 CCC summary token from [B, 18] projected to d_model
    Total: 32 input tokens.

    Architecture:
      - Cross-attention: learnable latents [32, d_model] attend to input tokens
      - 4 self-attention blocks (pre-norm: LN -> MHA -> residual -> LN -> FFN -> residual)
      - Decoder: single output query -> cross-attention to latents -> linear -> scalar
    """

    def __init__(
        self,
        n_cell_types: int = 31,
        gene_dim: int = 4796,
        ccc_dim: int = 18,
        d_model: int = 128,
        n_heads: int = 4,
        n_self_attn_blocks: int = 4,
        n_latents: int = 32,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-modality input projections
        self.pseudobulk_proj = nn.Linear(gene_dim, d_model)
        self.ccc_proj = nn.Linear(ccc_dim, d_model)

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        # Cross-attention: latents attend to input tokens
        self.cross_attn_norm_latent = nn.LayerNorm(d_model)
        self.cross_attn_norm_input = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )

        # Self-attention blocks
        self.self_attn_blocks = nn.ModuleList()
        for _ in range(n_self_attn_blocks):
            self.self_attn_blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "norm2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ffn_mult, d_model),
                ),
            }))

        # Decoder: single output query -> cross-attention to latents -> scalar
        self.output_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dec_cross_attn_norm_q = nn.LayerNorm(d_model)
        self.dec_cross_attn_norm_kv = nn.LayerNorm(d_model)
        self.dec_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, pseudobulk: torch.Tensor, ccc: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pseudobulk: [B, 31, 4796] cell-type expression profiles
            ccc: [B, 18] CCC summary features

        Returns:
            [B] scalar predictions
        """
        B = pseudobulk.shape[0]

        # Project inputs to d_model
        pb_tokens = self.pseudobulk_proj(pseudobulk)       # [B, 31, d_model]
        ccc_token = self.ccc_proj(ccc).unsqueeze(1)         # [B, 1, d_model]
        input_tokens = torch.cat([pb_tokens, ccc_token], dim=1)  # [B, 32, d_model]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, 32, d_model]

        # Cross-attention: latents attend to input tokens (pre-norm)
        latents_normed = self.cross_attn_norm_latent(latents)
        input_normed = self.cross_attn_norm_input(input_tokens)
        latents = latents + self.cross_attn(
            latents_normed, input_normed, input_normed,
        )[0]

        # Self-attention blocks
        for block in self.self_attn_blocks:
            # Pre-norm self-attention + residual
            x_norm = block["norm1"](latents)
            latents = latents + block["attn"](x_norm, x_norm, x_norm)[0]
            # Pre-norm FFN + residual
            x_norm = block["norm2"](latents)
            latents = latents + block["ffn"](x_norm)

        # Decoder cross-attention: output query attends to latents
        query = self.output_query.expand(B, -1, -1)  # [B, 1, d_model]
        query_normed = self.dec_cross_attn_norm_q(query)
        latents_normed = self.dec_cross_attn_norm_kv(latents)
        decoded = self.dec_cross_attn(
            query_normed, latents_normed, latents_normed,
        )[0]  # [B, 1, d_model]

        # Project to scalar
        out = self.output_head(decoded).squeeze(-1).squeeze(-1)  # [B]
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ROSMAPDataset(torch.utils.data.Dataset):
    """Simple dataset: pre-loaded pseudobulk + CCC summary arrays."""

    def __init__(
        self,
        pseudobulks: np.ndarray,   # [N, 31, 4796]
        ccc_summaries: np.ndarray,  # [N, 18]
        targets: np.ndarray,        # [N]
        subject_ids: list[str],
    ):
        self.pseudobulks = torch.from_numpy(pseudobulks).float()
        self.ccc_summaries = torch.from_numpy(ccc_summaries).float()
        self.targets = torch.from_numpy(targets).float()
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.pseudobulks[idx], self.ccc_summaries[idx], self.targets[idx]


def load_fold_data(
    data_dir: Path,
    subject_ids: list[str],
    targets_map: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load pseudobulk and CCC summary for a list of subjects.

    Returns:
        pseudobulks: [N, 31, 4796]
        ccc_summaries: [N, 18]
        targets: [N]
        valid_sids: subject IDs that were successfully loaded
    """
    pseudobulks = []
    ccc_summaries = []
    target_vals = []
    valid_sids = []

    for sid in subject_ids:
        if sid not in targets_map:
            continue
        pt_data = load_subject_pt(data_dir, sid)
        pb = pt_data["pseudobulk"]
        if isinstance(pb, torch.Tensor):
            pb = pb.numpy()
        pseudobulks.append(pb.astype(np.float32))
        ccc_summaries.append(extract_ccc_summary(pt_data))
        target_vals.append(targets_map[sid])
        valid_sids.append(sid)

    return (
        np.stack(pseudobulks),     # [N, 31, 4796]
        np.stack(ccc_summaries),   # [N, 18]
        np.array(target_vals, dtype=np.float32),
        valid_sids,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(
    model: PerceiverIORegressor,
    train_dataset: ROSMAPDataset,
    val_dataset: ROSMAPDataset,
    device: torch.device,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    patience: int = 15,
    batch_size: int = 32,
) -> dict:
    """Train one fold with early stopping on val MSE.

    Returns:
        best_state_dict for the model with lowest validation MSE.
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # ---- Train -----------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for pb, ccc, y in train_loader:
            pb, ccc, y = pb.to(device), ccc.to(device), y.to(device)
            pred = model(pb, ccc)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.shape[0]
            train_n += y.shape[0]

        # ---- Validate --------------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for pb, ccc, y in val_loader:
                pb, ccc, y = pb.to(device), ccc.to(device), y.to(device)
                pred = model(pb, ccc)
                loss = criterion(pred, y)
                val_loss_sum += loss.item() * y.shape[0]
                val_n += y.shape[0]

        val_mse = val_loss_sum / val_n
        train_mse = train_loss_sum / train_n

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"    Epoch {epoch:3d}: train_mse={train_mse:.6f}, val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"    Best val_mse={best_val_mse:.6f}")
    return best_state


@torch.no_grad()
def predict(
    model: PerceiverIORegressor,
    dataset: ROSMAPDataset,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate predictions for a dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for pb, ccc, _ in loader:
        pb, ccc = pb.to(device), ccc.to(device)
        pred = model(pb, ccc)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Perceiver IO baseline — ROSMAP regression")
    parser.add_argument("--data-dir", required=True, help="Directory with precomputed .pt files")
    parser.add_argument("--splits", required=True, help="Path to splits.json")
    parser.add_argument("--metadata-dir", required=True, help="Directory with metadata.csv")
    parser.add_argument("--results-dir", default="outputs/baselines/perceiver_io", help="Output dir")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = results_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # ---- Reproducibility -----------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # ---- Save config ---------------------------------------------------------
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Load metadata and splits --------------------------------------------
    targets = load_metadata(args.metadata_dir)
    splits = load_splits(args.splits)
    device = torch.device(args.device)
    print(f"Loaded {len(targets)} subjects with cogn_global targets")
    print(f"Running {len(splits['folds'])} folds on {device}\n")

    # ---- 5-fold CV -----------------------------------------------------------
    fold_results = []
    data_dir = Path(args.data_dir)

    for fold_idx, fold in enumerate(splits["folds"]):
        fold_num = fold_idx + 1
        print(f"{'='*60}")
        print(f"  Fold {fold_num}")
        print(f"{'='*60}")

        # ---- Load fold data --------------------------------------------------
        train_pb, train_ccc, train_y, train_sids = load_fold_data(
            data_dir, fold["train"], targets,
        )
        val_pb, val_ccc, val_y, val_sids = load_fold_data(
            data_dir, fold["val"], targets,
        )
        print(f"  Train: {len(train_sids)} subjects, Val: {len(val_sids)} subjects")
        print(f"  Pseudobulk shape: {train_pb.shape}, CCC shape: {train_ccc.shape}")

        train_dataset = ROSMAPDataset(train_pb, train_ccc, train_y, train_sids)
        val_dataset = ROSMAPDataset(val_pb, val_ccc, val_y, val_sids)

        # ---- Build model -----------------------------------------------------
        # Re-seed each fold for reproducibility
        torch.manual_seed(args.seed + fold_idx)
        np.random.seed(args.seed + fold_idx)

        model = PerceiverIORegressor(
            n_cell_types=train_pb.shape[1],
            gene_dim=train_pb.shape[2],
            ccc_dim=train_ccc.shape[1],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        if fold_idx == 0:
            print(f"  Model parameters: {n_params:,}")

        # ---- Train -----------------------------------------------------------
        t0 = time.time()
        best_state = train_fold(model, train_dataset, val_dataset, device)
        elapsed = time.time() - t0

        # ---- Restore best and predict ----------------------------------------
        model.load_state_dict(best_state)
        y_pred = predict(model, val_dataset, device)
        y_true = val_y

        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = fold_num
        metrics["train_time_s"] = round(elapsed, 1)
        fold_results.append(metrics)
        print(f"  R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  "
              f"r={metrics['pearson_r']:.4f}  rho={metrics['spearman_rho']:.4f}")

        # ---- Save per-fold predictions ---------------------------------------
        np.savez(
            preds_dir / f"fold{fold_idx}.npz",
            sample_ids=np.array(val_sids),
            y_true=y_true,
            y_pred=y_pred,
        )

        # ---- Save model state dict -------------------------------------------
        torch.save(best_state, preds_dir / f"fold{fold_idx}_model.pt")

        # ---- Cleanup ---------------------------------------------------------
        del model, best_state, train_dataset, val_dataset
        del train_pb, train_ccc, train_y, val_pb, val_ccc, val_y
        torch.cuda.empty_cache()
        print()

    # ---- Save aggregated results ---------------------------------------------
    save_results(fold_results, results_dir, "Perceiver IO")


if __name__ == "__main__":
    main()
