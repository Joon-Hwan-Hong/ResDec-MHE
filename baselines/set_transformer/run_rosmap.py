"""Set Transformer baseline for ROSMAP cognitive resilience regression.

Set Transformer (Lee et al. 2019) with ISAB + PMA pooling and MSE regression head,
on 30-dim scVI embeddings using our exact 5-fold splits.

Usage:
    uv run python baselines/set_transformer/run_rosmap.py \
        --data-h5ad baselines/shared/mixmil_input.h5ad \
        --splits outputs/splits.json \
        --results-dir outputs/baselines/set_transformer \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from mil_utils import load_data, run_5fold


class MAB(nn.Module):
    """Multi-head Attention Block."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(Q, K, K)
        Q = self.norm1(Q + out)
        Q = self.norm2(Q + self.ff(Q))
        return Q


class SetTransformer(nn.Module):
    """Set Transformer (Lee et al. 2019) adapted for regression.

    Uses Induced Set Attention Blocks (ISAB) for O(n*m) complexity,
    then Pooling by Multihead Attention (PMA) to produce fixed-size output.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_inducing: int = 16, num_isab: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.inducing = nn.Parameter(torch.randn(1, num_inducing, hidden_dim) * 0.01)
        self.isab_blocks = nn.ModuleList()
        for _ in range(num_isab):
            self.isab_blocks.append(nn.ModuleList([
                MAB(hidden_dim, num_heads),
                MAB(hidden_dim, num_heads),
            ]))
        self.pma_seed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        self.pma = MAB(hidden_dim, num_heads)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n_cells, input_dim] -> scalar prediction"""
        h = self.proj(x).unsqueeze(0)  # [1, n_cells, hidden_dim]
        inducing = self.inducing
        for mab_to_ind, mab_to_inst in self.isab_blocks:
            inducing_out = mab_to_ind(inducing.expand(h.size(0), -1, -1), h)
            h = mab_to_inst(h, inducing_out)
        z = self.pma(self.pma_seed.expand(h.size(0), -1, -1), h)
        return self.regressor(z.squeeze(0)).squeeze()


def main():
    parser = argparse.ArgumentParser(description="Set Transformer regression baseline")
    parser.add_argument("--data-h5ad", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--results-dir", default="outputs/baselines/set_transformer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pickle-cache", default=None, help="scPhase pickle cache for fast loading of raw genes")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w") as f:
        json.dump({**vars(args), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sample_ids, Xs, Y = load_data(args.data_h5ad, pickle_cache=args.pickle_cache)
    Q = Xs[0].shape[1]
    with open(args.splits) as f:
        splits = json.load(f)

    run_5fold(
        model_name="SetTransformer",
        model_cls=SetTransformer,
        model_kwargs={"input_dim": Q, "hidden_dim": 128, "num_heads": 4,
                       "num_inducing": 16, "num_isab": 2},
        sample_ids=sample_ids, Xs=Xs, Y=Y, splits=splits,
        results_dir=results_dir, device=torch.device(args.device),
        n_epochs=args.n_epochs, lr=args.lr, patience=args.patience, seed=args.seed,
    )


if __name__ == "__main__":
    main()
