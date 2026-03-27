"""ABMIL baseline for ROSMAP cognitive resilience regression.

Gated Attention-Based MIL (Ilse et al. 2018) with MSE regression head,
on 30-dim scVI embeddings using our exact 5-fold splits.

Usage:
    uv run python baselines/abmil/run_rosmap.py \
        --data-h5ad baselines/shared/mixmil_input.h5ad \
        --splits outputs/splits.json \
        --results-dir outputs/baselines/abmil \
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from mil_utils import load_data, run_5fold  # noqa: E402


class ABMIL(nn.Module):
    """Attention-Based MIL (Ilse et al. 2018) adapted for regression.

    Gated attention: a = softmax(W_2 * tanh(W_1 * h) ⊙ sigm(W_3 * h))
    Bag embedding: z = sum(a * h)
    Prediction: y = MLP(z)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, attn_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n_cells, input_dim] -> scalar prediction"""
        h = self.encoder(x)
        a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))
        a = F.softmax(a, dim=0)
        z = (a * h).sum(dim=0, keepdim=True)
        return self.regressor(z).squeeze()


def main():
    parser = argparse.ArgumentParser(description="ABMIL regression baseline")
    parser.add_argument("--data-h5ad", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--results-dir", default="outputs/baselines/abmil")
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
        model_name="ABMIL",
        model_cls=ABMIL,
        model_kwargs={"input_dim": Q, "hidden_dim": 128, "attn_dim": 64},
        sample_ids=sample_ids, Xs=Xs, Y=Y, splits=splits,
        results_dir=results_dir, device=torch.device(args.device),
        n_epochs=args.n_epochs, lr=args.lr, patience=args.patience, seed=args.seed,
    )


if __name__ == "__main__":
    main()
