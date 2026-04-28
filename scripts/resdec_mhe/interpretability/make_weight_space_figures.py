"""Orchestrator: render weight-space figures from per-fold checkpoints.

Loads the per-fold ``best-*.ckpt`` Lightning checkpoints under
``--canonical-dir``, flattens each ``state_dict`` into a parameter vector
(trainable float tensors only; buffers like ``_temperature_buf`` and
integer tensors are skipped), stacks into an ``(n_folds, n_params)``
matrix, and calls ``plot_checkpoint_weight_pca`` from
``src.visualization.weight_space_plots``.

Per-fold validation R² is read from each fold's ``best_summary.json``
(``val_results[0]["val/r2"]``) and used as a point annotation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme
from src.visualization.weight_space_plots import (
    plot_checkpoint_weight_pca,
)

logger = logging.getLogger(__name__)


def _flatten_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> np.ndarray:
    """Concatenate all float parameter tensors into one 1-D numpy vector.

    Skips integer buffers and non-tensor entries so the flattened vector
    contains only learnable params. Keys are sorted so the concatenation
    order is identical across checkpoints.
    """
    parts = []
    for key in sorted(state_dict.keys()):
        t = state_dict[key]
        if not isinstance(t, torch.Tensor):
            continue
        if not torch.is_floating_point(t):
            continue
        parts.append(t.detach().cpu().numpy().ravel())
    if not parts:
        raise RuntimeError("no float tensors found in state_dict")
    return np.concatenate(parts)


def _find_best_ckpt(fold_dir: Path) -> Path | None:
    matches = sorted((fold_dir / "checkpoints").glob("best-*.ckpt"))
    return matches[0] if matches else None


def _read_best_r2(fold_dir: Path) -> float | None:
    summary = fold_dir / "best_summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text())
    val_results = data.get("val_results") or []
    if not val_results:
        return None
    return float(val_results[0].get("val/r2", float("nan")))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-dir",
        default="outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/weight_space",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = Path(args.canonical_dir)
    if not canonical.exists():
        logger.error("canonical dir missing: %s", canonical)
        return 1

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    r2s: list[float] = []
    for f in range(args.n_folds):
        fold_dir = canonical / f"fold{f}"
        if not fold_dir.exists():
            logger.warning("skipping %s (missing)", fold_dir)
            continue
        ckpt_path = _find_best_ckpt(fold_dir)
        if ckpt_path is None:
            logger.warning("no best-*.ckpt in %s/checkpoints", fold_dir)
            continue
        logger.info("loading %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict")
        if sd is None:
            logger.warning("no state_dict in %s", ckpt_path)
            continue
        vec = _flatten_state_dict(sd)
        vectors.append(vec)
        labels.append(f"fold {f}")
        r2 = _read_best_r2(fold_dir)
        r2s.append(r2 if r2 is not None else float("nan"))

    if len(vectors) < 2:
        logger.error("need ≥2 checkpoints; got %d", len(vectors))
        return 1
    if not all(v.size == vectors[0].size for v in vectors):
        logger.error(
            "flattened vector sizes differ across folds: %s",
            [v.size for v in vectors],
        )
        return 1

    weight_matrix = np.stack(vectors)
    logger.info("weight matrix: %s; total params per fold: %d",
                weight_matrix.shape, weight_matrix.shape[1])

    fig = plot_checkpoint_weight_pca(
        weight_matrix,
        fold_labels=labels,
        r2_per_checkpoint=r2s,
        save_path=out_dir / "fig_checkpoint_weight_pca",
    )
    plt.close(fig)
    logger.info("rendered fig_checkpoint_weight_pca in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
