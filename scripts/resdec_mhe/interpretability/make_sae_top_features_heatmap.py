"""Figure D1 (lab-meeting supplementary): SAE top-N feature × 31-CT heatmap.

A 2D heatmap that PRESERVES BOTH AXES (no 1D collapse) so the audience can
SEE the per-feature × per-CT distribution texture: most features spread mass
across multiple CTs (the §31.11 distributed-representation finding).

Rows
    Top-N relaxed-interpretable SAE features (default N=50), sorted by
    interpretability score (mw_p_cognition ascending → most-significant first).

Columns
    31 cell types in canonical CELL_TYPE_ORDER.

Cell value
    Per-feature × per-CT contribution mass, computed faithfully against
    ``src.analysis.sparse_autoencoder.interpret_features``:

        per_ct_means[c, :]    = activations[:, c, :].mean(axis=0)   # [n]
        proj[j, c]            = per_ct_means[c, :] @ W_dec[:, j]
        sq[j, c]              = proj[j, c] ** 2
        mass[j, c]            = sq[j, c] / sq[j, :].sum()           # row-normalized

    Mass is the same quantity behind the JSON ``squared_projection`` entries
    in ``feature_report.json``; we just need it across ALL 31 CTs (the JSON
    only stores the top-3 per feature).

The Splatter-dominant feature (the lone 1/323 = 0.31% Splatter-top feature
flagged in §31.11) is highlighted with a red row border and a star marker.

Inputs
------
* ``feature_report.json`` (list of 2048 dicts) — used for the relaxed filter
  and the interpretability score per feature.
* ``sae_model.npz`` — provides ``W_dec`` of shape ``[n=64, m=2048]``.
* ``activations_fused_all_folds.npz`` — provides ``activations`` of shape
  ``[N, C=31, n=64]`` and ``cell_types``.

Outputs
-------
* ``fig_sae_top_features_heatmap.{png,pdf}``
* ``fig_sae_top_features_heatmap_data.json`` (chosen top-N feature indices,
  Splatter-dominant row index, value field used)

CLI
---
``--top-n``               number of features to plot (default 50)
``--feature-report``      path to feature_report.json
``--sae-model``           path to sae_model.npz
``--activations``         path to activations_fused_all_folds.npz
``--out-dir``             output directory
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Defaults (env-var / argparse driven per project rules)
# ===========================================================================

_DEFAULT_FEATURE_REPORT = os.environ.get(
    "SAE_SUPP_FEATURE_REPORT",
    str(_WORKTREE_ROOT
        / "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0"
        / "feature_report.json"),
)
_DEFAULT_SAE_MODEL = os.environ.get(
    "SAE_SUPP_SAE_MODEL",
    str(_WORKTREE_ROOT
        / "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0"
        / "sae_model.npz"),
)
_DEFAULT_ACTIVATIONS = os.environ.get(
    "SAE_SUPP_ACTIVATIONS",
    str(_WORKTREE_ROOT / "outputs/canonical/sae/activations_fused_all_folds.npz"),
)
_DEFAULT_OUT_DIR = os.environ.get(
    "SAE_SUPP_OUT_DIR",
    str(_WORKTREE_ROOT / "outputs/canonical/interpretability/figures/sae_supp"),
)
_DEFAULT_TOP_N = int(os.environ.get("SAE_SUPP_TOP_N", "50"))

# Constants matching src/analysis/sparse_autoencoder.py.
DEAD_FRACTION_THRESHOLD = 1e-4
RELAXED_FRAC_BAND = (DEAD_FRACTION_THRESHOLD, 0.5)
RELAXED_MW_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data loading + per-feature × per-CT mass computation
# ---------------------------------------------------------------------------


def load_feature_report(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def select_relaxed_features(
    feature_report: list[dict],
    top_n: int,
) -> tuple[list[int], list[dict]]:
    """Pick top-N relaxed-interpretable features sorted by mw_p_cognition asc.

    Relaxed criterion (matches feature_xref_consensus.json):
        - non-dead
        - mw_p_cognition < 0.05
        - fraction_active in [DEAD_FRACTION_THRESHOLD, 0.5]
    """
    relaxed = [
        f for f in feature_report
        if "dead" not in f.get("flags", [])
        and f.get("mw_p_cognition") is not None
        and f["mw_p_cognition"] < RELAXED_MW_THRESHOLD
        and RELAXED_FRAC_BAND[0] <= f.get("fraction_active", 0.0) <= RELAXED_FRAC_BAND[1]
    ]
    # Sort by mw_p_cognition ascending — most cognition-significant first
    relaxed_sorted = sorted(relaxed, key=lambda f: f["mw_p_cognition"])
    chosen = relaxed_sorted[:top_n]
    return [int(f["feature_idx"]) for f in chosen], chosen


def compute_per_feature_per_ct_mass(
    sae_model_path: Path,
    activations_path: Path,
    feature_indices: Sequence[int],
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Compute the per-feature × per-CT contribution mass matrix.

    Mass is derived faithfully from ``interpret_features`` in
    ``src/analysis/sparse_autoencoder.py``:

        proj  = per_ct_means @ W_dec[:, j]    # [C]
        sq    = proj ** 2                       # squared_projection
        mass  = sq / sq.sum()                   # row-normalized when normalize=True

    Parameters
    ----------
    sae_model_path
        Path to sae_model.npz with W_dec [n, m].
    activations_path
        Path to activations_fused_all_folds.npz with activations [N, C, n]
        and cell_types [C].
    feature_indices
        Sequence of feature indices j to include as rows.
    normalize
        If True, each row sums to 1.

    Returns
    -------
    mass : np.ndarray, shape (len(feature_indices), C)
    ct_order : list[str]
    feature_indices_out : list[int]
    """
    sae = np.load(sae_model_path)
    W_dec = np.asarray(sae["W_dec"], dtype=np.float64)  # [n, m]

    acts = np.load(activations_path, allow_pickle=True)
    A = np.asarray(acts["activations"], dtype=np.float64)  # [N, C, n]
    cell_types = list(map(str, acts["cell_types"]))
    per_ct_means = A.mean(axis=0)  # [C, n]

    feat_indices_arr = np.asarray(list(feature_indices), dtype=np.int64)
    sub_dec = W_dec[:, feat_indices_arr]  # [n, K]

    proj = per_ct_means @ sub_dec  # [C, K]
    sq = proj ** 2                  # [C, K]
    mass_t = sq.T                   # [K, C]
    if normalize:
        denom = mass_t.sum(axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1.0)
        mass_t = mass_t / denom
    return mass_t.astype(np.float32), cell_types, list(map(int, feat_indices_arr))


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def _find_splatter_dominant_row(
    mass: np.ndarray,
    ct_order: list[str],
) -> int | None:
    """Return the row index of the feature whose top-1 CT is Splatter, or None.

    If multiple features have Splatter as top-1, return the one with the
    highest Splatter mass.
    """
    if "Splatter" not in ct_order:
        return None
    splatter_col = ct_order.index("Splatter")
    top1 = np.argmax(mass, axis=1)
    matches = np.where(top1 == splatter_col)[0]
    if len(matches) == 0:
        return None
    # Pick the feature with the highest Splatter mass
    best = matches[np.argmax(mass[matches, splatter_col])]
    return int(best)


def _draw_heatmap(
    ax: plt.Axes,
    mass: np.ndarray,
    ct_order: list[str],
    feat_indices: list[int],
    *,
    splatter_row: int | None,
) -> object:
    """Render the per-feature × per-CT heatmap with Splatter row highlighted."""
    cmap = PALETTES["sequential"]  # viridis
    im = ax.imshow(mass, aspect="auto", cmap=cmap, interpolation="nearest")

    # Y-axis: feature indices (truncate to short labels for readability)
    ax.set_yticks(np.arange(len(feat_indices)))
    ax.set_yticklabels([f"feat {j}" for j in feat_indices], fontsize=5)
    ax.set_ylabel("SAE feature (sorted by mw_p_cognition asc)")

    # X-axis: 31 CT names rotated
    ax.set_xticks(np.arange(len(ct_order)))
    ax.set_xticklabels(ct_order, rotation=70, ha="right", fontsize=5.5)
    ax.set_xlabel("Cell type")

    # Highlight Splatter-dominant row with red border
    if splatter_row is not None:
        rect = mpatches.Rectangle(
            (-0.5, splatter_row - 0.5),
            len(ct_order), 1,
            linewidth=1.6, edgecolor="#d62728", facecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)
        # Place a red star to the LEFT of the heatmap, in axes-fraction
        ax.annotate(
            "*",
            xy=(-0.5, splatter_row),
            xytext=(-2.5, splatter_row),
            fontsize=12, color="#d62728", fontweight="bold",
            ha="right", va="center",
        )

    return im


def build_figure(
    feature_report_path: Path,
    sae_model_path: Path,
    activations_path: Path,
    out_dir: Path,
    *,
    top_n: int = _DEFAULT_TOP_N,
) -> dict:
    """Render the D1 heatmap. Returns metadata dict (also written as JSON)."""
    apply_theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_report = load_feature_report(feature_report_path)
    feat_indices, chosen = select_relaxed_features(feature_report, top_n=top_n)
    if not feat_indices:
        raise RuntimeError(
            "No relaxed-interpretable features found in feature_report; "
            f"check {feature_report_path}."
        )

    mass, ct_order, feat_indices_out = compute_per_feature_per_ct_mass(
        sae_model_path=sae_model_path,
        activations_path=activations_path,
        feature_indices=feat_indices,
        normalize=True,
    )

    splatter_row = _find_splatter_dominant_row(mass, ct_order)

    fig, ax = plt.subplots(figsize=(12.0, 14.0))
    im = _draw_heatmap(
        ax, mass, ct_order, feat_indices_out, splatter_row=splatter_row,
    )
    fmt_axes(ax, hide_spines=("top", "right"), grid_major=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Per-CT contribution mass (row-normalized squared projection)",
                   fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Title (Splatter row callout)
    if splatter_row is not None:
        feat_idx = feat_indices_out[splatter_row]
        title = (
            f"SAE top-{len(feat_indices_out)} interpretable features × 31 CTs "
            f"(distributed mass)\n"
            f"red border = lone Splatter-top feature "
            f"(feature_idx {feat_idx}, row {splatter_row})"
        )
    else:
        title = (
            f"SAE top-{len(feat_indices_out)} interpretable features × 31 CTs "
            f"(distributed mass)\n"
            f"no Splatter-top feature in this slice"
        )
    fig.suptitle(title, fontsize=10, y=0.99)

    fig.subplots_adjust(top=0.94, bottom=0.18, left=0.10, right=0.95)

    save_fig(fig, out_dir / "fig_sae_top_features_heatmap")
    plt.close(fig)

    # Metadata sidecar
    meta = {
        "top_n": int(len(feat_indices_out)),
        "feature_indices": feat_indices_out,
        "cell_type_order": ct_order,
        "splatter_dominant_row": splatter_row,
        "splatter_dominant_feature_idx": (
            feat_indices_out[splatter_row] if splatter_row is not None else None
        ),
        "value_field": "row_normalized_squared_projection",
        "row_sort_key": "mw_p_cognition_ascending",
        "filter": "relaxed (non-dead, mw_p_cog<0.05, fraction_active in [1e-4, 0.5])",
        "deviation_note": (
            "User spec said 'mark with red border or star'; we did both — "
            "red border around the Splatter-top row + a red asterisk to its "
            "left. The 'lone 1/323 Splatter-dominant feature' framing assumes "
            "the relaxed-set top-N still contains it; if top_n is small enough "
            "to exclude it, splatter_dominant_row will be None and the title "
            "is updated accordingly."
        ),
    }
    (out_dir / "fig_sae_top_features_heatmap_data.json").write_text(
        json.dumps(meta, indent=2)
    )
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--feature-report", default=_DEFAULT_FEATURE_REPORT)
    p.add_argument("--sae-model", default=_DEFAULT_SAE_MODEL)
    p.add_argument("--activations", default=_DEFAULT_ACTIVATIONS)
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    p.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N,
                   help="Number of relaxed-interpretable features to plot")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    t0 = time.perf_counter()
    meta = build_figure(
        feature_report_path=Path(args.feature_report),
        sae_model_path=Path(args.sae_model),
        activations_path=Path(args.activations),
        out_dir=Path(args.out_dir),
        top_n=args.top_n,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Rendered fig_sae_top_features_heatmap.{png,pdf} in %.2fs "
        "(N=%d features, splatter_row=%s)",
        elapsed, meta["top_n"], meta["splatter_dominant_row"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
