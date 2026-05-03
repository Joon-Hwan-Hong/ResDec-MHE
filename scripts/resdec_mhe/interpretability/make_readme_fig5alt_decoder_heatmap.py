#!/usr/bin/env python
"""Render Figure 5 alt-1 (SAE decoder-direction CT projection heatmap).

11 rows (1 Splatter feature + 10 random control features) x 31 columns
(canonical cell types). Each cell is the per-CT *projection* of the SAE
decoder column ``W_dec[:, j]`` against that CT's mean fused activation
``mu_c`` -- i.e.  ``proj[c, j] = mu_c . W_dec[:, j]``. This is the
quantity reported in :func:`src.analysis.sparse_autoencoder.interpret_features`
(see ``feature_report.json::top_cell_types``); we expand the per-feature
top-3 down to the full 31-CT vector by computing the projection from the
SAE state-dict and the persisted fused activations.

The Splatter feature (idx 572) is the lone SAE feature whose top-CT match
falls on the Splatter cell type under the "relaxed" CT-dominance filter;
the 10 random controls (drawn from the live, non-dead, non-ubiquitous
pool) act as the visual null. The (Splatter feature x Splatter CT) cell
is highlighted with a black box for emphasis.

Inputs
------
  - outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/sae_model.npz
      W_dec [n=64, m=2048], decoder weights of the canonical SAE.
  - outputs/canonical/sae/activations_fused_all_folds.npz
      activations [N, 31, 64], cell_types [31] -- used to compute the
      per-CT mean activation onto which decoder columns are projected.
  - outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/feature_report.json
      Cross-checked round-trip vs the per-feature top-3 entries already
      published (verified bit-equal on the in-row entries).

Outputs
-------
  - figures/fig5alt_decoder_heatmap.png  (~10 x 6 inches at 600 dpi)
  - Verification numbers printed to stdout.

Usage
-----
  PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_readme_fig5alt_decoder_heatmap.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Pin PYTHONHASHSEED defensively (matches sibling readme-fig orchestrators).
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
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


# Splatter SAE feature (the one the relaxed filter picked up) + 10
# random control features per the README revamp brief.
SPLATTER_FEATURE_IDX: int = 572
RANDOM_FEATURE_INDICES: tuple[int, ...] = (
    178, 1577, 183, 1340, 898, 883, 1431, 194, 415, 1750,
)


def _load_decoder_and_means(
    sae_npz: Path,
    activations_npz: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (W_dec [64, 2048], per_ct_means [31, 64], cell_types [31]).

    Per-CT means are computed across all (subject, fold) rows -- this is
    the same aggregation used in
    :func:`src.analysis.sparse_autoencoder.interpret_features` lines
    1175-1178 (``per_ct_means = activations.mean(axis=0)``).
    """
    sae = np.load(sae_npz, allow_pickle=True)
    if "W_dec" not in sae.files:
        raise KeyError(f"W_dec missing from {sae_npz!s}")
    W_dec = np.asarray(sae["W_dec"], dtype=np.float32)  # [64, 2048]

    acts = np.load(activations_npz, allow_pickle=True)
    if "activations" not in acts.files or "cell_types" not in acts.files:
        raise KeyError(
            f"activations/cell_types missing from {activations_npz!s}; "
            f"keys={list(acts.files)}"
        )
    activations = np.asarray(acts["activations"], dtype=np.float32)  # [N, 31, 64]
    if activations.ndim != 3 or activations.shape[1:] != (31, 64):
        raise ValueError(
            f"activations expected shape (N, 31, 64); got {activations.shape}"
        )
    if W_dec.shape != (activations.shape[2], W_dec.shape[1]):
        raise ValueError(
            f"W_dec/activations dim mismatch: W_dec.shape={W_dec.shape}, "
            f"activations.shape[-1]={activations.shape[2]}"
        )
    per_ct_means = activations.mean(axis=0).astype(np.float32)  # [31, 64]
    cell_types = [str(c) for c in list(acts["cell_types"])]
    if len(cell_types) != activations.shape[1]:
        raise ValueError(
            f"cell_types length {len(cell_types)} != activations CT axis "
            f"{activations.shape[1]}"
        )
    return W_dec, per_ct_means, cell_types


def _build_proj_matrix(
    W_dec: np.ndarray,
    per_ct_means: np.ndarray,
    feature_indices: list[int],
) -> np.ndarray:
    """For each requested feature j, compute mu @ W_dec[:, j] -> [11, 31].

    Output rows follow the order of ``feature_indices`` exactly so the
    caller controls Splatter-on-top placement; columns follow
    ``per_ct_means`` row order (i.e. the canonical 31-CT order).
    """
    cols = W_dec[:, feature_indices]  # [64, n_features]
    proj = per_ct_means @ cols  # [31, n_features]
    return proj.T  # -> [n_features, 31]


def _verify_top3_against_report(
    proj_row: np.ndarray,
    cell_types: list[str],
    expected_top3: list[dict],
    feature_idx: int,
) -> None:
    """Round-trip the projection against feature_report.json's top-3.

    Raises ``AssertionError`` on any mismatch (proj or sq differing by more
    than 1e-3 absolute) so the figure-builder fails loudly if the SAE
    state-dict and the persisted feature report drift apart.
    """
    sq = proj_row ** 2
    sorted_idx = np.argsort(-sq)
    for rank, item in enumerate(expected_top3):
        ct_name = item["cell_type"]
        if ct_name not in cell_types:
            raise AssertionError(
                f"feature {feature_idx}: report CT {ct_name!r} not in canonical 31"
            )
        ct_idx = cell_types.index(ct_name)
        # The argsort tie-break may shuffle within sq-equal rows but we
        # never expect ties at fp32 here -- so demand exact rank match.
        if sorted_idx[rank] != ct_idx:
            raise AssertionError(
                f"feature {feature_idx} top-{rank+1}: report says {ct_name!r} "
                f"(idx {ct_idx}), local says {cell_types[sorted_idx[rank]]!r} "
                f"(idx {sorted_idx[rank]})"
            )
        if abs(float(proj_row[ct_idx]) - float(item["projection"])) > 1e-3:
            raise AssertionError(
                f"feature {feature_idx} CT {ct_name!r}: proj mismatch "
                f"local={proj_row[ct_idx]:.6f} vs report={item['projection']:.6f}"
            )


def _draw_heatmap(
    ax: plt.Axes,
    proj: np.ndarray,
    feature_labels: list[str],
    cell_types: list[str],
    splatter_ct_idx: int,
) -> None:
    """Draw the (11 x 31) signed projection heatmap with PiYG diverging cmap.

    The colormap is centered at zero (decoder weights / projections are
    signed); ``vmax`` is set to the symmetric max-abs of the matrix so the
    color bar is balanced. The (Splatter feature, Splatter CT) cell is
    annotated with a 1.5-pt black box to flag the relaxed-filter pick.
    """
    cmap = PALETTES["diverging"]  # PiYG -- magenta(-) / green(+)
    vmax = float(np.max(np.abs(proj)))
    vmin = -vmax

    im = ax.imshow(
        proj,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    n_rows, n_cols = proj.shape

    # Axis labels.
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(feature_labels, fontsize=7)

    # White gridlines between cells (minor ticks at half-integer positions).
    ax.set_xticks(np.arange(-0.5, n_cols, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Caption-only style, all spines on (heatmap data-area frame), grids off
    # (handled manually via minor ticks above).
    fmt_axes(ax, hide_spines=(), grid_major=False, grid_minor=False)

    # Highlight the (Splatter feature row=0, Splatter CT column) cell with
    # a black square. Splatter feature is row 0 by construction.
    rect = mpatches.Rectangle(
        (splatter_ct_idx - 0.5, 0 - 0.5),
        1.0, 1.0,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(rect)

    # Compact colorbar to the right.
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.020, pad=0.015)
    cb.set_label(r"$\mu_c \cdot W_\mathrm{dec}[:,j]$  (signed)", fontsize=7)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0, labelsize=6)

    ax.set_xlabel("Cell type (canonical 31-CT order)", fontsize=8)
    ax.set_ylabel("SAE feature", fontsize=8)


def make_figure(
    proj: np.ndarray,
    feature_labels: list[str],
    cell_types: list[str],
    splatter_ct_idx: int,
) -> plt.Figure:
    """Build the 10 x 6 in figure with the single-panel heatmap."""
    apply_theme("paper")
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_heatmap(ax, proj, feature_labels, cell_types, splatter_ct_idx)
    fig.subplots_adjust(left=0.20, right=0.94, top=0.97, bottom=0.30)
    return fig


def _print_report(
    proj: np.ndarray,
    feature_indices: list[int],
    cell_types: list[str],
    decoder_method: str,
) -> None:
    """Print verification numbers for the operator."""
    print("=" * 72)
    print("README Figure 5 alt-1 -- SAE decoder-direction CT heatmap")
    print("=" * 72)
    print(f"  decoder_access_method : {decoder_method}")
    print(f"  feature_indices       : {feature_indices}")
    splatter_idx = cell_types.index("Splatter")

    # Splatter feature top-3.
    splatter_row = proj[0]
    sq = splatter_row ** 2
    top3 = np.argsort(-sq)[:3]
    print(f"  --- Splatter feature {feature_indices[0]} (top-3 CT, projection):")
    for r in top3:
        print(f"      {cell_types[r]:38s}  proj={splatter_row[r]:+.4f}  sq={sq[r]:.4f}")

    # Each random feature's top CT.
    print(f"  --- Random control features (top CT, projection):")
    for k, fi in enumerate(feature_indices[1:], start=1):
        row = proj[k]
        sq_row = row ** 2
        top1 = int(np.argmax(sq_row))
        print(f"      feat {fi:5d}  {cell_types[top1]:38s}  proj={row[top1]:+.4f}  "
              f"sq={sq_row[top1]:.4f}")

    print(f"  splatter_ct_idx       : {splatter_idx}  (column highlighted)")
    print(f"  proj.shape            : {proj.shape}  (rows=features, cols=CTs)")
    print(f"  proj.minmax           : [{proj.min():+.4f}, {proj.max():+.4f}]")
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument(
        "--sae-npz", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/sae_model.npz",
    )
    parser.add_argument(
        "--activations-npz", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/sae/activations_fused_all_folds.npz",
    )
    parser.add_argument(
        "--feature-report-json", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/feature_report.json",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig5alt_decoder_heatmap",
        help="Output path stem (no extension); save_fig appends .png.",
    )
    args = parser.parse_args()

    # Build the (n_features, 31) projection matrix from the SAE state-dict
    # + persisted fused activations.
    feature_indices: list[int] = [SPLATTER_FEATURE_IDX, *RANDOM_FEATURE_INDICES]

    logger.info("[fig5alt] loading SAE state-dict + per-CT means")
    W_dec, per_ct_means, cell_types = _load_decoder_and_means(
        args.sae_npz, args.activations_npz,
    )
    logger.info(
        "[fig5alt] W_dec.shape=%s, per_ct_means.shape=%s",
        W_dec.shape, per_ct_means.shape,
    )

    proj = _build_proj_matrix(W_dec, per_ct_means, feature_indices)  # [11, 31]
    logger.info("[fig5alt] proj.shape=%s", proj.shape)

    # Round-trip the locally-computed projection against the JSON top-3 to
    # guarantee the SAE state-dict and the published feature report are
    # consistent (this is *not* the fallback path; the figure uses the full
    # 31-CT vector regardless of the report's top-3).
    feature_report = json.loads(Path(args.feature_report_json).read_text())
    for k, fi in enumerate(feature_indices):
        report_entry = feature_report[fi]
        if int(report_entry["feature_idx"]) != fi:
            raise AssertionError(
                f"feature_report[{fi}] feature_idx mismatch: "
                f"{report_entry['feature_idx']}"
            )
        _verify_top3_against_report(
            proj[k], cell_types, report_entry["top_cell_types"], fi,
        )
    logger.info("[fig5alt] round-trip verified vs feature_report top-3 (all pass)")

    # Y-axis labels: "<idx> (Splatter top-CT)" for row 0, "<idx> (random)" for the rest.
    feature_labels: list[str] = [
        f"{feature_indices[0]} (Splatter top-CT)",
        *[f"{fi} (random)" for fi in feature_indices[1:]],
    ]

    splatter_ct_idx = cell_types.index("Splatter")

    fig = make_figure(proj, feature_labels, cell_types, splatter_ct_idx)

    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig5alt] removing preexisting %s", out_png)
        out_png.unlink()

    written = save_fig(fig, args.out_stem, formats=("png",))
    plt.close(fig)
    for w in written:
        logger.info("[fig5alt] wrote %s (size=%d B)", w, w.stat().st_size)

    _print_report(
        proj, feature_indices, cell_types,
        decoder_method="full SAE state_dict (W_dec[64,2048] x mu_c[31,64])",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
