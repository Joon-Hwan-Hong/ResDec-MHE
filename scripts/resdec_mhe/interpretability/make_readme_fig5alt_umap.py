#!/usr/bin/env python
"""Render Figure 5 alt-2 (SAE feature UMAP point cloud) for the README revamp.

Single-panel 2D UMAP projection of the SAE decoder columns for the 323
SAE features that pass the *relaxed* interpretability filter (same filter
as :file:`make_readme_fig5_sae_main.py`). Each point is one feature; the
position is the UMAP embedding of that feature's decoder vector
``W_dec[:, j]`` (shape ``[n=64]``); the color is the deterministic CT
color of the feature's top-CT identity (``top_cell_types[0]``); a faint
hexbin density layer underneath flags where features cluster in the 2D
embedding.

Highlighted markers
-------------------
- Splatter feature (idx 572) — the lone Splatter-top-CT feature in the
  relaxed pool — drawn as a red star with a black outline.
- 10 random control features that the EXP-042 causal-patching run used
  ([178, 1577, 183, 1340, 898, 883, 1431, 194, 415, 1750]) — drawn as
  grey diamonds with a black outline. Only 2 of those 10 (1340 and 883)
  are themselves in the relaxed-filter pool; for symmetry we still
  project ALL 10 onto the same UMAP via ``UMAP.transform`` so the
  control set is visualizable in context. The figure caption explains
  that these are the same controls used by the causal-patching null and
  are NOT part of the 323-point cloud.

Approach (decoder-vector UMAP, primary path)
--------------------------------------------
The SAE decoder weight matrix is ``W_dec`` with shape ``[n=64, m=2048]``
(input dim 64, hidden dim 2048). Per-feature decoder direction is
``W_dec[:, j]`` (shape ``[64]``). We assemble:

  - ``X_relaxed`` of shape ``(323, 64)`` — relaxed-filter feature decoder
    vectors, in canonical feature_idx ascending order.
  - ``X_random`` of shape ``(10, 64)`` — random control decoder vectors,
    in the deterministic order from
    ``RANDOM_FEATURE_INDICES``.

UMAP is fit on ``X_relaxed`` (323 x 64) with ``n_components=2,
n_neighbors=15, min_dist=0.1, random_state=42``. The 10 random control
vectors are then projected via ``reducer.transform(X_random)`` so they
share the same embedding manifold.

If the SAE checkpoint cannot be loaded (missing W_dec or shape
mismatch), the script falls back to the 31-dim ``top_cell_types``
representation: each feature's vector is a 31-d vector with the three
top-CT projections placed at the canonical CT indices and 0 elsewhere.
The fallback is markedly less informative (only 3 non-zeros per row)
but matches the spec's secondary path. The active code path is logged
to stdout under ``approach`` so the operator can confirm.

Inputs
------
  - ``outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/feature_report.json``
    Full per-feature metadata. Filtered down to the 323-feature pool.
  - ``outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0/sae_model.npz``
    SAE state-dict; ``W_dec`` is the per-feature decoder weight column.

Outputs
-------
  - ``figures/fig5alt_umap.png`` (10 x 8 in at 300 dpi)
  - Verification numbers printed to stdout.

Usage
-----
  PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_readme_fig5alt_umap.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Pin hash seed defensively for matplotlib/UMAP color-path determinism.
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.config import CELL_TYPE_COLORS  # noqa: E402
from src.visualization.theme import (  # noqa: E402
    apply_theme,
    fmt_axes,
    save_fig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — match sibling readme-fig orchestrators verbatim
# ---------------------------------------------------------------------------

# Lone Splatter-top-CT feature in the relaxed-filter pool (canonical).
SPLATTER_FEATURE_IDX: int = 572

# 10 random control features used by the EXP-042 causal-patching run.
# Order is preserved for deterministic plotting / printing.
RANDOM_FEATURE_INDICES: tuple[int, ...] = (
    178, 1577, 183, 1340, 898, 883, 1431, 194, 415, 1750,
)

# Splatter highlight: tab10 red, with a thick black outline so the single
# point reads above the 322 background spokes irrespective of CT color.
SPLATTER_HIGHLIGHT_COLOR: str = "#D62728"

# Random-control marker fill: neutral grey so it is clearly *not* a CT
# identity color — matches the panel B raincloud control color.
RANDOM_CONTROL_COLOR: str = "#7F7F7F"

# UMAP hyperparameters (per spec).
UMAP_N_COMPONENTS: int = 2
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_RANDOM_STATE: int = 42

# Figure dimensions (per spec: ~10 x 8 in single panel).
FIG_W_IN: float = 10.0
FIG_H_IN: float = 8.0


# ---------------------------------------------------------------------------
# Filter helpers (identical to make_readme_fig5_sae_main._passes_relaxed_filter)
# ---------------------------------------------------------------------------


def _passes_relaxed_filter(feat: dict) -> bool:
    """Return True if a feature_report entry passes the relaxed filter.

    Relaxed filter (per ``feature_xref_consensus.json::filter_definitions``):

      * non-dead (``"dead"`` not in ``flags``)
      * ``mw_p_cognition < 0.05``
      * ``fraction_active`` in [0.0001, 0.5]
      * ``ct_dominance <= 0.7``
    """
    flags = feat.get("flags") or []
    if "dead" in flags:
        return False
    p = feat.get("mw_p_cognition")
    if p is None or not (p < 0.05):
        return False
    fa = feat.get("fraction_active")
    if fa is None or not (0.0001 <= fa <= 0.5):
        return False
    dom = feat.get("ct_dominance")
    if dom is None or not (dom <= 0.7):
        return False
    return True


def _max_ct_identity(feat: dict) -> str | None:
    """Return ``top_cell_types[0].cell_type``, or None if no top CT recorded."""
    tcts = feat.get("top_cell_types") or []
    if not tcts:
        return None
    return tcts[0].get("cell_type")


def _load_relaxed_features(report_path: Path) -> list[dict]:
    """Load + relax-filter the SAE feature report.

    Returns the list of relaxed-filter feature dicts in feature_idx ascending
    order so downstream array assembly is deterministic and matches the
    SAE checkpoint's W_dec column order.
    """
    payload = json.loads(report_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected list at {report_path}; got {type(payload).__name__}"
        )
    filtered = [feat for feat in payload if _passes_relaxed_filter(feat)]
    filtered.sort(key=lambda feat: int(feat["feature_idx"]))
    return filtered


def _load_full_report_by_idx(report_path: Path) -> dict[int, dict]:
    """Return the full ``feature_report.json`` indexed by ``feature_idx``.

    Used to fetch decoder rows for the random control features (which may
    not be in the relaxed-filter pool — only 2 of 10 are by construction).
    The map is built once and reused for both vectors and label lookup.
    """
    payload = json.loads(report_path.read_text())
    return {int(feat["feature_idx"]): feat for feat in payload}


# ---------------------------------------------------------------------------
# Decoder-vector loading (primary path) + top_cell_types fallback
# ---------------------------------------------------------------------------


def _load_decoder_columns(
    sae_npz: Path,
    feature_indices: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Return ``W_dec[:, feature_indices].T`` of shape ``(len(indices), 64)``.

    The SAE checkpoint stores ``W_dec`` as ``[n=64, m=2048]`` (per
    :class:`src.analysis.sparse_autoencoder.SAEModel`); each *column* is a
    feature's decoder direction in input space. We transpose to
    ``(n_features, 64)`` so each row is a feature vector ready for UMAP.

    Raises
    ------
    KeyError
        ``W_dec`` missing from the checkpoint.
    ValueError
        Unexpected shape / out-of-range feature index.
    """
    sae = np.load(sae_npz, allow_pickle=True)
    if "W_dec" not in sae.files:
        raise KeyError(f"W_dec missing from {sae_npz!s}")
    W_dec = np.asarray(sae["W_dec"], dtype=np.float32)
    if W_dec.ndim != 2:
        raise ValueError(f"W_dec expected 2-D; got shape {W_dec.shape}")
    n_input, n_hidden = W_dec.shape
    indices = np.asarray(feature_indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("feature_indices is empty")
    if indices.max() >= n_hidden or indices.min() < 0:
        raise ValueError(
            f"feature_indices out of range [0, {n_hidden}); "
            f"got max={int(indices.max())}, min={int(indices.min())}"
        )
    cols = W_dec[:, indices]  # (n_input, n_features_requested)
    return cols.T.astype(np.float32)  # (n_features_requested, n_input)


def _build_top_ct_fallback_vectors(
    feats: list[dict],
    cell_type_order: list[str],
) -> np.ndarray:
    """Fallback vectors built from ``top_cell_types`` (3 non-zero per row).

    Each row is a ``(31,)`` vector with the feature's top-3 ``projection``
    values placed at the canonical CT index for those CTs and 0
    everywhere else. Only used when the SAE checkpoint cannot be loaded.

    Returns
    -------
    np.ndarray of shape ``(len(feats), 31)``.
    """
    n = len(feats)
    m = len(cell_type_order)
    if m != 31:
        raise ValueError(f"Expected 31 canonical CTs; got {m}")
    ct_to_idx = {ct: i for i, ct in enumerate(cell_type_order)}
    vecs = np.zeros((n, m), dtype=np.float32)
    for i, feat in enumerate(feats):
        for entry in feat.get("top_cell_types", []) or []:
            ct = entry.get("cell_type")
            if ct in ct_to_idx:
                vecs[i, ct_to_idx[ct]] = float(entry.get("projection", 0.0))
    return vecs


def _canonical_cell_type_order_from_feats(feats: list[dict]) -> list[str]:
    """Recover the canonical 31-CT order from the union of top_cell_types.

    Used only on the fallback path. We rely on the published
    :data:`src.visualization.config.CELL_TYPE_COLORS` to provide a
    deterministic 31-CT ordering when the SAE state-dict isn't available.
    """
    return list(CELL_TYPE_COLORS.keys())


# ---------------------------------------------------------------------------
# UMAP fit + transform
# ---------------------------------------------------------------------------


def _umap_fit_and_transform(
    X_fit: np.ndarray,
    X_extra: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Fit UMAP on ``X_fit`` and optionally embed ``X_extra``.

    Imported lazily so that callers without ``umap-learn`` can still do
    smoke-test imports of this module (the dependency error is reported
    with a clear actionable message).

    Returns
    -------
    embedding_fit
        ``(n_fit, 2)`` UMAP embedding of the fit set.
    embedding_extra
        ``(n_extra, 2)`` embedding of the additional vectors via
        ``UMAP.transform``, or ``None`` if ``X_extra`` is None.
    meta
        Dict with the UMAP hyperparameters that were used.
    """
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "umap-learn is required to render this figure. Install via "
            "`uv pip install umap-learn` from the worktree root."
        ) from exc

    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    embedding_fit = reducer.fit_transform(X_fit)
    embedding_extra = None
    if X_extra is not None and len(X_extra) > 0:
        embedding_extra = reducer.transform(X_extra)
    meta = {
        "n_components": UMAP_N_COMPONENTS,
        "n_neighbors": UMAP_N_NEIGHBORS,
        "min_dist": UMAP_MIN_DIST,
        "random_state": UMAP_RANDOM_STATE,
    }
    return embedding_fit, embedding_extra, meta


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def _draw_panel(
    ax: plt.Axes,
    embedding_fit: np.ndarray,
    feature_colors: list[str],
    splatter_xy: tuple[float, float],
    splatter_label: str,
    random_xy: np.ndarray,
    random_labels: list[str],
    *,
    n_relaxed: int,
    n_random: int,
) -> None:
    """Compose the hexbin density + scatter + highlights on a single axes.

    Layer order (bottom → top):
      1. ``hexbin`` density (Greys, alpha=0.4) — visual context only,
         emphasizes where the 323 features cluster.
      2. ``scatter`` of all 323 features colored by max-CT identity.
      3. Random-control diamonds (overlaid via UMAP transform).
      4. Splatter star (one of the 323 features, drawn last so it sits
         above any neighboring marker overlap).
      5. Text annotations next to highlighted markers.
    """
    # Layer 1 — density backdrop.
    ax.hexbin(
        embedding_fit[:, 0], embedding_fit[:, 1],
        gridsize=20, cmap="Greys", alpha=0.4, mincnt=1,
        linewidths=0.0, zorder=1,
    )

    # Layer 2 — main scatter.
    ax.scatter(
        embedding_fit[:, 0], embedding_fit[:, 1],
        c=feature_colors, s=22, alpha=0.85,
        edgecolor="black", linewidths=0.25,
        zorder=2,
    )

    # Layer 3 — random-control diamonds.
    if random_xy.size:
        ax.scatter(
            random_xy[:, 0], random_xy[:, 1],
            marker="D", s=80,
            color=RANDOM_CONTROL_COLOR,
            edgecolor="black", linewidths=0.9,
            label=f"Random controls (n={n_random})",
            zorder=4,
        )
        # Compact numeric labels next to each diamond.
        for (xx, yy), lbl in zip(random_xy, random_labels):
            ax.annotate(
                lbl, xy=(xx, yy), xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.0, color="black",
                zorder=6,
            )

    # Layer 4 — Splatter star.
    sx, sy = splatter_xy
    ax.scatter(
        [sx], [sy],
        marker="*", s=320,
        color=SPLATTER_HIGHLIGHT_COLOR,
        edgecolor="black", linewidths=1.0,
        label=f"Splatter feature (idx {SPLATTER_FEATURE_IDX})",
        zorder=5,
    )
    ax.annotate(
        splatter_label, xy=(sx, sy), xytext=(7, 7),
        textcoords="offset points",
        fontsize=7.5, color="black", fontweight="bold",
        zorder=7,
    )

    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    fmt_axes(ax, hide_spines=("top", "right"), grid_major=True)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="*", color=SPLATTER_HIGHLIGHT_COLOR,
            markersize=14, markeredgecolor="black",
            markeredgewidth=1.0, linewidth=0,
            label=f"Splatter feature (idx {SPLATTER_FEATURE_IDX})",
        ),
        Line2D(
            [0], [0],
            marker="D", color=RANDOM_CONTROL_COLOR,
            markersize=8, markeredgecolor="black",
            markeredgewidth=0.9, linewidth=0,
            label=f"Random controls (n={n_random}, EXP-042)",
        ),
        Line2D(
            [0], [0],
            marker="o", color="lightgray",
            markersize=7, markeredgecolor="black",
            markeredgewidth=0.25, linewidth=0,
            label=f"Relaxed-filter features (n={n_relaxed}, color = top-CT)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=7.5,
        frameon=True,
    )


def make_figure(
    embedding_fit: np.ndarray,
    feature_colors: list[str],
    splatter_xy: tuple[float, float],
    splatter_label: str,
    random_xy: np.ndarray,
    random_labels: list[str],
    n_relaxed: int,
    n_random: int,
) -> plt.Figure:
    """Build the 10 x 8 in figure with the single-panel UMAP."""
    apply_theme("paper")
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    _draw_panel(
        ax,
        embedding_fit=embedding_fit,
        feature_colors=feature_colors,
        splatter_xy=splatter_xy,
        splatter_label=splatter_label,
        random_xy=random_xy,
        random_labels=random_labels,
        n_relaxed=n_relaxed,
        n_random=n_random,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08)
    return fig


# ---------------------------------------------------------------------------
# Verification reporter
# ---------------------------------------------------------------------------


def _print_report(
    *,
    n_relaxed: int,
    embedding_shape: tuple[int, ...],
    splatter_xy: tuple[float, float],
    approach: str,
    umap_meta: dict,
    out_paths: list[Path],
) -> None:
    """Print verification numbers required by the brief."""
    print("=" * 72)
    print("README Figure 5 alt-2 -- SAE feature UMAP point cloud")
    print("=" * 72)
    print(f"  approach                : {approach}")
    print(f"  n_features (relaxed)    : {n_relaxed}")
    print(f"  embedding_shape         : {embedding_shape}")
    print(f"  splatter_idx            : {SPLATTER_FEATURE_IDX}")
    print(
        "  splatter_umap_xy        : "
        f"({splatter_xy[0]:+.6f}, {splatter_xy[1]:+.6f})"
    )
    print(f"  umap.n_components       : {umap_meta['n_components']}")
    print(f"  umap.n_neighbors        : {umap_meta['n_neighbors']}")
    print(f"  umap.min_dist           : {umap_meta['min_dist']}")
    print(f"  umap.random_state       : {umap_meta['random_state']}")
    for path in out_paths:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  Wrote: {path}  ({size_mb:.3f} MB)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--feature-report", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0"
            / "feature_report.json"
        ),
    )
    parser.add_argument(
        "--sae-npz", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0"
            / "sae_model.npz"
        ),
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig5alt_umap",
        help="Output PNG stem (no extension); save_fig appends .png.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="PNG resolution. Default 300 keeps the file under the 1 MB "
             "README target while preserving on-screen clarity at 10x8 in.",
    )
    parser.add_argument(
        "--force-fallback", action="store_true",
        help="Force the top_cell_types fallback path (skip W_dec). Mainly "
             "for testing the secondary code path; not used in CI.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load + filter the feature report.
    # ------------------------------------------------------------------
    logger.info("[fig5alt-umap] loading feature report: %s", args.feature_report)
    relaxed_feats = _load_relaxed_features(args.feature_report)
    n_relaxed = len(relaxed_feats)
    if n_relaxed != 323:
        # Surface this loudly if upstream artifacts have shifted; the
        # README narrative assumes the canonical 323-feature pool.
        raise ValueError(
            f"Expected exactly 323 relaxed-filter features; got {n_relaxed}. "
            "If feature_report.json has changed, re-derive the relaxed pool."
        )

    relaxed_indices = [int(f["feature_idx"]) for f in relaxed_feats]
    if SPLATTER_FEATURE_IDX not in relaxed_indices:
        raise ValueError(
            f"Splatter feature {SPLATTER_FEATURE_IDX} not in relaxed pool. "
            "Upstream feature_report has shifted."
        )

    full_by_idx = _load_full_report_by_idx(args.feature_report)

    # ------------------------------------------------------------------
    # Build feature vectors. Primary: SAE decoder columns. Fallback:
    # 31-d top_cell_types projections.
    # ------------------------------------------------------------------
    approach: str
    X_relaxed: np.ndarray
    X_random: np.ndarray | None

    if not args.force_fallback and args.sae_npz.is_file():
        try:
            X_relaxed = _load_decoder_columns(args.sae_npz, relaxed_indices)
            X_random = _load_decoder_columns(
                args.sae_npz, list(RANDOM_FEATURE_INDICES)
            )
            approach = (
                "full SAE decoder (W_dec[:, j] in input space, dim 64)"
            )
            logger.info(
                "[fig5alt-umap] using decoder-vector approach: "
                "X_relaxed.shape=%s, X_random.shape=%s",
                X_relaxed.shape, X_random.shape,
            )
        except (KeyError, ValueError) as exc:
            logger.warning(
                "[fig5alt-umap] decoder load failed (%s); falling back to "
                "top_cell_types 31-d projections.",
                exc,
            )
            X_relaxed = None  # type: ignore[assignment]
    else:
        X_relaxed = None  # type: ignore[assignment]

    if X_relaxed is None:
        ct_order = _canonical_cell_type_order_from_feats(relaxed_feats)
        X_relaxed = _build_top_ct_fallback_vectors(relaxed_feats, ct_order)
        random_feats = [
            full_by_idx[idx] for idx in RANDOM_FEATURE_INDICES
            if idx in full_by_idx
        ]
        X_random = _build_top_ct_fallback_vectors(random_feats, ct_order)
        approach = (
            "fallback: top_cell_types 31-d projections (3 non-zeros per row)"
        )
        logger.info(
            "[fig5alt-umap] using fallback approach: "
            "X_relaxed.shape=%s, X_random.shape=%s",
            X_relaxed.shape, X_random.shape,
        )

    # ------------------------------------------------------------------
    # UMAP fit on the 323; transform the 10 random controls.
    # ------------------------------------------------------------------
    embedding_fit, embedding_random, umap_meta = _umap_fit_and_transform(
        X_relaxed, X_random
    )
    if embedding_fit.shape != (n_relaxed, 2):
        raise ValueError(
            f"UMAP embedding shape {embedding_fit.shape} "
            f"!= expected ({n_relaxed}, 2)"
        )

    # ------------------------------------------------------------------
    # Pull out Splatter row for highlight overlay.
    # ------------------------------------------------------------------
    splatter_row = relaxed_indices.index(SPLATTER_FEATURE_IDX)
    splatter_xy = (
        float(embedding_fit[splatter_row, 0]),
        float(embedding_fit[splatter_row, 1]),
    )
    splatter_feat = full_by_idx[SPLATTER_FEATURE_IDX]
    splatter_top_ct = _max_ct_identity(splatter_feat) or "Splatter"
    splatter_label = f"feat {SPLATTER_FEATURE_IDX} ({splatter_top_ct})"

    # ------------------------------------------------------------------
    # Build per-feature colors (top-CT identity).
    # ------------------------------------------------------------------
    feature_colors: list[str] = []
    for feat in relaxed_feats:
        ct = _max_ct_identity(feat)
        feature_colors.append(CELL_TYPE_COLORS.get(ct or "", "#808080"))

    # Random-control labels (compact: just the index).
    random_labels = [str(idx) for idx in RANDOM_FEATURE_INDICES]
    random_xy = (
        np.asarray(embedding_random, dtype=np.float64)
        if embedding_random is not None
        else np.zeros((0, 2), dtype=np.float64)
    )

    # ------------------------------------------------------------------
    # Render + save.
    # ------------------------------------------------------------------
    fig = make_figure(
        embedding_fit=embedding_fit,
        feature_colors=feature_colors,
        splatter_xy=splatter_xy,
        splatter_label=splatter_label,
        random_xy=random_xy,
        random_labels=random_labels,
        n_relaxed=n_relaxed,
        n_random=len(RANDOM_FEATURE_INDICES),
    )

    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig5alt-umap] removing preexisting %s", out_png)
        out_png.unlink()
    written = save_fig(fig, args.out_stem, formats=("png",), dpi=args.dpi)
    plt.close(fig)

    _print_report(
        n_relaxed=n_relaxed,
        embedding_shape=tuple(int(s) for s in embedding_fit.shape),
        splatter_xy=splatter_xy,
        approach=approach,
        umap_meta=umap_meta,
        out_paths=written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
