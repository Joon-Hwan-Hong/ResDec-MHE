"""Lab-meeting supplementary figures D2 and D3 (CCC attention).

Two figures, both written to ``--out-dir``:

D2 — fig_ccc_per_edge_type_heatmap.{png,pdf}
    5-panel composite (one panel per CellChatDB edge type), each a 31×31
    heatmap of subject-averaged attention over (source CT × target CT).

    Edge types come from ``src/data/constants.py:ALL_EDGE_TYPES``:
        Secreted_Signaling, ECM_Receptor, Cell_Cell_Contact,
        Non_protein_Signaling, Novel_Uncharacterized

    NOTE: the user spec listed different names ("Cell-Cell-Contact,
    Co-expression, Curated-LRI, Predicted-LRI, Splatter-internal") but the
    actual canonical edge types in the data are CellChatDB-derived. The data
    itself (``edge_type_order`` in the npz) is the source of truth.

D3 — fig_ccc_subject_heterogeneity_strip.{png,pdf}
    Strip plot of N=516 subjects sorted by max CCC attention descending.
    Outliers above ``--outlier-threshold`` (default 0.01) — ~15 subjects per
    §15.1 — are colored differently to emphasise the structured-but-redundant
    heterogeneity finding.

Inputs
------
* ``per_subject_ccc_attention.npz`` — attention [N=516, 31, 31, 5] +
  ``cell_type_order``, ``edge_type_order``, ``subject_ids``, ``folds``.
* ``per_subject_ccc_attention_summary.json`` — per_subject list with
  max_attention.

Outputs
-------
* ``fig_ccc_per_edge_type_heatmap.{png,pdf}`` (D2)
* ``fig_ccc_subject_heterogeneity_strip.{png,pdf}`` (D3)
* ``ccc_supp_data.json`` (per-edge-type max ranges + outlier count)
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

import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import EDGE_TYPE_DISPLAY_NAMES  # noqa: E402
from src.visualization.composite import auto_letter, make_panel  # noqa: E402
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

_DEFAULT_NPZ = os.environ.get(
    "CCC_SUPP_NPZ",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc/per_subject_ccc_attention.npz"),
)
_DEFAULT_SUMMARY = os.environ.get(
    "CCC_SUPP_SUMMARY",
    str(_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc"
        / "per_subject_ccc_attention_summary.json"),
)
_DEFAULT_OUT_DIR = os.environ.get(
    "CCC_SUPP_OUT_DIR",
    str(_WORKTREE_ROOT / "outputs/canonical/interpretability/figures/ccc_supp"),
)
_DEFAULT_OUTLIER_THRESHOLD = float(os.environ.get("CCC_SUPP_OUTLIER_THRESHOLD", "0.01"))


# ---------------------------------------------------------------------------
# D2 — per-edge-type 31×31 attention heatmaps
# ---------------------------------------------------------------------------


def _load_attention(npz_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (mean_attention[31,31,5], cell_type_order, edge_type_order).

    The attention array is sparse: ~81% NaN (entries where the (src CT, tgt CT,
    edge type) combination has no actual edge in the data). We use ``nanmean``
    so the per-cell value is the subject-mean over subjects that DO have an
    edge there. Edges with NO subject (all-NaN slice) become 0.0 for plotting.
    """
    import warnings

    data = np.load(npz_path, allow_pickle=True)
    attention = np.asarray(data["attention"], dtype=np.float64)  # [N, C, C, E]
    ct_order = list(map(str, data["cell_type_order"]))
    et_order = list(map(str, data["edge_type_order"]))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean_att = np.nanmean(attention, axis=0)  # [C, C, E]
    # Replace remaining NaNs (all-NaN slices) with 0 so they render as the
    # cmap's lower bound rather than poisoning vmax/vmin.
    mean_att = np.where(np.isnan(mean_att), 0.0, mean_att)
    return mean_att.astype(np.float32), ct_order, et_order


def build_d2_figure(
    npz_path: Path,
) -> plt.Figure:
    """Render the 5-panel D2 figure and return it (without saving)."""
    apply_theme()
    mean_att, ct_order, et_order = _load_attention(npz_path)
    n_et = len(et_order)
    cmap = PALETTES["sequential"]

    # Per-panel global vmax = global max across panels (so colors are
    # comparable; per §15 the dynamic range varies a lot between edge types
    # so we still log per-panel ranges in the metadata).
    vmax = float(mean_att.max())

    panels: list[dict] = []
    for ei, et in enumerate(et_order):
        display = EDGE_TYPE_DISPLAY_NAMES.get(et, et)
        panel_data = mean_att[:, :, ei]

        def _draw(ax, data=panel_data, label=display, vmax=vmax):
            im = ax.imshow(
                data, aspect="equal", cmap=cmap,
                interpolation="nearest", vmin=0.0, vmax=vmax,
            )
            ax.set_xticks(np.arange(len(ct_order)))
            ax.set_yticks(np.arange(len(ct_order)))
            ax.set_xticklabels(ct_order, rotation=70, ha="right", fontsize=4)
            ax.set_yticklabels(ct_order, fontsize=4)
            ax.set_xlabel("target CT", fontsize=7)
            ax.set_ylabel("source CT", fontsize=7)
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(labelsize=6)

        panels.append({"draw": _draw, "title": display})

    fig = make_panel(
        panels,
        layout=(1, n_et),
        figsize=(16.0, 6.0),
        labels=True,
        wspace=0.45,
        hspace=0.30,
    )
    fig.suptitle(
        "CCC subject-mean attention per edge type "
        "(31 source CT × 31 target CT, mean across N=516 subjects)",
        fontsize=10, y=1.02,
    )
    return fig


def build_d2_per_edge_type_heatmap(
    npz_path: Path,
    out_dir: Path,
) -> dict:
    """Render and save the D2 figure. Returns per-edge-type metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_d2_figure(npz_path=npz_path)
    save_fig(fig, out_dir / "fig_ccc_per_edge_type_heatmap")
    plt.close(fig)

    mean_att, ct_order, et_order = _load_attention(npz_path)
    per_et_meta = {}
    for ei, et in enumerate(et_order):
        m = mean_att[:, :, ei]
        per_et_meta[et] = {
            "min": float(m.min()),
            "max": float(m.max()),
            "mean": float(m.mean()),
        }
    return per_et_meta


# ---------------------------------------------------------------------------
# D3 — per-subject heterogeneity strip
# ---------------------------------------------------------------------------


def _load_per_subject_max(summary_path: Path) -> tuple[list[str], np.ndarray]:
    """Return (subject_ids_sorted_desc, max_attention_sorted_desc)."""
    summary = json.loads(Path(summary_path).read_text())
    per_subject = summary["per_subject"]
    sids = np.array([p["subject_id"] for p in per_subject])
    max_att = np.array([float(p["max_attention"]) for p in per_subject], dtype=np.float64)
    order = np.argsort(-max_att)
    return list(sids[order]), max_att[order]


def count_outliers(
    summary_path: Path,
    threshold: float,
) -> int:
    _, max_att = _load_per_subject_max(summary_path)
    return int((max_att > threshold).sum())


def build_d3_subject_heterogeneity_strip(
    summary_path: Path,
    out_dir: Path,
    *,
    outlier_threshold: float = _DEFAULT_OUTLIER_THRESHOLD,
) -> dict:
    apply_theme()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sids_sorted, max_att_sorted = _load_per_subject_max(summary_path)
    n = len(max_att_sorted)
    is_outlier = max_att_sorted > outlier_threshold
    n_outliers = int(is_outlier.sum())

    palette = list(PALETTES["categorical"])
    bulk_color = "#bdbdbd"      # gray for the bulk
    outlier_color = palette[3]  # tab10 red — the 15 structured outliers

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    x = np.arange(n)
    ax.scatter(
        x[~is_outlier], max_att_sorted[~is_outlier],
        s=10, color=bulk_color, alpha=0.7, edgecolor="none",
        label=f"bulk (n={n - n_outliers})",
    )
    ax.scatter(
        x[is_outlier], max_att_sorted[is_outlier],
        s=22, color=outlier_color, edgecolor="white", linewidth=0.5,
        label=f"outliers > {outlier_threshold:g} (n={n_outliers})",
        zorder=5,
    )
    ax.axhline(
        outlier_threshold, color=outlier_color, linewidth=0.8,
        linestyle="--", alpha=0.6,
    )

    ax.set_xlabel(f"Subjects (rank-ordered by max attention, n={n})")
    ax.set_ylabel("Max CCC edge attention (per subject)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fmt_axes(ax)

    # Annotation per spec: structured-but-redundant heterogeneity per §15
    note = (
        f"{n_outliers}/{n} subjects with edges > {outlier_threshold:g}\n"
        "(structured deep-layer→microglia/OPC heterogeneity,\n"
        "but redundant for prediction per LOCO-ablation null in §15)"
    )
    ax.text(
        0.05, 0.55, note,
        transform=ax.transAxes, fontsize=7.5,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.5),
    )

    fig.suptitle(
        "Per-subject CCC heterogeneity: structured outliers, redundant signal",
        fontsize=10, y=1.00,
    )
    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.97)

    save_fig(fig, out_dir / "fig_ccc_subject_heterogeneity_strip")
    plt.close(fig)
    return {
        "n_subjects": int(n),
        "n_outliers": n_outliers,
        "outlier_threshold": float(outlier_threshold),
        "max_attention_min": float(max_att_sorted.min()),
        "max_attention_max": float(max_att_sorted.max()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_all(
    npz_path: Path,
    summary_path: Path,
    out_dir: Path,
    *,
    outlier_threshold: float = _DEFAULT_OUTLIER_THRESHOLD,
) -> dict:
    """Render D2 and D3, write a combined metadata JSON."""
    d2_meta = build_d2_per_edge_type_heatmap(npz_path=npz_path, out_dir=out_dir)
    d3_meta = build_d3_subject_heterogeneity_strip(
        summary_path=summary_path,
        out_dir=out_dir,
        outlier_threshold=outlier_threshold,
    )
    out = {"d2_per_edge_type": d2_meta, "d3_strip": d3_meta}
    (Path(out_dir) / "ccc_supp_data.json").write_text(json.dumps(out, indent=2))
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--npz", default=_DEFAULT_NPZ,
                   help="per_subject_ccc_attention.npz")
    p.add_argument("--summary", default=_DEFAULT_SUMMARY,
                   help="per_subject_ccc_attention_summary.json")
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    p.add_argument("--outlier-threshold", type=float,
                   default=_DEFAULT_OUTLIER_THRESHOLD,
                   help="D3 outlier threshold (default 0.01)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    t0 = time.perf_counter()
    meta = build_all(
        npz_path=Path(args.npz),
        summary_path=Path(args.summary),
        out_dir=Path(args.out_dir),
        outlier_threshold=args.outlier_threshold,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Rendered D2 + D3 in %.2fs (n_outliers=%d)",
        elapsed, meta["d3_strip"]["n_outliers"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
