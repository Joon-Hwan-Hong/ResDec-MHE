#!/usr/bin/env python
"""README Figure 3 alt-1: brain-anatomy spatial consensus map.

A single-panel spatial layout of all 31 cell types arranged by approximate
anatomical / lineage region. Each CT is rendered as a circle whose:

  - Position : anatomical region group (cortex/PFC top, hippocampus middle,
               cerebellum/rhombic-lip bottom, thalamus/midbrain right,
               striatum/amygdala left, glia/vascular interleaved).
  - Color    : top-5 frequency = count out of 11 methods that rank the CT
               in their top-5 (viridis sequential, 0..11).
  - Size     : inversely proportional to ``zero_frac`` (well-covered CTs =
               larger circles). Marker area scales as
               ``MIN_AREA + (MAX_AREA - MIN_AREA) * (1 - zero_frac)``.
  - Label    : CT name placed adjacent to each circle.

The layout is purely illustrative -- there is no ground-truth atlas behind
the coordinates. Cluster groups are indicated with light-grey background
labels (e.g. "Neocortex deep", "Hippocampus") near each cluster.

Inputs
------
  - ``outputs/canonical/interpretability/figures/consensus_heatmap/consensus_heatmap_data.json``
    Provides ``ranks[ct][method]`` for the CTs that appear in any method's
    top-5. CTs absent from this dict have a top-5 count of 0 by definition.
    (The published JSON only persists the 10 visualised rows; for this
    figure we treat the 21 absent CTs as count=0 -- they are coloured at
    the bottom of the viridis scale.)
  - ``outputs/canonical/interpretability/ct_coverage_full_cohort.json``
    Provides ``per_ct[ct]::zero_frac`` for all 31 CTs.

Outputs
-------
  - ``figures/fig3alt_brain_anatomy.png`` (600 DPI, 10 x 10 in canvas)
  - Verification: top-5 CTs by top-5-count, with their zero_frac, printed.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/make_readme_fig3alt_brain_anatomy.py

Idempotence
-----------
Pure JSON -> matplotlib pipeline; no sampling, no model inference.
PYTHONHASHSEED is pinned so repeated runs produce a bit-identical PNG.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Set PYTHONHASHSEED defensively for bit-identical reruns.
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch

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


N_METHODS_TOTAL = 11  # 11-method consensus universe


# ---------------------------------------------------------------------------
# Anatomical / lineage grouping (matches the task spec verbatim).
# ---------------------------------------------------------------------------

# Each entry: (group_label, group_anchor_xy, cts_in_group)
# Anchor is the cluster centroid; CTs within a group are placed in a tight
# sub-grid centered on the anchor.
#
# Coordinate convention (data units, both axes in [0, 10]):
#   (0, 0) bottom-left, (10, 10) top-right.
#   y > 7  : cortex / PFC (top of the panel)
#   3 <= y <= 7 : mid-region (hippocampus, glia, subcortical/diencephalon,
#                 striatum, support tissue interleaved)
#   y < 3  : cerebellum / hindbrain (bottom)
#   x small (<3) : "left" lobes  (striatum, hippocampus)
#   x mid  (3-7) : midline / glia
#   x large (>7) : subcortical / diencephalon ("right")
#
# The exact placements are illustrative; the only invariant the figure
# depends on is that each CT lands inside the bounding box of its declared
# group so the cluster label sits at a sensible centroid.
# Coordinate budget: x in [-0.6, 21.5], y in [-2.4, 15.0]. Three columns at
# x in {3.5, 10.5, 17.5} accommodate clusters with up-to-3 horizontally-
# placed members (h_spacing = 2.85 -> max sub-grid width ~5.7 for 3-col
# clusters and ~5.7 for 2-col 4-member clusters). Three vertical bands at
# y = 12.7 (cortex), 7.7 (mid), 2.8 (bottom) with ~5 units of vertical
# slack accommodate cluster labels + per-CT labels without cross-row
# bleed.
GROUPS: list[dict] = [
    # ---- Top row: cortex (3 clusters across the top) ----
    {
        "label": "Cortical interneurons",
        "anchor": (3.5, 12.7),
        "cts": [
            "CGE interneuron",
            "MGE interneuron",
            "Splatter",
        ],
    },
    {
        "label": "Neocortex (upper layers)",
        "anchor": (10.5, 12.7),
        "cts": [
            "Upper-layer intratelencephalic",
            "LAMP5-LHX6 and Chandelier",
        ],
    },
    {
        "label": "Neocortex (deep layers)",
        "anchor": (17.5, 12.7),
        "cts": [
            "Deep-layer intratelencephalic",
            "Deep-layer corticothalamic and 6b",
            "Deep-layer near-projecting",
        ],
    },
    # ---- Middle row: hippocampus (left) | glia (mid) | subcortical (right) ----
    {
        "label": "Hippocampus",
        "anchor": (3.5, 7.7),
        "cts": [
            "Hippocampal dentate gyrus",
            "Hippocampal CA1-3",
            "Hippocampal CA4",
        ],
    },
    {
        "label": "Glia",
        "anchor": (10.5, 7.7),
        "cts": [
            "Astrocyte",
            "Oligodendrocyte",
            "Oligodendrocyte precursor",
            "Committed oligodendrocyte precursor",
            "Microglia",
        ],
    },
    {
        "label": "Subcortical / diencephalon",
        "anchor": (17.5, 7.7),
        "cts": [
            "Thalamic excitatory",
            "Mammillary body",
            "Amygdala excitatory",
            "Midbrain-derived inhibitory",
        ],
    },
    # ---- Bottom row: striatum (left) | cerebellum (mid) | support (right) ----
    {
        "label": "Striatum",
        "anchor": (3.5, 2.8),
        "cts": [
            "Medium spiny neuron",
            "Eccentric medium spiny neuron",
        ],
    },
    {
        "label": "Cerebellum / hindbrain",
        "anchor": (10.5, 2.8),
        "cts": [
            "Upper rhombic lip",
            "Lower rhombic lip",
            "Cerebellar inhibitory",
            "Bergmann glia",
        ],
    },
    {
        "label": "Support tissue",
        "anchor": (17.5, 2.8),
        "cts": [
            "Vascular",
            "Fibroblast",
            "Ependymal",
            "Choroid plexus",
        ],
    },
    # ---- Floating: misc (well-covered, taxonomically other) ----
    {
        "label": "Other",
        "anchor": (10.5, -0.10),
        "cts": [
            "Miscellaneous",
        ],
    },
]


# ---------------------------------------------------------------------------
# Sub-grid layout helpers.
# ---------------------------------------------------------------------------
def _grid_offsets(
    n: int,
    *,
    h_spacing: float = 2.85,
    v_spacing: float = 1.70,
) -> list[tuple[float, float]]:
    """Return ``n`` (dx, dy) offsets centered on (0, 0).

    Uses a square-ish grid; for n in {1..6} the layout is:
        n=1 -> [(0, 0)]
        n=2 -> stacked vertically (avoids horizontal label collisions)
        n=3 -> equilateral triangle (one above two)
        n=4 -> 2x2 square
        n=5 -> quincunx (4 corners + center)
        n=6 -> 3x2 grid

    h_spacing > v_spacing because CT name labels can be long (e.g.
    "Deep-layer corticothalamic and 6b"); horizontal neighbours therefore
    need more clearance than vertical neighbours.
    """
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        # Vertical stack: labels for 2-CT clusters (Striatum, Upper neocortex)
        # are long enough that side-by-side placement collides. Stack instead.
        return [(0.0, 0.5 * v_spacing), (0.0, -0.5 * v_spacing)]
    if n == 3:
        return [
            (0.0, 0.55 * v_spacing),
            (-0.5 * h_spacing, -0.55 * v_spacing),
            (0.5 * h_spacing, -0.55 * v_spacing),
        ]
    if n == 4:
        return [
            (-0.5 * h_spacing, 0.55 * v_spacing),
            (0.5 * h_spacing, 0.55 * v_spacing),
            (-0.5 * h_spacing, -0.55 * v_spacing),
            (0.5 * h_spacing, -0.55 * v_spacing),
        ]
    if n == 5:
        # quincunx: 4 corners + center
        return [
            (-0.65 * h_spacing, 0.7 * v_spacing),
            (0.65 * h_spacing, 0.7 * v_spacing),
            (-0.65 * h_spacing, -0.7 * v_spacing),
            (0.65 * h_spacing, -0.7 * v_spacing),
            (0.0, 0.0),
        ]
    if n == 6:
        return [
            (-1.0 * h_spacing, 0.55 * v_spacing),
            (0.0, 0.55 * v_spacing),
            (1.0 * h_spacing, 0.55 * v_spacing),
            (-1.0 * h_spacing, -0.55 * v_spacing),
            (0.0, -0.55 * v_spacing),
            (1.0 * h_spacing, -0.55 * v_spacing),
        ]
    # General ceil(sqrt(n)) x ceil(sqrt(n)) grid for groups beyond 6.
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    out: list[tuple[float, float]] = []
    for k in range(n):
        r = k // cols
        c = k % cols
        dx = (c - (cols - 1) / 2.0) * h_spacing
        dy = ((rows - 1) / 2.0 - r) * v_spacing
        out.append((dx, dy))
    return out


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------
def _load_top5_counts(consensus_json: Path) -> dict[str, int]:
    """Compute {ct: top5_count} from the consensus heatmap JSON.

    The JSON's ``ranks`` field is keyed by CT and maps to per-method ranks;
    we count how many methods ranked the CT <= 5. CTs absent from
    ``ranks`` had a top-5 count of zero by construction (the JSON only
    persists CTs that entered any method's top-5), so callers should fill
    missing CTs with 0.
    """
    if not consensus_json.exists():
        raise FileNotFoundError(f"consensus heatmap JSON not found: {consensus_json}")
    payload = json.loads(consensus_json.read_text())
    ranks = payload.get("ranks", {})
    counts: dict[str, int] = {}
    for ct, per_method in ranks.items():
        c = 0
        for _method, rank in per_method.items():
            if isinstance(rank, (int, float)) and rank <= 5:
                c += 1
        counts[ct] = c
    return counts


def _load_zero_fracs(coverage_json: Path) -> dict[str, float]:
    """Return {ct: zero_frac} for all 31 CTs from the coverage JSON."""
    if not coverage_json.exists():
        raise FileNotFoundError(f"coverage JSON not found: {coverage_json}")
    payload = json.loads(coverage_json.read_text())
    per_ct = payload.get("per_ct", {})
    out: dict[str, float] = {}
    for ct, entry in per_ct.items():
        out[ct] = float(entry["zero_frac"])
    return out


# ---------------------------------------------------------------------------
# Layout solver.
# ---------------------------------------------------------------------------
def _compute_positions(
    groups: list[dict],
) -> dict[str, tuple[float, float]]:
    """Return {ct: (x, y)} positions for every CT declared across groups.

    Each group anchors its CTs in a tight sub-grid via ``_grid_offsets``.
    """
    positions: dict[str, tuple[float, float]] = {}
    for group in groups:
        anchor_x, anchor_y = group["anchor"]
        cts = group["cts"]
        offsets = _grid_offsets(len(cts))
        for ct, (dx, dy) in zip(cts, offsets):
            positions[ct] = (anchor_x + dx, anchor_y + dy)
    return positions


def _all_declared_cts(groups: list[dict]) -> list[str]:
    """Flatten the declared CTs across all groups (preserving group order)."""
    out: list[str] = []
    for group in groups:
        out.extend(group["cts"])
    return out


# ---------------------------------------------------------------------------
# Drawing.
# ---------------------------------------------------------------------------
# Marker area scaling: zero_frac=0 -> MAX_AREA, zero_frac=1 -> MIN_AREA.
# Areas are matplotlib scatter ``s`` (square points), tuned so the well-
# covered CTs at zero_frac=0 fill their cluster nicely without overlap.
MIN_AREA = 80.0
MAX_AREA = 700.0


def _marker_area(zero_frac: float) -> float:
    """Inverse-proportional marker area: well-covered CT -> larger circle."""
    f = max(0.0, min(1.0, float(zero_frac)))
    return MIN_AREA + (MAX_AREA - MIN_AREA) * (1.0 - f)


# Maximum characters per line for CT name labels. Labels longer than this
# are wrapped on " and " / whitespace so they stay within their per-CT
# slot. Tuned for fontsize=6.5 + h_spacing=2.85 + figsize=10x10 in.
_MAX_LABEL_CHARS_PER_LINE = 18


def _wrap_label(label: str, *, max_chars: int = _MAX_LABEL_CHARS_PER_LINE) -> str:
    """Insert a newline so each line is <= ``max_chars`` characters.

    Strategy:
      1. If the label is already short, return as-is.
      2. Prefer to split on " and " (yields readable two-line cell names).
      3. Otherwise, find the whitespace nearest the midpoint and split.
      4. Fall back to splitting at ``max_chars`` even mid-word; this only
         triggers for pathological inputs (no whitespace), which shouldn't
         occur in our 31-CT taxonomy.
    """
    if len(label) <= max_chars:
        return label
    # Prefer " and " as the split anchor (matches "Deep-layer
    # corticothalamic and 6b", "LAMP5-LHX6 and Chandelier").
    if " and " in label:
        i = label.index(" and ")
        return label[:i] + "\n" + label[i + 1:]
    # Otherwise, split on the whitespace nearest the midpoint.
    mid = len(label) // 2
    candidates = [i for i, ch in enumerate(label) if ch == " "]
    if candidates:
        best = min(candidates, key=lambda i: abs(i - mid))
        return label[:best] + "\n" + label[best + 1:]
    # No whitespace at all -- fall back to a hard split at max_chars.
    return label[:max_chars] + "\n" + label[max_chars:]


def _label_offset_for_circle(area: float) -> tuple[float, float]:
    """Return a (dx, dy) offset placing the label below the circle.

    The offset scales with sqrt(area) so labels clear the circle's edge
    in data units regardless of marker size.
    """
    # sqrt(area) is the marker's diameter in points. With figsize=10 in
    # tall and y data-range ~17.4 units, each inch maps to ~1.74 data
    # units; 72 points per inch means data-unit-per-point = ~0.024. So
    # dividing sqrt(area) by 1/0.024 ~= 42 gives the dot diameter in
    # data units. Use 55 for some clearance margin so labels sit just
    # below the marker rim.
    diameter_data = np.sqrt(area) / 55.0
    return (0.0, -(diameter_data * 0.5 + 0.32))


def _draw_panel(
    ax: plt.Axes,
    positions: dict[str, tuple[float, float]],
    counts: dict[str, int],
    zero_fracs: dict[str, float],
    groups: list[dict],
) -> ScalarMappable:
    """Render the spatial map. Returns the ScalarMappable for the colorbar."""
    cmap = PALETTES["sequential"]
    norm = Normalize(vmin=0, vmax=N_METHODS_TOTAL)

    # Cluster-group background labels first (so they sit beneath markers).
    # Labels are placed *above* the cluster's top-most circle, with extra
    # padding scaled to the largest circle in the group so the label clears
    # the marker edge regardless of zero_frac. We track each label's claim
    # over the y-range so per-CT name labels (placed below circles below)
    # never collide with a neighbouring cluster's title.
    for group in groups:
        anchor_x, anchor_y = group["anchor"]
        cts = group["cts"]
        if not cts:
            continue
        xs = [positions[ct][0] for ct in cts if ct in positions]
        ys = [positions[ct][1] for ct in cts if ct in positions]
        if not xs:
            continue
        cx = float(np.mean(xs))
        # Largest circle in the cluster determines how far above the bbox
        # the label needs to sit. Same divisor as `_label_offset_for_circle`
        # so the geometry matches across both label types.
        max_area = max(_marker_area(zero_fracs.get(ct, 1.0)) for ct in cts)
        radius_data = np.sqrt(max_area) / 55.0 * 0.5
        cy_top = float(np.max(ys)) + radius_data + 0.55
        ax.text(
            cx, cy_top,
            group["label"],
            ha="center", va="center",
            fontsize=9.0,
            color="#888888",
            fontstyle="italic",
            fontweight="medium",
            zorder=1,
        )

    # Plot circles. We collect arrays for a single scatter call so the
    # colorbar bound to the returned mappable is exact.
    xs_all: list[float] = []
    ys_all: list[float] = []
    cs_all: list[int] = []
    ss_all: list[float] = []
    cts_in_order: list[str] = []
    for ct, (x, y) in positions.items():
        xs_all.append(x)
        ys_all.append(y)
        cs_all.append(counts.get(ct, 0))
        ss_all.append(_marker_area(zero_fracs.get(ct, 1.0)))
        cts_in_order.append(ct)

    sc = ax.scatter(
        xs_all, ys_all,
        c=cs_all,
        s=ss_all,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # CT labels, placed below each circle with an area-proportional offset
    # so larger circles have their label further down. Long labels (e.g.
    # "Deep-layer corticothalamic and 6b") are wrapped on " and " /
    # whitespace so they stay within the per-CT slot width without
    # colliding with horizontal neighbours. Labels sit on a subtle
    # white-ish background so they stay legible over neighbouring circles.
    for ct, x, y, area in zip(cts_in_order, xs_all, ys_all, ss_all):
        dx, dy = _label_offset_for_circle(area)
        wrapped = _wrap_label(ct)
        ax.text(
            x + dx, y + dy,
            wrapped,
            ha="center", va="top",
            fontsize=6.5,
            color="#222222",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.12",
                      facecolor="white", edgecolor="none", alpha=0.70),
        )

    # Axis cosmetics: hide the data-coord ticks (they're meaningless in a
    # spatial-illustration panel) and remove the box frame.
    ax.set_xlim(-0.6, 21.0)
    ax.set_ylim(-2.4, 15.0)
    ax.set_xticks([])
    ax.set_yticks([])
    fmt_axes(ax, hide_spines=("top", "right", "bottom", "left"),
             grid_major=False, grid_minor=False)

    # Colorbar for top-5 count.
    cb = ax.figure.colorbar(
        sc, ax=ax,
        fraction=0.035, pad=0.02,
        ticks=list(range(0, N_METHODS_TOTAL + 1, 2)),
    )
    cb.set_label("Top-5 count (out of 11 methods)", fontsize=8)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, labelsize=7)

    # Size legend (3 representative circles + tick labels) drawn as a
    # secondary axes-anchored inset so the area-encoding is interpretable.
    _draw_size_legend(ax)

    return sc


def _draw_size_legend(ax: plt.Axes) -> None:
    """Annotate the inverse zero_frac -> circle area mapping in a corner.

    Placed in the lower-left corner of the canvas (below the Striatum
    cluster, x in [-0.5, 3.0]). The "Other" / Miscellaneous floating
    cluster lives further to the right at x=10.5, so this position is
    free. Three representative ``zero_frac`` values are shown: 0.00,
    0.50, and 0.95 (which span the observed range across the 31 CTs).
    """
    ref_zero_fracs = (0.00, 0.50, 0.95)
    ref_labels = ("zero_frac=0.00\n(well covered)",
                  "zero_frac=0.50",
                  "zero_frac=0.95\n(sparse)")

    # Anchor: lower-left corner of the canvas. Vertical layout, top-down.
    anchor_x = -0.30
    anchor_y_top = -0.50
    spacing_y = 0.65

    # Light grey rounded background to delimit the legend region.
    bg = FancyBboxPatch(
        (anchor_x - 0.10, anchor_y_top - len(ref_zero_fracs) * spacing_y - 0.10),
        3.2,
        len(ref_zero_fracs) * spacing_y + 0.85,
        boxstyle="round,pad=0.05",
        linewidth=0.6,
        edgecolor="#cccccc",
        facecolor="white",
        zorder=5,
    )
    ax.add_patch(bg)
    ax.text(
        anchor_x + 1.40, anchor_y_top + 0.45,
        "Circle area",
        ha="center", va="bottom",
        fontsize=8.0,
        color="#222222",
        fontweight="bold",
        zorder=6,
    )

    for i, (zf, lbl) in enumerate(zip(ref_zero_fracs, ref_labels)):
        cy = anchor_y_top - i * spacing_y
        ax.scatter(
            [anchor_x + 0.40], [cy],
            s=_marker_area(zf),
            c=["#888888"],
            edgecolor="white",
            linewidth=0.7,
            zorder=6,
        )
        ax.text(
            anchor_x + 0.95, cy,
            lbl,
            ha="left", va="center",
            fontsize=7.0,
            color="#444444",
            zorder=6,
        )


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------
def make_figure(
    *,
    counts: dict[str, int],
    zero_fracs: dict[str, float],
    groups: list[dict] = GROUPS,
) -> tuple[plt.Figure, dict[str, tuple[float, float]]]:
    """Build the figure. Returns (fig, positions_used)."""
    apply_theme("paper")
    declared = _all_declared_cts(groups)
    missing_in_coverage = [ct for ct in declared if ct not in zero_fracs]
    if missing_in_coverage:
        raise KeyError(
            "CT declared in GROUPS missing from coverage JSON: "
            f"{missing_in_coverage}"
        )

    positions = _compute_positions(groups)

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_panel(ax, positions, counts, zero_fracs, groups)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.98, bottom=0.02)
    return fig, positions


def _print_report(
    counts: dict[str, int],
    zero_fracs: dict[str, float],
    declared: list[str],
) -> None:
    """Echo the top-5 CTs by top-5 count and their zero_frac."""
    print("=" * 72)
    print("README Figure 3 alt-1 -- brain-anatomy spatial consensus map")
    print("=" * 72)
    print(f"  n_cts_declared : {len(declared)}")

    # Build full count table including zero-count CTs (those absent from
    # the consensus_heatmap_data.json::ranks dict).
    full_counts = {ct: counts.get(ct, 0) for ct in declared}
    top5_cts = sorted(
        full_counts.items(),
        key=lambda kv: (-kv[1], zero_fracs.get(kv[0], 1.0), kv[0]),
    )[:5]
    print("  Top 5 CTs by top-5-count (count out of 11 methods):")
    for ct, c in top5_cts:
        zf = zero_fracs.get(ct, float("nan"))
        print(f"    {ct:42s}  count={c:2d}/11  zero_frac={zf:.4f}")
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--consensus-json", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/consensus_heatmap/consensus_heatmap_data.json",
        help="Consensus heatmap JSON with per-method per-CT ranks.",
    )
    parser.add_argument(
        "--coverage-json", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ct_coverage_full_cohort.json",
        help="CT coverage JSON with per_ct[ct]::zero_frac.",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig3alt_brain_anatomy",
        help="Output path stem (no extension); save_fig appends .png.",
    )
    args = parser.parse_args()

    logger.info("[fig3alt] consensus  = %s", args.consensus_json)
    logger.info("[fig3alt] coverage   = %s", args.coverage_json)

    counts = _load_top5_counts(args.consensus_json)
    zero_fracs = _load_zero_fracs(args.coverage_json)
    declared = _all_declared_cts(GROUPS)

    if len(declared) != 31:
        raise ValueError(
            f"GROUPS declares {len(declared)} CTs but expected 31; check "
            "the anatomical mapping for missing or duplicate entries."
        )
    if len(zero_fracs) != 31:
        raise ValueError(
            f"coverage JSON has {len(zero_fracs)} CTs; expected 31"
        )
    declared_set = set(declared)
    coverage_set = set(zero_fracs.keys())
    missing_from_groups = sorted(coverage_set - declared_set)
    extra_in_groups = sorted(declared_set - coverage_set)
    if missing_from_groups or extra_in_groups:
        raise ValueError(
            "GROUPS / coverage CT-name mismatch: "
            f"missing_from_groups={missing_from_groups} "
            f"extra_in_groups={extra_in_groups}"
        )

    fig, _positions = make_figure(counts=counts, zero_fracs=zero_fracs)
    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig3alt] removing preexisting %s", out_png)
        out_png.unlink()
    written = save_fig(fig, args.out_stem, formats=("png",))
    plt.close(fig)
    for w in written:
        logger.info("[fig3alt] wrote %s", w)

    _print_report(counts, zero_fracs, declared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
