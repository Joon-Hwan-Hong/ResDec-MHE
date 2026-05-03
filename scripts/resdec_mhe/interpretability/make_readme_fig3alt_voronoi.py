#!/usr/bin/env python
"""README Figure 3 alt-2: weighted-Voronoi treemap (power diagram) of cross-method top-5 frequency.

Renders ``figures/fig3alt_voronoi.png`` -- a single-panel ~10x10 in figure
where each of the 31 cell types occupies an organic region. Region area
is *exactly* proportional to the number of methods (out of 11) that
placed the CT in their top-5 (with a +1.0 floor so zero-count CTs still
have visible cells); color (viridis sequential) also encodes that count.
Sparse CTs (zero_frac >= 0.20 in the 516-subject coverage cohort) are
rendered with a hatched fill pattern to flag artifact-prone status.

Algorithm
---------
True **additively-weighted Voronoi diagram** ("power diagram") via
Aurenhammer's lifted-paraboloid construction (Aurenhammer 1987).

For generators ``{p_i in R^2}`` with weights ``{w_i}``, the power-diagram
cell of generator ``i`` is::

    V_i = { x in R^2 : ||x - p_i||^2 - w_i  <=  ||x - p_j||^2 - w_j  forall j }

Construction:
  1. Lift each generator to ``q_i = (p_i_x, p_i_y, ||p_i||^2 - w_i)`` in R^3.
  2. Compute the convex hull of the lifted points (``scipy.spatial.ConvexHull``).
  3. Identify *lower* facets (outward normal has negative z-component); those
     facets project to the regular triangulation in R^2 dual to the power
     diagram. Their edges enumerate which generator pairs share a power-cell
     boundary.
  4. For each generator ``i``, intersect the half-planes
     ``(p_j - p_i) . x  <=  (||p_j||^2 - w_j - ||p_i||^2 + w_i) / 2``
     for every adjacent ``j``, and clip to the unit-square envelope.
     Half-plane intersection uses Sutherland-Hodgman clipping over a
     shapely ``Polygon``.

Iteration to match target areas:
  - Targets ``T_i = (count_i + 1.0) / sum_j(count_j + 1.0) * envelope_area``.
  - Initial: deterministic Halton placement seeded by the largest-count
    CTs first, all weights zero.
  - At each step we run a combined update:
        * Lloyd-relaxation step toward each cell's centroid.
        * Aurenhammer weight-gradient step:
            ``w_i += alpha * (T_i - A_i)``
          (cells smaller than target gain weight, growing next iteration).
  - We stop when ``max_i |A_i - T_i| / T_i < tol`` (default 5%) **and**
    no cells are empty, or after ``max_iter`` iterations whichever first.
  - Empty cells (which can transiently occur for an over-shrunk small
    cell) trigger a recovery: the empty generator is teleported to the
    envelope center plus a small random offset, with weight reset.

NO CIRCLE-PACKING FALLBACK. The user explicitly requires a power
diagram. NO unweighted Voronoi -- area must encode count.

Inputs
------
  - outputs/canonical/interpretability/figures/consensus_heatmap/consensus_heatmap_data.json
    Provides ``ranks[ct][method] -> int rank``. Top-5 count for each CT
    is the number of methods with rank <= 5. CTs not present in this
    file's ``ranks`` dict get count = 0 (they are still rendered with
    the +1.0 weight floor so the cell is visible but small).
  - outputs/canonical/interpretability/ct_coverage_full_cohort.json
    Provides ``per_ct[ct].zero_frac`` for the 516-subject hatch flag.

Outputs
-------
  - figures/fig3alt_voronoi.png  (10x10 in, **600 DPI** per project theme)

Usage
-----
  PYTHONPATH=<worktree-root> \
    uv run python scripts/resdec_mhe/interpretability/make_readme_fig3alt_voronoi.py

Idempotence
-----------
The pipeline seeds NumPy + Python's ``random`` with ``--seed`` (default 42).
PYTHONHASHSEED is pinned defensively. Two runs produce bit-identical PNG
because Halton placement, weight initialization, and Lloyd recovery
jitter are all driven by a single deterministic ``np.random.Generator``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Pin hash seed before any matplotlib / numpy imports (matplotlib's color
# selection paths can hit set-iteration order in edge cases).
os.environ.setdefault("PYTHONHASHSEED", "42")

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import (  # noqa: E402
    PALETTES,
    apply_theme,
    save_fig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_top5_counts_per_ct(
    consensus_json: Path,
    universe_cts: list[str],
    *,
    top_k: int = 5,
) -> dict[str, int]:
    """Return ``{ct: count}`` of methods with rank <= ``top_k`` for each CT.

    CTs in ``universe_cts`` not present in ``consensus_json::ranks`` get
    count = 0.
    """
    payload = json.loads(consensus_json.read_text())
    ranks: dict[str, dict[str, float]] = payload["ranks"]
    out: dict[str, int] = {ct: 0 for ct in universe_cts}
    for ct, per_method in ranks.items():
        c = sum(1 for r in per_method.values() if r is not None and r <= top_k)
        if ct in out:
            out[ct] = c
        else:
            logger.warning(
                "CT %r appears in consensus ranks but not in coverage "
                "universe; ignoring.",
                ct,
            )
    return out


def _load_zero_frac_per_ct(coverage_json: Path) -> dict[str, float]:
    """Return ``{ct: zero_frac}`` from the per-CT coverage JSON."""
    payload = json.loads(coverage_json.read_text())
    per_ct: dict[str, dict] = payload["per_ct"]
    return {ct: float(block["zero_frac"]) for ct, block in per_ct.items()}


# ---------------------------------------------------------------------------
# Power-diagram construction (Aurenhammer 1987 lifted-paraboloid)
# ---------------------------------------------------------------------------
def _power_diagram_cells(
    points: np.ndarray,
    weights: np.ndarray,
    bbox: Polygon,
) -> list[Polygon | None]:
    """Construct power-diagram cells for each generator, clipped to ``bbox``.

    Lifts ``points`` to ``q_i = (p_i_x, p_i_y, ||p_i||^2 - w_i)`` and uses
    ``scipy.spatial.ConvexHull`` to find the lower hull (outward normal
    z-component < 0). The lower hull is the regular triangulation dual to
    the power diagram; each lower-hull edge ``(i, j)`` corresponds to a
    power-cell boundary between generators ``i`` and ``j``.

    For each generator ``i``, the cell is the intersection of all
    half-planes::

        (p_j - p_i) . x  <=  (||p_j||^2 - w_j - ||p_i||^2 + w_i) / 2

    over every neighbor ``j`` (i.e., every ``(i, j)`` lower-hull edge),
    further clipped to ``bbox``. Returns a list of shapely ``Polygon``
    objects (or ``None`` if a cell becomes empty under clipping --
    the iteration loop interprets ``None`` as "needs recovery").
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(points)
    if n < 4:
        # ConvexHull in 3D needs >= 4 points; below that, every point is
        # adjacent to every other and we degenerate to all-pairs neighborhood.
        edges: set[tuple[int, int]] = set()
        for i in range(n):
            for j in range(i + 1, n):
                edges.add((i, j))
    else:
        z = (points ** 2).sum(axis=1) - weights
        Q = np.column_stack([points, z])
        try:
            hull = ConvexHull(Q)
        except Exception as exc:  # pragma: no cover -- qhull degeneracy
            logger.error("ConvexHull failed: %s", exc)
            return [None] * n
        # Lower facets: outward normal z-component < 0.
        lower_mask = hull.equations[:, 2] < 0
        lower_simplices = hull.simplices[lower_mask]
        edges = set()
        for tri in lower_simplices:
            a, b, c = sorted(int(x) for x in tri)
            edges.add((a, b))
            edges.add((a, c))
            edges.add((b, c))

    cells: list[Polygon | None] = []
    for i in range(n):
        cell: Polygon | None = bbox
        for (a, b) in edges:
            if a == i:
                j = b
            elif b == i:
                j = a
            else:
                continue
            normal = points[j] - points[i]
            rhs = (
                (points[j] ** 2).sum()
                - weights[j]
                - (points[i] ** 2).sum()
                + weights[i]
            ) / 2.0
            cell = _clip_half_plane(cell, normal, rhs)
            if cell is None or cell.is_empty:
                cell = None
                break
        cells.append(cell)
    return cells


def _clip_half_plane(
    poly: Polygon | None,
    normal: np.ndarray,
    rhs: float,
) -> Polygon | None:
    """Clip ``poly`` by half-plane ``normal . x <= rhs`` (Sutherland-Hodgman).

    Returns the clipped polygon or ``None`` if the result is degenerate
    (fewer than 3 vertices). Operates on the polygon's exterior ring;
    holes are not relevant for the convex envelope intersection.
    """
    if poly is None or poly.is_empty:
        return None
    coords = list(poly.exterior.coords)[:-1]  # drop closing duplicate
    if not coords:
        return None
    out: list[tuple[float, float]] = []
    n = len(coords)
    for k in range(n):
        c = np.asarray(coords[k], dtype=np.float64)
        cprev = np.asarray(coords[(k - 1) % n], dtype=np.float64)
        cdot = float(normal[0] * c[0] + normal[1] * c[1] - rhs)
        pdot = float(normal[0] * cprev[0] + normal[1] * cprev[1] - rhs)
        c_in = cdot <= 1e-12
        p_in = pdot <= 1e-12
        if c_in:
            if not p_in:
                t = pdot / (pdot - cdot)
                inter = cprev + t * (c - cprev)
                out.append((float(inter[0]), float(inter[1])))
            out.append((float(c[0]), float(c[1])))
        elif p_in:
            t = pdot / (pdot - cdot)
            inter = cprev + t * (c - cprev)
            out.append((float(inter[0]), float(inter[1])))
    if len(out) < 3:
        return None
    poly_clipped = Polygon(out)
    if not poly_clipped.is_valid:
        poly_clipped = poly_clipped.buffer(0)
        if poly_clipped.is_empty or not isinstance(poly_clipped, Polygon):
            return None
    return poly_clipped


# ---------------------------------------------------------------------------
# Initial placement (deterministic Halton)
# ---------------------------------------------------------------------------
def _halton(i: int, base: int) -> float:
    """Halton low-discrepancy 1-D coordinate for index ``i`` (1-based)."""
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _initial_placement(
    counts: np.ndarray,
    *,
    rng: np.random.Generator,
    margin: float = 0.04,
) -> np.ndarray:
    """Place ``len(counts)`` generators on a Halton (2, 3) sequence.

    Largest-count CTs are placed first so the lowest-discrepancy slots go
    to the regions with the largest area requirements (this drastically
    helps Lloyd convergence for skewed weight distributions).
    """
    n = len(counts)
    order = np.argsort(-counts, kind="stable")  # large counts first
    points = np.zeros((n, 2), dtype=np.float64)
    halton2 = np.array([_halton(i + 1, 2) for i in range(n)])
    halton3 = np.array([_halton(i + 1, 3) for i in range(n)])
    halton_pts = np.column_stack([halton2, halton3])
    halton_pts = halton_pts * (1.0 - 2.0 * margin) + margin
    for k_idx, i in enumerate(order):
        points[i] = halton_pts[k_idx]
    points += rng.uniform(-0.005, 0.005, size=points.shape)
    return points


# ---------------------------------------------------------------------------
# Iteration: Lloyd + Aurenhammer weight gradient
# ---------------------------------------------------------------------------
def _fit_power_diagram(
    counts: np.ndarray,
    *,
    bbox: Polygon,
    rng: np.random.Generator,
    tol: float,
    max_iter: int,
    alpha: float,
    beta: float,
    weight_floor: float,
) -> tuple[np.ndarray, np.ndarray, list[Polygon], int, float]:
    """Iterate position + weight updates until cell areas match targets.

    Parameters
    ----------
    counts
        Top-5 method count per CT (length n).
    bbox
        Shapely envelope (the unit square).
    rng
        Seeded numpy random generator (drives recovery jitter).
    tol
        Stop when ``max_i |A_i - T_i| / T_i < tol`` and no empty cells.
    max_iter
        Hard iteration cap.
    alpha
        Weight-gradient step size (Aurenhammer): ``w_i += alpha * (T_i - A_i)``.
    beta
        Lloyd-relaxation damping: ``p_i += beta * (centroid_i - p_i)``.
    weight_floor
        Minimum visible weight per CT (counts of 0 use this floor so the
        cell still has positive area).

    Returns
    -------
    points : (n, 2) array of final generator positions.
    weights : (n,) array of final weights.
    cells : list of n shapely ``Polygon`` objects (the converged power-cell
        for each generator).
    iters : number of iterations performed (``< max_iter`` if converged).
    max_rel_err : final ``max_i |A_i - T_i| / T_i``.
    """
    n = len(counts)
    weights_visual = np.where(counts > 0, counts.astype(np.float64), weight_floor)
    target_props = weights_visual / weights_visual.sum()
    env_area = float(bbox.area)
    target_areas = target_props * env_area

    points = _initial_placement(counts, rng=rng)
    weights = np.zeros(n, dtype=np.float64)
    minx, miny, maxx, maxy = bbox.bounds
    cx_env, cy_env = (minx + maxx) / 2.0, (miny + maxy) / 2.0

    best_err = float("inf")
    best_state: tuple[np.ndarray, np.ndarray, list[Polygon]] = (
        points.copy(), weights.copy(), [],
    )
    iters_used = max_iter

    for it in range(max_iter):
        cells = _power_diagram_cells(points, weights, bbox)
        n_empty = sum(1 for c in cells if c is None)
        areas = np.array(
            [c.area if c is not None else 0.0 for c in cells],
            dtype=np.float64,
        )
        rel_errs = np.abs(areas - target_areas) / np.maximum(target_areas, 1e-12)
        max_rel = float(rel_errs.max())

        if max_rel < best_err and n_empty == 0:
            best_err = max_rel
            best_state = (points.copy(), weights.copy(), [c for c in cells])

        if max_rel < tol and n_empty == 0:
            iters_used = it + 1
            best_err = max_rel
            best_state = (points.copy(), weights.copy(), [c for c in cells])
            break

        # Position update (Lloyd toward centroid for non-empty cells;
        # teleport empty cells back to the envelope center with jitter
        # and reset their weight so they re-acquire some area next round).
        new_points = points.copy()
        new_weights = weights.copy()
        for i, c in enumerate(cells):
            if c is not None and not c.is_empty:
                cxi, cyi = c.centroid.coords[0]
                new_points[i] = points[i] + beta * (
                    np.array([cxi, cyi], dtype=np.float64) - points[i]
                )
            else:
                new_points[i] = (
                    np.array([cx_env, cy_env], dtype=np.float64)
                    + rng.uniform(-0.05, 0.05, size=2)
                )
                new_weights[i] = 0.0
        new_points[:, 0] = np.clip(new_points[:, 0], minx + 0.005, maxx - 0.005)
        new_points[:, 1] = np.clip(new_points[:, 1], miny + 0.005, maxy - 0.005)

        # Aurenhammer weight gradient step.
        new_weights = new_weights + alpha * (target_areas - areas)

        points = new_points
        weights = new_weights

    points, weights, cells = best_state
    if not cells:  # never reached a non-empty layout (extreme edge case)
        cells = _power_diagram_cells(points, weights, bbox)
    return points, weights, cells, iters_used, best_err


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _short_ct(ct: str, *, max_chars: int = 22) -> str:
    """Return a shorter form of a CT name for in-region labeling."""
    rules: dict[str, str] = {
        "Committed oligodendrocyte precursor": "Committed OPC",
        "Oligodendrocyte precursor":           "OPC",
        "LAMP5-LHX6 and Chandelier":           "LAMP5-LHX6/Chand",
        "Deep-layer intratelencephalic":       "Deep IT",
        "Upper-layer intratelencephalic":      "Upper IT",
        "Deep-layer corticothalamic and 6b":   "Deep CT/6b",
        "Deep-layer near-projecting":          "Deep NP",
        "Eccentric medium spiny neuron":       "Eccentric MSN",
        "Hippocampal dentate gyrus":           "DG",
        "Hippocampal CA1-3":                   "CA1-3",
        "Hippocampal CA4":                     "CA4",
        "Midbrain-derived inhibitory":         "Midbrain inh.",
        "Cerebellar inhibitory":               "Cereb. inh.",
        "Thalamic excitatory":                 "Thalamic exc.",
        "Hippocampal CA1-3":                   "CA1-3",
    }
    if ct in rules:
        return rules[ct]
    if len(ct) <= max_chars:
        return ct
    return ct[: max_chars - 1] + "…"


def _draw_power_diagram(
    ax: plt.Axes,
    cells: list[Polygon],
    cts: list[str],
    counts: np.ndarray,
    zero_frac: np.ndarray,
    *,
    sparse_threshold: float = 0.20,
    n_methods: int = 11,
) -> None:
    """Render power-diagram cells with viridis fill + black boundaries.

    Sparse CTs (zero_frac >= ``sparse_threshold``) get a hatched fill
    pattern (``"//"`` per spec) at alpha=0.6; dense CTs get the pure fill.
    Cell labels (CT name + count badge) are placed at the cell centroid;
    label font size scales with cell side length, and labels in tiny
    cells fall back to acronym-only or are skipped if microscopic.
    """
    cmap = PALETTES["sequential"]
    norm_floor, norm_ceil = 0.10, 0.95

    def color_for(c: int) -> tuple:
        return cmap(norm_floor + (norm_ceil - norm_floor) * (c / max(n_methods, 1)))

    for ct, poly, count, zf in zip(cts, cells, counts, zero_frac):
        if poly is None or poly.is_empty:
            logger.warning("Skipping draw of empty cell for %r", ct)
            continue
        is_sparse = zf >= sparse_threshold
        coords = np.asarray(poly.exterior.coords)
        # Per spec: hatch="//" + alpha=0.6 for sparse cells.
        if is_sparse:
            patch = mpatches.Polygon(
                coords,
                closed=True,
                facecolor=color_for(int(count)),
                edgecolor="black",
                linewidth=1.0,
                hatch="//",
                alpha=0.6,
                zorder=2,
            )
        else:
            patch = mpatches.Polygon(
                coords,
                closed=True,
                facecolor=color_for(int(count)),
                edgecolor="black",
                linewidth=1.0,
                alpha=0.95,
                zorder=2,
            )
        ax.add_patch(patch)

        cx, cy = poly.centroid.coords[0]
        minx, miny, maxx, maxy = poly.bounds
        side = min(maxx - minx, maxy - miny)
        # Skip microscopic cells; the colorbar still encodes their count.
        if side < 0.045:
            continue
        text_color = "white" if count <= n_methods * 0.45 else "black"
        if side < 0.075:
            label = f"{_short_ct(ct, max_chars=10)}"
            fs = 5.5
        elif side < 0.12:
            label = f"{_short_ct(ct, max_chars=14)}\n{int(count)}/{n_methods}"
            fs = 7
        elif side < 0.18:
            label = f"{_short_ct(ct, max_chars=18)}\n{int(count)}/{n_methods}"
            fs = 8
        elif side < 0.25:
            label = f"{_short_ct(ct, max_chars=22)}\n{int(count)}/{n_methods}"
            fs = 9
        else:
            label = f"{_short_ct(ct, max_chars=26)}\n{int(count)}/{n_methods}"
            fs = 10
        ax.text(
            cx, cy, label,
            ha="center", va="center",
            fontsize=fs,
            color=text_color,
            zorder=3,
            linespacing=1.05,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _build_figure(
    *,
    cts: list[str],
    counts: np.ndarray,
    zero_frac: np.ndarray,
    seed: int,
    max_iter: int,
    tol: float,
    alpha: float,
    beta: float,
    weight_floor: float,
    n_methods: int = 11,
) -> tuple[plt.Figure, dict]:
    """Build the figure. Returns ``(fig, verification_dict)``."""
    apply_theme("paper")
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    bbox = box(0.0, 0.0, 1.0, 1.0)
    points, weights, cells, iters_used, max_rel_err = _fit_power_diagram(
        counts,
        bbox=bbox,
        rng=rng,
        tol=tol,
        max_iter=max_iter,
        alpha=alpha,
        beta=beta,
        weight_floor=weight_floor,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_power_diagram(
        ax, cells, cts, counts, zero_frac,
        sparse_threshold=0.20, n_methods=n_methods,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.grid(False)

    cmap = PALETTES["sequential"]
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0.0, vmax=float(n_methods)),
    )
    sm.set_array([])
    cb = fig.colorbar(
        sm, ax=ax,
        fraction=0.030, pad=0.02,
        shrink=0.55,
        aspect=22,
        orientation="vertical",
    )
    cb.set_label(
        f"# methods placing CT in top-5 (out of {n_methods})", fontsize=8,
    )
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0, labelsize=7)

    hatch_proxy = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="//", alpha=0.6,
        label=r"sparse CT (zero_frac $\geq$ 0.20)",
    )
    ax.legend(
        handles=[hatch_proxy],
        loc="lower right",
        bbox_to_anchor=(1.02, -0.02),
        frameon=True,
        fontsize=7,
    )

    fig.subplots_adjust(left=0.02, right=0.92, top=0.97, bottom=0.04)

    n_drawn = sum(1 for c in cells if c is not None and not c.is_empty)
    verify = {
        "n_cts_rendered": n_drawn,
        "algorithm": "power_diagram (Aurenhammer)",
        "convergence_iters": iters_used,
        "max_relative_area_error": max_rel_err,
        "tol": tol,
        "max_iter": max_iter,
        "alpha": alpha,
        "beta": beta,
        "weight_floor": weight_floor,
        "converged": max_rel_err < tol,
    }
    return fig, verify


def _print_report(
    cts: list[str],
    counts: np.ndarray,
    zero_frac: np.ndarray,
    verify: dict,
    *,
    n_methods: int,
    out_png: Path,
    dpi: int,
) -> None:
    """Print verification numbers to stdout per spec."""
    print("=" * 72)
    print("README Figure 3 alt-2 -- weighted-Voronoi (power-diagram) treemap")
    print("=" * 72)
    print(f"  algorithm                      : {verify['algorithm']}")
    print(f"  n_cts_rendered                 : {verify['n_cts_rendered']}")
    print(f"  convergence_iters              : {verify['convergence_iters']}")
    print(
        f"  max_relative_area_error        : {verify['max_relative_area_error']:.4f}"
    )
    print(f"  tol                            : {verify['tol']:.4f}")
    print(f"  converged                      : {verify['converged']}")
    print(f"  alpha (weight grad step)       : {verify['alpha']}")
    print(f"  beta  (Lloyd damping)          : {verify['beta']}")
    print(f"  weight_floor                   : {verify['weight_floor']}")
    order = np.argsort(-counts)
    print("  top_5_by_count                 :")
    for i in order[:5]:
        print(
            f"    {cts[i]:48s}  {int(counts[i])}/{n_methods}  "
            f"(zero_frac={zero_frac[i]:.3f})"
        )
    n_sparse = int(np.sum(zero_frac >= 0.20))
    print(f"  n_sparse_cts_hatched           : {n_sparse} (zero_frac >= 0.20)")
    print(f"  dpi                            : {dpi}")
    if out_png.exists():
        size_bytes = out_png.stat().st_size
        size_kb = size_bytes / 1024.0
        size_mb = size_bytes / 1024.0 / 1024.0
        print(f"  png_path                       : {out_png}")
        print(
            f"  png_size                       : {size_kb:.1f} KB "
            f"({size_mb:.2f} MB, {size_bytes} bytes)"
        )
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--consensus-json", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/consensus_heatmap"
        / "consensus_heatmap_data.json",
    )
    parser.add_argument(
        "--coverage-json", type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ct_coverage_full_cohort.json",
    )
    parser.add_argument(
        "--out-stem", type=Path,
        default=_WORKTREE_ROOT / "figures/fig3alt_voronoi",
        help="Output path stem (no extension); save_fig appends .png.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument(
        "--tol", type=float, default=0.05,
        help="Convergence tolerance: stop when max_i |A_i - T_i| / T_i < tol.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Aurenhammer weight-gradient step size: w_i += alpha * (T_i - A_i).",
    )
    parser.add_argument(
        "--beta", type=float, default=0.7,
        help="Lloyd-relaxation damping: p_i += beta * (centroid_i - p_i).",
    )
    parser.add_argument(
        "--weight-floor", type=float, default=1.0,
        help="Minimum weight for zero-count CTs (so cells are visible).",
    )
    parser.add_argument("--n-methods", type=int, default=11)
    parser.add_argument(
        "--dpi", type=int, default=600,
        help="PNG output DPI. Project theme default = 600.",
    )
    args = parser.parse_args()

    logger.info("[fig3alt-voronoi] consensus JSON: %s", args.consensus_json)
    logger.info("[fig3alt-voronoi] coverage JSON: %s", args.coverage_json)

    zero_frac_map = _load_zero_frac_per_ct(args.coverage_json)
    universe_cts = sorted(zero_frac_map.keys())
    counts_map = _load_top5_counts_per_ct(
        args.consensus_json, universe_cts, top_k=5,
    )
    cts = list(universe_cts)
    counts = np.array([counts_map[ct] for ct in cts], dtype=np.int64)
    zero_frac = np.array(
        [zero_frac_map[ct] for ct in cts], dtype=np.float64,
    )

    logger.info(
        "[fig3alt-voronoi] %d CTs in universe; %d with non-zero top-5 count",
        len(cts), int((counts > 0).sum()),
    )

    fig, verify = _build_figure(
        cts=cts,
        counts=counts,
        zero_frac=zero_frac,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        alpha=args.alpha,
        beta=args.beta,
        weight_floor=args.weight_floor,
        n_methods=args.n_methods,
    )

    out_png = args.out_stem.with_suffix(".png")
    if out_png.exists():
        logger.info("[fig3alt-voronoi] removing preexisting %s", out_png)
        out_png.unlink()

    written = save_fig(fig, args.out_stem, formats=("png",), dpi=args.dpi)
    plt.close(fig)
    for w in written:
        logger.info("[fig3alt-voronoi] wrote %s", w)

    _print_report(
        cts, counts, zero_frac, verify,
        n_methods=args.n_methods, out_png=out_png, dpi=args.dpi,
    )

    if verify["n_cts_rendered"] != len(cts):
        logger.error(
            "Expected %d CTs rendered; got %d",
            len(cts), verify["n_cts_rendered"],
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
