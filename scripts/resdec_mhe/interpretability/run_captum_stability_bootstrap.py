#!/usr/bin/env python
"""Subject-level bootstrap stability analysis for Captum IG top-gene rankings.

Background
----------
The canonical Captum Integrated Gradients composite attribution
(``composite_attributions.npz``, shape ``[516, 31, 4785]`` per-subject signed)
yields a top-K gene ranking per cell type by computing
``mean_abs_attribution[c, g] = mean_n |attr[n, c, g]|`` and sorting along the
gene axis.  A natural reviewer concern is whether these rankings are stable to
subject resampling: if you remove ~5% of the cohort, do the leaderboards stay
the same, or do they shuffle?  This script answers that question via subject-
level bootstrap-with-replacement and reports per-CT, per-gene stability.

Method
------
1. Load the per-subject Captum attribution tensor ``attr`` (shape
   ``[N=516, C=31, G=4785]``) and apply ``np.abs(attr)`` element-wise (the
   saved tensor is RAW signed; the producer aggregates to ``mean(|·|)`` only
   inside the summary JSON — see ``captum_composite_attribution.py`` lines
   228-232).  Call this ``A``.
2. Compute the canonical per-CT importance ``A.mean(axis=0)``, shape
   ``[C, G]``.  Per-CT top-K = ``argsort(-importance[c])[:K]``.  We use K=50.
3. For each bootstrap iteration b = 1..B (default B=1000):
     a. Draw indices ``idx ~ Uniform({0, .., N-1})`` of size N WITH
        replacement, using ``np.random.default_rng(seed=42)``.
     b. Compute ``boot_imp[c, g] = mean_{n in idx} A[n, c, g]``, equivalent
        to ``A[idx].mean(axis=0)``.
     c. Per-CT top-K via argsort along the gene axis.
4. For each canonical top-K (CT, gene) pair, the **inclusion frequency** is
   the fraction of bootstraps in which that gene also lies in the bootstrap
   top-K for the same CT.  We additionally record the median rank (with IQR)
   across bootstraps in which the gene is included.
5. Per-CT stability score = mean inclusion frequency over the canonical
   top-K.  "Rock-solid" pairs have inclusion frequency ≥ 0.95; "fragile"
   pairs have inclusion frequency < 0.5.

Reproducibility
---------------
We use ``numpy.random.default_rng(seed=42)`` (PCG64) everywhere; a single RNG
streams all B index draws.  The full bootstrap is therefore fully
deterministic for given (B, seed, K).

Outputs
-------
* ``--out-json`` (default
  ``outputs/canonical/interpretability/captum_stability_bootstrap.json``)
* ``--out-md`` (default
  ``outputs/canonical/interpretability/captum_stability_bootstrap.md``)
* ``--out-fig-dir`` (default
  ``outputs/canonical/interpretability/figures/captum_stability/``)
    fig_captum_stability_bootstrap.{png,pdf}  (2-panel, 600 DPI)

Usage
-----
    PYTHONPATH=<worktree-root> uv run python \
        scripts/resdec_mhe/interpretability/run_captum_stability_bootstrap.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.utils.provenance import git_sha  # noqa: E402
from src.visualization.theme import apply_theme, fmt_axes  # noqa: E402

logger = logging.getLogger(__name__)


# =============================================================================
# Constants — surfaced here so a reviewer doesn't have to spelunk
# =============================================================================

DEFAULT_TOP_K = 50            # canonical top-K per CT
DEFAULT_N_BOOT = 1000         # bootstrap draws
DEFAULT_SEED = 42
ROCK_SOLID_THRESHOLD = 0.95   # inclusion freq ≥ this → "rock-solid"
FRAGILE_THRESHOLD = 0.50      # inclusion freq <  this → "fragile"
SPLATTER_CT_NAME = "Splatter"  # focus CT for panel B


# =============================================================================
# Data loading
# =============================================================================


def load_attribution_tensor(npz_path: Path) -> np.ndarray:
    """Load the Captum IG attribution tensor (raw, signed).

    Returns ``attr`` of shape ``(N, C, G)``, float32.  Caller is responsible
    for taking ``np.abs(attr)`` before aggregating.
    """
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    if "attributions" not in d.files:
        raise KeyError(
            f"{npz_path}: missing required key 'attributions' "
            f"(found: {list(d.files)})"
        )
    attr = np.asarray(d["attributions"]).astype(np.float32)
    if attr.ndim != 3:
        raise ValueError(
            f"attributions must be 3D [N, C, G]; got shape {attr.shape}"
        )
    return attr


def load_gene_names(precomputed_dir: Path, n_genes: int) -> list[str]:
    """Load gene-name vector of length ``n_genes`` from the precomputed sidecar.

    Falls back to ``gene_<i>`` placeholders if no sidecar is present.  Mirrors
    the contract of ``run_sex_disparity_attribution.load_gene_names``.
    """
    candidates = [
        precomputed_dir / "gene_names.npy",
        precomputed_dir / "gene_names.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".npy":
            names = np.load(p, allow_pickle=True).tolist()
        else:
            names = json.loads(p.read_text())
        if isinstance(names, list) and len(names) >= n_genes:
            return [str(n) for n in names[:n_genes]]
    logger.warning(
        "No gene_names sidecar found in %s; using gene_<i> placeholders.",
        precomputed_dir,
    )
    return [f"gene_{i}" for i in range(n_genes)]


# =============================================================================
# Statistical primitives
# =============================================================================


def canonical_top_k_per_ct(
    abs_attr: np.ndarray, top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical per-CT top-K gene indices and their importance values.

    Parameters
    ----------
    abs_attr : (N, C, G) ndarray of |attr| (already abs-applied).
    top_k : K, e.g. 50.

    Returns
    -------
    top_idx : (C, K) int — gene indices ranked descending by mean(|attr|).
    top_imp : (C, K) float — the corresponding mean(|attr|) values.
    """
    if abs_attr.ndim != 3:
        raise ValueError(f"abs_attr must be 3D; got shape {abs_attr.shape}")
    n, c_dim, g_dim = abs_attr.shape
    if top_k <= 0 or top_k > g_dim:
        raise ValueError(f"top_k must satisfy 0 < K ≤ G={g_dim}; got {top_k}")
    importance = abs_attr.mean(axis=0)  # (C, G)
    # argsort ascending → take the last K and reverse → descending order
    order = np.argsort(importance, axis=1)[:, -top_k:][:, ::-1]
    top_imp = np.take_along_axis(importance, order, axis=1)
    return order.astype(np.int64), top_imp.astype(np.float64)


def bootstrap_inclusion_and_ranks(
    abs_attr: np.ndarray,
    canonical_top_idx: np.ndarray,
    *,
    n_boot: int,
    top_k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[list[list[int]]]]:
    """Run subject-level bootstrap and return inclusion + rank matrices.

    For each bootstrap b, draw N indices with replacement, recompute per-CT
    importance, and find the bootstrap top-K per CT.  For each canonical
    (CT, rank) pair, record:

      * whether the canonical gene is in the bootstrap top-K,
      * if so, the bootstrap rank (0 = best, K-1 = worst).

    Parameters
    ----------
    abs_attr : (N, C, G) ndarray.
    canonical_top_idx : (C, K) int — canonical top-K gene indices per CT.
    n_boot : number of bootstrap draws.
    top_k : K (must match canonical_top_idx.shape[1]).
    rng : ``numpy.random.Generator``.

    Returns
    -------
    inclusion : (C, K) float — inclusion frequency of each canonical gene
        across the n_boot bootstraps.
    boot_ranks : list of length C; each entry is a list of length K; each
        entry is a list of bootstrap ranks (0..K-1) for the bootstraps in
        which the gene was in the bootstrap top-K.  Used for median + IQR
        downstream.
    """
    n, c_dim, g_dim = abs_attr.shape
    if canonical_top_idx.shape != (c_dim, top_k):
        raise ValueError(
            f"canonical_top_idx shape {canonical_top_idx.shape} ≠ "
            f"({c_dim}, {top_k})"
        )

    # Inclusion counts per (CT, canonical rank-r) pair.
    incl_counts = np.zeros((c_dim, top_k), dtype=np.int64)
    # Per-(CT, canonical rank): list of bootstrap ranks across hits.
    boot_ranks: list[list[list[int]]] = [
        [[] for _ in range(top_k)] for _ in range(c_dim)
    ]

    # Vectorize over CTs within each bootstrap iteration.  Per iteration:
    #   1. Sample N indices.
    #   2. Compute boot_imp[c, g] = abs_attr[idx].mean(axis=0).  This is
    #      (N, C, G) sum/N along axis 0 — O(N * C * G) per iter.
    #   3. argsort along gene axis once per CT to get bootstrap top-K,
    #      then use np.isin against canonical[c] to record inclusions.
    log_every = max(1, n_boot // 20)
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n, endpoint=False)
        boot_imp = abs_attr[idx].mean(axis=0)  # (C, G)
        # Per CT: argpartition for top-K, then refine with sort within K.
        # argpartition on -boot_imp gives the indices of the K largest
        # values in any order in the first K slots; we then sort that slice
        # by descending importance to get the actual rank.
        # argpartition kth: smallest K of (-boot_imp) → largest K of boot_imp.
        part = np.argpartition(-boot_imp, kth=top_k - 1, axis=1)[:, :top_k]
        part_vals = np.take_along_axis(-boot_imp, part, axis=1)
        order_within = np.argsort(part_vals, axis=1)
        boot_top_idx = np.take_along_axis(part, order_within, axis=1)  # (C, K)

        # Convert to per-CT lookup: bootstrap rank for each canonical gene.
        # For each c, build a dict {gene_idx: rank}; check membership.
        for c in range(c_dim):
            boot_top_c = boot_top_idx[c]  # (K,)
            # Map gene -> rank via positional indexing.  Since K is small
            # (50), a Python loop here is cheap relative to the matmul
            # already performed.
            pos_map = {int(g): int(r) for r, g in enumerate(boot_top_c)}
            canon_c = canonical_top_idx[c]
            for r_canon, g_canon in enumerate(canon_c):
                rank_b = pos_map.get(int(g_canon))
                if rank_b is not None:
                    incl_counts[c, r_canon] += 1
                    boot_ranks[c][r_canon].append(rank_b)

        if (b + 1) % log_every == 0 or b == n_boot - 1:
            elapsed = time.time() - t0
            logger.info(
                "bootstrap %d/%d (%.1fs elapsed, %.2fs/iter)",
                b + 1, n_boot, elapsed, elapsed / (b + 1),
            )

    inclusion = incl_counts.astype(np.float64) / n_boot
    return inclusion, boot_ranks


def median_iqr(ranks: list[int]) -> tuple[float, float, float]:
    """Return (median, Q1, Q3) of a list of integer ranks.

    Returns ``(nan, nan, nan)`` if the list is empty.  IQR = Q3 - Q1 is left
    to the caller.
    """
    if len(ranks) == 0:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(ranks, dtype=np.float64)
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
    )


# =============================================================================
# JSON / MD assembly
# =============================================================================


def build_payload(
    *,
    canonical_top_idx: np.ndarray,
    canonical_top_imp: np.ndarray,
    inclusion: np.ndarray,
    boot_ranks: list[list[list[int]]],
    ct_names: list[str],
    gene_names: list[str],
    n_subjects: int,
    n_boot: int,
    top_k: int,
    seed: int,
    git: str,
) -> dict:
    """Assemble the JSON-serializable payload."""
    c_dim, k_dim = canonical_top_idx.shape
    if k_dim != top_k:
        raise ValueError(
            f"canonical_top_idx K mismatch: {k_dim} vs {top_k}"
        )

    per_ct: list[dict] = []
    for c in range(c_dim):
        gene_records = []
        for r in range(top_k):
            g = int(canonical_top_idx[c, r])
            med, q1, q3 = median_iqr(boot_ranks[c][r])
            gene_records.append({
                "rank_canonical": r,
                "gene_idx": g,
                "gene": gene_names[g],
                "mean_abs_attribution": float(canonical_top_imp[c, r]),
                "inclusion_frequency": float(inclusion[c, r]),
                "n_inclusions": int(round(inclusion[c, r] * n_boot)),
                "boot_rank_median": med,
                "boot_rank_q1": q1,
                "boot_rank_q3": q3,
                "boot_rank_iqr": (
                    float("nan") if (np.isnan(q3) or np.isnan(q1)) else q3 - q1
                ),
                "rock_solid": bool(inclusion[c, r] >= ROCK_SOLID_THRESHOLD),
                "fragile": bool(inclusion[c, r] < FRAGILE_THRESHOLD),
            })
        per_ct.append({
            "cell_type_idx": c,
            "cell_type": ct_names[c],
            "stability_score": float(np.mean(inclusion[c])),
            "n_rock_solid": int(np.sum(inclusion[c] >= ROCK_SOLID_THRESHOLD)),
            "n_fragile": int(np.sum(inclusion[c] < FRAGILE_THRESHOLD)),
            "top_genes": gene_records,
        })

    # Cross-CT pair lists ranked by inclusion frequency.
    all_pairs = []
    for c in range(c_dim):
        for r in range(top_k):
            g = int(canonical_top_idx[c, r])
            all_pairs.append({
                "cell_type": ct_names[c],
                "cell_type_idx": c,
                "rank_canonical": r,
                "gene": gene_names[g],
                "gene_idx": g,
                "mean_abs_attribution": float(canonical_top_imp[c, r]),
                "inclusion_frequency": float(inclusion[c, r]),
            })
    most_stable = sorted(
        all_pairs,
        key=lambda p: (-p["inclusion_frequency"], -p["mean_abs_attribution"]),
    )[:10]
    fragile_pairs = sorted(
        [p for p in all_pairs if p["inclusion_frequency"] < FRAGILE_THRESHOLD],
        key=lambda p: (p["inclusion_frequency"], -p["mean_abs_attribution"]),
    )[:10]

    overall_mean_stability = float(
        np.mean([entry["stability_score"] for entry in per_ct])
    )
    n_rock_solid_total = int(sum(e["n_rock_solid"] for e in per_ct))
    n_fragile_total = int(sum(e["n_fragile"] for e in per_ct))
    most_stable_ct = max(per_ct, key=lambda e: e["stability_score"])
    least_stable_ct = min(per_ct, key=lambda e: e["stability_score"])

    payload = {
        "config": {
            "n_subjects": int(n_subjects),
            "n_cell_types": int(c_dim),
            "n_genes_total": int(len(gene_names)),
            "top_k": int(top_k),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "rock_solid_threshold": ROCK_SOLID_THRESHOLD,
            "fragile_threshold": FRAGILE_THRESHOLD,
            "git_sha": git,
        },
        "summary": {
            "mean_stability_across_cts": overall_mean_stability,
            "n_rock_solid_pairs": n_rock_solid_total,
            "n_fragile_pairs": n_fragile_total,
            "n_pairs_total": int(c_dim * top_k),
            "most_stable_ct": {
                "cell_type": most_stable_ct["cell_type"],
                "stability_score": most_stable_ct["stability_score"],
            },
            "least_stable_ct": {
                "cell_type": least_stable_ct["cell_type"],
                "stability_score": least_stable_ct["stability_score"],
            },
        },
        "per_cell_type": per_ct,
        "top_10_most_stable_pairs": most_stable,
        "top_10_fragile_pairs": fragile_pairs,
    }
    return payload


def render_md(payload: dict) -> str:
    """Render the JSON payload as a human-readable Markdown report."""
    cfg = payload["config"]
    summ = payload["summary"]
    lines: list[str] = []
    lines.append("# Captum IG Top-Gene Stability Bootstrap")
    lines.append("")
    lines.append("Subject-level bootstrap-with-replacement of the per-CT "
                 "top-K (K = "
                 f"{cfg['top_k']}) gene leaderboards from the "
                 "ResDec-MHE composite Captum Integrated Gradients tensor.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Subjects (N): {cfg['n_subjects']}")
    lines.append(f"- Cell types (C): {cfg['n_cell_types']}")
    lines.append(f"- Genes (G): {cfg['n_genes_total']}")
    lines.append(f"- Top-K per CT: {cfg['top_k']}")
    lines.append(f"- Bootstrap draws: {cfg['n_boot']}")
    lines.append(f"- RNG seed: {cfg['seed']} (numpy.random.default_rng PCG64)")
    lines.append(
        f"- Rock-solid threshold: ≥ {cfg['rock_solid_threshold']}; "
        f"fragile threshold: < {cfg['fragile_threshold']}"
    )
    lines.append(f"- Git SHA: `{cfg['git_sha']}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Mean per-CT stability score across "
        f"{cfg['n_cell_types']} CTs: **{summ['mean_stability_across_cts']:.4f}**"
    )
    lines.append(
        f"- Rock-solid (CT, gene) pairs (inclusion ≥ "
        f"{cfg['rock_solid_threshold']}): "
        f"**{summ['n_rock_solid_pairs']}** of {summ['n_pairs_total']}"
    )
    lines.append(
        f"- Fragile (CT, gene) pairs (inclusion < "
        f"{cfg['fragile_threshold']}): "
        f"**{summ['n_fragile_pairs']}** of {summ['n_pairs_total']}"
    )
    lines.append(
        f"- Most-stable CT: **{summ['most_stable_ct']['cell_type']}** "
        f"(score = {summ['most_stable_ct']['stability_score']:.4f})"
    )
    lines.append(
        f"- Least-stable CT: **{summ['least_stable_ct']['cell_type']}** "
        f"(score = {summ['least_stable_ct']['stability_score']:.4f})"
    )
    lines.append("")

    lines.append("## Per-cell-type stability scores")
    lines.append("")
    lines.append(
        "| Rank | Cell type | Stability score | "
        "n rock-solid | n fragile |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    sorted_ct = sorted(
        payload["per_cell_type"],
        key=lambda e: -e["stability_score"],
    )
    for i, entry in enumerate(sorted_ct, 1):
        lines.append(
            f"| {i} | {entry['cell_type']} | "
            f"{entry['stability_score']:.4f} | "
            f"{entry['n_rock_solid']} | {entry['n_fragile']} |"
        )
    lines.append("")

    lines.append("## Top-10 most-stable (CT, gene) pairs")
    lines.append("")
    lines.append(
        "| # | Cell type | Gene | Canonical rank | "
        "Inclusion freq | mean(|attr|) |"
    )
    lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
    for i, p in enumerate(payload["top_10_most_stable_pairs"], 1):
        lines.append(
            f"| {i} | {p['cell_type']} | {p['gene']} | "
            f"{p['rank_canonical']} | {p['inclusion_frequency']:.4f} | "
            f"{p['mean_abs_attribution']:.6f} |"
        )
    lines.append("")

    lines.append("## Top-10 fragile (CT, gene) pairs (inclusion < "
                 f"{cfg['fragile_threshold']})")
    lines.append("")
    if not payload["top_10_fragile_pairs"]:
        lines.append("_None — all canonical top-K pairs cleared the "
                     "fragile threshold._")
    else:
        lines.append(
            "| # | Cell type | Gene | Canonical rank | "
            "Inclusion freq | mean(|attr|) |"
        )
        lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
        for i, p in enumerate(payload["top_10_fragile_pairs"], 1):
            lines.append(
                f"| {i} | {p['cell_type']} | {p['gene']} | "
                f"{p['rank_canonical']} | "
                f"{p['inclusion_frequency']:.4f} | "
                f"{p['mean_abs_attribution']:.6f} |"
            )
    lines.append("")

    return "\n".join(lines) + "\n"


# =============================================================================
# Figure
# =============================================================================


def make_figure(
    payload: dict,
    out_dir: Path,
    *,
    focus_ct: str = SPLATTER_CT_NAME,
    dpi: int = 600,
) -> tuple[Path, Path]:
    """Write the 2-panel stability figure (PNG + PDF, 600 DPI)."""
    apply_theme(style="paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ct = payload["per_cell_type"]
    sorted_ct = sorted(per_ct, key=lambda e: -e["stability_score"])
    ct_labels = [e["cell_type"] for e in sorted_ct]
    ct_scores = [e["stability_score"] for e in sorted_ct]

    # Panel B: focus CT (Splatter by default if present else most-stable CT).
    focus = next((e for e in per_ct if e["cell_type"] == focus_ct), None)
    if focus is None:
        focus = sorted_ct[0]

    # Subplot mosaic: A wide on top, B wide on bottom.
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 13), gridspec_kw={"height_ratios": [1.0, 1.4]},
    )
    ax_a, ax_b = axes

    # ---------- Panel A: per-CT mean stability score ----------
    y_pos_a = np.arange(len(ct_labels))
    bars_a = ax_a.barh(y_pos_a, ct_scores, color="#3b6cb7", edgecolor="black",
                       linewidth=0.4)
    ax_a.set_yticks(y_pos_a)
    ax_a.set_yticklabels(ct_labels, fontsize=8)
    ax_a.invert_yaxis()
    ax_a.axvline(ROCK_SOLID_THRESHOLD, color="#2e7d32", linestyle="--",
                 linewidth=1.0, alpha=0.7,
                 label=f"rock-solid ≥ {ROCK_SOLID_THRESHOLD}")
    ax_a.axvline(FRAGILE_THRESHOLD, color="#c62828", linestyle=":",
                 linewidth=1.0, alpha=0.7,
                 label=f"fragile < {FRAGILE_THRESHOLD}")
    ax_a.set_xlabel("Per-CT mean inclusion frequency over canonical top-"
                    f"{payload['config']['top_k']} genes")
    ax_a.set_title("A. Per-cell-type stability score (subject-level "
                   f"bootstrap, B = {payload['config']['n_boot']})",
                   loc="left", fontsize=11, fontweight="bold")
    ax_a.set_xlim(0.0, 1.02)
    ax_a.legend(loc="lower right", fontsize=8, frameon=True)
    fmt_axes(ax_a)
    for bar, score in zip(bars_a, ct_scores):
        ax_a.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2.0,
            f"{score:.3f}", va="center", fontsize=7,
        )

    # ---------- Panel B: focus-CT canonical-rank inclusion freq ----------
    genes = focus["top_genes"]
    gene_labels = [f"{g['rank_canonical'] + 1}. {g['gene']}" for g in genes]
    incl = [g["inclusion_frequency"] for g in genes]
    y_pos_b = np.arange(len(gene_labels))
    colors = [
        "#2e7d32" if v >= ROCK_SOLID_THRESHOLD
        else ("#c62828" if v < FRAGILE_THRESHOLD else "#8a8a8a")
        for v in incl
    ]
    ax_b.barh(y_pos_b, incl, color=colors, edgecolor="black", linewidth=0.4)
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(gene_labels, fontsize=7)
    ax_b.invert_yaxis()
    ax_b.axvline(ROCK_SOLID_THRESHOLD, color="#2e7d32", linestyle="--",
                 linewidth=1.0, alpha=0.7)
    ax_b.axvline(FRAGILE_THRESHOLD, color="#c62828", linestyle=":",
                 linewidth=1.0, alpha=0.7)
    ax_b.set_xlim(0.0, 1.02)
    ax_b.set_xlabel("Inclusion frequency across "
                    f"{payload['config']['n_boot']} bootstraps")
    ax_b.set_title(
        f"B. {focus['cell_type']} top-{len(genes)} canonical genes — "
        "rank-1 (top) to rank-50 (bottom)",
        loc="left", fontsize=11, fontweight="bold",
    )
    fmt_axes(ax_b)

    fig.tight_layout()
    png_path = out_dir / "fig_captum_stability_bootstrap.png"
    pdf_path = out_dir / "fig_captum_stability_bootstrap.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attributions-npz",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_ig"
        / "composite_attributions.npz",
        help="Per-subject Captum IG NPZ (default: canonical).",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=Path,
        default=_WORKTREE_ROOT / "data/precomputed",
        help="Directory containing gene_names.npy.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability"
        / "captum_stability_bootstrap.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability"
        / "captum_stability_bootstrap.md",
    )
    parser.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/captum_stability",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--focus-ct", type=str, default=SPLATTER_CT_NAME,
        help="Cell-type for figure panel B (default: Splatter).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading attribution tensor: %s", args.attributions_npz)
    attr = load_attribution_tensor(args.attributions_npz)
    n, c_dim, g_dim = attr.shape
    logger.info("attributions shape = (N=%d, C=%d, G=%d)", n, c_dim, g_dim)

    if c_dim != len(CELL_TYPE_ORDER):
        raise ValueError(
            f"NPZ has {c_dim} CTs but CELL_TYPE_ORDER has "
            f"{len(CELL_TYPE_ORDER)} — schema mismatch"
        )
    ct_names = list(CELL_TYPE_ORDER)
    gene_names = load_gene_names(args.precomputed_dir, g_dim)

    logger.info("Computing |attr| and canonical top-%d per CT...", args.top_k)
    abs_attr = np.abs(attr)
    canon_idx, canon_imp = canonical_top_k_per_ct(abs_attr, args.top_k)

    logger.info(
        "Bootstrap: B=%d × N=%d × C=%d × top-K=%d, seed=%d",
        args.n_boot, n, c_dim, args.top_k, args.seed,
    )
    rng = np.random.default_rng(args.seed)
    inclusion, boot_ranks = bootstrap_inclusion_and_ranks(
        abs_attr, canon_idx,
        n_boot=args.n_boot, top_k=args.top_k, rng=rng,
    )

    payload = build_payload(
        canonical_top_idx=canon_idx,
        canonical_top_imp=canon_imp,
        inclusion=inclusion,
        boot_ranks=boot_ranks,
        ct_names=ct_names,
        gene_names=gene_names,
        n_subjects=n,
        n_boot=args.n_boot,
        top_k=args.top_k,
        seed=args.seed,
        git=git_sha(_WORKTREE_ROOT),
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_md.write_text(render_md(payload))
    logger.info("Wrote JSON: %s", args.out_json)
    logger.info("Wrote MD : %s", args.out_md)

    png, pdf = make_figure(payload, args.out_fig_dir, focus_ct=args.focus_ct)
    logger.info("Wrote PNG: %s", png)
    logger.info("Wrote PDF: %s", pdf)

    summ = payload["summary"]
    logger.info(
        "DONE. mean stability=%.4f, rock-solid=%d/%d, fragile=%d/%d, "
        "most-stable CT=%s (%.4f), least-stable CT=%s (%.4f)",
        summ["mean_stability_across_cts"],
        summ["n_rock_solid_pairs"], summ["n_pairs_total"],
        summ["n_fragile_pairs"], summ["n_pairs_total"],
        summ["most_stable_ct"]["cell_type"],
        summ["most_stable_ct"]["stability_score"],
        summ["least_stable_ct"]["cell_type"],
        summ["least_stable_ct"]["stability_score"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
