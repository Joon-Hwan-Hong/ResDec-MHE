#!/usr/bin/env python
"""Cross-method gene-level agreement: pairwise Jaccard + multi-way intersection.

Quantifies gene-level cross-method agreement at the manuscript level for the
EXP-016 cross-method consensus. EXP-034 Panel D limited the comparison to four
methods (Captum IG / Wasserstein / DE-Wilcoxon / DE-DESeq2) on the Splatter CT;
this experiment expands to **all gene-rankable methods used in the EXP-016
consensus**, computes pairwise Jaccard for every method pair, and tallies
multi-way intersections.

# Methods included

The EXP-016 cross-method consensus enumerates 11 methods, but five of them are
**cell-type-only rankings** with no gene-axis output: AttnLRP, GMAR, GAF AF,
GAF AGF, GAF GF (per
``outputs/canonical/interpretability/attention_attribution/per_subject_attribution.npz``
which has shape ``[516 subjects × 31 cell-types]`` for each of those five
keys; no gene dim). Two further EXP-016 methods (LOCO zero-out, raw-pseudobulk
CMI) are likewise CT-only and are explicitly excluded by the task brief.

That leaves **six gene-rankable methods**:

1. **Captum IG** — per-CT top-50 from ``captum_ig/composite_attribution_summary.json``
2. **GradientSHAP** — per-CT top-50 from ``captum_robustness/gradientshap_attribution_summary.json``
3. **SmoothGrad** — per-CT top-50 from ``captum_robustness/smoothgrad_attribution_summary.json``
4. **Wasserstein** — per-CT top-10 only (the canonical pseudobulk JSON caps the
   per-CT list at ``wasserstein_per_gene_top10``; this is the same cap flagged
   in EXP-034 Panel D)
5. **DE Wilcoxon** — per-CT top-50 by p-value from ``de_resilient_vs_vulnerable/CT_NN_de.csv``
6. **DE DESeq2** — per-CT top-50 by p-value from ``de_resilient_vs_vulnerable_deseq2/CT_NN_de.csv``

For each method we build the **union of the per-CT top-K sets across all 31
cell types** as the method's gene set; this is the natural unit because every
source method ranks within each CT (gradient-attribution methods aggregate to
``top_genes_per_cell_type``; DE / Wasserstein are inherently per-CT). The
six-method matrix is therefore 6×6 (15 unique upper-triangle pairs); we
preserve the 11-method label in script names + outputs to anchor the EXP-016
mapping, and the JSON / MD report explicitly enumerates the five missing
gene-non-rankable methods.

# Computations

1. Build six gene sets ``S_method`` = union of per-CT top-K members.
2. Pairwise Jaccard for all 15 pairs:
   ``J(A, B) = |A∩B| / |A∪B|`` (returns ``0.0`` for empty union; both empty
   sets gives 0.0 by convention here).
3. Multi-way intersection histogram: for each gene observed in ≥1 method,
   count how many methods support it; tally counts for ≥1, ≥2, ..., ≥M
   methods (where M = number of methods loaded).
4. Top-N consensus genes: list every gene supported by ≥6 / N (the strictest
   threshold) along with which methods support it.

# Outputs

- ``--out-json`` / ``--out-md`` — JSON / MD report containing the matrix,
  histogram, and consensus-gene list
- ``--out-fig-dir`` — 2-panel figure (Jaccard heatmap with hierarchical
  clustering + multi-way intersection bar chart) at 600 DPI PNG + PDF

# CLI

    PYTHONPATH=<worktree-root> uv run python \\
        scripts/resdec_mhe/interpretability/run_11method_gene_jaccard.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme, fmt_axes, style_paper_axes  # noqa: E402

logger = logging.getLogger(__name__)

# Per-CT top-K cap for the union construction. Wasserstein top-K is hard-capped
# at 10 by the canonical JSON schema (``wasserstein_per_gene_top10``); the
# other five gene-rankable methods support top-50.
TOP_K_GENE_RANKABLE = 50
TOP_K_WASSERSTEIN = 10

# The five EXP-016 methods that do NOT have gene-axis output; documented for
# transparency in the JSON / MD report.
EXP_016_CT_ONLY_METHODS = (
    "AttnLRP", "GMAR", "GAF AF", "GAF AGF", "GAF GF",
)

# The two EXP-016 methods explicitly excluded by the task brief (also CT-only).
EXP_016_TASK_EXCLUDED = (
    "LOCO zero-out", "raw-pseudobulk CMI",
)


def _captum_per_ct_top_k(
    summary_path: Path, top_k: int = TOP_K_GENE_RANKABLE,
) -> dict[str, set[str]]:
    """Return ``{cell_type: top-k gene set}`` from a Captum-family summary JSON.

    The Captum IG / GradientSHAP / SmoothGrad summaries share the exact same
    schema (the same code path emits all three): each
    ``top_genes_per_cell_type[cell_type]`` is a list of ``{gene,
    mean_abs_attribution}`` records sorted descending. We keep at most the
    first ``top_k``; if the source list is shorter we keep what is there
    (which is why the loader returns a set of length ``min(top_k, len)``
    rather than asserting an exact size).
    """
    summary = json.loads(Path(summary_path).read_text())
    blocks = summary["top_genes_per_cell_type"]
    return {
        ct: {entry["gene"] for entry in records[:top_k]}
        for ct, records in blocks.items()
    }


def _wasserstein_per_ct_top_k(
    json_path: Path, top_k: int = TOP_K_WASSERSTEIN,
) -> dict[str, set[str]]:
    """Return ``{cell_type: top-k gene set}`` from the Wasserstein per-CT JSON.

    Schema: ``per_cell_type`` is a list of records with ``cell_type`` and
    ``wasserstein_per_gene_top10`` = list of ``[gene, value]``. The list is
    capped at 10 upstream so ``top_k > 10`` is silently truncated to 10
    (and the report makes the cap explicit).
    """
    summary = json.loads(Path(json_path).read_text())
    out: dict[str, set[str]] = {}
    for block in summary["per_cell_type"]:
        ct = block["cell_type"]
        pairs = block.get("wasserstein_per_gene_top10", [])
        out[ct] = {pair[0] for pair in pairs[:top_k]}
    return out


def _resolve_ct_index(de_summary_csv: Path) -> dict[str, int]:
    """Map ``cell_type -> cell_type_index`` from per-CT summary CSV."""
    out: dict[str, int] = {}
    with open(de_summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[r["cell_type"]] = int(r["cell_type_index"])
    return out


def _de_per_ct_top_k(
    de_dir: Path, top_k: int = TOP_K_GENE_RANKABLE,
) -> dict[str, set[str]]:
    """Return ``{cell_type: top-k gene set by p_value}`` from a DE directory.

    Reads ``per_ct_summary.csv`` for the cell_type → index mapping, then for
    each (CT, index) pair reads ``CT_NN_de.csv`` (~4785 rows: gene,
    log2_fold_change, p_value, padj_fdr, ...), sorts ascending by p_value,
    and keeps the first ``top_k`` gene names. Any row with non-finite
    p_value is dropped.
    """
    de_dir = Path(de_dir)
    summary_csv = de_dir / "per_ct_summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"DE summary CSV missing: {summary_csv}")
    ct_index = _resolve_ct_index(summary_csv)
    out: dict[str, set[str]] = {}
    for ct, idx in ct_index.items():
        ct_csv = de_dir / f"CT_{idx:02d}_de.csv"
        if not ct_csv.is_file():
            out[ct] = set()
            continue
        rows: list[tuple[str, float]] = []
        with open(ct_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gene = r.get("gene")
                if not gene:
                    continue
                try:
                    p = float(r["p_value"])
                except (KeyError, ValueError, TypeError):
                    continue
                if not np.isfinite(p):
                    continue
                rows.append((gene, p))
        rows.sort(key=lambda kv: kv[1])
        out[ct] = {g for g, _ in rows[:top_k]}
    return out


def per_ct_to_union(per_ct: dict[str, set[str]]) -> set[str]:
    """Union of per-CT gene sets into a single method-level set."""
    out: set[str] = set()
    for genes in per_ct.values():
        out.update(genes)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity J(A,B) = |A∩B| / |A∪B|; 0.0 when union is empty."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:  # defensive — both empty handled above
        return 0.0
    return len(a & b) / union


def pairwise_jaccard_matrix(
    gene_sets: dict[str, set[str]],
) -> tuple[np.ndarray, list[str]]:
    """Build symmetric Jaccard matrix; diagonal = 1.0 by convention.

    Returns
    -------
    matrix : np.ndarray, shape (M, M)
    labels : list[str], length M (matches matrix row/col order)
    """
    labels = sorted(gene_sets.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mat[i, i] = 1.0
    for i, j in combinations(range(n), 2):
        v = jaccard(gene_sets[labels[i]], gene_sets[labels[j]])
        mat[i, j] = v
        mat[j, i] = v
    return mat, labels


def multiway_support_counts(
    gene_sets: dict[str, set[str]],
) -> tuple[Counter, dict[int, int]]:
    """Count per-gene support and ``≥k`` method-count buckets.

    Returns
    -------
    per_gene : Counter
        ``gene -> integer support count`` over the supplied methods.
    at_least : dict[int, int]
        ``k -> number of genes supported by ≥k methods``, for k=1..M.
    """
    per_gene: Counter = Counter()
    for genes in gene_sets.values():
        for g in genes:
            per_gene[g] += 1
    n_methods = len(gene_sets)
    at_least: dict[int, int] = {}
    for k in range(1, n_methods + 1):
        at_least[k] = sum(1 for cnt in per_gene.values() if cnt >= k)
    return per_gene, at_least


def consensus_genes_at_threshold(
    gene_sets: dict[str, set[str]], threshold: int,
) -> list[dict]:
    """Return list of records ``{gene, count, methods}`` with count ≥ threshold.

    Sorted descending by count, then ascending by gene name (stable tie-break).
    """
    per_gene, _ = multiway_support_counts(gene_sets)
    records: list[dict] = []
    for gene, cnt in per_gene.items():
        if cnt < threshold:
            continue
        supporters = sorted(name for name, s in gene_sets.items() if gene in s)
        records.append({"gene": gene, "count": int(cnt), "methods": supporters})
    records.sort(key=lambda r: (-r["count"], r["gene"]))
    return records


def _format_pairwise_table(
    matrix: np.ndarray, labels: list[str],
) -> tuple[float, float, float]:
    """Return (min, max, median) over the unique upper-triangle off-diagonal pairs."""
    n = len(labels)
    vals: list[float] = []
    for i, j in combinations(range(n), 2):
        vals.append(float(matrix[i, j]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.min()), float(arr.max()), float(np.median(arr))


def _build_panel_a(ax, matrix: np.ndarray, labels: list[str]) -> None:
    """Hierarchically-clustered Jaccard heatmap with annotation labels.

    Uses average linkage on the distance matrix ``1 - J``; the resulting leaf
    order is applied symmetrically to rows and columns. Diagonal cells (J=1.0)
    are shown in light grey to reduce visual noise.
    """
    n = len(labels)
    if n < 2:
        ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title("Pairwise Jaccard")
        return

    dist = 1.0 - matrix
    np.fill_diagonal(dist, 0.0)
    # squareform requires zero diagonal + symmetric matrix; clamp tiny
    # floating-point asymmetry that scipy might reject.
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    if condensed.size > 0:
        linkage = hierarchy.linkage(condensed, method="average")
        order = hierarchy.leaves_list(linkage)
    else:
        order = np.arange(n)
    ordered_labels = [labels[i] for i in order]
    ordered_matrix = matrix[np.ix_(order, order)]

    im = ax.imshow(ordered_matrix, vmin=0, vmax=max(0.001, ordered_matrix[~np.eye(n, dtype=bool)].max()), cmap="viridis", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right")
    ax.set_yticklabels(ordered_labels)
    # Cell-value annotations
    for i in range(n):
        for j in range(n):
            v = ordered_matrix[i, j]
            color = "white" if v < 0.5 * ordered_matrix.max() else "black"
            if i == j:
                ax.text(
                    j, i, "—", ha="center", va="center",
                    color="#444", fontsize=8,
                )
            else:
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=7,
                )
    ax.set_title("(A) Pairwise gene-set Jaccard\n(hierarchical clustering, average linkage)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard J(A, B)", fontsize=8)


def _build_panel_b(
    ax,
    at_least: dict[int, int],
    n_methods: int,
    consensus_threshold: int,
) -> None:
    """Bar chart of #genes supported by ≥k methods for k=1..n_methods."""
    ks = list(range(1, n_methods + 1))
    counts = [at_least[k] for k in ks]
    bar_colors = [
        "#cccccc" if k < consensus_threshold else "#d62728"
        for k in ks
    ]
    bars = ax.bar(ks, counts, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(ks)
    ax.set_xlabel("Min. #methods supporting gene (≥k)")
    ax.set_ylabel("# genes")
    ax.set_title(
        f"(B) Multi-method support histogram\n"
        f"({n_methods} methods; consensus threshold ≥{consensus_threshold} highlighted)"
    )
    for k, cnt in zip(ks, counts):
        ax.text(k, cnt, f"{cnt}", ha="center", va="bottom", fontsize=8)
    fmt_axes(ax)
    # Headroom so the highest annotation does not collide with the spine.
    ymax = max(counts) if counts else 1
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)


def _make_figure(
    matrix: np.ndarray,
    labels: list[str],
    at_least: dict[int, int],
    consensus_threshold: int,
    out_fig_dir: Path,
) -> list[Path]:
    apply_theme(style="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _build_panel_a(axes[0], matrix, labels)
    _build_panel_b(axes[1], at_least, len(labels), consensus_threshold)
    fig.tight_layout()
    style_paper_axes(fig)
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    stem = out_fig_dir / "fig_cross_method_gene_jaccard"
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        paths.append(out)
    plt.close(fig)
    return paths


def _build_md_report(
    *,
    n_methods: int,
    method_set_sizes: dict[str, int],
    method_top_k: dict[str, int],
    matrix: np.ndarray,
    labels: list[str],
    pairwise_summary: tuple[float, float, float],
    at_least: dict[int, int],
    consensus_threshold: int,
    consensus_records: list[dict],
    excluded_ct_only: list[str],
    excluded_task_brief: list[str],
) -> str:
    """Render the markdown report."""
    lines: list[str] = []
    lines.append("# Cross-method gene-level agreement (EXP-040)")
    lines.append("")
    lines.append(
        "Pairwise Jaccard + multi-way intersection over the gene-rankable"
        " EXP-016 cross-method consensus methods."
    )
    lines.append("")
    lines.append(f"- **Gene-rankable methods loaded:** {n_methods}")
    lines.append(
        f"- **Consensus threshold:** ≥{consensus_threshold} / {n_methods} methods"
    )
    lines.append("")
    lines.append("## Methods excluded from the gene-level matrix")
    lines.append("")
    lines.append(
        "**EXP-016 methods that are CT-only (no gene-axis output):** "
        + ", ".join(excluded_ct_only)
    )
    lines.append("")
    lines.append(
        "**EXP-016 methods explicitly excluded by the task brief:** "
        + ", ".join(excluded_task_brief)
    )
    lines.append("")
    lines.append(
        "These seven methods produce per-cell-type rankings only; their per-"
        "subject artefacts are shape `[N_subjects × N_celltypes]` (no gene "
        "dim) so they cannot contribute a gene set to a pairwise Jaccard."
    )
    lines.append("")
    lines.append("## Per-method gene-set sizes (union of per-CT top-K)")
    lines.append("")
    lines.append("| Method | top-K per CT | |union across 31 CTs| |")
    lines.append("| --- | ---: | ---: |")
    for label in sorted(method_set_sizes.keys()):
        lines.append(
            f"| {label} | {method_top_k[label]} | {method_set_sizes[label]} |"
        )
    lines.append("")
    # Jaccard matrix (sorted by labels for stable ordering, NOT clustered —
    # the figure shows the clustered version).
    lines.append(f"## {n_methods}×{n_methods} Jaccard matrix (label-sorted)")
    lines.append("")
    header = "| | " + " | ".join(labels) + " |"
    sep = "| --- | " + " | ".join("---" for _ in labels) + " |"
    lines.append(header)
    lines.append(sep)
    for i, row_label in enumerate(labels):
        cells = [row_label]
        for j in range(len(labels)):
            v = matrix[i, j]
            cells.append("—" if i == j else f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lo, hi, med = pairwise_summary
    lines.append(
        f"- **Pairwise Jaccard summary** ({len(labels)*(len(labels)-1)//2} unique"
        f" upper-triangle pairs): **min** = {lo:.4f}, **max** = {hi:.4f},"
        f" **median** = {med:.4f}"
    )
    lines.append("")
    lines.append("## Multi-method support histogram")
    lines.append("")
    lines.append("| Min. #methods (≥k) | # genes |")
    lines.append("| ---: | ---: |")
    for k in sorted(at_least.keys()):
        lines.append(f"| {k} | {at_least[k]} |")
    lines.append("")
    lines.append(
        f"## Consensus genes (supported by ≥ {consensus_threshold} / "
        f"{n_methods} methods)"
    )
    lines.append("")
    if not consensus_records:
        lines.append(
            f"**No genes are supported by ≥ {consensus_threshold} of the "
            f"{n_methods} methods.**"
        )
    else:
        lines.append("| Gene | # methods | Methods |")
        lines.append("| --- | ---: | --- |")
        for rec in consensus_records:
            lines.append(
                f"| {rec['gene']} | {rec['count']} | "
                f"{', '.join(rec['methods'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captum-ig",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_ig/composite_attribution_summary.json",
    )
    parser.add_argument(
        "--gradientshap",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_robustness/gradientshap_attribution_summary.json",
    )
    parser.add_argument(
        "--smoothgrad",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_robustness/smoothgrad_attribution_summary.json",
    )
    parser.add_argument(
        "--wasserstein",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/distributional_resilience/wasserstein_per_celltype_pseudobulk.json",
    )
    parser.add_argument(
        "--de-wilcoxon-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable",
    )
    parser.add_argument(
        "--de-deseq2-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/de_resilient_vs_vulnerable_deseq2",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/cross_method_gene_jaccard.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/cross_method_gene_jaccard.md",
    )
    parser.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/cross_method_jaccard",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_GENE_RANKABLE,
        help=(
            "Per-CT top-K cap for gene-rankable methods (default 50; "
            "Wasserstein hard-capped at 10 by source schema)."
        ),
    )
    parser.add_argument(
        "--top-k-wasserstein", type=int, default=TOP_K_WASSERSTEIN,
        help="Per-CT top-K cap for Wasserstein (default 10, source-capped).",
    )
    parser.add_argument(
        "--consensus-threshold", type=int, default=6,
        help=(
            "Threshold for the 'consensus genes' table (≥ N methods)."
            " Default 6 = all 6 gene-rankable methods (strictest)."
        ),
    )
    args = parser.parse_args()

    # Validate inputs
    inputs = {
        "Captum IG": args.captum_ig,
        "GradientSHAP": args.gradientshap,
        "SmoothGrad": args.smoothgrad,
        "Wasserstein": args.wasserstein,
        "DE Wilcoxon": args.de_wilcoxon_dir,
        "DE DESeq2": args.de_deseq2_dir,
    }
    for label, path in inputs.items():
        if not path.exists():
            logger.error("Missing input for %s: %s", label, path)
            return 1

    # Load per-CT top-K for each method
    per_ct_sets: dict[str, dict[str, set[str]]] = {}
    method_top_k: dict[str, int] = {}

    logger.info("Loading Captum IG per-CT top-%d ...", args.top_k)
    per_ct_sets["Captum IG"] = _captum_per_ct_top_k(args.captum_ig, args.top_k)
    method_top_k["Captum IG"] = args.top_k

    logger.info("Loading GradientSHAP per-CT top-%d ...", args.top_k)
    per_ct_sets["GradientSHAP"] = _captum_per_ct_top_k(args.gradientshap, args.top_k)
    method_top_k["GradientSHAP"] = args.top_k

    logger.info("Loading SmoothGrad per-CT top-%d ...", args.top_k)
    per_ct_sets["SmoothGrad"] = _captum_per_ct_top_k(args.smoothgrad, args.top_k)
    method_top_k["SmoothGrad"] = args.top_k

    logger.info(
        "Loading Wasserstein per-CT top-%d (source-capped at 10) ...",
        args.top_k_wasserstein,
    )
    per_ct_sets["Wasserstein"] = _wasserstein_per_ct_top_k(
        args.wasserstein, args.top_k_wasserstein,
    )
    method_top_k["Wasserstein"] = min(args.top_k_wasserstein, TOP_K_WASSERSTEIN)

    logger.info("Loading DE Wilcoxon per-CT top-%d ...", args.top_k)
    per_ct_sets["DE Wilcoxon"] = _de_per_ct_top_k(args.de_wilcoxon_dir, args.top_k)
    method_top_k["DE Wilcoxon"] = args.top_k

    logger.info("Loading DE DESeq2 per-CT top-%d ...", args.top_k)
    per_ct_sets["DE DESeq2"] = _de_per_ct_top_k(args.de_deseq2_dir, args.top_k)
    method_top_k["DE DESeq2"] = args.top_k

    # Build per-method union sets
    gene_sets: dict[str, set[str]] = {
        label: per_ct_to_union(per_ct) for label, per_ct in per_ct_sets.items()
    }
    method_set_sizes = {label: len(s) for label, s in gene_sets.items()}
    logger.info("Per-method union sizes: %s", method_set_sizes)

    # Pairwise Jaccard
    matrix, labels = pairwise_jaccard_matrix(gene_sets)
    lo, hi, med = _format_pairwise_table(matrix, labels)
    logger.info(
        "Pairwise Jaccard: min=%.4f max=%.4f median=%.4f over %d pairs",
        lo, hi, med, len(labels) * (len(labels) - 1) // 2,
    )

    # Multi-method support
    per_gene, at_least = multiway_support_counts(gene_sets)
    logger.info(
        "Multi-method support: %s",
        {k: at_least[k] for k in sorted(at_least.keys())},
    )

    # Consensus genes at threshold
    consensus_records = consensus_genes_at_threshold(
        gene_sets, args.consensus_threshold,
    )
    logger.info(
        "Consensus genes (≥%d methods): %d genes",
        args.consensus_threshold, len(consensus_records),
    )

    n_methods = len(gene_sets)
    pairwise_summary = (lo, hi, med)

    # Top-5 most-cross-method-validated genes (regardless of threshold)
    top5_records = consensus_genes_at_threshold(gene_sets, threshold=1)[:5]

    # Pairwise pair list for JSON (deterministic order)
    pair_records: list[dict] = []
    for i, j in combinations(range(n_methods), 2):
        a = labels[i]
        b = labels[j]
        pair_records.append({
            "method_a": a,
            "method_b": b,
            "jaccard": float(matrix[i, j]),
            "intersection_size": len(gene_sets[a] & gene_sets[b]),
            "union_size": len(gene_sets[a] | gene_sets[b]),
        })

    # Write JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "config": {
            "n_methods": n_methods,
            "method_top_k": method_top_k,
            "consensus_threshold": args.consensus_threshold,
            "excluded_ct_only_methods": list(EXP_016_CT_ONLY_METHODS),
            "excluded_task_brief": list(EXP_016_TASK_EXCLUDED),
            "inputs": {label: str(path) for label, path in inputs.items()},
        },
        "labels_label_sorted": labels,
        "method_set_sizes": method_set_sizes,
        "jaccard_matrix": [
            [float(matrix[i, j]) for j in range(n_methods)]
            for i in range(n_methods)
        ],
        "pairwise_summary": {
            "n_pairs": n_methods * (n_methods - 1) // 2,
            "min_jaccard": lo,
            "max_jaccard": hi,
            "median_jaccard": med,
        },
        "pairwise_pairs": pair_records,
        "multiway_at_least_k": {str(k): v for k, v in at_least.items()},
        "consensus_genes": consensus_records,
        "top5_most_cross_method_validated": top5_records,
    }
    args.out_json.write_text(json.dumps(json_payload, indent=2, sort_keys=False))
    logger.info("Wrote %s", args.out_json)

    # Write MD
    md = _build_md_report(
        n_methods=n_methods,
        method_set_sizes=method_set_sizes,
        method_top_k=method_top_k,
        matrix=matrix,
        labels=labels,
        pairwise_summary=pairwise_summary,
        at_least=at_least,
        consensus_threshold=args.consensus_threshold,
        consensus_records=consensus_records,
        excluded_ct_only=list(EXP_016_CT_ONLY_METHODS),
        excluded_task_brief=list(EXP_016_TASK_EXCLUDED),
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    logger.info("Wrote %s", args.out_md)

    # Render figure (PNG + PDF, 600 DPI)
    figure_paths = _make_figure(
        matrix=matrix,
        labels=labels,
        at_least=at_least,
        consensus_threshold=args.consensus_threshold,
        out_fig_dir=args.out_fig_dir,
    )
    for p in figure_paths:
        logger.info("Wrote %s", p)

    # Stdout summary
    print()
    print("=" * 78)
    print(
        f"Cross-method gene-level Jaccard ({n_methods} methods × "
        f"31 CTs × per-CT top-K)"
    )
    print("=" * 78)
    print(f"Set sizes (union over CTs): {method_set_sizes}")
    print(
        f"Pairwise Jaccard: min={lo:.4f} max={hi:.4f} median={med:.4f}"
        f" over {n_methods * (n_methods - 1) // 2} pairs"
    )
    print(f"Multi-method support: {at_least}")
    print(
        f"Consensus genes (≥{args.consensus_threshold}/{n_methods}): "
        f"{len(consensus_records)}"
    )
    if top5_records:
        print("Top-5 most-cross-method-validated genes:")
        for rec in top5_records:
            print(f"  {rec['gene']:<10s} {rec['count']:>2d}/{n_methods}  "
                  f"({', '.join(rec['methods'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
