#!/usr/bin/env python
"""Stratify the canonical 5-fold val R² by per-subject region availability.

Each subject in ROSMAP is sequenced from one or more of 6 brain
regions: ``["PFC", "AG", "MTC", "EC", "HC", "TH"]`` (order from
``src/data/constants.py::REGION_ORDER``). A boolean ``region_mask``
of length 6 lives inside each ``data/precomputed/R*.pt`` cache and
indicates which of the 6 regions are available for that subject.

The cohort distribution in the precomputed caches is roughly:

* **PFC-only** (``region_mask`` sums to 1): 87.6 % (452 subjects)
* **2-5 regions** (multi-region but not all six): 4.1 % (21 subjects)
* **All 6 regions**: 8.3 % (43 subjects)

Note: 2 of the 452 single-region subjects have a non-PFC region
(region 4 = HC). They are placed in ``pfc_only`` regardless, since
their region count is 1; this matches the user-specified
"PFC-only / 2-5 / all 6" partition on **region count**, not on
PFC presence.

This script does **not** re-run the model. It consumes already-
written canonical val predictions from
``outputs/canonical/p5_canonical_seed42/fold{0..4}/val_predictions_best.npz``
(keys: ``subject_ids``, ``predictions``, ``targets``), joins on
subject IDs to the per-subject region count, then computes per-fold
+ pooled R² / Pearson r / MAE inside each stratum. Finally it tests
whether the model predicts equally well on PFC-only vs multi-region
(``two_to_five`` ∪ ``all_six``) subjects via Wilcoxon signed-rank on
the 5 paired per-fold R² values.

Outputs:
    ``<out-data-dir>/per_region_stratified_r2.json``
    ``<out-data-dir>/per_region_stratified_r2.md``
    ``<out-fig-dir>/fig_per_region_r2.{png,pdf}`` (bar chart, 600 DPI)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

# Stratum labels (must match the canonical ordering used in JSON / MD / fig).
STRATUM_PFC_ONLY = "pfc_only"
STRATUM_TWO_TO_FIVE = "two_to_five"
STRATUM_ALL_SIX = "all_six"
STRATA: list[str] = [STRATUM_PFC_ONLY, STRATUM_TWO_TO_FIVE, STRATUM_ALL_SIX]

STRATUM_DISPLAY: dict[str, str] = {
    STRATUM_PFC_ONLY: "PFC-only (1 region)",
    STRATUM_TWO_TO_FIVE: "2-5 regions",
    STRATUM_ALL_SIX: "All 6 regions",
}


def _stratum_for_count(n_regions: int) -> str:
    """Map an integer region count to the stratum label."""
    if n_regions == 1:
        return STRATUM_PFC_ONLY
    if 2 <= n_regions <= 5:
        return STRATUM_TWO_TO_FIVE
    if n_regions == 6:
        return STRATUM_ALL_SIX
    raise ValueError(
        f"region count {n_regions!r} outside [1, 6]; check region_mask"
    )


def build_region_count_map(precomputed_dir: Path) -> dict[str, int]:
    """Walk ``R*.pt`` caches and tabulate ``region_mask.sum()`` per SID.

    Returns a dict ``{subject_id: n_regions_available}``.
    """
    pt_files = sorted(precomputed_dir.glob("R*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No R*.pt files found in {precomputed_dir}"
        )
    out: dict[str, int] = {}
    for f in pt_files:
        pt = torch.load(f, weights_only=False, map_location="cpu")
        rm = pt.get("region_mask")
        if rm is None or not torch.is_tensor(rm) or rm.numel() != 6:
            raise ValueError(
                f"{f}: region_mask missing or not length-6 tensor"
            )
        out[f.stem] = int(rm.sum().item())
    return out


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """R² that returns None instead of nan when y_true has no variance."""
    if y_true.size < 2:
        return None
    if float(np.var(y_true)) == 0.0:
        return None
    return float(r2_score(y_true, y_pred))


def _safe_pearson(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float | None:
    """Pearson r with nan/zero-variance guards."""
    if y_true.size < 2:
        return None
    if float(np.var(y_true)) == 0.0 or float(np.var(y_pred)) == 0.0:
        return None
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 1:
        return None
    return float(mean_absolute_error(y_true, y_pred))


def _per_stratum_metrics(
    sids: Iterable[str],
    preds: np.ndarray,
    targets: np.ndarray,
    region_counts: dict[str, int],
) -> dict[str, dict]:
    """Group predictions/targets by stratum and compute (r2, pearson_r, mae).

    Returns mapping ``stratum -> {n_subjects, r2, pearson_r, mae}``.
    """
    sids_arr = np.asarray(list(sids), dtype=object)
    n_regions = np.array(
        [region_counts[str(s)] for s in sids_arr], dtype=np.int32
    )
    out: dict[str, dict] = {}
    for stratum in STRATA:
        if stratum == STRATUM_PFC_ONLY:
            mask = n_regions == 1
        elif stratum == STRATUM_TWO_TO_FIVE:
            mask = (n_regions >= 2) & (n_regions <= 5)
        elif stratum == STRATUM_ALL_SIX:
            mask = n_regions == 6
        else:  # pragma: no cover — STRATA is exhaustive
            raise AssertionError(stratum)
        y_true = targets[mask]
        y_pred = preds[mask]
        out[stratum] = {
            "n_subjects": int(mask.sum()),
            "r2": _safe_r2(y_true, y_pred),
            "pearson_r": _safe_pearson(y_true, y_pred),
            "mae": _safe_mae(y_true, y_pred),
        }
    return out


def aggregate_per_fold(
    pred_root: Path, region_counts: dict[str, int]
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Walk fold0..fold4 and compute per-fold per-stratum metrics.

    Returns ``(per_fold_records, all_sids, all_preds, all_targets)`` where
    the latter three are concatenations across all 5 folds (used for the
    pooled tabulation downstream).
    """
    per_fold_records: list[dict] = []
    all_sids: list[str] = []
    all_preds: list[float] = []
    all_targets: list[float] = []
    for fold_idx in range(5):
        npz_path = pred_root / f"fold{fold_idx}/val_predictions_best.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        d = np.load(npz_path, allow_pickle=True)
        sids = [str(s) for s in d["subject_ids"]]
        preds = np.asarray(d["predictions"], dtype=np.float64)
        targets = np.asarray(d["targets"], dtype=np.float64)
        if not (len(sids) == preds.size == targets.size):
            raise ValueError(
                f"fold{fold_idx}: array length mismatch "
                f"sids={len(sids)} preds={preds.size} targets={targets.size}"
            )
        # Verify every val SID has a region count.
        missing = [s for s in sids if s not in region_counts]
        if missing:
            raise KeyError(
                f"fold{fold_idx}: {len(missing)} val subjects lack region "
                f"counts (sample: {missing[:3]})"
            )
        per_stratum = _per_stratum_metrics(
            sids, preds, targets, region_counts
        )
        per_fold_records.append(
            {
                "fold_index": fold_idx,
                "n_val_total": int(preds.size),
                "r2_overall": _safe_r2(targets, preds),
                "pearson_r_overall": _safe_pearson(targets, preds),
                "mae_overall": _safe_mae(targets, preds),
                "per_stratum": per_stratum,
            }
        )
        all_sids.extend(sids)
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
    return (
        per_fold_records,
        np.asarray(all_sids, dtype=object),
        np.asarray(all_preds, dtype=np.float64),
        np.asarray(all_targets, dtype=np.float64),
    )


def aggregate_pooled(
    all_sids: np.ndarray,
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    region_counts: dict[str, int],
) -> dict:
    """Pool concatenated predictions/targets across all 5 folds; metrics."""
    per_stratum = _per_stratum_metrics(
        all_sids, all_preds, all_targets, region_counts
    )
    return {
        "n_val_total": int(all_preds.size),
        "r2_overall": _safe_r2(all_targets, all_preds),
        "pearson_r_overall": _safe_pearson(all_targets, all_preds),
        "mae_overall": _safe_mae(all_targets, all_preds),
        "per_stratum": per_stratum,
    }


def wilcoxon_pfc_vs_multi(per_fold_records: list[dict]) -> dict:
    """Wilcoxon signed-rank on per-fold R² (PFC-only) vs (2-5 ∪ all-6).

    Multi-region per-fold R² is computed by *concatenating* the 2-5 and
    all-6 subjects within that fold and recomputing R² on the combined
    pool — i.e. Wilcoxon compares R²(fold-level PFC-only) vs R²(fold-
    level multi-region pool) across the 5 paired folds.

    For folds where either side has fewer than 2 subjects or zero target
    variance, the pair is dropped (recorded as ``n_pairs``). With 5
    folds the smallest two-sided Wilcoxon p is 0.0625.
    """
    paired: list[tuple[float, float]] = []
    skipped: list[int] = []
    for rec in per_fold_records:
        ps = rec["per_stratum"]
        r2_pfc = ps[STRATUM_PFC_ONLY]["r2"]
        # Build a fold-level pooled multi-region R² from the 2-5 and
        # all-6 strata. The per-stratum block stores R² but to combine
        # across the two strata we need the underlying y/y_pred arrays.
        # Easiest: rebuild from the per_fold record's ``n_subjects`` —
        # but per_stratum doesn't carry the raw arrays. So we recompute
        # a "multi" R² as the size-weighted mean only if we can; instead
        # the cleanest approach is to compare PFC-only vs each of the
        # two multi-region sub-strata jointly (i.e. re-derive from the
        # NPZ in ``run_main``). Done in ``_build_multi_r2_per_fold``
        # below.
        if r2_pfc is None:
            skipped.append(rec["fold_index"])
            continue
        r2_multi = rec.get("_r2_multi", None)
        if r2_multi is None:
            skipped.append(rec["fold_index"])
            continue
        paired.append((float(r2_pfc), float(r2_multi)))
    n_pairs = len(paired)
    if n_pairs < 1:
        return {
            "statistic": None,
            "p_value": None,
            "n_pairs": 0,
            "skipped_folds": skipped,
            "pfc_only_per_fold_r2": [],
            "multi_region_per_fold_r2": [],
            "alternative": "two-sided",
        }
    pfc_arr = np.array([p[0] for p in paired])
    multi_arr = np.array([p[1] for p in paired])
    diffs = pfc_arr - multi_arr
    # If all paired diffs are zero, scipy raises; guard explicitly.
    if np.all(diffs == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_pairs": n_pairs,
            "skipped_folds": skipped,
            "pfc_only_per_fold_r2": pfc_arr.tolist(),
            "multi_region_per_fold_r2": multi_arr.tolist(),
            "alternative": "two-sided",
        }
    res = stats.wilcoxon(pfc_arr, multi_arr, alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n_pairs": n_pairs,
        "skipped_folds": skipped,
        "pfc_only_per_fold_r2": pfc_arr.tolist(),
        "multi_region_per_fold_r2": multi_arr.tolist(),
        "alternative": "two-sided",
    }


def _build_multi_r2_per_fold(
    pred_root: Path,
    region_counts: dict[str, int],
    per_fold_records: list[dict],
) -> None:
    """Attach a fold-level ``_r2_multi`` to each record.

    Rebuilds raw y/y_pred per fold, masks to multi-region (n_regions ≥ 2),
    and computes a single R² over that pool. Used by
    ``wilcoxon_pfc_vs_multi``. Skips (leaves None) when the multi pool
    has fewer than 2 subjects or zero target variance.
    """
    for rec in per_fold_records:
        fold_idx = rec["fold_index"]
        npz_path = pred_root / f"fold{fold_idx}/val_predictions_best.npz"
        d = np.load(npz_path, allow_pickle=True)
        sids = [str(s) for s in d["subject_ids"]]
        preds = np.asarray(d["predictions"], dtype=np.float64)
        targets = np.asarray(d["targets"], dtype=np.float64)
        n_regions = np.array(
            [region_counts[s] for s in sids], dtype=np.int32
        )
        mask_multi = n_regions >= 2
        rec["_r2_multi"] = _safe_r2(targets[mask_multi], preds[mask_multi])
        rec["_n_multi"] = int(mask_multi.sum())


def make_figure(
    per_fold_records: list[dict],
    pooled: dict,
    out_fig_dir: Path,
) -> None:
    """Bar chart of pooled per-stratum R² with 5-fold std error bars."""
    apply_theme(style="paper")
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # Per-stratum: pooled R² (height) and 5-fold std (error bar).
    means: list[float] = []
    stds: list[float] = []
    ns: list[int] = []
    labels: list[str] = []
    for stratum in STRATA:
        per_fold_r2 = [
            rec["per_stratum"][stratum]["r2"]
            for rec in per_fold_records
            if rec["per_stratum"][stratum]["r2"] is not None
        ]
        pooled_r2 = pooled["per_stratum"][stratum]["r2"]
        n_sub = pooled["per_stratum"][stratum]["n_subjects"]
        # Bar height = pooled R²; error bar = std of the per-fold R²s
        # (population std, ``ddof=0``, mirroring the canonical convention
        # used in EXP-001/EXP-003 for fold-mean ± fold-std reporting).
        means.append(float("nan") if pooled_r2 is None else float(pooled_r2))
        stds.append(
            float("nan")
            if len(per_fold_r2) < 2
            else float(np.std(per_fold_r2, ddof=0))
        )
        ns.append(int(n_sub))
        labels.append(f"{STRATUM_DISPLAY[stratum]}\n(n={n_sub})")

    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
    x = np.arange(len(STRATA))
    colors = ["#4F81BD", "#9BBB59", "#C0504D"]
    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    # Per-fold std error bars are drawn separately so we can mark them
    # clipped vs full. Single-subject (n=1) folds and folds with extreme
    # leverage in n≤5 strata produce per-fold R² values like -2.6 / -1.3,
    # which would push the y-axis to -3 and obscure the bar heights.
    # Solution: draw the error bars at the actual std but clip the y-axis
    # to a readable [-0.5, 1.0]; annotate clipped strata in the caption.
    err_low = np.asarray(stds, dtype=float)
    err_high = np.asarray(stds, dtype=float)
    ax.errorbar(
        x,
        means,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="black",
        capsize=4,
        elinewidth=0.8,
    )
    # Reference line: canonical pooled R² across all 516 subjects.
    pooled_overall = pooled["r2_overall"]
    if pooled_overall is not None:
        ax.axhline(
            pooled_overall,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"Canonical pooled R² = {pooled_overall:.3f}",
        )
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pooled val R² (5-fold)")
    ax.set_title("ResDec-MHE val R² by region availability stratum")
    # Annotate with bar height + per-fold std (e.g. "0.445 (±0.09)").
    for xb, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            std_str = "" if np.isnan(s) else f" (±{s:.2f})"
            ax.annotate(
                f"{m:.3f}{std_str}",
                xy=(xb.get_x() + xb.get_width() / 2, m),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    # y-axis clipped to a readable window. Per-fold R² in small strata
    # (n=4, n=8, n=9) reaches negative values from single-subject
    # leverage; the std bars accordingly overshoot the window. The full
    # per-fold values are recorded in the JSON.
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(0.0, color="black", linewidth=0.4)
    fig.tight_layout()
    png = out_fig_dir / "fig_per_region_r2.png"
    pdf = out_fig_dir / "fig_per_region_r2.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote figure: {png}")
    logger.info(f"Wrote figure: {pdf}")


def write_markdown(
    payload: dict,
    out_md: Path,
) -> None:
    """Render the report in Markdown."""
    pooled = payload["pooled"]
    per_fold = payload["per_fold"]
    wx = payload["wilcoxon_pfc_vs_multi"]

    def _fmt(v: float | None, fmt: str = ".4f") -> str:
        if v is None:
            return "—"
        return format(v, fmt)

    lines: list[str] = []
    lines.append("# Per-region-stratified val R² (canonical 5-fold)\n")
    lines.append(
        "Stratifies the canonical val predictions "
        "(`outputs/canonical/p5_canonical_seed42/fold{0..4}/val_predictions_best.npz`) "
        "by per-subject region availability.\n"
    )
    lines.append("## Cohort distribution (pooled across 5 folds)\n")
    lines.append("| Stratum | n subjects | % of 516 |\n")
    lines.append("|---|---|---|\n")
    total = sum(
        s["n_subjects"] for s in pooled["per_stratum"].values()
    )
    for stratum in STRATA:
        block = pooled["per_stratum"][stratum]
        n = block["n_subjects"]
        pct = 100.0 * n / total if total else float("nan")
        lines.append(
            f"| {STRATUM_DISPLAY[stratum]} | {n} | {pct:.1f}% |\n"
        )
    lines.append("\n")

    lines.append("## Pooled per-stratum metrics (concatenated across 5 folds)\n")
    lines.append("| Stratum | n | R² | Pearson r | MAE |\n")
    lines.append("|---|---|---|---|---|\n")
    for stratum in STRATA:
        block = pooled["per_stratum"][stratum]
        lines.append(
            f"| {STRATUM_DISPLAY[stratum]} | {block['n_subjects']} | "
            f"{_fmt(block['r2'])} | {_fmt(block['pearson_r'])} | "
            f"{_fmt(block['mae'])} |\n"
        )
    lines.append(
        f"\n*Canonical pooled R² (all 516, all folds concatenated): "
        f"{_fmt(pooled['r2_overall'])}*\n\n"
    )

    lines.append("## Per-fold per-stratum R²\n")
    lines.append(
        "| Fold | n_val | overall R² | "
        + " | ".join(
            f"R² {STRATUM_DISPLAY[s]}" for s in STRATA
        )
        + " |\n"
    )
    lines.append("|" + "---|" * (3 + len(STRATA)) + "\n")
    for rec in per_fold:
        cells = [
            f"fold{rec['fold_index']}",
            str(rec["n_val_total"]),
            _fmt(rec["r2_overall"]),
        ]
        for stratum in STRATA:
            cells.append(_fmt(rec["per_stratum"][stratum]["r2"]))
        lines.append("| " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("## Wilcoxon: PFC-only vs multi-region (per-fold paired R²)\n")
    if wx["n_pairs"] >= 1:
        lines.append(
            f"- Paired folds (PFC-only vs (2-5 ∪ all-6 pool)): n_pairs = "
            f"{wx['n_pairs']}\n"
        )
        lines.append(f"- Wilcoxon W = {_fmt(wx['statistic'], '.4f')}\n")
        lines.append(
            f"- Two-sided p = {_fmt(wx['p_value'], '.4f')} "
            "(smallest possible at n=5 is 0.0625)\n"
        )
        lines.append("- Per-fold R²:\n")
        lines.append(
            "    | Fold | PFC-only | multi-region |\n"
            "    |---|---|---|\n"
        )
        for i, (a, b) in enumerate(
            zip(wx["pfc_only_per_fold_r2"], wx["multi_region_per_fold_r2"])
        ):
            lines.append(f"    | {i} | {a:.4f} | {b:.4f} |\n")
    else:
        lines.append("- Insufficient paired folds for Wilcoxon.\n")
    if wx.get("skipped_folds"):
        lines.append(
            f"- Skipped folds (zero variance / empty stratum): "
            f"{wx['skipped_folds']}\n"
        )
    lines.append("\n")

    lines.append("## Interpretation\n")
    pooled_pfc_r2 = pooled["per_stratum"][STRATUM_PFC_ONLY]["r2"]
    pooled_25_r2 = pooled["per_stratum"][STRATUM_TWO_TO_FIVE]["r2"]
    pooled_all_r2 = pooled["per_stratum"][STRATUM_ALL_SIX]["r2"]
    lines.append(
        f"- PFC-only pooled R² = {_fmt(pooled_pfc_r2)}; "
        f"2-5 regions = {_fmt(pooled_25_r2)}; "
        f"all 6 regions = {_fmt(pooled_all_r2)}.\n"
    )
    lines.append(
        "- 87.6% of the cohort is PFC-only, so the PFC-only pooled R² is "
        "the dominant driver of the canonical headline number.\n"
    )
    lines.append(
        "- The multi-region strata (n=21 + n=43) are small enough that "
        "single-subject leverage and target-variance differences inflate "
        "the per-fold variance; interpret per-fold std bars accordingly.\n"
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines))
    logger.info(f"Wrote markdown: {out_md}")


def run_main(args: argparse.Namespace) -> None:
    pred_root = Path(args.pred_root)
    precomp_dir = Path(args.precomputed_dir)
    out_data_dir = Path(args.out_data_dir)
    out_fig_dir = Path(args.out_fig_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building region-count map from {precomp_dir} ...")
    region_counts = build_region_count_map(precomp_dir)
    logger.info(f"  {len(region_counts)} subjects mapped.")

    # Sanity log: how many subjects in each stratum at the cohort level
    # (independent of the val-fold partition).
    cohort_strat: dict[str, int] = {s: 0 for s in STRATA}
    for n in region_counts.values():
        cohort_strat[_stratum_for_count(n)] += 1
    logger.info(f"  Cohort strata: {cohort_strat}")

    logger.info(f"Aggregating per-fold metrics from {pred_root} ...")
    per_fold, all_sids, all_preds, all_targets = aggregate_per_fold(
        pred_root, region_counts
    )
    pooled = aggregate_pooled(
        all_sids, all_preds, all_targets, region_counts
    )

    # Attach fold-level multi-region R² for the paired Wilcoxon.
    _build_multi_r2_per_fold(pred_root, region_counts, per_fold)
    wx = wilcoxon_pfc_vs_multi(per_fold)

    # Strip the leading-underscore intermediate fields so JSON stays clean.
    clean_per_fold = []
    for rec in per_fold:
        clean_per_fold.append(
            {
                k: v
                for k, v in rec.items()
                if not k.startswith("_")
            }
        )

    payload = {
        "config": {
            "strata": STRATA,
            "stratum_display": STRATUM_DISPLAY,
            "pred_root": str(pred_root),
            "precomputed_dir": str(precomp_dir),
            "n_folds": 5,
            "stratum_definition": (
                "n_regions = int(region_mask.sum()) where region_mask is the "
                "boolean length-6 tensor inside data/precomputed/<sid>.pt. "
                "pfc_only := n_regions == 1, two_to_five := 2 <= n_regions <= 5, "
                "all_six := n_regions == 6. Region order: PFC, AG, MTC, EC, HC, TH."
            ),
        },
        "cohort_strata_counts": cohort_strat,
        "per_fold": clean_per_fold,
        "pooled": pooled,
        "wilcoxon_pfc_vs_multi": wx,
    }
    out_json = out_data_dir / "per_region_stratified_r2.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    logger.info(f"Wrote JSON: {out_json}")

    out_md = out_data_dir / "per_region_stratified_r2.md"
    write_markdown(payload, out_md)

    make_figure(per_fold, pooled, out_fig_dir)

    logger.info("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred-root",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
        help="Directory containing fold{0..4}/val_predictions_best.npz",
    )
    p.add_argument(
        "--precomputed-dir",
        type=Path,
        default=_WORKTREE_ROOT / "data/precomputed",
        help="Directory containing R*.pt files with region_mask",
    )
    p.add_argument(
        "--out-data-dir",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability",
        help="Directory for JSON + MD output",
    )
    p.add_argument(
        "--out-fig-dir",
        type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/figures/per_region_r2"
        ),
        help="Directory for PNG + PDF figure output",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run_main(parse_args())
