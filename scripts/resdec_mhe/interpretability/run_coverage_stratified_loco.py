#!/usr/bin/env python
"""Coverage-stratified LOCO (leave-one-cell-type-out) zero-out ablation.

Extension of ``run_loco_zero_out.py``: for each of 31 cell types, computes
the LOCO ΔR² on (a) the full N=516 cohort (matching EXP-013 baseline) and
(b) the subset of subjects whose precomputed cache has ``cell_counts[CT] >= 1``.
Restricted-cohort analysis answers a reviewer concern: for CTs with
median 0 cells per subject, the gradient-from-zero is ill-defined; does
the per-CT importance ranking change when restricted to subjects who
actually have that CT?

The semantics match EXP-013 exactly:

* Per-fold canonical R² and per-fold LOCO R² are computed on the val
  set (no retraining, single forward pass per CT per fold).
* Composite predictions = ResDec residual (model output) + TabPFN outer
  prediction.
* ΔR² = ``mean_over_folds(loco_r2_fold) - mean_over_folds(canon_r2_fold)``.
  This equals the mean of per-fold paired ΔR²s by linearity.

Restricted-cohort ΔR²: same formula, but at each fold the subset of val
subjects with ``cell_counts[CT] >= 1`` is selected, and a *restricted-set
canonical R²* is computed for the same subset to avoid bias from subset
composition. ΔR² is the difference of these restricted-set R²s averaged
across folds.

Outputs:

* ``<out-data-dir>/coverage_stratified_loco.json``
* ``<out-data-dir>/coverage_stratified_loco.md``
* ``<out-fig-dir>/fig_coverage_stratified_loco.{png,pdf}``  (3-panel, 600 DPI)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import r2_score

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.data.datamodule import CognitiveResilienceDataModule  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.training.resdec_lightning_module import ResDecLightningModule  # noqa: E402
from src.utils.provenance import git_sha, pick_max_r2_ckpt  # noqa: E402
from src.visualization.theme import apply_theme  # noqa: E402

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Forward-pass helpers (match run_loco_zero_out semantics exactly)
# -----------------------------------------------------------------------


def _zero_out_and_predict(
    model: ResDecLightningModule,
    val_batches: list[dict],
    ct_to_zero: int | None,
    tabpfn_outer_map: dict[str, float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward over val batches with cell type ``ct_to_zero`` zeroed.

    Mirrors the canonical ``run_loco_zero_out._zero_out_and_predict``:
    zeroing happens on both ``pseudobulk`` ``[B, n_ct, n_genes]`` and
    ``region_pseudobulk`` ``[B, n_regions, n_ct, n_genes]`` tensors at
    the per-CT axis. Composite = residual + TabPFN outer.
    """
    sids_all: list[str] = []
    comp_all: list[float] = []
    true_all: list[float] = []
    with torch.no_grad():
        for batch in val_batches:
            batch_d = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            if ct_to_zero is not None:
                for key in ("pseudobulk", "region_pseudobulk"):
                    v = batch_d.get(key)
                    if v is None or not torch.is_tensor(v):
                        continue
                    v_mod = v.clone()
                    if key == "pseudobulk":
                        v_mod[:, ct_to_zero, :] = 0.0
                    else:
                        v_mod[:, :, ct_to_zero, :] = 0.0
                    batch_d[key] = v_mod
            out = model(batch_d)
            residual = out["prediction"].detach().cpu().numpy().reshape(-1)
            for i, sid in enumerate(batch["subject_ids"]):
                sid_str = str(sid)
                ytab = tabpfn_outer_map.get(sid_str, np.nan)
                composite = float(residual[i]) + float(ytab)
                sids_all.append(sid_str)
                comp_all.append(composite)
                true_all.append(
                    float(batch_d["cognition"][i].item())
                    if "cognition" in batch_d else np.nan,
                )
    return (
        np.asarray(sids_all, dtype=object),
        np.asarray(comp_all, dtype=np.float64),
        np.asarray(true_all, dtype=np.float64),
    )


def _load_tabpfn_outer_map(tabpfn_dir: Path, fold: int) -> dict[str, float]:
    path = tabpfn_dir / f"tabpfn_outer_fold{fold}.npz"
    d = np.load(path, allow_pickle=True)
    return {
        str(s): float(v) for s, v in zip(d["val_subject_ids"], d["y_tabpfn"])
    }


# -----------------------------------------------------------------------
# Per-subject coverage matrix (cell_counts per CT)
# -----------------------------------------------------------------------


def build_subject_cell_counts(
    precomputed_dir: Path, n_cell_types: int = 31,
) -> tuple[list[str], np.ndarray]:
    """Walk ``R*.pt`` caches; return (subject_ids, counts[N, n_cell_types]).

    ``counts[i, ct]`` is the number of cells of cell type ``ct`` in subject
    ``i``'s precomputed cache.
    """
    pt_files = sorted(precomputed_dir.glob("R*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No R*.pt files in {precomputed_dir}")
    sids: list[str] = []
    counts_rows: list[np.ndarray] = []
    for f in pt_files:
        pt = torch.load(f, weights_only=False, map_location="cpu")
        cc = pt.get("cell_counts")
        if cc is None or not torch.is_tensor(cc) or cc.numel() != n_cell_types:
            raise ValueError(
                f"{f}: cell_counts missing or not length-{n_cell_types}"
            )
        sids.append(f.stem)
        counts_rows.append(cc.cpu().numpy().astype(np.int64))
    return sids, np.stack(counts_rows, axis=0)


# -----------------------------------------------------------------------
# Per-fold workhorse: build state, capture canonical + LOCO predictions
# -----------------------------------------------------------------------


def _run_fold(
    fold: int,
    cfg: Any,
    canonical_dir: Path,
    splits_path: Path,
    tabpfn_dir: Path,
    device: torch.device,
    n_cell_types: int,
) -> dict[str, Any]:
    """Run canonical + per-CT LOCO forward passes for one fold.

    Returns a dict with::

        {
          "fold": int,
          "subject_ids": np.ndarray[N_val],
          "true_y": np.ndarray[N_val],
          "comp_canon": np.ndarray[N_val],
          "comp_loco": np.ndarray[n_cell_types, N_val],
        }
    """
    fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(fold_cfg, False)
    fold_cfg.data.fold = fold

    fold_dir = canonical_dir / f"fold{fold}"
    ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("fold %d: loading %s", fold, ckpt_path.name)

    splits = load_splits(str(splits_path))
    metadata = pd.read_csv(Path(fold_cfg.data.metadata_path) / "metadata.csv")
    dm = CognitiveResilienceDataModule(
        config=fold_cfg, metadata=metadata, splits=splits,
        fold_idx=fold,
        precomputed_dir=fold_cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=fold_cfg, map_location="cpu",
    ).to(device).eval()

    val_batches: list[dict] = list(dm.val_dataloader())
    tabpfn_map = _load_tabpfn_outer_map(tabpfn_dir, fold)

    # Canonical (no zero) — gives subject ordering and true_y.
    sids, comp_canon, true_y = _zero_out_and_predict(
        model, val_batches, None, tabpfn_map, device,
    )

    # Per-CT LOCO. Store the composite predictions in [n_ct, N_val] aligned
    # with the canonical ``sids`` ordering — _zero_out_and_predict is
    # deterministic-batch-iteration so SIDs come back in identical order.
    comp_loco = np.zeros((n_cell_types, len(sids)), dtype=np.float64)
    for ct in range(n_cell_types):
        sids_ct, comp_ct, _ = _zero_out_and_predict(
            model, val_batches, ct, tabpfn_map, device,
        )
        if not np.array_equal(sids_ct, sids):
            raise RuntimeError(
                f"fold {fold} ct {ct}: SID order drift between canonical "
                f"and LOCO forward passes"
            )
        comp_loco[ct] = comp_ct

    del model
    torch.cuda.empty_cache()
    return {
        "fold": fold,
        "subject_ids": sids,
        "true_y": true_y,
        "comp_canon": comp_canon,
        "comp_loco": comp_loco,
    }


# -----------------------------------------------------------------------
# Aggregation: per-CT full-cohort + restricted ΔR²
# -----------------------------------------------------------------------


def aggregate_loco(
    fold_payloads: list[dict[str, Any]],
    subject_counts: dict[str, np.ndarray],
    n_cell_types: int,
    min_cells_threshold: int = 1,
) -> dict[str, Any]:
    """Compute full-cohort + restricted-cohort per-CT ΔR² across folds.

    Parameters
    ----------
    fold_payloads
        List of dicts produced by ``_run_fold``.
    subject_counts
        Map ``sid -> np.ndarray[n_cell_types]`` of cell counts.
    n_cell_types
        Number of cell types (31).
    min_cells_threshold
        Threshold for "subject has cells of this CT". Default 1.

    Returns
    -------
    Dict with::

        {
          "n_folds": int,
          "canonical_per_fold_full": list[float],
          "canonical_mean_full": float,
          "per_cell_type": [
            {
              "cell_type_index": int,
              "cell_type": str,
              "loco_per_fold_full": list[float],
              "loco_mean_full": float,
              "delta_r2_full": float,
              "delta_r2_full_per_fold": list[float],
              "n_val_full": list[int],  # per-fold val count (full cohort)
              "loco_per_fold_restricted": list[float | None],
              "canonical_per_fold_restricted": list[float | None],
              "delta_r2_restricted": float | None,
              "delta_r2_restricted_per_fold": list[float | None],
              "n_val_restricted": list[int],  # per-fold val count (restricted)
              "n_val_restricted_total": int,
            }, ...
          ]
        }
    """
    # Per-fold canonical R² over the FULL val set (these are the EXP-013
    # canonical R² values, exactly).
    canonical_per_fold_full: list[float] = []
    for fp in fold_payloads:
        canonical_per_fold_full.append(
            float(r2_score(fp["true_y"], fp["comp_canon"]))
        )
    canonical_mean_full = float(np.mean(canonical_per_fold_full))

    per_ct_results: list[dict[str, Any]] = []
    for ct in range(n_cell_types):
        ct_name = CELL_TYPE_ORDER[ct]
        loco_full: list[float] = []
        loco_full_delta: list[float] = []
        n_val_full: list[int] = []
        loco_restricted: list[float | None] = []
        canon_restricted: list[float | None] = []
        n_val_restricted: list[int] = []
        delta_restricted_per_fold: list[float | None] = []

        for fold_idx, fp in enumerate(fold_payloads):
            sids = fp["subject_ids"]
            true_y = fp["true_y"]
            comp_canon = fp["comp_canon"]
            comp_loco_ct = fp["comp_loco"][ct]

            # Full-cohort fold R² + ΔR² (per-fold paired).
            r2_full = float(r2_score(true_y, comp_loco_ct))
            loco_full.append(r2_full)
            loco_full_delta.append(r2_full - canonical_per_fold_full[fold_idx])
            n_val_full.append(len(sids))

            # Restricted to subjects with cell_counts[ct] >= threshold.
            keep_mask = np.zeros(len(sids), dtype=bool)
            for i, sid in enumerate(sids):
                cnt = subject_counts.get(sid)
                if cnt is None:
                    continue
                if int(cnt[ct]) >= min_cells_threshold:
                    keep_mask[i] = True
            n_keep = int(keep_mask.sum())
            n_val_restricted.append(n_keep)

            if n_keep < 2:
                # r2_score requires >= 2 samples; otherwise undefined.
                loco_restricted.append(None)
                canon_restricted.append(None)
                delta_restricted_per_fold.append(None)
                continue

            r2_canon_r = float(
                r2_score(true_y[keep_mask], comp_canon[keep_mask])
            )
            r2_loco_r = float(
                r2_score(true_y[keep_mask], comp_loco_ct[keep_mask])
            )
            canon_restricted.append(r2_canon_r)
            loco_restricted.append(r2_loco_r)
            delta_restricted_per_fold.append(r2_loco_r - r2_canon_r)

        # Restricted aggregate: mean over folds where r2 is defined.
        valid_deltas = [d for d in delta_restricted_per_fold if d is not None]
        if valid_deltas:
            delta_r2_restricted = float(np.mean(valid_deltas))
        else:
            delta_r2_restricted = None

        per_ct_results.append({
            "cell_type_index": ct,
            "cell_type": ct_name,
            "loco_per_fold_full": loco_full,
            "loco_mean_full": float(np.mean(loco_full)),
            "delta_r2_full": float(np.mean(loco_full)) - canonical_mean_full,
            "delta_r2_full_per_fold": loco_full_delta,
            "n_val_full": n_val_full,
            "loco_per_fold_restricted": loco_restricted,
            "canonical_per_fold_restricted": canon_restricted,
            "delta_r2_restricted": delta_r2_restricted,
            "delta_r2_restricted_per_fold": delta_restricted_per_fold,
            "n_val_restricted": n_val_restricted,
            "n_val_restricted_total": int(sum(n_val_restricted)),
        })

    return {
        "n_folds": len(fold_payloads),
        "canonical_per_fold_full": canonical_per_fold_full,
        "canonical_mean_full": canonical_mean_full,
        "per_cell_type": per_ct_results,
    }


# -----------------------------------------------------------------------
# Coverage stats merge + rank-shift detection
# -----------------------------------------------------------------------


def merge_coverage_stats(
    aggregated: dict[str, Any], coverage_json_path: Path,
) -> dict[str, Any]:
    """Merge ``ct_coverage_full_cohort.json`` stats into per-CT rows."""
    with coverage_json_path.open() as f:
        cov = json.load(f)
    per_ct_cov = cov["per_ct"]
    for row in aggregated["per_cell_type"]:
        nm = row["cell_type"]
        c = per_ct_cov.get(nm, {})
        row["median_cells"] = c.get("median_cells")
        row["n_subj_with_cells"] = c.get("n_subj_with_cells")
        row["zero_frac"] = c.get("zero_frac")
        row["q90_cells"] = c.get("q90_cells")
        row["total_cells"] = c.get("total_cells")
    aggregated["coverage_n_subjects_total"] = cov.get("n_subjects")
    return aggregated


def compute_rank_shift(
    aggregated: dict[str, Any],
) -> dict[str, Any]:
    """For each CT, rank by full-cohort ΔR² and restricted ΔR² (most negative
    = rank 1) and report the rank shift.

    Cell types with ``delta_r2_restricted is None`` (restricted cohort
    too small / undefined R²) are excluded from the restricted ranking
    but kept in the full-cohort ranking with rank-shift = None.
    """
    rows = aggregated["per_cell_type"]
    n_ct = len(rows)

    # Full-cohort rank: most negative ΔR² = rank 1.
    order_full = sorted(
        range(n_ct), key=lambda i: rows[i]["delta_r2_full"]
    )
    full_rank = {idx: r + 1 for r, idx in enumerate(order_full)}

    # Restricted rank: drop None entries.
    valid_idxs = [
        i for i in range(n_ct)
        if rows[i]["delta_r2_restricted"] is not None
    ]
    order_rest = sorted(
        valid_idxs, key=lambda i: rows[i]["delta_r2_restricted"]
    )
    rest_rank = {idx: r + 1 for r, idx in enumerate(order_rest)}

    for i, row in enumerate(rows):
        row["full_rank"] = full_rank[i]
        if i in rest_rank:
            row["restricted_rank"] = rest_rank[i]
            row["rank_shift"] = full_rank[i] - rest_rank[i]
        else:
            row["restricted_rank"] = None
            row["rank_shift"] = None
    aggregated["n_cell_types_with_valid_restricted"] = len(valid_idxs)
    return aggregated


# -----------------------------------------------------------------------
# Output writers (JSON, MD, figure)
# -----------------------------------------------------------------------


def write_json(aggregated: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2))


def _fmt(v: float | int | None, fmt: str = "{:.4f}") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return fmt.format(v)


def write_md(aggregated: dict[str, Any], out_path: Path) -> None:
    rows = aggregated["per_cell_type"]
    rows_sorted = sorted(rows, key=lambda r: r["delta_r2_full"])

    lines: list[str] = []
    lines.append("# Coverage-stratified LOCO ablation\n")
    canon_full = aggregated["canonical_mean_full"]
    canon_pf = aggregated["canonical_per_fold_full"]
    n_folds = aggregated["n_folds"]
    n_total = aggregated.get("coverage_n_subjects_total", "?")
    lines.append(
        f"Canonical 5-fold mean R² (full cohort, N={n_total}): "
        f"**{canon_full:+.4f}**\n"
    )
    lines.append(
        "Per-fold canonical R²: ["
        + ", ".join(f"{x:+.4f}" for x in canon_pf)
        + f"]  (n_folds={n_folds})\n"
    )
    lines.append(
        f"Cell types with a valid restricted ΔR² (≥ 2 val subjects in "
        f"≥ 1 fold): "
        f"**{aggregated['n_cell_types_with_valid_restricted']}**/{len(rows)}\n"
    )

    lines.append("\n## Per-CT ΔR² (sorted by full-cohort ΔR², ascending)\n")
    lines.append(
        "| CT | median cells | n_subj/516 ≥1 cell | zero_frac | "
        "full ΔR² | restricted ΔR² | restricted n_subj (sum across folds) | "
        "rank shift (full→restricted) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows_sorted:
        n_rest = r["n_val_restricted_total"]
        rs = r.get("rank_shift")
        rs_str = "—" if rs is None else f"{rs:+d}"
        lines.append(
            f"| {r['cell_type']} | {_fmt(r.get('median_cells'), '{:.0f}')} | "
            f"{_fmt(r.get('n_subj_with_cells'), '{:.0f}')} | "
            f"{_fmt(r.get('zero_frac'), '{:.4f}')} | "
            f"{r['delta_r2_full']:+.5f} | "
            f"{_fmt(r['delta_r2_restricted'], '{:+.5f}')} | "
            f"{n_rest} | {rs_str} |"
        )
    lines.append("")

    # Rank-shift > 5 entries.
    big_shifts = [
        r for r in rows
        if r.get("rank_shift") is not None and abs(r["rank_shift"]) > 5
    ]
    big_shifts_sorted = sorted(
        big_shifts, key=lambda r: -abs(r["rank_shift"])
    )
    lines.append(
        f"\n## Cell types with |rank shift| > 5 between rankings: "
        f"**{len(big_shifts)}**\n"
    )
    if big_shifts_sorted:
        lines.append(
            "| CT | full rank | restricted rank | shift | median cells |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for r in big_shifts_sorted:
            lines.append(
                f"| {r['cell_type']} | {r['full_rank']} | "
                f"{r['restricted_rank']} | {r['rank_shift']:+d} | "
                f"{_fmt(r.get('median_cells'), '{:.0f}')} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def make_figure(
    aggregated: dict[str, Any],
    out_dir: Path,
    dpi: int = 600,
) -> None:
    """3-panel figure: (A) full vs restricted ΔR² scatter coloured by
    log(median_cells); (B) rank-shift bar chart; (C) ΔR² flip / sign-change
    callouts.
    """
    apply_theme(style="paper")
    rows = aggregated["per_cell_type"]
    n = len(rows)

    full_d = np.array([r["delta_r2_full"] for r in rows])
    rest_d = np.array(
        [
            r["delta_r2_restricted"]
            if r["delta_r2_restricted"] is not None
            else np.nan
            for r in rows
        ]
    )
    median_cells = np.array(
        [r.get("median_cells") if r.get("median_cells") is not None else 0
         for r in rows], dtype=np.float64
    )
    log_mc = np.log10(np.clip(median_cells, 0.5, None))
    ct_names = [r["cell_type"] for r in rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(
        1, 3, figsize=(18, 6), constrained_layout=True
    )
    # Panel A: scatter.
    ax = axs[0]
    valid = np.isfinite(rest_d)
    sc = ax.scatter(
        full_d[valid], rest_d[valid],
        c=log_mc[valid], cmap="viridis", s=70, edgecolor="black",
        linewidth=0.5, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("log10(median cells / subject)")
    lo = float(min(np.nanmin(full_d), np.nanmin(rest_d)))
    hi = float(max(np.nanmax(full_d), np.nanmax(rest_d)))
    pad = 0.05 * max(hi - lo, 1e-3)
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad],
        "--", color="grey", lw=0.8, label="y = x", zorder=1,
    )
    ax.axhline(0, color="black", lw=0.5, zorder=2)
    ax.axvline(0, color="black", lw=0.5, zorder=2)
    # Annotate top-3 |shift| CTs.
    shifts = [r.get("rank_shift") for r in rows]
    abs_shifts = [
        abs(s) if s is not None else -1 for s in shifts
    ]
    top_shift_idx = np.argsort(abs_shifts)[::-1][:5]
    for k in top_shift_idx:
        if not valid[k]:
            continue
        ax.annotate(
            ct_names[k],
            xy=(full_d[k], rest_d[k]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8, fontweight="bold",
        )
    ax.set_xlabel("full-cohort ΔR² (vs canonical, N=516)")
    ax.set_ylabel("restricted ΔR² (subjects with ≥1 cell of CT)")
    ax.set_title("A. Full vs restricted ΔR² (31 CTs)")
    ax.legend(loc="lower right", fontsize=8)

    # Panel B: rank shift bar chart, ordered by full-cohort rank.
    ax = axs[1]
    full_ranks = np.array(
        [r["full_rank"] for r in rows], dtype=int
    )
    rank_shifts = np.array(
        [r.get("rank_shift") if r.get("rank_shift") is not None else 0
         for r in rows], dtype=float
    )
    # Order rows by full-cohort rank (rank 1 = most adversarial / "load-bearing").
    order = np.argsort(full_ranks)
    rank_shifts_ord = rank_shifts[order]
    ct_order_names = [ct_names[i] for i in order]
    colors = ["#d62728" if s < 0 else "#2ca02c" if s > 0 else "#777777"
              for s in rank_shifts_ord]
    ax.barh(
        np.arange(len(order)), rank_shifts_ord,
        color=colors, edgecolor="black", linewidth=0.4,
    )
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(ct_order_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("rank shift (full_rank − restricted_rank)")
    ax.set_title(
        "B. Rank shift between full and restricted ΔR² rankings\n"
        "(rows ordered by full-cohort rank)"
    )

    # Panel C: ΔR² sign flips between full and restricted.
    # A flip = full_d * rest_d < 0 (different signs of ΔR²).
    ax = axs[2]
    flip_mask = (full_d * rest_d < 0) & valid
    rs_arr = np.array(
        [r.get("rank_shift") if r.get("rank_shift") is not None else 0
         for r in rows]
    )
    big_mask = np.abs(rs_arr) > 5
    cat_mask = flip_mask | big_mask
    if cat_mask.any():
        idxs = np.where(cat_mask)[0]
        x_pos = np.arange(len(idxs))
        width = 0.4
        ax.bar(
            x_pos - width / 2, full_d[idxs], width=width,
            color="#1f77b4", label="full ΔR²", edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x_pos + width / 2,
            np.where(np.isfinite(rest_d[idxs]), rest_d[idxs], 0),
            width=width, color="#ff7f0e", label="restricted ΔR²",
            edgecolor="black", linewidth=0.4,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [ct_names[i] for i in idxs], rotation=45,
            ha="right", fontsize=8,
        )
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("ΔR²")
        ax.set_title(
            f"C. CTs with sign-flipped or |Δrank|>5 ΔR² "
            f"(n={int(cat_mask.sum())})"
        )
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(
            0.5, 0.5,
            "No CT has sign-flipped\nΔR² or |Δrank| > 5",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_axis_off()
        ax.set_title("C. Sign-flip / large-shift callouts")

    fig.suptitle(
        "Coverage-stratified LOCO ablation (EXP-043)",
        fontsize=13, y=1.02,
    )
    png = out_dir / "fig_coverage_stratified_loco.png"
    pdf = out_dir / "fig_coverage_stratified_loco.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--config", type=Path,
        default=_WORKTREE_ROOT / "configs/resdec_mhe/canonical.yaml",
    )
    p.add_argument(
        "--canonical-dir", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument(
        "--tabpfn-dir", type=Path,
        default=_WORKTREE_ROOT / "data/canonical",
    )
    p.add_argument(
        "--splits-path", type=Path,
        default=_WORKTREE_ROOT / "outputs/splits.json",
    )
    p.add_argument(
        "--precomputed-dir", type=Path,
        default=_WORKTREE_ROOT / "data/precomputed",
    )
    p.add_argument(
        "--coverage-json", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/ct_coverage_full_cohort.json"
        ),
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-cell-types", type=int, default=31)
    p.add_argument(
        "--min-cells-threshold", type=int, default=1,
        help="Subject is kept in restricted cohort if cell_counts[CT] >= "
             "this threshold (default 1).",
    )
    p.add_argument("--device", default="cuda:1")
    p.add_argument(
        "--out-data-dir", type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability",
        help="Directory for coverage_stratified_loco.{json,md} (note: no "
             "subfolder, mirrors EXP-013 LOCO output style).",
    )
    p.add_argument(
        "--out-fig-dir", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/figures"
            / "coverage_stratified_loco"
        ),
    )
    p.add_argument(
        "--smoke-fold-only", type=int, default=None,
        help="If set, run only the given fold (for fast smoke tests).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    out_data_dir = Path(args.out_data_dir)
    out_fig_dir = Path(args.out_fig_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t_start = time.time()

    # Build subject-cell_count map (one walk over R*.pt).
    logger.info("Building subject cell_counts map from %s", args.precomputed_dir)
    sids_all, counts_all = build_subject_cell_counts(
        Path(args.precomputed_dir), n_cell_types=args.n_cell_types,
    )
    subject_counts: dict[str, np.ndarray] = {
        sid: counts_all[i] for i, sid in enumerate(sids_all)
    }
    logger.info(
        "loaded cell_counts for %d subjects (matrix shape %s)",
        len(sids_all), counts_all.shape,
    )

    cfg = OmegaConf.merge(
        OmegaConf.load(_WORKTREE_ROOT / "configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"

    # Folds — sequential on a single device.
    fold_payloads: list[dict[str, Any]] = []
    fold_iter = (
        [args.smoke_fold_only] if args.smoke_fold_only is not None
        else range(args.n_folds)
    )
    for fold in fold_iter:
        fold_payloads.append(_run_fold(
            fold=fold, cfg=cfg,
            canonical_dir=Path(args.canonical_dir),
            splits_path=Path(args.splits_path),
            tabpfn_dir=Path(args.tabpfn_dir),
            device=device,
            n_cell_types=args.n_cell_types,
        ))

    aggregated = aggregate_loco(
        fold_payloads, subject_counts,
        n_cell_types=args.n_cell_types,
        min_cells_threshold=args.min_cells_threshold,
    )
    aggregated = merge_coverage_stats(aggregated, Path(args.coverage_json))
    aggregated = compute_rank_shift(aggregated)

    elapsed_min = round((time.time() - t_start) / 60, 2)
    aggregated["provenance"] = {
        "n_folds": args.n_folds,
        "n_cell_types": args.n_cell_types,
        "min_cells_threshold": args.min_cells_threshold,
        "device": str(device),
        "elapsed_min": elapsed_min,
        "git_commit": git_sha(_WORKTREE_ROOT),
        "config_path": str(args.config),
        "canonical_dir": str(args.canonical_dir),
        "smoke_fold_only": args.smoke_fold_only,
    }

    json_path = out_data_dir / "coverage_stratified_loco.json"
    md_path = out_data_dir / "coverage_stratified_loco.md"
    write_json(aggregated, json_path)
    write_md(aggregated, md_path)
    make_figure(aggregated, out_fig_dir)
    logger.info(
        "wrote %s, %s, %s/fig_coverage_stratified_loco.png; elapsed %.2f min",
        json_path, md_path, out_fig_dir, elapsed_min,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
