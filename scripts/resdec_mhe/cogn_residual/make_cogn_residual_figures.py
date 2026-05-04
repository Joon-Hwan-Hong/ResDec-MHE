"""Variant-specific figures (Task 13 of cogn-residual-variant plan).

Five figure types per plan:
  1. Residualized-target distribution histogram with resilient/vulnerable tail
     labels (top/bottom 25 %).
  2. Variant predicted-vs-actual scatter (per fold) + R² annotated.
  3. Variant A perm-null collapse: histogram of N=20 null R² + canonical line.
  4. Cross-variant DCR slope chart: per-method canonical→variant CT-rank shift.
  5. Cross-variant DAE volcano per gradient method: mean_diff vs -log10(padj_bh).

All figures use apply_theme("paper") + save_fig (600 DPI by theme default).
Per the no-default-protagonist rule: no specific CT names in titles or callouts;
labels are abstract entity references where possible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.visualization.theme import apply_theme, save_fig  # noqa: E402


def _load_variant_target(variant: str) -> tuple[np.ndarray, list[str]]:
    """Pool per-fold residualized targets per subject (variants) or
    metadata cogn_global (canonical).
    """
    if variant == "canonical":
        meta = pd.read_csv(_ROOT / "data/metadata_ROSMAP/metadata.csv")
        splits = json.loads((_ROOT / "outputs/splits.json").read_text())
        cohort = sorted({s for f in splits["folds"] for s in f["train"] + f["val"]})
        m = meta.set_index("ROSMAP_IndividualID")["cogn_global"]
        return np.array([m.get(s, np.nan) for s in cohort]), cohort

    cache = _ROOT / f"outputs/canonical/cogn_residual/{variant}/cache"
    splits = json.loads((_ROOT / "outputs/splits.json").read_text())
    cohort = sorted({s for f in splits["folds"] for s in f["train"] + f["val"]})
    fold_arr = []
    for f in range(len(splits["folds"])):
        d = np.load(cache / f"residual_target_fold{f}.npz", allow_pickle=True)
        m = {s: float(t) for s, t in zip(d["subject_ids"].tolist(), d["target"])}
        fold_arr.append([m.get(s, np.nan) for s in cohort])
    return np.nanmean(np.array(fold_arr), axis=0), cohort


def _fig_residual_distribution(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (variant, label) in zip(
        axes,
        [("canonical", "Canonical (raw cogn_global)"),
         ("gpath_only", "Variant A (residualized: gpath only)"),
         ("multi_axis", "Variant B (residualized: gpath + tangsqrt + amylsqrt)")],
    ):
        target, _ = _load_variant_target(variant)
        finite = target[np.isfinite(target)]
        n_take = max(1, int(round(len(finite) * 0.25)))
        order = np.argsort(finite)
        bottom = finite[order[:n_take]]
        top = finite[order[-n_take:]]
        ax.hist(finite, bins=40, color="#999999", alpha=0.6,
                edgecolor="black", linewidth=0.4)
        ax.hist(top, bins=40, color="#2ca02c", alpha=0.7,
                edgecolor="black", linewidth=0.4, label=f"resilient (n={len(top)})")
        ax.hist(bottom, bins=40, color="#d62728", alpha=0.7,
                edgecolor="black", linewidth=0.4, label=f"vulnerable (n={len(bottom)})")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("target value")
        ax.set_ylabel("subjects")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Cohort target distribution by variant + quartile labels", fontsize=11)
    save_fig(fig, out_dir / "fig_target_distribution")


def _fig_predicted_vs_actual(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (variant, label) in zip(
        axes,
        [("gpath_only", "Variant A: gpath_only residualized"),
         ("multi_axis", "Variant B: multi_axis residualized")],
    ):
        d = json.loads(
            (_ROOT / f"outputs/canonical/cogn_residual/{variant}/p5_seed42/best_vs_tabpfn_summary.json").read_text()
        )
        for f in d["per_fold"]:
            preds_path = (
                _ROOT / f"outputs/canonical/cogn_residual/{variant}/p5_seed42/fold{f['fold']}/val_predictions_best.npz"
            )
            npz = np.load(preds_path, allow_pickle=True)
            ax.scatter(npz["targets"], npz["predictions"], s=10, alpha=0.6,
                       label=f"fold {f['fold']}: R²={f['ours']['r2']:+.3f}")
        lim = [-2.5, 2.5]
        ax.plot(lim, lim, "k--", linewidth=0.6, alpha=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true residualized target")
        ax.set_ylabel("predicted")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    save_fig(fig, out_dir / "fig_predicted_vs_actual")


def _fig_perm_null_collapse(out_dir: Path) -> None:
    summary = json.loads(
        (_ROOT / "outputs/canonical/cogn_residual/gpath_only/permutation_test_n20/permutation_summary.json").read_text()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    canon = summary["canonical_mean_r2"]
    n = summary["n_permutations"]
    null_mean = summary["null_mean"]
    null_std = summary["null_std"]
    z = summary["z_under_null"]
    p = summary["p_value_one_sided"]

    rows = []
    for shard in ("shard_a", "shard_b"):
        sj = json.loads(
            (_ROOT / f"outputs/canonical/cogn_residual/gpath_only/permutation_test_n20/{shard}/permutation_results.json").read_text()
        )
        rows.extend(r["mean_r2_true"] for r in sj if "mean_r2_true" in r)
    null_means = np.array(rows)

    ax.hist(null_means, bins=15, color="#999999", alpha=0.7,
            edgecolor="black", linewidth=0.4, label=f"N={n} null perms")
    ax.axvline(canon, color="#1f77b4", linewidth=2,
               label=f"canonical R²={canon:+.4f}")
    ax.axvline(null_mean, color="#d62728", linewidth=1, linestyle="--",
               label=f"null mean={null_mean:+.4f} ± {null_std:.4f}")
    ax.set_xlabel("mean R² (vs TRUE residualized target)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Variant A permutation null (N={n})  →  z={z:+.2f}, p={p:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    save_fig(fig, out_dir / "fig_perm_null_collapse")


def _fig_dcr_slope_chart(out_dir: Path) -> None:
    dcr = json.loads(
        (_ROOT / "outputs/canonical/cogn_residual/differential/gpath_only/dcr_canonical_vs_gpath_only.json").read_text()
    )
    methods = sorted(dcr.keys())
    rhos = [dcr[m]["spearman_rho"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if r < 0.5 else "#2ca02c" for r in rhos]
    ax.barh(methods, rhos, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.5, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ (canonical CT-rank vs Variant A CT-rank)")
    ax.set_title(
        "Per-method CT-rank preservation across canonical → Variant A",
        fontsize=10,
    )
    for i, (m, r) in enumerate(zip(methods, rhos)):
        ax.text(r + (0.02 if r > 0 else -0.02), i, f"{r:+.3f}",
                va="center", ha="left" if r > 0 else "right", fontsize=8)
    save_fig(fig, out_dir / "fig_dcr_slope_chart")


def _fig_dae_volcano(out_dir: Path) -> None:
    methods = ["captum_ig", "gradient_shap", "smoothgrad"]
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    for ax, m in zip(axes, methods):
        path = _ROOT / f"outputs/canonical/cogn_residual/differential/gpath_only/dae_canonical_vs_gpath_only__{m}.csv"
        if not path.is_file():
            ax.text(0.5, 0.5, f"missing: {m}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(m, fontsize=10)
            continue
        df = pd.read_csv(path)
        ax.scatter(df["mean_diff"], -np.log10(np.maximum(df["padj_bh"], 1e-300)),
                   s=2, alpha=0.4, color="#666666")
        ax.axhline(-np.log10(0.05), color="red", linewidth=0.6, linestyle="--",
                   label="padj=0.05")
        n_sig = int((df["padj_bh"] < 0.05).sum())
        ax.set_title(f"{m} (n_sig={n_sig}/{len(df)})", fontsize=10)
        ax.set_xlabel("variant - canonical mean |attribution|")
        ax.set_ylabel("-log10 padj_bh")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("DAE volcanoes: canonical vs Variant A per gradient method",
                 fontsize=11)
    save_fig(fig, out_dir / "fig_dae_volcano")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path,
                   default=_ROOT / "outputs/canonical/cogn_residual/figures")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_theme("paper")

    print("[1/5] target distribution")
    _fig_residual_distribution(args.out_dir)
    print("[2/5] predicted-vs-actual")
    _fig_predicted_vs_actual(args.out_dir)
    print("[3/5] perm null collapse")
    _fig_perm_null_collapse(args.out_dir)
    print("[4/5] DCR slope chart")
    _fig_dcr_slope_chart(args.out_dir)
    print("[5/5] DAE volcano")
    _fig_dae_volcano(args.out_dir)
    print(f"\nwrote 5 figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
