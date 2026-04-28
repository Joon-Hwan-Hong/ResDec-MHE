"""Aggregate the SAE sweep grid into a Pareto frontier + best-config summary.

Reads ``outputs/redesign/sae/{architecture}/{layer}/exp{E}_k{K}_seed{S}/reconstruction_metrics.json``
for every completed sweep config; outputs:

* ``outputs/redesign/sae/sweep_summary.json`` — full grid as JSON
* ``outputs/redesign/sae/sweep_summary.csv``  — flat table for plotting
* ``outputs/redesign/sae/figures/sae_pareto.{png,pdf}`` — L0 vs FVE Pareto plot per (arch, layer)

Best config selection: per (architecture, layer), pick the config with the highest FVE
that satisfies dead_fraction < 0.5 and L0 < 0.5 × dictionary_size. If none, pick the
highest-FVE config with the largest L0.

Cross-fold stability: per config, ratio of mean per-fold FVE to overall FVE. Closer to
1.0 means the pooled-fold SAE generalizes; far below 1.0 means per-fold variation is
large (Option D fallback to per-fold SAE may be needed).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Single source of truth for the canonical sweep grid. Used both to compute
# ``n_total`` and (in principle) by any driver that wants to enumerate the
# grid without copy-pasting literals.
ARCHITECTURES: tuple[str, ...] = ("topk", "batch_topk")
LAYERS: tuple[str, ...] = ("attended", "fused")
EXPANSIONS: tuple[int, ...] = (8, 16, 32)
K_VALUES: tuple[int, ...] = (4, 8, 16, 32, 64)
N_TOTAL: int = (
    len(ARCHITECTURES) * len(LAYERS) * len(EXPANSIONS) * len(K_VALUES)
)


def _read_input_dim(run_dir: Path) -> int:
    """Read decoder input dim ``n`` from ``sae_model.npz`` in ``run_dir``.

    Falls back to ``stat_fraction_active.shape[0] / expansion`` when the
    decoder array is unavailable; raises if neither is readable. Avoids the
    previous hardcoded ``n = 64`` that broke as soon as the architecture's
    ``d_fused`` changed.
    """
    npz_path = run_dir / "sae_model.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"sae_model.npz missing — cannot derive input dim n: {npz_path}"
        )
    npz = np.load(npz_path, allow_pickle=True)
    if "W_dec" in npz.files:
        return int(np.asarray(npz["W_dec"]).shape[0])
    if "stat_fraction_active" in npz.files:
        return int(np.asarray(npz["stat_fraction_active"]).shape[0])
    raise KeyError(
        f"npz at {npz_path} carries neither 'W_dec' nor "
        "'stat_fraction_active'; cannot derive input dim n."
    )


def gather_sweep(sae_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for arch_dir in sorted(sae_root.iterdir()):
        if not arch_dir.is_dir() or arch_dir.name not in set(ARCHITECTURES):
            continue
        for layer_dir in sorted(arch_dir.iterdir()):
            if not layer_dir.is_dir() or layer_dir.name not in set(LAYERS):
                continue
            for run_dir in sorted(layer_dir.iterdir()):
                metrics_file = run_dir / "reconstruction_metrics.json"
                if not metrics_file.exists():
                    continue
                d = json.loads(metrics_file.read_text())
                cfg = d["config"]
                full = d["full"]
                per_fold = d.get("per_fold", {})
                per_fold_fves = [v["fve"] for v in per_fold.values() if "fve" in v]
                # Read input dim n from the SAE npz; this avoids hard-coding
                # n = d_fused = 64 and lets the aggregator track changes to
                # fused-dim if/when ResDec-MHE is reconfigured.
                try:
                    n_input = _read_input_dim(run_dir)
                except (FileNotFoundError, KeyError):
                    n_input = None
                rows.append({
                    "architecture": cfg["architecture"],
                    "layer": d["layer"],
                    "expansion": cfg["expansion"],
                    "k": cfg["k"],
                    "seed": cfg["seed"],
                    "n_input": n_input,
                    "n_train_rows": d["n_train_rows"],
                    "fve_full": full["fve"],
                    "mse_full": full["mse"],
                    "l0_mean_full": full["l0_mean"],
                    "dead_fraction_full": full["dead_fraction"],
                    "per_fold_fve_mean": float(np.mean(per_fold_fves)) if per_fold_fves else None,
                    "per_fold_fve_std": float(np.std(per_fold_fves, ddof=1)) if len(per_fold_fves) > 1 else None,
                    "cross_fold_stability_ratio": (
                        float(np.mean(per_fold_fves) / max(full["fve"], 1e-12))
                        if per_fold_fves else None
                    ),
                    "train_minutes": d.get("train_minutes"),
                    "config_id": f"{cfg['architecture']}/{d['layer']}/exp{cfg['expansion']}_k{cfg['k']}",
                })
    return pd.DataFrame(rows)


def select_best_per_arch_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Best config per (architecture, layer): max FVE under dead<0.5 AND L0/m<0.5; else max FVE."""
    best_rows: list[dict] = []
    for (arch, layer), g in df.groupby(["architecture", "layer"]):
        g = g.copy()
        # Use per-row n_input read from the SAE npz; falls back to 64 only
        # when n_input is unavailable for legacy npz files. This avoids
        # silently encoding d_fused=64 as a literal.
        n_per_row = g["n_input"].fillna(64).astype(int)
        g["dict_size"] = g["expansion"].astype(int) * n_per_row
        g["l0_fraction"] = g["l0_mean_full"] / g["dict_size"]
        sane = g[(g["dead_fraction_full"] < 0.5) & (g["l0_fraction"] < 0.5)]
        if len(sane):
            best = sane.sort_values("fve_full", ascending=False).iloc[0]
        else:
            best = g.sort_values("fve_full", ascending=False).iloc[0]
        best_rows.append(dict(best))
    return pd.DataFrame(best_rows)


def plot_pareto(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(5.5 * len(layers), 4.0), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        sub = df[df["layer"] == layer]
        for arch, g in sub.groupby("architecture"):
            ax.scatter(
                g["l0_mean_full"], g["fve_full"],
                s=60, label=arch, alpha=0.85,
            )
            for _, row in g.iterrows():
                ax.annotate(
                    f"e{row['expansion']}_k{row['k']}",
                    (row["l0_mean_full"], row["fve_full"]),
                    fontsize=6, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points",
                )
        ax.set_xlabel("L0 (mean active features per sample)")
        ax.set_ylabel("Fraction-of-variance-explained")
        ax.set_title(f"layer={layer}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"SAE sweep: L0 vs FVE Pareto (n_configs={len(df)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sae-root", default="outputs/redesign/sae")
    p.add_argument(
        "--summary-json",
        default="outputs/redesign/sae/sweep_summary.json",
    )
    p.add_argument(
        "--summary-csv",
        default="outputs/redesign/sae/sweep_summary.csv",
    )
    p.add_argument(
        "--pareto-png",
        default="outputs/redesign/sae/figures/sae_pareto.png",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    sae_root = Path(args.sae_root)
    df = gather_sweep(sae_root)
    if df.empty:
        logger.error("no completed sweep configs found under %s", sae_root)
        return 1
    logger.info("gathered %d completed configs", len(df))
    df.to_csv(Path(args.summary_csv), index=False)

    best = select_best_per_arch_layer(df)
    summary = {
        "n_configs_completed": len(df),
        "n_total": int(N_TOTAL),
        "per_arch_layer_best": best.drop(columns=["dict_size", "l0_fraction"]).to_dict(orient="records"),
        "all_configs": df.to_dict(orient="records"),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, default=float))
    logger.info("wrote %s + %s", args.summary_json, args.summary_csv)

    plot_pareto(df, Path(args.pareto_png))
    logger.info("wrote Pareto plot at %s (+ .pdf)", args.pareto_png)

    print("\nBest config per (architecture, layer):")
    print(best[["architecture", "layer", "expansion", "k", "fve_full", "l0_mean_full",
                "dead_fraction_full", "cross_fold_stability_ratio"]].to_string(index=False))

    print("\nCross-fold stability summary (ratio per_fold_fve_mean / fve_full):")
    print(df.groupby(["architecture", "layer"])["cross_fold_stability_ratio"]
          .agg(["min", "median", "max"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
