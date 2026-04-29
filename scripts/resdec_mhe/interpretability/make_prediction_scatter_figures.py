"""Lab-meeting slide-4 prediction scatter figures.

Renders TWO figures from canonical 5-fold val predictions joined to
ROSMAP metadata:

    1. fig_predicted_vs_actual_addx — colored by AD diagnosis (cogdx).
    2. fig_predicted_vs_actual_sex  — colored by sex (msex).

Both figures: square scatter + KDE marginals on top + right + identity
line + fitted regression line + R^2 in legend + RMSE/MAE/R^2 annotation.

Reads:
  - outputs/canonical/p5_canonical_seed42/fold{0..4}/val_predictions_best.npz
       keys: subject_ids (R-prefix), predictions, targets
  - data/metadata_ROSMAP/metadata.csv
       join key: ROSMAP_IndividualID

Note on Y-target semantics: ``predictions`` is already the COMPOSITE
(Σ f̂_k + ŷ_tabpfn) — do NOT add y_tabpfn. ``targets`` is the actual
held-out cognition residual.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.prediction_plots import plot_predicted_vs_actual
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)


# ROSMAP cogdx coding — from `dcfdx_lv` and the ROSMAP data dictionary.
COGDX_LABEL = {
    1: "NCI",
    2: "MCI",
    3: "AD",
    4: "AD + other",
    5: "Other dementia",
    6: "Other",
}

# Palettes — drawn from the project theme color tokens.
# AD-dx: ordinal severity → tab10 sequence (no NaN expected after merge).
COGDX_PALETTE = {
    "NCI":            "#2ca02c",  # green — cognitively normal
    "MCI":            "#ff7f0e",  # orange — mild impairment
    "AD":             "#d62728",  # red — AD
    "AD + other":     "#9467bd",  # purple — AD + other dementia
    "Other dementia": "#8c564b",  # brown — non-AD dementia
    "Other":          "#7f7f7f",  # gray — other
}

# Sex: F/M, two distinct theme accent colors.
SEX_PALETTE = {
    "F": "#E76A7B",  # ACCENT_CORAL
    "M": "#189584",  # ACCENT_TEAL
}


def load_predictions(canonical_dir: Path, n_folds: int) -> dict[str, np.ndarray]:
    """Concatenate val_predictions_best.npz across folds."""
    preds, actuals, sids, fold_ids = [], [], [], []
    for f in range(n_folds):
        p_path = canonical_dir / f"fold{f}/val_predictions_best.npz"
        if not p_path.exists():
            logger.warning("missing %s", p_path)
            continue
        d = np.load(p_path, allow_pickle=True)
        preds.append(np.asarray(d["predictions"], dtype=np.float64))
        actuals.append(np.asarray(d["targets"], dtype=np.float64))
        sids.append(np.asarray(d["subject_ids"]))
        fold_ids.append(np.full(d["predictions"].shape[0], f, dtype=np.int64))
    if not preds:
        raise RuntimeError(f"no per-fold predictions under {canonical_dir}")
    return {
        "predictions": np.concatenate(preds),
        "actual":      np.concatenate(actuals),
        "subject_id":  np.concatenate(sids),
        "fold":        np.concatenate(fold_ids),
    }


def join_metadata(pred_df: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    """Inner-join predictions ↔ ROSMAP metadata on R-prefix individual id."""
    md = pd.read_csv(metadata_csv)
    if "ROSMAP_IndividualID" not in md.columns:
        raise KeyError("metadata.csv missing ROSMAP_IndividualID column")
    cols = ["ROSMAP_IndividualID", "cogdx", "msex"]
    md = md[cols].rename(columns={"ROSMAP_IndividualID": "subject_id"})
    merged = pred_df.merge(md, on="subject_id", how="left")
    return merged


def make_addx_figure(
    df: pd.DataFrame,
    out_stem: Path,
    figsize: tuple[float, float] = (6.0, 6.0),
) -> dict[str, float]:
    """Predicted-vs-actual colored by AD diagnosis (cogdx).

    Per user pref for the lab-meeting deliverable: no identity line, no
    in-axes legend (covers the data), small scatter points. A separate
    legend-only PNG is also written so the audience can see the colour
    key in PowerPoint.
    """
    cogdx_int = df["cogdx"].astype("Int64")
    labels = cogdx_int.map(COGDX_LABEL).fillna("Unknown").to_numpy()
    fig = plot_predicted_vs_actual(
        predicted_mean=df["predictions"].to_numpy(),
        actual=df["actual"].to_numpy(),
        figsize=figsize,
        title="",
        add_marginals=True,
        color_by=labels,
        color_label="AD diagnosis",
        color_palette=COGDX_PALETTE,
        show_identity=False,
        show_legend=False,
        scatter_size=10,
    )
    _save_png(fig, out_stem)
    plt.close(fig)
    _save_legend_only(
        labels=labels,
        palette=COGDX_PALETTE,
        title="AD diagnosis",
        out_stem=out_stem.with_name(out_stem.name + "_legend"),
    )
    return _summary_metrics(df["predictions"].to_numpy(), df["actual"].to_numpy())


def make_sex_figure(
    df: pd.DataFrame,
    out_stem: Path,
    figsize: tuple[float, float] = (6.0, 6.0),
) -> dict[str, float]:
    """Predicted-vs-actual colored by sex (msex). See ``make_addx_figure``."""
    sex_labels = df["msex"].map({0: "F", 1: "M"}).fillna("Unknown").to_numpy()
    fig = plot_predicted_vs_actual(
        predicted_mean=df["predictions"].to_numpy(),
        actual=df["actual"].to_numpy(),
        figsize=figsize,
        title="",
        add_marginals=True,
        color_by=sex_labels,
        color_label="Sex",
        color_palette=SEX_PALETTE,
        show_identity=False,
        show_legend=False,
        scatter_size=10,
    )
    _save_png(fig, out_stem)
    plt.close(fig)
    _save_legend_only(
        labels=sex_labels,
        palette=SEX_PALETTE,
        title="Sex",
        out_stem=out_stem.with_name(out_stem.name + "_legend"),
    )
    return _summary_metrics(df["predictions"].to_numpy(), df["actual"].to_numpy())


def _save_png(fig: plt.Figure, stem: Path) -> None:
    """Save fig as PNG at 600 DPI (PDF intentionally dropped per user pref)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


def _save_legend_only(
    *,
    labels: np.ndarray,
    palette: dict,
    title: str,
    out_stem: Path,
) -> None:
    """Render and save just the categorical legend as a stand-alone PNG.

    The user wants the predicted-vs-actual scatter without any in-axes
    legend covering the data. Pasting this side-by-side in PowerPoint
    preserves the colour key.
    """
    from matplotlib.patches import Patch
    categories = list(dict.fromkeys(np.asarray(labels).tolist()))
    handles = [
        Patch(facecolor=palette.get(c, "#777777"), edgecolor="white", label=str(c))
        for c in categories
    ]
    fig_l, ax_l = plt.subplots(figsize=(2.5, 2.0))
    ax_l.axis("off")
    ax_l.legend(
        handles=handles, title=title,
        loc="center", frameon=True, fontsize=10, title_fontsize=11,
    )
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig_l.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig_l)


def _summary_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Pooled-across-folds R^2, RMSE, MAE."""
    valid = np.isfinite(pred) & np.isfinite(actual)
    p = pred[valid]
    a = actual[valid]
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    return {"r2": r2, "rmse": rmse, "mae": mae, "n": int(valid.sum())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-dir",
        default="outputs/canonical/p5_canonical_seed42",
        help="Directory containing fold{N}/val_predictions_best.npz",
    )
    p.add_argument(
        "--metadata-csv",
        default="data/metadata_ROSMAP/metadata.csv",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/prediction",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()

    canonical_dir = Path(args.canonical_dir)
    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)

    preds = load_predictions(canonical_dir, args.n_folds)
    pred_df = pd.DataFrame(preds)
    logger.info("loaded %d predictions across folds %s",
                len(pred_df), sorted(pred_df["fold"].unique().tolist()))

    df = join_metadata(pred_df, metadata_csv)
    n_missing_cogdx = int(df["cogdx"].isna().sum())
    n_missing_msex = int(df["msex"].isna().sum())
    if n_missing_cogdx or n_missing_msex:
        logger.warning("metadata coverage: %d missing cogdx, %d missing msex",
                       n_missing_cogdx, n_missing_msex)

    addx_metrics = make_addx_figure(df, out_dir / "fig_predicted_vs_actual_addx")
    sex_metrics = make_sex_figure(df, out_dir / "fig_predicted_vs_actual_sex")

    logger.info("addx figure: r2=%.4f rmse=%.4f mae=%.4f n=%d",
                addx_metrics["r2"], addx_metrics["rmse"],
                addx_metrics["mae"], addx_metrics["n"])
    logger.info("sex  figure: r2=%.4f rmse=%.4f mae=%.4f n=%d",
                sex_metrics["r2"], sex_metrics["rmse"],
                sex_metrics["mae"], sex_metrics["n"])
    logger.info("output dir: %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
