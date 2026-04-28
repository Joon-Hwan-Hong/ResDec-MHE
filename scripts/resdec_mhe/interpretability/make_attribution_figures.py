"""Orchestrator: render attribution figures from canonical artefacts.

Calls four functions from ``src.visualization.attribution_plots``:
  - subject waterfall (one example subject)
  - per-subject TabPFN-vs-residual stacked bar
  - resilience signature radar (top-N attribution genes by residual quartile)
  - per-prediction-quintile attribution heatmap

Inputs (defaults; CLI-overridable):
  --canonical-dir   outputs/canonical/p5_canonical_seed42
  --captum-npz      outputs/canonical/interpretability/captum_ig/composite_attributions.npz
  --residual-csv    outputs/canonical/interpretability/residual_per_subject.csv
  --tabpfn-dir      data/canonical
  --out-dir         outputs/canonical/interpretability/figures/attribution
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

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.attribution_plots import (
    plot_attribution_stability_heatmap,
    plot_captum_de_concordance,
    plot_per_quintile_attribution,
    plot_resilience_signature_radar,
    plot_subject_waterfall,
    plot_tabpfn_vs_residual_stack,
)
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)


def _load_canonical(canonical_dir: Path, n_folds: int = 5) -> dict:
    subj, preds, true_y = [], [], []
    for f in range(n_folds):
        p = canonical_dir / f"fold{f}/val_predictions_best.npz"
        if not p.exists():
            logger.warning("missing %s", p)
            continue
        d = np.load(p, allow_pickle=True)
        subj.extend([str(s) for s in d["subject_ids"]])
        preds.extend(np.asarray(d["predictions"], dtype=np.float64).tolist())
        true_y.extend(np.asarray(d["targets"], dtype=np.float64).tolist())
    return {
        "subject_ids": np.array(subj),
        "predictions": np.array(preds),
        "targets": np.array(true_y),
    }


def _load_tabpfn_subj_to_pred(tabpfn_dir: Path, n_folds: int = 5) -> dict:
    out = {}
    for f in range(n_folds):
        p = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        for s, v in zip(d["val_subject_ids"], d["y_tabpfn"]):
            out[str(s)] = float(v)
    return out


def _load_ct_names(captum_summary_path: Path) -> list[str] | None:
    """Return axis-aligned cell-type names (index-ordered).

    Uses ``src.data.constants.CELL_TYPE_ORDER`` (same convention as the
    pseudobulk loader, model forward, and all per-CT indexing elsewhere).
    The captum summary's ``cell_types_ranked_by_total_attribution`` is
    ranked by attribution magnitude, NOT by index — wrong for axis-
    aligned labeling of per-CT plots.
    """
    del captum_summary_path  # unused; kept for backward compat
    from src.data.constants import CELL_TYPE_ORDER
    return list(CELL_TYPE_ORDER)


def _load_gene_names(captum_summary_path: Path) -> list[str] | None:
    p = Path("data/precomputed/gene_names.npy")
    if p.exists():
        return list(np.load(p, allow_pickle=True))
    if captum_summary_path.exists():
        s = json.loads(captum_summary_path.read_text())
        return s.get("gene_names") or s.get("genes")
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--canonical-dir", default="outputs/canonical/p5_canonical_seed42")
    p.add_argument(
        "--captum-npz",
        default="outputs/canonical/interpretability/captum_ig/composite_attributions.npz",
    )
    p.add_argument(
        "--residual-csv",
        default="outputs/canonical/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--tabpfn-dir", default="data/canonical")
    p.add_argument(
        "--out-dir", default="outputs/canonical/interpretability/figures/attribution",
    )
    p.add_argument(
        "--de-dir",
        default="outputs/canonical/interpretability/de_resilient_vs_vulnerable",
        help="Directory with per-CT DE CSVs (CT_00_de.csv … CT_NN_de.csv); "
             "used for the Captum-vs-DE concordance plot.",
    )
    p.add_argument("--example-subject", default=None)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canon = _load_canonical(Path(args.canonical_dir), args.n_folds)
    if canon["subject_ids"].size == 0:
        logger.error("no canonical predictions found; aborting")
        return 1
    tabpfn_map = _load_tabpfn_subj_to_pred(Path(args.tabpfn_dir), args.n_folds)
    composite_tabpfn = np.array(
        [tabpfn_map.get(s, np.nan) for s in canon["subject_ids"]],
    )
    captum_npz = Path(args.captum_npz)
    captum_summary = captum_npz.parent / "composite_attribution_summary.json"
    ct_names = _load_ct_names(captum_summary)
    gene_names = _load_gene_names(captum_summary)
    captum_data = (
        {k: np.load(captum_npz, allow_pickle=True)[k]
         for k in np.load(captum_npz, allow_pickle=True).files}
        if captum_npz.exists() else None
    )
    residual_df = (
        pd.read_csv(args.residual_csv) if Path(args.residual_csv).exists() else None
    )
    rendered = []

    # 1. Subject waterfall.
    if (
        captum_data is not None
        and "attributions" in captum_data
        and gene_names is not None
        and ct_names is not None
    ):
        attrs_all = captum_data["attributions"]
        subj_attr_ids = (
            list(captum_data["subject_ids"]) if "subject_ids" in captum_data
            else list(canon["subject_ids"])
        )
        sel_id = args.example_subject
        if sel_id is None:
            residual_contrib_arr = canon["predictions"] - composite_tabpfn
            order = np.argsort(np.nan_to_num(residual_contrib_arr, nan=0.0))[::-1]
            sel_id = str(canon["subject_ids"][order[0]])
        if sel_id in [str(s) for s in subj_attr_ids]:
            i_attr = [str(s) for s in subj_attr_ids].index(sel_id)
            i_comp = list(canon["subject_ids"]).index(sel_id)
            try:
                fig = plot_subject_waterfall(
                    sel_id, attrs_all[i_attr],
                    cell_type_names=ct_names, gene_names=gene_names,
                    tabpfn_pred=float(composite_tabpfn[i_comp]),
                    composite_pred=float(canon["predictions"][i_comp]),
                    true_y=float(canon["targets"][i_comp]),
                    save_path=out_dir / "fig_subject_waterfall",
                )
                plt.close(fig)
                rendered.append("fig_subject_waterfall")
            except (ValueError, KeyError) as exc:
                logger.warning("waterfall: %s", exc)

    # 2. TabPFN-vs-residual stack.
    if np.isfinite(composite_tabpfn).any():
        try:
            fig = plot_tabpfn_vs_residual_stack(
                canon["subject_ids"], composite_tabpfn,
                canon["predictions"], canon["targets"],
                save_path=out_dir / "fig_tabpfn_vs_residual_stack",
            )
            plt.close(fig)
            rendered.append("fig_tabpfn_vs_residual_stack")
        except ValueError as exc:
            logger.warning("stack: %s", exc)

    # 3. Resilience signature radar.
    if (
        captum_data is not None
        and "attributions" in captum_data
        and residual_df is not None
        and gene_names is not None and ct_names is not None
    ):
        try:
            attrs_all = captum_data["attributions"]
            res_map = dict(zip(
                residual_df["ROSMAP_IndividualID"].astype(str),
                residual_df["residual"].astype(float),
            ))
            subj_attr_ids = (
                list(captum_data["subject_ids"])
                if "subject_ids" in captum_data else list(canon["subject_ids"])
            )
            res_per = np.array([res_map.get(str(s), np.nan) for s in subj_attr_ids])
            fig = plot_resilience_signature_radar(
                attrs_all, res_per, ct_names, gene_names,
                save_path=out_dir / "fig_resilience_signature_radar",
            )
            plt.close(fig)
            rendered.append("fig_resilience_signature_radar")
        except (ValueError, KeyError) as exc:
            logger.warning("radar: %s", exc)

    # 4. Per-prediction-quintile attribution heatmap.
    if (
        captum_data is not None
        and "attributions" in captum_data
        and ct_names is not None
    ):
        try:
            attrs_all = captum_data["attributions"]
            subj_attr_ids = (
                list(captum_data["subject_ids"])
                if "subject_ids" in captum_data else list(canon["subject_ids"])
            )
            subj_to_pred = dict(zip(
                [str(s) for s in canon["subject_ids"]], canon["predictions"],
            ))
            preds_for_attrs = np.array([
                subj_to_pred.get(str(s), np.nan) for s in subj_attr_ids
            ])
            fig = plot_per_quintile_attribution(
                attrs_all, preds_for_attrs, ct_names,
                save_path=out_dir / "fig_per_quintile_attribution",
            )
            plt.close(fig)
            rendered.append("fig_per_quintile_attribution")
        except (ValueError, KeyError) as exc:
            logger.warning("per-quintile: %s", exc)

    # 5. Cross-fold attribution stability heatmap.
    if (
        captum_data is not None
        and "attributions" in captum_data
        and "fold" in captum_data
        and ct_names is not None
        and gene_names is not None
    ):
        try:
            fig = plot_attribution_stability_heatmap(
                captum_data["attributions"],
                np.asarray(captum_data["fold"], dtype=int),
                ct_names, gene_names,
                save_path=out_dir / "fig_attribution_stability_heatmap",
            )
            plt.close(fig)
            rendered.append("fig_attribution_stability_heatmap")
        except (ValueError, KeyError) as exc:
            logger.warning("stability heatmap: %s", exc)

    # 6. Captum vs DE concordance — cross-validates model attributions
    #    against classical resilient-vs-vulnerable DE rankings.
    de_dir = Path(args.de_dir)
    if (
        captum_data is not None
        and "attributions" in captum_data
        and ct_names is not None
        and gene_names is not None
        and de_dir.exists()
    ):
        de_per_ct: list[pd.DataFrame | None] = []
        n_ct_attr = captum_data["attributions"].shape[1]
        for ct in range(n_ct_attr):
            de_csv = de_dir / f"CT_{ct:02d}_de.csv"
            de_per_ct.append(pd.read_csv(de_csv) if de_csv.exists() else None)
        n_loaded = sum(d is not None for d in de_per_ct)
        if n_loaded == 0:
            logger.warning("captum-vs-DE: no DE CSVs found under %s", de_dir)
        else:
            try:
                fig = plot_captum_de_concordance(
                    captum_data["attributions"],
                    de_per_ct, ct_names, gene_names,
                    save_path=out_dir / "fig_captum_de_concordance",
                )
                plt.close(fig)
                rendered.append("fig_captum_de_concordance")
            except (ValueError, KeyError) as exc:
                logger.warning("captum-vs-DE: %s", exc)

    logger.info("rendered %d attribution figures: %s", len(rendered), rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
