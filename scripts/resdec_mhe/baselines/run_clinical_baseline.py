"""Clinical-only baseline: regress cogn_global on demographics + APOE-e4 + Braak.

Predictors (all from `data/metadata_ROSMAP/metadata.csv`, columns verified to be
non-null on the 516 train_val_pool):
    - apoe_genotype  -> APOE-e4 dosage in {0, 1, 2}, parsed from the two digits
                        (e.g. 34 -> 1, 44 -> 2, 33 -> 0). One subject in the pool
                        has apoe_genotype = NaN; ε4 dosage is imputed to the
                        train-fold mean.
    - age_death      -> standardized (z-score on train, applied to val)
    - msex           -> {0, 1} as-is (no scaling)
    - educ           -> standardized
    - braaksc        -> standardized

Two estimators are evaluated on the canonical 5-fold splits at
`outputs/splits.json`:
    1. Linear regression (sklearn.linear_model.LinearRegression — no penalty).
    2. ElasticNet with inner 5-fold CV via ElasticNetCV over a small
       (alpha x l1_ratio) grid for principled regularization choice.

Per-fold metrics (R^2, MAE, RMSE, Pearson r, Spearman rho) are written along
with paired-Wilcoxon comparisons against:
    - ResDec-MHE canonical seed42 (per-fold from
      `outputs/canonical/p5_canonical_seed42/best_vs_tabpfn_summary.json`).
    - TabPFN-2.6 standalone, gene-expression top-2K features
      (per-fold `tab_ge` block of the same file; matches the reference
      mean R^2 = 0.399 reported in the paper baseline table).

Outputs (in --output-dir, default `outputs/canonical/clinical_baseline/`):
    - `clinical_baseline_summary.json`
    - `clinical_baseline_summary.md`

Usage:
    uv run python scripts/resdec_mhe/baselines/run_clinical_baseline.py
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Repo root: this file is at <repo>/scripts/resdec_mhe/baselines/run_clinical_baseline.py
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_METADATA = REPO_ROOT / "data" / "metadata_ROSMAP" / "metadata.csv"
DEFAULT_SPLITS = REPO_ROOT / "outputs" / "splits.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "canonical" / "clinical_baseline"
DEFAULT_REFERENCE_JSON = (
    REPO_ROOT
    / "outputs"
    / "canonical"
    / "p5_canonical_seed42"
    / "best_vs_tabpfn_summary.json"
)

SUBJECT_COLUMN = "ROSMAP_IndividualID"
TARGET_COLUMN = "cogn_global"

# Demographic / clinical predictors. The order is fixed for downstream
# coefficient interpretation. Continuous columns are z-scored on the train
# fold; binary `msex` is left as-is; `apoe4_dosage` is integer {0,1,2}
# (mean-imputed for the single missing subject).
CONTINUOUS_PREDICTORS = ("age_death", "educ", "braaksc")
PASSTHROUGH_PREDICTORS = ("msex", "apoe4_dosage")
ALL_PREDICTORS = (*CONTINUOUS_PREDICTORS, *PASSTHROUGH_PREDICTORS)


# ── APOE genotype parsing ───────────────────────────────────────────────────


def parse_apoe4_dosage(genotype: float | int | str | None) -> float:
    """Parse APOE genotype encoding into ε4 allele count {0, 1, 2}.

    ROSMAP `apoe_genotype` is stored as a two-digit number where each digit is
    one allele, e.g. 34 -> alleles {3, 4} -> dosage 1; 44 -> dosage 2; 33 ->
    dosage 0; 22 -> 0; 23 -> 0; 24 -> 1.

    Returns NaN for missing input. Raises ValueError on unexpected encodings.
    """
    if genotype is None or (isinstance(genotype, float) and np.isnan(genotype)):
        return float("nan")

    # Coerce to int; allow float64 (e.g. 34.0) or str ("34").
    try:
        gint = int(round(float(genotype)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot parse APOE genotype: {genotype!r}") from exc

    if not 22 <= gint <= 44:
        raise ValueError(f"APOE genotype {gint} outside expected range [22, 44]")

    a1, a2 = divmod(gint, 10)
    if a1 not in (2, 3, 4) or a2 not in (2, 3, 4):
        raise ValueError(f"APOE genotype {gint} contains non-{{2,3,4}} alleles")

    return float(int(a1 == 4) + int(a2 == 4))


# ── Data assembly ───────────────────────────────────────────────────────────


@dataclass
class FoldData:
    """Train / val tensors for a single CV fold."""

    fold_idx: int
    X_train: np.ndarray
    y_train: np.ndarray
    train_ids: list[str]
    X_val: np.ndarray
    y_val: np.ndarray
    val_ids: list[str]
    feature_names: list[str] = field(default_factory=list)


def build_predictor_frame(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble subject -> predictor frame keyed by SUBJECT_COLUMN.

    Adds an `apoe4_dosage` column derived from `apoe_genotype`. Returns a frame
    with columns [SUBJECT_COLUMN, TARGET_COLUMN, *ALL_PREDICTORS]. NaNs are
    preserved at this stage; per-fold imputation (train-fold mean for
    apoe4_dosage) is handled in `prepare_fold`.
    """
    required = [SUBJECT_COLUMN, TARGET_COLUMN, "apoe_genotype", *CONTINUOUS_PREDICTORS, "msex"]
    missing = [c for c in required if c not in metadata_df.columns]
    if missing:
        raise KeyError(f"metadata.csv missing required columns: {missing}")

    frame = metadata_df[required].copy()
    frame["apoe4_dosage"] = frame["apoe_genotype"].apply(parse_apoe4_dosage)
    keep_cols = [SUBJECT_COLUMN, TARGET_COLUMN, *ALL_PREDICTORS]
    return frame[keep_cols]


def prepare_fold(
    predictors_df: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    fold_idx: int,
) -> FoldData:
    """Build one fold's train/val arrays.

    - Drops subjects with missing target.
    - apoe4_dosage missing -> impute using train-fold mean.
    - CONTINUOUS_PREDICTORS z-scored on train fold then applied to val fold.
    - msex kept as 0/1.
    """
    indexed = predictors_df.set_index(SUBJECT_COLUMN)

    train_present = [sid for sid in train_ids if sid in indexed.index]
    val_present = [sid for sid in val_ids if sid in indexed.index]
    missing_train = set(train_ids) - set(train_present)
    missing_val = set(val_ids) - set(val_present)
    if missing_train or missing_val:
        logger.warning(
            "fold %d: %d train + %d val ids missing from metadata",
            fold_idx,
            len(missing_train),
            len(missing_val),
        )

    train_df = indexed.loc[train_present].copy()
    val_df = indexed.loc[val_present].copy()

    # Drop subjects with missing target.
    train_df = train_df[~train_df[TARGET_COLUMN].isna()]
    val_df = val_df[~val_df[TARGET_COLUMN].isna()]

    # Impute apoe4_dosage missing -> train mean.
    train_apoe_mean = float(train_df["apoe4_dosage"].mean())
    train_df["apoe4_dosage"] = train_df["apoe4_dosage"].fillna(train_apoe_mean)
    val_df["apoe4_dosage"] = val_df["apoe4_dosage"].fillna(train_apoe_mean)

    # Standardize continuous predictors on train; transform val.
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train_df[list(CONTINUOUS_PREDICTORS)].to_numpy(dtype=float))
    val_cont = scaler.transform(val_df[list(CONTINUOUS_PREDICTORS)].to_numpy(dtype=float))

    train_pass = train_df[list(PASSTHROUGH_PREDICTORS)].to_numpy(dtype=float)
    val_pass = val_df[list(PASSTHROUGH_PREDICTORS)].to_numpy(dtype=float)

    X_train = np.concatenate([train_cont, train_pass], axis=1)
    X_val = np.concatenate([val_cont, val_pass], axis=1)

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=float)

    feature_names = [f"{c}_z" for c in CONTINUOUS_PREDICTORS] + list(PASSTHROUGH_PREDICTORS)

    return FoldData(
        fold_idx=fold_idx,
        X_train=X_train,
        y_train=y_train,
        train_ids=list(train_df.index),
        X_val=X_val,
        y_val=y_val,
        val_ids=list(val_df.index),
        feature_names=feature_names,
    )


# ── Metrics & estimators ────────────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Standard regression metrics. Returns NaN safely on degenerate inputs."""
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        out["pearson_r"] = float("nan")
        out["spearman_rho"] = float("nan")
    else:
        out["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
        out["spearman_rho"] = float(spearmanr(y_true, y_pred)[0])
    return out


def fit_predict_linreg(fold: FoldData) -> dict:
    """Fit ordinary least-squares linear regression on the fold."""
    model = LinearRegression()
    model.fit(fold.X_train, fold.y_train)
    y_pred = model.predict(fold.X_val)
    metrics = compute_metrics(fold.y_val, y_pred)
    metrics["coef"] = dict(zip(fold.feature_names, model.coef_.tolist(), strict=True))
    metrics["intercept"] = float(model.intercept_)
    return metrics


def fit_predict_elasticnet(fold: FoldData, *, random_state: int = 42) -> dict:
    """Fit ElasticNet with inner 5-fold CV alpha/l1_ratio selection on train."""
    n_train = fold.X_train.shape[0]
    cv_inner = min(5, max(2, n_train // 20))
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=100,  # 100-point automatic alpha path (sklearn default semantics)
        cv=cv_inner,
        max_iter=10_000,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(fold.X_train, fold.y_train)
    y_pred = model.predict(fold.X_val)
    metrics = compute_metrics(fold.y_val, y_pred)
    metrics["coef"] = dict(zip(fold.feature_names, model.coef_.tolist(), strict=True))
    metrics["intercept"] = float(model.intercept_)
    metrics["alpha"] = float(model.alpha_)
    metrics["l1_ratio"] = float(model.l1_ratio_)
    return metrics


# ── Reference fold R² (ResDec-MHE + TabPFN) ────────────────────────────────


def load_reference_per_fold(reference_path: Path) -> dict[str, list[float]]:
    """Read ResDec-MHE and TabPFN-2.6 standalone per-fold R² from canonical JSON.

    Returns {"resdec_mhe": [...], "tabpfn": [...]} sorted by fold index.
    Missing reference file -> empty dict (paired Wilcoxon will be skipped).
    """
    if not reference_path.exists():
        logger.warning("Reference summary not found: %s", reference_path)
        return {}
    payload = json.loads(reference_path.read_text())
    per_fold = sorted(payload["per_fold"], key=lambda d: d["fold"])
    resdec = [float(rec["ours"]["r2"]) for rec in per_fold]
    # `tab_ge` (gene expression top-2K features) is the standalone TabPFN
    # baseline reported as 0.399 in the paper baseline table.
    tabpfn = [float(rec["tab_ge"]["r2"]) for rec in per_fold]
    return {"resdec_mhe": resdec, "tabpfn": tabpfn}


def paired_wilcoxon(ours: list[float], theirs: list[float]) -> dict:
    """One-sided paired Wilcoxon: H1: theirs > ours (snRNA model > clinical).

    Returns statistic + p-value + summary; falls back to NaN with the
    statistic-not-defined cause if both vectors are identical.
    """
    if len(ours) != len(theirs):
        raise ValueError(f"length mismatch: {len(ours)} vs {len(theirs)}")
    if not ours:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": 0, "note": "empty input"}
    diffs = np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float)
    if np.allclose(diffs, 0.0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n": len(diffs),
            "note": "all paired diffs are zero",
        }
    res = wilcoxon(ours, theirs, alternative="less", zero_method="wilcox")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "alternative": "less (clinical < reference)",
        "n": len(diffs),
        "mean_diff": float(diffs.mean()),
    }


# ── Orchestration ───────────────────────────────────────────────────────────


def run_clinical_baseline(
    metadata_path: Path,
    splits_path: Path,
    output_dir: Path,
    reference_path: Path,
    *,
    random_state: int = 42,
) -> dict:
    """Top-level driver. Returns the summary payload."""
    logger.info("Loading metadata: %s", metadata_path)
    metadata_df = pd.read_csv(metadata_path)
    predictors_df = build_predictor_frame(metadata_df)

    logger.info("Loading splits: %s", splits_path)
    splits = json.loads(splits_path.read_text())
    folds = splits["folds"]
    pool = splits["train_val_pool"]
    logger.info("Pool size: %d, folds: %d", len(pool), len(folds))

    pool_predictors = predictors_df[predictors_df[SUBJECT_COLUMN].isin(pool)]
    n_in_pool = len(pool_predictors)
    n_target_missing = int(pool_predictors[TARGET_COLUMN].isna().sum())
    n_apoe_missing = int(pool_predictors["apoe4_dosage"].isna().sum())
    logger.info(
        "Predictor coverage on 516 pool: %d rows; target NaN=%d; apoe4 NaN=%d",
        n_in_pool,
        n_target_missing,
        n_apoe_missing,
    )

    per_fold_records: list[dict] = []
    for fold_idx, fold_def in enumerate(folds):
        train_ids = fold_def["train"]
        val_ids = fold_def["val"]
        fold_data = prepare_fold(predictors_df, train_ids, val_ids, fold_idx)
        logger.info(
            "fold %d: train n=%d, val n=%d, p=%d",
            fold_idx,
            fold_data.X_train.shape[0],
            fold_data.X_val.shape[0],
            fold_data.X_train.shape[1],
        )
        linreg_metrics = fit_predict_linreg(fold_data)
        enet_metrics = fit_predict_elasticnet(fold_data, random_state=random_state)
        per_fold_records.append(
            {
                "fold": fold_idx,
                "n_train": int(fold_data.X_train.shape[0]),
                "n_val": int(fold_data.X_val.shape[0]),
                "linreg": linreg_metrics,
                "elasticnet": enet_metrics,
            }
        )

    per_fold_summary = _summarize_per_fold(per_fold_records)

    reference = load_reference_per_fold(reference_path)
    paired_stats = {}
    if reference:
        clinical_linreg_r2 = [rec["linreg"]["r2"] for rec in per_fold_records]
        clinical_enet_r2 = [rec["elasticnet"]["r2"] for rec in per_fold_records]
        for ref_name, ref_vals in reference.items():
            if len(ref_vals) != len(clinical_linreg_r2):
                logger.warning(
                    "Reference %s has %d folds, clinical has %d — skipping paired test",
                    ref_name,
                    len(ref_vals),
                    len(clinical_linreg_r2),
                )
                continue
            paired_stats[f"linreg_vs_{ref_name}"] = paired_wilcoxon(
                clinical_linreg_r2, ref_vals
            )
            paired_stats[f"elasticnet_vs_{ref_name}"] = paired_wilcoxon(
                clinical_enet_r2, ref_vals
            )
        # Also tabulate the deltas for the report.
        for ref_name, ref_vals in reference.items():
            paired_stats[f"delta_r2_{ref_name}_minus_linreg"] = float(
                np.mean(np.asarray(ref_vals) - np.asarray(clinical_linreg_r2))
            )
            paired_stats[f"delta_r2_{ref_name}_minus_elasticnet"] = float(
                np.mean(np.asarray(ref_vals) - np.asarray(clinical_enet_r2))
            )

    summary = {
        "predictors": list(ALL_PREDICTORS),
        "n_pool": n_in_pool,
        "n_target_missing": n_target_missing,
        "n_apoe_missing_pool": n_apoe_missing,
        "per_fold": per_fold_records,
        "summary": per_fold_summary,
        "reference_per_fold_r2": reference,
        "paired_wilcoxon": paired_stats,
        "data_paths": {
            "metadata": str(metadata_path),
            "splits": str(splits_path),
            "reference": str(reference_path),
        },
        "config": {
            "random_state": random_state,
            "estimators": ["linreg", "elasticnet"],
            "elasticnet_grid": {
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "alphas": "100-point automatic ElasticNetCV path",
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "clinical_baseline_summary.json"
    md_path = output_dir / "clinical_baseline_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=False, default=float))
    md_path.write_text(_render_markdown(summary))
    logger.info("Wrote: %s", json_path)
    logger.info("Wrote: %s", md_path)
    return summary


def _summarize_per_fold(per_fold: list[dict]) -> dict:
    """Aggregate mean / std of each metric across folds for both estimators."""
    metric_keys = ("r2", "mae", "rmse", "pearson_r", "spearman_rho")
    out: dict = {}
    for est in ("linreg", "elasticnet"):
        est_block = {}
        for k in metric_keys:
            vals = np.asarray([rec[est][k] for rec in per_fold], dtype=float)
            est_block[f"{k}_mean"] = float(vals.mean())
            est_block[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            est_block[f"{k}_per_fold"] = vals.tolist()
        out[est] = est_block
    return out


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _render_markdown(summary: dict) -> str:
    """Compose the human-readable markdown report."""
    lines: list[str] = []
    lines.append("# Clinical-only Baseline (cogn_global)")
    lines.append("")
    lines.append(
        f"Predictors: `{', '.join(summary['predictors'])}` "
        f"(continuous z-scored on train fold; msex / apoe4_dosage passthrough; "
        f"apoe4 NaN imputed to train mean)."
    )
    lines.append(
        f"Pool: n={summary['n_pool']}, target NaN={summary['n_target_missing']}, "
        f"apoe4 NaN in pool={summary['n_apoe_missing_pool']}."
    )
    lines.append("")
    lines.append("## Cross-fold metrics (mean ± std across 5 folds)")
    lines.append("")
    s = summary["summary"]
    lines.append("| Estimator | R² | MAE | RMSE | Pearson r | Spearman ρ |")
    lines.append("|---|---|---|---|---|---|")
    for label, key in (("Clinical (LinReg)", "linreg"), ("Clinical (ElasticNet)", "elasticnet")):
        block = s[key]
        lines.append(
            f"| {label} | {_fmt(block['r2_mean'])} ± {_fmt(block['r2_std'])} | "
            f"{_fmt(block['mae_mean'])} ± {_fmt(block['mae_std'])} | "
            f"{_fmt(block['rmse_mean'])} ± {_fmt(block['rmse_std'])} | "
            f"{_fmt(block['pearson_r_mean'])} ± {_fmt(block['pearson_r_std'])} | "
            f"{_fmt(block['spearman_rho_mean'])} ± {_fmt(block['spearman_rho_std'])} |"
        )
    lines.append("")
    lines.append("## Per-fold R²")
    lines.append("")
    ref = summary.get("reference_per_fold_r2", {}) or {}
    header = ["fold"]
    rows: list[list[str]] = []
    n_folds = len(summary["per_fold"])
    header.extend(["LinReg", "ElasticNet"])
    if "tabpfn" in ref:
        header.append("TabPFN-2.6 (ref)")
    if "resdec_mhe" in ref:
        header.append("ResDec-MHE (ref)")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i in range(n_folds):
        rec = summary["per_fold"][i]
        row = [str(i), _fmt(rec["linreg"]["r2"]), _fmt(rec["elasticnet"]["r2"])]
        if "tabpfn" in ref:
            row.append(_fmt(ref["tabpfn"][i]))
        if "resdec_mhe" in ref:
            row.append(_fmt(ref["resdec_mhe"][i]))
        rows.append(row)
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")

    if summary.get("paired_wilcoxon"):
        lines.append("## Paired Wilcoxon (clinical vs reference, alternative='less')")
        lines.append("")
        lines.append("Tests H1: clinical R² < reference R² across the 5 paired folds.")
        lines.append("")
        lines.append("| Comparison | n | mean Δ | W | p-value |")
        lines.append("|---|---|---|---|---|")
        for key, val in summary["paired_wilcoxon"].items():
            if not isinstance(val, dict):
                continue
            lines.append(
                f"| {key} | {val.get('n', 'n/a')} | {_fmt(val.get('mean_diff', float('nan')))} | "
                f"{_fmt(val.get('statistic', float('nan')))} | {_fmt(val.get('p_value', float('nan')), 5)} |"
            )
        lines.append("")
        deltas = {k: v for k, v in summary["paired_wilcoxon"].items() if not isinstance(v, dict)}
        if deltas:
            lines.append("### Mean ΔR² (reference − clinical)")
            lines.append("")
            for k, v in deltas.items():
                lines.append(f"- `{k}`: {_fmt(v)}")
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lin_r2 = s["linreg"]["r2_mean"]
    enet_r2 = s["elasticnet"]["r2_mean"]
    if "resdec_mhe" in ref:
        resdec_mean = float(np.mean(ref["resdec_mhe"]))
        gap_lin = resdec_mean - lin_r2
        gap_enet = resdec_mean - enet_r2
        lines.append(
            f"ResDec-MHE mean R² = {_fmt(resdec_mean)} vs clinical LinReg "
            f"{_fmt(lin_r2)} (Δ = {_fmt(gap_lin)}) and clinical ElasticNet "
            f"{_fmt(enet_r2)} (Δ = {_fmt(gap_enet)}). "
            f"snRNA-seq adds R² ≈ {_fmt(max(gap_lin, gap_enet))} over the best clinical baseline."
        )
    else:
        lines.append("Reference per-fold R² unavailable; gap not computed.")
    return "\n".join(lines) + "\n"


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Path to metadata.csv (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=DEFAULT_SPLITS,
        help=f"Path to splits.json (default: {DEFAULT_SPLITS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for summary.{{json,md}} (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=DEFAULT_REFERENCE_JSON,
        help=(
            "ResDec-MHE / TabPFN per-fold summary JSON "
            f"(default: {DEFAULT_REFERENCE_JSON})"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state passed to ElasticNetCV (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    return run_clinical_baseline(
        metadata_path=args.metadata_path,
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        reference_path=args.reference_path,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
