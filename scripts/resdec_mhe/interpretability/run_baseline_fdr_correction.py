#!/usr/bin/env python
"""Baseline-panel multiple-comparison correction (BH-FDR + Bonferroni).

Companion to the per-baseline Stouffer-combined p-values reported in
``EXP-002`` / ``EXP-003`` and stored in
``outputs/canonical/interpretability/seed_variation_wilcoxon_all_baselines.json``.

EXP-002 reports a per-baseline Stouffer-combined p across 5 seeds for
each of the 22 baselines, but does NOT correct for multiple comparisons
across the panel — Reviewer 2 will require this. This script applies:

1. **Benjamini–Hochberg FDR** at α=0.05 on the 22 Stouffer p-values
   (``scipy.stats.false_discovery_control(method='bh')`` — equivalent to
   ``statsmodels.stats.multitest.multipletests(method='fdr_bh')`` to
   machine precision; we cross-check both inside the test suite).
2. **Bonferroni** at α=0.05/M=0.05/22 ≈ 0.00227, flagging each baseline
   with a boolean ``bonferroni_significant``.
3. **Lost-to-FDR diagnostic** — baselines whose unadjusted Stouffer p
   < 0.05 but BH-q ≥ 0.05 (i.e. dropped by FDR correction).

Reads
-----
``outputs/canonical/interpretability/seed_variation_wilcoxon_all_baselines.json``

Writes
------
- ``<out-dir>/baseline_fdr_correction.json`` — full machine-readable record
  (per-baseline Stouffer p, BH q, Bonferroni boolean, plus panel-level
  summary counts).
- ``<out-dir>/baseline_fdr_correction.md`` — paper-ready markdown table
  with the four columns | Baseline | Stouffer p | BH-FDR q |
  Bonferroni-sig | Notes |.

Usage
-----
    PYTHONPATH=<worktree-root> uv run python \\
        scripts/resdec_mhe/interpretability/run_baseline_fdr_correction.py \\
        --in-path outputs/canonical/interpretability/seed_variation_wilcoxon_all_baselines.json \\
        --out-dir outputs/canonical/interpretability
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import false_discovery_control

logger = logging.getLogger(__name__)

# Default α used for both BH-FDR thresholding and the Bonferroni divisor.
DEFAULT_ALPHA = 0.05

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Apply Benjamini–Hochberg FDR to ``p_values`` and return q-values.

    Uses ``scipy.stats.false_discovery_control(method='bh')`` which is the
    standard linear-step-up procedure: q_i = min_{k≥i} (M · p_(k) / k),
    where p_(k) is the k-th order statistic of M sorted p-values.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape {p.shape}")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p_values must be in [0, 1].")
    return false_discovery_control(p, method="bh")


def bonferroni_threshold(alpha: float, m: int) -> float:
    """Return the per-test Bonferroni threshold α/M."""
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return alpha / float(m)


def _load_per_baseline_stouffer(in_path: Path) -> dict[str, dict]:
    """Read the per-baseline Stouffer + per-seed Wilcoxon record."""
    if not in_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")
    with in_path.open("r") as fh:
        payload = json.load(fh)
    if "per_baseline" not in payload:
        raise KeyError(f"Expected key 'per_baseline' in {in_path}")
    return payload


def correct_panel(
    payload: dict,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Apply BH-FDR + Bonferroni correction to the 22-baseline panel.

    Parameters
    ----------
    payload : dict
        Loaded JSON payload from ``seed_variation_wilcoxon_all_baselines.json``.
        Must have a ``per_baseline`` mapping; each value must contain
        ``stouffer_p_one_sided`` and ``per_seed`` with the 5 per-seed Wilcoxon
        p-values under key ``wilcoxon_p_one_sided_greater``.
    alpha : float
        Family-wise α for both BH-FDR thresholding and Bonferroni divisor
        (default 0.05).

    Returns
    -------
    dict with keys:
        - ``method``        : str
        - ``alpha``         : float
        - ``m_baselines``   : int
        - ``bonferroni_threshold`` : float
        - ``per_baseline``  : ordered list of dicts (one per baseline)
        - ``summary``       : panel-level counts
    """
    per = payload["per_baseline"]
    baselines = list(per.keys())
    m = len(baselines)
    seeds = payload.get("seeds", [])

    stouffer_p = np.asarray(
        [float(per[b]["stouffer_p_one_sided"]) for b in baselines],
        dtype=np.float64,
    )

    bh_q = bh_fdr(stouffer_p)
    bonf_thr = bonferroni_threshold(alpha, m)
    bonf_sig = stouffer_p < bonf_thr  # strict <, matches Bonferroni convention

    # Lost-to-FDR: unadjusted p < α but corrected q ≥ α.
    lost_to_fdr = (stouffer_p < alpha) & (bh_q >= alpha)

    per_baseline: list[dict] = []
    for i, name in enumerate(baselines):
        per_seed_ps = [
            float(per[name]["per_seed"][str(s)]["wilcoxon_p_one_sided_greater"])
            for s in seeds
        ]
        notes_parts: list[str] = []
        if not bonf_sig[i]:
            notes_parts.append("Bonferroni-fail")
        if lost_to_fdr[i]:
            notes_parts.append("Lost-to-FDR")
        per_baseline.append(
            {
                "baseline": name,
                "per_seed_wilcoxon_p_one_sided_greater": per_seed_ps,
                "stouffer_p_one_sided": float(stouffer_p[i]),
                "bh_q_value": float(bh_q[i]),
                "bonferroni_significant": bool(bonf_sig[i]),
                "lost_to_fdr": bool(lost_to_fdr[i]),
                "notes": "; ".join(notes_parts) if notes_parts else "",
            }
        )

    n_bh_pass = int(np.sum(bh_q < alpha))
    n_bonf_pass = int(np.sum(bonf_sig))
    n_lost = int(np.sum(lost_to_fdr))
    # "Most-conservative baseline among the BH-significant set" = the one
    # with the largest q-value while still q < α (closest to the FDR
    # rejection boundary).
    bh_pass_mask = bh_q < alpha
    if bh_pass_mask.any():
        idx = int(np.argmax(np.where(bh_pass_mask, bh_q, -np.inf)))
        most_conservative = {
            "baseline": baselines[idx],
            "stouffer_p_one_sided": float(stouffer_p[idx]),
            "bh_q_value": float(bh_q[idx]),
        }
    else:
        most_conservative = None

    return {
        "method": (
            "Benjamini-Hochberg FDR (scipy.stats.false_discovery_control, "
            "method='bh') + Bonferroni at alpha/M on Stouffer-combined "
            "per-baseline p-values."
        ),
        "alpha": float(alpha),
        "m_baselines": m,
        "bonferroni_threshold": float(bonf_thr),
        "seeds": seeds,
        "input_path": "(set by caller)",
        "per_baseline": per_baseline,
        "summary": {
            "n_bh_significant": n_bh_pass,
            "n_bonferroni_significant": n_bonf_pass,
            "n_lost_to_fdr": n_lost,
            "most_conservative_significant_baseline": most_conservative,
        },
    }


def _format_p(p: float) -> str:
    """Format a p- or q-value for the markdown column.

    Use scientific notation when below 1e-3 (matches the per-baseline
    table convention; reads cleanly for very-small values like 1.6e-05).
    """
    if not np.isfinite(p):
        return "—"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


def write_markdown(
    record: dict,
    md_path: Path,
    *,
    in_path_str: str | None = None,
) -> None:
    """Write the markdown table to ``md_path``."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    summary = record["summary"]
    m = record["m_baselines"]
    alpha = record["alpha"]
    bonf_thr = record["bonferroni_threshold"]

    lines: list[str] = [
        "# Baseline-panel multiple-comparison correction",
        "",
        (
            f"Source: {in_path_str or '(per-call argument)'} "
            f"(per-baseline Stouffer-combined p over {len(record['seeds'])} "
            "seeds × 5 folds)."
        ),
        "",
        (
            f"Methods: Benjamini–Hochberg FDR at α={alpha} "
            f"(`scipy.stats.false_discovery_control(method='bh')`), "
            f"Bonferroni at α/M = {alpha}/{m} = {bonf_thr:.4e}."
        ),
        "",
        f"M = {m} baselines.",
        "",
        "| Baseline | Stouffer p | BH-FDR q | Bonferroni-sig | Notes |",
        "|---|---|---|---|---|",
    ]
    for row in record["per_baseline"]:
        bonf = "yes" if row["bonferroni_significant"] else "no"
        notes = row["notes"] if row["notes"] else "—"
        lines.append(
            f"| {row['baseline']} "
            f"| {_format_p(row['stouffer_p_one_sided'])} "
            f"| {_format_p(row['bh_q_value'])} "
            f"| {bonf} "
            f"| {notes} |"
        )
    lines.extend(
        [
            "",
            "## Panel-level summary",
            "",
            f"- BH-FDR significant (q < {alpha}): "
            f"**{summary['n_bh_significant']} / {m}**.",
            f"- Bonferroni significant (p < {bonf_thr:.4e}): "
            f"**{summary['n_bonferroni_significant']} / {m}**.",
            f"- Lost to FDR correction (Stouffer p < {alpha} but BH-q ≥ "
            f"{alpha}): **{summary['n_lost_to_fdr']}**.",
        ]
    )
    most = summary["most_conservative_significant_baseline"]
    if most is not None:
        lines.append(
            f"- Most-conservative BH-significant baseline (largest q): "
            f"**{most['baseline']}** "
            f"(q = {_format_p(most['bh_q_value'])}, "
            f"Stouffer p = {_format_p(most['stouffer_p_one_sided'])})."
        )
    else:
        lines.append("- Most-conservative BH-significant baseline: N/A — no baselines passed BH.")

    md_path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--in-path",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/seed_variation_wilcoxon_all_baselines.json",
        help="Per-baseline Stouffer + per-seed Wilcoxon JSON.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/interpretability",
        help="Directory to write baseline_fdr_correction.{json,md}.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Family-wise alpha for BH and Bonferroni divisor (default {DEFAULT_ALPHA}).",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "baseline_fdr_correction.json"
    out_md = out_dir / "baseline_fdr_correction.md"

    payload = _load_per_baseline_stouffer(in_path)
    record = correct_panel(payload, alpha=float(args.alpha))
    record["input_path"] = str(in_path)
    record["written_at_utc"] = datetime.now(tz=timezone.utc).isoformat()

    out_json.write_text(json.dumps(record, indent=2))
    write_markdown(record, out_md, in_path_str=str(in_path))

    summary = record["summary"]
    most = summary["most_conservative_significant_baseline"]
    most_str = (
        f"{most['baseline']} (q={most['bh_q_value']:.4e})"
        if most is not None
        else "N/A"
    )
    logger.info(
        "M=%d baselines | BH-pass=%d | Bonferroni-pass=%d | lost-to-FDR=%d | "
        "most-conservative-BH=%s",
        record["m_baselines"],
        summary["n_bh_significant"],
        summary["n_bonferroni_significant"],
        summary["n_lost_to_fdr"],
        most_str,
    )
    logger.info("Wrote %s + %s", out_json, out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
