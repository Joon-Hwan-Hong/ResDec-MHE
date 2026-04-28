"""Cross-reference SAE interpretable features against the 11-method §31.9 (CT, gene) consensus.

For each interpretable SAE feature in the trained best-config run (seed=0,
``batch_topk/fused/exp32_k64``), extract the top-1 cell type from
``top_cell_types`` (decoder-direction projection magnitude) and compute:

  * Per-CT count of dominant interpretable features.
  * Splatter concentration fraction.
  * Concentration in the §31.9 top-3 consensus (Splatter, Fibroblast, ≥6/11).
  * The same metrics for the matched-as-close-as-possible random-encoder
    baseline (``random_encoder/topk/attended/exp8_k16_seed0``).

DEVIATION FROM USER BRIEF (flagged):

The brief instructs "filter to ``interpretable_candidate`` flagged features"
and "extract its top-1 cell type from ``top_cell_types``". Empirically the
fused-layer best-config feature_report has **zero** ``interpretable_candidate``
features because ``interpret_features`` requires ``ct_dominance > 0.7`` while
the highest observed CT dominance in the run is 0.638 (no feature concentrates
70 % of decoder mass into 3 CTs).

To preserve the spirit of the comparison without silently simplifying, we
report **two filtering tiers**:

  1. ``strict`` — ``interpretable_candidate`` flag exactly as the
     ``interpret_features`` function emits it. This may yield 0.
  2. ``relaxed`` — non-dead AND ``mw_p_cognition`` < 0.05 AND
     ``fraction_active`` in [1e-4, 0.5]. This is the ``interpretable_candidate``
     definition with the ``ct_dominance > 0.7`` clause removed, used to
     surface CT distribution at the canonical sparsity / cognition criteria
     when no feature meets the strict CT-dominance bar.

The canonical random-encoder comparator is now also a fused-layer SAE
matched to the trained-encoder best-config (``batch_topk/fused/exp32_k64_seed0``),
so CT-distribution metrics are valid on both sides. (The earlier
``topk/attended/exp8_k16_seed0`` random-null produced no
``top_cell_types`` and was an apples-to-oranges comparison.)

Outputs
-------
``<out-path>/feature_xref_consensus.json``
    Per-CT counts (trained vs random) at both tiers, Splatter-concentration
    fractions, concentration in §31.9 top-3 consensus, and a brief
    markdown summary appended at ``markdown_summary``.

Usage
-----
    PYTHONPATH=<worktree-root> uv run python \\
        scripts/resdec_mhe/interpretability/run_feature_xref_consensus.py \\
        --trained-report outputs/redesign/sae/batch_topk/fused/exp32_k64_seed0/feature_report.json \\
        --random-report  outputs/redesign/sae/random_encoder/batch_topk/fused/exp32_k64_seed0/feature_report.json \\
        --out-path       outputs/redesign/sae/feature_xref_consensus.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.utils.provenance import git_sha  # noqa: E402

logger = logging.getLogger(__name__)


# §31.9 11-method (CT, gene) consensus from MASTER-INFO. These CTs reflect
# top-5 placement across N methods (Captum, GradShap, GS-SmoothGrad, attention
# attribution, raw-pseudobulk CMI max+vector, CCC, counterfactuals
# {relative,absolute}, Wasserstein-1, LOCO).
CONSENSUS_TOP3: tuple[str, ...] = ("Splatter", "Fibroblast", "Upper rhombic lip")
CONSENSUS_HIGH_FREQ: dict[str, int] = {
    "Splatter": 11,
    "Fibroblast": 10,
    "Upper rhombic lip": 7,
    "Miscellaneous": 7,
    "Committed oligodendrocyte precursor": 6,
    "LAMP5-LHX6 and Chandelier": 5,  # attention-only
}

# Lower bound for non-dead activation; matches DEAD_FRACTION_THRESHOLD.
DEAD_FRACTION_THRESHOLD: float = 1e-4


def _load_report(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Feature report missing: {path}")
    return json.loads(path.read_text())


def _strict_filter(reports: list[dict]) -> list[dict]:
    return [r for r in reports if "interpretable_candidate" in r.get("flags", [])]


def _relaxed_filter(reports: list[dict]) -> list[dict]:
    out = []
    for r in reports:
        if "dead" in r.get("flags", []):
            continue
        p = r.get("mw_p_cognition")
        if p is None or p >= 0.05:
            continue
        f = r.get("fraction_active", 0.0)
        if not (DEAD_FRACTION_THRESHOLD <= f <= 0.5):
            continue
        out.append(r)
    return out


def _per_ct_counts(features: list[dict]) -> dict[str, int]:
    """Count top-1 CT across features. Skips features whose top_cell_types is empty."""
    counts: dict[str, int] = {ct: 0 for ct in CELL_TYPE_ORDER}
    n_no_ct = 0
    for r in features:
        tcts = r.get("top_cell_types") or []
        if not tcts:
            n_no_ct += 1
            continue
        # top_cell_types is sorted by squared_projection desc in interpret_features.
        ct = tcts[0].get("cell_type")
        if ct is not None and ct in counts:
            counts[ct] += 1
    return counts, n_no_ct


def _splatter_concentration(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return float(counts.get("Splatter", 0) / total) if total > 0 else 0.0


def _consensus_concentration(counts: dict[str, int]) -> float:
    """Fraction of dominant-CT mass in §31.9 top-3 consensus."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    in_consensus = sum(counts.get(ct, 0) for ct in CONSENSUS_TOP3)
    return float(in_consensus / total)


def _top5_cts_by_count(counts: dict[str, int]) -> list[tuple[str, int]]:
    """Return the top-5 CTs by descending count (ties broken by CT order)."""
    items = [(ct, c) for ct, c in counts.items() if c > 0]
    items.sort(key=lambda x: (-x[1], CELL_TYPE_ORDER.index(x[0])))
    return items[:5]


def _build_markdown_summary(payload: dict) -> str:
    lines = [
        "# SAE feature × §31.9 (CT, gene) consensus cross-reference",
        "",
        f"git_commit: `{payload.get('git_commit', '?')}`",
        "",
        "## Trained SAE (best config: batch_topk/fused/exp32_k64_seed0)",
        "",
        f"  * Total features: {payload['trained']['n_total']}",
        f"  * Strict `interpretable_candidate` count: {payload['trained']['strict']['n_features']}",
        f"  * Relaxed (cog<0.05, non-dead, frac in [1e-4, 0.5]): {payload['trained']['relaxed']['n_features']}",
        "",
        "### Splatter concentration",
        f"  * Strict: {_fmt_concentration(payload['trained']['strict']['splatter_concentration'])}",
        f"  * Relaxed: {_fmt_concentration(payload['trained']['relaxed']['splatter_concentration'])}",
        "",
        "### §31.9 top-3 consensus concentration",
        f"  * Strict: {_fmt_concentration(payload['trained']['strict']['consensus_top3_concentration'])}",
        f"  * Relaxed: {_fmt_concentration(payload['trained']['relaxed']['consensus_top3_concentration'])}",
        "",
        "### Top-5 dominant CTs (relaxed)",
    ]
    for ct, c in payload["trained"]["relaxed"].get("top5_cts") or []:
        lines.append(f"  * {ct}: {c}")
    lines += [
        "",
        "## Random-encoder baseline (matched: batch_topk/fused/exp32_k64_seed0)",
        "",
        f"  * Total features: {payload['random']['n_total']}",
        f"  * Strict `interpretable_candidate` count: {payload['random']['strict']['n_features']}",
        f"  * Relaxed: {payload['random']['relaxed']['n_features']}",
        "",
        "### Splatter concentration (random)",
        f"  * Strict: {_fmt_concentration(payload['random']['strict']['splatter_concentration'])}",
        f"  * Relaxed: {_fmt_concentration(payload['random']['relaxed']['splatter_concentration'])}",
        "",
        "### §31.9 top-3 consensus concentration (random)",
        f"  * Strict: {_fmt_concentration(payload['random']['strict']['consensus_top3_concentration'])}",
        f"  * Relaxed: {_fmt_concentration(payload['random']['relaxed']['consensus_top3_concentration'])}",
        "",
        "Apples-to-apples (both fused) — interpretation is meaningful for "
        "both flag-count ratios AND CT-distribution metrics.",
    ]
    return "\n".join(lines)


def _fmt_concentration(v: float | None) -> str:
    """Render a fraction as ``0.123`` or ``N/A`` when missing."""
    if v is None:
        return "N/A"
    return f"{v:.3f}"


def _build_summary(reports: list[dict], label: str, has_ct: bool) -> dict:
    strict = _strict_filter(reports)
    relaxed = _relaxed_filter(reports)
    out: dict = {
        "label": label,
        "n_total": len(reports),
        "strict": {"n_features": len(strict)},
        "relaxed": {"n_features": len(relaxed)},
    }
    if has_ct:
        for tier_name, tier_features in [("strict", strict), ("relaxed", relaxed)]:
            counts, n_no_ct = _per_ct_counts(tier_features)
            out[tier_name].update({
                "per_ct_counts": counts,
                "n_features_without_top_cts": n_no_ct,
                "splatter_concentration": _splatter_concentration(counts),
                "consensus_top3_concentration": _consensus_concentration(counts),
                "top5_cts": _top5_cts_by_count(counts),
            })
    else:
        for tier_name in ("strict", "relaxed"):
            out[tier_name].update({
                "per_ct_counts": None,
                "n_features_without_top_cts": None,
                "splatter_concentration": None,
                "consensus_top3_concentration": None,
                "top5_cts": None,
                "note": "attended-layer SAE has no top_cell_types (no decoder-direction CT decomposition)",
            })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--trained-report",
        required=True,
        help="Path to the trained-SAE feature_report.json (best config seed=0).",
    )
    p.add_argument(
        "--random-report",
        required=True,
        help="Path to the random-encoder feature_report.json comparator.",
    )
    p.add_argument(
        "--out-path",
        required=True,
        help="Output JSON file path.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    )

    trained_path = Path(args.trained_report)
    if not trained_path.is_absolute():
        trained_path = _WORKTREE_ROOT / trained_path
    random_path = Path(args.random_report)
    if not random_path.is_absolute():
        random_path = _WORKTREE_ROOT / random_path
    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = _WORKTREE_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading trained report: %s", trained_path)
    trained_reports = _load_report(trained_path)
    logger.info("Loading random-encoder report: %s", random_path)
    random_reports = _load_report(random_path)

    # Detect whether each run carries top_cell_types (only fused-layer does).
    trained_has_ct = any(r.get("top_cell_types") for r in trained_reports)
    random_has_ct = any(r.get("top_cell_types") for r in random_reports)

    trained_summary = _build_summary(
        trained_reports, "trained_batch_topk_fused_exp32_k64_seed0",
        has_ct=trained_has_ct,
    )
    random_summary = _build_summary(
        random_reports, "random_encoder_topk_attended_exp8_k16_seed0",
        has_ct=random_has_ct,
    )

    payload = {
        "git_commit": git_sha(_WORKTREE_ROOT),
        "consensus_reference": {
            "source": "MASTER-INFO §31.9 11-method (CT, gene) consensus",
            "consensus_top3": list(CONSENSUS_TOP3),
            "high_frequency_cts": CONSENSUS_HIGH_FREQ,
        },
        "deviation_note": (
            "Strict `interpretable_candidate` flag at fused-layer best-config returns "
            "0 features because the function-level threshold `ct_dominance > 0.7` is "
            "unreachable in this run (max observed dominance ~0.64). Relaxed tier "
            "drops the ct_dominance clause but keeps cognition Mann-Whitney p < 0.05 "
            "and the [1e-4, 0.5] sparsity band. Random-encoder comparator is on "
            "`attended` layer (no per-CT decomposition) so only flag-count ratios are "
            "meaningful for that side."
        ),
        "filter_definitions": {
            "strict": "`interpretable_candidate` flag from `interpret_features`.",
            "relaxed": (
                "non-dead AND mw_p_cognition < 0.05 AND fraction_active in "
                f"[{DEAD_FRACTION_THRESHOLD}, 0.5]; ct_dominance > 0.7 dropped."
            ),
        },
        "trained": trained_summary,
        "random": random_summary,
    }
    payload["markdown_summary"] = _build_markdown_summary(payload)

    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %s (trained strict=%d relaxed=%d; random strict=%d relaxed=%d)",
        out_path,
        trained_summary["strict"]["n_features"],
        trained_summary["relaxed"]["n_features"],
        random_summary["strict"]["n_features"],
        random_summary["relaxed"]["n_features"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
