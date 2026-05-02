#!/usr/bin/env python
"""Aggregate permutation-null shards into a single summary (N=50, N=100, ...).

Default reads:
    outputs/canonical/permutation_test_n50_full/shard_a/permutation_results.json (perms 0-24)
    outputs/canonical/permutation_test_n50_full/shard_b/permutation_results.json (perms 25-49)

Default writes:
    outputs/canonical/permutation_test_n50_full/permutation_results.json (combined)
    outputs/canonical/permutation_test_n50_full/permutation_summary.json (aggregate stats)

Override via CLI flags for N=100 / 4-shard variants without source edits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

WT = Path(__file__).resolve().parents[3]
ROOT = WT / "outputs/canonical/permutation_test_n50_full"
CANONICAL_PERMNULL_PATH = WT / "outputs/canonical/permutation_test/permutation_summary.json"


def _load_canonical_r2(canonical_summary_path: Path = CANONICAL_PERMNULL_PATH) -> float:
    """Load canonical 5-fold mean R² from an existing perm summary.

    Single source of truth across N=10 / N=50 / N=100 perm tests. Avoids
    a hardcoded literal that goes stale if the canonical pipeline is rerun.
    """
    if not canonical_summary_path.exists():
        raise FileNotFoundError(
            f"Canonical perm summary missing at {canonical_summary_path}; "
            "cannot derive canonical R² for the aggregated z-score."
        )
    return float(json.loads(canonical_summary_path.read_text())["canonical_mean_r2"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--output-base",
        type=Path,
        default=ROOT,
        help=(
            "Parent directory containing each shard's subdirectory. "
            "Combined results.json + summary.json are written here. "
            f"Default: {ROOT}."
        ),
    )
    p.add_argument(
        "--shards",
        nargs="+",
        default=["shard_a", "shard_b"],
        help=(
            "Per-shard subdirectory names under --output-base. Each must "
            "contain a permutation_results.json. Default: shard_a shard_b."
        ),
    )
    p.add_argument(
        "--canonical-summary",
        type=Path,
        default=CANONICAL_PERMNULL_PATH,
        help=(
            "Path to canonical perm summary holding canonical_mean_r2. "
            f"Default: {CANONICAL_PERMNULL_PATH}."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    output_base: Path = args.output_base
    shard_paths = [
        output_base / shard / "permutation_results.json"
        for shard in args.shards
    ]
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        present = "\n  ".join(
            f"{p}: {p.exists()}" for p in shard_paths
        )
        print(f"Shard files not yet present:\n  {present}")
        return
    combined: list = []
    for path in shard_paths:
        combined.extend(json.loads(path.read_text()))

    # Filter to successful perms (skip those with 'error' field). The
    # ``"error"`` key is an explicit convention from
    # ``run_permutation_test.py``: failure records are written as
    # ``{"perm_seed": k, "error": str(exc), "elapsed_min": ...}``. Successful
    # records carry ``mean_r2_true``. A typed dataclass would be more robust
    # but the convention has held since 2026-04-24 and is exercised by all
    # downstream consumers.
    successful = [r for r in combined if "error" not in r and r.get("mean_r2_true") is not None]
    failed = [r for r in combined if "error" in r]

    null_r2s = [r["mean_r2_true"] for r in successful]
    n = len(null_r2s)
    if n < 2:
        print(f"Not enough successful perms (n={n}); aborting aggregation.")
        return

    canonical_r2 = _load_canonical_r2(args.canonical_summary)
    null_mean = float(statistics.mean(null_r2s))
    null_std = float(statistics.pstdev(null_r2s))
    z_under_null = (canonical_r2 - null_mean) / null_std if null_std > 0 else float("inf")
    n_ge = sum(1 for r in null_r2s if r >= canonical_r2)
    p_one_sided = (n_ge + 1) / (n + 1)

    # Per-perm raw R² array (sorted by perm_seed). Required by figure consumers
    # (make_interpretability_capstone_composite.py +
    # make_remaining_lab_meeting_figures.py) which draw histograms of the null
    # distribution. Kept consistent with the N=10 + N=100 schemas.
    successful_sorted = sorted(successful, key=lambda r: r["perm_seed"])
    null_mean_r2_per_perm: list[float] = [
        float(r["mean_r2_true"]) for r in successful_sorted
    ]
    perm_seeds_used: list[int] = [int(r["perm_seed"]) for r in successful_sorted]

    summary = {
        "method": "full-pipeline permutation null (shuffle labels + retrain)",
        "perm_shard_strategy": (
            f"shards={list(args.shards)} under {output_base}"
        ),
        "canonical_mean_r2": canonical_r2,
        "n_permutations": n,
        "n_failed": len(failed),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_under_null": z_under_null,
        "p_value_one_sided": p_one_sided,
        "p_value_formula": "(1 + #perms≥obs) / (N + 1)",
        "n_perms_ge_canonical": n_ge,
        "null_mean_r2_per_perm": null_mean_r2_per_perm,
        "perm_seeds_used": perm_seeds_used,
    }

    # Write combined results
    results_path = output_base / "permutation_results.json"
    summary_path = output_base / "permutation_summary.json"
    results_path.write_text(json.dumps(combined, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Summary: N={n} successful (failed={len(failed)})")
    print(f"  canonical R² = {canonical_r2:.4f}")
    print(f"  null mean    = {null_mean:+.4f} ± {null_std:.4f}")
    print(f"  z_under_null = {z_under_null:.2f}")
    print(f"  p_one_sided  = {p_one_sided:.4f}")
    print(f"  n_ge         = {n_ge}")


if __name__ == "__main__":
    main()
