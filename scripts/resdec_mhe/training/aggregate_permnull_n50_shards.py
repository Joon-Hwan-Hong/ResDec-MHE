#!/usr/bin/env python
"""Aggregate two permutation-null shards into a single N=50 summary.

Reads:
    outputs/canonical/permutation_test_n50_full/shard_a/permutation_results.json (perms 0-24)
    outputs/canonical/permutation_test_n50_full/shard_b/permutation_results.json (perms 25-49)

Writes:
    outputs/canonical/permutation_test_n50_full/permutation_results.json (combined)
    outputs/canonical/permutation_test_n50_full/permutation_summary.json (aggregate stats)
"""
from __future__ import annotations
import json
from pathlib import Path
import statistics

WT = Path(__file__).resolve().parents[3]
ROOT = WT / "outputs/canonical/permutation_test_n50_full"
CANONICAL_PERMNULL_PATH = WT / "outputs/canonical/permutation_test/permutation_summary.json"


def _load_canonical_r2() -> float:
    """Load canonical 5-fold mean R² from existing N=10 perm summary.

    Single source of truth across N=10 / N=50 / N=100 perm tests. Avoids
    a hardcoded literal that goes stale if the canonical pipeline is rerun.
    """
    if not CANONICAL_PERMNULL_PATH.exists():
        raise FileNotFoundError(
            f"Canonical N=10 perm summary missing at {CANONICAL_PERMNULL_PATH}; "
            "cannot derive canonical R² for the N=50 z-score."
        )
    return float(json.loads(CANONICAL_PERMNULL_PATH.read_text())["canonical_mean_r2"])


def main():
    shard_a_path = ROOT / "shard_a/permutation_results.json"
    shard_b_path = ROOT / "shard_b/permutation_results.json"
    if not shard_a_path.exists() or not shard_b_path.exists():
        print(f"Shard files not yet present:\n  {shard_a_path}: {shard_a_path.exists()}\n  {shard_b_path}: {shard_b_path.exists()}")
        return
    a = json.loads(shard_a_path.read_text())
    b = json.loads(shard_b_path.read_text())
    combined = a + b

    # Filter to successful perms (skip those with 'error' field)
    successful = [r for r in combined if "error" not in r and r.get("mean_r2_true") is not None]
    failed = [r for r in combined if "error" in r]

    null_r2s = [r["mean_r2_true"] for r in successful]
    n = len(null_r2s)
    if n < 2:
        print(f"Not enough successful perms (n={n}); aborting aggregation.")
        return

    canonical_r2 = _load_canonical_r2()
    null_mean = float(statistics.mean(null_r2s))
    null_std = float(statistics.pstdev(null_r2s))
    z_under_null = (canonical_r2 - null_mean) / null_std if null_std > 0 else float("inf")
    n_ge = sum(1 for r in null_r2s if r >= canonical_r2)
    p_one_sided = (n_ge + 1) / (n + 1)

    summary = {
        "method": "full-pipeline permutation null (shuffle labels + retrain)",
        "perm_shard_strategy": "perms 0-24 on GPU 0 (shard_a), perms 25-49 on GPU 1 (shard_b)",
        "canonical_mean_r2": canonical_r2,
        "n_permutations": n,
        "n_failed": len(failed),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_under_null": z_under_null,
        "p_value_one_sided": p_one_sided,
        "p_value_formula": "(1 + #perms≥obs) / (N + 1)",
        "n_perms_ge_canonical": n_ge,
        "perm_seeds_used": sorted([r["perm_seed"] for r in successful]),
    }

    # Write combined results
    (ROOT / "permutation_results.json").write_text(json.dumps(combined, indent=2))
    (ROOT / "permutation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {ROOT / 'permutation_results.json'}")
    print(f"Wrote {ROOT / 'permutation_summary.json'}")
    print(f"Summary: N={n} successful (failed={len(failed)})")
    print(f"  canonical R² = {canonical_r2:.4f}")
    print(f"  null mean    = {null_mean:+.4f} ± {null_std:.4f}")
    print(f"  z_under_null = {z_under_null:.2f}")
    print(f"  p_one_sided  = {p_one_sided:.4f}")
    print(f"  n_ge         = {n_ge}")


if __name__ == "__main__":
    main()
