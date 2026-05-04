"""Aggregate variant perm-null shard outputs and compute z + empirical p.

Reads <shards-dir>/shard_a/permutation_results.json + shard_b/, combines into
one results list, drops duplicates by perm_seed, and writes a summary JSON
with z (vs canonical mean R²) and one-sided empirical p (= (1 + n_null_ge_canonical) / (1 + N)).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards-dir", type=Path, required=True,
                   help="Directory holding shard_a/, shard_b/.")
    p.add_argument("--canonical-summary", type=Path, required=True,
                   help="Path to variant best_vs_tabpfn_summary.json (provides "
                        "canonical mean R² of the variant for the z computation).")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    rows: list[dict] = []
    seen: set[int] = set()
    for shard in ("shard_a", "shard_b"):
        path = args.shards_dir / shard / "permutation_results.json"
        if not path.is_file():
            print(f"warning: {path} missing; skipping", flush=True)
            continue
        for r in json.loads(path.read_text()):
            if "error" in r:
                continue
            ps = r["perm_seed"]
            if ps in seen:
                continue
            seen.add(ps)
            rows.append(r)

    if not rows:
        raise SystemExit("no usable perm rows found across both shards")

    canonical_summary = json.loads(args.canonical_summary.read_text())
    canon_r2_per_fold = [f["ours"]["r2"] for f in canonical_summary["per_fold"]]
    canon_mean_r2 = statistics.fmean(canon_r2_per_fold)

    null_means = np.array([r["mean_r2_true"] for r in rows], dtype=float)
    null_mean = float(null_means.mean())
    null_std = float(null_means.std(ddof=0))
    n = int(len(null_means))
    n_ge = int((null_means >= canon_mean_r2).sum())
    p_one_sided = (n_ge + 1) / (n + 1)
    z = (canon_mean_r2 - null_mean) / null_std if null_std > 0 else float("inf")

    summary = {
        "canonical_mean_r2": canon_mean_r2,
        "canonical_per_fold_r2": canon_r2_per_fold,
        "n_permutations": n,
        "null_mean": null_mean,
        "null_std": null_std,
        "n_null_ge_canonical": n_ge,
        "z_under_null": z,
        "p_value_one_sided": p_one_sided,
        "p_floor_at_n": 1.0 / (n + 1),
        "perm_seeds": sorted(int(r["perm_seed"]) for r in rows),
        "shards_dir": str(args.shards_dir),
        "canonical_summary_path": str(args.canonical_summary),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")
    print(f"canonical mean R² = {canon_mean_r2:+.4f}")
    print(f"null mean         = {null_mean:+.4f} ± {null_std:.4f} (N={n})")
    print(f"z under null      = {z:+.3f}")
    print(f"empirical p (1-sided) = {p_one_sided:.4f}  (floor at N={n}: {1.0 / (n + 1):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
