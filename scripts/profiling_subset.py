"""Pre-select subjects with diverse edge counts for reproducible profiling.

Creates a fixed subset of subjects spanning the edge count distribution
(p0, p10, p25, p50, p75, p90, p95, p99, max) so that profiling runs
see representative batch variance without random sampling noise.

The subset is saved to a JSON file and reused across profiling experiments.

Usage:
    # Generate the subset (only needed once):
    .venv/bin/python scripts/profiling_subset.py \
        --precomputed-dir data/precomputed/rosmap/ \
        --n-subjects 64

    # Profile scripts use it via --profile-subset:
    .venv/bin/python scripts/profile_training.py \
        --profile-subset outputs/profiling/profile_subset.json ...
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_profiling_subset(
    feature_dir: Path,
    n_subjects: int = 64,
    seed: int = 42,
) -> dict:
    """Select subjects spanning the edge count distribution.

    Strategy:
    1. Scan all .npz files for edge counts
    2. Pick one subject at each percentile (p0, p10, p25, p50, p75, p90, p95, p99, max)
       as "anchors" — these are always included
    3. Fill remaining slots by stratified sampling from quartile bins

    Returns dict with subject_ids list and edge count metadata.
    """
    rng = np.random.RandomState(seed)

    # Scan edge counts
    subjects = []
    npz_files = sorted(feature_dir.glob("*.npz"))
    for f in npz_files:
        if f.stem == "gene_names":
            continue
        with np.load(f, allow_pickle=True) as data:
            n_edges = data["edge_index"].shape[1]
        subjects.append((f.stem, n_edges))

    if not subjects:
        raise ValueError(f"No .npz files found in {feature_dir}")

    subjects.sort(key=lambda x: x[1])
    all_edges = np.array([e for _, e in subjects])

    # Anchor percentiles
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    anchor_indices = set()
    for p in percentiles:
        target = np.percentile(all_edges, p)
        idx = np.argmin(np.abs(all_edges - target))
        anchor_indices.add(idx)

    # Fill remaining slots from quartile bins
    n_remaining = n_subjects - len(anchor_indices)
    if n_remaining > 0:
        bin_edges = np.percentile(all_edges, [0, 25, 50, 75, 100])
        bins = np.digitize(all_edges, bin_edges[1:-1])  # 0-3 for 4 quartiles

        per_bin = max(1, n_remaining // 4)
        for b in range(4):
            bin_candidates = [
                i for i in range(len(subjects))
                if bins[i] == b and i not in anchor_indices
            ]
            n_pick = min(per_bin, len(bin_candidates))
            if n_pick > 0:
                picked = rng.choice(bin_candidates, size=n_pick, replace=False)
                anchor_indices.update(picked)

    selected = sorted(anchor_indices)[:n_subjects]
    selected_subjects = [subjects[i] for i in selected]

    result = {
        "subject_ids": [s[0] for s in selected_subjects],
        "edge_counts": {s[0]: s[1] for s in selected_subjects},
        "stats": {
            "n_selected": len(selected_subjects),
            "n_total": len(subjects),
            "edge_min": int(all_edges.min()),
            "edge_p25": int(np.percentile(all_edges, 25)),
            "edge_median": int(np.median(all_edges)),
            "edge_p75": int(np.percentile(all_edges, 75)),
            "edge_p95": int(np.percentile(all_edges, 95)),
            "edge_max": int(all_edges.max()),
            "selected_edge_min": int(min(e for _, e in selected_subjects)),
            "selected_edge_max": int(max(e for _, e in selected_subjects)),
        },
        "seed": seed,
    }
    return result


def load_profiling_subset(path: Path) -> list[str]:
    """Load pre-selected subject IDs from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["subject_ids"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate profiling subset with diverse edge counts"
    )
    parser.add_argument(
        "--precomputed-dir", type=str, required=True,
        help="Path to precomputed feature directory",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=64,
        help="Number of subjects to select (default: 64 = 4 batches of 16)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/profiling/profile_subset.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = build_profiling_subset(
        Path(args.precomputed_dir),
        n_subjects=args.n_subjects,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    stats = result["stats"]
    print(f"\nProfiling subset: {stats['n_selected']} / {stats['n_total']} subjects")
    print(f"Population edges: min={stats['edge_min']}, median={stats['edge_median']}, "
          f"p95={stats['edge_p95']}, max={stats['edge_max']}")
    print(f"Selected edges:   min={stats['selected_edge_min']}, "
          f"max={stats['selected_edge_max']}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
