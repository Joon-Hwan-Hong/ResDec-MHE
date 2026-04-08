"""Update ablation configs in-place to match a chosen Stage 5 winner config
while preserving each ablation's structural override.

Each ablation represents the effect of removing/changing ONE model component
while holding everything else fixed. When Stage 5 finds a new HPO winner,
the ablation HPs become stale. This script re-bakes each ablation YAML as:

    updated_ablation = winner_config + ABLATION_OVERRIDES[name]

The ablation's structural override (e.g., disable HGT, swap fusion type) is
preserved by applying it LAST on top of the winner config. Everything else —
optimizer HPs, loss HPs, schedule, gene_gate temp, etc. — is inherited from
the winner so that the ablation measures pure component effect.

Usage:
    uv run python scripts/training/update_ablation_configs.py \\
        --winner outputs/pipeline/best_config_d128.yaml \\
        --ablation-dir configs/ablations \\
        --output-dir configs/ablations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Each ablation's structural override. Verified against
# configs/ablations/ablation_*.yaml on 2026-04-07 by diffing against
# HPO7 Rank 3 baseline.
ABLATION_OVERRIDES: dict[str, list[tuple[str, object]]] = {
    "ablation_ct_only": [
        ("model.use_hgt_encoder", False),
    ],
    "ablation_hgt_only": [
        ("model.use_cell_transformer", False),
    ],
    "ablation_no_gene_gate": [
        ("model.gene_gate.initial_temperature", 100.0),
    ],
    "ablation_no_pathology_attention": [
        ("model.use_pathology_attention", False),
    ],
    "ablation_fusion_cross_attention": [
        ("model.fusion.type", "cross_attention"),
    ],
    "ablation_fusion_crossfuse": [
        ("model.fusion.type", "crossfuse"),
    ],
    "ablation_fusion_crossfuse_blend": [
        ("model.fusion.type", "crossfuse_blend"),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", required=True, type=Path,
                        help="Path to the Stage 5 winner config YAML")
    parser.add_argument("--ablation-dir", required=True, type=Path,
                        help="Directory containing ablation_*.yaml files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to write the updated ablation configs")
    args = parser.parse_args()

    if not args.winner.is_file():
        logger.error("Winner config not found: %s", args.winner)
        return 1

    winner = OmegaConf.load(args.winner)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ablation_name, overrides in ABLATION_OVERRIDES.items():
        src = args.ablation_dir / f"{ablation_name}.yaml"
        if not src.is_file():
            logger.warning("Missing ablation file: %s (skipping)", src)
            continue

        # Start from a deep-copy of the winner, then apply the ablation override.
        updated = OmegaConf.create(OmegaConf.to_container(winner, resolve=True))

        # Preserve the _ablation metadata from the original ablation YAML so
        # the provenance stays intact (name + description)
        orig = OmegaConf.load(src)
        if "_ablation" in orig:
            OmegaConf.update(updated, "_ablation", orig["_ablation"], merge=True)

        # Set the ablation's run_name to reflect both the ablation and the
        # winner's d_embed basin
        d_embed_val = winner.model.get("d_embed", 64)
        OmegaConf.update(
            updated, "experiment.run_name",
            f"{ablation_name}_d{d_embed_val}",
        )

        # Apply the structural override last
        for key, value in overrides:
            OmegaConf.update(updated, key, value)
            logger.info("[%s] set %s = %r", ablation_name, key, value)

        out_path = args.output_dir / f"{ablation_name}.yaml"
        OmegaConf.save(updated, out_path)
        logger.info("Wrote %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
