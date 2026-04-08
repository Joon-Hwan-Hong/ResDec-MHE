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

import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

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


def update_one_ablation(
    winner: DictConfig,
    src: Path,
    out_path: Path,
    ablation_name: str,
    overrides: list[tuple[str, object]],
    winner_filename: str,
    d_embed_val: int,
) -> bool:
    """Update one ablation YAML. Returns True on success, False on skip/error."""
    try:
        orig = OmegaConf.load(src)
    except (OSError, ValueError, yaml.YAMLError, OmegaConfBaseException) as e:
        logger.warning("Failed to parse %s: %s — skipping", src, e)
        return False

    # resolve=True freezes interpolations so each ablation YAML is self-contained
    cfg = OmegaConf.create(OmegaConf.to_container(winner, resolve=True))

    # Drop stale provenance. The winner deep-copy can carry a top-level
    # _hpo_provenance from the winner itself (e.g., if someone passes a
    # non-canonical winner config). In the production Stage 5 pipeline this
    # is empty, but we enforce the drop unconditionally so the emitted YAML
    # never claims provenance that doesn't match its HPs. See Task 12 design
    # decision D: _ablation.base_config carries the only provenance hint.
    if "_hpo_provenance" in cfg:
        del cfg["_hpo_provenance"]

    # Preserve the _ablation metadata from the original ablation YAML so
    # the provenance stays intact (name + description).
    #
    # _hpo_provenance is intentionally dropped: it would be stale (HPs now come
    # from the Stage 5 winner, not the original HPO trial). Overwrite
    # _ablation.base_config to reflect the new source so the YAML isn't lying
    # about where its HPs came from. Full Stage 5 winner trial_id threading is
    # deferred to a follow-up task.
    if "_ablation" in orig:
        OmegaConf.update(cfg, "_ablation", orig["_ablation"])
    OmegaConf.update(
        cfg, "_ablation.base_config",
        f"{winner_filename} (Stage 5)",
    )

    run_name = f"{ablation_name}_d{d_embed_val}"
    OmegaConf.update(cfg, "experiment.run_name", run_name)
    logger.info("[%s] set experiment.run_name = %r", ablation_name, run_name)

    # Apply the structural override last
    for key, value in overrides:
        OmegaConf.update(cfg, key, value)
        logger.info("[%s] set %s = %r", ablation_name, key, value)

    try:
        OmegaConf.save(cfg, out_path)
    except OSError as e:
        logger.warning("Failed to save %s: %s — skipping", out_path, e)
        return False
    logger.info("Wrote %s", out_path)
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(__doc__.splitlines()[0] if __doc__ and __doc__.strip() else None),
    )
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

    if not args.ablation_dir.is_dir():
        logger.error("Ablation dir not found: %s", args.ablation_dir)
        return 1

    try:
        winner = OmegaConf.load(args.winner)
    except (OSError, ValueError, yaml.YAMLError, OmegaConfBaseException) as e:
        logger.error("Failed to parse winner %s: %s", args.winner, e)
        return 1

    if "model" not in winner or "d_embed" not in winner.model:
        logger.error("Winner config missing model.d_embed: %s", args.winner)
        return 1
    d_embed_val = int(winner.model.d_embed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir.resolve() == args.ablation_dir.resolve():
        logger.warning(
            "Output dir == ablation dir (%s); updates will overwrite originals in place",
            args.output_dir,
        )

    n_ok = 0
    n_total = len(ABLATION_OVERRIDES)
    for ablation_name, overrides in ABLATION_OVERRIDES.items():
        src = args.ablation_dir / f"{ablation_name}.yaml"
        if not src.is_file():
            logger.warning("Missing ablation file: %s (skipping)", src)
            continue

        out_path = args.output_dir / f"{ablation_name}.yaml"
        if update_one_ablation(
            winner=winner,
            src=src,
            out_path=out_path,
            ablation_name=ablation_name,
            overrides=overrides,
            winner_filename=args.winner.name,
            d_embed_val=d_embed_val,
        ):
            n_ok += 1

    logger.info("Updated %d/%d ablations", n_ok, n_total)
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
