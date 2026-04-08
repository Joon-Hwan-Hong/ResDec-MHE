"""Extract the top-1 completed trial for each d_embed value from a Ray Tune
results directory, and save each as a standalone training config YAML.

Used in Stage 5 of the pipeline when the Claude agent analysis failed to
produce ``best_config_d{64,128,256}.yaml``. Walks the ray_results directory,
finds each trial's ``result.json`` (last finite val_nll wins; first record
with a config key wins for config), groups by ``d_embed``, picks the lowest
val_nll per group, loads the base config, applies the winning HPs via
``build_config_from_ray``, and saves the result.

Trials whose logged config lacks a ``d_embed`` key (e.g., legacy HPO6/7/8
runs that hardcoded d_embed outside the search space) fall back to
``WARM_START_BASELINE_D_EMBED`` from ``scripts.training.hpo`` and emit an
INFO log line so the operator can see the fallback was applied. This
mirrors the warm-start defaults path used by ``hpo.py`` itself.

Usage:
    uv run python scripts/training/extract_best_per_d_embed.py \\
        --ray-dir outputs/pipeline/ray_results/cognitive_resilience \\
        --base-config configs/default.yaml \\
        --output-dir outputs/pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Make scripts.training importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.hpo import (  # noqa: E402
    WARM_START_BASELINE_D_EMBED,
    build_config_from_ray,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRIAL_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def load_trial(trial_dir: Path) -> tuple[dict[str, object], float] | None:
    """Return (config, final_val_nll) or None if the trial is incomplete."""
    result_file = trial_dir / "result.json"
    if not result_file.exists():
        return None
    final_nll = float("inf")
    config: dict[str, object] = {}
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = record.get("val_nll")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                final_nll = v
            if not config and "config" in record:
                config = record["config"]
    if not config or not math.isfinite(final_nll):
        return None
    return config, final_nll


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--ray-dir", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    ray_dir: Path = args.ray_dir
    if not ray_dir.is_dir():
        logger.error("Ray dir %s does not exist", ray_dir)
        return 1

    # Find latest experiment timestamp to filter old trials
    state_files = sorted(ray_dir.glob("experiment_state-*.json"))
    latest_ts = None
    if state_files:
        latest_ts = state_files[-1].stem.replace("experiment_state-", "")
        logger.info("Filtering to experiment %s", latest_ts)

    # Group trials by d_embed
    by_demb: dict[int, list[tuple[dict, float]]] = {}
    for trial_dir in sorted(ray_dir.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("train_fn_"):
            continue
        if latest_ts:
            ts_match = TRIAL_TIMESTAMP_RE.search(trial_dir.name)
            if ts_match and ts_match.group(1) < latest_ts:
                continue
        loaded = load_trial(trial_dir)
        if loaded is None:
            continue
        config, nll = loaded
        demb = config.get("d_embed")
        if demb is None:
            # Legacy HPO runs (HPO6/7/8) hardcoded d_embed outside the search
            # space, so the logged config dict has no d_embed key. Fall back
            # to the baseline constant from hpo.py — same convention as the
            # warm-start defaults path at hpo.py:976-983. INFO-level so the
            # fallback is visible but not alarmist.
            demb = WARM_START_BASELINE_D_EMBED
            logger.info(
                "Trial %s missing d_embed — using WARM_START_BASELINE_D_EMBED=%d",
                trial_dir.name, demb,
            )
        if demb not in (64, 128, 256):
            logger.warning("Unexpected d_embed=%s in trial %s — including anyway", demb, trial_dir.name)
        by_demb.setdefault(demb, []).append((config, nll))

    if not by_demb:
        logger.error("No completed trials found in %s", ray_dir)
        return 1

    # Pick the best (lowest val_nll) per d_embed
    base = OmegaConf.load(args.base_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for demb in sorted(by_demb):
        trials = by_demb[demb]
        trials.sort(key=lambda t: t[1])
        best_config, best_nll = trials[0]
        logger.info(
            "d_embed=%d: best val_nll=%.4f (from %d trials)",
            demb, best_nll, len(trials),
        )
        merged = build_config_from_ray(best_config, base)
        if merged is None:
            logger.warning(
                "d_embed=%d best trial is invalid (skip-signal from builder)",
                demb,
            )
            continue
        OmegaConf.update(merged, "experiment.run_name", f"HPO_d{demb}", merge=True)
        out_path = args.output_dir / f"best_config_d{demb}.yaml"
        OmegaConf.save(merged, out_path)
        logger.info("Wrote %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
