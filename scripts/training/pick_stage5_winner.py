"""Pick the overall-best Stage 5 production winner across d_embed groups.

For each d_embed value, scans the pipeline directory for Stage 5 run dirs
(matching ``20*HPO_d{demb}*``), extracts per-fold best val_nll from the
Lightning checkpoint filenames, averages across folds, and picks the d_embed
with the lowest mean val_nll — subject to a --min-folds completeness guard.

Prints the winner's best_config path + mean val_nll (tab-separated) to stdout.
Diagnostics (per-d_embed skip reasons, mean val_nll per group) go to stderr.
Exits 1 if no d_embed has enough completed folds.

Note: the fold-count guard is strict — a d_embed with a better observed mean
val_nll but fewer than --min-folds completed folds is SKIPPED, not selected,
even if it would win on the observed folds. This prevents declaring a winner
from a partially-failed production sweep.

Usage:
    uv run python scripts/training/pick_stage5_winner.py \\
        --pipeline-dir outputs/pipeline \\
        --d-embeds 64 128 256 \\
        --min-folds 5
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

VAL_NLL_RE = re.compile(r"val_nll=([0-9]+\.[0-9]+)")


def best_val_nll_in_run(run_dir: Path) -> float | None:
    """Return the lowest val_nll found in run_dir/checkpoints/epoch=*.ckpt, or None."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    best: float | None = None
    for ckpt in ckpt_dir.glob("epoch=*-val_nll=*.ckpt"):
        m = VAL_NLL_RE.search(ckpt.name)
        if not m:
            continue
        try:
            v = float(m.group(1))
        except ValueError as e:
            logger.warning("Malformed val_nll in %s: %s", ckpt.name, e)
            continue
        if best is None or v < best:
            best = v
    return best


def mean_val_nll_for_d_embed(
    pipeline_dir: Path, demb: int, min_folds: int
) -> tuple[float, int] | str:
    """Return (mean, n_folds) on success, or a skip-reason string on failure."""
    cfg_path = pipeline_dir / f"best_config_d{demb}.yaml"
    if not cfg_path.exists():
        return f"no best_config_d{demb}.yaml"

    # 20* matches any year 2000-2099 so the pipeline script doesn't break in 2027.
    # run_sensitivity.sh produces dirs like "20260410_HHMMSS_<hash>_HPO_d{N}_*".
    run_dirs = sorted(
        d for d in pipeline_dir.glob(f"20*HPO_d{demb}*") if d.is_dir()
    )
    fold_nlls: list[float] = []
    for run_dir in run_dirs:
        nll = best_val_nll_in_run(run_dir)
        if nll is not None:
            fold_nlls.append(nll)

    if len(fold_nlls) < min_folds:
        return (
            f"only {len(fold_nlls)}/{min_folds} folds complete "
            f"(scanned {len(run_dirs)} run dirs)"
        )
    return sum(fold_nlls) / len(fold_nlls), len(fold_nlls)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,  # CRITICAL: diagnostics go to stderr, winner goes to stdout
    )
    parser = argparse.ArgumentParser(
        description=(
            __doc__.splitlines()[0] if __doc__ and __doc__.strip() else None
        ),
    )
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument(
        "--d-embeds",
        required=True,
        type=int,
        nargs="+",
        help="d_embed values to compare (e.g., 64 128 256)",
    )
    parser.add_argument(
        "--min-folds",
        type=int,
        default=5,
        help="Minimum completed folds required to qualify as winner (default: 5)",
    )
    args = parser.parse_args()

    if not args.pipeline_dir.is_dir():
        logger.error("Pipeline dir not found: %s", args.pipeline_dir)
        return 1

    best_demb: int | None = None
    best_mean: float | None = None
    best_cfg: Path | None = None
    skip_reasons: dict[int, str] = {}

    for demb in args.d_embeds:
        result = mean_val_nll_for_d_embed(
            args.pipeline_dir, demb, args.min_folds,
        )
        if isinstance(result, str):
            logger.info("skipping d_embed=%d: %s", demb, result)
            skip_reasons[demb] = result
            continue
        mean_nll, n_folds = result
        logger.info(
            "d_embed=%d mean val_nll=%.6f (from %d folds)",
            demb,
            mean_nll,
            n_folds,
        )
        if best_mean is None or mean_nll < best_mean:
            best_demb = demb
            best_mean = mean_nll
            best_cfg = args.pipeline_dir / f"best_config_d{demb}.yaml"

    if best_demb is None:
        logger.error("No d_embed qualified as winner.")
        for d, r in skip_reasons.items():
            logger.error("  d_embed=%d: %s", d, r)
        return 1

    logger.info(
        "WINNER d_embed=%d mean_val_nll=%.6f cfg=%s",
        best_demb,
        best_mean,
        best_cfg,
    )
    # Tab-separated result on stdout for shell capture
    print(f"{best_cfg}\t{best_mean:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
