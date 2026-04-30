#!/usr/bin/env python
"""Render Step B counterfactual trajectory-convergence figure.

Reads the trajectory-recorded fold-0 counterfactual JSON (produced when
``run_counterfactuals.py`` is invoked with ``--record-trajectory``) and
plots squared-loss ``(f(x_t) - y_target)^2`` as a function of the
λ-doubling step for each subject.

Two panels (mode = relative, mode = absolute, both at δ=0.5).
Each panel: 20 subject lines, color-coded by regime
(resilient = blue, vulnerable = red), styled differently for converged
(success=True, solid) vs hit-λ_max (success=False, dashed).

Output:
    outputs/canonical/interpretability/figures/step_b_trajectory/
        fig_step_b_trajectory.{png,pdf}    (600 DPI)

If both source files are missing, a single-panel placeholder PNG/PDF is
produced with a "trajectory data not available" annotation; the script
still exits 0 so it can be wired into a downstream orchestrator.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

REGIME_COLORS: dict[str, str] = {
    "resilient": "#1f77b4",
    "vulnerable": "#d62728",
}


def _trajectory_to_arrays(traj: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of [lambda, signed_gap] pairs to (steps, squared_loss).

    The file stores ``[lambda_value, signed_gap_at_that_lambda]`` pairs
    where ``signed_gap = f(x_t) - y_target`` (verified empirically:
    initial gap = target_y - y_init = -0.5 in fold-0 relative δ=0.5; the
    file logs +0.499 at λ=1e-3 with the optimizer not yet moving). To
    match the spec ``(f(x_t) - y_target)^2`` we square the second
    column. We index by step (0..len-1) so the x-axis is the
    λ-doubling step regardless of the actual λ ladder used.
    """
    if not traj:
        return np.array([]), np.array([])
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.array([]), np.array([])
    steps = np.arange(arr.shape[0])
    values = arr[:, 1] ** 2  # squared loss
    return steps, values


def _render_placeholder(
    out_dir: Path,
    stem: str,
    msg: str,
) -> tuple[Path, Path]:
    """Render a single-panel placeholder figure when no data is available."""
    apply_theme(style="paper", use_scienceplots=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    ax.text(
        0.5,
        0.5,
        msg,
        ha="center",
        va="center",
        fontsize=14,
        color="dimgray",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "Step B — counterfactual trajectory convergence (no data)",
        fontsize=12,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _render_panel(
    ax: plt.Axes,
    results: Iterable[dict],
    *,
    title: str,
) -> tuple[int, int]:
    """Plot one trajectory panel; return (n_success, n_failure)."""
    n_succ = 0
    n_fail = 0
    plotted_any = False
    for r in results:
        traj = r.get("trajectory") or []
        steps, values = _trajectory_to_arrays(traj)
        if steps.size == 0:
            continue
        regime = r.get("regime", "unknown")
        success = bool(r.get("success", False))
        color = REGIME_COLORS.get(regime, "gray")
        if success:
            ls = "-"
            n_succ += 1
        else:
            ls = "--"
            n_fail += 1
        ax.plot(
            steps,
            values,
            color=color,
            linestyle=ls,
            linewidth=1.1,
            alpha=0.75,
        )
        plotted_any = True
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1.0)
    ax.set_xlabel("λ-doubling step", fontsize=11)
    ax.set_ylabel(r"$(f(x_t) - y_{\mathrm{target}})^2$", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(linestyle=":", alpha=0.5)

    # Manual legend (regime + success-flag styling)
    handles = [
        plt.Line2D([], [], color=REGIME_COLORS["resilient"], linestyle="-", label="Resilient, converged"),
        plt.Line2D([], [], color=REGIME_COLORS["resilient"], linestyle="--", label="Resilient, hit λ_max"),
        plt.Line2D([], [], color=REGIME_COLORS["vulnerable"], linestyle="-", label="Vulnerable, converged"),
        plt.Line2D([], [], color=REGIME_COLORS["vulnerable"], linestyle="--", label="Vulnerable, hit λ_max"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)

    counts_label = f"converged={n_succ} | hit λ_max={n_fail}"
    ax.text(
        0.02,
        0.02,
        counts_label,
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
        ha="left",
        va="bottom",
    )
    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No trajectory data in this file",
            ha="center",
            va="center",
            fontsize=12,
            color="darkred",
            transform=ax.transAxes,
        )
    return n_succ, n_fail


def make_figure(payloads: dict[str, dict | None]) -> plt.Figure:
    """Render the 2-panel trajectory figure.

    ``payloads`` maps mode label -> loaded JSON dict (or None if missing).
    """
    apply_theme(style="paper", use_scienceplots=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    for ax, (mode_label, payload) in zip(axes, payloads.items()):
        title = f"{mode_label} (δ=0.5, fold 0)"
        if payload is None:
            ax.text(
                0.5,
                0.5,
                f"{mode_label}: file not found",
                ha="center",
                va="center",
                fontsize=12,
                color="darkred",
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        results = payload.get("results", [])
        _render_panel(ax, results, title=title)
    fig.suptitle(
        "Step B — counterfactual squared-loss trajectory vs λ-doubling step",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relative-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/counterfactuals_trajectory_relative_delta0p5/counterfactuals_fold0.json",
        help="Path to relative-mode trajectory JSON.",
    )
    parser.add_argument(
        "--absolute-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/counterfactuals_trajectory_absolute_delta0p5/counterfactuals_fold0.json",
        help="Path to absolute-mode trajectory JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/step_b_trajectory",
        help="Output directory for the figure files.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="fig_step_b_trajectory",
        help="File stem (without extension) for PNG and PDF outputs.",
    )
    args = parser.parse_args()

    rel_exists = args.relative_json.is_file()
    abs_exists = args.absolute_json.is_file()

    if not rel_exists and not abs_exists:
        logger.warning(
            "Neither trajectory JSON exists (rel=%s, abs=%s); writing placeholder.",
            args.relative_json,
            args.absolute_json,
        )
        png, pdf = _render_placeholder(
            args.out_dir,
            args.stem,
            "Step B trajectory data not available",
        )
        logger.info("Wrote %s", png)
        logger.info("Wrote %s", pdf)
        return 0

    payloads: dict[str, dict | None] = {}
    if rel_exists:
        try:
            payloads["Relative mode"] = json.loads(args.relative_json.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Relative JSON unreadable: %s", exc)
            payloads["Relative mode"] = None
    else:
        logger.warning("Relative JSON missing: %s", args.relative_json)
        payloads["Relative mode"] = None

    if abs_exists:
        try:
            payloads["Absolute mode"] = json.loads(args.absolute_json.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Absolute JSON unreadable: %s", exc)
            payloads["Absolute mode"] = None
    else:
        logger.warning("Absolute JSON missing: %s", args.absolute_json)
        payloads["Absolute mode"] = None

    fig = make_figure(payloads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"{args.stem}.png"
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", png_path)
    logger.info("Wrote %s", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
