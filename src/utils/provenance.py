"""Tiny utilities for run provenance metadata.

Used by orchestrators that emit JSON output and want a git-SHA stamp without
each one re-implementing the subprocess call.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def git_sha(cwd: Path | str | None = None) -> str:
    """Return the current `git rev-parse HEAD` SHA, or 'unknown' on failure.

    Parameters
    ----------
    cwd
        Optional working directory; defaults to ``Path.cwd()`` (which for
        worktree-based work is the worktree root).
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


_BEST_CKPT_RE = re.compile(r"^best-(\d+)-(-?\d+\.\d+)\.ckpt$")


def pick_max_r2_ckpt(ckpt_dir: Path) -> Path:
    """Return the ``best-{epoch}-{r2}.ckpt`` with the largest R² in ``ckpt_dir``.

    The canonical training pipeline writes per-fold checkpoints named
    ``best-{epoch:03d}-{val_r2:+.4f}.ckpt``; this helper picks the one with
    the highest val_r2 (e.g. ``best-052-+0.498.ckpt``).

    Raises
    ------
    FileNotFoundError
        If no matching checkpoint is found.
    """
    best: tuple[Path, float] | None = None
    for p in ckpt_dir.glob("best-*.ckpt"):
        m = _BEST_CKPT_RE.match(p.name)
        if not m:
            continue
        r2 = float(m.group(2))
        if best is None or r2 > best[1]:
            best = (p, r2)
    if best is None:
        raise FileNotFoundError(f"No best-*.ckpt in {ckpt_dir}")
    return best[0]
