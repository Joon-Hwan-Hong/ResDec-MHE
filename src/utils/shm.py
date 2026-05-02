"""Shared-memory (/dev/shm) cache cleanup utilities."""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# Regex for PID-tagged shm dirs: hpo_{pid}_{name}.
# Matches the convention from scripts/training/hpo.py shm-cache initialiser
# (HPO orchestrator names cache dirs with the worker PID so cleanup can
# tell live workers from stale ones).
_HPO_PID_RE = re.compile(r"^hpo_(\d+)_")


def cleanup_stale_shm(shm_root: Path | None = None) -> None:
    """Remove stale precomputed data caches from /dev/shm.

    Cleans up directories matching ``precomputed_*``, ``rosmap_*``, or
    ``hpo_{pid}_*`` patterns, which are created by DDP training or previous
    HPO runs.

    For PID-tagged dirs (``hpo_{pid}_{name}``), the directory is only removed
    if the owning PID is no longer alive. Legacy patterns (``precomputed_*``,
    ``rosmap_*``) without PID tags are always cleaned.

    Args:
        shm_root: Root directory to clean. Defaults to ``/dev/shm``.
    """
    import shutil

    shm_root = shm_root or Path("/dev/shm")
    if not shm_root.exists():
        # Non-Linux platforms (macOS, Windows) lack /dev/shm. Surface
        # the no-op at debug level so silent skip is at least visible.
        logger.debug(
            "cleanup_stale_shm: %s does not exist (non-Linux platform); "
            "no-op.", shm_root,
        )
        return
    removed = []
    for d in shm_root.iterdir():
        if not d.is_dir():
            continue
        # Legacy patterns (no PID): always clean
        if d.name.startswith("precomputed_") or d.name.startswith("rosmap_"):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
            continue
        # PID-tagged HPO dirs: only clean if owning PID is dead
        m = _HPO_PID_RE.match(d.name)
        if m:
            pid = int(m.group(1))
            if not _pid_alive(pid):
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d.name)
            else:
                logger.debug("Skipping /dev/shm/%s — PID %d still alive", d.name, pid)
    if removed:
        logger.info("Cleaned up stale /dev/shm caches: %s", removed)
