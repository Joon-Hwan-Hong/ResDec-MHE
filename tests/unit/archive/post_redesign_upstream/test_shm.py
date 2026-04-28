"""Tests for src/utils/shm.py — shared-memory cache cleanup utilities."""

import os

import pytest

from src.utils.shm import _pid_alive, cleanup_stale_shm


class TestCleanupStaleShm:
    """Tests for stale /dev/shm cache cleanup."""

    def test_cleanup_stale_shm_removes_legacy_dirs(self, tmp_path):
        """cleanup_stale_shm removes precomputed_* and rosmap_* dirs (legacy patterns)."""
        stale1 = tmp_path / "precomputed_rosmap"
        stale1.mkdir()
        (stale1 / "R001.pt").write_bytes(b"data")

        stale2 = tmp_path / "rosmap_precomputed"
        stale2.mkdir()
        (stale2 / "R002.pt").write_bytes(b"data")

        keep = tmp_path / "other_data"
        keep.mkdir()

        cleanup_stale_shm(tmp_path)
        assert not stale1.exists()
        assert not stale2.exists()
        assert keep.exists()

    def test_cleanup_stale_shm_removes_dead_pid_hpo_dirs(self, tmp_path):
        """cleanup_stale_shm removes hpo_{pid}_{name} dirs when PID is dead."""
        # Use PID 999999999 which is almost certainly not alive
        stale = tmp_path / "hpo_999999999_rosmap"
        stale.mkdir()
        (stale / "R001.pt").write_bytes(b"data")

        cleanup_stale_shm(tmp_path)
        assert not stale.exists()

    def test_cleanup_stale_shm_keeps_alive_pid_hpo_dirs(self, tmp_path):
        """cleanup_stale_shm skips hpo_{pid}_{name} dirs when PID is alive."""
        # Use our own PID (guaranteed alive)
        alive = tmp_path / f"hpo_{os.getpid()}_rosmap"
        alive.mkdir()
        (alive / "R001.pt").write_bytes(b"data")

        cleanup_stale_shm(tmp_path)
        assert alive.exists()

    def test_cleanup_stale_shm_noop_on_empty(self, tmp_path):
        """cleanup_stale_shm is a no-op when no matching dirs exist."""
        cleanup_stale_shm(tmp_path)  # should not raise

    def test_pid_alive_helper(self):
        """_pid_alive returns True for own PID, False for dead PID."""
        assert _pid_alive(os.getpid()) is True
        assert _pid_alive(999999999) is False
