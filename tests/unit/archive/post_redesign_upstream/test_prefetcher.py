"""Tests for src/data/prefetch.py — ThreadedPrefetcher abort behavior."""

import threading
import time

import pytest
import torch

from src.data.prefetch import ThreadedPrefetcher


class _SlowDataLoader:
    """Fake DataLoader that yields batches slowly, counting how many were produced."""

    def __init__(self, n_batches: int, delay: float = 0.1):
        self.n_batches = n_batches
        self.delay = delay
        self.produced = 0

    def __iter__(self):
        for i in range(self.n_batches):
            time.sleep(self.delay)
            self.produced += 1
            yield {"x": torch.tensor([float(i)])}

    def __len__(self):
        return self.n_batches


class TestPrefetcherAbort:
    """Tests that producer thread stops promptly when consumer stops iterating."""

    def test_early_exit_stops_producer(self):
        """If consumer only takes 2 of 20 batches, producer should stop quickly."""
        loader = _SlowDataLoader(n_batches=20, delay=0.05)
        prefetcher = ThreadedPrefetcher(loader, device=torch.device("cpu"), prefetch_count=2)

        consumed = []
        for batch in prefetcher:
            consumed.append(batch)
            if len(consumed) >= 2:
                break

        # Wait a moment for the abort to propagate
        time.sleep(0.5)

        assert len(consumed) == 2
        # Producer should not have produced all 20 batches.
        # With prefetch_count=2 and consuming 2, producer should have produced
        # at most ~4-6 (2 consumed + 2 buffered + a small margin).
        assert loader.produced < 10, (
            f"Producer made {loader.produced} batches after consumer took only 2 — "
            f"abort event not working"
        )

    def test_full_iteration_works(self):
        """Normal full iteration should yield all batches."""
        loader = _SlowDataLoader(n_batches=5, delay=0.01)
        prefetcher = ThreadedPrefetcher(loader, device=torch.device("cpu"), prefetch_count=2)

        results = list(prefetcher)
        assert len(results) == 5
        assert loader.produced == 5

    def test_exception_in_consumer_sets_abort(self):
        """If consumer raises, producer should not hang indefinitely."""
        loader = _SlowDataLoader(n_batches=20, delay=0.05)
        prefetcher = ThreadedPrefetcher(loader, device=torch.device("cpu"), prefetch_count=2)

        with pytest.raises(ValueError, match="test error"):
            for i, batch in enumerate(prefetcher):
                if i >= 1:
                    raise ValueError("test error")

        # Producer should stop promptly
        time.sleep(0.5)
        assert loader.produced < 10

    def test_no_active_threads_after_exit(self):
        """After early exit, no producer threads should remain active."""
        loader = _SlowDataLoader(n_batches=50, delay=0.05)
        prefetcher = ThreadedPrefetcher(loader, device=torch.device("cpu"), prefetch_count=2)

        # Count threads before
        threads_before = threading.active_count()

        for i, batch in enumerate(prefetcher):
            if i >= 1:
                break

        # Give thread time to exit
        time.sleep(1.0)
        threads_after = threading.active_count()

        # Should not have leaked threads
        assert threads_after <= threads_before, (
            f"Leaked threads: {threads_after} active vs {threads_before} before"
        )
