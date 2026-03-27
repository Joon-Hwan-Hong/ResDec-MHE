"""Bucket batch sampler for grouping subjects with similar edge counts.

Reduces padding waste by batching subjects with similar CCC edge counts together.
Without bucket batching, a single high-edge subject (134K edges) forces the entire
batch of 16 to pad to 134K, wasting compute on ~15 subjects that may have far fewer
edges. Sorting by edge count and batching neighbors reduces max/min ratio within
each batch from ~60x to ~2-3x.

Compatible with DDP: each rank gets a disjoint subset of batches.
"""

import math
from typing import Iterator

import numpy as np
import torch.utils.data


class EdgeCountBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batch sampler that groups samples by edge count to minimize padding.

    Strategy:
    1. Sort dataset indices by edge count
    2. Form contiguous batches from the sorted order
    3. Shuffle the batch order (not within-batch order) each epoch
    4. For DDP: each rank takes every world_size-th batch

    This preserves data diversity across epochs (batch order is shuffled)
    while ensuring each batch has minimal edge count variance.

    Args:
        edge_counts: Edge count per sample, indexed by dataset position.
        batch_size: Number of samples per batch.
        drop_last: If True, drop the last incomplete batch.
        shuffle: If True, shuffle the order of batches each epoch.
        seed: Random seed for batch-order shuffling.
        rank: DDP rank (0 if single-GPU).
        world_size: Number of DDP ranks (1 if single-GPU).
    """

    def __init__(
        self,
        edge_counts: list[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.edge_counts = edge_counts
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        # Sort indices by edge count (ascending)
        self._sorted_indices = np.argsort(edge_counts).tolist()

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic batch-order shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        # Form batches from sorted indices
        batches = []
        for i in range(0, len(self._sorted_indices), self.batch_size):
            batch = self._sorted_indices[i : i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)

        # Shuffle batch order (deterministic per epoch)
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            perm = rng.permutation(len(batches))
            batches = [batches[i] for i in perm]

        # DDP: each rank takes every world_size-th batch
        if self.world_size > 1:
            batches = batches[self.rank :: self.world_size]

        yield from batches

    def __len__(self) -> int:
        n_batches = len(self._sorted_indices) // self.batch_size
        if not self.drop_last and len(self._sorted_indices) % self.batch_size != 0:
            n_batches += 1
        if self.world_size > 1:
            n_batches = math.ceil(n_batches / self.world_size)
        return n_batches
