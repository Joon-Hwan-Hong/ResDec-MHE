"""Threaded batch prefetcher for overlapping collation with GPU compute.

With num_workers=0, DataLoader collation (torch.cat of ~1.4 GB cell_data)
runs synchronously in the main thread, blocking GPU compute.  This module
provides a wrapper that runs collation + device transfer in a background
thread so they overlap with forward/backward.

Works because:
- torch.cat/torch.stack release the GIL during C++ memcpy
- CUDA kernels (forward/backward) run without the GIL
- .to(device, non_blocking=True) releases the GIL during DMA

Benchmarked improvement (2x RTX 6000 Ada, ROSMAP dataset):
- 2-GPU without prefetch: 26.6 samples/sec (data_load=628ms/step)
- 2-GPU with prefetch:    45.6 samples/sec (data_load=103ms/step)
"""

import queue
import threading

import torch


class ThreadedPrefetcher:
    """Prefetch batches in a background thread.

    Wraps any iterable (typically a DataLoader) and yields batches that
    have already been moved to ``device``.  The next batch is collated
    and transferred while the current batch is being processed on the GPU.

    Args:
        dataloader: Iterable that yields dict batches.
        device: Target CUDA device for batch tensors.
        prefetch_count: Number of batches to buffer ahead.  Higher values
            use more GPU memory (~4 GB per buffered batch).  Default 2.
    """

    def __init__(
        self,
        dataloader,
        device: torch.device,
        prefetch_count: int = 2,
    ):
        self.dataloader = dataloader
        self.device = device
        self.prefetch_count = prefetch_count
        self._epoch = 0

    def __iter__(self):
        # Sync DistributedSampler epoch for proper shuffling across epochs.
        # When ThreadedPrefetcher wraps the train DataLoader, Lightning skips
        # its own set_epoch call (only applies to DataLoader instances).
        dl = self.dataloader
        if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
            dl.sampler.set_epoch(self._epoch)
            self._epoch += 1

        q: queue.Queue = queue.Queue(maxsize=self.prefetch_count)
        error_box: list = [None]

        def _producer():
            try:
                for batch in self.dataloader:
                    moved = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            moved[k] = v.to(self.device, non_blocking=True)
                        else:
                            moved[k] = v
                    q.put(moved)
            except Exception as e:
                error_box[0] = e
            finally:
                q.put(None)  # sentinel

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            batch = q.get()
            if batch is None:
                break
            if error_box[0] is not None:
                raise error_box[0]
            yield batch

        thread.join(timeout=30)
        if error_box[0] is not None:
            raise error_box[0]

    def __len__(self):
        return len(self.dataloader)
