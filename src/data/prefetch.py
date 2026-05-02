"""Threaded batch prefetcher for overlapping DataLoader retrieval + GPU transfer.

Moves batch retrieval from the DataLoader AND host-to-device transfer into a
background daemon thread, hiding both behind GPU forward/backward compute.
This is more aggressive than the simpler stream-based approach (NVIDIA/apex
pattern, which only overlaps H2D transfer via CUDA streams) because it also
hides the DataLoader batch retrieval latency.

Benchmarked on ROSMAP dataset (batch_size=20, 1x RTX 6000 Ada):
- No prefetching:              1254 ms/step
- Stream-based (NVIDIA/apex):  1007 ms/step  (20% faster)
- ThreadedPrefetcher (this):    916 ms/step  (27% faster, 9% over stream)

Design tradeoff: daemon threads hold GPU tensor references in their closure
and queue, which can leak ~8-10 GB per fold if shutdown() is not called.
A simpler stream-based approach avoids this lifecycle issue entirely.
However, when used with per-trial process isolation (e.g., Ray Tune), the
leak cannot accumulate across trials since the OS frees all GPU memory on
process exit. In that setting, ThreadedPrefetcher is strictly better than
stream-based: same simplicity guarantees with 9% more throughput.

Works because:
- torch.cat/torch.stack release the GIL during C++ memcpy
- CUDA kernels (forward/backward) run without the GIL
- .to(device, non_blocking=True) releases the GIL during DMA
"""

import logging
import queue
import threading

import torch

logger = logging.getLogger(__name__)


class ThreadedPrefetcher:
    """Prefetch batches in a background thread.

    Wraps any iterable (typically a DataLoader) and yields batches that
    have already been moved to ``device``.  The next batch is collated
    and transferred while the current batch is being processed on the GPU.

    Call ``shutdown()`` between folds/trials to release GPU tensor references
    held by producer threads. Without this, ~8-10 GB of CUDA memory leaks
    per fold from queued batches that were never consumed.

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
        self._abort_events: list[threading.Event] = []
        self._threads: list[threading.Thread] = []
        self._queues: list[queue.Queue] = []

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
        abort = threading.Event()

        def _producer():
            try:
                for batch in self.dataloader:
                    if abort.is_set():
                        break
                    moved = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            moved[k] = v.to(self.device, non_blocking=True)
                        else:
                            moved[k] = v
                    # Use timeout + abort check so producer can exit promptly
                    # when consumer stops iterating (early stopping, exception).
                    while not abort.is_set():
                        try:
                            q.put(moved, timeout=1)
                            break
                        except queue.Full:
                            continue
            except (RuntimeError, OSError, ValueError, KeyError, TypeError) as e:
                # Narrow except: deliberately catches data-loading failures
                # so the consumer can re-raise them. KeyboardInterrupt and
                # SystemExit propagate normally so a Ctrl-C kills the
                # process instead of becoming a silent stall.
                error_box[0] = e
            finally:
                q.put(None)  # sentinel

        # Track for shutdown()
        self._abort_events.append(abort)
        self._queues.append(q)
        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        self._threads.append(thread)

        try:
            while True:
                batch = q.get()
                if batch is None:
                    break
                if error_box[0] is not None:
                    raise error_box[0]
                yield batch
        finally:
            abort.set()
            thread.join(timeout=10)
            if thread.is_alive():
                # Thread did not honour the abort within the timeout; this
                # usually means the producer was blocked in dataloader I/O
                # (e.g., a stuck disk read). The thread becomes effectively
                # zombie until the I/O unblocks.
                logger.warning(
                    "ThreadedPrefetcher producer did not shut down within "
                    "10 s; thread is still alive (likely blocked in disk "
                    "I/O). Continuing — daemon=True ensures it does not "
                    "block process exit."
                )
            if error_box[0] is not None:
                raise error_box[0]

    def shutdown(self) -> None:
        """Stop all producer threads and drain queues to release GPU tensor refs.

        Must be called between folds/trials in HPO to prevent ~8-10 GB CUDA
        memory leak per fold from unconsumed batches in the prefetch queue.
        """
        # Signal all producers to stop
        for evt in self._abort_events:
            evt.set()

        # Join all threads (producers will exit after abort is set)
        for t in self._threads:
            t.join(timeout=10)

        # Drain all queues — this releases the GPU tensor references
        # held by the `moved` dicts that producers put on the queue
        for q in self._queues:
            while True:
                try:
                    item = q.get_nowait()
                    del item  # drop reference to GPU tensors
                except queue.Empty:
                    break

        self._abort_events.clear()
        self._threads.clear()
        self._queues.clear()

    def __len__(self):
        return len(self.dataloader)
