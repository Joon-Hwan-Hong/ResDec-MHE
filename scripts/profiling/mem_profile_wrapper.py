"""Monitor system + per-process memory during a profiling run.

Tracks:
- System free memory (ground truth for OOM risk)
- Per-process USS (unique, non-shared memory — true cost of each process)
- Per-process RSS (includes shared pages — inflated for forked processes)
- /dev/shm usage (PyTorch DataLoader shared memory tensors)
- Total USS across all children (true memory footprint)
"""

import os
import shutil
import subprocess
import sys
import time

import psutil


def get_descendant_pids(parent_pid):
    """Get all descendant PIDs of a process."""
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        return [parent] + children
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def get_shm_usage():
    """Get /dev/shm usage in GB."""
    usage = shutil.disk_usage("/dev/shm")
    return usage.used / 1e9


def monitor_loop(target_pid, interval=2.0, outfile=None):
    """Monitor memory until target process exits."""
    out = open(outfile, "w") if outfile else sys.stdout

    header = (
        f"{'Time':>10} {'FreeMem':>8} {'ShmUsed':>8} "
        f"{'NProc':>5} {'TotalUSS':>10} {'TotalRSS':>10} "
        f"{'MaxProcUSS':>11} {'MaxProcRSS':>11} {'MaxProcPID':>10}"
    )
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    out.flush()

    t0 = time.monotonic()
    while True:
        try:
            parent = psutil.Process(target_pid)
            if not parent.is_running():
                break
        except psutil.NoSuchProcess:
            break

        procs = get_descendant_pids(target_pid)
        elapsed = time.monotonic() - t0

        total_uss = 0
        total_rss = 0
        max_uss = 0
        max_rss = 0
        max_uss_pid = 0
        max_rss_pid = 0
        n_procs = 0

        for p in procs:
            try:
                mi = p.memory_full_info()
                uss = mi.uss
                rss = mi.rss
                total_uss += uss
                total_rss += rss
                n_procs += 1
                if uss > max_uss:
                    max_uss = uss
                    max_uss_pid = p.pid
                if rss > max_rss:
                    max_rss = rss
                    max_rss_pid = p.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        vm = psutil.virtual_memory()
        shm = get_shm_usage()

        line = (
            f"{elapsed:>9.1f}s {vm.available/1e9:>7.1f}G {shm:>7.2f}G "
            f"{n_procs:>5} {total_uss/1e9:>9.1f}G {total_rss/1e9:>9.1f}G "
            f"{max_uss/1e9:>10.2f}G {max_rss/1e9:>10.2f}G {max_uss_pid:>10}"
        )
        out.write(line + "\n")
        out.flush()

        time.sleep(interval)

    out.write("\n--- Process exited ---\n")
    if outfile:
        out.close()


if __name__ == "__main__":
    target_pid = int(sys.argv[1])
    outfile = sys.argv[2] if len(sys.argv) > 2 else None
    monitor_loop(target_pid, interval=2.0, outfile=outfile)
