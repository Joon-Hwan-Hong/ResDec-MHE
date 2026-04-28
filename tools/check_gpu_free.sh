#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Returns 0 if the target physical GPU index has at most $MAX_USED_MB used
# memory, otherwise 1. Use to gate launches on a GPU that's already busy.
#
# Usage:
#   tools/check_gpu_free.sh <gpu_index> [max_used_mb=2000]
#
# Examples:
#   tools/check_gpu_free.sh 0          # OK if GPU 0 has < 2 GB used
#   tools/check_gpu_free.sh 1 5000     # OK if GPU 1 has < 5 GB used
#
# Exit 0 = free; exit 1 = busy. Prints human summary either way.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <gpu_index> [max_used_mb=2000]" >&2
    exit 2
fi
GPU="$1"
MAX_USED_MB="${2:-2000}"

USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -d ' ')

if [[ -z "$USED_MB" ]]; then
    echo "[check_gpu_free] cannot read GPU $GPU memory; exit 2" >&2
    exit 2
fi

if (( USED_MB <= MAX_USED_MB )); then
    echo "[check_gpu_free] GPU $GPU: ${USED_MB} MB used (≤ ${MAX_USED_MB} MB) — OK"
    exit 0
else
    PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" 2>/dev/null | head -5 || true)
    echo "[check_gpu_free] GPU $GPU: ${USED_MB} MB used (> ${MAX_USED_MB} MB) — BUSY"
    echo "[check_gpu_free] processes:"
    echo "$PROCS" | sed 's/^/  /'
    exit 1
fi
