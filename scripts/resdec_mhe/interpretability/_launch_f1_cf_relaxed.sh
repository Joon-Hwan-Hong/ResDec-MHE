#!/usr/bin/env bash
# F1 vulnerable CF deep-dive — relaxed search.
# Tol 1e-2 (10x), lambda_max 1e4 (10x), max_steps 3000 (3x) vs original.
#
# Worktree root is derived from BASH_SOURCE so this script is portable
# across worktrees / cloned trees (mirrors _launch_f1_optimized_grid.sh).
# EXIT_STATUS captures the python exit code via PIPESTATUS[0]; tee's
# rc would always be 0 even when python crashes.
set -euo pipefail

# Tmux preflight: max_steps 3000 (3x canonical) means this run is likely
# > 30 min wall, which a SIGHUP from a closed SSH session would kill.
# Force tmux usage; override with FORCE_NO_TMUX=1 for short smoke tests.
if [ -z "${TMUX:-}" ] && [ -z "${FORCE_NO_TMUX:-}" ]; then
    echo "ERROR: This launcher may run > 30 min and must be in tmux." >&2
    echo "       Run inside tmux, or set FORCE_NO_TMUX=1 to override." >&2
    exit 1
fi

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$WORKTREE_ROOT"
export CUDA_VISIBLE_DEVICES=0
OUT_DIR=outputs/canonical/interpretability/counterfactuals_relative_relaxed
mkdir -p "${OUT_DIR}"
uv run python scripts/resdec_mhe/interpretability/run_counterfactuals.py \
  --target-mode relative \
  --tol 1e-2 \
  --lambda-max 1e4 \
  --max-steps 3000 \
  --out-dir "${OUT_DIR}" \
  --device cuda:0 \
  2>&1 | tee "${OUT_DIR}/run.log"
echo "EXIT_STATUS=${PIPESTATUS[0]}"
