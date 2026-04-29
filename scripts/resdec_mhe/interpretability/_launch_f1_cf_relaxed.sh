#!/usr/bin/env bash
# F1 vulnerable CF deep-dive — relaxed search
# Tol 1e-2 (10x), lambda_max 1e4 (10x), max_steps 3000 (3x) vs original
set -euo pipefail
cd /host/milan/tank/Joon/proj_ml_snrna/.worktrees/refinement-two
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
echo "EXIT_STATUS=$?"
