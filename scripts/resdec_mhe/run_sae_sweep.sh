#!/usr/bin/env bash
# 2-GPU-aware shard-able sweep over the SAE hyperparameter grid.
#
#  Total = 2 architectures (topk, batch_topk)
#        × 2 layers (attended, fused)
#        × 3 expansions (8, 16, 32)
#        × 5 K values (4, 8, 16, 32, 64)
#        = 60 runs (when GPU_INDEX is unset / NUM_GPUS=1; otherwise sharded).
#
# This driver is meant to be launched inside tmux (long-runs-need-tmux memory
# rule — anything >30 min wall must NOT be in a Bash run_in_background, since
# SSH disconnect / session refresh sends SIGHUP and kills the run).
#
# Skips runs whose ``reconstruction_metrics.json`` already exists, so the
# sweep can be safely resumed after interruption.
#
# 2-GPU sharding (memory rule feedback_parallelize_gpus.md):
#
#   Pass GPU_INDEX=<i> and NUM_GPUS=<N> to shard the global run order across
#   N parallel shells. Each shard processes runs whose global index satisfies
#   ``index % NUM_GPUS == GPU_INDEX``. Recommended pattern:
#
#       tmux new -s sae_sweep_g0
#       CUDA_VISIBLE_DEVICES=0 GPU_INDEX=0 NUM_GPUS=2 \
#           bash scripts/resdec_mhe/run_sae_sweep.sh
#
#       tmux new -s sae_sweep_g1
#       CUDA_VISIBLE_DEVICES=1 GPU_INDEX=1 NUM_GPUS=2 \
#           bash scripts/resdec_mhe/run_sae_sweep.sh
#
#   Each shell gets ~30 of the 60 runs (round-robin by global index).
#
# Required env vars (defaults shown):
#   ACTIVATIONS_DIR  outputs/redesign/sae
#   OUT_ROOT         outputs/redesign/sae
#   N_STEPS          50000
#   BATCH_SIZE       64
#   LEARNING_RATE    1e-4
#   AUX_LAMBDA       0.03125    (= 1/32)
#   AUX_K            256
#   SEED             0
#   CUDA_VISIBLE_DEVICES  0
#   GPU_INDEX        0      shard index in [0, NUM_GPUS)
#   NUM_GPUS         1      number of shards; set to 2 for 2-GPU parallel
#
# Usage (single-GPU; defaults match prior behaviour):
#   tmux new -s sae_sweep
#   cd /host/milan/tank/Joon/proj_ml_snrna/.worktrees/refinement-two
#   CUDA_VISIBLE_DEVICES=0 bash scripts/resdec_mhe/run_sae_sweep.sh
#
# To run fold-stratified extraction first:
#   uv run python scripts/resdec_mhe/interpretability/extract_sae_activations.py \
#       --layers attended fused

set -euo pipefail

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-outputs/redesign/sae}"
OUT_ROOT="${OUT_ROOT:-outputs/redesign/sae}"
N_STEPS="${N_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
AUX_LAMBDA="${AUX_LAMBDA:-0.03125}"
AUX_K="${AUX_K:-256}"
SEED="${SEED:-0}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
GPU_INDEX="${GPU_INDEX:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "${GPU_INDEX}" -ge "${NUM_GPUS}" ]]; then
    echo "[sweep] ERROR: GPU_INDEX (${GPU_INDEX}) must be < NUM_GPUS (${NUM_GPUS})" >&2
    exit 2
fi

ARCHITECTURES=("topk" "batch_topk")
LAYERS=("attended" "fused")
EXPANSIONS=(8 16 32)
K_VALUES=(4 8 16 32 64)
TOTAL_GRID=$((${#ARCHITECTURES[@]} * ${#LAYERS[@]} * ${#EXPANSIONS[@]} * ${#K_VALUES[@]}))

echo "[sweep] worktree: ${WORKTREE_ROOT}"
echo "[sweep] activations: ${ACTIVATIONS_DIR}"
echo "[sweep] out_root: ${OUT_ROOT}"
echo "[sweep] n_steps: ${N_STEPS}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}"
echo "[sweep] CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} GPU_INDEX=${GPU_INDEX} NUM_GPUS=${NUM_GPUS}"
echo "[sweep] grid total = ${TOTAL_GRID} runs"

cd "${WORKTREE_ROOT}"

n_total=0
n_in_shard=0
n_skipped=0
n_run=0
n_failed=0

for arch in "${ARCHITECTURES[@]}"; do
    for layer in "${LAYERS[@]}"; do
        for exp in "${EXPANSIONS[@]}"; do
            for k in "${K_VALUES[@]}"; do
                n_total=$((n_total + 1))
                # Round-robin shard: this index runs only if index % N == GPU_INDEX.
                if (( (n_total - 1) % NUM_GPUS != GPU_INDEX )); then
                    continue
                fi
                n_in_shard=$((n_in_shard + 1))
                run_dir="${OUT_ROOT}/${arch}/${layer}/exp${exp}_k${k}_seed${SEED}"
                metrics_file="${run_dir}/reconstruction_metrics.json"
                run_tag="${arch}/${layer}/exp${exp}_k${k}"

                if [[ -f "${metrics_file}" ]]; then
                    echo "[sweep][gpu${GPU_INDEX}] [${n_total}/${TOTAL_GRID}] SKIP existing: ${run_tag}"
                    n_skipped=$((n_skipped + 1))
                    continue
                fi

                # Per-run log under /tmp; tag includes architecture / layer / k / exp.
                log_hash=$(printf '%s' "${run_tag}" | tr '/' '_' | tr -d ' ')
                log_file="/tmp/sae_gpu${GPU_INDEX}_${log_hash}.log"

                echo "[sweep][gpu${GPU_INDEX}] [${n_total}/${TOTAL_GRID}] RUN ${run_tag} → ${run_dir}"
                echo "[sweep][gpu${GPU_INDEX}]    log: ${log_file}"

                set +e
                CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
                PYTHONPATH="${WORKTREE_ROOT}" \
                uv run python scripts/resdec_mhe/interpretability/run_sae_train.py \
                    --activations-dir "${ACTIVATIONS_DIR}" \
                    --layer "${layer}" \
                    --architecture "${arch}" \
                    --expansion "${exp}" \
                    --k "${k}" \
                    --seed "${SEED}" \
                    --n-steps "${N_STEPS}" \
                    --batch-size "${BATCH_SIZE}" \
                    --learning-rate "${LEARNING_RATE}" \
                    --aux-lambda "${AUX_LAMBDA}" \
                    --aux-k "${AUX_K}" \
                    --out-root "${OUT_ROOT}" \
                    > "${log_file}" 2>&1
                rc=$?
                set -e

                if [[ ${rc} -ne 0 ]]; then
                    echo "[sweep][gpu${GPU_INDEX}]    FAILED (rc=${rc}); see ${log_file}"
                    n_failed=$((n_failed + 1))
                else
                    n_run=$((n_run + 1))
                fi
            done
        done
    done
done

echo "[sweep][gpu${GPU_INDEX}] DONE: total=${n_total}, in_shard=${n_in_shard}, ran=${n_run}, skipped=${n_skipped}, failed=${n_failed}"
