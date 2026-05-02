#!/usr/bin/env bash
# 2-GPU-aware shard-able SMALLER-M / SMALLER-K stability sweep.
#
#  Total = 2 architectures (topk, batch_topk)
#        × 2 layers (attended, fused)
#        × 3 expansions (4, 8, 16)
#        × 5 K values (4, 8, 16, 32, 64)
#        × 3 seeds (0, 1, 2)
#        = 180 runs (when GPU_INDEX is unset / NUM_GPUS=1; otherwise sharded).
#
# Differs from the canonical run_sae_sweep.sh in two ways:
#   - Adds expansion=4 (smallest reasonable for stability headroom) and drops
#     expansion=32 (canonical max) to keep the sweep small. Defaults
#     {4, 8, 16}; override by exporting EXPANSIONS="..." with a bash-array-
#     compatible space-separated string before launching.
#   - Iterates over 3 SEEDS to characterise SAE-fit stochastic stability per
#     (arch, layer, exp, K) tuple.
#
# Outputs go to ``${OUT_ROOT:-outputs/canonical/sae/stability_smaller_m}/<arch>/<layer>/exp{m}_k{K}_seed{S}/``
# instead of polluting the canonical sweep directory.
#
# This driver is meant to be launched inside tmux (long-runs-need-tmux memory
# rule).
#
# 2-GPU sharding (memory rule feedback_parallelize_gpus.md):
#
#   tmux new -s sae_smaller_m_g0
#   CUDA_VISIBLE_DEVICES=0 GPU_INDEX=0 NUM_GPUS=2 \
#       bash scripts/resdec_mhe/run_sae_sweep_smaller_m.sh
#
#   tmux new -s sae_smaller_m_g1
#   CUDA_VISIBLE_DEVICES=1 GPU_INDEX=1 NUM_GPUS=2 \
#       bash scripts/resdec_mhe/run_sae_sweep_smaller_m.sh
#
# Required env vars (defaults shown):
#   ACTIVATIONS_DIR  outputs/canonical/sae
#   OUT_ROOT         outputs/canonical/sae/stability_smaller_m
#   EXPANSIONS       "4 8 16"
#   K_VALUES         "4 8 16 32 64"
#   SEEDS            "0 1 2"
#   ARCHITECTURES    "topk batch_topk"
#   LAYERS           "attended fused"
#   N_STEPS          50000
#   BATCH_SIZE       64
#   LEARNING_RATE    1e-4
#   AUX_LAMBDA       0.03125    (= 1/32)
#   AUX_K            256
#   CUDA_VISIBLE_DEVICES  0
#   GPU_INDEX        0      shard index in [0, NUM_GPUS)
#   NUM_GPUS         1      number of shards; set to 2 for 2-GPU parallel
#
# Usage (single-GPU; default smaller-m grid):
#   tmux new -s sae_smaller_m
#   cd /host/milan/tank/Joon/proj_ml_snrna/.worktrees/refinement-two
#   CUDA_VISIBLE_DEVICES=0 bash scripts/resdec_mhe/run_sae_sweep_smaller_m.sh

# Top-level shell uses ``set -euo pipefail`` (strict). The per-config
# uv-run-python invocation below is wrapped in ``set +e`` / ``set -e``
# (lines 149/168) so a SINGLE training failure does not abort the entire
# 180-config sweep — surrounding mkdir / printf / tr ops are checked
# strictly, while per-run failures are counted in n_failed and the sweep
# continues. (B-SM2: clarify that top-level is ``-e``, not ``-uo``.)
set -euo pipefail

# tmux preflight (feedback_long_runs_need_tmux.md): 180 configs × ~2-5 min
# each = 6-15 hr wall, an order of magnitude past the 30-min threshold.
if [ -z "${TMUX:-}" ]; then
    echo "ERROR: This sweep runs ~6-15 hr and must be in tmux to survive SSH disconnect." >&2
    echo "  tmux new -s sae_smaller_m 'bash $0'" >&2
    exit 1
fi

WORKTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Both ACTIVATIONS_DIR and OUT_ROOT are resolved relative to WORKTREE_ROOT
# (the cd on line 102 below is what makes that work). To pass an absolute
# path, override the env var with one starting at /.
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-outputs/canonical/sae}"
OUT_ROOT="${OUT_ROOT:-outputs/canonical/sae/stability_smaller_m}"
N_STEPS="${N_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
AUX_LAMBDA="${AUX_LAMBDA:-0.03125}"
AUX_K="${AUX_K:-256}"

# F11 parity: resolve metadata.csv ONCE and forward via --metadata-path so
# every per-config child skips its OmegaConf.merge.
METADATA_PATH="${METADATA_PATH:-${WORKTREE_ROOT}/data/metadata_ROSMAP/metadata.csv}"

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
GPU_INDEX="${GPU_INDEX:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

if [[ "${GPU_INDEX}" -ge "${NUM_GPUS}" ]]; then
    echo "[smaller-m] ERROR: GPU_INDEX (${GPU_INDEX}) must be < NUM_GPUS (${NUM_GPUS})" >&2
    exit 2
fi

# Bash arrays from space-separated env strings; allow override of any axis.
read -r -a ARCHITECTURES <<< "${ARCHITECTURES:-topk batch_topk}"
read -r -a LAYERS        <<< "${LAYERS:-attended fused}"
read -r -a EXPANSIONS    <<< "${EXPANSIONS:-4 8 16}"
read -r -a K_VALUES      <<< "${K_VALUES:-4 8 16 32 64}"
read -r -a SEEDS         <<< "${SEEDS:-0 1 2}"

TOTAL_GRID=$((${#ARCHITECTURES[@]} * ${#LAYERS[@]} * ${#EXPANSIONS[@]} * ${#K_VALUES[@]} * ${#SEEDS[@]}))

echo "[smaller-m] worktree: ${WORKTREE_ROOT}"
echo "[smaller-m] activations: ${ACTIVATIONS_DIR}"
echo "[smaller-m] out_root: ${OUT_ROOT}"
echo "[smaller-m] architectures: ${ARCHITECTURES[*]}"
echo "[smaller-m] layers: ${LAYERS[*]}"
echo "[smaller-m] expansions: ${EXPANSIONS[*]}"
echo "[smaller-m] K values: ${K_VALUES[*]}"
echo "[smaller-m] seeds: ${SEEDS[*]}"
echo "[smaller-m] CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} GPU_INDEX=${GPU_INDEX} NUM_GPUS=${NUM_GPUS}"
echo "[smaller-m] grid total = ${TOTAL_GRID} runs"

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
                for seed in "${SEEDS[@]}"; do
                    n_total=$((n_total + 1))
                    # Round-robin shard.
                    if (( (n_total - 1) % NUM_GPUS != GPU_INDEX )); then
                        continue
                    fi
                    n_in_shard=$((n_in_shard + 1))
                    run_dir="${OUT_ROOT}/${arch}/${layer}/exp${exp}_k${k}_seed${seed}"
                    metrics_file="${run_dir}/reconstruction_metrics.json"
                    run_tag="${arch}/${layer}/exp${exp}_k${k}_seed${seed}"

                    if [[ -f "${metrics_file}" ]]; then
                        echo "[smaller-m][gpu${GPU_INDEX}] [${n_total}/${TOTAL_GRID}] SKIP existing: ${run_tag}"
                        n_skipped=$((n_skipped + 1))
                        continue
                    fi

                    log_hash=$(printf '%s' "${run_tag}" | tr '/' '_' | tr -d ' ')
                    # Colocate logs with sweep outputs (avoids /tmp collisions
                    # across worktrees and survives reboot).
                    SWEEP_LOG_DIR="${OUT_ROOT}/_sae_sweep_logs"
                    mkdir -p "${SWEEP_LOG_DIR}"
                    log_file="${SWEEP_LOG_DIR}/sae_smaller_m_gpu${GPU_INDEX}_${log_hash}.log"

                    echo "[smaller-m][gpu${GPU_INDEX}] [${n_total}/${TOTAL_GRID}] RUN ${run_tag} → ${run_dir}"
                    echo "[smaller-m][gpu${GPU_INDEX}]    log: ${log_file}"

                    set +e
                    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
                    PYTHONPATH="${WORKTREE_ROOT}" \
                    uv run python scripts/resdec_mhe/interpretability/run_sae_train.py \
                        --activations-dir "${ACTIVATIONS_DIR}" \
                        --metadata-path "${METADATA_PATH}" \
                        --layer "${layer}" \
                        --architecture "${arch}" \
                        --expansion "${exp}" \
                        --k "${k}" \
                        --seed "${seed}" \
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
                        echo "[smaller-m][gpu${GPU_INDEX}]    FAILED (rc=${rc}); see ${log_file}"
                        n_failed=$((n_failed + 1))
                    else
                        n_run=$((n_run + 1))
                    fi
                done
            done
        done
    done
done

echo "[smaller-m][gpu${GPU_INDEX}] DONE: total=${n_total}, in_shard=${n_in_shard}, ran=${n_run}, skipped=${n_skipped}, failed=${n_failed}"
