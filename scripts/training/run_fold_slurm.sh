#!/usr/bin/env bash
#SBATCH --job-name=cog_res_cv
#SBATCH --array=0-4               # One task per fold (0-indexed)
#SBATCH --gres=gpu:1              # 1 GPU per fold
#SBATCH --cpus-per-task=12        # Match num_workers
#SBATCH --mem=64G                 # ~41 GB preload + headroom
#SBATCH --time=04:00:00           # Per-fold wall time
#SBATCH --output=outputs/logs/slurm/fold_%a_%j.out
#SBATCH --error=outputs/logs/slurm/fold_%a_%j.err
#
# Submit: sbatch scripts/training/run_fold_slurm.sh
# Monitor: squeue -u $USER
# Logs:    outputs/logs/slurm/fold_<FOLD>_<JOBID>.{out,err}
#
# Override defaults via sbatch flags:
#   sbatch --partition=gpu-large --mem=128G scripts/training/run_fold_slurm.sh
#   sbatch --array=2-4 scripts/training/run_fold_slurm.sh   # re-run failed folds only
set -euo pipefail

# Reproducibility: PYTHONHASHSEED must be set before Python starts.
export PYTHONHASHSEED=42

FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p outputs/logs/slurm

echo "=== Fold $FOLD on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-auto} ==="
echo "    SLURM Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "    Started at: $(date)"

# Activate environment — edit to match your cluster setup.
# Common options (uncomment one):
#   module load cuda/12.1 python/3.11
#   source /path/to/venv/bin/activate
#   conda activate cogres

uv run python scripts/training/train.py \
    --config configs/default.yaml \
    --splits-path outputs/splits.json \
    --precomputed-dir data/precomputed/rosmap/ \
    --fold "$FOLD" \
    training.strategy=auto training.devices=1

echo "=== Fold $FOLD finished at $(date) ==="
