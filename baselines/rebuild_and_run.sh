#!/usr/bin/env bash
# Rebuilds baselines/shared/{mixmil,scphase}_input.h5ad from the current splits,
# then runs the MixMIL and scPhase baselines sequentially.
#
# Use when the source AnnData or splits have changed since the shared h5ads were
# last built, or when the vendored baseline venvs need fresh results on the current
# subject set.
#
# Stages are chained with `&&`; any failure aborts the rest. Each stage echoes a
# header + timestamp for log auditability.

set -euo pipefail
cd /host/milan/tank/Joon/proj_ml_snrna

echo "=== stage 1: regenerate baseline h5ads from splits ==="
date
uv run python baselines/prepare_data.py \
    --adata data/snRNAseq/adata_ROSMAP_preprocessed.h5ad \
    --splits outputs/splits.json \
    --metadata data/metadata_ROSMAP/metadata.csv \
    --output-dir baselines/shared/

echo "=== stage 2: MixMIL ==="
date
baselines/mixmil/.venv/bin/python baselines/mixmil/run_rosmap.py \
    --data-h5ad baselines/shared/mixmil_input.h5ad \
    --splits outputs/splits.json \
    --results-dir outputs/baselines/mixmil \
    --device cuda:0

echo "=== stage 3: scPhase ==="
date
baselines/scPhase/.venv/bin/python baselines/scPhase/run_rosmap.py \
    --data-h5ad baselines/shared/scphase_input.h5ad \
    --splits outputs/splits.json \
    --results-dir outputs/baselines/scphase \
    --device-model cuda:0 --device-encoder cuda:1

echo "=== rebuild_and_run: all stages complete ==="
date
