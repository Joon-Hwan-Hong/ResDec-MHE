#!/usr/bin/env bash
# Phase 6: Variant A distributional analyses on raw pseudobulk vs residualized
# cognition quartiles. Wasserstein-1 per (cell type, gene) + raw-pseudobulk
# CMI per cell type. ~1 hr wall (mostly CPU). Reuses canonical
# run_distributional_resilience.py + conditional_mi.
set -uo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "ERROR: must be in tmux." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VARIANT_NAME="${VARIANT_NAME:-gpath_only}"

INTERP_OUT="$ROOT/outputs/canonical/cogn_residual/${VARIANT_NAME}/interpretability"
RESIDUAL_CSV="$INTERP_OUT/variant_residual_per_subject.csv"
DIST_OUT="$INTERP_OUT/distributional_resilience"
SENTINEL_DIR="$ROOT/outputs/canonical/cogn_residual/_chain_sentinels"
LOG="$INTERP_OUT/phase6_distributional.log"
mkdir -p "$DIST_OUT" "$SENTINEL_DIR"

export PYTHONPATH="$ROOT"
cd "$ROOT"

echo "[$(date -Iseconds)] === Phase 6: Variant ${VARIANT_NAME} distributional ===" | tee -a "$LOG"

if [ ! -f "$RESIDUAL_CSV" ]; then
    RESIDUAL_CACHE_DIR="$ROOT/outputs/canonical/cogn_residual/${VARIANT_NAME}/cache"
    echo "[$(date -Iseconds)] building $RESIDUAL_CSV" | tee -a "$LOG"
    if ! uv run python scripts/resdec_mhe/cogn_residual/build_variant_residual_csv.py \
        --residual-cache-dir "$RESIDUAL_CACHE_DIR" \
        --splits-path "$ROOT/outputs/splits.json" \
        --out-csv "$RESIDUAL_CSV" \
        >> "$LOG" 2>&1; then
        rc=$?
        echo "[$(date -Iseconds)] === Phase 6 residual-CSV build FAILED rc=$rc ===" | tee -a "$LOG"
        touch "$SENTINEL_DIR/PHASE6_FAIL_BUILD_${rc}"
        exit "$rc"
    fi
fi

CT_NAMES_SOURCE="$INTERP_OUT/captum_ig/composite_attribution_summary.json"
VARIANT_CANONICAL_DIR="$ROOT/outputs/canonical/cogn_residual/${VARIANT_NAME}/p5_seed42"
VARIANT_CAPTUM_NPZ="$INTERP_OUT/captum_ig/composite_attributions.npz"
N_FAILED=0

# 1. Wasserstein-1 per (CT, gene) between resilient and vulnerable quartiles
echo "[$(date -Iseconds)] Wasserstein-1 per (CT, gene)" | tee -a "$LOG"
if uv run python scripts/resdec_mhe/interpretability/run_distributional_resilience.py wasserstein \
    --residual-csv "$RESIDUAL_CSV" \
    --precomputed-dir "$ROOT/data/precomputed" \
    --gene-names-npy "$ROOT/data/precomputed/gene_names.npy" \
    --cell-type-names-source "$CT_NAMES_SOURCE" \
    --out-dir "$DIST_OUT" \
    >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] Wasserstein done" | tee -a "$LOG"
else
    rc=$?
    echo "[$(date -Iseconds)] Wasserstein FAILED rc=$rc" | tee -a "$LOG"
    N_FAILED=$((N_FAILED + 1))
fi

# 2. Raw-pseudobulk conditional MI per cell type (max + vector aggregation)
# `run_resilience_analyses.py cmi --source raw` writes
# conditional_mi_per_celltype_raw_<aggregation>.json to --out-dir.
# IMPORTANT: --canonical-dir + --captum-npz must point at the variant model,
# NOT the canonical default; cmd_cmi pulls Y from <canonical-dir>/foldX/
# val_predictions_best.npz and the subject ordering from <captum-npz>.
for AGG in max vector; do
    echo "[$(date -Iseconds)] CMI raw-pseudobulk (agg=${AGG})" | tee -a "$LOG"
    if uv run python scripts/resdec_mhe/interpretability/run_resilience_analyses.py \
        --canonical-dir "$VARIANT_CANONICAL_DIR" \
        --captum-npz "$VARIANT_CAPTUM_NPZ" \
        --residual-csv "$RESIDUAL_CSV" \
        --metadata-csv "$ROOT/data/metadata_ROSMAP/metadata.csv" \
        --out-dir "$INTERP_OUT" \
        cmi --source raw --aggregation "$AGG" --n-jobs 4 \
        --precomputed-dir "$ROOT/data/precomputed" \
        >> "$LOG" 2>&1; then
        echo "[$(date -Iseconds)] CMI ${AGG} done" | tee -a "$LOG"
    else
        rc=$?
        echo "[$(date -Iseconds)] CMI ${AGG} FAILED rc=$rc" | tee -a "$LOG"
        N_FAILED=$((N_FAILED + 1))
    fi
done

if [ "$N_FAILED" -eq 0 ]; then
    echo "[$(date -Iseconds)] === Phase 6 done ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE6_DONE"
else
    echo "[$(date -Iseconds)] === Phase 6 FAILED (${N_FAILED} subtasks) ===" | tee -a "$LOG"
    touch "$SENTINEL_DIR/PHASE6_FAIL_${N_FAILED}"
    exit 1
fi
