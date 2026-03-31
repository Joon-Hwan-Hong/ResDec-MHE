#!/usr/bin/env bash
# Full automated pipeline: blocked HVG preprocessing -> HPO -> production
# training -> ablations -> baselines -> analysis.
#
# Idempotent: each stage writes a sentinel file on success. Re-running the
# script resumes from the first incomplete stage.
#
# Usage:
#   tmux new-session -d -s pipeline 'bash scripts/run_full_pipeline.sh'
#   tmux attach -t pipeline   # to watch
set -euo pipefail
cd /host/milan/tank/Joon/proj_ml_snrna

SECONDS=0

PIPELINE_DIR="outputs/pipeline_$(date +%Y%m%d)"
LOG_DIR="$PIPELINE_DIR/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"; }
done_file() { echo "$PIPELINE_DIR/.stage_${1}.done"; }
mark_done() { touch "$(done_file "$1")"; log "Stage $1 COMPLETE"; }
is_done() { [ -f "$(done_file "$1")" ]; }

# ── Paths ───────────────────────────────────────────────────────────────────
ADATA_RAW="data/snRNAseq/adata_ROSMAP_merged.h5ad"
ADATA_PREP="data/snRNAseq/adata_ROSMAP_preprocessed.h5ad"
PRECOMPUTED="data/precomputed"
SPLITS="outputs/splits.json"
METADATA="data/metadata_ROSMAP"
HPO_CONFIG="$PIPELINE_DIR/hpo_config.yaml"
WARM_START_DIR="outputs/ray_results/cognitive_resilience_hpo7"

log "Pipeline started — output dir: $PIPELINE_DIR"

# ─── Stage 1: Preprocess (blocked HVG) ─────────────────────────────────────
if ! is_done 1; then
    log "Stage 1: Preprocessing with blocked HVG..."
    uv run python -u -m src.data.preprocessing preprocess-only \
        --input "$ADATA_RAW" \
        --output "$ADATA_PREP" \
        --hvg-flavor blocked \
        --hvg-per-type-n 5000 \
        --n-hvg 4000 \
        --splits-path "$SPLITS" \
        2>&1 | tee "$LOG_DIR/stage1_preprocess.log"
    mark_done 1
fi

# ─── Stage 2: Precompute .pt files ─────────────────────────────────────────
if ! is_done 2; then
    log "Stage 2: Precomputing per-subject .pt files..."
    # Clear old .pt files so gene-index mismatch cannot happen
    rm -rf "$PRECOMPUTED"/*.pt
    uv run python -u scripts/data/precompute_features.py \
        --config configs/default.yaml \
        --output-dir "$PRECOMPUTED" \
        --adata "$ADATA_PREP" \
        --liana-dir data/liana_cache/ \
        2>&1 | tee "$LOG_DIR/stage2_precompute.log"
    N_PT=$(ls "$PRECOMPUTED"/*.pt 2>/dev/null | wc -l)
    log "  Created $N_PT .pt files"
    mark_done 2
fi

# ─── Stage 3: Create splits (with 10% holdout) ─────────────────────────────
if ! is_done 3; then
    log "Stage 3: Creating splits with test_frac=0.1..."
    uv run python -u scripts/data/create_splits.py \
        --config configs/default.yaml \
        --precomputed-dir "$PRECOMPUTED" \
        --output "$SPLITS" \
        --test-frac 0.1 \
        2>&1 | tee "$LOG_DIR/stage3_splits.log"
    mark_done 3
fi

# ─── Stage 4: HPO (50 trials, 3-fold) ──────────────────────────────────────
if ! is_done 4; then
    log "Stage 4: HPO — 50 trials, 3-fold, 2 GPUs..."

    # Generate a merged HPO config: default.yaml + hpo search space overlay.
    # hpo.py takes a single --config, so we produce one self-contained YAML.
    uv run python -c "
from omegaconf import OmegaConf
base = OmegaConf.load('configs/default.yaml')
overlay = OmegaConf.create({
    'hpo': {
        'n_trials': 50,
        'per_trial_timeout': 7200,
        'search_space': {
            'lr':             {'type': 'loguniform', 'low': 5e-5,  'high': 5e-3},
            'd_embed':        {'type': 'categorical', 'choices': [64, 128]},
            'dropout':        {'type': 'uniform',    'low': 0.05, 'high': 0.4},
            'beta':           {'type': 'uniform',    'low': 0.1,  'high': 1.0},
            'weight_decay':   {'type': 'loguniform', 'low': 1e-6, 'high': 1e-3},
            'guide_lr':       {'type': 'loguniform', 'low': 5e-4, 'high': 0.05},
            'anneal_epochs':  {'type': 'int',        'low': 8,    'high': 30},
            'n_hgt_layers':   {'type': 'int',        'low': 2,    'high': 3},
            'n_inducing':     {'type': 'categorical', 'choices': [32, 64]},
            'gene_gate_temp': {'type': 'uniform',    'low': 0.3,  'high': 2.0},
        },
    },
})
merged = OmegaConf.merge(base, overlay)
OmegaConf.save(merged, '$HPO_CONFIG')
print(f'Wrote merged HPO config to $HPO_CONFIG')
"
    log "  Generated merged HPO config at $HPO_CONFIG"

    # Warm-start from HPO7 best config; output ray_results into pipeline dir.
    uv run python -u scripts/training/hpo.py \
        --config "$HPO_CONFIG" \
        --precomputed-dir "$PRECOMPUTED" \
        --splits-path "$SPLITS" \
        --n-trials 50 \
        --n-folds 3 \
        --n-gpus 2 \
        --warm-start "$WARM_START_DIR" \
        paths.output_dir="$PIPELINE_DIR" \
        2>&1 | tee "$LOG_DIR/stage4_hpo.log"

    # Agent analysis of HPO results
    claude -p "Read $LOG_DIR/stage4_hpo.log and $PIPELINE_DIR/ray_results/. Extract top 3 configs. Write analysis to $PIPELINE_DIR/hpo_analysis.md. Extract best config to $PIPELINE_DIR/best_config.yaml." \
        --model opus --effort max --output-format text \
        > "$LOG_DIR/stage4_agent.log" 2>&1 || true
    mark_done 4
fi

# ─── Stage 5: Production 5-fold (best HPO config) ──────────────────────────
if ! is_done 5; then
    log "Stage 5: Production 5-fold training..."

    # HPO auto-saves best config to paths.output_dir/best_config.yaml.
    # The claude agent in Stage 4 may also write/refine it.
    # run_sensitivity.sh takes config YAML(s) as positional args.
    BEST_CFG="$PIPELINE_DIR/best_config.yaml"
    if [ ! -f "$BEST_CFG" ]; then
        log "ERROR: best_config.yaml not found in $PIPELINE_DIR — cannot run production training"
        exit 1
    fi

    bash scripts/training/run_sensitivity.sh \
        --splits "$SPLITS" \
        --precomputed "$PRECOMPUTED" \
        --logdir "$LOG_DIR/production" \
        "$BEST_CFG" \
        2>&1 | tee "$LOG_DIR/stage5_production.log"

    claude -p "Read production results in $LOG_DIR/production/. Compute mean R2, Pearson r, Spearman rho across folds. Write to $PIPELINE_DIR/production_results.md." \
        --model opus --effort max --output-format text \
        > "$LOG_DIR/stage5_agent.log" 2>&1 || true
    mark_done 5
fi

# ─── Stage 6: Ablations (all ablation configs x 5 folds) ───────────────────
if ! is_done 6; then
    log "Stage 6: Ablation study..."
    for cfg in configs/ablations/ablation_*.yaml; do
        NAME=$(basename "$cfg" .yaml)
        log "  Running $NAME..."
        bash scripts/training/run_sensitivity.sh \
            --splits "$SPLITS" \
            --precomputed "$PRECOMPUTED" \
            --logdir "$LOG_DIR/ablations" \
            "$cfg" \
            2>&1 | tee -a "$LOG_DIR/stage6_ablations.log"
    done

    claude -p "Read ablation results in $LOG_DIR/ablations/. Compare all ablation configs vs full model. Write comprehensive ablation analysis to $PIPELINE_DIR/ablation_results.md." \
        --model opus --effort max --output-format text \
        > "$LOG_DIR/stage6_agent.log" 2>&1 || true
    mark_done 6
fi

# ─── Stage 6.5: Baseline data prep ─────────────────────────────────────────
if ! is_done 6.5; then
    log "Stage 6.5: Preparing baseline input data..."

    # scPhase h5ad (CPU, no scVI)
    uv run python -u baselines/prepare_data.py \
        --adata "$ADATA_PREP" \
        --splits "$SPLITS" \
        --metadata "$METADATA/metadata.csv" \
        --output-dir baselines/shared/ \
        --methods scphase \
        2>&1 | tee "$LOG_DIR/stage6.5_scphase_data.log"

    # MixMIL h5ad (scVI training on GPU)
    CUDA_VISIBLE_DEVICES=0 uv run python -u baselines/prepare_data.py \
        --adata "$ADATA_PREP" \
        --splits "$SPLITS" \
        --metadata "$METADATA/metadata.csv" \
        --output-dir baselines/shared/ \
        --methods mixmil \
        --scvi-epochs 50 --scvi-latent 30 \
        2>&1 | tee "$LOG_DIR/stage6.5_mixmil_data.log"
    mark_done 6.5
fi

# ─── Stage 7a: Classical baselines ──────────────────────────────────────────
if ! is_done 7a; then
    log "Stage 7a: Classical baselines (Ridge, ElasticNet, SVR, RF, XGBoost, PLS)..."
    uv run python -u scripts/analysis/run_baselines.py \
        --precomputed-dir "$PRECOMPUTED" \
        --splits-path "$SPLITS" \
        --metadata-path "$METADATA" \
        --output "$PIPELINE_DIR/baseline_results_classical.csv" \
        --cv-tune \
        2>&1 | tee "$LOG_DIR/stage7a_classical.log"
    mark_done 7a
fi

# ─── Stage 7b: CloudPred baselines ─────────────────────────────────────────
if ! is_done 7b; then
    log "Stage 7b: CloudPred + CloudPred per-type..."
    baselines/cloudpred/.venv/bin/python -u baselines/cloudpred/run_rosmap.py \
        --data-dir "$PRECOMPUTED" --splits "$SPLITS" --metadata-dir "$METADATA" \
        --results-dir "$PIPELINE_DIR/baselines/cloudpred" --device cuda:1 \
        2>&1 | tee "$LOG_DIR/stage7b_cloudpred.log"

    baselines/cloudpred/.venv/bin/python -u baselines/cloudpred/run_rosmap_pertype.py \
        --data-dir "$PRECOMPUTED" --splits "$SPLITS" --metadata-dir "$METADATA" \
        --results-dir "$PIPELINE_DIR/baselines/cloudpred_pertype" --device cuda:1 \
        --k-per-type 3 \
        2>&1 | tee "$LOG_DIR/stage7b_cloudpred_pertype.log"
    mark_done 7b
fi

# ─── Stage 7c: Perceiver IO + GPIO ─────────────────────────────────────────
if ! is_done 7c; then
    log "Stage 7c: Perceiver IO + GPIO..."
    baselines/perceiver_io/.venv/bin/python -u baselines/perceiver_io/run_rosmap.py \
        --data-dir "$PRECOMPUTED" --splits "$SPLITS" --metadata-dir "$METADATA" \
        --results-dir "$PIPELINE_DIR/baselines/perceiver_io" --device cuda:1 \
        2>&1 | tee "$LOG_DIR/stage7c_perceiver_io.log"

    baselines/gpio/.venv/bin/python -u baselines/gpio/run_rosmap.py \
        --data-dir "$PRECOMPUTED" --splits "$SPLITS" --metadata-dir "$METADATA" \
        --results-dir "$PIPELINE_DIR/baselines/gpio" --device cuda:1 \
        2>&1 | tee "$LOG_DIR/stage7c_gpio.log"
    mark_done 7c
fi

# ─── Stage 7d: MIL baselines ───────────────────────────────────────────────
if ! is_done 7d; then
    log "Stage 7d: MIL baselines (MixMIL, ABMIL, SetTransformer, scPhase)..."

    baselines/mixmil/.venv/bin/python -u baselines/mixmil/run_rosmap.py \
        --data-h5ad baselines/shared/mixmil_input.h5ad \
        --splits "$SPLITS" \
        --results-dir "$PIPELINE_DIR/baselines/mixmil" \
        2>&1 | tee "$LOG_DIR/stage7d_mixmil.log"

    baselines/abmil/.venv/bin/python -u baselines/abmil/run_rosmap.py \
        --data-h5ad baselines/shared/mixmil_input.h5ad \
        --splits "$SPLITS" \
        --results-dir "$PIPELINE_DIR/baselines/abmil" \
        --device cuda:1 \
        2>&1 | tee "$LOG_DIR/stage7d_abmil.log"

    baselines/set_transformer/.venv/bin/python -u baselines/set_transformer/run_rosmap.py \
        --data-h5ad baselines/shared/mixmil_input.h5ad \
        --splits "$SPLITS" \
        --results-dir "$PIPELINE_DIR/baselines/set_transformer" \
        --device cuda:1 \
        2>&1 | tee "$LOG_DIR/stage7d_set_transformer.log"

    baselines/scPhase/.venv/bin/python -u baselines/scPhase/run_rosmap.py \
        --data-h5ad baselines/shared/scphase_input.h5ad \
        --splits "$SPLITS" \
        --results-dir "$PIPELINE_DIR/baselines/scphase" \
        --device-model cuda:0 \
        --device-encoder cuda:1 \
        2>&1 | tee "$LOG_DIR/stage7d_scphase.log"
    mark_done 7d
fi

# ─── Stage 8: Final analysis ───────────────────────────────────────────────
if ! is_done 8; then
    log "Stage 8: Final cross-benchmark analysis..."
    claude -p "You are analyzing the complete results of a cognitive resilience prediction project after a blocked HVG preprocessing change.

Working directory: /host/milan/tank/Joon/proj_ml_snrna
Pipeline dir: $PIPELINE_DIR

TASKS:
1. Read all results: production ($LOG_DIR/production/), ablations ($LOG_DIR/ablations/), classical baselines ($PIPELINE_DIR/baseline_results_classical.csv), DL baselines ($PIPELINE_DIR/baselines/).
2. Create comprehensive docs/results/$(date +%Y-%m-%d)-blocked-hvg-results.md with:
   - Per-fold tables for every method
   - Ranking table sorted by R2
   - Comparison vs archive (old random-subsample HVG results in docs/results/)
   - Interpretation: did blocked HVG help? Which methods benefited most?
3. Write $PIPELINE_DIR/final_summary.txt (concise).

Be thorough." \
        --model opus --effort max --output-format text \
        > "$LOG_DIR/stage8_final_agent.log" 2>&1 || true
    mark_done 8
fi

ELAPSED=$SECONDS
log "Pipeline complete. Total elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s ($ELAPSED seconds)"
