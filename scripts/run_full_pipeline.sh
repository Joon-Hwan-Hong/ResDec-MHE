#!/usr/bin/env bash
# Full automated pipeline: blocked HVG preprocessing -> HPO -> production
# training -> ablations -> baselines -> analysis.
#
# Idempotent: each stage writes a sentinel file on success. Re-running the
# script resumes from the first incomplete stage.
#
# Usage:
#   tmux new-session -d -s pipeline 'bash scripts/run_full_pipeline.sh'
#   tmux new-session -d -s pipeline 'bash scripts/run_full_pipeline.sh --start-from 2'
#   tmux new-session -d -s pipeline 'bash scripts/run_full_pipeline.sh --fresh'
#   tmux attach -t pipeline   # to watch
set -euo pipefail
cd /host/milan/tank/Joon/proj_ml_snrna

SECONDS=0

# ── Parse flags ──────────────────────────────────────────────────────────────
START_FROM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh)      rm -f "${PIPELINE_DIR:-outputs/pipeline}"/.stage_*.done 2>/dev/null; shift ;;
        --start-from) START_FROM="$2"; shift 2 ;;
        *)            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Pipeline directory ───────────────────────────────────────────────────────
PIPELINE_DIR="${PIPELINE_DIR:-outputs/pipeline}"
mkdir -p "$PIPELINE_DIR"
LOG_DIR="$PIPELINE_DIR/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"; }
done_file() { echo "$PIPELINE_DIR/.stage_${1}.done"; }
mark_done() { touch "$(done_file "$1")"; log "Stage $1 COMPLETE"; }
is_done() { [ -f "$(done_file "$1")" ]; }

# Run inference on all trained checkpoints in $PIPELINE_DIR to produce predictions.csv
run_inference_on_checkpoints() {
    local PIDS=() GPU_IDX=0 N_INF=0
    for RUN_DIR in "$PIPELINE_DIR"/20*/; do
        [ -d "$RUN_DIR/checkpoints" ] || continue
        [ -f "$RUN_DIR/analysis/predictions.csv" ] && continue  # already done

        # Find best checkpoint (lowest val_nll)
        local BEST_CKPT="" BEST_NLL="999"
        for ckpt in "$RUN_DIR"/checkpoints/epoch=*-val_nll=*.ckpt; do
            [ -f "$ckpt" ] || continue
            [[ "$ckpt" == *"last"* ]] && continue
            local nll; nll=$(echo "$ckpt" | grep -oP 'val_nll=\K[0-9.]+')
            if python3 -c "exit(0 if $nll < $BEST_NLL else 1)" 2>/dev/null; then
                BEST_NLL="$nll"
                BEST_CKPT="$ckpt"
            fi
        done
        [ -z "$BEST_CKPT" ] && continue

        local CONFIG_YAML="$RUN_DIR/config.yaml"
        [ -f "$CONFIG_YAML" ] || continue
        local FOLD_IDX; FOLD_IDX=$(grep 'fold_idx:' "$CONFIG_YAML" | head -1 | awk '{print $2}')
        [ -z "$FOLD_IDX" ] && FOLD_IDX=0

        local OUTPUT_DIR="$RUN_DIR/analysis"
        mkdir -p "$OUTPUT_DIR"

        log "    Inference: $(basename "$RUN_DIR") fold=$FOLD_IDX -> GPU $GPU_IDX"
        CUDA_VISIBLE_DEVICES=$GPU_IDX uv run python -u scripts/inference/run_inference.py \
            --checkpoint "$BEST_CKPT" \
            --config "$CONFIG_YAML" \
            --output-dir "$OUTPUT_DIR" \
            --data-path "$PRECOMPUTED" \
            --splits-path "$SPLITS" \
            --split val \
            --fold "$FOLD_IDX" \
            > "$OUTPUT_DIR/inference.log" 2>&1 &
        PIDS+=($!)
        N_INF=$((N_INF + 1))
        GPU_IDX=$(( (GPU_IDX + 1) % N_GPUS ))

        if [ "${#PIDS[@]}" -ge "$N_GPUS" ]; then
            for pid in "${PIDS[@]}"; do wait "$pid" || true; done
            PIDS=()
        fi
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || true; done
    log "    Inference complete: $N_INF runs"
}
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N_GPUS" -eq 0 ] && N_GPUS=1

# --start-from: mark all stages before the target as done
if [ -n "$START_FROM" ]; then
    ALL_STAGES=(1 1.5 2 3 4a 4 5 6 6.5 7a 7b 7c 7d 8)
    for s in "${ALL_STAGES[@]}"; do
        if [ "$s" = "$START_FROM" ]; then
            break
        fi
        touch "$(done_file "$s")"
    done
    log "Starting from stage $START_FROM (marking prior stages as done)"
fi

# ── Paths ───────────────────────────────────────────────────────────────────
ADATA_RAW="data/snRNAseq/adata_ROSMAP_merged.raw.h5ad"
ADATA_PREP="data/snRNAseq/adata_ROSMAP_preprocessed.h5ad"
PRECOMPUTED="data/precomputed"
SPLITS="outputs/splits.json"
METADATA="data/metadata_ROSMAP"
HPO_CONFIG="$PIPELINE_DIR/hpo_config.yaml"
# Warm-start HPO from a prior run (set to empty string "" to skip warm-start)
WARM_START_DIR="${WARM_START_DIR:-outputs/pipeline/ray_results/cognitive_resilience}"

log "Pipeline started — output dir: $PIPELINE_DIR"

# ─── Stage 1: Preprocess (blocked HVG) ─────────────────────────────────────
if ! is_done 1; then
    log "Stage 1: Preprocessing with seurat_v3 HVG (1M subsample)..."
    # Note: no --splits-path here because splits.json doesn't exist yet.
    # HVG selection on all subjects is standard practice (unsupervised step).
    uv run python -u -m src.data.preprocessing preprocess-only \
        --input "$ADATA_RAW" \
        --output "$ADATA_PREP" \
        --hvg-flavor seurat_v3 \
        --hvg-subsample 1000000 \
        --n-hvg 4000 \
        2>&1 | tee "$LOG_DIR/stage1_preprocess.log"
    mark_done 1
fi

# ─── Stage 1.5: Run LIANA+ CCC analysis ────────────────────────────────────
if ! is_done 1.5; then
    log "Stage 1.5: Running LIANA+ CCC analysis on preprocessed data..."
    LIANA_DIR="data/liana_cache/rosmap_blocked_hvg"
    mkdir -p "$LIANA_DIR"
    uv run python -u scripts/data/run_liana.py \
        --config configs/default.yaml \
        --adata "$ADATA_PREP" \
        --output-dir "$LIANA_DIR" \
        --n-perms 100 \
        --n-jobs 8 \
        --overwrite \
        2>&1 | tee "$LOG_DIR/stage1.5_liana.log"
    N_LIANA=$(ls "$LIANA_DIR"/*.parquet 2>/dev/null | wc -l)
    log "  LIANA completed for $N_LIANA subjects"
    mark_done 1.5
fi

# ─── Stage 2: Precompute .pt files ─────────────────────────────────────────
if ! is_done 2; then
    log "Stage 2: Precomputing per-subject .pt files..."
    LIANA_DIR="data/liana_cache/rosmap_blocked_hvg"
    uv run python -u scripts/data/precompute_features.py \
        --config configs/default.yaml \
        --output-dir "$PRECOMPUTED" \
        --adata "$ADATA_PREP" \
        --liana-dir "$LIANA_DIR" \
        --overwrite \
        2>&1 | tee "$LOG_DIR/stage2_precompute.log"
    N_PT=$(ls "$PRECOMPUTED"/*.pt 2>/dev/null | wc -l)
    log "  Created $N_PT .pt files"

    # Auto-detect n_genes from the first .pt file
    FIRST_PT=$(ls "$PRECOMPUTED"/*.pt | head -1)
    N_GENES=$(uv run python -c "
import torch; pt = torch.load('$FIRST_PT', weights_only=False)
print(pt['pseudobulk'].shape[1])
")
    log "  Auto-detected n_genes=$N_GENES"
    # Save to a pipeline-local file (don't mutate tracked configs/default.yaml)
    echo "$N_GENES" > "$PIPELINE_DIR/n_genes.txt"
    mark_done 2
fi

# ─── Stage 3: Create splits (no holdout — all subjects in CV) ─────────────
if ! is_done 3; then
    log "Stage 3: Creating splits with test_frac=0 (no holdout)..."
    uv run python -u scripts/data/create_splits.py \
        --config configs/default.yaml \
        --precomputed-dir "$PRECOMPUTED" \
        --output "$SPLITS" \
        --test-frac 0 \
        2>&1 | tee "$LOG_DIR/stage3_splits.log"
    mark_done 3
fi

# ─── Stage 4a: d_embed smoke test (3 configs × 1 fold × 3 epochs) ─────────
if ! is_done 4a; then
    log "Stage 4a: Smoke test for d_embed in {64, 128, 256}..."

    N_GENES=$(cat "$PIPELINE_DIR/n_genes.txt")
    SMOKE_DIR="$PIPELINE_DIR/smoke_d_embed"
    mkdir -p "$SMOKE_DIR"

    for DEMB in 64 128 256; do
        SMOKE_CFG="$SMOKE_DIR/config_d${DEMB}.yaml"
        log "  Generating smoke config for d_embed=$DEMB"
        uv run python -c "
from omegaconf import OmegaConf
base = OmegaConf.load('configs/default.yaml')
overlay = OmegaConf.create({
    'experiment': {'run_name': f'smoke_d_embed_${DEMB}'},
    'model': {'n_genes': $N_GENES, 'd_embed': $DEMB, 'd_fused': $DEMB},
    'data': {'precomputed_dir': '$PRECOMPUTED'},
    'paths': {'output_dir': '$SMOKE_DIR'},
    'training': {'max_epochs': 3, 'early_stopping': {'min_epochs': 1, 'patience': 3}},
})
merged = OmegaConf.merge(base, overlay)
OmegaConf.save(merged, '$SMOKE_CFG')
"
        log "  Running smoke training (d_embed=$DEMB)..."
        CUDA_VISIBLE_DEVICES=0 uv run python -u scripts/training/train.py \
            --config "$SMOKE_CFG" \
            --splits-path "$SPLITS" \
            --precomputed-dir "$PRECOMPUTED" \
            --fold 0 \
            training.devices=1 training.strategy=auto \
            > "$SMOKE_DIR/smoke_d${DEMB}.log" 2>&1
        log "  Smoke d_embed=$DEMB OK"
    done

    log "  All 3 smoke runs passed — d_embed widening is safe to HPO"
    mark_done 4a
fi

# ─── Stage 4: HPO (90 trials, 3-fold, d_embed widening) ───────────────────
if ! is_done 4; then
    log "Stage 4: HPO — 90 trials, 3-fold, 2 GPUs (d_embed widening search)..."

    # Read auto-detected n_genes
    N_GENES=$(cat "$PIPELINE_DIR/n_genes.txt")
    log "  Using n_genes=$N_GENES (from stage 2 auto-detection)"

    # Generate a merged HPO config: default.yaml + hpo search space overlay.
    # hpo.py takes a single --config, so we produce one self-contained YAML.
    uv run python -c "
from omegaconf import OmegaConf
base = OmegaConf.load('configs/default.yaml')
overlay = OmegaConf.create({
    'model': {'n_genes': $N_GENES},
    'data': {'precomputed_dir': '$PRECOMPUTED'},
    'paths': {'output_dir': '$PIPELINE_DIR'},
    'hpo': {
        'n_trials': 90,
        'search_space': {
            'lr':             {'type': 'loguniform', 'low': 5e-5,  'high': 1e-3},
            'dropout':        {'type': 'uniform',    'low': 0.05, 'high': 0.4},
            'beta':           {'type': 'uniform',    'low': 0.1,  'high': 1.0},
            'weight_decay':   {'type': 'loguniform', 'low': 1e-6, 'high': 1e-3},
            'guide_lr':       {'type': 'loguniform', 'low': 5e-4, 'high': 0.01},
            'anneal_epochs':  {'type': 'int',        'low': 8,    'high': 25},
            'd_embed':        {'type': 'categorical','choices': [64, 128, 256]},
        },
    },
})
merged = OmegaConf.merge(base, overlay)
OmegaConf.save(merged, '$HPO_CONFIG')
print(f'Wrote merged HPO config to $HPO_CONFIG')
"
    log "  Generated merged HPO config at $HPO_CONFIG"

    # Build HPO command
    HPO_CMD=(uv run python -u scripts/training/hpo.py
        --config "$HPO_CONFIG"
        --precomputed-dir "$PRECOMPUTED"
        --splits-path "$SPLITS"
        --n-trials 90
        --n-folds 3
        --n-gpus 2)
    if [ -n "$WARM_START_DIR" ] && [ -d "$WARM_START_DIR" ]; then
        HPO_CMD+=(--warm-start "$WARM_START_DIR")
        log "  Warm-starting from $WARM_START_DIR"
    else
        log "  No warm-start (fresh HPO search)"
    fi
    "${HPO_CMD[@]}" 2>&1 | tee "$LOG_DIR/stage4_hpo.log"

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

    # Run inference on production checkpoints to get predictions.csv with R²
    log "  Running inference on production checkpoints..."
    run_inference_on_checkpoints 2>&1 | tee "$LOG_DIR/stage5_inference.log"
    mark_done 5
fi

# ─── Stage 6: Ablations (all ablation configs x 5 folds) ───────────────────
if ! is_done 6; then
    log "Stage 6: Ablation study..."
    N_GENES=$(cat "$PIPELINE_DIR/n_genes.txt")

    # Update all ablation configs to match current n_genes (may differ from hardcoded value)
    for cfg in configs/ablations/ablation_*.yaml; do
        sed -i "s/^  n_genes:.*/  n_genes: $N_GENES/" "$cfg"
    done
    log "  Updated ablation configs to n_genes=$N_GENES"

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

    # Run inference on ablation checkpoints to get predictions.csv with R²
    log "  Running inference on ablation checkpoints..."
    run_inference_on_checkpoints 2>&1 | tee "$LOG_DIR/stage6_inference.log"
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
    log "Stage 7a: Classical baselines (Ridge, ElasticNet, RF, XGBoost, PLS)..."
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
        2>&1 | tee "$LOG_DIR/stage7c_gpio.log" || log "  WARNING: GPIO failed (non-fatal)"
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
