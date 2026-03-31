# Pipeline Flow

## Data Preprocessing Pipeline

```
0a. Merge h5ads (src/data/preprocessing.py merge-and-preprocess)
    ├─ DLPFC h5ad + multiregion h5ad → adata_ROSMAP_merged.h5ad (raw counts, ~20K genes)
    └─ Only run once (69 GB output)

0b. Preprocess (src/data/preprocessing.py preprocess-only)
    ├─ HVG selection: "blocked" (default) — scran-style per-cell-type variance
    │   ├─ Stratified subsample: 5K cells per type (equal representation)
    │   ├─ Indicator matrix algebra: E[X²]-E[X]² (no for-loops)
    │   ├─ Loess normalization (remove mean-variance trend)
    │   └─ Top 4000 genes + CellChatDB L-R genes → ~4100-4800 final genes
    ├─ Normalize (target_sum=1e4) + log1p
    ├─ Store raw counts in .raw
    └─ Output: adata_ROSMAP_preprocessed.h5ad

0c. Precompute per-subject .pt files (scripts/data/precompute_features.py)
    ├─ Input: preprocessed h5ad + LIANA results
    └─ Output: data/precomputed/{subject_id}.pt (516 files)

0d. Create splits (scripts/data/create_splits.py)
    ├─ Stratified K-fold with optional holdout (test_frac=0.1)
    └─ Output: outputs/splits.json
```

## Production Pipeline

```
1. HPO (scripts/training/hpo.py)
   └─> Ray Tune results in outputs/ray_results/

2. Extract HPO results (scripts/inference/extract_hpo_results.py)
   └─> Top-K configs in outputs/hpo_analysis/{date}_{name}/top_configs/

3. 5-fold production training (scripts/training/run_sensitivity.sh → scripts/training/train.py)
   └─> Checkpoints in outputs/{timestamp}_{name}_{hash}/checkpoints/
   └─> Each run: config.yaml, checkpoints/, logs/, model/, figures/

4. Inference + extraction (scripts/inference/run_inference.py --extract-all)
   Input: checkpoint .ckpt + data/precomputed/*.pt
   Output: {experiment}/analysis/
     ├── predictions.parquet    # subject_id, predicted_mean, predicted_std, actual, residual, pathology
     ├── predictions.csv        # same as parquet, human-readable
     └── attention_weights.h5   # HGT attention, PMA attention, gene gate, embeddings, etc.

5. Post-hoc analysis (scripts/analysis/run_analysis.py)
   Input: predictions.parquet + attention_weights.h5
   Output: {experiment}/analysis/
     ├── cell_type_importance/
     ├── gene_importance/
     ├── ccc_importance/
     ├── resilience_signatures/
     ├── uncertainty/
     ├── embedding_analysis/
     └── gene_enrichment/ (if enabled)
```

## Baseline Pipeline

### Classical ML (GPU-accelerated)
```
scripts/analysis/run_baselines.py --cv-tune
  Input: data/precomputed/*.pt + outputs/splits.json + data/metadata_ROSMAP/
  Output: outputs/baseline_results.csv
  Models: Ridge, ElasticNet, SVR, RandomForest, XGBoost, PLS
  Feature sets: C (cell-type proportions), A (pseudobulk), A+C+E (all combined)
```

### DL Baselines (each has own venv)
```
baselines/{model}/.venv/bin/python baselines/{model}/run_rosmap.py
  Input: data/precomputed/*.pt + outputs/splits.json + data/metadata_ROSMAP/
  Output: outputs/baselines/{model}/results.csv + predictions/fold{0-4}.npz

  Models: mixmil, scPhase, abmil, set_transformer, cloudpred, perceiver_io, gpio
```

## Script Dependencies

| Script | Reads | Writes |
|--------|-------|--------|
| hpo.py | configs/*.yaml, data/precomputed/, outputs/splits.json | outputs/ray_results/ |
| extract_hpo_results.py | outputs/ray_results/, configs/*.yaml | outputs/hpo_analysis/ |
| run_sensitivity.sh | top_configs/*.yaml, data/precomputed/, outputs/splits.json | outputs/{exp}/checkpoints/ |
| run_inference.py | outputs/{exp}/checkpoints/*.ckpt, data/precomputed/ | outputs/{exp}/analysis/ |
| run_analysis.py | outputs/{exp}/analysis/predictions.parquet + attention_weights.h5 | outputs/{exp}/analysis/*/ |
| run_baselines.py | data/precomputed/, outputs/splits.json, data/metadata_ROSMAP/ | outputs/baseline_results.csv |

## Common Pitfalls

1. **Data path**: Training configs may have `precomputed_dir: data/precomputed/rosmap/` but actual data is at `data/precomputed/`. The `run_sensitivity.sh` script overrides with `--precomputed-dir` — make sure it points to the right path.

2. **Pyro guide device**: When loading checkpoints for inference, the guide must be explicitly moved to GPU. `model.to(device)` does NOT move the guide.

3. **HGT attention format**: Model returns `list[Tensor]` (flat across batch). Downstream code uses `split_batch_attention_by_subject()` to convert to per-subject format. Do not assume per-subject structure directly from model output.

4. **Inference runs on ALL subjects**: `run_inference.py` discovers all `.pt` files in the precomputed directory (not just the fold's val set). Filter to val subjects when computing fold-specific metrics.
