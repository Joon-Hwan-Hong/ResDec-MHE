# Scripts Inventory

## Directory Structure

```
scripts/
├── training/        # Model training, HPO, sensitivity analysis
├── inference/       # Post-training inference, HPO result extraction
├── analysis/        # Analysis pipeline, baselines, plots
├── data/            # Data preprocessing, splits, feature computation
├── profiling/       # Diagnostic/performance profiling
└── utils/           # Cleanup utilities
```

## scripts/training/ — Training & HPO

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `train.py` | Train model on one fold | config.yaml, precomputed/, splits.json | checkpoints/, logs/ |
| `hpo.py` | HPO via Ray Tune + Optuna | config.yaml, precomputed/, splits.json | ray_results/ |
| `run_sensitivity.sh` | Batch 5-fold training across configs | config yamls | checkpoints per fold |
| `run_ablation_2x2.sh` | OGM-GE × temp anneal ablation | configs | checkpoints |
| `run_ablation_pma_cells.sh` | PMA seeds × max_cells ablation | configs | checkpoints |
| `run_ablation_pma_inducing.sh` | PMA seeds × inducing points ablation | configs | checkpoints |
| `run_all_folds_ngpu.sh` | Generic K-fold across N GPUs | config | checkpoints |
| `run_fold_slurm.sh` | SLURM job submission for folds | config | checkpoints |
| `launch_xattn_5fold.sh` | Cross-attention fusion 5-fold | config | checkpoints |
| `run_fold_variance.sh` | Repeated stratified K-fold CV (multiple seeds) | config | per-seed results |

## scripts/ — Pipeline & Benchmarks (top-level)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `run_full_pipeline.sh` | Full automated pipeline: preprocess → HPO → production → ablations → baselines | adata, configs | all results |
| `run_benchmarks.py` | Benchmark orchestrator with status tracking | .pt files, splits | benchmark_status.json |
| `test_benchmark_startup.py` | Dry-run startup test for all baselines | venvs, data files | pass/fail report |
| `benchmark_agent_loop.sh` | Persistent Claude Code agent for benchmark monitoring | status files | analysis reports |

## scripts/inference/ — Inference & Results

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `run_inference.py` | Post-training inference + attention extraction | checkpoint .ckpt, precomputed/ | predictions.parquet, attention_weights.h5 |
| `extract_hpo_results.py` | Parse HPO results, export top-K configs | ray_results/, base config | hpo_analysis/ |

## scripts/analysis/ — Analysis & Baselines

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `run_analysis.py` | Post-hoc interpretability analysis | predictions.parquet, attention_weights.h5 | analysis subdirectories |
| `run_baselines.py` | Classical ML baselines (Ridge, XGB, etc.) | precomputed/, splits.json, metadata/ | baseline_results.csv |
| `analyze_fusion.py` | Standalone fusion weight analysis | checkpoint | fusion analysis |
| `generate_plots.py` | Plot generation | analysis outputs | figures |
| `run_cell_heterogeneity.py` | Standalone cell heterogeneity analysis | PMA attention | heterogeneity results |

## scripts/data/ — Data Preparation

| Script | Purpose |
|--------|---------|
| `preprocess_adata.py` | Raw AnnData preprocessing |
| `merge_adata.py` | Merge h5ad datasets with HVG + CellChatDB gene selection |
| `precompute_features.py` | Generate per-subject .pt files from AnnData |
| `create_splits.py` | Stratified 5-fold subject-level splits |
| `run_liana.py` | LIANA+ CCC analysis per subject |
| `convert_npz_to_pt.py` | .npz → .pt conversion (historical) |
| `convert_to_flat_npz.py` | Padded → flat .npz (historical) |

## scripts/profiling/ — Diagnostics

| Script | Purpose |
|--------|---------|
| `profile_dataloader.py` | DataLoader timing |
| `profile_ddp.py` | DDP multi-GPU scaling |
| `profile_training.py` | CUDA training step timing |
| `profile_vram.py` | VRAM usage profiling |
| `profiling_subset.py` | Select subjects for profiling |
| `mem_profile_wrapper.py` | System + per-process memory monitoring |

## scripts/utils/

| Script | Purpose |
|--------|---------|
| `cleanup_old_experiments.py` | Delete old experiment directories |
