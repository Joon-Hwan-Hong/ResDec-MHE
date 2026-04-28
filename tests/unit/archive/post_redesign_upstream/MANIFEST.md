# Post-Redesign Upstream Test Archive

Archived 2026-04-28. Tests for upstream / pre-redesign components that will not
be touched again now that the ResDec-MHE paper pipeline is locked. User
explicitly authorized aggressive archive + no mock data going forward.

Each entry cites a code-level justification: an import target or a class/script
under test that is no longer part of the canonical pipeline (canonical = the
ResDec-MHE Lightning module + ResDecMHEHead + scripts/resdec_mhe/...).

## Archived files

### tests/unit/data/ (all except `test_datamodule.py`)

- `test_cell_sampling.py` — cell-level sampling helpers; not used by canonical pseudobulk pipeline
- `test_collate.py` — pre-redesign collate fn (n-cells / n-regions stitching), pseudobulk path bypasses
- `test_datasets.py` — older Dataset implementations, superseded by canonical DataModule
- `test_enriched_features.py` — `src.data.enriched_features` (pre-redesign per-cell feature enrichment)
- `test_feature_loaders.py` — `src.data.feature_loaders` (HGT-era helpers)
- `test_flat_cells.py` — flat-cell concatenation pipeline (pre-pseudobulk)
- `test_io.py` — `src.utils.io.unpack_hgt_for_ccc` and HDF5 attention I/O (HGT/CCC era)
- `test_liana_processing.py` — LIANA preprocessing (CCC HGT branch, deprecated)
- `test_lr_annotation.py` — ligand-receptor annotation (CCC era)
- `test_metadata_wiring.py` — pre-redesign metadata-vector wiring
- `test_prefetcher.py` — `src.data.prefetch.ThreadedPrefetcher` (HGT era)
- `test_preprocessing.py` — older AnnData preprocessing helpers
- `test_splits.py` — splits utility tests; covered transitively by datamodule
- `test_tabpfn_input.py` — tabpfn_input flatten helpers; canonical reads precomputed caches via DataModule

### tests/unit/visualization/ (all except `test_composite.py`, `test_config.py`, `test_theme.py`)

- `test_activation_plots.py` — paper figures locked, smoke tests no longer maintained
- `test_attention_plots.py` — paper-locked attention figures
- `test_attribution_plots.py` — paper-locked Captum/DE concordance figure
- `test_counterfactual_plots.py` — paper-locked counterfactual figure
- `test_distributional_plots.py` — paper-locked Wasserstein/distributional figures
- `test_embedding_plots.py` — paper-locked embedding/UMAP figures
- `test_importance_plots.py` — paper-locked importance bars (also flagged as flaky by user)
- `test_learning_curve_plots.py` — paper-locked learning-curve figure
- `test_prediction_plots.py` — paper-locked predicted-vs-true scatter
- `test_training_curves.py` — paper-locked training-curve figure
- `test_weight_space_plots.py` — paper-locked weight-space PCA figure

### tests/unit/utils/

- `test_device.py` — `src.utils.device.move_batch_to_device` (small utility, stable, no future changes expected)
- `test_shm.py` — `src.utils.shm` shared-memory cleanup (system utility, stable)

### tests/unit/inference/

- `test_predict_script.py` — `scripts/inference/run_inference.py` import-style and split-mapping smoke. Canonical interpretability uses `ResDecLightningModule.load_from_checkpoint` directly (see `scripts/resdec_mhe/interpretability/captum_composite_attribution.py`); `Predictor.from_checkpoint` is deprecated for canonical pipeline.

### tests/unit/scripts/

- `test_run_permutation_test.py` — `scripts/run_permutation_test.py` shuffle helper. Permutation test was a one-time experiment (N=10, completed 2026-04-24); no future re-runs planned.

### tests/unit/training/

- `test_callbacks.py` — bulk of file tests `TemperatureAnnealing`, `GradientNormLogger`, `KLAnnealing-related`, `ResilienceModelCheckpoint`, Pyro restore/resync — all used by deprecated `scripts/training/train.py`. Canonical `scripts/resdec_mhe/training/train.py` only imports `MinEpochEarlyStopping`. User explicitly authorized archive.
- `test_generate_plots_script.py` — old `scripts/analysis/generate_plots.py` orchestrator
- `test_gradient_modulation.py` — `OGMGEModulator` / `GradientModulationCallback` from `src.training.gradient_modulation`. Verified zero matches in `src/training/resdec_lightning_module.py` and `scripts/resdec_mhe/training/train.py` — gradient modulation is not in canonical pipeline.
- `test_losses.py` — `BetaNLLLoss` heteroscedastic loss. Canonical `resdec_lightning_module.py` imports only `mse_loss` (line 48); `BetaNLLLoss` is used only by deprecated `src/training/lightning_module.py` (line 44).
- `test_metrics.py` — `ResilienceMetrics`. Used by deprecated `src/training/lightning_module.py` (line 45) and `scripts/analysis/run_analysis.py` (line 697). Canonical `resdec_lightning_module.py` does NOT import metrics module.
- `test_run_analysis_script.py` — old `scripts/analysis/run_analysis.py` orchestrator (HGT/CCC era plot pipeline)
