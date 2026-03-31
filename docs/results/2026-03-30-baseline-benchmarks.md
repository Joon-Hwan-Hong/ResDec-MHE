# Baseline Benchmarks — Pre-Blocked-HVG (ARCHIVED)

> **NOTE:** These results use the OLD random-subsample HVG (4,796 genes) with no holdout test set
> (5-fold CV, 516 subjects). A new pipeline with blocked HVG (scran-style per-cell-type variance)
> and 10% holdout is running — see `outputs/pipeline_20260331/` for updated results.

Reference: Our model (HPO7 production, old HVG) R² = 0.323 +/- 0.067

## Completed

### CloudPred (Toloşi & Bock 2023) — unstructured cell bag

- **Input:** All cells as unstructured bag → PCA(10) → GMM(K=10) density features → polynomial regression
- **What it sees:** Raw cell expression only. No cell type structure, no CCC graph, no pseudobulk.
- **Optimizations:** Vectorized mixture (batched Gaussians), GPU fine-tuning, GPU PCA (torch.pca_lowrank)

| Fold | R² | MAE | Pearson r | Spearman rho |
|------|-----|-----|-----------|--------------|
| 1 | 0.0382 | 0.9288 | 0.2275 | 0.2588 |
| 2 | 0.0330 | 0.9355 | 0.1915 | 0.2137 |
| 3 | 0.0735 | 0.9141 | 0.2761 | 0.2703 |
| 4 | 0.0252 | 0.8759 | 0.1692 | 0.1510 |
| 5 | 0.0803 | 0.9236 | 0.2952 | 0.3546 |
| **Mean +/- Std** | **0.050 +/- 0.025** | **0.916 +/- 0.024** | **0.232 +/- 0.054** | **0.250 +/- 0.075** |

### CloudPred per-type — cell-type-aware variant

- **Input:** Cells split by type (using cell_offsets) → global PCA(10) → per-type GMM(K=3) for 24 active types → 72 density features → polynomial regression
- **What it sees:** Raw cell expression + cell type structure. No CCC graph, no pseudobulk aggregation.
- **Optimizations:** Batched per-type mixture (all 24 types in one GPU kernel), GPU PCA

| Fold | R² | MAE | Pearson r | Spearman rho |
|------|-----|-----|-----------|--------------|
| 1 | 0.0772 | 0.8786 | 0.3008 | 0.2320 |
| 2 | 0.0426 | 0.9337 | 0.2190 | 0.2339 |
| 3 | 0.2376 | 0.8066 | 0.4875 | 0.4975 |
| 4 | 0.0400 | 0.8585 | 0.2081 | 0.1532 |
| 5 | 0.1158 | 0.8743 | 0.3552 | 0.4376 |
| **Mean +/- Std** | **0.103 +/- 0.082** | **0.870 +/- 0.046** | **0.314 +/- 0.114** | **0.311 +/- 0.148** |

**Takeaway:** Adding cell type structure doubles R² (0.050 → 0.103) and improves Pearson r (0.232 → 0.314). High fold variance (0.04–0.24) suggests the density-based approach is sensitive to fold composition.

## Running

### Classical ML baselines (cuML GPU + XGBoost)

- **Models:** Ridge, ElasticNet, RandomForest, XGBoost, PLS
- **Feature sets:**
  - C: cell-type proportions [31]
  - A: flattened pseudobulk [148,607]
  - A+C+E: all combined [148,656]
- **Status:** Running with --cv-tune. ElasticNet on feature set A is slow (148K coordinate descent). Ridge/ElasticNet on C complete.

## Pending

### Perceiver IO (Jaegle et al. 2021)
- **Input:** 31 cell-type pseudobulk tokens [31, 4796] + 1 CCC summary token → cross-attention to latents → regression
- **What it sees:** Pseudobulk + CCC summary (18 features). No individual cells.
- **Script:** `baselines/perceiver_io/run_rosmap.py` — ready to launch

### GPIO (Li et al. 2026)
- **Input:** CCC graph (31 nodes) with RWPE positional encoding + pseudobulk features → Perceiver IO encoder → regression
- **What it sees:** Full CCC graph topology + pseudobulk. No individual cells.
- **Script:** `baselines/gpio/run_rosmap.py` — ready to launch

### MixMIL (Engelmann et al. 2024)
- **Input:** Unstructured bag of cells with 30-dim scVI embeddings → GLMM + attention MIL
- **What it sees:** Cell-level scVI embeddings. No cell type structure, no CCC graph.
- **Blocker:** Needs `baselines/shared/mixmil_input.h5ad` (scVI embeddings from adata)

### ABMIL (Ilse et al. 2018)
- **Input:** Unstructured bag of cells with 30-dim scVI embeddings → gated attention pooling → regression
- **What it sees:** Same as MixMIL
- **Blocker:** Needs venv + `mixmil_input.h5ad`

### Set Transformer (Lee et al. 2019)
- **Input:** Unstructured bag of cells with 30-dim scVI embeddings → ISAB + PMA pooling → regression
- **What it sees:** Same as MixMIL
- **Blocker:** Needs venv + `mixmil_input.h5ad`

### scPhase (Berson et al. 2025)
- **Input:** All cells with 4797 raw genes → per-cell-type attention → classification-first design adapted for regression
- **What it sees:** Raw cell expression + cell type labels (internal)
- **Blocker:** Needs `baselines/shared/scphase_input.h5ad`

## Input comparison matrix

| Baseline | Individual cells | Cell type structure | Pseudobulk | CCC graph | Feature dims |
|----------|:---:|:---:|:---:|:---:|------|
| **Our model** | Yes | Yes (offsets) | Yes (31x4796) | Yes (full edges) | All |
| Classical ML | No | Proportions (31d) | Yes (148K flat) | Summary (18d) | 31–148K |
| CloudPred | Yes (bag) | No | No | No | PCA→10 |
| CloudPred per-type | Yes (per type) | Yes (offsets) | No | No | PCA→10/type |
| Perceiver IO | No | Yes (31 tokens) | Yes (31x4796) | Summary (1 token) | 4796 |
| GPIO | No | Yes (31 nodes) | Yes (31x4796) | Yes (edges+RWPE) | 4796+16 |
| MixMIL | Yes (bag) | No | No | No | scVI→30 |
| ABMIL | Yes (bag) | No | No | No | scVI→30 |
| Set Transformer | Yes (bag) | No | No | No | scVI→30 |
| scPhase | Yes | Yes (internal) | No | No | 4797 |
