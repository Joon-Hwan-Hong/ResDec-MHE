# CloudPred Baseline

CloudPred (Toloşi & Bock, 2023) adapted for ROSMAP cognitive resilience regression.

## Method Summary

CloudPred models patient-level phenotype from single-cell data by:
1. **PCA** — reduce per-cell gene expression from 4,796 to 10 dimensions
2. **GMM** — fit a K=10 Gaussian Mixture Model on training cells
3. **Density features** — represent each patient as a K-dim vector of mean mixture membership
4. **Polynomial classifier** — degree-2 polynomial maps density features to prediction

Training is two-phase:
- **Phase 1 (warmup):** precompute mixture features, train polynomial head only (1,000 SGD steps, lr=1e-3)
- **Phase 2 (fine-tune):** end-to-end stochastic SGD through both mixture and polynomial (1,000 iterations, lr=1e-4)

## Optimizations vs. Reference Implementation

The reference CloudPred repo (`repo/`) processes data entirely on CPU with Python-level loops. Our adapter (`run_rosmap.py`) applies three optimizations that preserve algorithmic fidelity while drastically improving runtime and memory:

### 1. Vectorized Mixture Computation

**Reference:** `Mixture.forward()` loops over K Gaussian components in Python:
```python
logp = torch.cat([c(x).unsqueeze(0) for c in self.component])  # K iterations
```

**Optimized:** `VectorizedMixture` computes all K Gaussians in a single batched operation:
```python
diff = mus.unsqueeze(1) - x.unsqueeze(0)            # [K, N, D]
logp = -0.5 * (const - log_det + (diff**2 * invvar)) # [K, N] — one op
```

Eliminates the Python for-loop. Numerically equivalent (same log-density formula, same softmax normalization, same mean-across-cells aggregation).

### 2. GPU Training

**Reference:** `train_classifier()` runs on CPU with `cuda=False` default. GPU support exists in the code but is never activated in any call path.

**Optimized:** Model and data are moved to the specified CUDA device. The fine-tuning loop (1,000 iterations x ~412 subjects) runs entirely on GPU. Stochastic SGD semantics are preserved (one subject per optimization step).

### 3. Per-Fold Data Loading + GPU PCA

**Reference:** Our initial adapter loaded all 516 subjects' raw cell data upfront (~30 GB), then concatenated ~1.9M cells x 4,796 genes for PCA (~36 GB), reaching ~47 GB peak RSS.

**Optimized:**
- Subjects are loaded per-fold and freed after PCA transform, cutting peak memory to ~25 GB
- PCA uses `torch.pca_lowrank` on GPU (randomized SVD) instead of `sklearn.decomposition.PCA` on CPU
- The ~36 GB concatenated array is freed immediately after PCA fitting

### Performance Impact

| Metric | Before Optimization | After Optimization |
|--------|--------------------|--------------------|
| Peak RAM | ~47 GB | ~25 GB |
| Per-fold time | ~45 min (est.) | ~8 min |
| 5-fold total | ~3.75 hr (est.) | ~40 min |
| GPU utilization | 0% | Active (fine-tune + PCA) |

### What Is NOT Changed

- **Algorithm:** GMM initialization, warmup schedule, stochastic SGD fine-tuning — all match the reference
- **Hyperparameters:** K=10 centers, 10 PCA dims, 1,000 warmup steps, 1,000 fine-tune iterations
- **Evaluation:** Same MSE loss, same R2/MAE/Pearson/Spearman metrics

## Usage

```bash
baselines/cloudpred/.venv/bin/python -u baselines/cloudpred/run_rosmap.py \
    --data-dir data/precomputed/ \
    --splits outputs/splits.json \
    --metadata-dir data/metadata_ROSMAP/ \
    --results-dir outputs/baselines/cloudpred \
    --device cuda:1
```

## Results (5-fold CV, 516 subjects)

| Metric | Mean +/- Std |
|--------|-------------|
| R2 | 0.050 +/- 0.025 |
| MAE | 0.916 +/- 0.024 |
| RMSE | 1.128 +/- 0.045 |
| Pearson r | 0.232 +/- 0.054 |
| Spearman rho | 0.250 +/- 0.075 |

## Vendored Repository

`repo/` contains the CloudPred source from https://github.com/solevillar/CloudPred.
Only the GMM, Mixture, and Polynomial classes are referenced for initialization logic.
The training loop is reimplemented with the optimizations above.
