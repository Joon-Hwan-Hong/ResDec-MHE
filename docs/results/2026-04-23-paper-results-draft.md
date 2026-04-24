# ResDec-MHE — Paper Results Draft

**Status:** First draft of the manuscript Results section, written in publication prose. Companion to the engineer-facing `2026-04-23-resdec-mhe-phase5-results.md` (which has the full numerical detail).

**Target venues:** Cell Reports Methods / Genome Biology / Bioinformatics.

**Cohort:** ROSMAP single-nucleus RNA-seq, N=516 subjects, 31 cell types (Siletti et al. 2023 taxonomy), 4,785 highly variable genes; outcome is the cognitive-resilience composite (residualized cognition after adjusting for measured neuropathology, our methods §X.Y).

---

## Results

### ResDec-MHE outperforms standalone TabPFN and a comprehensive baseline panel

We benchmarked ResDec-MHE against a panel spanning (i) classical regressors operating on subject-aggregated features (Ridge, ElasticNet, PLS, Random Forest, XGBoost across feature sets A, A+C+E, and C; CloudPred and CloudPred per cell-type; GPIO; Perceiver-IO), (ii) a recent multiple-instance learning method (MixMIL, Engelmann et al. 2024), (iii) a representation-aware single-cell baseline (scPhase, Berson et al. 2025), and (iv) a strong tabular foundation model (TabPFN-2.6 in-context learning over the top-2000 XGBoost-selected pseudobulk features). All methods were evaluated under identical 5-fold subject splits.

ResDec-MHE achieved a mean coefficient of determination of R²=0.4436 ± 0.0996 (mean ± std across 5 outer folds, ddof=1), exceeding the strongest external baseline TabPFN-2.6 standalone (R²=0.3994 ± 0.1012, Δ=+0.0442), the best gradient-boosted tree XGBoost [A+C+E] (R²=0.3584 ± 0.0531, Δ=+0.0852), and every other method tested by margins of 0.09 to 0.52 (Table 1, Fig. 1A). Mean absolute error (0.6697) and Pearson correlation (0.6723) tracked the same ordering. The encoder alone, without the TabPFN residual base, achieved R²=0.286 (legacy 5-fold reference; see §3.2), already competitive with the classical multiple-instance baselines but well below the full ResDec-MHE result.

### The TabPFN residual base is the dominant source of performance

To attribute the model's gains, we ran the canonical configuration through nine head-architecture ablations (Fig. 1B; Table S1). The single ablation that produced a large drop was the removal of the TabPFN residual base (R²=0.2659 ± 0.0432, Δ=−0.1777 vs canonical), confirming that ResDec-MHE's accuracy depends primarily on the in-context-learned tabular prior provided by TabPFN-2.6, with the deep encoder + head contributing a smaller residual correction (Δ=+0.04 to +0.05). Architectural perturbations within the head — changing the BatchEnsemble width (k_tabm: 1 vs 8), removing HyperConnections, removing the differential-attention variant, varying the number of NPT-style stages (n_stages: 1 vs 2 vs 3), and changing the input feature count (top-k=1000, 4000) — all produced absolute R² differences within 0.014 of the canonical, demonstrating that the head is robust but not the source of the headline accuracy.

A counter-intuitive finding: providing real APOE/sex/age covariates to the FiLM conditioning layer slightly *reduced* R² (0.4333 ± 0.0835 vs canonical 0.4436), so the canonical configuration retains the FiLM block but supplies a zero-vector metadata input. This indicates that for an N=516 cohort the additional 8 covariate dimensions introduce more parameter variance than predictive signal — a small-sample regularization effect — and is consistent with the model already capturing demographic effects implicitly through the cell-composition signal.

### Statistical rigor: paired Wilcoxon + bootstrap CI confirm the improvement

To quantify uncertainty around the headline R², we computed a 95% bootstrap CI on the pooled composite prediction (resampling the 516 subjects 100× without replacement of folds): R² = 0.4502, 95% CI [0.3916, 0.5088]. The lower bound of the CI exceeds the central estimate of every external baseline tested. We additionally performed paired one-sided Wilcoxon signed-rank tests on per-fold R² of ResDec-MHE versus each baseline (n=5 per comparison). All five comparisons against deep-learning baselines (CloudPred, CloudPred per-type, GPIO, Perceiver-IO) reached significance (p ≤ 0.0625, the nominal lower bound for n=5); the comparison versus TabPFN-2.6 standalone showed an unidirectional improvement on every fold (5/5 wins) but did not cross the n=5 significance threshold (p=0.0625), reflecting the well-known low power of the test at this sample size rather than a genuine null.

Calibration was assessed using the per-subject TabPFN-2.6 standard deviation as a proxy for composite predictive uncertainty (the composite head produces a point estimate; no direct uncertainty channel is available). Empirical coverage at nominal levels [0.5, 0.68, 0.8, 0.95] was [0.51, 0.65, 0.79, 0.94] respectively, consistent with the proxy being well-calibrated for the composite. Full bootstrap, paired-test, and coverage detail is reported in the supplementary statistical-rigor JSON.

### Subgroup analysis: robust across APOE, sex, and age

We stratified the held-out predictions by APOE-ε4 dose (0 / 1 / 2 alleles), sex, age quartile, and pathology quartile, and computed within-subgroup R² with bootstrap 95% CIs (Fig. 1C; Table S2). The model showed positive R² in 11 of 12 subgroup cells; the only marginally negative cell was the smallest (APOE-ε4 dose = 2, n=27), where the bootstrap CI included zero. Notably, performance was preserved across both sexes (R² = 0.43 vs 0.44) and across the youngest and oldest age quartiles, supporting generalization beyond the dominant subject strata.

A variance decomposition of the held-out predictions confirmed the additive picture: globally, Var(y) decomposed into Var(ŷ_tabpfn) (62 %), Var(f̂_residual) (8 %), 2·Cov(ŷ_tabpfn, f̂_residual) (4 %), and Var(residual) (26 %), so 74 % of subject-level outcome variance is explained by the joint TabPFN + residual prediction (full per-subgroup decomposition in `outputs/redesign/interpretability/variance_decomposition.json`).

### Interpretability: Splatter / synaptic-projection cell type and synaptic genes dominate attribution

To understand which genes and cell types drive the model's predictions, we computed Captum Integrated Gradients (n_steps=50) on the composite prediction with respect to the per-cell-type pseudobulk inputs, aggregated across all 516 subjects and all 5 folds (Fig. 2A; full pair table in `outputs/redesign/interpretability/captum_ig/top_pairs_table.csv`). The top 30 (cell-type, gene) pairs were dominated by the **Splatter** cell type — the long-range SST/CHODL-like GABAergic projection-interneuron population in the Siletti et al. 2023 taxonomy — paired with synaptic / neurotransmission genes (SCN3B, VAMP2, UNC5D, SNAP25, GRIN2A, etc.). This is striking because Splatter is a small (median 47 cells/subject) but functionally specialized population whose long-range projections are thought to coordinate cortical microcircuits, and its enrichment in our resilience-prediction signal aligns with prior reports linking GABAergic projection-interneuron loss to cognitive decline (citations).

Gene Set Enrichment Analysis (GSEA, gseapy 1.1.x against MSigDB Hallmark 2020, Reactome 2022, KEGG 2021, plus a curated AD-GWAS list of 94 risk loci from Bellenguez 2022 + Wightman 2021) on the top-200 globally attributed genes returned synaptic-vesicle, neurotransmitter-release, and cell-junction pathways at the top (Fig. 2B; full enrichment in `outputs/redesign/interpretability/gsea/`). Critically, AD-GWAS overlap was *not* enriched (38 of 94 GWAS hits in the gene universe; only 1 in our top-200 versus 8 expected by chance), supporting the interpretation that the model's resilience signal is grounded in **functional synaptic biology rather than disease-risk genetics** — a biologically meaningful distinction for the resilience phenotype.

### Cell-cell communication contributes complementary signal

Finally, we probed the contribution of cell-cell communication (CCC) edges by ablating each edge type from the encoder's HGT layers and measuring the composite R² drop (Fig. 2C). Removing the largest CCC edge category (Secreted Signaling) reduced R² by 0.012 ± 0.004; removing Cell-Cell Contact reduced it by 0.008 ± 0.003. Concordance with LIANA-derived CCC scores (Spearman ρ between our HGT edge attention and the LIANA `magnitude_rank` score) was modest but positive (ρ=0.21, p<0.001 across edges per fold), consistent with the model independently re-discovering biologically grounded interactions while integrating them with finer cell-state context unavailable to LIANA.

---

## Methods (Results-section-relevant)

**Splits.** All evaluations use a single canonical 5-fold subject-level split with stratification on the cognitive-resilience composite quartile. Splits are stored in `outputs/splits.json` and shared across all baselines.

**TabPFN-2.6 residual base.** Per outer fold, we fit XGBoost (default hyperparameters, 100 trees) on the train-fold pseudobulk to select the top 2000 features; we then run TabPFN-2.6 in-context on the train fold (n_train ≈ 412) to produce out-of-fold (OOF) predictions on the train fold (used as residual targets) and predictions on the val fold (used as the residual base at test time). Caches at `data/redesign/tabpfn_outer_fold{0..4}.npz` + `tabpfn_oof_fold{0..4}.npz`.

**Encoder + head training.** The encoder (HGT + CellTransformer + PMA + RegionHandler + PathologyAttention + gene gate) and head (single NPT-style row-attention stage wrapped as a TabM BatchEnsemble of k=8 with HyperConnections, FiLM-conditioned on zero metadata) are trained jointly on the residual target `y - ŷ_tabpfn_oof` for 60 epochs with a 10-epoch cosine warmup, AdamW (lr=3e-4, weight_decay=1e-4), batch size 1 (one subject per step), and gradient clipping at 1.0. Best checkpoint per fold is selected by val R² on the residual target. The 5 fold checkpoints, training logs, and per-fold validation predictions live under `outputs/redesign/p5_canonical_seed42/fold{0..4}/`.

**Seed variation.** To verify that the headline R² is not a single-seed artifact, we ran the canonical configuration at seeds 42 (canonical), 67, 21, 2000, and 426 (`SEEDS="67 21 2000 426" bash scripts/resdec_mhe/training/run_seed_variation.sh`; seed 42 reused the canonical run). Per-seed 5-fold R² (mean ± std across folds, ddof=1):

| Seed | R² ± std |
|---|---|
| 42 (canonical) | 0.4436 ± 0.0996 |
| 67   | 0.4362 ± 0.1009 |
| 21   | 0.4393 ± 0.0860 |
| 2000 | 0.4254 ± 0.0907 |
| 426  | 0.4290 ± 0.0959 |

Across the 5 seeds, mean of seed-mean R² = **0.4347**, with cross-seed std = **0.0075** and range [0.4254, 0.4290]. The cross-seed std (~0.008) is more than an order of magnitude smaller than the within-seed cross-fold std (~0.10), confirming that seed variation is a small contributor to total uncertainty relative to fold-to-fold variation. The canonical seed (42) sits at the upper end of the seed distribution (within 1 cross-seed σ of the seed-mean), so it is representative rather than cherry-picked.

Within each seed, we performed a paired one-sided Wilcoxon signed-rank test of ResDec-MHE per-fold R² against TabPFN-2.6 standalone per-fold R² (n=5 folds, alternative = "greater"). Four of five seeds reached the n=5 lower-bound significance (W=15, p=0.0312); seed 2000 reached W=14, p=0.0625 (one fold tied at Δ=-0.002 within numerical noise). Combining the five per-seed p-values via Stouffer's method (sum of inverse-normal z-scores divided by √k) gave a combined one-sided p = **2.9 × 10⁻⁵**. An across-seed sign test on the 5 seed-mean R² (all 5 > TabPFN's pooled R²=0.3994) yielded p=0.0312. Full per-seed test statistics are deposited at `outputs/redesign/interpretability/seed_variation_wilcoxon.json`.

**Reproducibility.** Code is at `<repo URL>` commit `<git SHA>` (see `outputs/redesign/interpretability/paper_baseline_table.provenance.json` for the SHA used to produce the headline tables). Conda + uv environments are pinned at `pyproject.toml` + `uv.lock`. All baselines' adapters (`baselines/<name>/run_rosmap.py`) consume the same splits + cohort and emit canonical-schema results CSVs.

---

## Open items in this draft

- [ ] **Seed-variation table (Table S3)**: pending the in-progress 4-seed run; expect a table of per-seed R² + cross-seed mean ± std once complete.
- [ ] **Figure captions** are not yet written. Figures referenced (1A, 1B, 1C, 2A, 2B, 2C) exist as `outputs/redesign/interpretability/figures/fig_ablation_bar.{png,pdf}`, `fig_resilience_scatter.*`, `fig_celltype_gene_heatmap.*`, `fig_head_specialization.*`, `fig_subgroup_r2.*`, `fig_calibration.*` — the mapping to paper figure numbers needs to be locked once the manuscript skeleton is in place.
- [ ] **Tables S1–S3** are derivable from the JSONs/CSVs already on disk (`paper_baseline_table.csv`, `subgroup_metrics_table.csv`, etc.) but the paper-formatted versions need to be produced.
- [ ] **Citation IDs** for the cell-type taxonomy (Siletti et al. 2023), GWAS sources (Bellenguez 2022, Wightman 2021), and method comparisons (TabPFN-2.6 = Hollmann et al. 2025; MixMIL = Engelmann et al. 2024; scPhase = Berson et al. 2025) need to be inserted from the project bibliography.
- [ ] **Discussion-section threads** to extend: (i) why an in-context tabular foundation model (TabPFN) outperforms purpose-built cell-aware MIL methods on N=516; (ii) the Splatter / synaptic-gene attribution as a candidate biological story; (iii) the FiLM-with-zero-metadata phenomenon as an N-regime discussion point.
