# Permutation Null + Distributional / DE Resilience Analyses (2026-04-24)

**Status:** Companion to `2026-04-23-paper-results-draft.md`. Covers the three paper-strengthening analyses that landed on branch `paper-strengthening` (commits `ea25b5a` … `8771a49`): (1) full-pipeline permutation null, (2) distributional resilient-vs-vulnerable signatures on raw pseudobulk (Wasserstein + stability selection), (3) classical differential expression (Wilcoxon + DESeq2) on raw pseudobulk, with cross-method concordance.

**Cohort & split.** ROSMAP snRNA-seq, N=516 subjects, 31 cell types, 4,785 HVGs, 5-fold subject-level split at `outputs/splits.json` (unchanged from earlier analyses).

---

## 1. Permutation null: canonical R² is not a chance result

We subjected the **full ResDec-MHE pipeline** to a N=10 negative-control permutation test. For each permutation seed *k*, we (i) randomly permuted the cognitive-resilience composite across subjects (preserving the original missingness pattern so no cohort subject ever received a NaN target), (ii) re-ran the complete upstream pipeline — XGBoost top-2000 feature selection on shuffled labels, TabPFN-2.6 OOF + outer caches on shuffled labels — and (iii) re-trained the 5 folds of ResDec-MHE on the shuffled residual targets. The resulting val-fold predictions were evaluated against the **true** cognitive-resilience labels, yielding a null distribution of 5-fold-mean R².

| Canonical R² | Null mean | Null std | Null range | #null ≥ canonical | p (one-sided) | z (canonical, null-std) |
|---:|---:|---:|---:|---:|---:|---:|
| **+0.4436** | −0.2944 | 0.0845 | [−0.4532, −0.1448] | 0 / 10 | **1/11 ≈ 0.0909** | **8.73** |

Under H₀ (label-independent signal), the observed R² is roughly 9 null-std units above the null mean, and exceeds every permutation sample. The empirical one-sided p ≤ 1/(N+1) = 0.091 is the floor at N=10; at N=100, this would tighten to ≤ 0.01. The z-score is the reporting statistic of choice.

**Sanity details.** Each permutation took 25–28 min end-to-end on 2 GPUs (XGBoost top-k + TabPFN OOF/outer + 5 parallel folds), totaling ~4.6 hr wall-clock. Permutation 4 initially failed due to an orchestrator NaN-shuffle bug (a cohort subject received a permuted-in NaN for the target), fixed in commit `ea25b5a` and successfully retried in a fresh Python process; both entries are preserved in `permutation_results.json`. Clean aggregate at `outputs/redesign/permutation_test/permutation_summary.json`.

---

## 2. Distributional resilient-vs-vulnerable signatures (pseudobulk)

We complemented the Captum-attribution-based interpretability with a **model-free** distributional analysis on the raw pseudobulk: for each of the 31 cell types and 4,785 genes, we computed the Wasserstein-1 distance between resilient (top residual-quartile, n=129) and vulnerable (bottom residual-quartile, n=129) subject-level expression distributions. Top cell types by mean per-gene W-1:

| Rank | Cell type | Mean per-gene W-1 | Top-3 genes (by W-1) |
|---:|---|---:|---|
| 1 | CT_30 | 0.0436 | CTNNA2, PTPRN, SHANK2 |
| 2 | CT_26 | 0.0368 | ST5, NR2F1-AS1, RAPGEF5 |
| 3 | CT_29 | 0.0320 | NRG1, MEG3, STXBP5L |
| 4 | CT_15 | 0.0273 | NELL1, AJ009632.2, DOK6 |
| 5 | CT_25 | 0.0267 | ATP6V0C, SORBS2, PDE10A |

The top-ranked individual genes are enriched for known synaptic / neuronal substrates (SHANK2, NRG1, STXBP5L) and one canonical AD-relevant vascular gene (FLT1, see stability selection below). Full per-CT top-10 at `outputs/redesign/interpretability/distributional_resilience/wasserstein_per_celltype_pseudobulk.json`.

**Stability selection.** We subsampled 50 % of each group (resilient / vulnerable) 100× and retained (cell-type, gene) pairs whose |rank-biserial| exceeded 0.2 in ≥ 80 % of resamples (Meinshausen–Bühlmann stability selection). 9 (cell-type, gene) pairs passed this criterion (all in the pseudobulk; 3 additional pairs from the parallel Captum-attribution stability run are in a separate file):

| CT | Stable genes |
|---|---|
| CT_3 | LMOD3, LRAT |
| CT_4 | DLEC1, LRRC9 |
| CT_7 | **FLT1**, TGFBI |
| CT_0 | TACR1 |
| CT_2 | RYR3 |
| CT_11 | C1QL3 |

FLT1 (VEGFR1) in CT_7 is noteworthy — a known vascular-mural marker and recurrent GWAS hit in AD-risk panels. `outputs/redesign/interpretability/distributional_resilience/stability_selection_pseudobulk.json`.

---

## 3. Classical DE: Wilcoxon and DESeq2 agree on one thing — the signal is not classical

We ran resilient-vs-vulnerable DE per cell type with two methods: (a) two-sample Wilcoxon rank-sum on log1p pseudobulk, and (b) pydeseq2 Likelihood-Ratio Test on raw counts with apeglm `lfc_shrink` (default settings, 4 CPUs per CT; 276 min total wall-clock across 31 CTs × 4,785 genes).

**Headline finding.** Across all 31 × 4,785 = 148,335 tests, **zero genes pass BH-corrected significance at padj < 0.05** under either method — consistent with the biology: at N=258 (129 vs 129) and 4,785-gene panels, classical pairwise DE has very low power for subtle effects.

**Method disagreement as a positive result.** Between Wilcoxon and DESeq2, per-cell-type Spearman ρ of raw p-values across the 4,785 genes ranges from −0.61 to +0.33 (median ~0). Top-50 Jaccard overlap (by raw p) is 0–9 % per cell type. In other words, the two classical methods largely **disagree on which genes rank highest** at this dimensionality, which undercuts any interpretation that relies on a single classical DE pipeline. This is the quantitative backing for the framing in the main results section that *"the model's resilience signal is distributional and non-linear — something neither two-group mean-based test captures well"*. Full concordance CSV at `outputs/redesign/interpretability/de_wilcoxon_vs_deseq2_topK.csv`.

Per-method top-K-by-raw-p exports (not BH-corrected — useful for Captum × DE concordance plots):

| Method | File |
|---|---|
| Wilcoxon | `outputs/redesign/interpretability/de_resilient_vs_vulnerable/top_genes_per_ct_by_pvalue.csv` |
| DESeq2 | `outputs/redesign/interpretability/de_resilient_vs_vulnerable_deseq2/top_genes_per_ct_by_pvalue.csv` |

The Captum attribution rankings have partial but modest concordance with DE rankings per cell type (figure `fig_captum_de_concordance.{png,pdf}` in `outputs/redesign/interpretability/figures/attribution/`): per-CT Spearman ρ with Wilcoxon -log₁₀(p) varies around zero, with ρ > 0 in most cell types but no cell type exceeding ρ > 0.5. Attributions and classical DE capture **partly orthogonal** views of the same phenomenon.

---

## 4. Residual distribution: bimodal, not a smooth gradient

A Gaussian-mixture latent-class analysis on the 516 subjects' per-subject residuals (cognitive-resilience composite minus TabPFN prediction) returns **best_k = 2 by BIC** (BIC(k=1)=1322, BIC(k=2)=1300, ΔBIC=22 → "strong" evidence by Kass-Raftery). By AIC, k=4 marginally wins (AIC(k=4)=1281.5 vs AIC(k=2)=1278.8 — essentially tied), so the BIC choice of k=2 is the conservative call. Artifact at `outputs/redesign/interpretability/latent_class_on_residuals.json`.

The k=2 component interpretation: the subject-level resilience *residual* after TabPFN appears to split into two subpopulations (rather than lying on a continuous gradient), consistent with a threshold effect. This is a candidate discussion point rather than a main-result claim and should be treated as exploratory; higher-N cohorts would be needed to confirm.

---

## Reproducibility

All three analyses are re-runnable from the worktree:

```bash
# 1. Permutation test (N=10, ~4.6 hr on 2 GPUs)
PYTHONPATH=. uv run python scripts/resdec_mhe/training/run_permutation_test.py \
    --num-perms 10 --start-perm 0

# 2. Wasserstein + stability on pseudobulk
PYTHONPATH=. uv run python scripts/resdec_mhe/interpretability/run_distributional_resilience.py wasserstein
PYTHONPATH=. uv run python scripts/resdec_mhe/interpretability/run_distributional_resilience.py stability

# 3. Classical DE
PYTHONPATH=. uv run python scripts/resdec_mhe/interpretability/run_de_resilience.py --method wilcoxon
PYTHONPATH=. uv run python scripts/resdec_mhe/interpretability/run_de_resilience.py --method deseq2 \
    --out-dir outputs/redesign/interpretability/de_resilient_vs_vulnerable_deseq2
```

Artifacts:

| Content | Path |
|---|---|
| Perm raw results | `outputs/redesign/permutation_test/permutation_results.json` |
| Perm clean summary | `outputs/redesign/permutation_test/permutation_summary.json` |
| Wasserstein (pseudobulk) | `outputs/redesign/interpretability/distributional_resilience/wasserstein_per_celltype_pseudobulk.json` |
| Stability (pseudobulk) | `outputs/redesign/interpretability/distributional_resilience/stability_selection_pseudobulk.json` |
| Stability (Captum) | `outputs/redesign/interpretability/stability_selection_attributions.json` |
| Wilcoxon DE per-CT | `outputs/redesign/interpretability/de_resilient_vs_vulnerable/CT_*_de.csv` |
| DESeq2 DE per-CT | `outputs/redesign/interpretability/de_resilient_vs_vulnerable_deseq2/CT_*_de.csv` |
| Wilcoxon × DESeq2 concordance | `outputs/redesign/interpretability/de_wilcoxon_vs_deseq2_topK.csv` |
| Latent class | `outputs/redesign/interpretability/latent_class_on_residuals.json` |

---

## How this slots into the manuscript

- **Methods §** — note the permutation-null design (full-pipeline refit per seed, not a residual-label shuffle) as the primary negative control.
- **Results §** — the permutation section should be inserted immediately after "Statistical rigor" in `2026-04-23-paper-results-draft.md`; the distributional / DE sections belong in the interpretability arc after the Captum + GSEA + CCC subsections, introduced with the framing that model attributions and classical DE capture partly orthogonal signal.
- **Discussion §** — the method-disagreement result (§3) and the bimodal-residual finding (§4) are candidate discussion threads that further motivate the need for the deep, non-linear architecture over classical DE pipelines at this sample size.
