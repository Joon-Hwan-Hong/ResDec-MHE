# Hyperparameter Optimization Log

Chronological record of all HPO rounds for the cognitive resilience model.

---

## HPO Round 1 — Broad Exploration
**Date:** 2026-03-08
**Architecture:** 3-branch (PseudoBulk + HGT + CellTransformer), concat+linear fusion
**Framework:** Optuna TPE + MedianPruner, SQLite backend
**Trials:** 100 (1 fold each, max 50 epochs)
**GPUs:** 2
**Data:** 517 subjects, 4,797 genes (HVG+CellChatDB), 1,000 max cells/type

**Search space (13 HPs, broad):**
| HP | Range |
|---|---|
| lr | [1e-5, 0.01] loguniform |
| d_embed | {64, 128, 256} |
| dropout | [0.0, 0.3] |
| n_hgt_layers | {2, 3, 4} |
| beta | [0.0, 1.0] |
| weight_decay | [1e-6, 0.01] loguniform |
| batch_size | {16} (OOM with 32/64) |
| n_heads | {2, 4, 8} |
| n_inducing | {16, 32, 64} |
| gene_gate_temp | [0.1, 2.0] |
| guide_lr | [0.001, 0.05] |
| selection_temperature | [0.5, 3.0] |
| ogm_alpha | [0.5, 2.0] |

**Key findings:**
- d_embed=64 dominated: 21/25 top trials (128/256 consistently worse)
- n_hgt_layers=2 dominated: 24/25 top trials (3-4 layers overfit)
- n_inducing=64 clearly best: mean_nll=0.65 vs 0.79 for 16
- lr sweet spot: 5e-5 to 1e-3

**Issues:** SQLite concurrency problems with parallel workers; switched to JournalFileBackend. No early stopping triggered (50 epochs too short).

---

## HPO Round 2 — Narrowed Search
**Date:** 2026-03-14
**Architecture:** Same as HPO1
**Framework:** Optuna TPE + MedianPruner, JournalFileBackend
**Trials:** 100 (1 fold, max 50 epochs)

**Changes from HPO1:**
- Fixed d_embed=64, n_hgt_layers=2, n_inducing=64 (converged)
- Narrowed continuous ranges to top-25% region from HPO1
- Added per-worker data caching (eliminated per-trial disk I/O)

**Search space (narrowed):**
| HP | Range |
|---|---|
| lr | [5e-5, 1e-3] loguniform |
| dropout | [0.08, 0.28] |
| beta | [0.03, 0.6] |
| weight_decay | [5e-5, 0.003] loguniform |
| gene_gate_temp | [0.2, 1.0] |
| guide_lr | [0.001, 0.0025] loguniform |
| selection_temperature | [0.5, 1.4] |
| ogm_alpha | [0.5, 1.7] |

**Issues:** OGM-GE investigation revealed it was harmful (see OGM-GE ablation below). Max epochs still too short (no early stopping).

---

## HPO Round 3 — Environment Fix
**Date:** 2026-03-16
**Architecture:** 3-branch, concat+linear fusion
**Framework:** Optuna TPE + MedianPruner
**Trials:** 150 (1 fold, max 100 epochs)
**Data:** max_cells_per_type=2000

**Critical fix:** Corrected training environment mismatch:
- Set deterministic=True, benchmark=False (was reversed during HPO1-2)
- Disabled OGM-GE via config (was unintentionally active in HPO1-2)
- Made temperature annealing (tau_min, anneal_epochs) searchable via Optuna
- Extended max_epochs to 100

**Search space:** Same as HPO2 + added tau_min [1.5, 2.0] and anneal_epochs [10, 30].

**Outcome:** Established correct environment baseline. Warm-started HPO4 with top 5 d_embed=64 configs.

---

## OGM-GE Ablation (2026-03-15)

Investigated gradient modulation (OGM-GE, Peng et al. CVPR 2022).

| | Temp Anneal | No Temp Anneal |
|---|---|---|
| **No OGM-GE** | R²=0.29, r=0.68 | **R²=0.36, r=0.69** |
| **OGM-GE** | R²=-0.01, r=0.50 | R²=-0.02, r=0.58 |

**Conclusion:** OGM-GE causes ~0.35 R² drop — harmful. Gradient imbalance (PB/HGT >> CT) is natural, not pathological. CT converges early (fewer params). Fusion layer weights showed balanced contribution (34%/40%/26%) despite unequal gradients.

---

## HPO Round 4 — Production Quality
**Date:** 2026-03-19
**Architecture:** 3-branch (PB+HGT+CT), concat+linear fusion
**Framework:** Optuna TPE + MedianPruner
**Trials:** 100 (2 folds, max 100 epochs)
**Data:** 517 subjects, 4,797 genes, 1,000 max cells/type

**Changes from HPO3:**
- 2 folds per trial (more robust estimates)
- max_cells_per_type=1000 (2000 showed no benefit, 2x slower)
- Fixed n_heads=4 (7/10 top-10 in HPO3)
- Seeded with top 5 HPO3 configs via enqueue_trial
- n_hgt_layers expanded to {3, 4}

**Search space (10 HPs):**
| HP | Range |
|---|---|
| lr | [3e-4, 6e-3] loguniform |
| beta | [0.05, 0.8] |
| weight_decay | [1e-6, 1e-4] loguniform |
| selection_temperature | [0.5, 1.6] |
| tau_min | [1.5, 2.0] |
| anneal_epochs | [10, 30] int |
| gene_gate_temp | [0.3, 2.0] |
| dropout | [0.0, 0.3] |
| guide_lr | [1e-3, 0.035] loguniform |
| n_hgt_layers | {3, 4} |

**Results:** 71 completed, 29 pruned. Best trial 89: val_nll=0.5117.

**5-Fold Production (Rank 3 config):**
- R²=0.323 ± 0.091
- Pearson r=0.578 ± 0.073
- Spearman ρ=0.501 ± 0.086
- Sensitivity spread: 0.017 (tight)

---

## Branch Ablation Study (2026-03-20)

6 configs × 5 folds = 30 runs, using HPO4 Rank 3 HPs.

| Config | Branches | R² | Pearson r | Spearman ρ | val_nll |
|---|---|---|---|---|---|
| Full model | PB+HGT+CT | 0.323±0.091 | 0.578±0.073 | 0.501±0.086 | 0.452±0.082 |
| CT only | CT | 0.323±0.088 | 0.576±0.073 | 0.475±0.065 | 0.453±0.085 |
| HGT only | HGT | 0.317±0.090 | 0.569±0.073 | 0.466±0.093 | 0.458±0.081 |
| PB+HGT | PB+HGT | 0.318±0.086 | 0.571±0.076 | 0.470±0.090 | 0.454±0.081 |
| PB+CT | PB+CT | 0.289±0.098 | 0.554±0.102 | 0.452±0.074 | 0.473±0.083 |
| HGT+CT | HGT+CT | 0.299±0.080 | 0.554±0.070 | 0.412±0.060 | 0.469±0.077 |
| PB only | PB | 0.014±0.020 | 0.133±0.083 | 0.105±0.063 | 0.645±0.068 |

**Key findings:**
- CT alone and HGT alone match the full model — fusion adds nothing measurable
- PB alone fails dramatically (R²=0.014)
- PB contributes nothing complementary when combined with CT or HGT
- **Decision:** Remove PB branch, move to 2-branch architecture (HGT+CT)

---

## Baseline Comparison (2026-03-20)

| Method | Input | R² | Pearson r | Spearman ρ |
|---|---|---|---|---|
| **Our model** | 3 views | **0.323±0.091** | **0.578±0.073** | **0.501±0.086** |
| XGBoost (A+C+E) | 148k pseudobulk | 0.186±0.068 | 0.442±0.064 | 0.409±0.057 |
| ABMIL (scVI) | 30-dim | 0.145±0.086 | 0.404±0.109 | 0.403±0.086 |
| ABMIL (raw) | 4797 genes | 0.120±0.132 | 0.418±0.101 | 0.399±0.091 |
| MixMIL | 30-dim scVI | 0.110±0.038 | 0.359±0.072 | 0.344±0.037 |
| Set Transformer | 30-dim scVI | -0.008±0.027 | 0.029±0.175 | 0.000±0.200 |
| scPhase | 4797 genes | -0.059±0.093 | -0.010±0.103 | 0.025±0.122 |

---

## HPO Round 5 — Cross-Attention Fusion
**Date:** 2026-03-20
**Architecture:** 3-branch, cross-attention fusion
**Framework:** Optuna TPE + MedianPruner
**Trials:** 100 (2 folds)

**Purpose:** Test cross-attention fusion as alternative to concat+linear.

**Changes from HPO4:**
- lr range extended lower [1e-4, 6e-3] (fold 3 collapsed at HPO4 best LR)
- dropout extended upper [0.0, 0.4] (90K more params need more regularization)
- weight_decay extended upper [1e-6, 5e-4]
- Added fusion_n_heads {2, 4, 8}
- Warm-started with top 5 HPO4 configs adapted to cross-attention

**Outcome:** Did not outperform concat+linear. Cross-attention fusion's extra parameters didn't help with the small dataset (516 subjects).

---

## HPO Round 6 — 2-Branch Architecture
**Date:** 2026-03-25
**Architecture:** 2-branch (HGT + CellTransformer), concat_normalized fusion
**Framework:** Ray Tune + Optuna TPE + MedianStoppingRule
**Trials:** 120 (2 folds, max 100 epochs)
**GPUs:** 2 parallel
**Data:** 516 subjects, 4,796 genes (fixed CellChatDB parser), 1,000 max cells/type

**Major changes:**
- Removed PseudoBulk branch based on ablation results
- Switched from Optuna (native) to Ray Tune + OptunaSearch (process isolation)
- concat_normalized fusion (LayerNorm per branch before concat)
- Fixed CellChatDB gene parser (4,677 → 4,796 genes)
- Rebuilt data pipeline: merge → preprocess → LIANA → precompute → splits

**Search space (8 HPs):**
| HP | Range |
|---|---|
| lr | [1e-4, 6e-3] loguniform |
| dropout | [0.0, 0.4] |
| weight_decay | [1e-6, 5e-4] loguniform |
| beta | [0.05, 0.8] |
| guide_lr | [1e-3, 0.035] loguniform |
| tau_min | [1.5, 2.0] |
| anneal_epochs | [10, 30] int |
| gene_gate_temp | [0.3, 2.0] |

**Results (incomplete — killed at 22h):**
53/120 terminated, 64 paused, 2 running. Best val_nll=0.4125 (trial 267f19ca, epoch 52).

**Top 10 val_nll:** 0.4125, 0.4136, 0.4305, 0.4325, 0.4330, 0.4372, 0.4386, 0.4429, 0.4476, 0.4503

**Issues encountered:**
1. **Ax + HyperBand incompatibility** — GP never activated because HyperBand prunes before final metrics. All 120 trials used Sobol (random). Switched to BOHB, then to Optuna TPE.
2. **BOHB 1-epoch minimum budget** — Default max_t=100, eta=3 → min rung at 1 epoch (useless with 5-epoch warmup). Fixed to max_t=135.
3. **MedianStoppingRule PAUSED starvation** — `FIFOScheduler.choose_trial_to_run()` always picks PENDING over PAUSED. Paused trials never resume until all configs generated. Root cause: no `max_concurrent_trials` limit. Fix applied for future runs.
4. **Zero-stride tensor crash** — Degenerate subject R5026720 (1 cell) caused empty tensors with stride 0. Ray's zero-copy serialization crashed. Fixed with `.contiguous()`.
5. **22h stall** — After 53 terminated (6h43m), zero new terminations for 15+ hours. Trials resumed, ran a few epochs, then re-paused by MedianStoppingRule. Infinite pause/resume cycle.

**Decision:** Kill HPO6, warm-start HPO7.

**HP patterns from top 10:**
| HP | Top 10 pattern | Signal |
|---|---|---|
| lr | 0.0002-0.0012 (mean 0.0005) | Lower LR preferred |
| dropout | 0.20-0.30 (tight) | Moderate dropout sweet spot |
| weight_decay | 1e-6 to 1.5e-5 | Very light reg (Bayesian prior already regularizes) |
| beta | 0.27-0.79 (mean 0.62) | Higher β-NLL beta |
| guide_lr | No clear signal | Spread across range |
| tau_min | No clear signal | ~1.6-1.9 |
| anneal_epochs | 12-22 (mean 17) | Shorter annealing |
| gene_gate_temp | 0.31-1.18 (mean 0.61) | Low temp = sharper gene selection |

---

## HPO Round 7 — Warm-Start from HPO6
**Date:** 2026-03-26
**Architecture:** 2-branch (HGT + CellTransformer), concat_normalized fusion
**Framework:** Ray Tune + Optuna TPE + MedianStoppingRule
**Trials:** 120 (2 folds)
**GPUs:** 2 parallel

**Changes from HPO6:**
- **Warm-start:** 120 HPO6 trials injected via `OptunaSearch(points_to_evaluate, evaluated_rewards)`. TPE starts with full l(x)/g(x) models from trial 1.
- **Fixed:** `max_concurrent_trials=n_gpus` (no more PAUSED starvation)
- **Fixed:** Full metrics reporting (val_r2, val_pearson_r, val_spearman_rho, val_rmse)
- Same search space as HPO6

**Results:** 120 trials completed. Best val_nll=0.4044.

**5-Fold Production (Rank 3 config, 2-branch):**
- R²=0.323 ± 0.067
- Pearson r=0.575 ± 0.063
- Spearman ρ=0.498 ± 0.076

**Ablation Study (7 configs × 5 folds):**
See `docs/results/2026-03-30-hpo7-ablation-interpretability.md` for full results.

---

## HPO Round 8 — Blocked HVG Experiment
**Date:** 2026-03-31 to 2026-04-01
**Architecture:** 2-branch (HGT + CellTransformer), concat_normalized fusion
**Framework:** Ray Tune + Optuna TPE + MedianStoppingRule
**Trials:** 39 new + 11 warm-start = 50 total (3 folds, interleaved epoch-by-epoch)
**GPUs:** 2 parallel

**Motivation:** Ridge regression on flattened pseudobulk achieved R²=0.290 — only +0.033 below our model. Investigation revealed the random 100K-cell subsample for HVG selection was biased toward dominant cell types (upper-layer IT neurons = 25% of cells). Rare types contributed almost nothing to gene variance estimates.

**Changes from HPO7:**
- **Blocked HVG:** scran-style per-cell-type variance with loess normalization (4,135 genes, down from 4,796)
- **Holdout restored:** test_frac=0.1 (52 subjects held out)
- **3-fold HPO** (up from 2-fold) for more robust estimates
- **Fixed d_embed=64, n_hgt_layers=2, n_inducing=64** (converged in HPO1-4, not re-searched)
- **Interleaved fold-swap training:** manual training loop replaces Lightning trainer.fit(). One model on GPU at a time, K-1 fold states in CPU RAM. Reports mean val_nll per epoch to Ray for correct MedianStoppingRule pruning.
- **Weight decay param groups:** biases, LayerNorm, gate_logits excluded from weight decay (Loshchilov & Hutter 2019)
- **Temperature annealing fixed:** gate attribute names were wrong in prior HPO code (gene_gate_hgt → hgt_gene_gate, gene_gate_ct → cell_transformer.gene_gate). All prior HPO rounds had broken temperature annealing during HPO trials.

**Search space (7 HPs):**
| HP | Range |
|---|---|
| lr | [5e-5, 5e-3] loguniform |
| dropout | [0.05, 0.4] |
| beta | [0.1, 1.0] |
| weight_decay | [1e-6, 1e-3] loguniform |
| guide_lr | [5e-4, 0.05] loguniform |
| anneal_epochs | [8, 30] int |
| gene_gate_temp | [0.3, 2.0] |

**Status:** Running in tmux `pipeline` session.

---

## Architecture Timeline

```
HPO1-4 (Mar 8-19):  3-branch (PB + HGT + CT), concat+linear fusion
Ablation (Mar 20):   PB shown useless (R²=0.014 alone, no contribution in combination)
HPO5 (Mar 20):       3-branch, cross-attention fusion (didn't outperform concat)
HPO6 (Mar 25):       2-branch (HGT + CT), concat_normalized fusion
HPO7 (Mar 26):       2-branch (HGT + CT), concat_normalized fusion (warm-start)
HPO8 (Mar 31):       2-branch (HGT + CT), blocked HVG, holdout restored
```

```
HPO1-4 (Mar 8-19):  3-branch (PB + HGT + CT), concat+linear fusion
Ablation (Mar 20):   PB shown useless (R²=0.014 alone, no contribution in combination)
HPO5 (Mar 20):       3-branch, cross-attention fusion (didn't outperform concat)
HPO6 (Mar 25):       2-branch (HGT + CT), concat_normalized fusion
HPO7 (Mar 26):       2-branch (HGT + CT), concat_normalized fusion (warm-start)
```

## Framework Timeline

```
HPO1-5 (Mar 8-20):   Optuna (native) + MedianPruner, JournalFileBackend
HPO6 (Mar 25):        Ray Tune + OptunaSearch + MedianStoppingRule
HPO7 (Mar 26):        Ray Tune + OptunaSearch + MedianStoppingRule (warm-start)
HPO8 (Mar 31):        Ray Tune + OptunaSearch + MedianStoppingRule (1 seed, blocked HVG)
```
