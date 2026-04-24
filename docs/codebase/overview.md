# ResDec-MHE Project Overview

Comprehensive map of the repository: what lives where, how the model is structured, how worktrees + archives are used during development. Companion to the more focused docs in this directory (`data-formats.md`, `model-contracts.md`, `pipeline.md`, `scripts-inventory.md`).

## 1. Project goal

Continuous regression of cognitive resilience from snRNA-seq (ROSMAP cohort, N=516 subjects, 31 cell types, 4785 highly variable genes). Target audience: biology venues (Cell Reports Methods / Genome Biology / Bioinformatics).

The canonical model is **ResDec-MHE** (Residual-Decomposition + Multi-Head Ensemble). 5-fold mean R² = 0.4436 ± 0.10, beating TabPFN-2.6 standalone (0.399), XGBoost (0.358), MixMIL (0.157), scPhase (-0.074), and all other baselines tracked in `outputs/redesign/interpretability/paper_baseline_table.{csv,md}`.

## 2. Repository layout (top level)

```
proj_ml_snrna/
├── src/                  # Production source (importable as `src.*`)
├── scripts/              # CLI entrypoints + orchestration
├── configs/              # OmegaConf YAMLs (default + per-experiment)
├── tests/                # pytest tree (unit, integration, regression, smoke, negative)
├── baselines/            # Vendored or adapted external baselines
├── data/                 # Source AnnData + precomputed caches  (gitignored)
├── outputs/              # All training/inference/analysis outputs (gitignored)
├── docs/                 # Plans, decisions, codebase knowledge, results
├── notebooks/            # Exploratory analysis (not on critical path)
└── .worktrees/           # Isolated feature-branch workspaces (gitignored)
```

`/data` and `/outputs` are gitignored at the worktree level (each worktree symlinks them to the canonical paths in the main repo). `.worktrees/` itself is gitignored.

## 3. Source code (`src/`)

```
src/
├── analysis/             # Post-hoc analyses (variance decomp, subgroup R², CCC, etc.)
├── data/                 # Datasets, datamodule, collate, splits, AnnData loader
├── inference/            # Re-inference + prediction utilities
├── models/               # Encoder + heads
│   └── resdec_head/      # Residual-decomposition head (canonical)
├── training/             # Lightning module, callbacks, optimizers
├── utils/                # Shared helpers
└── visualization/        # Figure-drawing primitives
```

The encoder lives at `src/models/full_model.py::CognitiveResilienceModel`. The canonical head is `src/models/resdec_head/resdec_mhe_head.py::ResDecMHEHead`. Lightning glue is `src/training/resdec_lightning_module.py`. Datamodule + dataset is `src/data/{datamodule,datasets,collate}.py`.

`src/analysis/` modules are pure functions (decomposition + CIs + correlations + edge attention extraction). Orchestration that loads from disk and writes JSONs lives under `scripts/resdec_mhe/interpretability/`.

## 4. Scripts (`scripts/`)

```
scripts/
├── resdec_mhe/           # ResDec-MHE-specific orchestration
│   ├── training/         # train.py, run_5fold_parallel.sh, run_seed_variation.sh, reinfer
│   ├── interpretability/ # Phase B/C analyses + paper figures + baseline-table aggregator
│   └── tabpfn/           # OOF + outer TabPFN-2.6 cache builders
├── analysis/             # Cross-experiment aggregation + plots
├── data/                 # Data prep + cache rebuilders
├── inference/            # Standalone inference entrypoints
├── training/             # Legacy training drivers (HPO, sensitivity sweeps)
├── profiling/            # Performance probes
└── utils/                # Misc helpers
```

Use `scripts/resdec_mhe/training/run_5fold_parallel.sh` to run a single 5-fold training across both GPUs. Use `scripts/resdec_mhe/training/run_seed_variation.sh` for multi-seed runs (loops the 5-fold driver per seed).

The paper-table aggregator `scripts/resdec_mhe/interpretability/make_baseline_table.py` collects per-fold metrics from every baseline + ablation and emits `outputs/redesign/interpretability/paper_baseline_table.{csv,md,provenance.json}`. Figures are drawn by `scripts/resdec_mhe/interpretability/make_figures.py`.

## 5. Configs (`configs/`)

```
configs/
├── default.yaml          # Base config (everything inherits from this)
├── resdec_mhe/
│   ├── canonical.yaml    # Locked Phase 5 canonical (n_stages=1, k_tabm=8, vanilla MHA)
│   └── ablations/        # Per-ablation overrides (top-k, z-score, no FiLM, etc.)
├── hpo_round{6,7}.yaml   # Older HPO sweep configs
├── MapMyCells/           # Cell-type annotation configs
└── archived/             # Superseded configs kept for traceability
```

`configs/resdec_mhe/canonical.yaml` is what every Phase 5 run starts from. Ablations override one or two keys via OmegaConf merge.

## 6. Tests (`tests/`)

```
tests/
├── unit/                 # Per-module tests (data, models, training, analysis, ...)
│   ├── baselines/        # Tests for baselines/ adapters (e.g. scPhase summary writer)
│   ├── resdec_mhe/       # Tests for scripts/resdec_mhe/ orchestration
│   ├── models/resdec_head/  # Tests for ResDecMHE head
│   └── archive/          # Tests for code that's been moved to archive
├── integration/          # End-to-end pipeline smoke tests
├── regression/           # Numerical regression checks (frozen targets)
├── smoke/                # Quickest possible "does it import + run" checks
└── negative/             # Tests that verify failure modes
```

Test runner: `uv run pytest tests/...`. Per project rule: only run targeted tests for the files changed during development; full suite (`pytest tests/`) is reserved for the very end of a work session as one final regression check.

## 7. Baselines (`baselines/`)

```
baselines/
├── shared/               # Shared input AnnData (mixmil_input.h5ad, scphase_input.h5ad)
├── mixmil/               # MixMIL adapter (run_rosmap.py + .venv + repo/)
├── scPhase/              # scPhase adapter (run_rosmap.py + summary_canonical.py + repo/)
├── cloudpred/            # CloudPred (already produces results.csv directly)
├── gpio/                 # GPIO baseline
├── perceiver_io/         # Perceiver-IO baseline
├── set_transformer/      # Set-Transformer baseline
├── abmil/                # ABMIL baseline
├── prepare_data.py       # Builds shared input AnnDatas from canonical splits
├── rebuild_and_run.sh    # Full-cohort rebuild + sequential mixmil/scphase run
└── tabpfn/               # TabPFN-2.6 standalone baseline (under scripts/resdec_mhe/tabpfn/)
```

Each baseline directory has its own vendored repo + venv (gitignored) so the heavyweight conda/pip stacks of MixMIL, scPhase, etc. don't pollute the main `.venv`. `summary_canonical.py` bridges scPhase's idiosyncratic output schema into the canonical paper-aggregator schema.

## 8. Data + Outputs

Both gitignored. Canonical paths:

- **Raw AnnData**: `data/snRNAseq/adata_ROSMAP_preprocessed.h5ad` (~70 GB, 516 subjects, 31 cell types, 4785 HVGs)
- **Splits**: `outputs/splits.json` (5-fold + holdout)
- **Precomputed cache**: `data/precomputed/precomputed_dataset.pt` (per-subject pseudobulk + cell metadata, faster than loading AnnData every fold)
- **TabPFN caches**: `data/redesign/tabpfn_outer_fold{0..4}.npz` + `tabpfn_oof_fold{0..4}.npz` (per-fold predictions used as the residual base)
- **Per-experiment outputs**: `outputs/redesign/p5_*/` (each ablation/seed/canonical run gets its own dir)
- **Paper artifacts**: `outputs/redesign/interpretability/` (CSV/MD tables, figures, JSONs, GSEA, Captum, CCC)

Worktrees keep `data/` and `outputs/` as symlinks pointing back at the main repo paths so a single training set + outputs tree is shared across all worktrees.

## 9. ResDec-MHE model architecture

```
Input subject (cells × genes)
        │
        ▼
┌───────────────────────────────────┐
│ Encoder: CognitiveResilienceModel │
│   ├─ HGT (heterogeneous graph)    │  ← src/models/full_model.py
│   ├─ CellTransformer              │
│   ├─ PMA (set→token)              │
│   ├─ RegionHandler (6 scalars)    │
│   ├─ PathologyAttention           │
│   └─ gene_gate                    │
│   → z ∈ ℝ^64                      │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Head: ResDecMHEHead               │  ← src/models/resdec_head/
│   n_stages=1, k_tabm=8            │
│   use_diff_attn=False (vanilla)   │
│   use_hyper_conn=True             │
│   use_film=True (zero metadata)   │
│   → f̂_1 (residual)                │
└───────────────────────────────────┘
        │
        ▼
ŷ = ŷ_tabpfn_outer + f̂_1
```

**Residual-decomposition target**: `y_residual = y_true - ŷ_tabpfn_oof` (TabPFN trained on inner-OOF predictions per fold). The encoder + head learns to predict the residual; the final composite is the outer-fold TabPFN prediction plus the learned residual.

**Why this works**: TabPFN gives a strong tabular prior (R²≈0.40), and the deep model only has to learn the gap (~0.04 absolute). Ablating the TabPFN base drops R² from 0.444 to 0.266. FiLM with real metadata HURTS (0.4333 vs canonical 0.4436), so canonical FiLM uses zero metadata vector — kept structurally for symmetry.

## 10. Worktree strategy

Long-running feature work happens in isolated git worktrees under `.worktrees/`. This keeps the main repo's working tree clean for ad-hoc commands while lengthy refactors / ablations / experiments run elsewhere on their own branch.

```
.worktrees/
├── refactor-canonical-naming/    # Branch refactor/canonical-naming (now merged)
├── redesign-resdec-h3/           # Branch redesign/resdec-h3 (now merged)
└── investigation-beat-baselines/ # Removed 2026-04-23 (purpose closed)
```

Each worktree is bootstrapped by:
1. `git worktree add .worktrees/<name> <branch>`
2. `ln -s ../../data .worktrees/<name>/data`
3. `ln -s ../../outputs .worktrees/<name>/outputs`

After the branch is merged back to its parent + master, the worktree is removed via `git worktree remove .worktrees/<name>`.

## 11. Archive + rollback strategy

Before any disruptive refactor or experiment, the project state is preserved at three levels:

1. **Git tag** at the pre-change HEAD (e.g., `pre-refactor-2026-04-23`, `pre-cleanup-phase5`, `investigation-closed-2026-04-23`). Lets `git checkout <tag>` return to that exact state.
2. **Tarball** of the relevant `outputs/` subtrees under `outputs/archive/` (e.g., `pre-refactor-2026-04-23-outputs-redesign.tar.gz`, `scphase-oom-2026-04-23.tar.gz`). For when the refactor changes filenames or directory layouts and the historical artifacts need to be readable as-is.
3. **MEMORY.md notes** describing what changed and why, with absolute dates so future reads stay interpretable.

Archive tarballs live at `outputs/archive/` (gitignored). Tags live in the git refs of the main repo.

## 12. Key tags + commits (current as of 2026-04-23)

- `pre-refactor-2026-04-23` — HEAD before the canonical-naming refactor began.
- `pre-cleanup-phase5` — HEAD before the dead-code cleanup.
- `investigation-closed-2026-04-23` — HEAD of the now-removed beat-baselines investigation worktree (preserved for reference; branch still in repo).

Active branches:
- `master` — canonical state. After merging `redesign/resdec-h3` (2026-04-23), this contains everything.
- `redesign/resdec-h3` — Phase 5 work; merged to master and ready to be cleaned up.
- `refactor/canonical-naming` — refactor + cleanup; merged to redesign + master and ready to be cleaned up.
- `archive/2026-03-31-pre-hvg-experiment`, `archive/pre-squash-2026-03-27` — historical references.
- `investigation/beat-baselines` — closed investigation (worktree removed; branch retained for traceability).

## 13. Where to look next

- **Architecture details**: `docs/plans/2026-01-13-cognitive-resilience-model-design-part1-architecture.md`
- **Training/operations**: `docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md`
- **Phase 5 finish plan**: `docs/plans/2026-04-22-resdec-h3-phase5-finish.md`
- **Region handler design**: `docs/plans/2026-01-27-region-handler-design.md`
- **Data formats**: `docs/codebase/data-formats.md`
- **Model contracts**: `docs/codebase/model-contracts.md`
- **Pipeline flow**: `docs/codebase/pipeline.md`
- **Script inventory**: `docs/codebase/scripts-inventory.md`
- **Paper baseline table**: `outputs/redesign/interpretability/paper_baseline_table.md`
- **Field-gap analysis**: `docs/field-gap-analysis.md`
- **Prior art + baselines**: `docs/prior-art-and-baselines.md`
