# ResDec-H3 Phase 5 Finish — Interpretability + R²-ablations + Paper Prep

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Close out ResDec-H3 P5 evaluation by (1) wiring real metadata into FiLM and testing its effect, (2) building the missing interpretability analyses that are the paper's real contribution, (3) running R²-ablations for TabPFN input (top-k, z-score), (4) compiling paper-ready baseline table + figures.

**Architecture:** No architectural changes; this is completion work on top of the locked canonical (`n_stages=1 + TabM(k=8) + vanilla MHA + HyperConn + FiLM`, commit `9c1748e`). New analysis modules follow the existing `src/analysis/*.py` pattern; orchestration scripts live in a new `scripts/redesign/interpretability/` subdir (chosen for separation from training/orchestration as codebase grows).

**Tech Stack:** PyTorch · Lightning · Captum IG · gseapy · scipy.stats · TabPFN-2.6 · matplotlib/seaborn · uv for all Python invocations.

**Status before this plan (verified 2026-04-22):**
- Canonical 5-fold R² = 0.4436 ± 0.089 (seed 42; `outputs/redesign/p5_canonical_seed42/`).
- 9-variant Phase 5.3 ablation matrix complete (all dirs on disk).
- Captum IG composite script written but NOT RUN.
- Interpretability outputs (residual phenotype, attention extraction, head specialization) are STALE — run on the old canonical (with DiffAttn).
- FiLM receives zero metadata in canonical; `_get_metadata()` has a TODO(phase4) placeholder.
- `src/data/tabpfn_input.py::load_metadata_vector` exists and returns an 8-dim APOE/sex/age vector.

---

## Phase A — FiLM metadata wire-up

### Task A.1: Wire real metadata into datamodule → lightning module

**Files:**
- Modify: `src/data/datasets.py` — add `metadata` field to per-subject `__getitem__` return dict (call `load_metadata_vector(subject_id, meta_csv, age_mean, age_std)`). Pass `age_mean`/`age_std` through via dataset init from fold-train-set stats.
- Modify: `src/data/datamodule.py` — compute train-fold age_mean/age_std once at `setup()` and pass to datasets. DO NOT compute on val subjects (leakage).
- Modify: `src/data/collate.py` — stack per-subject `metadata` tensors into `[B, 8]` batch tensor.
- Modify: `src/training/resdec_lightning_module.py:283-301` — replace TODO placeholder with actual path that reads `batch["metadata"]`; keep the `md is None → zeros` fallback for robustness; drop the TODO comment.
- Test: `tests/unit/data/test_metadata_wiring.py` (new file)

**Step 1: Write the failing test** — three cases: (a) present subject produces non-zero 8-dim vector; (b) missing subject produces missingness-bit-only vector; (c) train/val age stats come from TRAIN ONLY (leakage check).

```python
# tests/unit/data/test_metadata_wiring.py
import numpy as np, pandas as pd, torch
from pathlib import Path
from src.data.tabpfn_input import load_metadata_vector

def test_load_metadata_vector_present_subject(tmp_path):
    df = pd.DataFrame({
        "ROSMAP_IndividualID": ["R0000001"], "apoe_genotype": [34],
        "msex": [1], "age_death": [86.0],
    })
    csv = tmp_path / "metadata.csv"; df.to_csv(csv, index=False)
    vec, fields = load_metadata_vector("R0000001", csv, age_mean=86.0, age_std=6.5)
    assert vec.shape == (8,)
    # APOE 34 → e3 and e4 present
    assert vec[1].item() == 1.0  # e3
    assert vec[2].item() == 1.0  # e4
    assert vec[0].item() == 0.0  # e2 absent
    assert vec[3].item() == 0.0  # apoe not missing
    assert vec[4].item() == 1.0  # sex=1
    assert vec[6].item() == pytest.approx(0.0)  # age z-scored at mean

def test_load_metadata_vector_missing_subject(tmp_path):
    df = pd.DataFrame({
        "ROSMAP_IndividualID": ["R9999"], "apoe_genotype": [33],
        "msex": [0], "age_death": [80.0],
    })
    csv = tmp_path / "metadata.csv"; df.to_csv(csv, index=False)
    vec, _ = load_metadata_vector("R0000001", csv)
    # Subject not in metadata: only missingness bits set
    assert vec[3].item() == 1.0 and vec[5].item() == 1.0 and vec[7].item() == 1.0
    assert vec[0:3].sum().item() == 0.0

def test_datamodule_age_stats_train_only(tmp_path):
    """Datamodule must compute age_mean/std from train split only."""
    # ... build tiny metadata with distinguishable train/val ages;
    # assert dataset.age_mean matches train-only mean, not pooled mean
```

**Step 2: Run test to verify it fails**

```
uv run pytest tests/unit/data/test_metadata_wiring.py -v
```
Expected: tests (a) and (b) should PASS out of the box (`load_metadata_vector` already implemented); test (c) FAILS because datamodule doesn't yet pass train-only stats.

**Step 3: Implement the wiring**

`src/data/datamodule.py` — in `setup("fit")`, after splitting train/val subjects, compute `train_age_mean = metadata.loc[metadata.ROSMAP_IndividualID.isin(train_ids), "age_death"].dropna().mean()` (and std). Pass into `CognitiveResilienceDataset(... age_mean=train_age_mean, age_std=train_age_std)`.

`src/data/datasets.py` — accept `age_mean`, `age_std`, `meta_csv` in `__init__`; in `__getitem__`, call `load_metadata_vector(subject_id, meta_csv, age_mean, age_std)` and add to the returned dict under key `metadata`.

`src/data/collate.py` — add `"metadata"` to the list of keys stacked via `torch.stack`; shape `[B, 8]`.

`src/training/resdec_lightning_module.py:283-301` — simplify `_get_metadata` to just return `batch["metadata"]` (it's now always present); keep the `None → zeros` fallback only for tests that don't pass metadata. Remove `TODO(phase4)` comment.

**Step 4: Run tests** — all three pass. Also run existing training-module tests:

```
uv run pytest tests/unit/training/test_resdec_lightning_module*.py tests/unit/data/test_metadata_wiring.py -x -q
```

**Step 5: Commit**

```
git add -u src/data/datamodule.py src/data/datasets.py src/data/collate.py \
           src/training/resdec_lightning_module.py \
           tests/unit/data/test_metadata_wiring.py
git commit -m "+ FiLM metadata wire-up: APOE+sex+age from load_metadata_vector, train-fold age stats"
```

---

### Task A.2: FiLM fold-0 pilot

**Files:**
- No code changes — just launch.
- Output dir: `outputs/redesign/p5_filmwired_pilot_fold0/`

**Step 1: Launch fold-0 training with wired metadata**

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 uv run python scripts/redesign/train_resdec.py \
    --config configs/redesign/p5_phase2_residual.yaml \
    --fold 0 --seed 42 \
    --output-dir outputs/redesign/p5_filmwired_pilot_fold0/fold0
```

**Step 2: Reinfer best ckpt**

```bash
CONFIG=configs/redesign/p5_phase2_residual.yaml \
OUTROOT=outputs/redesign/p5_filmwired_pilot_fold0 \
TABPFN_DIR=data/redesign \
PYTHONPATH=. uv run python scripts/redesign/reinfer_best_ckpt.py \
    --config $CONFIG --fold 0 --pred-root $OUTROOT --tabpfn-dir $TABPFN_DIR
```

**Step 3: Compare to canonical fold-0**

Canonical fold-0 R² = 0.4842 (verified from `outputs/redesign/p5_canonical_seed42/best_vs_tabpfn_summary.json`). Pilot fold-0 must be compared numerically.

**Decision gate:**
- If pilot R² − canonical R² > +0.01 → proceed to Task A.3 (full 5-fold)
- If within ±0.01 → FiLM is noise-level at real metadata too; SKIP A.3, document in paper as "FiLM with APOE/sex/age conditioning did not improve R²; retained for architectural consistency"
- If pilot R² − canonical R² < −0.01 → metadata hurts (possible if small-N overfitting on 8 extra dims); SKIP A.3, document as "FiLM marginally hurts; evaluated at zero metadata in final"

**Step 4: Commit** (regardless of outcome)

```
git add outputs/redesign/p5_filmwired_pilot_fold0/fold0/best_summary.json
git commit -m "+ FiLM fold-0 pilot: real metadata, R²=<value>"
```

---

### Task A.3: FiLM 5-fold (conditional on A.2 passing gate)

**Files:**
- No code changes.
- Output dir: `outputs/redesign/p5_canonical_filmwired_seed42/`

**Step 1: Launch 5-fold via existing parallel driver**

```bash
CONFIG=configs/redesign/p5_phase2_residual.yaml \
OUTROOT=outputs/redesign/p5_canonical_filmwired_seed42 \
MAX_EPOCHS=60 SEED=42 RUN_REINFER=1 \
  bash scripts/redesign/run_phase2_5fold_parallel.sh
```

**Step 2: Verify all 5 folds completed**

```bash
for f in 0 1 2 3 4; do
  test -f outputs/redesign/p5_canonical_filmwired_seed42/fold${f}/val_predictions_best.npz \
    && echo "fold${f} OK" || echo "fold${f} MISSING"
done
```

**Step 3: If 5-fold mean R² > canonical R² by > 0.01 → promote to final canonical**

Update `configs/redesign/p5_phase2_residual.yaml` header comment to reflect the new canonical numbers.

**Step 4: Commit**

```
git add outputs/redesign/p5_canonical_filmwired_seed42/best_vs_tabpfn_summary.json \
        configs/redesign/p5_phase2_residual.yaml
git commit -m "+ FiLM-wired canonical 5-fold: R²=<value> ± <std> (vs <baseline>)"
```

---

## Phase B — Directory cleanup + stale interpretability re-runs

### Task B.0: Move existing interpretability scripts into subdir

**Files:**
- Move:
  - `scripts/redesign/captum_composite_attribution.py` → `scripts/redesign/interpretability/captum_composite_attribution.py`
  - `scripts/redesign/resilience_residual_phenotype.py` → `scripts/redesign/interpretability/resilience_residual_phenotype.py`
  - `scripts/redesign/extract_pathology_attention.py` → `scripts/redesign/interpretability/extract_pathology_attention.py`
  - `scripts/redesign/analyze_pathology_attention_heads.py` → `scripts/redesign/interpretability/analyze_pathology_attention_heads.py`
- Create: `scripts/redesign/interpretability/__init__.py` (empty)

**Step 1: `git mv` each file**

```bash
mkdir -p scripts/redesign/interpretability
git mv scripts/redesign/captum_composite_attribution.py scripts/redesign/interpretability/
git mv scripts/redesign/resilience_residual_phenotype.py scripts/redesign/interpretability/
git mv scripts/redesign/extract_pathology_attention.py scripts/redesign/interpretability/
git mv scripts/redesign/analyze_pathology_attention_heads.py scripts/redesign/interpretability/
touch scripts/redesign/interpretability/__init__.py
```

**Step 2: Update docstring usage examples in each moved file** to reflect the new path.

**Step 3: Commit**

```
git add -u scripts/redesign/ scripts/redesign/interpretability/__init__.py
git commit -m "+ move interpretability scripts into scripts/redesign/interpretability/ (2C org)"
```

---

### Task B.1: Re-run residual phenotype on canonical

Use the post-FiLM canonical from A.3 if available, else `outputs/redesign/p5_canonical_seed42/`.

```bash
PYTHONPATH=. uv run python scripts/redesign/interpretability/resilience_residual_phenotype.py \
    --pred-root outputs/redesign/<final_canonical_dir>
```

Outputs land in `outputs/redesign/interpretability/` (overwrites stale). ~3 min, CPU.

**Commit:** `+ re-run residual phenotype on final canonical`

---

### Task B.2: Re-run pathology attention extraction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python \
    scripts/redesign/interpretability/extract_pathology_attention.py \
    --pred-root outputs/redesign/<final_canonical_dir>
```

~5 min, 1 GPU.

**Commit:** `+ re-extract pathology attention on final canonical`

---

### Task B.3: Re-run head specialization analysis

```bash
PYTHONPATH=. uv run python scripts/redesign/interpretability/analyze_pathology_attention_heads.py
```

~2 min, CPU (reads `pathology_attention_per_subject.npz` from B.2).

**Commit:** `+ re-run head specialization + Splatter deep-dive on final canonical`

---

## Phase C — New interpretability modules (TDD)

All new `src/analysis/` modules must have unit tests with synthetic data. All orchestration scripts live in `scripts/redesign/interpretability/`.

### Task C.1: Variance decomposition

**What it computes:** `Var(y) = Var(ŷ_tabpfn) + Var(f̂_1) + 2·Cov(ŷ_tabpfn, f̂_1) + Var(resid)`, reported globally and per subgroup (APOE-ε4 count, sex, age-quartile).

**Files:**
- Create: `src/analysis/resdec_variance_decomposition.py` — pure-function `decompose_variance(y_true, y_tabpfn, f1_residual, *, subgroups=None) -> dict`.
- Create: `scripts/redesign/interpretability/variance_decomposition.py` — orchestration: loads per-fold predictions from `val_predictions_best.npz` + TabPFN outer from `data/redesign/`, joins with metadata, calls the analysis function, writes `outputs/redesign/interpretability/variance_decomposition.json`.
- Test: `tests/unit/analysis/test_resdec_variance_decomposition.py`

**Step 1: Write failing test** — two cases: (a) identity case where `f1=0` (prediction == TabPFN) should give `Var(resid) == Var(y - y_tabpfn)`; (b) known-covariance case with crafted arrays.

**Step 2: Run to verify FAIL** (module doesn't exist).

**Step 3: Implement** the decomposition. Use `np.var(..., ddof=1)` for unbiased sample variance. Subgroup handling: dict of masks.

**Step 4: Run tests PASS.**

**Step 5: Implement orchestration script** (load predictions, compose, write JSON).

**Step 6: Run orchestration script end-to-end on canonical predictions; verify JSON keys include `global`, `by_apoe_e4_count`, `by_msex`, `by_age_quartile`, each with `var_y`, `var_tabpfn`, `var_f1`, `cov_tabpfn_f1`, `var_resid`, `total_explained_fraction`**.

**Step 7: Commit**

```
git commit -m "+ resdec variance decomposition: Var(y) split into TabPFN, f_1, covariance, residual (+ subgroup)"
```

---

### Task C.2: Subgroup R² stratification

**What it computes:** per-subgroup (APOE-ε4 count, sex, age-quartile, pathology-quartile) R², RMSE, Pearson, Spearman of composite predictions vs `y_true`. Bootstrap 1000× within each subgroup.

**Files:**
- Create: `src/analysis/resdec_subgroup_analysis.py` — `stratified_metrics(y_true, y_pred, subgroup_masks, *, n_bootstrap=1000, seed=42) -> dict`.
- Create: `scripts/redesign/interpretability/subgroup_r2.py` — loads composite predictions + metadata, runs, dumps `outputs/redesign/interpretability/subgroup_metrics.json`.
- Test: `tests/unit/analysis/test_resdec_subgroup_analysis.py`

**Step 1: Write failing test** — three cases: (a) trivial single-subgroup case matches overall metric; (b) two-subgroup case returns two different R²s; (c) bootstrap CIs are wider with smaller subgroup n.

**Step 2–4:** TDD as above.

**Step 5: Orchestration script.** APOE-ε4 count groups from `apoe_genotype`; sex from `msex`; age-quartile from `age_death`; pathology-quartile from `gpath`. Output JSON + `subgroup_metrics_table.csv`.

**Step 6: Run on canonical predictions.**

**Step 7: Commit**

```
git commit -m "+ resdec subgroup R²: APOE/sex/age/pathology stratified metrics with bootstrap CIs"
```

---

### Task C.3: Statistical rigor (paired tests + bootstrap CI + calibration)

**What it computes:**
1. Paired Wilcoxon signed-rank (per-fold R² of ours vs each baseline); one-sided, `alternative="greater"`. n=5 per comparison (under-powered but required for honesty).
2. Bootstrap 95% CI on global R² (resample 516 subjects 1000×).
3. Calibration plot: residual vs TabPFN σ (σ_tabpfn from inner-OOF cache as proxy for outer σ; or extract from TabPFN-2.6 outer cache keys).

**Files:**
- Create: `src/analysis/resdec_statistical_rigor.py` — functions `paired_wilcoxon(fold_r2s_ours, fold_r2s_baseline, alternative="greater") -> dict`, `bootstrap_r2_ci(y_true, y_pred, n_boot=1000, conf=0.95, seed=42) -> dict`, `calibration_coverage(y_true, y_pred, sigma, nominal=[0.5, 0.68, 0.8, 0.95]) -> dict`.
- Create: `scripts/redesign/interpretability/paired_tests_and_bootstrap.py` — orchestration: loads per-fold R² for ours + each baseline (XGBoost/TabPFN-2.6 standalone/MixMIL/scPhase), computes all three, writes `outputs/redesign/interpretability/statistical_rigor.json` + a paper-ready markdown table.
- Test: `tests/unit/analysis/test_resdec_statistical_rigor.py`

**Step 1: Write failing tests** — (a) paired Wilcoxon with identical arrays → p=1.0; (b) bootstrap CI contains true R² on synthetic data; (c) well-calibrated Gaussian residuals hit nominal coverage.

**Steps 2–7:** TDD + orchestration + commit.

```
git commit -m "+ resdec statistical rigor: paired Wilcoxon + bootstrap R² CI + calibration coverage"
```

---

### Task C.4: CCC interpretability

**What it computes:**
1. HGT edge attention per subject: extract per-edge-type attention weights from the encoder's HGT layers by hooking `CognitiveResilienceModel`'s HGT forward.
2. Per-edge-type ablation: predict with each edge type zero'd out, measure R² drop.
3. Correlation against LIANA CCC scores (exist at `data/precomputed/liana_scores.csv` or similar — verify path during implementation).

**Files:**
- Create: `src/analysis/resdec_ccc_importance.py` — functions `extract_hgt_edge_attention(model, batch) -> dict`, `per_edge_type_ablation(lit_module, dataloader, edge_types, device) -> dict`, `liana_correlation(our_importance, liana_df) -> dict`.
- Create: `scripts/redesign/interpretability/ccc_composite_attribution.py` — 5-fold orchestration.
- Test: `tests/unit/analysis/test_resdec_ccc_importance.py` (synthetic forward hook test; full ablation too slow for unit).

**Step 1: Investigate before coding** — read `src/models/full_model.py` HGT section, identify hook-able layers, verify LIANA data location.

**Step 2–7:** TDD for the deterministic pieces (extraction API, correlation function); manual verification for ablation.

Time: ~3-4 hr. This is the biggest new module; may need to split into C.4a/C.4b if needed.

```
git commit -m "+ resdec CCC importance: HGT edge attention + per-edge-type ablation + LIANA correlation"
```

---

### Task C.5: Captum IG composite attribution on final canonical

**Files:**
- No code changes; script already exists at `scripts/redesign/interpretability/captum_composite_attribution.py` (post-B.0 rename).

**Step 1: Launch** — output into `outputs/redesign/interpretability/captum_ig/` (not the base `interpretability/` dir, to avoid mixing with phenotype/attention files):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python \
    scripts/redesign/interpretability/captum_composite_attribution.py \
    --pred-root outputs/redesign/<final_canonical_dir> \
    --out-dir outputs/redesign/interpretability/captum_ig \
    --n-steps 50 --internal-batch-size 4
```

Expected runtime: 30-60 min (5 folds × ~7-12 min/fold).

**Step 2: Verify outputs** — check `composite_attributions.npz`, `composite_attribution_summary.json`, `top_pairs_table.csv` exist; sanity-check top-10 cell-type × gene pairs.

**Step 3: Commit**

```
git add outputs/redesign/interpretability/captum_ig/composite_attribution_summary.json \
        outputs/redesign/interpretability/captum_ig/top_pairs_table.csv
git commit -m "+ Captum IG composite attribution on final canonical (5-fold)"
```

*(Note: do NOT commit the full `.npz` — ~300 MB; gitignore it.)*

---

### Task C.6: GSEA adapter from Captum output

**Files:**
- Create: `scripts/redesign/interpretability/gsea_from_captum.py` — reads `composite_attributions.npz` + gene-name list; for each top-attributed gene set (top-200 global, top-50 per-cell-type), runs `gseapy.enrichr` against Hallmark / Reactome / KEGG / AD-GWAS-hits (Bellenguez 2022 + Wightman 2021, manually curated as small gene list).
- Reuses: `src/analysis/gene_enrichment.py` (existing).

**Step 1: Verify gene names loadable** — check `data/precomputed/gene_names.json` or equivalent.

**Step 2: Implement adapter** — thin wrapper that builds gene-list from Captum summary + calls existing enrichment code.

**Step 3: Run**:

```bash
PYTHONPATH=. uv run python scripts/redesign/interpretability/gsea_from_captum.py \
    --captum-npz outputs/redesign/interpretability/captum_ig/composite_attributions.npz \
    --out-dir outputs/redesign/interpretability/gsea
```

**Step 4: Commit**

```
git commit -m "+ GSEA adapter: Captum top-attributed genes → Hallmark/Reactome/KEGG + AD GWAS overlap"
```

---

## Phase D — R²-ablations (track II, concurrent with Phase C)

All three ablations follow the same pattern: adjust TabPFN input, recompute OOF + outer caches, retrain 5-fold canonical with new residual targets, reinfer, dump JSON.

### Task D.1: top-k=1000 TabPFN feature ablation

**Files:**
- No code changes; `compute_top_k_features.py` already parameterizes `--top-k`.

**Step 1: Compute top-1000 features for 5 folds:**

```bash
PYTHONPATH=. uv run python scripts/redesign/compute_top_k_features.py \
    --top-k 1000 --feature-set A \
    --splits-path outputs/splits.json \
    --precomputed-dir data/precomputed \
    --metadata-csv data/metadata_ROSMAP/metadata.csv \
    --output-dir data/redesign/top_k_variants/k1000
```

**Step 2: Recompute TabPFN OOF + outer with k=1000** (modify CLI flags to point at new top-k file):

```bash
TOP_K=1000 TOP_K_DIR=data/redesign/top_k_variants/k1000 \
  PYTHONPATH=. uv run python scripts/redesign/compute_tabpfn_oof.py \
    --top-k-dir $TOP_K_DIR --output-dir data/redesign/top_k_variants/k1000 [...]

TOP_K=1000 TOP_K_DIR=data/redesign/top_k_variants/k1000 \
  PYTHONPATH=. uv run python scripts/redesign/compute_tabpfn_outer.py \
    --top-k-dir $TOP_K_DIR --output-dir data/redesign/top_k_variants/k1000 [...]
```

*(Implementer: verify flag names from actual scripts — may need 1-line config override.)*

**Step 3: Create config** `configs/redesign/p5_ablation_topk_1000.yaml` — inherit canonical, set `data.tabpfn_oof_dir: data/redesign/top_k_variants/k1000`, `data.tabpfn_outer_dir: data/redesign/top_k_variants/k1000`, run_name `p5_ablation_topk_1000`.

**Step 4: Launch 5-fold + reinfer:**

```bash
CONFIG=configs/redesign/p5_ablation_topk_1000.yaml \
OUTROOT=outputs/redesign/p5_ablation_topk_1000 \
MAX_EPOCHS=60 SEED=42 RUN_REINFER=1 \
  bash scripts/redesign/run_phase2_5fold_parallel.sh
```

**Step 5: Commit**

```
git add configs/redesign/p5_ablation_topk_1000.yaml \
        outputs/redesign/p5_ablation_topk_1000/best_vs_tabpfn_summary.json
git commit -m "+ ablation top-k=1000: R²=<value> ± <std>"
```

---

### Task D.2: top-k=4000 TabPFN feature ablation

Identical to D.1 with `--top-k 4000`, `data/redesign/top_k_variants/k4000`, config `p5_ablation_topk_4000.yaml`, outroot `p5_ablation_topk_4000`.

Note: 4000 features exceeds TabPFN-2.6 documented safe range more than 2000 does; may require `ignore_pretraining_limits=True` config knob (plan doc Risk #6). Verify during implementation.

**Commit:** `+ ablation top-k=4000: R²=<value> ± <std>`

---

### Task D.3: Per-feature z-score on TabPFN input

**Files:**
- Modify: `scripts/redesign/compute_tabpfn_oof.py` — add `--zscore` flag; when set, fit `StandardScaler` on train-fold X, apply to train+val X (inner 5-fold OOF uses inner-fold train stats; NO POOLED stats).
- Modify: `scripts/redesign/compute_tabpfn_outer.py` — same, with outer-fold train stats.
- Test: `tests/unit/redesign/test_tabpfn_zscore_no_leakage.py` — verify that z-score stats come from train-only, not pooled.

**Step 1: Write failing test** — mock train/val X with distinguishable means; assert that val-transformed mean reflects TRAIN mean, not val mean.

**Step 2: Implement `--zscore` flag** in both scripts.

**Step 3: Run tests PASS.**

**Step 4: Generate zscored caches:**

```bash
PYTHONPATH=. uv run python scripts/redesign/compute_tabpfn_oof.py \
    --zscore --output-dir data/redesign/zscore [...]

PYTHONPATH=. uv run python scripts/redesign/compute_tabpfn_outer.py \
    --zscore --output-dir data/redesign/zscore [...]
```

**Step 5: Create config** `configs/redesign/p5_ablation_zscore.yaml` — canonical + point at `data/redesign/zscore`.

**Step 6: Launch 5-fold + reinfer.**

**Step 7: Commit**

```
git commit -m "+ ablation TabPFN z-score input: R²=<value> ± <std>"
```

---

## Phase E — Paper synthesis

### Task E.1: Paper baseline table

**Files:**
- Create: `scripts/redesign/interpretability/make_baseline_table.py` — gathers per-fold R² from:
  - Ridge / ElasticNet / PLS / RF — `outputs/baseline_results_classical.csv` (existing, verify path)
  - XGBoost — existing baseline result
  - MixMIL — existing
  - scPhase — existing
  - CloudPred — existing (verify)
  - TabPFN-2.6 standalone — `outputs/redesign/<tabpfn_baseline_dir>/` (verify path)
  - Current encoder alone (R²=0.286)
  - Ours (ResDec-H3 P5 canonical)
  - Ablations #2 (no TabPFN), #5 (k_tabm=1), #6 (no DiffAttn — current canonical!), #7 (no HyperConn), #8 (no FiLM), #9 (no aug-U n=2), n=2, n=3, old canonical with DiffAttn
  - D.1 (k=1000), D.2 (k=4000), D.3 (z-score)
- Output: `outputs/redesign/interpretability/paper_baseline_table.csv` + `.md`.

**Step 1: List all existing result paths**, verify each by `ls` + reading first JSON/CSV.

**Step 2: Implement table builder.**

**Step 3: Generate both formats.**

**Step 4: Commit**

```
git commit -m "+ paper baseline table: all baselines + ablations (R², MAE, RMSE, Pearson, Spearman)"
```

---

### Task E.2: Paper figures

**Files:**
- Create: `scripts/redesign/interpretability/make_figures.py` — generates:
  1. `fig_ablation_bar.png/.pdf` — bar chart of ablation R² with error bars, sorted by R².
  2. `fig_resilience_scatter.png/.pdf` — y_true vs y_pred colored by residual (resilient/vulnerable quadrants).
  3. `fig_celltype_gene_heatmap.png/.pdf` — top-30 (cell-type, gene) pairs from Captum IG as a 30-row heatmap.
  4. `fig_head_specialization.png/.pdf` — per-head top-3 cell-type attention stacked bar or radar.
  5. `fig_subgroup_r2.png/.pdf` — APOE/sex/age-quartile R² with bootstrap CIs.
  6. `fig_calibration.png/.pdf` — residual vs TabPFN σ + coverage plot.
- Output: `outputs/redesign/interpretability/figures/`

**Step 1: Sketch each figure** (matplotlib), verify with canonical data.

**Step 2: Generate all six**, tune for publication readability (fonts, colors, titles).

**Step 3: Commit**

```
git commit -m "+ paper figures: ablation bar, resilience scatter, CT×gene heatmap, head spec, subgroup R², calibration"
```

---

## Execution order summary

```
Phase A (FiLM):           A.1 → A.2 (pilot) → [gate] → A.3 (cond. 5-fold)
                                                        │
Phase B (cleanup + re-run): B.0 (git mv) → B.1 + B.2 + B.3 (parallel, CPU/GPU light)
                                                        │
Phase C (new analyses):    C.1 → C.2 → C.3 → C.4 → C.5 (Captum) → C.6 (GSEA)
                                                        │
Phase D (R²-abls):         D.1 ∥ D.2 ∥ D.3  (run concurrently on GPU-1 while Phase C uses GPU-0)
                                                        │
Phase E (paper):           E.1 (table) → E.2 (figures)
```

**Estimated wall time** (2 GPUs, parallelized):
- Phase A: 15 min (A.1) + 10 min (A.2) + 1 hr (A.3, conditional)
- Phase B: 15 min total
- Phase C: ~6-8 hr (C.1-C.3 = ~3 hr of coding + test; C.4 = ~3-4 hr; C.5 = 30-60 min compute; C.6 = 30 min)
- Phase D: ~3-4 hr compute, mostly wall-parallelizable with Phase C
- Phase E: ~2 hr

**Total session estimate:** ~1.5-2 days of focused work.

---

## Critical reminders (from MEMORY.md)

- **Every subagent dispatch prompt MUST start with:** `"FIRST: Read /home/bic/joonh/.claude/projects/-host-milan-tank-Joon-proj-ml-snrna/memory/MEMORY.md and follow ALL rules listed there."`
- No full test suite during dev — only targeted tests for changed modules.
- No sleep+tail polling — use `run_in_background` or `tail -f`.
- Commit format: `+ {short description}` — one line, no Co-Authored-By.
- pathlib over os.path / glob.glob / string concat.
- Use both GPUs for parallel work (existing drivers do this).
- Flag decisions, don't resolve silently.
- Research integrity: implement exactly what's specified, no substitutions.

## File inventory (what this plan creates)

**New source modules:**
- `src/analysis/resdec_variance_decomposition.py`
- `src/analysis/resdec_subgroup_analysis.py`
- `src/analysis/resdec_statistical_rigor.py`
- `src/analysis/resdec_ccc_importance.py`

**New orchestration scripts (`scripts/redesign/interpretability/`):**
- `__init__.py`
- `variance_decomposition.py`
- `subgroup_r2.py`
- `paired_tests_and_bootstrap.py`
- `ccc_composite_attribution.py`
- `gsea_from_captum.py`
- `make_baseline_table.py`
- `make_figures.py`
- (moved) `captum_composite_attribution.py`
- (moved) `resilience_residual_phenotype.py`
- (moved) `extract_pathology_attention.py`
- (moved) `analyze_pathology_attention_heads.py`

**New tests:**
- `tests/unit/data/test_metadata_wiring.py`
- `tests/unit/analysis/test_resdec_variance_decomposition.py`
- `tests/unit/analysis/test_resdec_subgroup_analysis.py`
- `tests/unit/analysis/test_resdec_statistical_rigor.py`
- `tests/unit/analysis/test_resdec_ccc_importance.py`
- `tests/unit/redesign/test_tabpfn_zscore_no_leakage.py`

**New configs:**
- `configs/redesign/p5_ablation_topk_1000.yaml`
- `configs/redesign/p5_ablation_topk_4000.yaml`
- `configs/redesign/p5_ablation_zscore.yaml`

**Modified existing:**
- `src/data/datamodule.py`, `src/data/datasets.py`, `src/data/collate.py`
- `src/training/resdec_lightning_module.py:283-301`
- `scripts/redesign/compute_tabpfn_oof.py`, `compute_tabpfn_outer.py` (add `--zscore` flag)

**New output directories** (under `outputs/redesign/`):
- `p5_filmwired_pilot_fold0/` (A.2)
- `p5_canonical_filmwired_seed42/` (A.3, conditional)
- `p5_ablation_topk_1000/`, `p5_ablation_topk_4000/`, `p5_ablation_zscore/` (D.1-D.3)
- `interpretability/captum_ig/`, `interpretability/gsea/`, `interpretability/figures/` (C.5, C.6, E.2)
- Shared `outputs/redesign/interpretability/` gets new JSONs + CSVs (C.1-C.4, E.1)

---

**End of plan.**
