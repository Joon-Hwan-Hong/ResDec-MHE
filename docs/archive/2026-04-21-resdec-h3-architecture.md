# ResDec-H3 Architecture — Implementation Plan (P5 revision)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a cognitive-resilience regression architecture for ROSMAP snRNA-seq that beats the XGBoost baseline (R²=0.358) by adding three novel head mechanisms — **TabPFN-2.6 residual-decomposition base + H3 3-stage jointly-trained detached-residual head + TabM ensembling** — on top of the already-validated `CognitiveResilienceModel` encoder (R²=0.286). The encoder is kept unchanged; novelty is concentrated in the head stack for clean research attribution.

**Architecture (P5):** The existing `src/models/full_model.py` encoder is reused verbatim (HGT + CellTransformer + PMA + RegionHandler + PathologyAttention + shared gene gate). Its subject-level embedding `z ∈ ℝ^d_subject` is consumed by a new head stack: FiLM metadata conditioning → three sequential stages with cross-stage attention (each stage = TabM-wrapped NPT row-attention with Differential attention + Hyper-Connections) → scalar prediction. Each stage predicts a residual `f̂_k` added to `ŷ_tabpfn` (from TabPFN-2.6 in-context inference on top-2K features of flat pseudobulk). Auxiliary per-stage losses use `.detach()` on prior-stage outputs to preserve boosting semantics within a single training run. Uncertainty via TabM-ensemble disagreement. Interpretability via Captum Integrated Gradients on the composite prediction (existing `src/analysis/gene_attribution.py` infrastructure reused).

**Tech Stack:** PyTorch 2.x · Lightning · OmegaConf · Pyro (Bayesian head existing, not used in new heads) · TabPFN-2.6 (`tabpfn==7.1.1`, `ModelVersion.V2_6`) · Captum · XGBoost · uv for all Python invocations.

---

## Background context

### Why P5 (not original flat-tabular plan)

**Data-driven rejection of flat-tabular redesign:**
- 87.6% of 516 subjects are PFC-only (single region); only 8.3% have all 6 regions. Original plan's `[6 × 31 × 4785 = 890,310]` zero-padded features meant 744K constant-zero features for 452 subjects → dead gradients through any per-feature STE gate.
- Honest compute microbenchmark (documented in `scripts/redesign/bench_encoder_options.py`):
  - Flat 895K first-layer: **121.74M params**, 9.24 ms/step, 2.46 GB peak
  - Flat 153K (B-lean, aggregate only): 20.84M params, 1.56 ms/step, 0.43 GB peak
  - Current full encoder: 2.58M params, 28.53 ms/step, 0.70 GB peak
- Current encoder already handles regional availability correctly via `RegionHandler` masked softmax (6 learnable scalars, proven). Re-engineering it in flat-tabular form costs parameters for the same functional capability.

**Research-attribution argument for P5:**
- The paper's novelty is TabPFN residual-decomposition + H3 boosting + TabM ensembling, not encoder redesign.
- Reusing the proven encoder means each new head component gets isolated measurement against the known `R²=0.286` baseline.
- If we redesigned encoder + heads together, we'd conflate gains. P5 gives clean attribution.

### Key design references
- TabPFN-2.5 paper — arXiv 2511.08667 (Nov 2025). Package `tabpfn==7.1.1` defaults to `ModelVersion.V2_6` (shipped minor iteration on HuggingFace `Prior-Labs/tabpfn_2_6`).
- Hyper-Connections — arXiv 2409.19606 (ICLR 2025)
- Differential Transformer — arXiv 2410.05258 (ICLR 2025 Oral)
- TabM BatchEnsemble — arXiv 2410.24210 (ICLR 2025)
- NPT row-attention — NeurIPS 2021
- BoostTransformer — arXiv 2508.02924 (prior art for transformer-boosting)
- Captum IG — already wired via `src/analysis/gene_attribution.py`

### Design decisions committed (from 2026-04-21 brainstorming + P5 revision)

| Decision | Choice | Rationale |
|---|---|---|
| Encoder | **Reuse `CognitiveResilienceModel` unchanged** (P5) | Proven R²=0.286 baseline; clean research attribution |
| Head-stack novelty | H3 3-stage detached-residual boosting + TabPFN residual base + TabM ensembling | Original plan's load-bearing contributions, preserved |
| TabPFN version | V2.6 (default in `tabpfn==7.1.1`) | User approval 2026-04-21; newer minor iteration |
| TabPFN input | flat `pseudobulk [31 × 4785] = 148,335` → XGBoost top-2K per fold | Standard high-dim→TabPFN recipe |
| Regional handling | via existing `RegionHandler` in current encoder | Already correct; 6 learnable scalars + masked softmax |
| Metadata conditioning | FiLM on encoder output; APOE + sex + age | Pathology is already inside encoder (PathologyAttention) |
| Cell-level branch | **Kept** (inside current encoder via CellTransformer) | Already in encoder |
| Uncertainty | B3: TabM ensemble disagreement (k=8 members) | Plan's original choice; no Pyro in new heads |
| Seeds | 42 (dev) + 43 (final sanity) | Existing 5-seed baseline variance ±0.009 |
| Dropped components | STE hierarchical gate, MoE origin routing, per-feature z-score | Motivated by flat-tabular framing; redundant in P5 |

---

## Target inputs (all from existing `data/precomputed/*.pt`, do NOT regenerate)

**For the encoder path** (unchanged from current model):
- `region_{0..5}_pseudobulk` (only regions in `available_regions` exist; others padded to zero by existing collate)
- `pseudobulk [31, 4785]` (aggregate)
- `ccc_edge_index`, `ccc_edge_type`, `ccc_edge_attr`
- `cell_data`, `cell_offsets`
- `cell_type_mask`, `cell_counts`, `region_mask`
- `pathology [3]` (gpath, amylsqrt, tangsqrt from `data/metadata_ROSMAP/metadata.csv`)

**For the TabPFN residual base path** (new):
- `pseudobulk.flatten() = 148,335` features per subject
- XGBoost top-2K indices per fold (from Task 0.5) → 2,000 features per subject for TabPFN

**For FiLM metadata conditioning** (new):
- APOE one-hot (3-way: e2, e3, e4 + missingness bit)
- sex (binary + missingness)
- age (z-scored + missingness)
- Total metadata vector: ~8 dims

**Splits:** `outputs/splits.json` (5-fold, no holdout, 412 train / 104 val per fold).

---

## Final architecture (P5)

```
Per-subject input:
  Existing collate → current encoder inputs (region_pseudobulk, CCC, cells, pathology, ...)
  + Metadata (APOE, sex, age) → FiLM conditioning vector
  + pseudobulk.flatten()[top_2k_indices] → TabPFN residual base input (148K → 2K)

Encoder (UNCHANGED from src/models/full_model.py):
  CognitiveResilienceModel(
    HGT + CellTransformer + PMA + RegionHandler + PathologyAttention + gene_gate
  )
  → subject embedding z ∈ ℝ^d_subject  (d_subject = d_embed * 2 ≈ 64 by default)
  → optional Bayesian head output (not used in new heads; kept for backwards compat)

Head stack (NEW, this plan):
  z → FiLM(APOE, sex, age) → z_cond ∈ ℝ^d_subject
  
  Stage 1:
    h_1 = TabMWrappedStage(z_cond)                    # TabM ensemble k=8 over NPT+DiffAttn+HyperConn
    f̂_1 = readout(h_1)                                # scalar per subject
  
  Stage 2:
    ctx_2 = cross_stage_attention(z_cond, h_1_latent)
    h_2 = TabMWrappedStage(z_cond + ctx_2)
    f̂_2 = readout(h_2)
  
  Stage 3:
    ctx_3 = cross_stage_attention(z_cond, [h_1_latent, h_2_latent])
    h_3 = TabMWrappedStage(z_cond + ctx_3)
    f̂_3 = readout(h_3)

Prediction:
  ŷ_tabpfn = TabPFN-2.6(top_2k_features, in-context train+test)
  ŷ = ŷ_tabpfn + f̂_1 + f̂_2 + f̂_3

Loss (single backward pass):
  L = L_main(ŷ, y)
    + λ_1 · MSE(f̂_1, y − ŷ_tabpfn)
    + λ_2 · MSE(f̂_2, (y − ŷ_tabpfn − f̂_1.detach())) · w(σ_tabpfn)     [aug-U]
    + λ_3 · MSE(f̂_3, (y − ŷ_tabpfn − f̂_1.detach() − f̂_2.detach())) · w(σ_tabpfn)

Defaults: λ_main = λ_1 = λ_2 = λ_3 = 1.0 (tunable)
```

---

# Phase 0 — Setup (mostly done)

### Task 0.1: Create redesign worktree [DONE 2026-04-21]
### Task 0.2: Install TabPFN-2.6 (`tabpfn==7.1.1`) [DONE 2026-04-21]
### Task 0.3: Trigger TabPFN weights download to shared cache [DONE 2026-04-21]

Worktree at `/host/milan/tank/Joon/proj_ml_snrna/.worktrees/redesign-resdec-h3/`. Weights at `/host/milan/tank/Joon/__external_programs/tabpfn/tabpfn-v2.6-regressor-v2.6_default.ckpt` (50 MB).

---

### Task 0.4: Flat-pseudobulk + metadata loader for TabPFN input

**Files:**
- Create: `src/data/tabpfn_input.py`
- Create: `tests/unit/data/test_tabpfn_input.py`

Under P5, this module is much smaller than the original plan's 895K flat-tabular builder. Two responsibilities:
1. Flatten `pseudobulk [31, 4785] → [148,335]` per subject (for XGBoost top-2K selection → TabPFN input)
2. Load FiLM metadata (APOE, sex, age + missingness indicators) from `data/metadata_ROSMAP/metadata.csv`

Much of (1) already exists in `scripts/analysis/run_baselines.py:extract_features_a` — that function flattens pseudobulk exactly as needed.

**Step 1: Write failing tests**

```python
# tests/unit/data/test_tabpfn_input.py
import pytest
import torch
import pandas as pd
from pathlib import Path
from src.data.tabpfn_input import flatten_pseudobulk, load_metadata_vector

REAL_PT = Path("data/precomputed/R1015854.pt")
META_CSV = Path("data/metadata_ROSMAP/metadata.csv")

def test_flatten_pseudobulk_shape():
    pt = torch.load(REAL_PT, weights_only=False)
    flat = flatten_pseudobulk(pt)
    assert flat.shape == (148335,)  # 31 * 4785
    assert flat.dtype == torch.float32

def test_flatten_pseudobulk_matches_baseline():
    """Matches existing extract_features_a output byte-for-byte."""
    import sys; sys.path.insert(0, "scripts/analysis")
    from run_baselines import extract_features_a
    pt = torch.load(REAL_PT, weights_only=False)
    flat_new = flatten_pseudobulk(pt).numpy()
    flat_old = extract_features_a(pt)
    assert flat_new.shape == flat_old.shape
    assert (flat_new == flat_old).all()

def test_load_metadata_vector_shape():
    vec, field_names = load_metadata_vector("R1015854", META_CSV)
    assert vec.ndim == 1
    # APOE 3-way one-hot (e2, e3, e4 presence) + 1 missing bit = 4
    # sex: 1 + 1 missing = 2
    # age: 1 + 1 missing = 2
    # total = 8
    assert vec.shape[0] == 8
    assert len(field_names) == 8

def test_load_metadata_handles_missing():
    vec, _ = load_metadata_vector("R99999999", META_CSV)
    # All-missing subject: all missingness bits = 1, actual values = 0
    assert vec[3] == 1.0  # APOE missing bit
    assert vec[5] == 1.0  # sex missing bit
    assert vec[7] == 1.0  # age missing bit
```

**Step 2:** Run tests, confirm fail (ImportError).

```bash
uv run pytest tests/unit/data/test_tabpfn_input.py -x -q
```

**Step 3:** Implement `src/data/tabpfn_input.py`.

```python
# src/data/tabpfn_input.py
"""Minimal loaders for ResDec-H3's TabPFN residual path and FiLM metadata.

Under the P5 plan, the heavy origin-tagged flat-tabular builder is NOT needed:
TabPFN consumes the flat aggregate pseudobulk (via top-2K XGBoost importance
selection per fold), and the new head's FiLM conditioning takes a small
metadata vector.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

METADATA_FIELDS = [
    "apoe_e2", "apoe_e3", "apoe_e4", "apoe_missing",
    "sex", "sex_missing",
    "age", "age_missing",
]  # total 8 dims

# Reference age stats for z-scoring (cohort-wide; fit once, frozen)
_AGE_MEAN = 86.0  # ROSMAP cohort approx; recomputed at fit time if needed
_AGE_STD = 6.5


def flatten_pseudobulk(pt_subject: dict) -> torch.Tensor:
    """Flatten pseudobulk [31, 4785] → [148_335] as float32 tensor.

    Returned tensor is on CPU; caller moves to device as needed.
    """
    pb = pt_subject["pseudobulk"]
    if not isinstance(pb, torch.Tensor):
        pb = torch.as_tensor(pb)
    return pb.float().flatten().contiguous()


def load_metadata_vector(subject_id: str, meta_csv: Path) -> tuple[torch.Tensor, list[str]]:
    """Build an 8-dim metadata vector for FiLM conditioning.

    Fields (in order): APOE one-hot presence (e2, e3, e4, missing), sex (val, missing),
    age (z-scored, missing). All missingness indicators are 1 when the field
    is NaN in metadata.csv, 0 otherwise.

    Subject ID format: splits/precomputed use "R<digits>" (e.g. "R1015854");
    metadata.csv has BOTH `projid` (integer, unrelated to the R-prefix digits)
    AND `ROSMAP_IndividualID` (string, matches the precomputed file names exactly).
    Use `ROSMAP_IndividualID` as the join key — do NOT strip the "R" prefix.
    The numeric apoe_genotype column encodes allele pairs as two-digit integers:
    22, 23, 24, 33, 34, 44 (digits 2, 3, 4 map to e2, e3, e4).

    Returns (vector [8], field_names).
    """
    df = pd.read_csv(meta_csv)
    # Join on ROSMAP_IndividualID (string, matches .pt filenames exactly).
    # The metadata.csv's `projid` column is a DIFFERENT integer unrelated to the
    # R-prefix digits — e.g. R1015854 has projid=45115248. Do NOT strip "R".
    row = df.loc[df["ROSMAP_IndividualID"] == subject_id]
    vec = torch.zeros(len(METADATA_FIELDS), dtype=torch.float32)

    if len(row) == 0:
        # Subject not in metadata — all missing
        vec[3] = 1.0  # apoe_missing
        vec[5] = 1.0  # sex_missing
        vec[7] = 1.0  # age_missing
        return vec, METADATA_FIELDS

    r = row.iloc[0]

    # APOE: apoe_genotype is numeric (22, 23, 24, 33, 34, 44). Decompose into digit pairs
    # and set the presence bit for each allele observed.
    apoe = r.get("apoe_genotype")
    if pd.isna(apoe):
        vec[3] = 1.0
    else:
        code = int(apoe)
        d1, d2 = code // 10, code % 10
        for d in (d1, d2):
            if d == 2:
                vec[0] = 1.0  # e2 present
            elif d == 3:
                vec[1] = 1.0  # e3 present
            elif d == 4:
                vec[2] = 1.0  # e4 present

    # Sex
    sex = r.get("msex")
    if pd.isna(sex):
        vec[5] = 1.0
    else:
        vec[4] = float(sex)  # 0 or 1

    # Age (z-scored)
    age = r.get("age_death")
    if pd.isna(age):
        vec[7] = 1.0
    else:
        vec[6] = (float(age) - _AGE_MEAN) / _AGE_STD

    return vec, METADATA_FIELDS
```

**Step 4:** Run tests, verify pass.

```bash
uv run pytest tests/unit/data/test_tabpfn_input.py -x -q
```

**Step 5: Commit**

```bash
git add src/data/tabpfn_input.py tests/unit/data/test_tabpfn_input.py
git commit -m "+ tabpfn_input: flat pseudobulk + FiLM metadata loaders for ResDec-H3"
```

---

### Task 0.5: Precompute top-2K feature indices per fold (XGBoost importance on flat pseudobulk)

**Files:**
- Create: `scripts/redesign/compute_top_k_features.py`
- Output: `data/redesign/top_2000_features_fold{0..4}.json`

**Step 1:** Write script.

```python
# scripts/redesign/compute_top_k_features.py
"""Compute top-K feature indices per CV fold using XGBoost importance on the
flat pseudobulk (148,335 features = 31 cell types × 4785 genes).
Output used as input selector for TabPFN-2.6 residual-base predictions."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
import pandas as pd

from src.data.splits import load_splits
from src.data.tabpfn_input import flatten_pseudobulk


def _load_all_flat_features(precomputed_dir: Path, subject_ids: list[str]) -> dict:
    out = {}
    for sid in subject_ids:
        pt_path = precomputed_dir / f"{sid}.pt"
        if not pt_path.exists():
            continue
        pt = torch.load(pt_path, weights_only=False)
        out[sid] = flatten_pseudobulk(pt).numpy()
    return out


def _load_targets(meta_csv: Path, subject_ids: list[str]) -> dict:
    """Load cogn_global target per subject. Join key is ROSMAP_IndividualID
    (string, matches .pt filenames like 'R1015854'). Do NOT confuse with projid."""
    df = pd.read_csv(meta_csv)
    wanted = set(subject_ids)
    return {
        r["ROSMAP_IndividualID"]: float(r["cogn_global"])
        for _, r in df.iterrows()
        if r["ROSMAP_IndividualID"] in wanted and not pd.isna(r["cogn_global"])
    }


def main(args):
    splits = load_splits(args.splits_path)
    precomputed_dir = Path(args.precomputed_dir)
    meta_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all subject IDs from splits (strings like 'R1015854')
    all_ids = sorted({sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]})
    features = _load_all_flat_features(precomputed_dir, all_ids)
    targets = _load_targets(meta_csv, all_ids)

    for fold_idx, fold_split in enumerate(splits["folds"]):
        train_ids = [s for s in fold_split["train"] if s in features and s in targets]
        X_train = np.stack([features[s] for s in train_ids])
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)

        reg = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            n_jobs=-1, tree_method="hist",
            random_state=args.seed,
        )
        reg.fit(X_train, y_train)
        imp = reg.feature_importances_
        top_k_idx = np.argsort(imp)[::-1][: args.top_k].tolist()

        out_path = output_dir / f"top_{args.top_k}_features_fold{fold_idx}.json"
        out_path.write_text(json.dumps({
            "fold": fold_idx,
            "top_k": args.top_k,
            "n_features_total": X_train.shape[1],
            "indices": top_k_idx,
            "seed": args.seed,
        }))
        print(f"fold {fold_idx}: wrote {out_path} (top {args.top_k} of {X_train.shape[1]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--output-dir", default="data/redesign")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
```

**Step 2:** Run.

```bash
cd /host/milan/tank/Joon/proj_ml_snrna/.worktrees/redesign-resdec-h3
mkdir -p data/redesign
uv run python scripts/redesign/compute_top_k_features.py --top-k 2000
```

Expected: 5 JSON files in `data/redesign/top_2000_features_fold{0..4}.json`, each with 2000 integer indices in `[0, 148335)`. Total runtime ~5-10 min.

**Step 3:** Verify.

```bash
uv run python -c "
import json
for f in range(5):
    d = json.loads(open(f'data/redesign/top_2000_features_fold{f}.json').read())
    assert len(d['indices']) == 2000
    assert d['n_features_total'] == 148335
    assert all(0 <= i < 148335 for i in d['indices'][:10])
    print(f'fold {f}: {len(d[\"indices\"])} indices, first 3: {d[\"indices\"][:3]}')
"
```

**Step 4: Commit**

```bash
git add scripts/redesign/compute_top_k_features.py
git commit -m "+ compute_top_k_features: XGBoost top-2K feature selection on flat pseudobulk"
```

---

### Task 0.6: Precompute TabPFN-2.6 OOF predictions per fold

**Files:**
- Create: `scripts/redesign/compute_tabpfn_oof.py`
- Output: `data/redesign/tabpfn_oof_fold{0..4}.npz`

**Step 1:** Write script.

```python
# scripts/redesign/compute_tabpfn_oof.py
"""Pre-compute TabPFN-2.6 out-of-fold predictions on training subjects per CV fold.
Uses 5-fold-within-train OOF.
Outputs .npz with subject_ids, y_true, y_tabpfn_oof, sigma_tabpfn_oof."""
import json
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from src.data.splits import load_splits
from src.data.tabpfn_input import flatten_pseudobulk


def _load_all_flat_features(precomputed_dir: Path, subject_ids: list[str]) -> dict:
    out = {}
    for sid in subject_ids:
        pt_path = precomputed_dir / f"{sid}.pt"
        if not pt_path.exists():
            continue
        pt = torch.load(pt_path, weights_only=False)
        out[sid] = flatten_pseudobulk(pt).numpy()
    return out


def _load_targets(meta_csv: Path, subject_ids: list[str]) -> dict:
    """Load cogn_global per subject via ROSMAP_IndividualID (NOT projid)."""
    df = pd.read_csv(meta_csv)
    wanted = set(subject_ids)
    return {
        r["ROSMAP_IndividualID"]: float(r["cogn_global"])
        for _, r in df.iterrows()
        if r["ROSMAP_IndividualID"] in wanted and not pd.isna(r["cogn_global"])
    }


def main(args):
    # Ensure env vars for TabPFN are set
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/host/milan/tank/Joon/__external_programs/tabpfn")

    splits = load_splits(args.splits_path)
    precomputed_dir = Path(args.precomputed_dir)
    meta_csv = Path(args.metadata_csv)
    top_k_dir = Path(args.top_k_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = sorted({sid for fold in splits["folds"] for sid in fold["train"] + fold["val"]})
    features = _load_all_flat_features(precomputed_dir, all_ids)
    targets = _load_targets(meta_csv, all_ids)

    for fold_idx, fold_split in enumerate(splits["folds"]):
        train_ids = [s for s in fold_split["train"] if s in features and s in targets]
        top_k = json.loads(
            (top_k_dir / f"top_{args.top_k}_features_fold{fold_idx}.json").read_text()
        )["indices"]

        X_train_full = np.stack([features[s] for s in train_ids])[:, top_k]
        y_train = np.array([targets[s] for s in train_ids], dtype=np.float32)

        # 5-fold-within-train OOF
        oof_mean = np.zeros_like(y_train, dtype=np.float32)
        oof_std = np.zeros_like(y_train, dtype=np.float32)

        inner_kf = KFold(n_splits=args.n_inner_folds, shuffle=True, random_state=args.seed)
        for inner_fold, (tr_idx, va_idx) in enumerate(inner_kf.split(X_train_full)):
            reg = TabPFNRegressor(device="cuda", model_version=ModelVersion.V2_6)
            reg.fit(X_train_full[tr_idx], y_train[tr_idx])
            # Extract point prediction + per-sample std via output_type="full"
            try:
                pred_dict = reg.predict(X_train_full[va_idx], output_type="full")
                pred_mean = pred_dict.get("median", pred_dict.get("mean"))
                pred_std = pred_dict.get("std", np.ones_like(pred_mean))
            except Exception:
                # Fallback: point prediction only
                pred_mean = reg.predict(X_train_full[va_idx])
                pred_std = np.ones_like(pred_mean)
            oof_mean[va_idx] = pred_mean
            oof_std[va_idx] = pred_std
            print(f"  fold {fold_idx} inner {inner_fold}: predicted {len(va_idx)} val subjects")

        out_path = output_dir / f"tabpfn_oof_fold{fold_idx}.npz"
        np.savez(
            out_path,
            subject_ids=np.array(train_ids),
            y_true=y_train,
            y_tabpfn_oof=oof_mean,
            sigma_tabpfn_oof=oof_std,
        )
        from sklearn.metrics import r2_score
        print(f"fold {fold_idx}: wrote {out_path}  OOF R² = {r2_score(y_train, oof_mean):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--top-k-dir", default="data/redesign")
    p.add_argument("--output-dir", default="data/redesign")
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--n-inner-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
```

**Step 2:** Run.

```bash
uv run python scripts/redesign/compute_tabpfn_oof.py
```

Expected: 5 .npz files, each with 412-ish OOF predictions + uncertainties. ~30-90 min total depending on TabPFN-2.6 speed.

**Step 3:** Write + run verification test.

```python
# tests/unit/redesign/test_tabpfn_oof_outputs.py
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

def test_tabpfn_oof_shapes_and_sanity():
    rs = []
    for f in range(5):
        d = np.load(f"data/redesign/tabpfn_oof_fold{f}.npz", allow_pickle=True)
        assert len(d["subject_ids"]) == len(d["y_true"]) == len(d["y_tabpfn_oof"]) == len(d["sigma_tabpfn_oof"])
        assert not np.any(np.isnan(d["y_tabpfn_oof"]))
        # Predictions should not grossly deviate
        assert d["y_tabpfn_oof"].min() >= d["y_true"].min() - 3
        assert d["y_tabpfn_oof"].max() <= d["y_true"].max() + 3
        rs.append(r2_score(d["y_true"], d["y_tabpfn_oof"]))
    # Mean OOF R² should land in a sensible range
    assert 0.15 < np.mean(rs) < 0.55, f"unexpected TabPFN OOF R²: {np.mean(rs):.4f}"
```

**Step 4:** Record TabPFN standalone baseline R² (side benefit).

```bash
uv run python -c "
import numpy as np
from sklearn.metrics import r2_score
rs = []
for f in range(5):
    d = np.load(f'data/redesign/tabpfn_oof_fold{f}.npz')
    rs.append(r2_score(d['y_true'], d['y_tabpfn_oof']))
print(f'TabPFN-2.6 OOF R² per fold: {[f\"{r:.4f}\" for r in rs]}')
print(f'Mean ± std: {np.mean(rs):.4f} ± {np.std(rs):.4f}')
"
```

Save to `outputs/pipeline/baseline_results_tabpfn.csv` for the paper table.

**Step 5: Commit**

```bash
git add scripts/redesign/compute_tabpfn_oof.py tests/unit/redesign/test_tabpfn_oof_outputs.py
git commit -m "+ compute_tabpfn_oof: TabPFN-2.6 in-fold OOF predictions for stage-1 residual targets"
```

---

# Phase 1 — Core head-stack components (parallelizable within, but dispatch serially per subagent-driven-development)

### Task 1.1: FiLM metadata conditioning module

**Files:**
- Create: `src/models/resdec_head/film_metadata.py`
- Create: `tests/unit/models/resdec_head/test_film_metadata.py`

**Step 1: Write failing test**

```python
# tests/unit/models/resdec_head/test_film_metadata.py
import torch
from src.models.resdec_head.film_metadata import FiLMMetadata

def test_film_metadata_shape_and_modulation():
    film = FiLMMetadata(d_subject=64, d_metadata=8)
    z = torch.randn(4, 64)
    m = torch.randn(4, 8)
    z_cond = film(z, m)
    assert z_cond.shape == (4, 64)
    # Without metadata (m=0), modulation should be near-identity (gamma≈1, beta≈0 via small init)
    # Implementation note: init gamma close to 1, beta close to 0.

def test_film_metadata_gradient_flow():
    film = FiLMMetadata(d_subject=64, d_metadata=8)
    z = torch.randn(4, 64, requires_grad=True)
    m = torch.randn(4, 8, requires_grad=True)
    z_cond = film(z, m)
    loss = z_cond.sum()
    loss.backward()
    assert z.grad is not None
    assert m.grad is not None
```

**Step 2:** Run, confirm fail.

**Step 3:** Implement.

```python
# src/models/resdec_head/film_metadata.py
import torch
import torch.nn as nn


class FiLMMetadata(nn.Module):
    """FiLM conditioning: z_cond = γ(m) ⊙ z + β(m).

    Initialized so γ ≈ 1, β ≈ 0 → near-identity at start of training.
    """

    def __init__(self, d_subject: int, d_metadata: int):
        super().__init__()
        self.gamma_proj = nn.Linear(d_metadata, d_subject)
        self.beta_proj = nn.Linear(d_metadata, d_subject)
        # Near-identity init
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, z: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(metadata)  # [B, d_subject]
        beta = self.beta_proj(metadata)    # [B, d_subject]
        return gamma * z + beta
```

**Step 4:** Run tests, verify pass.

**Step 5: Commit.**

```bash
git add src/models/resdec_head/film_metadata.py tests/unit/models/resdec_head/test_film_metadata.py
git commit -m "+ resdec_head.film_metadata: FiLM conditioning for APOE/sex/age"
```

---

### Task 1.2: Hyper-Connections module

**Files:**
- Create: `src/models/resdec_head/hyper_connections.py`
- Create: `tests/unit/models/resdec_head/test_hyper_connections.py`

**Step 1: Write failing test**

```python
def test_hyper_connections_shape():
    from src.models.resdec_head.hyper_connections import HyperConnection
    hc = HyperConnection(d_model=64, n_streams=4)
    x = torch.randn(4, 64)
    # Apply a dummy sublayer
    sublayer = nn.Linear(64, 64)
    y = hc(x, sublayer)
    assert y.shape == (4, 64)
```

**Step 2:** Run, confirm fail.

**Step 3:** Implement per arXiv 2409.19606.

```python
# src/models/resdec_head/hyper_connections.py
"""Hyper-Connections (Zhu et al., ICLR 2025) — dynamic learnable residual replacement.

Replaces `x_out = x_in + sublayer(x_in)` with a learned combination of N streams.
"""
import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    def __init__(self, d_model: int, n_streams: int = 4):
        super().__init__()
        self.n_streams = n_streams
        # Learned stream weights, init as equal
        self.alpha = nn.Parameter(torch.ones(n_streams) / n_streams)
        self.beta = nn.Parameter(torch.ones(n_streams) / n_streams)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        # Expand to streams: [B, d] → [B, N, d]
        streams = x.unsqueeze(1).expand(-1, self.n_streams, -1)
        # Apply sublayer to each stream (parallel)
        stream_outputs = torch.stack([sublayer(streams[:, i, :]) for i in range(self.n_streams)], dim=1)
        # Weighted combine: sum over streams with softmax weights
        weights = torch.softmax(self.alpha, dim=0)  # [N]
        combined = (stream_outputs * weights.view(1, -1, 1)).sum(dim=1)  # [B, d]
        return combined
```

**Step 4:** Run tests.

**Step 5: Commit.**

```bash
git commit -m "+ resdec_head.hyper_connections: Hyper-Connections (Zhu 2024)"
```

---

### Task 1.3: Differential Attention module

**Files:**
- Create: `src/models/resdec_head/differential_attention.py`
- Create: `tests/unit/models/resdec_head/test_differential_attention.py`

**Step 1: Write failing test**

```python
def test_differential_attention_shape():
    from src.models.resdec_head.differential_attention import DifferentialAttention
    attn = DifferentialAttention(d_model=64, n_heads=4, lambda_init=0.8)
    x = torch.randn(4, 16, 64)  # [B, seq, d]
    y = attn(x)
    assert y.shape == x.shape
```

**Step 2:** Run, confirm fail.

**Step 3:** Implement per arXiv 2410.05258 (ICLR 2025 Oral).

```python
# src/models/resdec_head/differential_attention.py
"""Differential Transformer attention (Ye et al., ICLR 2025)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialAttention(nn.Module):
    """Two attention maps subtracted via learned λ to cancel noise.

    attn_1, attn_2 = softmax(QK_1.T / √d), softmax(QK_2.T / √d)
    output = (attn_1 - λ · attn_2) @ V
    """

    def __init__(self, d_model: int, n_heads: int = 4, lambda_init: float = 0.8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q1 = self.q1(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k1 = self.k1(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        q2 = self.q2(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k2 = self.k2(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** -0.5
        attn1 = F.softmax(q1 @ k1.transpose(-2, -1) * scale, dim=-1)
        attn2 = F.softmax(q2 @ k2.transpose(-2, -1) * scale, dim=-1)
        attn = attn1 - self.lambda_param * attn2

        out = attn @ v  # [B, H, N, d_head]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)
```

**Step 4:** Run tests.

**Step 5: Commit.**

```bash
git commit -m "+ resdec_head.differential_attention: Diff-Transformer attention (Ye 2024)"
```

---

### Task 1.4: NPT row-attention stage with Differential attention + Hyper-Connections

**Files:**
- Create: `src/models/resdec_head/npt_stage.py`
- Create: `tests/unit/models/resdec_head/test_npt_stage.py`

**Step 1:** Write test — verify forward shape + full-cohort NPT mode (attend over subjects in the batch).

**Step 2:** Implement. Full-cohort NPT: batch contains all train subjects; attention is over the subject axis (not within-subject sequence).

```python
# src/models/resdec_head/npt_stage.py
import torch
import torch.nn as nn
from .differential_attention import DifferentialAttention
from .hyper_connections import HyperConnection


class NPTStage(nn.Module):
    """Single head stage: NPT row-attention (subjects as tokens) + Diff-Attn + HyperConn.

    Input:  z_cond [B, d_subject] — all subjects in one batch (full-cohort NPT)
    Output: latent [B, d_subject], scalar [B]
    """

    def __init__(self, d_subject: int = 64, n_heads: int = 4, n_hc_streams: int = 4,
                 lambda_init: float = 0.8):
        super().__init__()
        # Treat each subject as a token in a sequence of length B
        self.diff_attn = DifferentialAttention(d_subject, n_heads=n_heads, lambda_init=lambda_init)
        self.hc = HyperConnection(d_subject, n_streams=n_hc_streams)
        self.norm1 = nn.LayerNorm(d_subject)
        self.ffn = nn.Sequential(
            nn.Linear(d_subject, d_subject * 2),
            nn.GELU(),
            nn.Linear(d_subject * 2, d_subject),
        )
        self.norm2 = nn.LayerNorm(d_subject)
        self.readout = nn.Linear(d_subject, 1)

    def forward(self, z_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z_cond: [B, d_subject] → [1, B, d_subject] (treat batch as sequence for NPT-style attention)
        x = z_cond.unsqueeze(0)  # [1, B, d]
        x = x + self.diff_attn(self.norm1(x))  # NPT row-attention
        x = x.squeeze(0)  # [B, d]
        x = self.hc(x, lambda xx: self.ffn(self.norm2(xx)))  # Hyper-Connections over FFN
        scalar = self.readout(x).squeeze(-1)  # [B]
        return x, scalar
```

**Step 4:** Tests.

**Step 5: Commit.**

```bash
git commit -m "+ resdec_head.npt_stage: NPT row-attention stage with DiffAttn + HyperConn"
```

---

### Task 1.5: TabM BatchEnsemble wrapper

**Files:**
- Create: `src/models/resdec_head/tabm_wrapper.py`
- Create: `tests/unit/models/resdec_head/test_tabm_wrapper.py`

**Step 1:** Write test — verify k members produce distinct outputs.

**Step 2:** Implement BatchEnsemble scaling per TabM (arXiv 2410.24210).

```python
# src/models/resdec_head/tabm_wrapper.py
import torch
import torch.nn as nn


class TabMWrapper(nn.Module):
    """Wrap a submodule with BatchEnsemble k members.

    Each member gets a rank-1 (s_k, r_k) scaling. At training time, all members
    run in parallel; at inference, we average predictions and compute std for
    uncertainty.
    """

    def __init__(self, submodule: nn.Module, d_io: int, k: int = 8):
        super().__init__()
        self.submodule = submodule
        self.k = k
        self.s = nn.Parameter(torch.randn(k, d_io) * 0.01 + 1.0)  # input scaling
        self.r = nn.Parameter(torch.randn(k, d_io) * 0.01 + 1.0)  # output scaling

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, d_io]
        outputs = []
        for ki in range(self.k):
            scaled_in = x * self.s[ki]
            sub_out = self.submodule(scaled_in)
            if isinstance(sub_out, tuple):
                sub_out = sub_out[0]  # take the latent if tuple returned
            outputs.append(sub_out * self.r[ki])
        stacked = torch.stack(outputs, dim=1)  # [B, k, d_io]
        mean = stacked.mean(dim=1)
        std = stacked.std(dim=1)
        return mean, std
```

**Step 4:** Tests.

**Step 5: Commit.**

```bash
git commit -m "+ resdec_head.tabm_wrapper: BatchEnsemble wrapping for uncertainty"
```

---

### Task 1.6: Cross-stage attention module

**Files:**
- Create: `src/models/resdec_head/cross_stage_attention.py`
- Create: `tests/unit/models/resdec_head/test_cross_stage_attention.py`

**Step 1:** Write test — query=current stage's embedding, keys+values=prior stages' latents.

**Step 2:** Implement.

```python
# src/models/resdec_head/cross_stage_attention.py
import torch
import torch.nn as nn


class CrossStageAttention(nn.Module):
    """Single-layer cross-attention where query = current z_cond,
    keys+values = concatenation of prior stages' latents.
    """

    def __init__(self, d_subject: int = 64, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_subject // n_heads
        self.q = nn.Linear(d_subject, d_subject)
        self.kv = nn.Linear(d_subject, d_subject * 2)
        self.out = nn.Linear(d_subject, d_subject)

    def forward(self, z_cond: torch.Tensor, prior_latents: list[torch.Tensor]) -> torch.Tensor:
        """
        z_cond: [B, d_subject]
        prior_latents: list of [B, d_subject], one per prior stage
        Returns: context [B, d_subject]
        """
        B, D = z_cond.shape
        if len(prior_latents) == 0:
            return torch.zeros_like(z_cond)

        # Stack priors as seq: [B, n_prior, d]
        ctx = torch.stack(prior_latents, dim=1)
        q = self.q(z_cond).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, 1, d_head]
        kv = self.kv(ctx).view(B, ctx.size(1), 2, self.n_heads, self.d_head)
        k = kv[:, :, 0].transpose(1, 2)  # [B, H, n_prior, d_head]
        v = kv[:, :, 1].transpose(1, 2)

        scale = self.d_head ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # [B, H, 1, n_prior]
        out = attn @ v  # [B, H, 1, d_head]
        out = out.transpose(1, 2).contiguous().view(B, D)
        return self.out(out)
```

**Step 4:** Tests.

**Step 5: Commit.**

```bash
git commit -m "+ resdec_head.cross_stage_attention: prior-stage-attending for H3 heads"
```

---

### Task 1.7: PathologyAttention integration verification (encoder reuse check)

**Files:**
- Create: `tests/unit/models/resdec_head/test_encoder_integration.py`

Simple integration test — load current `CognitiveResilienceModel`, run a dummy batch, confirm the encoder's output embedding has expected shape + the PathologyAttention path is exercised. No code change to the encoder itself.

**Step 5: Commit** (test only).

```bash
git commit -m "+ test: integration check for current encoder feeding new head stack"
```

---

### Task 1.8: ResDec-H3 head-stack composer (single-stage, no TabPFN yet)

**Files:**
- Create: `src/models/resdec_head/resdec_head.py`
- Create: `tests/unit/models/resdec_head/test_resdec_head_smoke.py`

**Step 1:** Write smoke test — FiLM + 1 NPT stage + scalar output.

**Step 2:** Implement Phase-1 composer: takes encoder output + metadata, returns `{"prediction": f̂_1, "latent_1": h_1, "sigma_1": σ_1}`.

**Step 3:** Tests.

**Step 5: Commit.**

```bash
git commit -m "+ resdec_head.resdec_head: single-stage composer (no TabPFN, no boosting)"
```

---

### Task 1.9: End-to-end Phase 1 training (current encoder + bare head, no TabPFN)

**Files:**
- Create: `scripts/redesign/train_resdec.py`
- Create: `configs/redesign/p5_phase1_baseline.yaml`

**Step 1:** Minimal Lightning training loop that:
- Loads fold 0 using existing datamodule
- Builds current encoder + new bare head via composer
- Predicts directly (no TabPFN residual yet)
- Target: `cogn_global` directly
- Train 60 epochs, report val R²

**Step 2:** Run. Expected val R²: ~0.27-0.30 (should be close to encoder-alone baseline 0.286 since the bare head is just a linear readout over the encoder embedding).

**Step 3:** If passes, commit. If not, debug before Phase 2.

```bash
git commit -m "+ phase1: end-to-end training validates current encoder + bare head"
```

---

# Phase 2 — TabPFN residual integration (sequential after Phase 1)

### Task 2.1: Verify TabPFN-2.6 standalone baseline (from Task 0.6 outputs)

Run the R² computation from Task 0.6 outputs; record to `outputs/pipeline/baseline_results_tabpfn.csv`.

```bash
git commit -m "+ baselines: TabPFN-2.6 standalone R² recorded"
```

---

### Task 2.2: Add TabPFN-2.6 as inference-time residual base

**Files:**
- Modify: `src/models/resdec_head/resdec_head.py` (add TabPFN residual composition)
- Modify: `scripts/redesign/train_resdec.py` (load per-fold OOF, compose `ŷ_tabpfn + f̂_1`)
- Create: `tests/unit/models/resdec_head/test_tabpfn_residual.py`

Training loop now:
- Target at training: `y - y_tabpfn_oof`
- Prediction at val: `TabPFN(top_2K features of val subject, context=full train) + f̂_1(val embedding)`
- The TabPFN call at val is fast (~1 second per fold for 104 val subjects).

**Step 5: Commit.**

```bash
git commit -m "+ phase2: TabPFN-2.6 residual base integrated; single-stage residual training"
```

---

### Task 2.3: Uncertainty-weighted residual loss (aug-U)

Wrap stage aux loss with `w_i = 1 / (σ_tabpfn²_i + eps)` per subject. Test that high-uncertainty subjects get larger aux gradient.

```bash
git commit -m "+ phase2: aug-U uncertainty-weighted residual loss"
```

---

# Phase 3 — H3 3-stage extension with cross-stage attention + detached-aux loss

### Task 3.1: Extend composer to 3 stages with cross-stage attention

**Files:** modify `src/models/resdec_head/resdec_head.py`, add `tests/unit/models/resdec_head/test_multi_stage.py`

Verify: stage-2 gradient does NOT flow into stage-1's f̂_1 thanks to `.detach()`.

```bash
git commit -m "+ phase3: H3 3-stage composer with cross-stage attention + detached aux losses"
```

---

### Task 3.2: Full 3-stage training run + sanity checks

Run on fold 0. Verify:
- All 4 loss terms (main + 3 aux) roughly decreasing
- `corrcoef(f̂_1, f̂_2) < 0.3`, `corrcoef(f̂_1, f̂_3) < 0.3` (stages learn distinct signal)
- Val R² > Phase 2 single-stage R²

```bash
git commit -m "+ phase3: full H3 3-stage training validated on fold 0"
```

---

# Phase 4 — Augmentations (parallel within)

### Task 4.1: FiLM metadata wire-up verification (aug-M)

Integration test: train with `film.enabled=true` vs `film.enabled=false`, compare R² on fold 0. Evidence of FiLM contribution, or drop the module.

### Task 4.2: Captum IG interpretability harness

**Files:**
- Create: `src/analysis/resdec_attribution.py`

Adapt existing `src/analysis/gene_attribution.py`'s `_PseudobulkForwardWrapper` to the composite prediction `ŷ_tabpfn + f̂_1 + f̂_2 + f̂_3`. Since TabPFN is frozen at inference, attribution flows only through the H3 head's residual contributions. This means Captum IG captures "what the H3 head contributed" — a useful decomposition for biology claims.

```bash
git commit -m "+ resdec_attribution: Captum IG for composite residual-decomposition prediction"
```

---

# Phase 5 — Evaluation

### Task 5.1: 5-fold CV training (seed 42)

Loop over 5 folds, save per-fold predictions + σ + attributions. Target: mean R² > 0.358 (beating XGBoost).

### Task 5.2: Seed 43 sanity check

Re-run fold 0 with seed 43. Verify R² within ±0.011 of seed 42 (documented small-N variance).

### Task 5.3: Head-component ablation matrix (new structure for P5)

11 ablations × 5 folds. Each ablation turns off one new component:

1. **Full model** (current encoder + TabPFN residual + 3 stages + TabM + FiLM + cross-stage attn + HyperConn + DiffAttn)
2. **No TabPFN residual** (predict y directly from current encoder + H3 heads)
3. **No H3 boosting** (single stage, no stage-2/stage-3 aux loss)
4. **No cross-stage attention** (independent stages)
5. **No TabM ensemble** (single prediction instead of k=8)
6. **No Differential attention** (vanilla NPT)
7. **No Hyper-Connections** (standard residuals in head)
8. **No FiLM metadata** (drop aug-M)
9. **No aug-U** (uniform residual loss weighting)
10. **Current encoder alone** (no new heads — should match existing R²=0.286 baseline)
11. **TabPFN-2.6 standalone** (no new encoder or heads — from Task 0.6)

Parallelized across GPUs via existing `scripts/training/run_sensitivity.sh` pattern.

### Task 5.4: Analysis + paper figures

- Variance decomposition: `Var(ŷ_tabpfn)`, `Var(f̂_1)`, `Var(f̂_2)`, `Var(f̂_3)`
- Metadata-stratified analysis: per-APOE, per-sex stage contributions
- Cross-stage attention maps
- Paired t-tests vs XGBoost (R²=0.358), TabPFN-2.6 standalone, current full model (R²=0.286)

### Task 5.5: Paper baseline table

Expected rows:
- Ridge / ElasticNet / PLS / RF (from existing `baseline_results_classical.csv`)
- XGBoost (R²=0.358, existing)
- MixMIL (R²=0.110, existing)
- scPhase (R²=-0.059, existing)
- CloudPred (verify; existing)
- **TabPFN-2.6 standalone** (new, Task 2.1) — **measured 2026-04-21 outer-fold R² = 0.399 ± 0.091 (range 0.276–0.526)** on top-2K XGBoost-importance features from flat pseudobulk. Inner-OOF R² = 0.586 ± 0.020 (used for stage-1 residual targets, not for paper baseline comparison).
- **Current encoder alone** (R²=0.286, existing baseline; ablation #10)
- **Our model (ResDec-H3 P5)** — primary result
- Target: full model > XGBoost at paired t-test α=0.1

### Task 5.6: Interpretability validation

Captum infidelity + sensitivity metrics on the composite prediction. Validate top-gene × top-cell-type lists overlap with HPO7 Tier 1 findings (Upper-layer IT, oligodendrocytes, interneurons).

### Task 5.7: Post-MVP — reconsider dropped components

If P5 full model falls short of XGBoost target:
- Try adding **STE gate on the gene_gate output** (hard 0/1 selection for interpretability crispness)
- Try adding **per-feature z-score on TabPFN input** (may help TabPFN calibration)
- Skip **MoE routing** (architecturally redundant under P5)

Otherwise, these stay dropped.

---

# Appendices

## A. Hyperparameter defaults

| Component | Parameter | Value |
|---|---|---|
| Encoder | (reused from current model's production config) | — |
| FiLM | `d_metadata` | 8 |
| Hyper-Connections | `n_streams` | 4 |
| Differential attention | `n_heads`, `λ_init` | 4, 0.8 |
| NPT stage | `d_subject`, `n_heads` | 64, 4 |
| TabM wrapper | `k_members` | 8 |
| Cross-stage attn | `n_heads` | 4 |
| Loss weights | `λ_main, λ_1, λ_2, λ_3` | 1.0 each |
| Optimizer | type, lr, wd | AdamW, 0.0015, 5.6e-6 |
| Scheduler | type, warmup | cosine, 5 epochs |
| Training | batch, max_epochs, patience | 24, 60, 10 |
| Early stop | monitor, mode | val R², max |
| TabPFN | model_version | `ModelVersion.V2_6` |
| TabPFN | top_k | 2000 |

## B. File map

```
src/
  models/
    full_model.py                         # UNCHANGED (current encoder reused)
    components/                           # UNCHANGED (region_handler, pathology_attention, ...)
    resdec_head/                          # NEW
      __init__.py
      film_metadata.py
      hyper_connections.py
      differential_attention.py
      npt_stage.py
      tabm_wrapper.py
      cross_stage_attention.py
      resdec_head.py                      # H3 composer
  data/
    tabpfn_input.py                       # NEW (flat pseudobulk + metadata)
  analysis/
    gene_attribution.py                   # UNCHANGED (existing Captum IG)
    resdec_attribution.py                 # NEW (composite-prediction IG wrapper)
scripts/
  redesign/
    bench_encoder_options.py              # DONE (microbenchmark)
    bench_p5_full_model.py                # DONE (P5 reference timing)
    compute_top_k_features.py             # NEW (Task 0.5)
    compute_tabpfn_oof.py                 # NEW (Task 0.6)
    train_resdec.py                       # NEW (main training entry)
configs/
  redesign/
    p5_phase1_baseline.yaml               # NEW (Phase 1: encoder + bare head, no TabPFN)
    p5_phase2_residual.yaml               # NEW (Phase 2: + TabPFN residual)
    p5_h3_full.yaml                       # NEW (Phase 3+4+5 full model)
tests/
  unit/
    data/
      test_tabpfn_input.py                # NEW (Task 0.4)
    models/resdec_head/
      test_film_metadata.py               # NEW (Task 1.1)
      test_hyper_connections.py           # NEW (Task 1.2)
      test_differential_attention.py      # NEW (Task 1.3)
      test_npt_stage.py                   # NEW (Task 1.4)
      test_tabm_wrapper.py                # NEW (Task 1.5)
      test_cross_stage_attention.py       # NEW (Task 1.6)
      test_encoder_integration.py         # NEW (Task 1.7)
      test_resdec_head_smoke.py           # NEW (Task 1.8)
      test_tabpfn_residual.py             # NEW (Task 2.2)
      test_multi_stage.py                 # NEW (Task 3.1)
    redesign/
      test_tabpfn_oof_outputs.py          # NEW (Task 0.6)
docs/plans/
  2026-04-21-resdec-h3-architecture.md    # this file (P5 revision)
```

## C. Ablation matrix (Phase 5.3)

Listed in Task 5.3. Each ablation = one config variant run × 5 folds. Parallelize across 2 GPUs.

## D. Baselines for paper table

(Listed in Task 5.5.)

## E. Risks flagged

1. **Current encoder is a fixed confound under P5.** If encoder is the real bottleneck, P5 caps short of target. Fallback: Phase 6 = encoder redesign (original Option P3).
2. **TabPFN σ extraction API uncertainty.** Code attempts `output_type="full"` with a fallback to scalar 1.0 on failure. Flagged for Phase 2.3 refinement.
3. **Dropped components revisit.** STE gate, MoE, per-feature z-score skipped at MVP. Task 5.7 reconsiders at end only if target missed.
4. **Multi-hop CCC signal** — explicitly delegated to current encoder's HGT branch; not re-encoded in new heads.
5. **Seed variance.** Relying on existing 5-seed baseline ±0.009. Only 42 + 43 for new model. Expand if divergent.
6. **TabPFN pretraining limits.** If per-fold training subset (~412) or top-2K features exceeds TabPFN's documented safe range, we may hit soft warnings. Strategy: let `ignore_pretraining_limits=True` (off by default) be a config knob.

## F. Success criteria

- **Phase 1** (bare head, no TabPFN): val R² ≈ 0.27-0.30 on fold 0 (parity with encoder-alone baseline 0.286).
- **Phase 2** (single-stage residual): val R² improves over both TabPFN standalone and Phase 1 → "heads add signal beyond encoder alone AND beyond TabPFN alone."
- **Phase 3** (full H3): val R² improves further; `|corrcoef(f̂_k, f̂_j)| < 0.3` for k≠j.
- **Phase 5.3 ablation matrix**: each "no X" variant worse than full model by measurable margin (provides paper ablation table).
- **Phase 5.5 paper result**: full ResDec-H3 P5 mean 5-fold R² > 0.399 (TabPFN-2.6 standalone, measured 2026-04-21) AND paired t-test α=0.1 vs TabPFN alone AND XGBoost (R²=0.358). Bonus target: reduce per-fold variance (TabPFN alone: ±0.091).
- **Phase 5.6 interpretability**: Captum IG top-gene × top-cell-type overlap with HPO7 Tier 1 findings (Upper-layer IT, oligodendrocytes, interneurons).

---

**End of plan (P5 revision, 2026-04-21).**
