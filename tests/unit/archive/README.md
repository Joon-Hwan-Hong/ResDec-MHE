# Archived Tests

This directory holds tests that were archived during the 2026-04-23 canonical-naming
refactor (branch `refactor/canonical-naming`, base tag `pre-refactor-2026-04-23`).

## Why archive instead of delete?

- Git history + the `pre-refactor-2026-04-23` tag preserve the full pre-redesign
  test suite if anyone needs to resurrect a specific test for a regression hunt.
- Placing them in this folder (excluded from pytest via `--ignore=tests/unit/archive`)
  keeps them discoverable without running them in CI, and documents the rationale.

## Archived files

Every file here tests pre-redesign code paths that are NOT exercised by the
canonical ResDec-MHE pipeline. In some cases the imported classes still exist
in `src/` (kept for legacy baselines/analysis), but the canonical pipeline
bypasses them, so the tests no longer provide coverage of the live pipeline.

### Task 4 archive batch (2026-04-24, paper-strengthening)

15 files moved to `pre_redesign/` subdir. Every file in this batch instantiates
`CognitiveResilienceModel` (the pre-redesign full model) or
`CognitiveResilienceLightningModule`. These classes still exist: the ResDec-MHE
canonical pipeline uses `CognitiveResilienceModel` as a frozen encoder (via
`build_model_from_config` at `src/training/resdec_lightning_module.py:100`), but
its Bayesian/deterministic-head output is discarded under the canonical loss
path. Tests below exercise the pre-redesign loss path (Bayesian head, pathology
encoder output, Pyro SVI, etc.) and do not track canonical ResDec-MHE behaviour.

| File | Original path | Pre-redesign refs |
|---|---|---|
| `pre_redesign/test_mixed_precision.py` | `tests/unit/models/test_mixed_precision.py` | 41 |
| `pre_redesign/test_stress_scale.py` | `tests/unit/models/test_stress_scale.py` | 16 |
| `pre_redesign/test_full_model.py` | `tests/unit/models/test_full_model.py` | 35 |
| `pre_redesign/test_edge_cases_coverage.py` | `tests/unit/models/test_edge_cases_coverage.py` | 10 |
| `pre_redesign/test_serialization.py` | `tests/unit/models/test_serialization.py` | 82 |
| `pre_redesign/test_gpu_cuda.py` | `tests/unit/models/test_gpu_cuda.py` | 37 |
| `pre_redesign/test_lightning_module.py` | `tests/unit/training/test_lightning_module.py` | 118 |
| `pre_redesign/test_collate_training_integration.py` | `tests/integration/test_collate_training_integration.py` | 5 |
| `pre_redesign/test_data_to_full_model.py` | `tests/integration/test_data_to_full_model.py` | 18 |
| `pre_redesign/test_full_model_integration.py` | `tests/integration/test_full_model_integration.py` | 35 |
| `pre_redesign/test_end_to_end.py` | `tests/integration/test_end_to_end.py` | 4 |
| `pre_redesign/test_gradient_flow.py` | `tests/integration/test_gradient_flow.py` | 9 |
| `pre_redesign/test_interface_contracts.py` | `tests/integration/test_interface_contracts.py` | 13 |
| `pre_redesign/test_regression_guards.py` | `tests/regression/test_regression_guards.py` | 10 |
| `pre_redesign/test_pipeline_smoke.py` | `tests/smoke/test_pipeline_smoke.py` | 3 |

Tests NOT archived in this batch despite testing model components:
- `tests/integration/test_data_to_model.py` — tests only data→component shapes (CellTransformer, HGTEncoderTensor), no pre-redesign model instantiation
- `tests/integration/test_region_handler_integration.py` — tests `RegionHandler`, a shared component used by ResDec-MHE
- `tests/unit/models/test_cell_transformer.py` — tests `CellTransformer`, a shared component

### Task 3 archive batch

| File | Original path | Reason archived |
|---|---|---|
| `test_extract_attention.py` | `tests/unit/inference/test_extract_attention.py` | Tests call `aggregate_hgt_attention(edge_types=...)` but the current `src/inference/extract_attention.py::aggregate_hgt_attention` signature no longer accepts an `edge_types` kwarg, and its return dict no longer has `edge_type_names` / `n_samples_per_edge_type` keys (API drift). 10/10 `TestAggregateHGTAttention` / `TestEdgeTypeOrdering` / `TestEdgeCases` / `TestAbsentEdgeTypeNaN` tests fail. The function itself is still live (used by `src/inference/predict.py:556,661,1115`); writing fresh tests against the current API is left as a follow-up when the inference path is next touched. |

### Task 2 archive batch

| File | Original path | Reason archived |
|---|---|---|
| `test_bayesian_head.py` | `tests/unit/models/test_bayesian_head.py` | Canonical ResDec-MHE uses the `resdec_head/` stack (TabM+MHA+FiLM+HyperConn) on top of the encoder's `attended` vector. `BayesianPredictionHead` is still imported by `src/models/full_model.py`, but it is frozen and its output ignored under ResDec-MHE (see `src/training/resdec_lightning_module.py:114-119`). Test coverage is obsolete for the canonical pipeline. |
| `test_deterministic_head.py` | `tests/unit/models/test_deterministic_head.py` | Same as above — `DeterministicPredictionHead` exists and is the "built-but-frozen" head under ResDec-MHE; its behaviour is not part of the canonical loss path. |
| `test_set_transformer.py` | `tests/unit/models/test_set_transformer.py` | Set-Transformer primitives (MAB/ISAB/PMA/SetTransformerEncoder) are still used via `CellTransformer` inside the encoder, but the Set-Transformer-as-subject-pooler ablation was shelved. Low-level unit tests are no longer tracking canonical behaviour — integration via `test_full_model.py` suffices. |
| `test_hpo_script.py` | `tests/unit/training/test_hpo_script.py` | Ray Tune HPO is pre-redesign. Canonical pipeline is single-config (`configs/resdec_mhe/canonical.yaml`) + fold-parallel training; no HPO. `scripts/training/hpo.py` still exists for reproducibility of pre-redesign results but is not part of canonical ops. |
| `test_extract_hpo_results.py` | `tests/unit/scripts/test_extract_hpo_results.py` | Same reason — HPO result extraction is pre-redesign. |
| `test_train_script.py` | `tests/unit/training/test_train_script.py` | Tests `scripts/training/train.py` (pre-redesign entrypoint) with `head: {type: deterministic}` config fixtures. Canonical entrypoint is `scripts/resdec_mhe/training/train.py` which uses the `resdec_head:` config section; the two code paths diverge enough that the fixtures do not translate. The pre-redesign script still exists. |
| `test_predict.py` | `tests/unit/inference/test_predict.py` | Imports `src.training.lightning_module.CognitiveResilienceLightningModule` (pre-redesign) and uses `head: {type: bayesian}` config. Canonical inference path is `src.training.resdec_lightning_module.ResDecLightningModule` — different API surface. Pre-redesign `predict.py` still exists for historical inference but is not part of canonical ops. |

## Restoring a test

If a future question demands running one of these tests:

1. `git log --follow tests/unit/archive/test_<name>.py` — see its history.
2. `git mv` it back to its original location (listed in the table above).
3. Update imports/fixtures for current `src/` API if needed.

## Policy

- **Do not run tests in this directory.** pytest invocations should use
  `--ignore=tests/unit/archive` (or a matching `norecursedirs` rule).
- **Do not add new tests here.** Archive is write-once: entries should
  document historical tests, not accumulate new dead tests. If a test is
  obsolete, fix or delete it in its original location in the same PR that
  makes it obsolete.
