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
