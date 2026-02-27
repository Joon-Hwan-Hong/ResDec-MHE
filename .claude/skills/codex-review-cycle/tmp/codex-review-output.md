**Findings**
1. **High**: Optuna pruner fallback defaults are still epoch-oriented, which conflicts with the fold-level pruning design and can silently regress behavior when config keys are omitted.  
Evidence: [scripts/optuna_optimize.py:69](/host/milan/tank/Joon/proj_ml_snrna/scripts/optuna_optimize.py:69), [scripts/optuna_optimize.py:71](/host/milan/tank/Joon/proj_ml_snrna/scripts/optuna_optimize.py:71), [scripts/optuna_optimize.py:72](/host/milan/tank/Joon/proj_ml_snrna/scripts/optuna_optimize.py:72), design target [plan.md:1166](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1166), [plan.md:1167](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1167).  
Impact: Partial configs can prune too late and degrade HPO efficiency/consistency.

2. **High**: Subject ID alignment is strict and unnormalized across inference/analysis paths, so type/format mismatches can silently drop metadata/covariates.  
Evidence: [src/inference/predict.py:683](/host/milan/tank/Joon/proj_ml_snrna/src/inference/predict.py:683), [src/inference/predict.py:684](/host/milan/tank/Joon/proj_ml_snrna/src/inference/predict.py:684), [scripts/run_analysis.py:273](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_analysis.py:273), [src/data/datasets.py:554](/host/milan/tank/Joon/proj_ml_snrna/src/data/datasets.py:554).  
Impact: `region/split/pathology` alignment can degrade to nulls without a hard failure.

3. **High**: Data-flow contract drift: design doc still specifies prediction columns `mean/std`, but implementation and analysis consume `predicted_mean/predicted_std`.  
Evidence: contract [plan.md:2170](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:2170), writer [src/inference/predict.py:816](/host/milan/tank/Joon/proj_ml_snrna/src/inference/predict.py:816), [src/inference/predict.py:817](/host/milan/tank/Joon/proj_ml_snrna/src/inference/predict.py:817), [src/inference/predict.py:821](/host/milan/tank/Joon/proj_ml_snrna/src/inference/predict.py:821), consumer [scripts/run_analysis.py:987](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_analysis.py:987), [scripts/run_analysis.py:1027](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_analysis.py:1027).  
Impact: External integrations following the doc contract will break.

4. **Medium**: Dataset/DataLoader assembly logic is duplicated across modules instead of using a shared factory.  
Evidence: [src/data/loaders.py:19](/host/milan/tank/Joon/proj_ml_snrna/src/data/loaders.py:19), [src/data/datamodule.py:109](/host/milan/tank/Joon/proj_ml_snrna/src/data/datamodule.py:109), [scripts/run_inference.py:112](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_inference.py:112), [scripts/run_inference.py:117](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_inference.py:117).  
Impact: Higher drift risk and more maintenance overhead.

5. **Medium**: Deprecated loader API is still exported from `src.data`, encouraging stale usage.  
Evidence: [src/data/__init__.py:4](/host/milan/tank/Joon/proj_ml_snrna/src/data/__init__.py:4).  
Impact: Mixed patterns remain in ecosystem/tests and make migration incomplete.

6. **Medium**: `run_analysis.py` still contains legacy `edge_metadata.parquet` ingestion path that the current pipeline does not produce.  
Evidence: [scripts/run_analysis.py:701](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_analysis.py:701), [scripts/run_analysis.py:704](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_analysis.py:704), [scripts/run_analysis.py:706](/host/milan/tank/Joon/proj_ml_snrna/scripts/run_analysis.py:706).  
Impact: Extra branch complexity and stale-file confusion.

7. **Medium**: Placeholder/degraded analysis modes can produce “successful” outputs even when required upstream artifacts were not extracted.  
Evidence: [src/analysis/ccc_importance.py:198](/host/milan/tank/Joon/proj_ml_snrna/src/analysis/ccc_importance.py:198), [src/analysis/ccc_importance.py:206](/host/milan/tank/Joon/proj_ml_snrna/src/analysis/ccc_importance.py:206), [src/analysis/regional_analysis.py:150](/host/milan/tank/Joon/proj_ml_snrna/src/analysis/regional_analysis.py:150), [src/analysis/regional_analysis.py:157](/host/milan/tank/Joon/proj_ml_snrna/src/analysis/regional_analysis.py:157).  
Impact: Easy to miss upstream extraction/config mistakes.

8. **Medium**: Optuna script validation is incomplete for keys it later dereferences (`optuna`, `paths`).  
Evidence: [scripts/optuna_optimize.py:382](/host/milan/tank/Joon/proj_ml_snrna/scripts/optuna_optimize.py:382), then direct use at [scripts/optuna_optimize.py:384](/host/milan/tank/Joon/proj_ml_snrna/scripts/optuna_optimize.py:384) and [scripts/optuna_optimize.py:497](/host/milan/tank/Joon/proj_ml_snrna/scripts/optuna_optimize.py:497).  
Impact: Late runtime failures instead of explicit config-validation errors.

9. **Medium**: Full-loop verification plan is only partially realized as dedicated suites; no standalone `e2e/performance/property/edge_cases` directories despite design sections.  
Evidence: expected sections [plan.md:1483](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1483), [plan.md:1533](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1533), [plan.md:1588](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1588), [plan.md:1666](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1666), current test root [tests](/host/milan/tank/Joon/proj_ml_snrna/tests).  
Impact: Weaker confidence for production-scale and long-run regressions.

10. **Low**: Smoke test comments are stale and describe an old manual-SVI path.  
Evidence: [tests/smoke/test_pipeline_smoke.py:304](/host/milan/tank/Joon/proj_ml_snrna/tests/smoke/test_pipeline_smoke.py:304), [tests/smoke/test_pipeline_smoke.py:330](/host/milan/tank/Joon/proj_ml_snrna/tests/smoke/test_pipeline_smoke.py:330), current behavior [src/training/lightning_module.py:76](/host/milan/tank/Joon/proj_ml_snrna/src/training/lightning_module.py:76).  
Impact: Maintainer confusion.

11. **Low**: Design doc implementation tables are stale in places relative to code/changelog decisions.  
Evidence: CRPS task still listed [plan.md:2236](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:2236), but losses only include BetaNLL/MSE [src/training/losses.py:18](/host/milan/tank/Joon/proj_ml_snrna/src/training/losses.py:18), with CRPS metric retained [src/training/metrics.py:88](/host/milan/tank/Joon/proj_ml_snrna/src/training/metrics.py:88).  
Also `n_selected_types` still appears in doc search space [plan.md:302](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:302), [plan.md:1186](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1186), but has no implementation mapping [scripts/optuna_optimize.py:159](/host/milan/tank/Joon/proj_ml_snrna/scripts/optuna_optimize.py:159).  
Impact: Alignment ambiguity for future work.

12. **Low**: Planned utility test bucket exists but is empty.  
Evidence: planned utility tests [plan.md:1333](/host/milan/tank/Joon/proj_ml_snrna/docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md:1333), current folder [tests/unit/utils](/host/milan/tank/Joon/proj_ml_snrna/tests/unit/utils).  
Impact: Missing direct coverage for utility APIs in their expected location.

**Open Questions / Assumptions**
1. Is strict string normalization of subject IDs guaranteed upstream across metadata, split JSON, and precomputed `.npz` stems? If not, ID-alignment hardening should be treated as a release blocker.
2. Should placeholder/degraded analysis outputs remain default behavior, or should they require explicit opt-in flags in production runs?

**Answers to your (1)–(7)**
1. **Did everything in the design doc get implemented?**  
Core training/inference/analysis architecture is largely implemented. Main gaps are doc-alignment/ops gaps: stale contract (`mean/std`), stale task/search-space items (`CRPS` loss class, `n_selected_types`), and missing dedicated test suites (`e2e/performance/property/edge_cases`).

2. **Any systems design issues?**  
Yes: duplicated loader orchestration, incomplete key validation in Optuna script, and permissive placeholder fallbacks that can hide upstream extraction errors.

3. **Where is functionality re-implemented instead of reused?**  
Primary case: dataset/dataloader assembly duplicated across `DataModule`, `src/data/loaders.py`, and `scripts/run_inference.py` instead of one canonical builder.

4. **Stale code or deprecated functions with equivalents elsewhere?**  
Yes: deprecated `create_fold_dataloaders` is still exported; legacy `edge_metadata.parquet` branch persists; stale smoke comments reference old Bayesian behavior; multiple backward-compat paths remain.

5. **Integration issues between modules?**  
Yes: subject-ID alignment fragility and documented-vs-actual prediction schema drift are the main integration risks.

6. **Scaffolding/TODOs intended to be resolved later?**  
No active `TODO/FIXME` markers found, but there is active scaffolding/degraded behavior (CCC/regional placeholders, legacy edge metadata path) that should be explicitly decided as keep/remove now.

7. **Readiness for a full training loop?**  
**Good but not fully hardened**. For preprocessed ROSMAP-style data and existing scripts, the loop is close to operational. For robust “full-loop” confidence, address the high/medium items above, especially pruning defaults, ID normalization, schema contract cleanup, and dedicated end-to-end/performance tests.

Static review only; I did not execute the training/inference/test suites in this pass.  

1. Update pruning defaults and Optuna validation requirements.  
2. Enforce canonical subject-ID normalization at dataset load boundaries.  
3. Consolidate dataloader assembly into a single reusable factory and deprecate/remove the old export path.