# Interleaved Multi-Fold HPO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor HPO `train_fn` to interleave folds epoch-by-epoch, reporting mean val_nll per epoch to Ray for correct MedianStoppingRule pruning.

**Architecture:** Instead of running each fold sequentially to completion (fold 0 all epochs → fold 1 all epochs → ...), create K trainers upfront and step each by 1 epoch per outer loop iteration. Report mean val_nll across all folds after each epoch. Per-fold EarlyStopping still works via Lightning — stopped folds just reuse their last val_nll. Remove `TuneReportCheckpointCallback` (was reporting per-fold per-epoch, causing fold-boundary confusion for the scheduler).

**Tech Stack:** Lightning Trainer (kept), Ray Tune, Pyro SVI (unchanged)

---

### Task 1: Refactor `train_fn` to interleaved epoch loop

**Files:**
- Modify: `scripts/training/hpo.py` — `train_fn` function (lines 354-511)

**Step 1: Replace sequential fold loop with interleaved epoch loop**

Replace the current `train_fn` body (from `max_epochs = config.training.max_epochs` onward, lines 434-511) with:

```python
    max_epochs = config.training.max_epochs
    min_epochs = config.training.early_stopping.get("min_epochs", 1)
    es_patience = config.training.early_stopping.get("patience", 15)
    es_min_delta = config.training.early_stopping.get("min_delta", 0.0001)

    # --- Build K trainers, models, and data modules upfront ---
    trainers = []
    models = []
    dms = []

    for fold_idx in range(n_folds):
        set_seed(seed + fold_idx, deterministic=False, benchmark=True)

        module = CognitiveResilienceLightningModule(config)

        # Filter callbacks: exclude checkpointing, LR monitor, gradient logger,
        # AND TuneReportCheckpointCallback (we report to Ray manually).
        callbacks = [
            cb for cb in setup_callbacks(config)
            if not isinstance(cb, _EXCLUDED_TRIAL_CALLBACKS)
        ]
        # NOTE: No _make_tune_report_callback() — we report mean val_nll ourselves

        trial_logger = CSVLogger(
            save_dir=str(Path(config.paths.get("logs_dir", "outputs/logs")) / "hpo_trials"),
            name="ray_trial",
            version=f"fold_{fold_idx}",
        )

        trainer = pl.Trainer(
            max_epochs=1,  # Will be incremented each outer epoch
            min_epochs=1,  # Per-fold min_epochs handled by MinEpochEarlyStopping callback
            accelerator="auto",
            devices="auto",
            strategy="auto",
            precision=config.training.get("precision", "32-true"),
            gradient_clip_val=config.training.get("gradient_clip_val", None),
            gradient_clip_algorithm="norm",
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=trial_logger,
            log_every_n_steps=config.training.get("logging", {}).get("log_every_n_steps", 10),
            val_check_interval=config.training.get("logging", {}).get("val_check_interval", 1.0),
            deterministic=False,
            benchmark=True,
        )

        dm = CognitiveResilienceDataModule(
            config=config,
            metadata=metadata,
            splits=splits,
            fold_idx=fold_idx,
            adata=None,
            precomputed_dir=config.data.get("precomputed_dir"),
            preloaded_cache=preloaded_cache,
        )

        trainers.append(trainer)
        models.append(module)
        dms.append(dm)

    # --- Interleaved epoch loop ---
    # Track which folds are still active (not early-stopped or OOM'd)
    fold_active = [True] * n_folds
    fold_last_nll = [float("inf")] * n_folds  # Last known val_nll per fold

    # Mean-level early stopping state
    best_mean_nll = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        epoch_fold_nlls = []

        for fold_idx in range(n_folds):
            if not fold_active[fold_idx]:
                # Fold early-stopped or OOM'd — reuse last known val_nll
                epoch_fold_nlls.append(fold_last_nll[fold_idx])
                continue

            # Advance this fold's trainer by 1 epoch
            trainers[fold_idx].fit_loop.max_epochs = epoch + 1

            try:
                trainers[fold_idx].fit(models[fold_idx], datamodule=dms[fold_idx])
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM at fold %d epoch %d — marking fold inactive", fold_idx, epoch)
                torch.cuda.empty_cache()
                fold_active[fold_idx] = False
                epoch_fold_nlls.append(float("inf"))
                fold_last_nll[fold_idx] = float("inf")
                continue

            # Check if this fold early-stopped
            if trainers[fold_idx].should_stop:
                fold_active[fold_idx] = False

            # Collect val_nll
            val_nll = trainers[fold_idx].callback_metrics.get("val_nll")
            if val_nll is not None:
                nll_val = val_nll.item()
                fold_last_nll[fold_idx] = nll_val
                epoch_fold_nlls.append(nll_val)
            else:
                epoch_fold_nlls.append(fold_last_nll[fold_idx])

        # Compute mean across folds and report to Ray
        mean_nll = sum(epoch_fold_nlls) / len(epoch_fold_nlls)

        # Also compute per-fold R² etc. for logging (from active folds only)
        report_dict = {
            "val_nll": mean_nll,
            "val_nll_std": float(np.std(epoch_fold_nlls)),
        }
        # Aggregate other metrics from active folds
        for metric_name in ("val_r2", "val_pearson_r", "val_spearman_rho"):
            metric_vals = []
            for fold_idx in range(n_folds):
                v = trainers[fold_idx].callback_metrics.get(metric_name)
                if v is not None:
                    metric_vals.append(v.item())
            if metric_vals:
                report_dict[metric_name] = float(np.mean(metric_vals))

        tune.report(report_dict)

        # Mean-level early stopping (independent of per-fold EarlyStopping)
        if epoch >= min_epochs:
            if mean_nll < best_mean_nll - es_min_delta:
                best_mean_nll = mean_nll
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= es_patience:
                    logger.info(
                        "Mean val_nll early stopping at epoch %d (patience %d exhausted)",
                        epoch, es_patience,
                    )
                    break

        # If all folds stopped, no point continuing
        if not any(fold_active):
            logger.info("All folds early-stopped at epoch %d", epoch)
            break

    # Cleanup all folds
    for fold_idx in range(n_folds):
        _cleanup_fold(trainers[fold_idx], models[fold_idx], dms[fold_idx])
```

**Step 2: Add `tune` import at top of `train_fn`**

The current code imports `tune` from `ray` at the top of the function body. Ensure `from ray import tune` is present (it already is at line 372, but verify it's accessible in the new scope since `tune.report` is now called in the outer loop, not inside the fold loop).

**Step 3: Remove `_make_tune_report_callback` function**

Delete the `_make_tune_report_callback` function (lines 240-257) — it's no longer used. The Ray reporting is now done manually in the outer epoch loop.

**Step 4: Update docstring**

Update the `train_fn` docstring (lines 355-358) to:

```python
    """Per-trial training function invoked by Ray Tune.

    Trains N folds interleaved epoch-by-epoch: each epoch, all folds train
    for one epoch, and the mean val_nll across folds is reported to Ray.
    This gives MedianStoppingRule a clean, comparable metric at every
    iteration without fold-boundary spikes.

    Early stopping operates at two levels:
    - Per-fold: Lightning's MinEpochEarlyStopping stops individual folds
      (stopped folds reuse their last val_nll for mean computation)
    - Mean-level: if mean val_nll doesn't improve for `patience` epochs
      after `min_epochs`, the entire trial stops

    Data is passed via tune.with_parameters (uses ray.put internally).
    With RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=1, tensors in the cache are
    backed by shared object store memory — no per-worker copy.
    """
```

**Step 5: Run targeted tests**

```bash
uv run pytest tests/unit/training/ -x -q --tb=short 2>&1 | tail -20
```

Verify no tests directly test `train_fn` internals (it's an integration-level function). If any tests import `_make_tune_report_callback`, they'll need updating.

**Step 6: Verify no other code references `_make_tune_report_callback`**

```bash
grep -r "_make_tune_report_callback" src/ scripts/ tests/
```

Should only show the deletion site. If tests reference it, update them.

**Step 7: Commit**

```bash
git add scripts/training/hpo.py
git commit -m "+ hpo: interleaved multi-fold epochs with mean val_nll reporting to Ray"
```

---

### Task 2: Verify with a dry-run (post-pipeline)

After the current pipeline HPO completes, run a quick 3-trial smoke test:

```bash
uv run python scripts/training/hpo.py \
    --config outputs/pipeline/hpo_config.yaml \
    --splits-path outputs/splits.json \
    --precomputed-dir data/precomputed/ \
    --n-gpus 1 --n-trials 3 --n-folds 2
```

Verify:
1. Both folds train in lockstep (check log output)
2. Ray table shows `iter` incrementing by 1 per epoch (not per-fold-epoch)
3. val_nll is a mean (no fold-boundary spikes)
4. Trials complete or get pruned cleanly
5. Mean-level early stopping fires when expected

---

## Design Decisions

1. **Per-fold EarlyStopping kept:** Each fold's MinEpochEarlyStopping still works. When a fold stops, its last val_nll is frozen and reused for mean computation. This is correct — a stopped fold's best performance should count.

2. **Mean-level early stopping added:** Independent of per-fold stopping. Checks if the mean val_nll across all folds has stopped improving. Uses the same patience/min_delta from config.

3. **trainer.fit() called repeatedly with incrementing max_epochs:** Lightning supports this — the trainer continues from where it left off. Optimizer state, scheduler state, callback state (TemperatureAnnealing epoch counter, KLAnnealing) are all preserved.

4. **No TuneReportCheckpointCallback:** Replaced by manual `tune.report()` in outer loop. This is the key change that gives the scheduler clean per-epoch mean metrics.
