#!/usr/bin/env python
"""Run scPhase baseline on our ROSMAP cognitive resilience 5-fold splits.

Generates config, loads data via scPhase's load_data(), injects our exact
fold assignments, trains per-fold, aggregates metrics, and runs scPhase's
built-in Captum interpretability (gene attributions + cell attention).

Usage:
    baselines/scPhase/.venv/bin/python baselines/scPhase/run_rosmap.py \
        --data-h5ad baselines/shared/scphase_input.h5ad \
        --splits outputs/splits.json \
        --results-dir outputs/baselines/scphase \
        --device-model cuda:0 --device-encoder cuda:1
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# sys.path setup — scPhase uses flat imports (e.g. `from data_loader import ...`)
# We add repo/scphase/ so that scPhase's internal imports resolve correctly.
# ---------------------------------------------------------------------------
_SCPHASE_SRC = str(Path(__file__).resolve().parent / "repo" / "scphase")
if _SCPHASE_SRC not in sys.path:
    sys.path.insert(0, _SCPHASE_SRC)

from data_loader import AttnMoE_Dataset, load_data, sparse_collate_fn
from model import SCMIL_AttnMoE
from run_cv import (
    _save_fold_prediction_data,
    _save_predictions_csv,
    train_and_evaluate_fold,
)
from train_utils import (
    calculate_cell_attention,
    calculate_gene_attributions,
    calculate_sample_gene_attributions,
    ensemble_cell_attentions,
    ensemble_gene_attributions,
    ensemble_sample_gene_attributions,
    plot_ensemble_cell_attention_umaps,
    plot_ensemble_gene_attributions,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("SCMIL_Workflow")

# ---------------------------------------------------------------------------
# Config generation — paper defaults adapted for ROSMAP
# ---------------------------------------------------------------------------

def build_config(
    data_h5ad: str,
    results_dir: str,
    device_model: str,
    device_encoder: str,
    num_folds: int = 5,
    seed: int = 3407,
) -> dict:
    """Build scPhase config dict with paper defaults for our ROSMAP data.

    Adaptations from paper defaults:
    - input_dim: 4797 (our gene count, not their default 5000)
    - n_classes: 1 (regression output)
    - task_type: "regression"
    - use_domain_adaptation: false (single cohort ROSMAP; code also auto-disables
      for regression + single group)
    - Model and encoder can be split across GPUs
    - batch_col: set to "batch_int" (non-existent column) so data_loader falls
      back to 0 for all samples, avoiding .astype(int) failure on string "ROSMAP"
    """
    pickle_path = os.path.join(os.path.abspath(results_dir), "scphase_data_cache.pkl")

    return {
        "path_params": {
            "data_h5ad_file": os.path.abspath(data_h5ad),
            "_master_pickle_file": pickle_path,
            "RESULTS_DIR": os.path.abspath(results_dir),
            "LOGNAME": "training_ROSMAP.log",
            "MODEL_NAME": "scPhase_ROSMAP",
        },
        "data_params": {
            "sample_col": "sample_id",
            "label_col": "phenotype",
            # Use non-existent column so data_loader assigns batch=0 for all
            # samples (avoids .astype(int) failure on string "ROSMAP").
            "batch_col": "batch_int",
        },
        "run_params": {
            "seed": seed,
            "skip_groups": [],
            "device_model": device_model,
            "device_encoder": device_encoder,
            "task_type": "regression",
            "num_folds": num_folds,
        },
        "ablation_params": {
            "use_moe": True,
            "attention_type": "linformer",
            "use_domain_adaptation": False,
        },
        "training_params": {
            "epochs": 150,
            "batch_size": 1,
            "num_workers": 1,  # low to avoid OOM on fork (53GB resident); their code requires >0 for prefetch_factor
            "early_stopping_patience": 20,
            "val_size": 0.15,
        },
        "optimizer_params": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999],
            "clip_grad_norm": 1.0,
        },
        "scheduler_params": {
            "warmup_epochs": 10,
        },
        "model_params": {
            "input_dim": 4797,
            "encoder_dims": [1024, 512],
            "classifier_dims": [128, 64],
            "hidden_dim": 256,
            "n_classes": 1,
            "max_instances": 20000,
            "instance_encoder_dropout_rates": [0.1, 0.1, 0.1],
            "classifier_dropout_rates": [0.1, 0.1],
            "instance_dropout_rate": 0.2,
            "mha_dropout": 0.3,
            "linformer_dropout": 0.3,
            "num_heads": 4,
            "linformer_k": 256,
            "moe_num_experts": 4,
            "moe_dropout": 0.2,
        },
    }


# ---------------------------------------------------------------------------
# Fold mapping — translate our splits.json subject IDs to scPhase indices
# ---------------------------------------------------------------------------

def map_splits_to_indices(
    splits: dict,
    sample_ids: list[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map our splits.json fold assignments to scPhase DataList indices.

    Our fold's `val` (93 subjects) = scPhase's `test_idx`.
    Our fold's `train` (372 subjects) = scPhase's `train_idx`.
    scPhase further splits train_idx internally into train/val for early stopping.

    Returns list of (train_idx, test_idx) numpy arrays, one per fold.
    """
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    fold_indices = []
    for fold_i, fold_spec in enumerate(splits["folds"]):
        train_sids = fold_spec["train"]
        test_sids = fold_spec["val"]

        # Validate all IDs exist in DataList
        missing_train = [s for s in train_sids if s not in sid_to_idx]
        missing_test = [s for s in test_sids if s not in sid_to_idx]
        if missing_train:
            raise ValueError(
                f"Fold {fold_i}: {len(missing_train)} train subject IDs not found in "
                f"scPhase DataList (first 5: {missing_train[:5]})"
            )
        if missing_test:
            raise ValueError(
                f"Fold {fold_i}: {len(missing_test)} val/test subject IDs not found in "
                f"scPhase DataList (first 5: {missing_test[:5]})"
            )

        train_idx = np.array([sid_to_idx[s] for s in train_sids])
        test_idx = np.array([sid_to_idx[s] for s in test_sids])

        # Sanity check: no overlap
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            raise ValueError(
                f"Fold {fold_i}: {len(overlap)} indices appear in both train and test"
            )

        fold_indices.append((train_idx, test_idx))

    return fold_indices


# ---------------------------------------------------------------------------
# Cross-validation with our exact splits
# ---------------------------------------------------------------------------

def run_cv_with_our_splits(
    config: dict,
    splits: dict,
) -> pd.DataFrame:
    """Run scPhase 5-fold CV using our exact fold assignments."""

    logger.info("--- Starting ROSMAP CV with injected splits ---")
    logger.info(f"Configuration:\n{json.dumps(config, indent=2)}")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Forcing CPU.")
        config["run_params"]["device_model"] = "cpu"
        config["run_params"]["device_encoder"] = "cpu"

    # Clear any leftover prediction storage from prior runs
    if hasattr(_save_fold_prediction_data, "prediction_storage"):
        _save_fold_prediction_data.prediction_storage.clear()

    # Load data via scPhase's data_loader
    logger.info("Loading data via scPhase load_data()...")
    DataList, DataLabel, DataBatch, SampleIDs = load_data(config)
    logger.info(f"Data loaded: {len(DataList)} subjects, {DataList[0].shape[1]} genes")

    # Map our splits to scPhase indices
    fold_indices = map_splits_to_indices(splits, SampleIDs)
    logger.info(f"Mapped {len(fold_indices)} folds from splits.json")
    for i, (tr, te) in enumerate(fold_indices):
        logger.info(f"  Fold {i}: train={len(tr)}, test={len(te)}")

    # Domain adaptation: disabled for regression + single cohort
    use_domain_adaptation = False
    num_groups = len(np.unique(DataBatch))
    logger.info(
        f"Single data group detected (n_groups={num_groups}). "
        "Domain adaptation disabled for regression task."
    )

    # --- CV Loop ---
    all_results = []
    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        t0 = time.time()
        fold_results = train_and_evaluate_fold(
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            DataList=DataList,
            DataLabel=DataLabel,
            DataBatch=DataBatch,
            SampleIDs=SampleIDs,
            cfg=config,
            use_domain_adaptation=use_domain_adaptation,
        )
        elapsed = time.time() - t0
        logger.info(f"Fold {fold + 1} completed in {elapsed / 60:.1f} min")

        fold_with_meta = {
            "model_name": config["path_params"]["MODEL_NAME"],
            "fold": fold + 1,
            **fold_results,
        }
        all_results.append(fold_with_meta)

    # Save all-folds CSV
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(
        config["path_params"]["RESULTS_DIR"],
        f"AllFolds_{config['path_params']['MODEL_NAME']}.csv",
    )
    results_df.to_csv(results_path, index=False)
    logger.info(f"All-folds results saved to: {results_path}")

    # Save per-sample predictions
    _save_predictions_csv(config)

    # Summary statistics
    metric_cols = [c for c in results_df.columns if c not in ("model_name", "fold")]
    summary = {"model_name": config["path_params"]["MODEL_NAME"]}
    logger.info("\n--- FINAL CV SUMMARY ---")
    for metric in metric_cols:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        summary[f"mean_{metric}"] = mean_val
        summary[f"std_{metric}"] = std_val
        logger.info(f"  {metric.upper():>8s}: {mean_val:.4f} (+/- {std_val:.4f})")

    summary_path = os.path.join(
        config["path_params"]["RESULTS_DIR"],
        f"Summary_{config['path_params']['MODEL_NAME']}.csv",
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")

    return results_df


# ---------------------------------------------------------------------------
# Interpretability — mirrors run_interpretation_ensemble() from run.py
# ---------------------------------------------------------------------------

def run_interpretability(config: dict) -> None:
    """Run scPhase's built-in interpretability: gene attributions, cell attention,
    sample-level attributions, ensembled across all folds.

    Mirrors run_interpretation_ensemble() from baselines/scPhase/repo/run.py.
    """
    logger.info("--- Starting Ensemble Interpretability Analysis ---")

    # anndata >= 0.11 requires opt-in for nullable string writing
    import anndata
    anndata.settings.allow_write_nullable_strings = True

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Forcing CPU.")
        config["run_params"]["device_model"] = "cpu"
        config["run_params"]["device_encoder"] = "cpu"

    path_cfg = config["path_params"]
    run_cfg = config["run_params"]
    model_cfg = config["model_params"]
    device = run_cfg["device_model"]

    # Find fold model checkpoints
    pattern = os.path.join(
        path_cfg["RESULTS_DIR"],
        f"BestModel_{path_cfg['MODEL_NAME']}_Fold*.pt",
    )
    fold_model_paths = sorted(
        glob.glob(pattern),
        key=lambda x: int(x.split("Fold")[1].split(".")[0]),
    )
    if not fold_model_paths:
        raise FileNotFoundError(
            f"No fold model checkpoints found at {pattern}. "
            "Run CV training first."
        )
    logger.info(f"Found {len(fold_model_paths)} fold models for ensemble analysis.")

    # Load data
    DataList, DataLabel, DataBatch, sample_ids = load_data(config)
    adata = sc.read_h5ad(config["path_params"]["data_h5ad_file"])
    gene_list = adata.var_names.tolist()

    # For regression, label_map is not used but the function signature requires it
    label_map = None

    all_fold_attributions = []
    all_fold_attention_arrays = []
    all_fold_sample_attributions = []

    for fold_idx, model_path in enumerate(fold_model_paths):
        logger.info(f"Processing fold {fold_idx + 1}/{len(fold_model_paths)}: {model_path}")

        # Load checkpoint once; reuse for num_domains inference and state_dict loading
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            domain_key = "domain_classifier.domain_classifier.6.weight"
            if domain_key in checkpoint:
                num_domains = checkpoint[domain_key].shape[0]
                logger.info(
                    f"Inferred num_domains={num_domains} from checkpoint for fold {fold_idx + 1}"
                )
            else:
                num_domains = len(np.unique(DataBatch))
                logger.warning(
                    f"Could not infer num_domains from checkpoint, "
                    f"using {num_domains} from data"
                )
        except Exception as e:
            logger.error(f"Error loading checkpoint to infer num_domains: {e}")
            num_domains = len(np.unique(DataBatch))
            checkpoint = None

        # Construct model and load weights (reuse checkpoint loaded above)
        network = SCMIL_AttnMoE(
            model_cfg,
            config["ablation_params"],
            num_domains,
            run_cfg["device_encoder"],
            device,
        )
        if checkpoint is not None:
            network.load_state_dict(checkpoint)
        else:
            network.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
        del checkpoint
        network.eval()

        # Create DataLoader over ALL subjects for interpretability
        interpret_dataset = AttnMoE_Dataset(DataList, DataLabel, DataBatch, sample_ids)
        interpret_loader = DataLoader(
            interpret_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config["training_params"]["num_workers"],
            collate_fn=sparse_collate_fn,
        )

        # 1. Gene attributions (Integrated Gradients)
        logger.info(f"Computing gene attributions for fold {fold_idx + 1}...")
        fold_attr = calculate_gene_attributions(
            network, interpret_loader, gene_list, config, device, label_map
        )
        all_fold_attributions.append(fold_attr)

        # 2. Sample-level gene attributions
        logger.info(f"Computing sample gene attributions for fold {fold_idx + 1}...")
        sample_attr = calculate_sample_gene_attributions(
            network, interpret_loader, gene_list, config, device
        )
        all_fold_sample_attributions.append(sample_attr)

        # 3. Cell attention weights
        logger.info(f"Computing cell attentions for fold {fold_idx + 1}...")
        fold_attention_array = calculate_cell_attention(
            network, interpret_loader, device, config, adata
        )
        all_fold_attention_arrays.append(fold_attention_array)

        # Cleanup
        del network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Ensemble across folds ---
    logger.info("Ensembling gene attributions across folds...")
    ensemble_attributions = ensemble_gene_attributions(all_fold_attributions, config)

    logger.info("Ensembling cell attentions across folds...")
    updated_adata = ensemble_cell_attentions(all_fold_attention_arrays, config, adata)

    logger.info("Ensembling sample gene attributions across folds...")
    ensemble_sample_gene_attributions(all_fold_sample_attributions, config)

    # --- Plots ---
    logger.info("Generating ensemble gene attribution plots...")
    plot_ensemble_gene_attributions(config, ensemble_attributions)

    logger.info("Generating ensemble cell attention UMAP plots...")
    try:
        plot_ensemble_cell_attention_umaps(config, updated_adata)
    except Exception as e:
        # May fail if no X_umap in adata (our data may not have UMAP precomputed)
        logger.warning(
            f"Cell attention UMAP plot failed (likely no X_umap in adata): {e}"
        )

    logger.info("--- Ensemble interpretability analysis completed! ---")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run scPhase baseline on ROSMAP cognitive resilience task",
    )
    parser.add_argument(
        "--data-h5ad",
        required=True,
        help="Path to scphase_input.h5ad (from prepare_data.py)",
    )
    parser.add_argument(
        "--splits",
        required=True,
        help="Path to outputs/splits.json with our 5-fold assignments",
    )
    parser.add_argument(
        "--results-dir",
        default="outputs/baselines/scphase",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--device-model",
        default="cuda:0",
        help="CUDA device for model/classifier (default: cuda:0)",
    )
    parser.add_argument(
        "--device-encoder",
        default="cuda:1",
        help="CUDA device for attention encoder (default: cuda:1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (default: 3407, scPhase paper default)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip CV training (only run interpretability on existing checkpoints)",
    )
    parser.add_argument(
        "--skip-interpret",
        action="store_true",
        help="Skip interpretability (only run CV training)",
    )
    args = parser.parse_args()

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Build config
    config = build_config(
        data_h5ad=args.data_h5ad,
        results_dir=args.results_dir,
        device_model=args.device_model,
        device_encoder=args.device_encoder,
        seed=args.seed,
    )

    # Save config for reproducibility
    config_path = os.path.join(args.results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Setup logging (uses scPhase's setup_logging which writes to RESULTS_DIR)
    setup_logging(config)

    set_seed(config["run_params"]["seed"])
    logger.info(f"Global seed set to: {config['run_params']['seed']}")
    logger.info(f"Config saved to: {config_path}")

    # Load our splits
    with open(args.splits) as f:
        splits = json.load(f)
    logger.info(
        f"Loaded splits: {len(splits['train_val_pool'])} subjects, "
        f"{len(splits['folds'])} folds"
    )

    # --- Phase 1: Cross-Validation ---
    if not args.skip_cv:
        results_df = run_cv_with_our_splits(config, splits)
        logger.info(f"\nCV complete. Results:\n{results_df.to_string()}")
    else:
        logger.info("Skipping CV training (--skip-cv)")

    # --- Phase 2: Interpretability ---
    if not args.skip_interpret:
        run_interpretability(config)
    else:
        logger.info("Skipping interpretability (--skip-interpret)")

    logger.info("--- ROSMAP scPhase baseline run complete ---")


if __name__ == "__main__":
    main()
