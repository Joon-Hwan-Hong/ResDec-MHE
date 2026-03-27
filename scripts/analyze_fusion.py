"""
Fusion analysis: branch embeddings, CKA, per-sample predictions.

Investigates branch complementarity in the 2-branch architecture (HGT + CellTransformer).
Determines whether branches learn redundant representations or whether fusion
extracts complementary signal.

Analyses:
1. CKA between branch embeddings (hgt, cell) in the full model
2. Per-sample prediction comparison: full model vs ablation models
3. Branch output distributions (magnitude, variance, activation patterns)
4. Inference-time ablation (zero out branches in full model)

Usage:
    uv run python scripts/analyze_fusion.py [--device cuda:0]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import pyro
import pyro.poutine
from pyro.infer.autoguide import AutoDiagonalNormal

from src.models.full_model import build_model_from_config
from src.data.splits import load_splits, get_fold_subjects
from src.data.datasets import PrecomputedDataset
from src.data.collate import collate_for_hgt_multiregion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rank 3 (trial 83) 5-fold production run directories, ordered by fold_idx
FOLD_DIRS = [
    "20260319_040457_184349_5FoldProd_rank3_trial83_9213642df1b0f8fb641da669726f1f0eedbc21fb81f037c80fc3b763dda3b28f",
    "20260319_040457_190739_5FoldProd_rank3_trial83_b28a73d4d450dee70081d53d3dd48a3c7dd963ac8aabe58995eddf8bb5daf83c",
    "20260319_043337_564432_5FoldProd_rank3_trial83_9fe82237d5a71eeb2ce0a381402c708df56f4eef73cfd0095e76684a2fbf149d",
    "20260319_043337_586279_5FoldProd_rank3_trial83_73ff091e52df93798a72608d674be692c806b646854e8b141d0d0a3c1f1e8466",
    "20260319_051130_032837_5FoldProd_rank3_trial83_bccf5db2b163de81dcf32e9d7ce2ae6eac1735a62fdd8f50e289ba84e5fb7c55",
]

# Best checkpoints per fold (lowest val_nll)
BEST_CKPTS = [
    "epoch=66-val_nll=0.3563.ckpt",
    "epoch=17-val_nll=0.5811.ckpt",
    "epoch=92-val_nll=0.4391.ckpt",
    "epoch=18-val_nll=0.3801.ckpt",
    "epoch=11-val_nll=0.5021.ckpt",
]

# Ablation run directories (CT-only and HGT-only) — will find dynamically
ABLATION_PATTERNS = {
    "cell_transformer_only": "cell_transformer_only",
    "hgt_only": "hgt_only",
}


# ── CKA Implementation ──────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA (Kornblith et al., 2019).

    Args:
        X: [n_samples, n_features_x] — centered by this function
        Y: [n_samples, n_features_y] — centered by this function

    Returns:
        CKA similarity in [0, 1]. Higher = more similar representations.
    """
    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Frobenius inner product of gram matrices: <XX^T, YY^T>_F
    # Efficient computation: ||Y^T X||_F^2
    YtX = Y.T @ X
    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: str | Path, device: str = "cpu"):
    """Load CognitiveResilienceModel from checkpoint with Pyro guide.

    Returns:
        (model, guide, config) — model in eval mode with posterior median.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(ckpt["model_config"])
    full_config = OmegaConf.create(ckpt["full_config"])

    # Build model
    model = build_model_from_config(config)
    model.eval()

    # Load state dict (filter to model keys)
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v  # Strip "model." prefix
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # Set up Pyro guide — must prototype before restoring param store
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    guide.to(device)

    # Store param store state for deferred loading after prototyping
    param_store_state = ckpt.get("pyro_param_store")

    return model, guide, full_config, param_store_state


def conditioned_forward(model, guide, batch_kwargs, **extra_kwargs):
    """Forward pass using posterior median (MAP estimate)."""
    median = guide.median()
    conditioned = pyro.poutine.condition(model, data=median)
    return conditioned(**batch_kwargs, **extra_kwargs)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_fold_data(config, fold_idx: int, device: str = "cpu", split: str = "val"):
    """Load train or val subjects for a given fold using PrecomputedDataset.

    Args:
        config: Experiment config
        fold_idx: Fold index (0-4)
        device: Target device
        split: "val" or "train"

    Returns:
        (subject_ids, dataloader) — batched DataLoader for efficient processing.
    """
    splits = load_splits(PROJECT_ROOT / "outputs" / "splits.json")
    subject_ids = get_fold_subjects(splits, fold_idx=fold_idx, split_type=split)

    metadata_path = PROJECT_ROOT / config.data.metadata_path
    metadata = pd.read_csv(metadata_path / "metadata.csv")

    precomputed_dir = PROJECT_ROOT / config.data.precomputed_dir / "rosmap"

    dataset = PrecomputedDataset(
        feature_dir=precomputed_dir,
        subject_ids=subject_ids,
        metadata=metadata,
        subject_column=config.data.get("subject_column", "ROSMAP_IndividualID"),
        target_column=config.data.get("target_column", "cogn_global"),
        pathology_columns=list(config.data.get("pathology_columns", [])),
    )

    # Use dataset's filtered subject_ids (may remove degenerate subjects)
    actual_ids = dataset.subject_ids

    # Batch size 8 — cell transformer needs ~6GB per subject on GPU
    # (31 types × 1000 cells × 4797 genes), so 8 is safe for 48GB GPU
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_for_hgt_multiregion,
        num_workers=0,
    )

    return actual_ids, dataloader, dataset


def move_batch_to_device(batch: dict, device: str) -> dict:
    """Move all tensors in batch to device."""
    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(device)
        else:
            batch_device[k] = v
    return batch_device


def batch_to_model_kwargs(batch: dict) -> dict:
    """Extract model forward-pass kwargs from batch dict."""
    return {
        "region_pseudobulk": batch.get("region_pseudobulk"),
        "region_mask": batch.get("region_mask"),
        "pseudobulk": batch.get("pseudobulk"),
        "ccc_edge_index": batch.get("ccc_edge_index"),
        "ccc_edge_type": batch.get("ccc_edge_type"),
        "ccc_edge_attr": batch.get("ccc_edge_attr"),
        "cell_type_mask": batch.get("cell_type_mask"),
        "pathology": batch.get("pathology"),
        "cognition": batch.get("cognition"),
        "cell_data": batch["cell_data"],
        "cell_offsets": batch["cell_offsets"],
    }


# ── Analysis Functions ───────────────────────────────────────────────────────

def analyze_fusion_weights(model):
    """Analyze fusion layer parameters for the 2-branch architecture.

    Returns dict with fusion type info and parameter statistics.
    For concat-based fusion, decomposes projection weights by branch (HGT, CT).
    For attention-based fusion, reports attention parameter statistics.
    """
    fusion = model.fusion_layer
    fusion_type = type(fusion).__name__

    results = {
        "fusion_type": fusion_type,
        "d_fused": getattr(fusion, 'd_fused', None),
    }

    # Concat-based fusion: decompose projection by branch
    if hasattr(fusion, 'proj'):
        W = fusion.proj.weight.detach().cpu().numpy()
        b = fusion.proj.bias.detach().cpu().numpy()
        d_embed = getattr(fusion, 'd_embed', W.shape[1] // 2)
        d_cell = getattr(fusion, 'd_cell_emb', W.shape[1] - d_embed)

        W_hgt = W[:, :d_embed]
        W_ct = W[:, d_embed:d_embed + d_cell]

        results["weight_frobenius"] = {
            "hgt": float(np.linalg.norm(W_hgt, "fro")),
            "cell_transformer": float(np.linalg.norm(W_ct, "fro")),
        }
        results["weight_mean_abs"] = {
            "hgt": float(np.mean(np.abs(W_hgt))),
            "cell_transformer": float(np.mean(np.abs(W_ct))),
        }
        results["bias_stats"] = {
            "mean": float(np.mean(b)),
            "std": float(np.std(b)),
        }
        results["d_embed"] = d_embed
        results["d_cell_emb"] = d_cell
        results["W_shape"] = list(W.shape)

        total_frob = sum(results["weight_frobenius"].values())
        results["contribution_fraction"] = {
            k: v / total_frob for k, v in results["weight_frobenius"].items()
        }
    else:
        # Attention-based fusion: report total parameter count
        n_params = sum(p.numel() for p in fusion.parameters())
        results["n_parameters"] = n_params

    return results


def extract_embeddings_and_predictions(
    model, guide, subject_ids, dataloader, device: str = "cpu",
):
    """Run forward pass on all subjects (batched), extract embeddings + predictions.

    Returns:
        dict with keys: subject_ids, targets, predictions, branch_embeddings, etc.
    """
    model.eval()
    all_data = {
        "subject_ids": list(subject_ids),
        "targets": [],
        "predictions": [],
        "stds": [],
        "hgt_emb": [],  # [n_subjects, 31, d_embed]
        "ct_emb": [],   # [n_subjects, 31, d_embed]
        "fused": [],    # [n_subjects, 31, d_fused]
        "attended": [],  # [n_subjects, d_fused]
        "attention_weights": [],  # [n_subjects, n_heads, 31]
    }

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            kwargs = batch_to_model_kwargs(batch)
            output = conditioned_forward(
                model, guide, kwargs,
                return_embeddings=True,
                return_hgt_attention=False,
                return_pma_attention=False,
            )

            all_data["targets"].extend(batch["cognition"].cpu().numpy().tolist())
            all_data["predictions"].extend(output["mean"].cpu().numpy().flatten().tolist())
            if "std" in output:
                all_data["stds"].extend(output["std"].cpu().numpy().flatten().tolist())

            emb = output["embeddings"]
            all_data["hgt_emb"].append(emb["hgt"].cpu().numpy())
            all_data["ct_emb"].append(emb["cell"].cpu().numpy())
            all_data["fused"].append(emb["fused"].cpu().numpy())
            all_data["attended"].append(emb["attended"].cpu().numpy())

            if output.get("attention_weights") is not None:
                all_data["attention_weights"].append(
                    output["attention_weights"].cpu().numpy()
                )

    # Concatenate batch arrays
    for key in ["hgt_emb", "ct_emb", "fused", "attended"]:
        all_data[key] = np.concatenate(all_data[key], axis=0)
    if all_data["attention_weights"]:
        all_data["attention_weights"] = np.concatenate(
            all_data["attention_weights"], axis=0
        )

    return all_data


def compute_branch_cka(all_data: dict) -> dict:
    """Compute CKA between 2-branch embedding pairs (HGT, CellTransformer).

    Flattens [n_subjects, 31, d_embed] -> [n_subjects, 31*d_embed] for global CKA.
    Also computes per-cell-type CKA: [n_subjects, d_embed] for each of 31 types.
    """
    n = all_data["hgt_emb"].shape[0]
    d_embed = all_data["hgt_emb"].shape[2]

    # Global CKA (all cell types flattened)
    hgt_flat = all_data["hgt_emb"].reshape(n, -1)
    ct_flat = all_data["ct_emb"].reshape(n, -1)
    fused_flat = all_data["fused"].reshape(n, -1)

    global_cka = {
        "hgt_vs_ct": linear_cka(hgt_flat, ct_flat),
        "hgt_vs_fused": linear_cka(hgt_flat, fused_flat),
        "ct_vs_fused": linear_cka(ct_flat, fused_flat),
    }

    # Per-cell-type CKA
    n_cell_types = all_data["hgt_emb"].shape[1]
    per_type_cka = {"hgt_vs_ct": []}
    for ct_idx in range(n_cell_types):
        hgt_ct = all_data["hgt_emb"][:, ct_idx, :]
        cell_ct = all_data["ct_emb"][:, ct_idx, :d_embed]

        per_type_cka["hgt_vs_ct"].append(linear_cka(hgt_ct, cell_ct))

    return {"global": global_cka, "per_cell_type": per_type_cka}


def analyze_branch_output_distributions(all_data: dict) -> dict:
    """Analyze distribution of branch outputs (magnitude, variance, sparsity)."""
    results = {}
    for branch_name, key in [("hgt", "hgt_emb"),
                              ("cell_transformer", "ct_emb"), ("fused", "fused")]:
        emb = all_data[key]  # [n_subjects, 31, d]
        results[branch_name] = {
            "mean_magnitude": float(np.mean(np.abs(emb))),
            "std_magnitude": float(np.std(np.abs(emb))),
            "l2_norm_per_subject": float(np.mean(np.linalg.norm(
                emb.reshape(emb.shape[0], -1), axis=1
            ))),
            "variance_across_subjects": float(np.mean(np.var(emb, axis=0))),
            "variance_across_cell_types": float(np.mean(np.var(emb, axis=1))),
            "sparsity_fraction": float(np.mean(np.abs(emb) < 1e-3)),
            "max_activation": float(np.max(np.abs(emb))),
        }
    return results


def effective_branch_contributions(model, all_data: dict) -> dict:
    """Compute effective contribution of each branch through fusion projection.

    For concat-based fusion: fused ≈ W_hgt @ hgt + W_ct @ ct + b
    Compute the L2 norm of each branch's contribution to the output.
    For attention-based fusion, uses embedding L2 norms directly.
    """
    fusion = model.fusion_layer

    if hasattr(fusion, 'proj'):
        # Concat-based fusion: decompose by branch
        W = fusion.proj.weight.detach().cpu().numpy()
        d_embed = getattr(fusion, 'd_embed', W.shape[1] // 2)
        d_cell = getattr(fusion, 'd_cell_emb', W.shape[1] - d_embed)

        W_hgt = W[:, :d_embed]
        W_ct = W[:, d_embed:d_embed + d_cell]

        hgt_out = np.einsum("ijk,lk->ijl", all_data["hgt_emb"], W_hgt)
        ct_out = np.einsum("ijk,lk->ijl", all_data["ct_emb"], W_ct)

        hgt_arr = np.mean(np.linalg.norm(hgt_out, axis=2), axis=1)
        ct_arr = np.mean(np.linalg.norm(ct_out, axis=2), axis=1)
    else:
        # Attention-based fusion: use raw embedding norms
        hgt_arr = np.mean(np.linalg.norm(all_data["hgt_emb"], axis=2), axis=1)
        ct_arr = np.mean(np.linalg.norm(all_data["ct_emb"], axis=2), axis=1)

    total = hgt_arr + ct_arr

    return {
        "mean_contribution_norm": {
            "hgt": float(np.mean(hgt_arr)),
            "cell_transformer": float(np.mean(ct_arr)),
        },
        "mean_contribution_fraction": {
            "hgt": float(np.mean(hgt_arr / total)),
            "cell_transformer": float(np.mean(ct_arr / total)),
        },
        "std_contribution_fraction": {
            "hgt": float(np.std(hgt_arr / total)),
            "cell_transformer": float(np.std(ct_arr / total)),
        },
    }


def inference_time_ablation(model, guide, subject_ids, dataloader, device: str = "cpu"):
    """Zero out each branch at inference time in the full model.

    Different from training-time ablation: uses the SAME model weights,
    just sets branch output to zero. Shows how much the full model
    actually relies on each branch for its predictions.
    """
    model.eval()

    # Store original branch flags
    orig_hgt = model.use_hgt_encoder
    orig_ct = model.use_cell_transformer

    configs = {
        "full": (True, True),
        "no_hgt": (False, True),
        "no_cell_transformer": (True, False),
        "hgt_only": (True, False),
        "ct_only": (False, True),
    }

    results = {name: {"predictions": []} for name in configs}

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            kwargs = batch_to_model_kwargs(batch)
            for name, (use_hgt, use_ct) in configs.items():
                model.use_hgt_encoder = use_hgt
                model.use_cell_transformer = use_ct

                output = conditioned_forward(model, guide, kwargs)
                results[name]["predictions"].extend(
                    output["mean"].cpu().numpy().flatten().tolist()
                )

    # Restore original flags
    model.use_hgt_encoder = orig_hgt
    model.use_cell_transformer = orig_ct

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def find_ablation_checkpoints(ablation_name: str) -> list[tuple[int, Path]]:
    """Find best checkpoint for each fold of an ablation experiment."""
    outputs_dir = PROJECT_ROOT / "outputs"
    fold_ckpts = []

    for run_dir in sorted(outputs_dir.glob(f"20260320_*ablation_{ablation_name}*")):
        # Find fold index from config
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        fold_idx = cfg.experiment.fold_idx

        # Find best checkpoint (lowest val_nll)
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        best_ckpt = None
        best_nll = float("inf")
        for ckpt_path in ckpt_dir.glob("epoch=*-val_nll=*.ckpt"):
            nll_str = ckpt_path.stem.split("val_nll=")[1]
            nll = float(nll_str)
            if nll < best_nll:
                best_nll = nll
                best_ckpt = ckpt_path

        if best_ckpt is not None:
            fold_ckpts.append((fold_idx, best_ckpt))

    return sorted(fold_ckpts, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="Fusion analysis")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--output-dir", default="outputs/fusion_analysis",
                        help="Output directory for results")
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Analyze all 5 folds ──────────────────────────────────────────────
    all_fold_results = {}

    for fold_idx in range(5):
        logger.info(f"{'='*60}")
        logger.info(f"Processing fold {fold_idx}")
        logger.info(f"{'='*60}")

        # Load model
        ckpt_path = (
            PROJECT_ROOT / "outputs" / FOLD_DIRS[fold_idx]
            / "checkpoints" / BEST_CKPTS[fold_idx]
        )
        logger.info(f"Loading checkpoint: {ckpt_path.name}")
        model, guide, config, param_store_state = load_model_from_checkpoint(ckpt_path, device)

        # Load BOTH train and val subjects
        val_ids, val_loader, val_ds = load_fold_data(config, fold_idx, device, "val")
        train_ids, train_loader, train_ds = load_fold_data(config, fold_idx, device, "train")
        logger.info(f"Loaded {len(val_ids)} val + {len(train_ids)} train subjects")

        # Prototype guide with a single sample, then restore saved param store
        logger.info("Prototyping guide...")
        single_sample = collate_for_hgt_multiregion([val_ds[0]])
        single_sample = move_batch_to_device(single_sample, device)
        guide(**batch_to_model_kwargs(single_sample))
        del single_sample
        # Now restore saved posterior parameters
        if param_store_state:
            pyro.get_param_store().set_state(param_store_state)

        # ── 1. Extract embeddings on BOTH train and val ──────────────────
        logger.info("Extracting val embeddings and predictions...")
        val_data = extract_embeddings_and_predictions(
            model, guide, val_ids, val_loader, device
        )
        logger.info("Extracting train embeddings and predictions...")
        train_data = extract_embeddings_and_predictions(
            model, guide, train_ids, train_loader, device
        )

        # Merge for CKA (all subjects — maximizes statistical power)
        merged_data = {}
        for key in ["hgt_emb", "ct_emb", "fused", "attended"]:
            merged_data[key] = np.concatenate([train_data[key], val_data[key]], axis=0)
        merged_data["subject_ids"] = train_data["subject_ids"] + val_data["subject_ids"]
        merged_data["targets"] = train_data["targets"] + val_data["targets"]
        merged_data["predictions"] = train_data["predictions"] + val_data["predictions"]

        # ── 2. CKA analysis (on ALL subjects for power) ──────────────────
        logger.info(f"Computing CKA on {merged_data['hgt_emb'].shape[0]} subjects...")
        cka_results = compute_branch_cka(merged_data)

        # Also compute CKA on val-only for unbiased comparison
        cka_val_only = compute_branch_cka(val_data)

        # ── 3. Fusion weight analysis ────────────────────────────────────
        logger.info("Analyzing fusion weights...")
        weight_results = analyze_fusion_weights(model)

        # ── 4. Branch output distributions ───────────────────────────────
        logger.info("Analyzing branch output distributions...")
        dist_results = analyze_branch_output_distributions(merged_data)

        # ── 5. Effective branch contributions ────────────────────────────
        logger.info("Computing effective branch contributions...")
        contrib_results = effective_branch_contributions(model, merged_data)

        # ── 6. Inference-time ablation (on val — metrics need held-out) ──
        logger.info("Running inference-time ablation on val subjects...")
        inf_ablation = inference_time_ablation(
            model, guide, val_ids, val_loader, device
        )

        # ── Compile fold results ─────────────────────────────────────────
        n_total = merged_data["hgt_emb"].shape[0]
        fold_result = {
            "fold_idx": fold_idx,
            "n_val_subjects": len(val_ids),
            "n_train_subjects": len(train_ids),
            "n_total_subjects": n_total,
            "cka_all_subjects": cka_results,
            "cka_val_only": cka_val_only,
            "fusion_weights": weight_results,
            "branch_distributions": dist_results,
            "effective_contributions": contrib_results,
            "predictions_val": {
                "subject_ids": val_data["subject_ids"],
                "targets": val_data["targets"],
                "full_model": val_data["predictions"],
            },
            "predictions_train": {
                "subject_ids": train_data["subject_ids"],
                "targets": train_data["targets"],
                "full_model": train_data["predictions"],
            },
            "inference_ablation": {
                name: data["predictions"]
                for name, data in inf_ablation.items()
            },
        }

        # Compute fold-level metrics for inference-time ablation (val only)
        targets = np.array(val_data["targets"]).flatten()
        for name, preds_list in fold_result["inference_ablation"].items():
            preds = np.array(preds_list).flatten()
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - targets.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            pearson_r = float(np.corrcoef(targets, preds)[0, 1]) if len(targets) > 1 else 0.0
            fold_result[f"inference_ablation_r2_{name}"] = r2
            fold_result[f"inference_ablation_pearson_{name}"] = pearson_r

        all_fold_results[f"fold_{fold_idx}"] = fold_result

        # Print key results for this fold
        logger.info(f"\n── Fold {fold_idx} Results (CKA on {n_total} subjects) ──")
        logger.info(f"CKA (global): HGT-CT={cka_results['global']['hgt_vs_ct']:.3f}")
        logger.info(f"CKA (branch-fused): "
                     f"HGT={cka_results['global']['hgt_vs_fused']:.3f}, "
                     f"CT={cka_results['global']['ct_vs_fused']:.3f}")
        if "contribution_fraction" in weight_results:
            logger.info(f"Fusion weight contribution: "
                         f"HGT={weight_results['contribution_fraction']['hgt']:.3f}, "
                         f"CT={weight_results['contribution_fraction']['cell_transformer']:.3f}")
        logger.info(f"Effective data contribution: "
                     f"HGT={contrib_results['mean_contribution_fraction']['hgt']:.3f}, "
                     f"CT={contrib_results['mean_contribution_fraction']['cell_transformer']:.3f}")

        for name in ["full", "ct_only", "hgt_only", "no_hgt", "no_cell_transformer"]:
            r2_key = f"inference_ablation_r2_{name}"
            logger.info(f"Inference-ablation R² ({name}): {fold_result[r2_key]:.3f}")

        # Cleanup GPU memory
        del model, guide, val_data, train_data, merged_data
        torch.cuda.empty_cache() if "cuda" in device else None

    # ── Aggregate across folds ───────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE RESULTS (mean ± std across 5 folds)")
    logger.info(f"{'='*60}")

    # Aggregate CKA (all subjects)
    cka_keys = ["hgt_vs_ct", "hgt_vs_fused", "ct_vs_fused"]
    agg_cka = {}
    for key in cka_keys:
        vals = [all_fold_results[f"fold_{i}"]["cka_all_subjects"]["global"][key] for i in range(5)]
        agg_cka[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        logger.info(f"CKA {key} (all): {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # CKA val-only for comparison
    logger.info("\nCKA (val-only for unbiased comparison):")
    for key in cka_keys:
        vals = [all_fold_results[f"fold_{i}"]["cka_val_only"]["global"][key] for i in range(5)]
        logger.info(f"CKA {key} (val): {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Aggregate fusion weight contributions (only for concat-based fusion)
    if "contribution_fraction" in all_fold_results.get("fold_0", {}).get("fusion_weights", {}):
        for branch in ["hgt", "cell_transformer"]:
            vals = [all_fold_results[f"fold_{i}"]["fusion_weights"]["contribution_fraction"][branch]
                    for i in range(5)]
            logger.info(f"Fusion weight fraction ({branch}): {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Aggregate effective contributions
    for branch in ["hgt", "cell_transformer"]:
        vals = [all_fold_results[f"fold_{i}"]["effective_contributions"]["mean_contribution_fraction"][branch]
                for i in range(5)]
        logger.info(f"Effective contribution ({branch}): {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Aggregate inference-time ablation R²
    logger.info("\nInference-time ablation R² (using FULL model weights, zeroing branches):")
    for name in ["full", "ct_only", "hgt_only",
                  "no_hgt", "no_cell_transformer"]:
        vals = [all_fold_results[f"fold_{i}"][f"inference_ablation_r2_{name}"] for i in range(5)]
        logger.info(f"  {name}: R²={np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ── Per-sample prediction correlation across branches ────────────────
    logger.info("\nPer-sample prediction correlations (full model vs inference-ablation, val set):")
    for name in ["ct_only", "hgt_only"]:
        corrs = []
        for i in range(5):
            fold = all_fold_results[f"fold_{i}"]
            full_preds = np.array(fold["predictions_val"]["full_model"])
            abl_preds = np.array(fold["inference_ablation"][name])
            if len(full_preds) > 2:
                corrs.append(float(np.corrcoef(full_preds, abl_preds)[0, 1]))
        logger.info(f"  full vs {name}: r={np.mean(corrs):.3f} ± {np.std(corrs):.3f}")

    # CT-only vs HGT-only correlation
    ct_hgt_corrs = []
    for i in range(5):
        fold = all_fold_results[f"fold_{i}"]
        ct_preds = np.array(fold["inference_ablation"]["ct_only"])
        hgt_preds = np.array(fold["inference_ablation"]["hgt_only"])
        if len(ct_preds) > 2:
            ct_hgt_corrs.append(float(np.corrcoef(ct_preds, hgt_preds)[0, 1]))
    logger.info(f"  ct_only vs hgt_only: r={np.mean(ct_hgt_corrs):.3f} ± {np.std(ct_hgt_corrs):.3f}")

    # ── Save results ─────────────────────────────────────────────────────
    # Convert numpy arrays to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    output_data = {
        "aggregate": {
            "cka": agg_cka,
        },
        "per_fold": make_serializable(all_fold_results),
    }

    output_path = output_dir / "fusion_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    # Save per-sample predictions as CSV for easy analysis
    all_preds_rows = []
    for i in range(5):
        fold = all_fold_results[f"fold_{i}"]
        # Val subjects with inference-time ablation
        val_preds = fold["predictions_val"]
        for j in range(len(val_preds["subject_ids"])):
            row = {
                "fold": i,
                "split": "val",
                "subject_id": val_preds["subject_ids"][j],
                "target": val_preds["targets"][j],
                "full_model": val_preds["full_model"][j],
            }
            for name in ["ct_only", "hgt_only",
                          "no_hgt", "no_cell_transformer"]:
                row[f"inf_ablation_{name}"] = fold["inference_ablation"][name][j]
            all_preds_rows.append(row)
        # Train subjects (full model only)
        train_preds = fold["predictions_train"]
        for j in range(len(train_preds["subject_ids"])):
            row = {
                "fold": i,
                "split": "train",
                "subject_id": train_preds["subject_ids"][j],
                "target": train_preds["targets"][j],
                "full_model": train_preds["full_model"][j],
            }
            all_preds_rows.append(row)

    preds_df = pd.DataFrame(all_preds_rows)
    preds_path = output_dir / "per_sample_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"Per-sample predictions saved to {preds_path}")


if __name__ == "__main__":
    main()
