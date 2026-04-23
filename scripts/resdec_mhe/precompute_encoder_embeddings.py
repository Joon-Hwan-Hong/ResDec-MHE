"""Precompute CognitiveResilienceModel encoder embeddings for all subjects.

Loads each precomputed .pt file in ``data/precomputed/``, wraps it as a
single-subject batch, runs the encoder forward with ``torch.no_grad()`` in
``.eval()`` mode, and extracts ``output['attended']`` ([1, d_fused=64]).
Writes all subjects to a single ``.npz`` file.

This is option (2) of the full-cohort NPT OOM fix: freeze the encoder and
precompute per-subject embeddings once, then train the ResDec-MHE head at
full-cohort batch (500) without OOM.

Why this is safe:
- The encoder is used as a frozen feature extractor under P5.
- ``attended`` is a per-subject function of per-subject inputs only; no
  cross-subject interactions occur inside the encoder proper.
- Dropout is disabled by ``.eval()``, so cached values are deterministic.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/precompute_encoder_embeddings.py

Output
------
    data/redesign/encoder_embeddings.npz with keys:
        subject_ids:          [N] object array of ROSMAP IDs
        embeddings:           [N, d_subject] float32
        d_subject:            int
        encoder_config_hash:  string (hashlib.sha256 of the encoder config YAML)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.constants import N_REGIONS, PFC_REGION_IDX
from src.models.full_model import build_model_from_config

logger = logging.getLogger(__name__)


def _build_single_subject_batch(pt_data: dict, device: torch.device) -> dict:
    """Wrap a single precomputed .pt subject dict as a batch-of-1 on device.

    Mirrors the output shape of ``collate_for_hgt_multiregion`` but for a
    single subject. Key mapping:
        - region_pseudobulk: [1, N_REGIONS, C, G] — assembled from pseudobulk
          + region_{idx}_pseudobulk keys, falling back to PFC for single-
          region subjects.
        - region_mask: [1, N_REGIONS] bool
        - ccc_edge_*: [2, E] / [E] / [E, 1] (no node offsets needed for B=1)
        - cell_data: [total_cells, G], cell_offsets: [1, C+1]
        - cell_type_mask: [1, C] bool
        - pathology: [1, 3] (zeros — not consumed by any branch relevant to
          `attended`)
        - cognition: [1, 1] (zeros — target, only needed for Bayesian head
          which is disabled under deterministic head mode)
    """
    pb = pt_data["pseudobulk"]              # [C, G] tensor
    n_cell_types, n_genes = pb.shape

    # Assemble multi-region pseudobulk [1, N_REGIONS, C, G] from per-region
    # keys, falling back to the aggregate pseudobulk for PFC when only PFC
    # data was saved.
    region_pb = torch.zeros(1, N_REGIONS, n_cell_types, n_genes, dtype=torch.float32)
    region_mask = torch.zeros(1, N_REGIONS, dtype=torch.bool)
    avail = list(pt_data.get("available_regions", [PFC_REGION_IDX]))
    for ridx in avail:
        rkey = f"region_{ridx}_pseudobulk"
        if rkey in pt_data:
            region_pb[0, ridx] = pt_data[rkey]
            region_mask[0, ridx] = True
        elif ridx == PFC_REGION_IDX:
            region_pb[0, ridx] = pb
            region_mask[0, ridx] = True

    cell_type_mask = pt_data["cell_type_mask"].bool().unsqueeze(0)    # [1, C]
    cell_data = pt_data["cell_data"].float()                          # [n_cells, G]
    cell_offsets = pt_data["cell_offsets"].long().unsqueeze(0)        # [1, C+1]

    ccc_edge_index = pt_data["ccc_edge_index"].long()                 # [2, E]
    ccc_edge_type = pt_data["ccc_edge_type"].long()                   # [E]
    ccc_edge_attr = pt_data["ccc_edge_attr"].float()                  # [E, 1]

    pathology = torch.zeros(1, 3, dtype=torch.float32)
    cognition = torch.zeros(1, 1, dtype=torch.float32)

    batch = {
        "region_pseudobulk": region_pb.to(device),
        "region_mask": region_mask.to(device),
        "ccc_edge_index": ccc_edge_index.to(device),
        "ccc_edge_type": ccc_edge_type.to(device),
        "ccc_edge_attr": ccc_edge_attr.to(device),
        "cell_type_mask": cell_type_mask.to(device),
        "cell_data": cell_data.to(device),
        "cell_offsets": cell_offsets.to(device),
        "pathology": pathology.to(device),
        "cognition": cognition.to(device),
    }
    return batch


def _encoder_config_hash(cfg) -> str:
    """Return a short hash of the encoder config for reproducibility metadata."""
    yaml_str = OmegaConf.to_yaml(cfg)
    return hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()[:16]


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ------------------------------------------------------------------ #
    # Config + model                                                     #
    # ------------------------------------------------------------------ #
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg.model, False)
    # Force deterministic head so the Bayesian SVI path (which needs a Pyro
    # guide and real targets) is not engaged. The ResDec-MHE frozen path only
    # needs the `attended` embedding; the encoder's own prediction_head is
    # ignored.
    cfg.model.head.type = "deterministic"
    cfg.model.n_genes = int(cfg.model.get("n_genes") or 4785)
    cfg.model.n_cell_types = int(cfg.model.get("n_cell_types") or 31)
    # Gradient checkpointing is a training-time memory optimization; disable
    # it for inference so we don't waste time saving activations.
    cfg.model.use_gradient_checkpointing = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Building encoder on %s (n_genes=%d, n_cell_types=%d)",
                device, cfg.model.n_genes, cfg.model.n_cell_types)
    model = build_model_from_config(cfg.model).to(device).eval()

    # Load trained encoder weights from a Lightning checkpoint if provided.
    # Expected source: ResDecLightningModule checkpoint (live-encoder mode) where
    # the encoder state_dict lives under the `encoder.*` prefix within the LM.
    if args.encoder_ckpt is not None:
        logger.info("Loading encoder weights from %s", args.encoder_ckpt)
        ckpt = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Extract keys under the "encoder." prefix (Lightning nesting)
        encoder_sd = {
            k[len("encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        if not encoder_sd:
            # Checkpoint may already be just the encoder state dict (no prefix)
            logger.warning("No 'encoder.' prefix found; attempting direct load of full state_dict")
            encoder_sd = state_dict
        missing, unexpected = model.load_state_dict(encoder_sd, strict=False)
        # prediction_head keys may be missing if frozen during training, and
        # Bayesian-head keys may be unexpected if ckpt came from a different head type.
        if missing:
            logger.info("  load_state_dict: %d missing keys (first 5): %s",
                        len(missing), list(missing)[:5])
        if unexpected:
            logger.info("  load_state_dict: %d unexpected keys (first 5): %s",
                        len(unexpected), list(unexpected)[:5])
        logger.info("Encoder weights loaded (strict=False)")
    else:
        logger.warning(
            "No --encoder-ckpt provided; using RANDOM-INIT encoder. "
            "Downstream head training will see random-projection features. "
            "For real training pass --encoder-ckpt <path-to-trained.ckpt>."
        )

    d_subject = int(cfg.model.d_fused)
    cfg_hash = _encoder_config_hash(cfg.model)
    logger.info("encoder_config_hash=%s, d_subject=%d", cfg_hash, d_subject)

    # ------------------------------------------------------------------ #
    # Enumerate precomputed subjects                                     #
    # ------------------------------------------------------------------ #
    precomputed_dir = Path(args.precomputed_dir)
    pt_files = sorted(precomputed_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {precomputed_dir}")
    logger.info("Found %d precomputed .pt files in %s", len(pt_files), precomputed_dir)

    # ------------------------------------------------------------------ #
    # Encode each subject                                                #
    # ------------------------------------------------------------------ #
    subject_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    t0 = time.monotonic()
    with torch.no_grad():
        for i, pt_path in enumerate(pt_files):
            sid = pt_path.stem
            try:
                pt_data = torch.load(pt_path, weights_only=False, map_location="cpu")
            except Exception as e:
                logger.warning("Skipping %s: failed to load (%s)", sid, e)
                continue

            batch = _build_single_subject_batch(pt_data, device)
            out = model(
                region_pseudobulk=batch["region_pseudobulk"],
                region_mask=batch["region_mask"],
                ccc_edge_index=batch["ccc_edge_index"],
                ccc_edge_type=batch["ccc_edge_type"],
                ccc_edge_attr=batch["ccc_edge_attr"],
                cell_type_mask=batch["cell_type_mask"],
                cell_data=batch["cell_data"],
                cell_offsets=batch["cell_offsets"],
                pathology=batch["pathology"],
                cognition=batch["cognition"],
            )
            attended = out["attended"]  # [1, d_subject]
            if attended.shape != (1, d_subject):
                raise RuntimeError(
                    f"Subject {sid}: expected attended shape [1, {d_subject}], "
                    f"got {tuple(attended.shape)}"
                )
            subject_ids.append(sid)
            embeddings.append(attended.squeeze(0).detach().float().cpu().numpy())

            if (i + 1) % 50 == 0 or (i + 1) == len(pt_files):
                elapsed = time.monotonic() - t0
                logger.info("Encoded %d/%d (%.1fs)", i + 1, len(pt_files), elapsed)

    if not embeddings:
        raise RuntimeError("No subjects were encoded — check precomputed_dir")

    # ------------------------------------------------------------------ #
    # Save                                                               #
    # ------------------------------------------------------------------ #
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subject_ids_arr = np.array(subject_ids, dtype=object)
    embeddings_arr = np.stack(embeddings).astype(np.float32)

    np.savez(
        out_path,
        subject_ids=subject_ids_arr,
        embeddings=embeddings_arr,
        d_subject=np.int32(d_subject),
        encoder_config_hash=np.array(cfg_hash, dtype=object),
    )
    elapsed = time.monotonic() - t0
    logger.info(
        "Wrote %d subjects to %s (%.2f MB, %.1fs wall-clock)",
        len(subject_ids),
        out_path,
        out_path.stat().st_size / 1e6,
        elapsed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Model config (default: configs/default.yaml). Must define model.d_fused "
             "and match the encoder architecture used for downstream frozen-head training.",
    )
    p.add_argument(
        "--precomputed-dir",
        default="data/precomputed/",
        help="Directory containing per-subject .pt files.",
    )
    p.add_argument(
        "--output",
        default="data/redesign/encoder_embeddings.npz",
        help="Path to write the cached embeddings npz.",
    )
    p.add_argument(
        "--encoder-ckpt",
        default=None,
        help="Path to a Lightning checkpoint from live-encoder Phase 1 training. "
             "If provided, loads encoder.* keys into the model before running forward passes. "
             "If omitted, uses random init (sanity-check mode only).",
    )
    main(p.parse_args())
