"""Dataset over cached encoder embeddings (ResDec-MHE frozen-encoder path).

Consumes the ``.npz`` written by
``scripts/resdec_mhe/precompute_encoder_embeddings.py`` and yields, per subject:
    - ``attended``:   [d_subject] cached encoder embedding (float32)
    - ``metadata``:   [d_metadata] FiLM-conditioning vector (APOE/sex/age)
    - ``cognition``:  [1] regression target

Used with ``EmbeddingDataModule`` + ``ResDecFrozenLightningModule`` to train
the ResDec-MHE head only, at full-cohort batch, without re-running the frozen
encoder per step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.tabpfn_input import METADATA_FIELDS, load_metadata_vector


class EmbeddingDataset(Dataset):
    """Minimal dataset yielding (embedding, metadata, cogn_global) per subject.

    Assumes encoder embeddings have been precomputed. Subjects missing from
    either the embeddings file or the targets dict are dropped silently — the
    caller is responsible for passing the intersection of the current fold
    split, the cached subjects, and subjects with non-NaN targets.

    Args:
        subject_ids: Subjects to include (typically a fold train/val list).
        embeddings_npz: Path to the cache produced by
            ``scripts/resdec_mhe/precompute_encoder_embeddings.py``.
        targets: Mapping ``subject_id -> cogn_global`` (already filtered to
            non-NaN).
        meta_csv: Path to ``metadata.csv`` (used for FiLM metadata lookup).
    """

    def __init__(
        self,
        subject_ids: list[str],
        embeddings_npz: str | Path,
        targets: dict[str, float],
        meta_csv: str | Path,
    ):
        data = np.load(embeddings_npz, allow_pickle=True)
        all_ids = [str(s) for s in data["subject_ids"]]
        all_embs = data["embeddings"]
        id_to_emb = dict(zip(all_ids, all_embs))

        # Keep only subjects with both a cached embedding AND a target
        self.subject_ids: list[str] = [
            s for s in subject_ids if s in id_to_emb and s in targets
        ]
        if not self.subject_ids:
            raise ValueError(
                f"EmbeddingDataset: no overlap between subject_ids (n={len(subject_ids)}), "
                f"embeddings (n={len(id_to_emb)}), and targets (n={len(targets)})"
            )

        self.embeddings = np.stack(
            [id_to_emb[s] for s in self.subject_ids]
        ).astype(np.float32)
        self.targets = np.array(
            [targets[s] for s in self.subject_ids], dtype=np.float32
        )
        # Precompute 8-dim metadata vectors once per subject (no per-step cost).
        # NOTE: the canonical helper is ``_build_metadata_vectors`` in
        # ``src/data/datasets.py``; this dataset differs in signature (meta_csv
        # is required, no age_mean/age_std propagation) so we inline a minimal
        # variant here rather than bending the helper API to fit both callers.
        meta_csv = Path(meta_csv)
        metas = [load_metadata_vector(s, meta_csv)[0] for s in self.subject_ids]
        self.metadata = torch.stack(metas).float()

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "subject_id": self.subject_ids[idx],
            "attended": torch.from_numpy(self.embeddings[idx]),      # [d_subject]
            "metadata": self.metadata[idx],                           # [d_metadata]
            "cognition": torch.tensor(
                self.targets[idx], dtype=torch.float32
            ).unsqueeze(-1),                                          # [1]
        }


__all__ = ["EmbeddingDataset", "METADATA_FIELDS"]
