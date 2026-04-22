"""LightningDataModule over cached encoder embeddings.

Sibling of ``CognitiveResilienceDataModule`` used only for the ResDec-H3
frozen-encoder path (option 2 of the full-cohort NPT OOM fix). Loads the
precomputed ``.npz`` embedding cache, intersects with fold subjects, and
yields (attended, metadata, cognition) tuples via ``EmbeddingDataset``.
"""
from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import DataLoader

from src.data.embedding_dataset import EmbeddingDataset
from src.data.splits import load_splits


class EmbeddingDataModule(pl.LightningDataModule):
    """DataModule for training the ResDec-H3 head on cached embeddings.

    Args:
        embeddings_npz: Path to cache from
            ``scripts/redesign/precompute_encoder_embeddings.py``.
        splits_path: Path to 5-fold ``splits.json``.
        meta_csv: Path to ``metadata.csv`` (for FiLM metadata + targets).
        fold: CV fold index (0-indexed).
        batch_size: Per-step batch size. Default 500 is a ceiling for
            full-cohort NPT — actual train/val fold sizes are smaller so
            the loader will emit one batch per epoch.
        num_workers: DataLoader workers. Default 0 because the dataset is
            tiny (N x 64 float32 in RAM) and fork() adds no benefit.
        target_col: Metadata CSV column with the regression target.
    """

    def __init__(
        self,
        embeddings_npz: str | Path,
        splits_path: str | Path,
        meta_csv: str | Path,
        fold: int,
        batch_size: int = 500,
        num_workers: int = 0,
        target_col: str = "cogn_global",
    ):
        super().__init__()
        self.embeddings_npz = Path(embeddings_npz)
        self.splits_path = Path(splits_path)
        self.meta_csv = Path(meta_csv)
        self.fold = int(fold)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.target_col = target_col
        self._train_ds: EmbeddingDataset | None = None
        self._val_ds: EmbeddingDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        splits = load_splits(self.splits_path)
        folds = splits["folds"]
        if self.fold < 0 or self.fold >= len(folds):
            raise IndexError(
                f"fold={self.fold} out of range for {len(folds)} folds"
            )
        fold_split = folds[self.fold]

        df = pd.read_csv(self.meta_csv)
        targets: dict[str, float] = {
            r["ROSMAP_IndividualID"]: float(r[self.target_col])
            for _, r in df.iterrows()
            if not pd.isna(r.get(self.target_col))
        }

        self._train_ds = EmbeddingDataset(
            fold_split["train"], self.embeddings_npz, targets, self.meta_csv,
        )
        self._val_ds = EmbeddingDataset(
            fold_split["val"], self.embeddings_npz, targets, self.meta_csv,
        )

    @property
    def train_dataset(self) -> EmbeddingDataset | None:
        return self._train_ds

    @property
    def val_dataset(self) -> EmbeddingDataset | None:
        return self._val_ds

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "setup() must be called first"
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "setup() must be called first"
        # Use the full val set as a single batch — the ResDec-H3 head is
        # cheap enough that there's no benefit to mini-batching on validation.
        return DataLoader(
            self._val_ds,
            batch_size=max(1, len(self._val_ds)),
            shuffle=False,
            num_workers=self.num_workers,
        )


__all__ = ["EmbeddingDataModule"]
