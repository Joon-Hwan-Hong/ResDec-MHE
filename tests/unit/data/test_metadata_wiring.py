"""Tests for Task A.1 metadata wiring: datamodule → dataset → lightning module.

Three cases:
    (a) ``load_metadata_vector`` for a present subject produces the expected
        non-zero 8-dim vector (APOE e3+e4, sex, z-scored age at mean = 0).
    (b) ``load_metadata_vector`` for a missing subject returns a vector with
        only the three missingness bits set.
    (c) Datamodule computes ``age_mean``/``age_std`` from TRAIN subjects only
        (leakage check) and the resulting dataset returns per-subject
        ``metadata`` tensors z-scored with those train-only stats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.data.tabpfn_input import load_metadata_vector


def test_load_metadata_vector_present_subject(tmp_path):
    """Present subject with APOE 34 produces the expected 8-dim vector."""
    df = pd.DataFrame({
        "ROSMAP_IndividualID": ["R0000001"],
        "apoe_genotype": [34],
        "msex": [1],
        "age_death": [86.0],
    })
    csv = tmp_path / "metadata.csv"
    df.to_csv(csv, index=False)

    vec, _ = load_metadata_vector(
        "R0000001", csv, age_mean=86.0, age_std=6.5,
    )
    assert vec.shape == (8,)
    # APOE 34 → e3 and e4 present, e2 absent, not missing
    assert vec[0].item() == 0.0  # e2 absent
    assert vec[1].item() == 1.0  # e3 present
    assert vec[2].item() == 1.0  # e4 present
    assert vec[3].item() == 0.0  # apoe not missing
    # Sex present, not missing
    assert vec[4].item() == 1.0
    assert vec[5].item() == 0.0
    # Age at the mean → z-score = 0, not missing
    assert vec[6].item() == pytest.approx(0.0)
    assert vec[7].item() == 0.0


def test_load_metadata_vector_missing_subject(tmp_path):
    """Unknown subject → only missingness bits (indices 3, 5, 7) set."""
    df = pd.DataFrame({
        "ROSMAP_IndividualID": ["R9999"],
        "apoe_genotype": [33],
        "msex": [0],
        "age_death": [80.0],
    })
    csv = tmp_path / "metadata.csv"
    df.to_csv(csv, index=False)

    vec, _ = load_metadata_vector("R0000001", csv)
    assert vec[3].item() == 1.0  # apoe_missing
    assert vec[5].item() == 1.0  # sex_missing
    assert vec[7].item() == 1.0  # age_missing
    # Value slots must stay zero
    assert vec[0:3].sum().item() == 0.0
    assert vec[4].item() == 0.0
    assert vec[6].item() == 0.0


def _minimal_datamodule_config() -> OmegaConf:
    """Build a minimal OmegaConf for CognitiveResilienceDataModule."""
    return OmegaConf.create({
        "data": {
            "cell_type_column": "supercluster_name",
            "subject_column": "ROSMAP_IndividualID",
            "target_column": "cogn_global",
            "pathology_columns": [],
            "metadata_path": None,  # will be injected per-test
            "cell_sampling": {
                "max_cells_per_type": 10,
                "min_cells_threshold": 2,
                "sampling_strategy": "random",
            },
            "dataloader": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "prefetch_factor": None,
            },
        },
        "experiment": {"seed": 42},
    })


def _write_precomputed_fixture(
    dir_path: Path,
    subject_ids: list[str],
    n_cell_types: int = 31,
    n_genes: int = 4,
) -> None:
    """Write minimal per-subject .pt files for PrecomputedDataset."""
    from src.data.constants import CELL_TYPE_ORDER, REGION_ORDER

    max_cells = 3
    n_regions = len(REGION_ORDER)
    total_cells = n_cell_types * max_cells

    for sid in subject_ids:
        cell_counts = torch.full((n_cell_types,), max_cells, dtype=torch.long)
        cell_offsets = torch.zeros(n_cell_types + 1, dtype=torch.long)
        for ct in range(n_cell_types):
            cell_offsets[ct + 1] = cell_offsets[ct] + max_cells
        torch.save(
            {
                "pseudobulk": torch.randn(n_cell_types, n_genes),
                "cell_type_mask": torch.ones(n_cell_types, dtype=torch.bool),
                "cell_counts": cell_counts,
                "region_mask": torch.tensor(
                    [True] + [False] * (n_regions - 1), dtype=torch.bool,
                ),
                "cell_data": torch.randn(total_cells, n_genes),
                "cell_offsets": cell_offsets,
                "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long),
                "ccc_edge_type": torch.zeros(0, dtype=torch.long),
                "ccc_edge_attr": torch.zeros(0, 1),
                "cell_type_order": list(CELL_TYPE_ORDER),
                "available_regions": [0],
            },
            dir_path / f"{sid}.pt",
        )


def test_datamodule_age_stats_train_only(tmp_path):
    """Train/val age_mean & age_std must come from TRAIN subjects only.

    Build a tiny fixture where TRAIN subjects have ages ~60 and VAL subjects
    have ages ~100. Pooled mean would be ~80; train-only mean should be ~60.
    After wiring, the dataset's metadata tensor for a train subject at the
    train-only mean must z-score to ~0.0, not to ~(-20/std).
    """
    from src.data.datamodule import CognitiveResilienceDataModule

    rng = np.random.default_rng(0)
    train_ids = [f"R_train_{i:02d}" for i in range(10)]
    val_ids = [f"R_val_{i:02d}" for i in range(5)]
    all_ids = train_ids + val_ids

    # Train ages ~60, val ages ~100: pooled mean drifts well above train-only mean.
    train_ages = np.linspace(58.0, 62.0, len(train_ids))
    val_ages = np.linspace(98.0, 102.0, len(val_ids))
    ages = list(train_ages) + list(val_ages)

    metadata = pd.DataFrame({
        "ROSMAP_IndividualID": all_ids,
        "cogn_global": rng.standard_normal(len(all_ids)),
        "apoe_genotype": [33] * len(all_ids),
        "msex": [0] * len(all_ids),
        "age_death": ages,
    })

    meta_dir = tmp_path / "metadata_ROSMAP"
    meta_dir.mkdir()
    metadata.to_csv(meta_dir / "metadata.csv", index=False)

    pt_dir = tmp_path / "precomputed"
    pt_dir.mkdir()
    _write_precomputed_fixture(pt_dir, all_ids)

    splits = {
        "holdout_test": [],
        "train_val_pool": all_ids,
        "folds": [{"train": train_ids, "val": val_ids}],
    }

    cfg = _minimal_datamodule_config()
    OmegaConf.set_struct(cfg, False)
    cfg.data.metadata_path = str(meta_dir)

    dm = CognitiveResilienceDataModule(
        config=cfg,
        metadata=metadata,
        splits=splits,
        fold_idx=0,
        precomputed_dir=pt_dir,
    )
    dm.setup(stage="fit")

    # Datamodule must expose the train-only age stats somewhere (property or
    # attribute) — without this, leakage occurred silently.
    expected_mean = float(np.mean(train_ages))
    expected_std = float(np.std(train_ages, ddof=0))

    # Train dataset receives the train-only stats.
    train_ds = dm._train_ds
    assert hasattr(train_ds, "age_mean"), (
        "Dataset is missing age_mean attribute (leakage guard not wired)."
    )
    assert hasattr(train_ds, "age_std"), (
        "Dataset is missing age_std attribute (leakage guard not wired)."
    )
    assert float(train_ds.age_mean) == pytest.approx(expected_mean, abs=1e-4)
    assert float(train_ds.age_std) == pytest.approx(expected_std, abs=1e-4)

    # Val dataset must reuse the SAME train-only stats (not its own val-only
    # stats, which would also be leakage of a different flavour).
    val_ds = dm._val_ds
    assert float(val_ds.age_mean) == pytest.approx(expected_mean, abs=1e-4)
    assert float(val_ds.age_std) == pytest.approx(expected_std, abs=1e-4)

    # A train subject whose age equals the train-only mean should z-score to 0.
    # Find the train subject closest to the train-only mean for a clean check.
    closest_idx = int(np.argmin(np.abs(np.array(train_ages) - expected_mean)))
    closest_sid = train_ids[closest_idx]
    sample = train_ds[train_ds.subject_ids.index(closest_sid)]
    assert "metadata" in sample, (
        "Dataset __getitem__ must include 'metadata' key after wiring."
    )
    md = sample["metadata"]
    assert md.shape == (8,)
    # Age z-score using train-only stats ≈ 0 for subject at train-only mean.
    expected_z = (train_ages[closest_idx] - expected_mean) / expected_std
    assert md[6].item() == pytest.approx(expected_z, abs=1e-4)

    # Sanity: the pooled mean (all ages) differs from train-only mean, so if
    # the wiring were pooling (leakage), age z would NOT be ~0 here.
    pooled_mean = float(np.mean(ages))
    assert abs(pooled_mean - expected_mean) > 1.0, (
        "Fixture is degenerate — pooled and train-only means should differ."
    )


def test_cognitive_resilience_dataset_metadata_wiring(tmp_path):
    """CognitiveResilienceDataset.__getitem__ returns correct FiLM metadata.

    Builds a minimal AnnData fixture (a few subjects, few cells, few genes) and
    constructs the dataset with explicit ``meta_csv`` / ``age_mean`` / ``age_std``.
    Retrieves one sample via ``__getitem__`` and asserts the ``metadata`` tensor
    has the correct shape, the age slot matches the expected z-score, and the
    APOE slot reflects the subject's apoe_genotype (34 → e3 + e4 set).
    """
    import anndata
    from src.data.constants import CELL_TYPE_ORDER
    from src.data.datasets import CognitiveResilienceDataset

    rng = np.random.default_rng(0)

    # Three subjects: two train-like (APOE 34 for the target subject, 33 for
    # a second), one with a different APOE to verify per-row wiring.
    subject_ids = ["R_A", "R_B", "R_C"]
    ages = [86.0, 80.0, 92.0]
    apoe = [34, 33, 24]
    sexes = [1, 0, 1]

    # Minimal AnnData: 2 cells per subject across a couple of cell types.
    n_cells_per_subject = 2
    n_genes = 5
    total_cells = len(subject_ids) * n_cells_per_subject
    X = rng.standard_normal((total_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({
        "ROSMAP_IndividualID": np.repeat(subject_ids, n_cells_per_subject),
        "supercluster_name": np.tile([CELL_TYPE_ORDER[0]] * n_cells_per_subject, len(subject_ids)),
        "BrainRegion": ["PFC"] * total_cells,
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    metadata_df = pd.DataFrame({
        "ROSMAP_IndividualID": subject_ids,
        "apoe_genotype": apoe,
        "msex": sexes,
        "age_death": ages,
        "gpath": [0.1, 0.2, 0.3],
        "amylsqrt": [0.4, 0.5, 0.6],
        "tangsqrt": [0.7, 0.8, 0.9],
        "cogn_global": [0.0, -0.5, 0.5],
    })

    meta_csv = tmp_path / "metadata.csv"
    metadata_df.to_csv(meta_csv, index=False)

    age_mean = 85.0
    age_std = 5.0

    ds = CognitiveResilienceDataset(
        adata=adata,
        metadata=metadata_df,
        subject_ids=subject_ids,
        # Relax the cell-count threshold so the tiny fixture is admissible.
        min_cells_threshold=1,
        max_cells_per_type=8,
        meta_csv=meta_csv,
        age_mean=age_mean,
        age_std=age_std,
    )

    # Drive __getitem__ (not just __init__) to verify the metadata slot is
    # threaded through to the per-sample dict.
    target_idx = ds.subject_ids.index("R_A")
    sample = ds[target_idx]
    assert "metadata" in sample, (
        "CognitiveResilienceDataset.__getitem__ must include 'metadata' when "
        "meta_csv is wired."
    )
    md = sample["metadata"]
    assert md.shape == (8,)
    # Age slot (index 6) = (age - age_mean) / age_std
    expected_z = (ages[0] - age_mean) / age_std
    assert md[6].item() == pytest.approx(expected_z, abs=1e-5)
    assert md[7].item() == 0.0  # age_missing
    # APOE 34 → indices 1 (e3) and 2 (e4) set, e2 absent, apoe_missing = 0
    assert md[0].item() == 0.0  # e2 absent
    assert md[1].item() == 1.0  # e3 present
    assert md[2].item() == 1.0  # e4 present
    assert md[3].item() == 0.0  # apoe_missing
