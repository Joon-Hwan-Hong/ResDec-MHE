"""Unit + integration tests for datamodule residualized-target loading."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.feature_loaders import load_residualized_targets


def test_load_residualized_targets_returns_dict_for_known_subjects(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sids = ["A", "B", "C"]
    np.savez(
        cache_dir / "residual_target_fold0.npz",
        fold=0,
        subject_ids=np.array(sids, dtype=object),
        target=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        alpha=0.0, beta_gpath=-0.5,
    )

    out = load_residualized_targets(
        subject_ids=["A", "C"], cache_dir=cache_dir, fold_idx=0,
    )
    assert set(out.keys()) == {"A", "C"}
    assert abs(out["A"] - 0.1) < 1e-6
    assert abs(out["C"] - 0.3) < 1e-6


def test_load_residualized_targets_skips_nan_subjects(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sids = ["A", "B", "C"]
    np.savez(
        cache_dir / "residual_target_fold0.npz",
        fold=0,
        subject_ids=np.array(sids, dtype=object),
        target=np.array([0.1, np.nan, 0.3], dtype=np.float32),
        alpha=0.0, beta_gpath=-0.5,
    )
    out = load_residualized_targets(
        subject_ids=sids, cache_dir=cache_dir, fold_idx=0,
    )
    assert "A" in out and "C" in out
    assert "B" not in out


def test_load_residualized_targets_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_residualized_targets(
            subject_ids=["A"], cache_dir=tmp_path, fold_idx=0,
        )


@pytest.mark.slow
def test_datamodule_uses_residualized_target_when_configured(tmp_path):
    """End-to-end: datamodule reads residualized target when cfg.data.residualize_against is set."""
    pytest.importorskip("lightning")
    from omegaconf import OmegaConf
    from src.data.datamodule import CognitiveResilienceDataModule

    _WT_ROOT = Path(__file__).resolve().parents[3]
    splits = json.loads((_WT_ROOT / "outputs/splits.json").read_text())
    metadata = pd.read_csv(_WT_ROOT / "data/metadata_ROSMAP/metadata.csv")

    cache_dir = _WT_ROOT / "outputs/canonical/variants/gpath_only/cache"
    if not (cache_dir / "residual_target_fold0.npz").exists():
        pytest.skip("residual cache missing; Task 2 smoke run not done")

    cfg = OmegaConf.merge(
        OmegaConf.load(_WT_ROOT / "configs/default.yaml"),
        OmegaConf.load(_WT_ROOT / "configs/resdec_mhe/canonical.yaml"),
        OmegaConf.load(_WT_ROOT / "configs/resdec_mhe/variants/gpath_only.yaml"),
    )
    OmegaConf.set_struct(cfg, False)

    precomputed_dir = _WT_ROOT / "data/precomputed_4796"
    if not precomputed_dir.exists():
        precomputed_dir = _WT_ROOT / "data/precomputed"
    if not precomputed_dir.exists():
        pytest.skip("no precomputed data dir found")

    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata, splits=splits,
        fold_idx=0,
        precomputed_dir=str(precomputed_dir),
        adata=None,
    )
    dm.setup(stage="fit")

    # Validate: dataset target_array should match residualized targets, NOT raw cogn_global.
    train_targets = dm._train_ds._target_array
    raw_cogn = metadata.set_index("ROSMAP_IndividualID")["cogn_global"]

    # Pull a sample of train subject ids
    sids = dm._train_ds.subject_ids[:10]
    residuals = load_residualized_targets(
        subject_ids=list(sids), cache_dir=cache_dir, fold_idx=0,
    )
    for i, sid in enumerate(sids):
        if sid in residuals:
            assert abs(train_targets[i] - residuals[sid]) < 1e-4, \
                f"sid={sid}: dataset target {train_targets[i]} != residual {residuals[sid]}"
            assert abs(train_targets[i] - raw_cogn[sid]) > 1e-6, \
                f"sid={sid}: dataset target {train_targets[i]} == raw cogn {raw_cogn[sid]}"


def test_final_mode_with_variant_config_raises_not_implemented():
    """final_mode + cfg.data.residualize_against must raise NotImplementedError."""
    pytest.importorskip("lightning")
    from omegaconf import OmegaConf
    from src.data.datamodule import CognitiveResilienceDataModule

    _WT_ROOT = Path(__file__).resolve().parents[3]
    splits = json.loads((_WT_ROOT / "outputs/splits.json").read_text())
    metadata = pd.read_csv(_WT_ROOT / "data/metadata_ROSMAP/metadata.csv")
    cfg = OmegaConf.merge(
        OmegaConf.load(_WT_ROOT / "configs/default.yaml"),
        OmegaConf.load(_WT_ROOT / "configs/resdec_mhe/canonical.yaml"),
        OmegaConf.load(_WT_ROOT / "configs/resdec_mhe/variants/gpath_only.yaml"),
    )
    OmegaConf.set_struct(cfg, False)

    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata, splits=splits,
        fold_idx=0,
        precomputed_dir=str(_WT_ROOT / "data/precomputed"),
        adata=None, final_mode=True,
    )
    with pytest.raises(NotImplementedError, match="final_mode=True"):
        dm.setup(stage="fit")
