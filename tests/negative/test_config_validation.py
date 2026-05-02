"""Negative tests for config validation.

Verifies that validate_config rejects bad configurations with clear errors
and accepts valid ones.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from omegaconf import OmegaConf, DictConfig

from src.utils.config import validate_config

# Standard required keys used in training entrypoint
REQUIRED_KEYS = ["experiment", "data", "model", "training", "paths"]


@pytest.fixture
def default_config(default_config_path) -> DictConfig:
    """Load the default config as a mutable DictConfig.

    Path resolves via the ``default_config_path`` fixture (worktree-rooted),
    so this test works regardless of pytest's invocation cwd.
    """
    with open(default_config_path) as f:
        return OmegaConf.create(yaml.safe_load(f))

# ── 1. Missing required key ─────────────────────────────────────────────────

def test_missing_required_key(default_config: DictConfig) -> None:
    """Removing the 'model' top-level key should raise ValueError listing it."""
    del default_config["model"]
    with pytest.raises(ValueError, match=r"Missing required keys.*model"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 2. lr is string ─────────────────────────────────────────────────────────

def test_lr_is_string(default_config: DictConfig) -> None:
    """Setting lr to a string should raise ValueError mentioning type."""
    default_config.training.optimizer.lr = "fast"
    with pytest.raises(ValueError, match=r"training\.optimizer\.lr.*expected.*got str"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 3. lr is negative ───────────────────────────────────────────────────────

def test_lr_is_negative(default_config: DictConfig) -> None:
    """Negative lr should raise ValueError mentioning invalid value."""
    default_config.training.optimizer.lr = -0.001
    with pytest.raises(ValueError, match=r"training\.optimizer\.lr.*invalid value"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 4. dropout above one ────────────────────────────────────────────────────

def test_dropout_above_one(default_config: DictConfig) -> None:
    """dropout=1.5 should fail the 0 <= v < 1 range check."""
    default_config.model.dropout = 1.5
    with pytest.raises(ValueError, match=r"model\.dropout.*invalid value"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 5. head type invalid ────────────────────────────────────────────────────

def test_head_type_invalid(default_config: DictConfig) -> None:
    """head.type='transformer' is not in the allowed enum."""
    default_config.model.head.type = "transformer"
    with pytest.raises(ValueError, match=r"model\.head\.type.*invalid value.*transformer"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 6. loss type invalid ────────────────────────────────────────────────────

def test_loss_type_invalid(default_config: DictConfig) -> None:
    """loss.type='l1' is not in the allowed enum."""
    default_config.training.loss.type = "l1"
    with pytest.raises(ValueError, match=r"training\.loss\.type.*invalid value.*l1"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 7. batch_size zero ──────────────────────────────────────────────────────

def test_batch_size_zero(default_config: DictConfig) -> None:
    """batch_size=0 should fail the v > 0 range check."""
    default_config.data.dataloader.batch_size = 0
    with pytest.raises(ValueError, match=r"data\.dataloader\.batch_size.*invalid value"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 8. max_epochs float ─────────────────────────────────────────────────────

def test_max_epochs_float(default_config: DictConfig) -> None:
    """max_epochs=10.5 should fail the int type check."""
    default_config.training.max_epochs = 10.5
    with pytest.raises(ValueError, match=r"training\.max_epochs.*expected int.*got float"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 9. precision invalid ────────────────────────────────────────────────────

def test_precision_invalid(default_config: DictConfig) -> None:
    """precision='8-bit' is not in the allowed enum."""
    default_config.training.precision = "8-bit"
    with pytest.raises(ValueError, match=r"training\.precision.*invalid value.*8-bit"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 10. valid config passes ─────────────────────────────────────────────────

def test_valid_config_passes(default_config: DictConfig) -> None:
    """The default config should pass validation without error."""
    # Should not raise
    validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 11. scheduler type invalid ────────────────────────────────────────────

def test_scheduler_type_invalid(default_config: DictConfig) -> None:
    """scheduler.type='warmup' is not in the allowed enum."""
    default_config.training.scheduler.type = "warmup"
    with pytest.raises(ValueError, match=r"training\.scheduler\.type.*invalid value.*warmup"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 12. optimizer type invalid ────────────────────────────────────────────

def test_optimizer_type_invalid(default_config: DictConfig) -> None:
    """optimizer.type='rmsprop' is not in the allowed enum."""
    default_config.training.optimizer.type = "rmsprop"
    with pytest.raises(ValueError, match=r"training\.optimizer\.type.*invalid value.*rmsprop"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 13. sampling strategy invalid ────────────────────────────────────────

def test_sampling_strategy_invalid(default_config: DictConfig) -> None:
    """sampling_strategy='balanced' is not in the allowed enum."""
    default_config.data.cell_sampling.sampling_strategy = "balanced"
    with pytest.raises(ValueError, match=r"data\.cell_sampling\.sampling_strategy.*invalid value"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 14. temperature annealing schedule invalid ───────────────────────────

def test_annealing_schedule_invalid(default_config: DictConfig) -> None:
    """schedule='step' is not in the allowed enum."""
    default_config.training.temperature_annealing.schedule = "step"
    with pytest.raises(ValueError, match=r"training\.temperature_annealing\.schedule.*invalid value"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 15. test_frac out of range ─────────────────────────────────────────

def test_split_test_frac_out_of_range(default_config: DictConfig) -> None:
    """test_frac must be in (0, 1)."""
    default_config.data.splits.test_frac = 1.5
    with pytest.raises(ValueError, match=r"data\.splits\.test_frac.*invalid value"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 16. tau_min >= tau_max ───────────────────────────────────────────────

def test_tau_min_gte_tau_max(default_config: DictConfig) -> None:
    """tau_min >= tau_max should fail cross-field validation."""
    default_config.training.temperature_annealing.tau_min = 5.0
    default_config.training.temperature_annealing.tau_max = 2.0
    with pytest.raises(ValueError, match=r"tau_min.*must be.*tau_max"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 17. eta_min >= lr ────────────────────────────────────────────────────

def test_eta_min_gte_lr(default_config: DictConfig) -> None:
    """eta_min >= lr should fail cross-field validation."""
    default_config.training.scheduler.eta_min = 0.01
    default_config.training.optimizer.lr = 0.001
    with pytest.raises(ValueError, match=r"eta_min.*must be.*lr"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)

# ── 18. n_regions mismatch ──────────────────────────────────────────────

def test_n_regions_mismatch(default_config: DictConfig) -> None:
    """model.n_regions != N_REGIONS constant should fail validation."""
    default_config.model.n_regions = 99
    with pytest.raises(ValueError, match=r"n_regions.*fixed by dataset schema"):
        validate_config(default_config, required_keys=REQUIRED_KEYS)
