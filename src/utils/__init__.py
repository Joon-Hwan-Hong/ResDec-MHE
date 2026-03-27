"""Utility functions and helpers."""

from src.utils.config import (
    load_config,
    save_config,
    validate_config,
)
from src.utils.hashing import (
    generate_experiment_hash,
    hash_config,
)
from src.utils.reproducibility import (
    set_seed,
    get_rng_states,
    set_rng_states,
)
from src.utils.experiment import (
    Experiment,
    ExperimentManager,
)
from src.utils.io import (
    save_attention_weights,
    load_attention_weights,
    save_json,
    load_json,
    # DataFrame utilities
    save_dataframe,
    load_dataframe,
    save_dataframes_multi_format,
)
from src.utils.device import (
    move_batch_to_device,
)
from src.utils.statistics import (
    CalibrationResult,
    CALIBRATION_LEVELS,
    compute_calibration_metrics,
    calibration_error,
    gini_coefficient,
    cohens_d,
    cohens_d_with_ci,
    cohens_d_vectorized,
    cohens_d_ci_vectorized,
    benjamini_hochberg,
    attention_entropy,
    derive_resilience_groups,
)

__all__ = [
    # Config
    "load_config",
    "save_config",
    "validate_config",
    # Hashing
    "generate_experiment_hash",
    "hash_config",
    # Reproducibility
    "set_seed",
    "get_rng_states",
    "set_rng_states",
    # Experiment
    "Experiment",
    "ExperimentManager",
    # IO
    "save_attention_weights",
    "load_attention_weights",
    "save_json",
    "load_json",
    # DataFrame utilities
    "save_dataframe",
    "load_dataframe",
    "save_dataframes_multi_format",
    # Device
    "move_batch_to_device",
    # Statistics
    "CalibrationResult",
    "CALIBRATION_LEVELS",
    "compute_calibration_metrics",
    "calibration_error",
    "gini_coefficient",
    "cohens_d",
    "cohens_d_with_ci",
    "cohens_d_vectorized",
    "cohens_d_ci_vectorized",
    "benjamini_hochberg",
    "attention_entropy",
    "derive_resilience_groups",
]