"""Utility functions and helpers."""

from src.utils.config import (
    load_config,
    save_config,
    merge_configs,
    validate_config,
    flatten_config,
)
from src.utils.hashing import (
    generate_experiment_hash,
    hash_config,
    hash_file,
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
    save_checkpoint,
    load_checkpoint,
    save_attention_weights,
    load_attention_weights,
    save_json,
    load_json,
    # DataFrame utilities
    save_dataframe,
    load_dataframe,
    save_dataframes_multi_format,
)
from src.utils.gpu import (
    get_available_gpus,
    get_gpu_memory_info,
    clear_gpu_memory,
    select_device,
    set_visible_gpus,
    estimate_batch_size,
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
    benjamini_hochberg,
)

__all__ = [
    # Config
    "load_config",
    "save_config",
    "merge_configs",
    "validate_config",
    "flatten_config",
    # Hashing
    "generate_experiment_hash",
    "hash_config",
    "hash_file",
    # Reproducibility
    "set_seed",
    "get_rng_states",
    "set_rng_states",
    # Experiment
    "Experiment",
    "ExperimentManager",
    # IO
    "save_checkpoint",
    "load_checkpoint",
    "save_attention_weights",
    "load_attention_weights",
    "save_json",
    "load_json",
    # DataFrame utilities
    "save_dataframe",
    "load_dataframe",
    "save_dataframes_multi_format",
    # GPU
    "get_available_gpus",
    "get_gpu_memory_info",
    "clear_gpu_memory",
    "select_device",
    "set_visible_gpus",
    "estimate_batch_size",
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
    "benjamini_hochberg",
]