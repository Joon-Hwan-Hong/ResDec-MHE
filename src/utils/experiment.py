"""
Experiment management utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any

from src.utils.hashing import generate_experiment_hash
from src.utils.config import save_config


@dataclass
class Experiment:
    """
    Represents a single experiment with its directory structure and metadata.
    """

    exp_dir: Path
    config: dict[str, Any]
    exp_hash: str
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def checkpoints_dir(self) -> Path:
        return self.exp_dir / "checkpoints"

    @property
    def model_dir(self) -> Path:
        return self.exp_dir / "model"

    @property
    def logs_dir(self) -> Path:
        return self.exp_dir / "logs"

    @property
    def tensorboard_dir(self) -> Path:
        return self.logs_dir / "tensorboard"


class ExperimentManager:
    """
    Manages experiment directories and artifacts.

    Handles:
    - Creating experiment directory structures
    - Saving/loading configurations
    - Tracking experiment metadata
    """

    def __init__(self, base_dir: str | Path = "experiments"):
        """
        Initialize experiment manager.

        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, config: dict[str, Any]) -> Experiment:
        """
        Create a new experiment with directory structure.

        Args:
            config: Experiment configuration

        Returns:
            Experiment object with paths and metadata
        """
        exp_hash = generate_experiment_hash(config)
        exp_dir = self.base_dir / exp_hash

        # Create subdirectories (figure dirs match generate_plots.py categories)
        subdirs = [
            "checkpoints",
            "model",
            "analysis",
            "figures/training",
            "figures/attention",
            "figures/importance",
            "figures/prediction",
            "figures/embedding",
            "figures/resilience",
            "preprocessing",
            "logs/tensorboard",
        ]

        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(config, exp_dir / "config.yaml")

        return Experiment(
            exp_dir=exp_dir,
            config=config,
            exp_hash=exp_hash,
        )

