"""
Experiment management utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any

from src.utils.hashing import generate_experiment_hash
from src.utils.config import load_config, save_config


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
    def analysis_dir(self) -> Path:
        return self.exp_dir / "analysis"

    @property
    def preprocessing_dir(self) -> Path:
        return self.exp_dir / "preprocessing"

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

        # Create subdirectories
        subdirs = [
            "checkpoints",
            "model",
            "analysis",
            "analysis/cell_heterogeneity",
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

    def load_experiment(self, exp_hash: str) -> Experiment:
        """
        Load an existing experiment by hash.

        Args:
            exp_hash: Experiment hash identifier

        Returns:
            Experiment object

        Raises:
            FileNotFoundError: If experiment doesn't exist
        """
        exp_dir = self.base_dir / exp_hash
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {exp_hash}")

        config_path = exp_dir / "config.yaml"
        config = load_config(config_path)

        return Experiment(
            exp_dir=exp_dir,
            config=config,
            exp_hash=exp_hash,
        )

    def list_experiments(self) -> list[str]:
        """
        List all experiment hashes in base directory.

        Returns:
            List of experiment hash strings
        """
        return [
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "config.yaml").exists()
        ]

    def get_latest_experiment(self) -> Experiment | None:
        """
        Get the most recently created experiment.

        Returns:
            Latest Experiment or None if no experiments exist
        """
        experiments = self.list_experiments()
        if not experiments:
            return None

        # Sort by timestamp prefix (YYYYMMDD_HHMMSS)
        latest_hash = sorted(experiments)[-1]
        return self.load_experiment(latest_hash)