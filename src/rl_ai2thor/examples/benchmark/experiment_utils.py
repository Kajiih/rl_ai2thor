"""Utilities for running experiments."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# TODO: Handle config path better
experiment_config_path = Path(__file__).parent / "experiment_config.yaml"


@dataclass
class Exp:
    """
    Class for experiment configuration.

    Attributes:
        dataset (str): Dataset name.
        model (str): Model name.
        skills (int): Number of skills.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        temperature (float): Temperature.
        timestamp (str): Timestamp of the start of the experiment.
        id (str): Experiment ID.

    """

    model: str
    tasks: list[str]
    scenes: list[str]
    job_type: str = "train"
    id: str | None = None

    def __post_init__(self) -> None:
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%Z")
        self.day = datetime.now().strftime("%Y-%m-%d")
        if self.id is None:
            self.id = str(uuid.uuid4())
        self.config = self.load_config()

        # === Type Annotations ===
        self.timestamp: str
        self.day: str
        self.config: dict[str, Any]

    @property
    def name(self) -> str:
        """Return the name of the experiment."""
        return f"{self.model}_{self.tasks}_{self.scenes}_{self.timestamp}"

    # TODO: Improve group naming
    @property
    def group(self) -> str:
        """Return the group of the experiment."""
        return f"{self.tasks}_{self.scenes}"

    @property
    def log_dir(self) -> Path:
        """Return the log directory of the experiment."""
        return Path(f"runs/{self.day}/{self.name}_({self.id})")

    @property
    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory of the experiment."""
        return Path(f"checkpoints/{self.day}/{self.name}_({self.id})")

    def load_config(self) -> dict[str, Any]:
        """Load the experiment configuration."""
        with experiment_config_path.open("r") as file:
            config = yaml.safe_load(file)
        config.update({
            "model": self.model,
            "tasks": self.tasks,
            "scenes": self.scenes,
            "job_type": self.job_type,
        })
        return config
