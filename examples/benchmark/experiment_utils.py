"""Utilities for running experiments."""

import csv
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rl_thor.envs.ai2thor_envs import BaseAI2THOREnv

# TODO: Handle config path better
experiment_config_path = Path(__file__).parent / "config/experiment_config.yaml"


@dataclass
class Exp:
    """
    Class for experiment configuration.

    Attributes:
        model (str): Name of the model to use.
        tasks (Iterable[str]): Tasks to train on.
        scenes (set[str]): Scenes to train on.
        job_type (str): Type of job to run (train or eval).
        id (str): Unique identifier for the experiment.
        experiment_config_path (Path): Path to the experiment configuration file.
        project_name (str): Name of the project.
        group_name (str): Name of the group.
    """

    model: str
    tasks: Iterable[str]
    scenes: set[str]
    job_type: str = "train"
    id: str | None = None
    experiment_config_path: Path = experiment_config_path
    project_name: str | None = None
    group_name: str | None = None

    def __post_init__(self) -> None:
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%Z")
        self.day = datetime.now().strftime("%Y-%m-%d")
        if self.id is None:
            self.id = str(uuid.uuid4())
        self.config = self.load_config()
        self.sorted_scenes = sorted(self.scenes)

        # === Type Annotations ===
        self.timestamp: str
        self.day: str
        self.config: dict[str, Any]

    @property
    def name(self) -> str:
        """Return the name of the experiment."""
        return f"{self.model}_{"-".join(self.tasks)}_{"-".join(self.sorted_scenes)}_{self.timestamp}"

    # TODO: Improve group naming
    @property
    def group(self) -> str:
        """Return the group of the experiment."""
        return f"{"-".join(self.tasks)}"

    @property
    def log_dir(self) -> Path:
        """Return the log directory of the experiment."""
        return Path(f"runs/{self.project_name}/{self.group_name}/{self.name}_({self.id})")

    @property
    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory of the experiment."""
        return Path(f"checkpoints/{self.project_name}/{self.group_name}/{self.name}_({self.id})")

    def load_config(self) -> dict[str, Any]:
        """Load the experiment configuration."""
        with self.experiment_config_path.open("r") as file:
            config = yaml.safe_load(file)
        # Update project and group names
        if self.project_name is not None:
            config["wandb"]["project"] = self.project_name
        else:
            self.project_name = config["project_name"]
        if self.group_name is not None:
            config["wandb"]["group"] = self.group_name
        else:
            self.group_name = config["group_name"]
        # Add other attributes
        config.update({
            "model": self.model,
            "tasks": self.tasks,
            "scenes": self.scenes,
            "job_type": self.job_type,
        })

        return config


# %% SB3 callbacks
from stable_baselines3.common.callbacks import BaseCallback


#!! No more used
class LogSpeedPerformanceCallback(BaseCallback):
    """
    Callback for logging performance (graph tasks computation time, rendering time and scene initialization time).

    Those data are found in the `info` dictionary of the environment after each step (self.model.ep_info_buffer).

    Attributes:
        log_dir (Path): Log directory.
        verbose (int): Verbosity level.

    """

    def __init__(
        self,
        log_dir: Path | str,
        verbose: int = 0,
    ) -> None:
        """
        Initialize the callback.

        Args:
            log_dir (Path): Log directory.
            verbose (int): Verbosity level.

        """
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_file_path = self.log_dir / "performance_log.csv"

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        with self.log_file_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["num_timesteps", "reward_computation_time", "action_execution_time"])

    def _on_step(self) -> bool:
        """
        Log the performance after each step.

        Returns:
            bool: Whether or not the callback should continue.

        """
        # Log the performance
        speed_performance_info = self.locals["infos"][0]["speed_performance"]

        reward_computation_time = speed_performance_info.get("reward_computation_time", None)
        action_execution_time = speed_performance_info.get("action_execution_time", None)

        # Write the performance metrics to CSV
        with self.log_file_path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.num_timesteps, reward_computation_time, action_execution_time])

        return True


#!! Untested
class LogStepsRewardCallback(BaseCallback):
    """
    Callback for logging the reward after each step.

    Attributes:
        log_dir (Path): Log directory.
        verbose (int): Verbosity level.
    """

    def __init__(
        self,
        log_dir: Path | str,
        verbose: int = 0,
    ) -> None:
        """
        Initialize the callback.

        Args:
            log_dir (Path): Log directory.
            verbose (int): Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_file_path = self.log_dir / "steps_reward_log.csv"

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        with self.log_file_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["num_timesteps", "reward"])

    def _on_step(self) -> bool:
        """
        Log the performance after each step.

        Returns:
            bool: Whether or not the callback should continue.

        """
        # Log the performance
        reward = self.locals["reward"]

        # Write the performance metrics to CSV
        with self.log_file_path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.num_timesteps, reward])

        return True


# %% Environment wrappers
import gymnasium as gym


class FullMetricsLogWrapper(gym.Wrapper, BaseAI2THOREnv):
    """
    Wrapper for logging several metrics after each step.

    Attributes:
        log_dir (Path): Log directory.
    """

    def __init__(
        self,
        env: gym.Env,
        log_dir: Path | str,
    ) -> None:
        """
        Initialize the wrapper.

        Args:
            env (gym.Env): Environment to wrap.
            log_dir (Path): Log directory.
        """
        super().__init__(env)
        self.log_dir = Path(log_dir)
        self.log_file_path = self.log_dir / "full_metrics_log.csv"

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        with self.log_file_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "episode_step",
                "reward",
                "max_task_advancement",
                "task_advancement",
                "terminated",
                "truncated",
                "task_type",
                "task_args",
                "task_description",
                "scene_initialization_time",
                "reward_computation_time",
                "action_execution_time",
                "scene",
            ])

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """
        Step the environment.

        Args:
            action (Any): Action to take.

        Returns:
            tuple[Any, float, bool, bool, dict]: Observation, reward, termination status, truncated status, and info.

        """
        # Step the environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._log_full_step_metrics(
            info=info,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
        )

        return observation, float(reward), terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        """
        Reset the environment.

        Args:
            **kwargs (Any): Additional arguments for the reset.

        Returns:
            observation (Any): Initial observation.
            info (dict): Reset information.

        """
        observation, info = self.env.reset(**kwargs)
        self._log_full_step_metrics(
            info=info,
            reward=0,
            terminated=False,
            truncated=False,
        )

        return observation, info

    def _log_full_step_metrics(
        self,
        info: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Log the full step metrics.

        Args:
            info (dict): Information about the step.
            reward (float): Reward obtained.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
        """
        # Log the metrics
        episode_step = info.get("episode_step")
        max_task_advancement = info.get("max_task_advancement")
        task_advancement = info.get("task_advancement")
        task_type = info.get("task_type")
        task_args = info.get("task_args")
        task_description = info.get("task_description")
        scene_initialization_time = info["speed_performance"].get("scene_initialization_time")
        reward_computation_time = info["speed_performance"].get("reward_computation_time")
        action_execution_time = info["speed_performance"].get("action_execution_time")
        scene = info.get("scene")

        # Write the full step metrics to CSV
        with self.log_file_path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                episode_step,
                reward,
                max_task_advancement,
                task_advancement,
                terminated,
                truncated,
                task_type,
                task_args,
                task_description,
                scene_initialization_time,
                reward_computation_time,
                action_execution_time,
                scene,
            ])
