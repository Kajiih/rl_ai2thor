"""Utilities for running experiments."""

import csv
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

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
    seed: int
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
        self.sorted_scenes = sorted(self.scenes)[:5]

        # === Type Annotations ===
        self.timestamp: str
        self.day: str
        self.config: dict[str, Any]

    @property
    def name(self) -> str:
        """Return the name of the experiment."""
        # return f"{self.model}_{"-".join(self.tasks)}_{"-".join(self.sorted_scenes)}_{self.seed}_{self.timestamp}"
        return f"{self.model}_{"-".join(self.tasks)}_{len(self.sorted_scenes)}-scenes_{self.seed}_{self.timestamp}"

    # TODO: Improve group naming
    @property
    def group(self) -> str:
        """Return the group of the experiment."""
        # Limit to 50 characters
        # return f"{"-".join(self.tasks)}"
        return f"{self.model}_{"-".join(self.tasks)}"

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
            self.project_name = config["wandb"]["project"]
        if self.group_name is not None:
            config["wandb"]["group"] = self.group_name
        else:
            self.group_name = config["wandb"]["group"]
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


class EvalOnEachTaskAndSceneCallback(BaseCallback):
    """
    Evaluate the model on each task and scene after each evaluation interval.

    To be used with EvalCallback as callback_after_eval.
    """

    def __init__(
        self,
        eval_env: DummyVecEnv,
        log_dir: str | Path,
        verbose=0,
    ) -> None:
        """
        Initialize the callback.

        Args:
            eval_env (DummyVecEnv): Evaluation environment.
            log_dir (str | Path): Log directory.
            verbose (int): Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = Path(log_dir) / "eval_data"

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Perform evaluation after each training step
        self._evaluate_all_scenes()
        return True

    def _evaluate_all_scenes(self) -> None:
        nb_files = len(list(self.log_dir.glob("*.csv")))
        log_file = self.log_dir / f"eval_data_{nb_files}.csv"
        eval_env = self.eval_env.envs[0]

        # Initialize CSV file with headers
        with log_file.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "scene",
                "task_idx",
                "task_type",
                "episode_max_return",
                "episode_max_task_advancement",
                "task_maximum_advancement",
                "task_completed",
            ])

            print("Evaluating model on each task and scene...")
            for task_idx, task_blueprint in enumerate(self.eval_env.get_attr("task_blueprints")[0]):
                task_type = task_blueprint.task_type
                for scene in task_blueprint.scenes:
                    # obs, info = self.eval_env.reset(forced_scene=scene, forced_task_idx=task_idx)
                    reset_options = {"forced_scene": scene, "forced_task_idx": task_idx}
                    obs, info = self.eval_env.env_method("reset", options=reset_options)[0]
                    terminated = info["is_success"]
                    truncated = False
                    episode_reward = 0
                    episode_max_reward = 0
                    max_task_advancement = 0
                    task_maximum_advancement = info.get("task_maximum_advancement", None)

                    while not terminated and not truncated:
                        obs_copy = obs.copy()
                        action, _states = self.model.predict(obs_copy, deterministic=True)
                        # Make action parameters scalars
                        action = {k: v.item() for k, v in action.items()} if isinstance(action, dict) else action.item()

                        # obs, reward, terminated, truncated, info = self.eval_env.step(action)
                        obs, reward, terminated, truncated, info = self.eval_env.env_method("step", action)[0]
                        episode_reward += reward
                        episode_max_reward = max(episode_max_reward, reward)
                        task_advancement = info.get("task_advancement", 0)
                        if task_advancement is not None:
                            max_task_advancement = max(max_task_advancement, task_advancement)

                    result = {
                        "scene": scene,
                        "task_idx": task_idx,
                        "task_type": task_type.__name__,
                        "episode_max_return": episode_max_reward,
                        "episode_max_task_advancement": max_task_advancement,
                        "task_maximum_advancement": task_maximum_advancement,
                        "task_completed": terminated,
                    }
                    self._log_result(writer, result)

            # Reset the environment at the end of the evaluation
            self.eval_env.reset()

    def _log_result(self, writer, result: dict[str, Any]) -> None:  # noqa: PLR6301, ANN001
        writer.writerow([
            result["scene"],
            result["task_idx"],
            result["task_type"],
            result["episode_max_return"],
            result["episode_max_task_advancement"],
            result["task_completed"],
        ])


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
