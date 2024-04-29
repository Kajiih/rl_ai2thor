"""
Run a stable-baselines3 agent in the AI2THOR RL environment on several tasks.

# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any

import gymnasium as gym
import typer
import wandb
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
from rl_ai2thor.envs.tasks.tasks import TaskType
from rl_ai2thor.envs.wrappers import SimpleActionSpaceWrapper
from rl_ai2thor.examples.benchmark.experiment_utils import Exp

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


model_config = {
    "policy_type": "MultiInputPolicy",
    "verbose": 1,
    "progress_bar": True,
}

action_categories = {
    # === Navigation actions ===
    "crouch_actions": False,
    # === Object manipulation actions ===
    "drop_actions": False,
    "throw_actions": False,
    "push_pull_actions": False,
    # === Object interaction actions ===
    "open_close_actions": True,
    "toggle_actions": True,
    "slice_actions": True,
    "use_up_actions": False,
    "liquid_manipulation_actions": False,
}
override_config = {
    "discrete_actions": True,
    "target_closest_object": True,
    "action_categories": action_categories,
    "simple_movement_actions": True,
    "max_episode_steps": 1000,
    "random_agent_spawn": True,
    # "tasks" is added dynamically
}


class ModelType(StrEnum):
    """SB3 compatible models."""

    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"


class AvailableTask(StrEnum):
    """Available tasks for training."""

    PREPARE_MEAL = TaskType.PREPARE_MEAL
    PREPARE_WATCHING_TV = TaskType.PREPARE_WATCHING_TV
    PREPARE_GOING_TO_BED = TaskType.PREPARE_GOING_TO_BED
    PREPARE_FOR_SHOWER = TaskType.PREPARE_FOR_SHOWER


def get_model(model_name: ModelType) -> type[PPO] | type[A2C] | type[DQN]:
    """Return the SB3 model class."""
    match model_name:
        case ModelType.PPO:
            return PPO
        case ModelType.A2C:
            return A2C
        case ModelType.DQN:
            return DQN


def get_scenes(task: AvailableTask) -> list[str]:
    """Return the scenes for the task."""
    match task:
        case AvailableTask.PREPARE_MEAL:
            return ["FloorPlan1"]
        case AvailableTask.PREPARE_WATCHING_TV:
            return ["FloorPlan201"]
        case AvailableTask.PREPARE_GOING_TO_BED:
            return ["FloorPlan301"]
        case AvailableTask.PREPARE_FOR_SHOWER:
            return ["FloorPlan401"]


def make_env(override_config: dict[str, Any]) -> ITHOREnv:
    """Create the environment for single task and simple action space training with stable-baselines3."""
    env: ITHOREnv = gym.make("rl_ai2thor/ITHOREnv-v0.1_sb3_ready", override_config=override_config)  # type: ignore
    env = SimpleActionSpaceWrapper(env)
    return env


def main(
    model_name: Annotated[ModelType, typer.Option("--model", case_sensitive=False)] = ModelType.PPO,
    total_timesteps: Annotated[int, typer.Option("--timesteps", "-s")] = 1_000_000,
    record: bool = False,
) -> None:
    """
    Train the agent.

    TODO: Improve docstring.
    """
    tasks = list(AvailableTask)
    task_config = [
        {
            "type": task,
            "args": {},
            "scenes": get_scenes(task),
        }
        for task in tasks
    ]
    override_config["tasks"] = task_config
    # === Load the experiment configuration ===
    scenes = [scene for task_dict in task_config for scene in task_dict["scenes"]]
    experiment = Exp(model=model_name, tasks=tasks, scenes=scenes)
    wandb_config = experiment.config["wandb"]
    run: Run = wandb.init(  # type: ignore
        config=experiment.config,
        project=wandb_config["project"],
        sync_tensorboard=wandb_config["sync_tensorboard"],
        monitor_gym=wandb_config["monitor_gym"],
        save_code=wandb_config["save_code"],
        name=experiment.name,
        group=experiment.group,
        job_type=experiment.job_type,
        tags=["simple_actions", "multi-tasks", experiment.job_type, model_name, *scenes, *tasks],
        notes=f"Simple {model_name} agent for RL THOR benchmarking on {tasks} multi-task.",
    )

    # === Instantiate the environment ===
    env = DummyVecEnv([lambda: make_env(override_config)])
    if record:
        record_config = experiment.config["record"]
        env = VecVideoRecorder(
            venv=env,
            video_folder=str(experiment.log_dir / "videos"),  # TODO: Check if we need the str() conversion
            record_video_trigger=lambda x: x % record_config["frequency"] == 0,
            video_length=record_config["length"],
            name_prefix=record_config["prefix"],
        )

    # === Instantiate the model ===
    wandb_callback_config = wandb_config["sb3_callback"]
    sb3_model = get_model(model_name)
    train_model = sb3_model(
        policy=model_config["policy_type"],
        env=env,
        verbose=model_config["verbose"],
        tensorboard_log=str(experiment.log_dir),  # TODO: Check if we need the str() conversion
    )
    train_model.learn(
        total_timesteps=total_timesteps,
        progress_bar=model_config["progress_bar"],
        callback=WandbCallback(
            verbose=wandb_callback_config["verbose"],
            model_save_path=f"models/{run.id}",
            model_save_freq=wandb_callback_config["gradient_save_freq"],
            gradient_save_freq=wandb_callback_config["gradient_save_freq"],
        ),
    )
    env.close()
    run.finish()


if __name__ == "__main__":
    typer.run(main)
