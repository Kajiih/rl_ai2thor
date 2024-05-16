"""Run a stable-baselines3 agent in the AI2THOR RL environment."""
# TODO: Make compatible with multi-task training

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import gymnasium as gym
import typer
import wandb
import yaml
from experiment_utils import Exp, LogSpeedPerformanceCallback
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from rl_thor.envs.tasks.tasks import TaskType
from rl_thor.envs.wrappers import SimpleActionSpaceWrapper, SingleTaskWrapper

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

config_path = Path("examples/benchmark/config/environment_config.yaml")
with config_path.open("r") as file:
    env_config = yaml.safe_load(file)


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
    MULTI_TASK = "MultiTask"


model_config = {
    "verbose": 1,
    "progress_bar": True,
}


def get_model(model_name: ModelType) -> type[PPO] | type[A2C] | type[DQN]:
    """Return the SB3 model class."""
    match model_name:
        case ModelType.PPO:
            return PPO
        case ModelType.A2C:
            return A2C
        case ModelType.DQN:
            return DQN


task_blueprints_configs = {
    TaskType.PREPARE_MEAL: {
        "task_type": TaskType.PREPARE_MEAL,
        "args": {},
        "scenes": ["FloorPlan1"],
    },
    TaskType.PREPARE_WATCHING_TV: {
        "task_type": TaskType.PREPARE_WATCHING_TV,
        "args": {},
        "scenes": ["FloorPlan201"],
    },
    TaskType.PREPARE_GOING_TO_BED: {
        "task_type": TaskType.PREPARE_GOING_TO_BED,
        "args": {},
        "scenes": ["FloorPlan301"],
    },
    TaskType.PREPARE_FOR_SHOWER: {
        "task_type": TaskType.PREPARE_FOR_SHOWER,
        "args": {},
        "scenes": ["FloorPlan401"],
    },
}


def get_task_blueprint_config(task: AvailableTask) -> list[dict[str, Any]]:
    """Return the scenes for the task."""
    match task:
        case AvailableTask.PREPARE_MEAL:
            return [task_blueprints_configs[TaskType.PREPARE_MEAL]]
        case AvailableTask.PREPARE_WATCHING_TV:
            return [task_blueprints_configs[TaskType.PREPARE_WATCHING_TV]]
        case AvailableTask.PREPARE_GOING_TO_BED:
            return [task_blueprints_configs[TaskType.PREPARE_GOING_TO_BED]]
        case AvailableTask.PREPARE_FOR_SHOWER:
            return [task_blueprints_configs[TaskType.PREPARE_FOR_SHOWER]]
        case AvailableTask.MULTI_TASK:
            return [task_blueprints_configs[task] for task in task_blueprints_configs]


def make_env(
    config_path: str | Path, config_override: dict[str, Any], experiment: Exp, is_single_task: bool
) -> gym.Env:
    """Create the environment for single task and simple action space training with stable-baselines3."""
    env = gym.make(
        "rl_thor/ITHOREnv-v0.1_sb3_ready",
        config_path=config_path,
        config_override=config_override,
    )  # type: ignore
    env = SimpleActionSpaceWrapper(env)
    if is_single_task:
        env = SingleTaskWrapper(env)
    env = Monitor(
        env,
        filename=str(experiment.log_dir / "monitor.csv"),
        info_keywords=("task_advancement", "is_success"),
    )
    return env


def main(
    task: AvailableTask,
    model_name: Annotated[ModelType, typer.Option("--model", case_sensitive=False)] = ModelType.PPO,
    total_timesteps: Annotated[int, typer.Option("--timesteps", "-s")] = 1_000_000,
    record: bool = False,
    log_speed_performance: Annotated[bool, typer.Option("--log-speed", "-l")] = False,
    no_task_advancement_reward: Annotated[bool, typer.Option("--no-adv", "-n")] = False,
    seed: int = 0,
) -> None:
    """
    Train the agent.

    Args:
        task (AvailableTask): Task to train the agent on.
        model_name (ModelType): Model to use for training.
        total_timesteps (int): Total number of timesteps to train the agent.
        record (bool): Record the training.
        log_speed_performance (bool): Log the speed performance of the agent.
        seed (int): Seed for reproducibility.
    """
    is_single_task = task != AvailableTask.MULTI_TASK
    if is_single_task:
        model_config["policy_type"] = "CnnPolicy"
    else:
        model_config["policy_type"] = "MultiInputPolicy"

    task_blueprint_config = get_task_blueprint_config(task)
    scenes = {scenes for task_config in task_blueprint_config for scenes in task_config["scenes"]}

    # === Load the environment and experiment configurations ===
    experiment = Exp(model=model_name, tasks=[task], scenes=scenes)
    config_override: dict[str, Any] = {"tasks": {"task_blueprints": task_blueprint_config}}
    config_override["no_task_advancement_reward"] = no_task_advancement_reward
    wandb_config = experiment.config["wandb"]
    tags = ["simple_actions", "single_task", model_name, *scenes, task, experiment.job_type]
    tags.append("single_task" if is_single_task else "multi_task")
    run: Run = wandb.init(  # type: ignore
        config=experiment.config | env_config | {"tasks": {"task_blueprints": task_blueprint_config}},
        project=wandb_config["project"],
        sync_tensorboard=wandb_config["sync_tensorboard"],
        monitor_gym=wandb_config["monitor_gym"],
        save_code=wandb_config["save_code"],
        name=experiment.name,
        # group=experiment.group,
        job_type=experiment.job_type,
        tags=tags,
        notes=f"Simple {model_name} agent for RL THOR benchmarking on {task} task.",
    )

    # === Instantiate the environment ===
    env = DummyVecEnv([lambda: make_env(config_path, config_override, experiment, is_single_task=is_single_task)])
    if record:
        record_config = experiment.config["video_recorder"]
        env = VecVideoRecorder(
            venv=env,
            video_folder=str(experiment.log_dir / "videos"),
            record_video_trigger=lambda x: x % record_config["frequency"] == 0,
            video_length=record_config["length"],
            name_prefix=record_config["prefix"],
        )

    # === Instantiate the model ===
    sb3_model = get_model(model_name)
    train_model = sb3_model(
        policy=model_config["policy_type"],
        env=env,
        verbose=model_config["verbose"],
        tensorboard_log=str(experiment.log_dir),
        seed=seed,
    )
    wandb_callback_config = wandb_config["sb3_callback"]
    eval_callback_config = experiment.config["evaluation"]
    # TODO? Add a callback for saving the model instead of using the parameter in WandbCallback?
    eval_env = DummyVecEnv([lambda: make_env(config_path, config_override, experiment, is_single_task=is_single_task)])
    callbacks = [
        # TODO: Check EvalCallback really works with different tasks
        EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=eval_callback_config["nb_episodes"],
            eval_freq=eval_callback_config["frequency"],
            log_path=str(experiment.log_dir),
            best_model_save_path=str(experiment.checkpoint_dir),
            deterministic=eval_callback_config["deterministic"],
            verbose=eval_callback_config["verbose"],
        ),
        WandbCallback(
            verbose=wandb_callback_config["verbose"],
            model_save_path=str(experiment.checkpoint_dir),
            model_save_freq=wandb_callback_config["gradient_save_freq"],
            gradient_save_freq=wandb_callback_config["gradient_save_freq"],
        ),
    ]
    if log_speed_performance:
        callbacks.append(LogSpeedPerformanceCallback(experiment.log_dir, verbose=1))

    train_model.learn(
        total_timesteps=total_timesteps,
        progress_bar=model_config["progress_bar"],
        callback=CallbackList(callbacks),
    )
    env.close()
    run.finish()


if __name__ == "__main__":
    typer.run(main)
