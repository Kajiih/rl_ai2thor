"""Run a stable-baselines3 agent in the AI2THOR RL environment."""
# TODO: Make compatible with multi-task training

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import gymnasium as gym
import typer
import wandb
import yaml
from experiment_utils import Exp, FullMetricsLogWrapper
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from rl_thor.envs.sim_objects import SimObjectType
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
    RANDOM = "Random"


class AvailableTask(StrEnum):
    """Available tasks for training."""

    # Complex tasks
    PREPARE_MEAL = TaskType.PREPARE_MEAL
    PREPARE_WATCHING_TV = TaskType.PREPARE_WATCHING_TV
    PREPARE_GOING_TO_BED = TaskType.PREPARE_GOING_TO_BED
    PREPARE_FOR_SHOWER = TaskType.PREPARE_FOR_SHOWER
    MULTI_TASK = "MultiTask"

    # Gradual tasks
    PICKUP_KNIFE = "PickupKnife"
    PICKUP_MUG = "PickupMug"
    PLACE_KNIFE_IN_SINK = "PlaceKnifeInSink"
    PLACE_MUG_IN_SINK = "PlaceMugInSink"
    PLACE_KNIFE_IN_FILLED_SINK = "PlaceKnifeInFilledSink"
    PLACE_MUG_IN_FILLED_SINK = "PlaceMugInFilledSink"
    PLACE_KNIFE_BOWL_MUG_IN_FILLED_SINK = "PlaceKnifeBowlMugInFilledSink"


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
        case ModelType.RANDOM:
            return DQN


task_blueprints_configs = {
    AvailableTask.PREPARE_MEAL: {
        "task_type": TaskType.PREPARE_MEAL,
        "args": {},
        "scenes": ["FloorPlan1", "FloorPlan2"],
    },
    AvailableTask.PREPARE_WATCHING_TV: {
        "task_type": TaskType.PREPARE_WATCHING_TV,
        "args": {},
        "scenes": ["FloorPlan201", "FloorPlan203"],
    },
    AvailableTask.PREPARE_GOING_TO_BED: {
        "task_type": TaskType.PREPARE_GOING_TO_BED,
        "args": {},
        "scenes": ["FloorPlan301", "FloorPlan302"],
    },
    AvailableTask.PREPARE_FOR_SHOWER: {
        "task_type": TaskType.PREPARE_FOR_SHOWER,
        "args": {},
        "scenes": ["FloorPlan401", "FloorPlan402"],
    },
    AvailableTask.PICKUP_KNIFE: {
        "task_type": TaskType.PICKUP,
        "args": {"picked_up_object_type": SimObjectType.BUTTER_KNIFE},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PICKUP_MUG: {
        "task_type": TaskType.PICKUP,
        "args": {"picked_up_object_type": SimObjectType.MUG},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_KNIFE_IN_SINK: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.BUTTER_KNIFE, "receptacle_type": SimObjectType.SINK_BASIN},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_MUG_IN_SINK: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.MUG, "receptacle_type": SimObjectType.SINK_BASIN},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_MUG_IN_FILLED_SINK: {
        "task_type": TaskType.PLACE_IN_FILLED_SINK,
        "args": {"placed_object_type": SimObjectType.MUG},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_KNIFE_BOWL_MUG_IN_FILLED_SINK: {
        "task_type": TaskType.PLACE_IN_FILLED_SINK,
        "args": {
            "placed_object_type_1": SimObjectType.BUTTER_KNIFE,
            "placed_object_type_2": SimObjectType.BOWL,
            "placed_object_type_3": SimObjectType.MUG,
        },
        "scenes": ["FloorPlan1"],
    },
}


def get_task_blueprint_config(task: AvailableTask) -> list[dict[str, Any]]:
    """Return the scenes for the task."""
    match task:
        case AvailableTask.MULTI_TASK:
            return [
                task_blueprints_configs[task]
                for task in (
                    AvailableTask.PREPARE_MEAL,
                    AvailableTask.PREPARE_WATCHING_TV,
                    AvailableTask.PREPARE_GOING_TO_BED,
                    AvailableTask.PREPARE_FOR_SHOWER,
                )
            ]
        case _:
            return [task_blueprints_configs[task]]


def get_action_groups_override_config(task: AvailableTask) -> dict[str, Any]:
    """Return the action groups for the task."""
    match task:
        case (
            AvailableTask.PICKUP_KNIFE
            | AvailableTask.PICKUP_MUG
            | AvailableTask.PLACE_KNIFE_IN_SINK
            | AvailableTask.PLACE_MUG_IN_SINK
            | AvailableTask.PLACE_KNIFE_IN_FILLED_SINK
            | AvailableTask.PLACE_MUG_IN_FILLED_SINK
            | AvailableTask.PLACE_KNIFE_BOWL_MUG_IN_FILLED_SINK
        ):
            action_groups = {
                "open_close_actions": False,
                "toggle_actions": False,
                "slice_actions": False,
            }
        case _:
            action_groups = {}
    return {"action_groups": action_groups}


def make_env(
    config_path: str | Path,
    config_override: dict[str, Any],
    experiment: Exp,
    is_single_task: bool,
    log_full_metrics: bool,
    eval_env: bool = False,
) -> gym.Env:
    """Create the environment for single task and simple action space training with stable-baselines3."""
    env = gym.make(
        "rl_thor/ITHOREnv-v0.1_sb3_ready",
        config_path=config_path,
        config_override=config_override,
    )  # type: ignore

    log_dir = experiment.log_dir / "eval" if eval_env else experiment.log_dir / "train"
    if log_full_metrics:
        env = FullMetricsLogWrapper(env, log_dir)

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
    rollout_length: Annotated[Optional[int], typer.Option("--rollout", "-r")] = None,  # noqa: UP007
    total_timesteps: Annotated[int, typer.Option("--timesteps", "-s")] = 1_000_000,
    record: bool = False,
    log_full_env_metrics: Annotated[bool, typer.Option("--log-metrics", "-l")] = False,
    no_task_advancement_reward: Annotated[bool, typer.Option("--no-adv", "-n")] = False,
    seed: int = 0,
    group_name: Annotated[Optional[str], typer.Option("--group", "-g")] = None,  # noqa: UP007
) -> None:
    """
    Train the agent.

    Args:
        task (AvailableTask): Task to train the agent on.
        model_name (ModelType): Model to use for training.
        total_timesteps (int): Total number of timesteps to train the agent.
        record (bool): Record the training.
        log_full_env_metrics (bool): Log full environment metrics.
        no_task_advancement_reward (bool): Do not use the task advancement reward.
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
    if rollout_length is not None:
        config_override["max_episode_steps"] = rollout_length
    # Add action groups override config
    config_override.update(get_action_groups_override_config(task))
    wandb_config = experiment.config["wandb"]
    tags = ["simple_actions", "single_task", model_name, *scenes, task, experiment.job_type]
    tags.extend((
        "single_task" if is_single_task else "multi_task",
        group_name if group_name is not None else "no_group",
        "no_task_advancement_reward" if no_task_advancement_reward else "with_task_advancement_reward",
    ))
    run: Run = wandb.init(  # type: ignore
        config=experiment.config | env_config | {"tasks": {"task_blueprints": task_blueprint_config}},
        project=wandb_config["project"],
        sync_tensorboard=wandb_config["sync_tensorboard"],
        monitor_gym=wandb_config["monitor_gym"],
        save_code=wandb_config["save_code"],
        name=experiment.name,
        group=group_name,
        job_type=experiment.job_type,
        tags=tags,
        notes=f"Simple {model_name} agent for RL THOR benchmarking on {task} task.",
    )

    # === Instantiate the environment ===
    env = DummyVecEnv([
        lambda: make_env(
            config_path,
            config_override,
            experiment,
            is_single_task=is_single_task,
            log_full_metrics=log_full_env_metrics,
            eval_env=False,
        )
    ])
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
    model_args = {
        "policy": model_config["policy_type"],
        "env": env,
        "verbose": model_config["verbose"],
        "tensorboard_log": str(experiment.log_dir),
        "seed": seed,
    }
    if model_name == ModelType.RANDOM:
        model_args["learning_starts"] = total_timesteps
        model_args["learning_rate"] = 0.0
    train_model = sb3_model(**model_args)

    wandb_callback_config = wandb_config["sb3_callback"]
    eval_callback_config = experiment.config["evaluation"]
    # TODO? Add a callback for saving the model instead of using the parameter in WandbCallback?
    eval_env = DummyVecEnv([
        lambda: make_env(
            config_path,
            config_override,
            experiment,
            is_single_task=is_single_task,
            log_full_metrics=log_full_env_metrics,
            eval_env=True,
        )
    ])
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

    train_model.learn(
        total_timesteps=total_timesteps,
        progress_bar=model_config["progress_bar"],
        callback=CallbackList(callbacks),
    )
    env.close()
    run.finish()


if __name__ == "__main__":
    typer.run(main)
