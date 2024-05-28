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
from sb3_contrib import QRDQN
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from rl_thor.agents.agents import RandomAgent
from rl_thor.envs.sim_objects import SimObjectType
from rl_thor.envs.tasks.tasks import TaskType
from rl_thor.envs.wrappers import SimpleActionSpaceWrapper, SingleTaskWrapper

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from rl_thor.envs.ai2thor_envs import ITHOREnv

config_path = Path("examples/benchmark/config/environment_config.yaml")
with config_path.open("r") as file:
    env_config = yaml.safe_load(file)


class ModelType(StrEnum):
    """SB3 compatible models."""

    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"
    QRDQN = "QRDQN"
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
    PICKUP_POTATO = "PickupPotato"
    PLACE_POTATO_IN_MICROWAVE = "PlacePotatoInMicrowave"
    COOK_POTATO = "CookPotato"
    SLICE_AND_COOK_POTATO = "SliceAndCookPotato"


model_config = {
    "verbose": 1,
    "progress_bar": True,
}


def get_model(model_name: ModelType) -> type[PPO] | type[A2C] | type[DQN] | type[QRDQN]:
    """Return the SB3 model class."""
    match model_name:
        case ModelType.PPO:
            return PPO
        case ModelType.A2C:
            return A2C
        case ModelType.DQN:
            return DQN
        case ModelType.QRDQN:
            return QRDQN
        case ModelType.RANDOM:
            raise ValueError("Random agent doesn't need a model.")


task_blueprints_configs = {
    AvailableTask.PREPARE_MEAL: {
        "task_type": TaskType.PREPARE_MEAL,
        "args": {},
        "scenes": [
            "FloorPlan1",
            "FloorPlan2",
            "FloorPlan3",
            "FloorPlan4",
            "FloorPlan5",
            "FloorPlan6",
            "FloorPlan7",
            "FloorPlan8",
            "FloorPlan9",
            "FloorPlan10",
            "FloorPlan11",
            "FloorPlan12",
            "FloorPlan13",
            "FloorPlan14",
            "FloorPlan15",
            "FloorPlan16",
            "FloorPlan17",
            "FloorPlan18",
            "FloorPlan19",
            "FloorPlan20",
            "FloorPlan21",
            "FloorPlan22",
            "FloorPlan23",
            "FloorPlan24",
            "FloorPlan25",
            "FloorPlan26",
            "FloorPlan27",
            "FloorPlan28",
            "FloorPlan29",
            "FloorPlan30",
        ],
    },
    AvailableTask.PREPARE_WATCHING_TV: {
        "task_type": TaskType.PREPARE_WATCHING_TV,
        "args": {},
        "scenes": [
            "FloorPlan201",
            "FloorPlan203",
            "FloorPlan209",
            "FloorPlan210",
            "FloorPlan211",
            "FloorPlan212",
            "FloorPlan214",
            "FloorPlan215",
            "FloorPlan216",
            "FloorPlan218",
            "FloorPlan219",
            "FloorPlan222",
            "FloorPlan224",
            "FloorPlan225",
            "FloorPlan226",
            "FloorPlan227",
            "FloorPlan228",
            "FloorPlan230",
        ],
    },
    AvailableTask.PREPARE_GOING_TO_BED: {
        "task_type": TaskType.PREPARE_GOING_TO_BED,
        "args": {},
        "scenes": [
            # "FloorPlan201",
            # "FloorPlan224",
            "FloorPlan301",
            "FloorPlan302",
            "FloorPlan303",
            "FloorPlan304",
            "FloorPlan305",
            "FloorPlan306",
            "FloorPlan307",
            "FloorPlan308",
            "FloorPlan309",
            "FloorPlan310",
            "FloorPlan311",
            "FloorPlan312",
            "FloorPlan313",
            "FloorPlan314",
            "FloorPlan315",
            "FloorPlan316",
            "FloorPlan317",
            "FloorPlan318",
            "FloorPlan319",
            "FloorPlan320",
            "FloorPlan321",
            "FloorPlan322",
            "FloorPlan323",
            "FloorPlan324",
            "FloorPlan325",
            "FloorPlan326",
            "FloorPlan327",
            "FloorPlan328",
            "FloorPlan329",
            "FloorPlan330",
        ],
    },
    AvailableTask.PREPARE_FOR_SHOWER: {
        "task_type": TaskType.PREPARE_FOR_SHOWER,
        "args": {},
        "scenes": [
            "FloorPlan401",
            "FloorPlan402",
            "FloorPlan403",
            "FloorPlan404",
            "FloorPlan407",
            "FloorPlan413",
            "FloorPlan415",
            "FloorPlan419",
            "FloorPlan422",
            "FloorPlan423",
            "FloorPlan426",
            "FloorPlan427",
        ],
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
    AvailableTask.PICKUP_POTATO: {
        "task_type": TaskType.PICKUP,
        "args": {"picked_up_object_type": SimObjectType.POTATO},
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
    AvailableTask.PLACE_POTATO_IN_MICROWAVE: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.POTATO, "receptacle_type": SimObjectType.MICROWAVE},
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
    AvailableTask.COOK_POTATO: {
        "task_type": TaskType.COOK,
        "args": {"cooked_object_type": SimObjectType.POTATO},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.SLICE_AND_COOK_POTATO: {
        "task_type": TaskType.SLICE_AND_COOK_POTATO,
        "args": {},
        "scenes": ["FloorPlan1"],
    },
}


def keep_only_n_scenes(task_blueprint_config: dict[str, Any], nb_scenes: int) -> dict[str, Any]:
    """Return a copy of the task blueprint config with only the first n scenes."""
    task_blueprint_config = task_blueprint_config.copy()
    task_blueprint_config["scenes"] = task_blueprint_config["scenes"][:nb_scenes]
    return task_blueprint_config


def get_task_blueprint_config(task: AvailableTask, nb_scenes: int) -> list[dict[str, Any]]:
    """Return the scenes for the task."""
    match task:
        case AvailableTask.MULTI_TASK:
            return [
                keep_only_n_scenes(task_blueprints_configs[task], nb_scenes)
                for task in (
                    AvailableTask.PREPARE_MEAL,
                    AvailableTask.PREPARE_WATCHING_TV,
                    AvailableTask.PREPARE_GOING_TO_BED,
                    AvailableTask.PREPARE_FOR_SHOWER,
                )
            ]
        case _:
            return [keep_only_n_scenes(task_blueprints_configs[task], nb_scenes)]


def get_action_groups_override_config(task: AvailableTask) -> dict[str, Any]:
    """Return the action groups for the task."""
    action_groups = {
        "open_close_actions": False,
        "toggle_actions": False,
        "slice_actions": False,
    }
    # === Enable opening and closing ===
    if task in {
        AvailableTask.PLACE_POTATO_IN_MICROWAVE,
        AvailableTask.COOK_POTATO,
        AvailableTask.SLICE_AND_COOK_POTATO,
        AvailableTask.PREPARE_MEAL,
        AvailableTask.PREPARE_WATCHING_TV,
        AvailableTask.PREPARE_GOING_TO_BED,
        AvailableTask.PREPARE_FOR_SHOWER,
    }:
        action_groups["open_close_actions"] = True

    # === Enable toggling ===
    if task in {
        AvailableTask.PLACE_KNIFE_IN_FILLED_SINK,
        AvailableTask.PLACE_MUG_IN_FILLED_SINK,
        AvailableTask.PLACE_KNIFE_BOWL_MUG_IN_FILLED_SINK,
        AvailableTask.COOK_POTATO,
        AvailableTask.SLICE_AND_COOK_POTATO,
        AvailableTask.PREPARE_MEAL,
        AvailableTask.PREPARE_WATCHING_TV,
        AvailableTask.PREPARE_GOING_TO_BED,
        AvailableTask.PREPARE_FOR_SHOWER,
    }:
        action_groups["toggle_actions"] = True

    # === Enable slicing ===
    if task in {
        AvailableTask.SLICE_AND_COOK_POTATO,
        AvailableTask.PREPARE_MEAL,
        AvailableTask.PREPARE_WATCHING_TV,
        AvailableTask.PREPARE_GOING_TO_BED,
        AvailableTask.PREPARE_FOR_SHOWER,
    }:
        action_groups["slice_actions"] = True

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
    nb_scenes: int = 1,
    model_name: Annotated[ModelType, typer.Option("--model", case_sensitive=False)] = ModelType.PPO,
    rollout_length: Annotated[Optional[int], typer.Option("--rollout", "-r")] = None,  # noqa: UP007
    total_timesteps: Annotated[int, typer.Option("--timesteps", "-s")] = 1_000_000,
    record: bool = False,
    log_full_env_metrics: Annotated[bool, typer.Option("--log-metrics", "-l")] = False,
    no_task_advancement_reward: Annotated[bool, typer.Option("--no-adv", "-n")] = False,
    seed: int = 0,
    project_name: Annotated[Optional[str], typer.Option("--project", "-p")] = None,  # noqa: UP007
    group_name: Annotated[Optional[str], typer.Option("--group", "-g")] = None,  # noqa: UP007
    do_eval: Annotated[bool, typer.Option("--eval", "-e")] = False,
    randomize_agent_position: Annotated[bool, typer.Option("--randomize-agent")] = False,
) -> None:
    """
    Train the agent.

    Args:
        task (AvailableTask): Task to train the agent on.
        nb_scenes (int): Number of scenes per task to use for training.
        model_name (ModelType): Model to use for training.
        rollout_length (Optional[int]): Maximum number of steps per episode.
        total_timesteps (int): Total number of timesteps to train the agent.
        record (bool): Record the training.
        log_full_env_metrics (bool): Log full environment metrics.
        no_task_advancement_reward (bool): Do not use the task advancement reward.
        seed (int): Seed for reproducibility.
        project_name (Optional[str]): Project name for the run in WandB.
        group_name (Optional[str]): Group name for the run in WandB.
        do_eval (bool): Evaluate the agent. !! Don't eval with a different environment in a Docker container, both rendering windows might be mixed up.
        randomize_agent_position (bool): Randomize the agent position in the environment.
    """
    is_single_task = task != AvailableTask.MULTI_TASK
    if is_single_task:
        model_config["policy_type"] = "CnnPolicy"
    else:
        model_config["policy_type"] = "MultiInputPolicy"

    task_blueprint_config = get_task_blueprint_config(task, nb_scenes)
    scenes = {scenes for task_config in task_blueprint_config for scenes in task_config["scenes"]}

    # === Load the environment and experiment configurations ===
    experiment = Exp(model=model_name, tasks=[task], scenes=scenes)
    config_override: dict[str, Any] = {"tasks": {"task_blueprints": task_blueprint_config}}
    config_override["no_task_advancement_reward"] = no_task_advancement_reward
    if rollout_length is not None:
        config_override["max_episode_steps"] = rollout_length
    if randomize_agent_position:
        config_override["scene_randomization"] = {"random_agent_spawn": True}
    # Add action groups override config
    config_override.update(get_action_groups_override_config(task))
    wandb_config = experiment.config["wandb"]
    if project_name is not None:
        wandb_config["project"] = project_name
    tags = ["simple_actions", "single_task", model_name, *scenes, task, experiment.job_type, wandb_config["project"]]
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
    # Save infos about the run
    experiment.log_dir.mkdir(parents=True, exist_ok=True)
    run_info_path = experiment.log_dir / "run_info.yaml"
    run_info = {"tags": tags, "env_config": env_config, "experiment_config": experiment.config}
    with run_info_path.open("w") as f:
        yaml.dump(run_info, f)

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

    # === Run a random agent if the model is random ===
    if model_name == ModelType.RANDOM:
        single_env: ITHOREnv = env.envs[0]
        single_env.reset(seed=seed)
        random_agent = RandomAgent(single_env, seed=seed)
        random_agent.run_episode(
            nb_episodes=total_timesteps // single_env.config.max_episode_steps, total_max_steps=total_timesteps
        )
    else:
        # === Instantiate the model ===
        sb3_model = get_model(model_name)
        model_args = {
            "policy": model_config["policy_type"],
            "env": env,
            "verbose": model_config["verbose"],
            "tensorboard_log": str(experiment.log_dir),
            "seed": seed,
        }
        train_model = sb3_model(**model_args)

        wandb_callback_config = wandb_config["sb3_callback"]
        # TODO? Add a callback for saving the model instead of using the parameter in WandbCallback?
        callbacks: list[BaseCallback] = [
            WandbCallback(
                verbose=wandb_callback_config["verbose"],
                model_save_path=str(experiment.checkpoint_dir),
                model_save_freq=wandb_callback_config["gradient_save_freq"],
                gradient_save_freq=wandb_callback_config["gradient_save_freq"],
            ),
        ]
        if do_eval:
            eval_callback_config = experiment.config["evaluation"]
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
            callbacks.append(
                # TODO: Check EvalCallback really works with different tasks
                EvalCallback(
                    eval_env=eval_env,
                    n_eval_episodes=eval_callback_config["nb_episodes"],
                    eval_freq=eval_callback_config["frequency"],
                    log_path=str(experiment.log_dir),
                    best_model_save_path=str(experiment.checkpoint_dir),
                    deterministic=eval_callback_config["deterministic"],
                    verbose=eval_callback_config["verbose"],
                )
            )

        train_model.learn(
            total_timesteps=total_timesteps,
            progress_bar=model_config["progress_bar"],
            callback=CallbackList(callbacks),
        )

    env.close()
    run.finish()


if __name__ == "__main__":
    typer.run(main)
