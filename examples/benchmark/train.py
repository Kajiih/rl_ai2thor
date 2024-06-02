"""Run a stable-baselines3 agent in the AI2THOR RL environment."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import gymnasium as gym
import typer
import wandb
import yaml
from experiment_utils import EvalOnEachTaskAndSceneCallback, Exp, FullMetricsLogWrapper
from model_info import MODEL_CONFIG, ModelType, get_model
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from tasks_info import AvailableTask, get_action_groups_override_config, get_task_blueprint_config
from wandb.integration.sb3 import WandbCallback

from rl_thor.agents.agents import RandomAgent
from rl_thor.envs.wrappers import SimpleActionSpaceWrapper, SingleTaskWrapper

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from rl_thor.envs.ai2thor_envs import ITHOREnv

config_path = Path("examples/benchmark/config/environment_config.yaml")
with config_path.open("r") as file:
    env_config = yaml.safe_load(file)


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
    is_single_task = task not in {AvailableTask.MULTI_TASK_4, AvailableTask.MULTI_TASK_8}
    if is_single_task:
        MODEL_CONFIG["policy_type"] = "CnnPolicy"
    else:
        MODEL_CONFIG["policy_type"] = "MultiInputPolicy"

    task_blueprint_config = get_task_blueprint_config(task, nb_scenes)
    scenes = {scenes for task_config in task_blueprint_config for scenes in task_config["scenes"]}

    # === Load the environment and experiment configurations ===
    experiment = Exp(model=model_name, tasks=[task], scenes=scenes, project_name=project_name, group_name=group_name)
    config_override: dict[str, Any] = {"tasks": {"task_blueprints": task_blueprint_config}}
    config_override["no_task_advancement_reward"] = no_task_advancement_reward
    if rollout_length is not None:
        config_override["max_episode_steps"] = rollout_length
    if randomize_agent_position:
        config_override["scene_randomization"] = {"random_agent_spawn": True}
    # Add action groups override config
    config_override.update(get_action_groups_override_config(task))
    wandb_config = experiment.config["wandb"]
    tags = ["simple_actions", "single_task", model_name, *scenes, task, experiment.job_type, wandb_config["project"]]
    tags.extend((
        "single_task" if is_single_task else "multi_task",
        experiment.group_name if experiment.group_name is not None else "no_group",
        experiment.project_name,
        "no_task_advancement_reward" if no_task_advancement_reward else "with_task_advancement_reward",
        f"{nb_scenes}_scenes",
        "do_eval" if do_eval else "no_eval",
        "randomize_agent_position" if randomize_agent_position else "no_randomize_agent_position",
    ))

    run: Run = wandb.init(  # type: ignore
        config=experiment.config | env_config | {"tasks": {"task_blueprints": task_blueprint_config}},
        project=experiment.project_name,
        sync_tensorboard=wandb_config["sync_tensorboard"],
        monitor_gym=wandb_config["monitor_gym"],
        save_code=wandb_config["save_code"],
        name=experiment.name,
        group=experiment.group_name,
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
            "policy": MODEL_CONFIG["policy_type"],
            "env": env,
            "verbose": MODEL_CONFIG["verbose"],
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
            # eval_env = DummyVecEnv([
            #     lambda: make_env(
            #         config_path,
            #         config_override,
            #         experiment,
            #         is_single_task=is_single_task,
            #         log_full_metrics=log_full_env_metrics,
            #         eval_env=True,
            #     )
            # ])
            callbacks.append(
                # TODO: Check EvalCallback really works with different tasks
                EvalCallback(
                    eval_env=env,
                    n_eval_episodes=eval_callback_config["nb_episodes"],
                    eval_freq=eval_callback_config["frequency"],
                    log_path=str(experiment.log_dir),
                    best_model_save_path=str(experiment.checkpoint_dir),
                    deterministic=eval_callback_config["deterministic"],
                    verbose=eval_callback_config["verbose"],
                    callback_after_eval=EvalOnEachTaskAndSceneCallback(
                        eval_env=env, log_dir=experiment.log_dir, verbose=1
                    ),
                )
            )

        train_model.learn(
            total_timesteps=total_timesteps,
            progress_bar=MODEL_CONFIG["progress_bar"],
            callback=CallbackList(callbacks),
        )

    env.close()
    run.finish()


if __name__ == "__main__":
    typer.run(main)
