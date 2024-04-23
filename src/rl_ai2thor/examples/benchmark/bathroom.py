"""Run a stable-baselines3 agent in the AI2-THOR RL environment."""

from datetime import datetime
from typing import TYPE_CHECKING

import gymnasium as gym
import stable_baselines3 as sb3
import wandb
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.wandb_run import Run

from rl_ai2thor.envs.tasks.tasks import TaskType
from rl_ai2thor.envs.wrappers import SimpleActionSpaceWrapper, SingleTaskWrapper

if TYPE_CHECKING:
    from rl_ai2thor.envs.ai2thor_envs import ITHOREnv


model_name = "PPO"  # "PPO", "A2C", "DQN"
if model_name == "PPO":
    sb3_model = sb3.PPO
elif model_name == "A2C":
    sb3_model = sb3.A2C
elif model_name == "DQN":
    sb3_model = sb3.DQN
else:
    raise ValueError(f"Model name '{model_name}' not supported.")

task = TaskType.PREPARE_FOR_SHOWER
scenes = ["FloorPlan401"]  # "FloorPlan7", "FloorPlan8", "Kitchen", "Bathroom"

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 500_000,
    "env_name": "rl_ai2thor/ITHOREnv-v0.1_sb3_ready",
    "gradient_save_freq": 100,
}
run: Run = wandb.init(  # type: ignore
    project="rl_ai2thor",
    config=config,
    sync_tensorboard=True,
    # monitor_gym=True,
    save_code=True,
    tags=[model_name, "sb3", "simple_actions", "benchmark", "randomized spawn", *scenes, task],
    notes=f"Simple {model_name} agent for AI2-THOR benchmarking on {task} task.",
    name=f"{model_name}_{task}_[{",".join(scenes)}]_{datetime.now().strftime("%Y%m%d-%H%M%S")}",
    group=f"{task}_{scenes}",
)

action_categories = {
    # === Navigation actions ===
    "crouch_actions": False,
    # === Object manipulation actions ===
    "drop_actions": False,
    "throw_actions": False,
    "push_pull_actions": False,
    # === Object interaction actions ===
    "open_close_actions": False,
    "toggle_actions": True,
    "slice_actions": False,
    "use_up_actions": True,
    "liquid_manipulation_actions": False,
}
task_config = {
    "type": task,
    "args": {},
    "scenes": scenes,
}
override_config = {
    "discrete_actions": True,
    "target_closest_object": True,
    "tasks": [task_config],
    "action_categories": action_categories,
    "simple_movement_actions": True,
    "max_episode_steps": 1000,
    "random_agent_spawn": True,
}


def main() -> None:
    """Run the DQN agent in the AI2-THOR environment."""
    env: ITHOREnv = gym.make(config["env_name"], override_config=override_config)  # type: ignore

    env = SingleTaskWrapper(SimpleActionSpaceWrapper(env))

    model = sb3_model(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=config["gradient_save_freq"],
            verbose=2,
            model_save_path=f"models/{run.id}",
        ),
    )
    env.close()


if __name__ == "__main__":
    main()
    run.finish()
