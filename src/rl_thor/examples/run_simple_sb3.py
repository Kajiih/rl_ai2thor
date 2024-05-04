"""Run a stable-baselines3 agent in the AI2-THOR RL environment."""

from typing import TYPE_CHECKING

import gymnasium as gym
import stable_baselines3 as sb3
import wandb
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.wandb_run import Run

from rl_thor.envs.wrappers import SimpleActionSpaceWrapper, SingleTaskWrapper

if TYPE_CHECKING:
    from rl_thor.envs.ai2thor_envs import ITHOREnv


model_name = "PPO"  # "PPO", "A2C", "DQN"
if model_name == "PPO":
    sb3_model = sb3.PPO
elif model_name == "A2C":
    sb3_model = sb3.A2C
elif model_name == "DQN":
    sb3_model = sb3.DQN
else:
    raise ValueError(f"Model name '{model_name}' not supported.")

scenes = ["FloorPlan7"]  # "FloorPlan7", "FloorPlan8", "Kitchen"

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 20_000,
    "env_name": "rl_thor/ITHOREnv-v0.1_sb3_ready",
    "gradient_save_freq": 10,
}
run: Run = wandb.init(  # type: ignore
    project="rl_thor",
    config=config,
    sync_tensorboard=True,
    # monitor_gym=True,
    save_code=True,
    tags=[model_name, "sb3", "simple_action", "single_task"],
    notes=f"Simple {model_name} agent",
    name=f"{model_name}_simple_single_task_[{",".join(scenes)}]",
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
    "toggle_actions": False,
    "slice_actions": False,
    "use_up_actions": False,
    "liquid_manipulation_actions": False,
}
task_config = {
    "type": "Pickup",
    "args": {
        "picked_up_object_type": "Tomato",  # Mug
    },
    "scenes": scenes,
}
config_override = {
    "discrete_actions": True,
    "target_closest_object": True,
    "tasks": [task_config],
    "action_categories": action_categories,
    "simple_movement_actions": True,
    "max_episode_steps": 500,
}


def main() -> None:
    """Run the DQN agent in the AI2-THOR environment."""
    env: ITHOREnv = gym.make(config["env_name"], config_override=config_override)  # type: ignore

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
