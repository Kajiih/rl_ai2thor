"""Run a stable-baselines3 DQN agent in the AI2THOR RL environment."""

from typing import TYPE_CHECKING

import gymnasium as gym
import stable_baselines3 as sb3
import wandb
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.wandb_run import Run

from rl_ai2thor.envs.wrappers import SimpleActionSpaceWrapper, SingleTaskWrapper

if TYPE_CHECKING:
    from rl_ai2thor.envs.ai2thor_envs import ITHOREnv

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 25_000,
    "env_name": "rl_ai2thor/ITHOREnv-v0.1_sb3_ready",
    "gradient_save_freq": 100,
}
run: Run = wandb.init(  # type: ignore
    project="rl_ai2thor",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
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
        "picked_object_type": "Mug",
    },
    "scenes": "FloorPlan7",
}
override_config = {
    "discrete_actions": True,
    "target_closest_object": True,
    "tasks": [task_config],
    "action_categories": action_categories,
    "simple_movement_actions": True,
    "max_episode_steps": 500,
}


def main() -> None:
    """Run the DQN agent in the AI2THOR environment."""
    env: ITHOREnv = gym.make(config["env_name"], override_config=override_config)  # type: ignore

    env = SingleTaskWrapper(SimpleActionSpaceWrapper(env))

    model = sb3.DQN(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
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
