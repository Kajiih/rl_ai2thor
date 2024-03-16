"""Run a stable-baselines3 DQN agent in the AI2THOR RL environment."""

import gymnasium as gym
import stable_baselines3 as sb3

from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
from rl_ai2thor.envs.wrappers import SimpleActionSpaceWrapper, SingleTaskWrapper

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
    "type": "PlaceIn",
    "args": {
        "placed_object_type": "Mug",
        "receptacle_type": "DiningTable",
    },
    "scenes": "FloorPlan7",
}
override_config = {
    "discrete_actions": True,
    "target_closest_object": True,
    "tasks": [task_config],
    "action_categories": action_categories,
    "simple_movement_actions": True,
}

env: ITHOREnv = gym.make("rl_ai2thor/ITHOREnv-v0.1_sb3_ready", override_config=override_config)  # type: ignore
env = SingleTaskWrapper(SimpleActionSpaceWrapper(env))

model = sb3.DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
env.close()
