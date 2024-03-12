"""Main module for AI2THOR RL environment."""

from gymnasium.envs.registration import register

register(
    id="rl_ai2thor/ITHOREnv-v0.1",
    entry_point="rl_ai2thor.envs.ai2thor_envs:ITHOREnv",
    # TODO: Check if we add kwargs
    # TODO: Add additional_wrappers
)
