"""Wrappers for AI2THOR RL environments."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, Wrapper
from numpy.typing import NDArray

from rl_ai2thor.envs.ai2thor_envs import ITHOREnv


# %% === Wrappers ===
class SingleTaskWrapper(ObservationWrapper):
    """
    Wrapper for single task AI2THOR environment.

    It simply replaces the dictionary form observation with only environment's observation (frames) to simplify the observation space.
    It also checks that there is only one task.

    This has to be used as last wrapper in the chain because it changes the observation space.
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        # Check that there is only one task
        if len(self.task_blueprints) > 1:
            raise MoreThanOneTaskBlueprintError(self.config)
        if max(len(arg_values) for arg_values in self.task_blueprints[0].task_args.values()) > 1:
            raise MoreThanOneArgumentValueError(self.config)
        self.env: ITHOREnv  # Only for type hinting

        self.observation_space = self.env.observation_space.spaces["env_obs"]

    @staticmethod
    def observation(observation: dict[str, Any]) -> NDArray[np.uint8]:
        """Return only the environment's observation."""
        return observation["env_obs"]


class FirstChannelObservationWrapper(ObservationWrapper):
    """
    Wrapper for taking only the first channel of the observation.

    It works for both single and multi channel observations (in case grayscale is supported).
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        self.env: ITHOREnv  # Only for type hinting
        env_obs_space: gym.spaces.Box = self.env.observation_space.spaces["env_obs"]  # type: ignore
        space_shape = (env_obs_space.shape[-1], *env_obs_space.shape[:-1])
        self.observation_space.spaces["env_obs"] = gym.spaces.Box(
            low=0, high=255, shape=space_shape, dtype=env_obs_space.dtype
        )

    @staticmethod
    def observation(observation: dict[str, NDArray[np.uint8] | str]) -> dict[str, NDArray[np.uint8] | str]:
        """Return only the first channel of the observation."""
        env_obs = np.moveaxis(observation["env_obs"], -1, 0)  # type: ignore
        observation["env_obs"] = env_obs
        return observation


class NormalizeActionWrapper(ActionWrapper):
    """
    Wrapper for normalizing the action space.

    It particular, it enables using values from [-1, 1] instead of [0, 1] for target_object_position.

    It cannot be used if target_closest_object is True in the environment's config.
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        self.env: ITHOREnv  # Only for type hinting
        self.action_space.spaces["target_closest_object"] = gym.spaces.Box(low=-1, high=1, shape=(2,))

    @staticmethod
    def action(action: dict[str, Any]) -> dict[str, Any]:
        """Convert from [-1, 1] to [0, 1] for target_object_position."""
        action["target_object_position"] = (action["target_object_position"] + 1) / 2
        return action


# TODO: Rewrite this wrapper and remove the part in the main code that handles the different environment modes discrete_actions/target_closest_object
class SimpleActionSpaceWrapper(ActionWrapper):
    """
    Wrapper to use when the environment is in discrete_actions and target_closest_object mode.

    It removes the "target_closest_object" action from the action space.
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        self.env: ITHOREnv  # Only for type hinting
        self.action_space = self.env.action_space.space["action_index"]

    @staticmethod
    def action(action: int) -> dict[str, Any]:
        """Convert from action index to dictionary."""
        return {"action_index": action}


# %% === Exceptions ===
class MoreThanOneTaskBlueprintError(Exception):
    """Exception raised when there is more than one task blueprint in an environment wrapped by SingleTaskWrapper."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def __str__(self) -> str:
        return f"The environment has more than one task blueprint, which is incompatible with {SingleTaskWrapper.__name__}; config: {self.config}"


class MoreThanOneArgumentValueError(Exception):
    """Exception raised when there is more than one possible argument value for the task in an environment wrapped by SingleTaskWrapper."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def __str__(self) -> str:
        return f"The environment's task has more than one possible argument value, which is incompatible with {SingleTaskWrapper.__name__}; config: {self.config}"
