"""Wrappers for RL-THOR environments."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper
from numpy.typing import NDArray

from rl_thor.envs._config import EnvConfig
from rl_thor.envs.ai2thor_envs import ITHOREnv


# %% === Wrappers ===
class SingleTaskWrapper(ObservationWrapper, ITHOREnv):
    """
    Wrapper for single task AI2-THOR environment.

    It simply replaces the dictionary form observation with only environment's observation (frames) to simplify the observation space.
    It also checks that there is only one task.

    This has to be used as last wrapper in the chain because it changes the observation space.
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        self.env: ITHOREnv  # Only for type hinting
        unwrapped_env: ITHOREnv = self.unwrapped  # type: ignore
        # Check that there is only one task
        if len(unwrapped_env.task_blueprints) > 1:
            raise MoreThanOneTaskBlueprintError(unwrapped_env.config)

        self.observation_space = self.env.observation_space.spaces["env_obs"]

    def observation(self, observation: dict[str, Any]) -> NDArray[np.uint8]:  # noqa: PLR6301
        """Return only the environment's observation."""
        return observation["env_obs"]


class ChannelFirstObservationWrapper(ObservationWrapper, ITHOREnv):
    """
    Wrapper for changing the observation space from channel last to channel first.

    It works for both single and multi channel observations (in case grayscale is supported).
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        self.env: ITHOREnv  # Only for type hinting
        self.observation_space: gym.spaces.Dict
        env_obs_space: gym.spaces.Box = self.env.observation_space.spaces["env_obs"]  # type: ignore
        space_shape = (env_obs_space.shape[-1], *env_obs_space.shape[:-1])
        self.observation_space.spaces["env_obs"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=space_shape,
            dtype=env_obs_space.dtype,  # type: ignore
        )

    def observation(self, observation: dict[str, NDArray[np.uint8] | str]) -> dict[str, NDArray[np.uint8] | str]:  # noqa: PLR6301
        """Return only the first channel of the observation."""
        env_obs = np.moveaxis(observation["env_obs"], -1, 0)  # type: ignore
        observation["env_obs"] = env_obs
        return observation


class NormalizeActionWrapper(ActionWrapper, ITHOREnv):
    """
    Wrapper for normalizing the action space.

    It particular, it enables using values from [-1, 1] instead of [0, 1] for target_object_coordinates.

    It cannot be used if target_closest_object is True in the environment's config.
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        self.env: ITHOREnv  # Only for type hinting
        self.action_space: gym.spaces.Dict
        self.action_space.spaces["target_object_coordinates"] = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def action(self, action: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR6301
        """Convert from [-1, 1] to [0, 1] for target_object_coordinates."""
        action["target_object_coordinates"] = [(coord + 1) / 2 for coord in action["target_object_coordinates"]]
        return action


# TODO: Rewrite this wrapper and remove the part in the main code that handles the different environment modes discrete_actions/target_closest_object
class SimpleActionSpaceWrapper(ActionWrapper, ITHOREnv):
    """
    Wrapper to use when the environment is in discrete_actions and target_closest_object mode.

    It removes the "target_closest_object" action from the action space.
    """

    def __init__(self, env: ITHOREnv) -> None:
        """Initialize the wrapper."""
        super().__init__(env)
        self.env: ITHOREnv  # Only for type hinting
        self.unwrapped: ITHOREnv  # Only for type hinting
        self.action_space = self.unwrapped.action_space.spaces["action_index"]  # type: ignore
        if not (
            self.unwrapped.config.action_modifiers.discrete_actions
            and self.unwrapped.config.action_modifiers.target_closest_object
        ):
            raise NotSimpleActionEnvironmentMode(self.unwrapped.config)

    def action(self, action: int) -> dict[str, Any]:  # noqa: PLR6301
        """Convert from action index to dictionary."""
        return {"action_index": action}


# %% === Exceptions ===
class MoreThanOneTaskBlueprintError(Exception):
    """Exception raised when there is more than one task blueprint in an environment wrapped by SingleTaskWrapper."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def __str__(self) -> str:
        return f"The environment has more than one task blueprint, which is incompatible with {SingleTaskWrapper.__name__}; task blueprints: {self.config.tasks.task_blueprints}"


# TODO: Delete; deprecated since task blueprint don't support multiple arguments values anymore
class MoreThanOneArgumentValueError(Exception):
    """Exception raised when there is more than one possible value for a task argument in an environment wrapped by SingleTaskWrapper."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def __str__(self) -> str:
        return f"The environment's task has more than one possible argument value, which is incompatible with {SingleTaskWrapper.__name__}; config['tasks']]: {self.config["tasks"]}"


class NotSimpleActionEnvironmentMode(Exception):
    """Exception raised when the environment is not in discrete actions and target closest object mode and it is wrapped by SimpleActionSpaceWrapper."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def __str__(self) -> str:
        return f"The environment is not in the required mode (discrete_actions=True and target_closest_object=True) for {SimpleActionSpaceWrapper.__name__}; action modifiers: {self.config.action_modifiers}"
