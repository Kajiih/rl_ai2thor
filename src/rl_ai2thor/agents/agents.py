"""Agents in AI2THOR RL environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from rl_ai2thor.agents.callbacks import BaseCallback

if TYPE_CHECKING:
    import gymnasium as gym

    from rl_ai2thor.envs.ai2thor_envs import BaseAI2THOREnv


# TODO: Improve type hints with more specific types
# %% === Agents ===
class BaseAgent[ObsType, ActType](ABC):
    """Base class for agents in the AI2THOR RL environment."""

    def __init__(self, env: BaseAI2THOREnv[ObsType, ActType], callback: BaseCallback | None = None) -> None:
        """
        Initialize the agent.

        Args:
            env (BaseAI2THOREnv): Environment to interact with.
            callback (BaseCallback, optional): Callback to use. Defaults to None.
        """
        self.env = env
        if callback is None:
            callback = BaseCallback()
        self.callback = callback

    @abstractmethod
    def __call__(self, obs: ObsType) -> ActType:
        """
        Return the action to take given the current observation.

        Args:
            obs (ObsType): Current observation.

        Returns:
            action (ActType): Action to take.
        """

    def continue_episode(
        self,
        obs: ObsType,
        max_steps: int | None = None,
    ) -> tuple[float, int, ObsType, float, bool, bool, dict[str, Any]]:
        """
        Continue the episode and return the final state of the environment and the number of steps taken.

        Args:
            obs (ObsType): Current observation.
            max_steps (int, optional): Maximum number of steps to run. Defaults to None.

        Returns:
            sum_reward (float): Sum of the rewards obtained throughout the episode.
            nb_steps (int): Number of steps taken.
            obs (ObsType): Observation at the end of the episode.
            reward (float): Extra reward obtained on the last step.
            terminated (bool): Whether the episode has terminated on the last step.
            truncated (bool): Whether the episode has been truncated on the last step.
            info (dict[str, Any]): Additional information about the last step.
        """
        terminated, truncated = False, False
        nb_steps = 0
        sum_reward = 0
        while not terminated and not truncated:
            action, obs, reward, terminated, truncated, info = self.step(obs)
            sum_reward += reward
            nb_steps += 1
            if max_steps is not None and nb_steps >= max_steps:
                break
        return sum_reward, nb_steps, obs, reward, terminated, truncated, info

    def run_episode(
        self,
        nb_episodes: int = 1,
        total_max_steps: int | None = None,
    ) -> tuple[float, int, ObsType, float, bool, bool, dict[str, Any]]:
        """
        Start and run a certain number of episodes and return the final state of the environment and the number of steps taken.

        Args:
            nb_episodes (int, optional): Number of episodes to run. Defaults to 1.
            total_max_steps (int, optional): Maximum number of steps to run in total. Defaults to None.

        Returns:
            total_reward (float): Sum of the rewards obtained throughout the episodes.
            total_nb_steps (int): Total number of steps taken.
            obs (ObsType): Observation at the end of the last episode.
            reward (float): Extra reward obtained on the last step.
            terminated (bool): Whether the last episode has terminated on the last step.
            truncated (bool): Whether the last episode has been truncated on the last step.
            info (dict[str, Any]): Additional information about the last step.
        """
        total_reward = 0
        total_nb_steps = 0

        for _ep in range(nb_episodes):
            obs, info = self.reset()
            remaining_steps = total_max_steps - total_nb_steps if total_max_steps is not None else None
            sum_reward, nb_steps, obs, reward, terminated, truncated, info = self.continue_episode(obs, remaining_steps)
            total_reward += sum_reward
            total_nb_steps += nb_steps
            if total_max_steps is not None and total_nb_steps >= total_max_steps:
                break

        return total_reward, total_nb_steps, obs, reward, terminated, truncated, info

    def step(self, obs: ObsType) -> tuple[ActType, ObsType, float, bool, bool, dict[str, Any]]:
        """
        Return the next action to take and the resulting state of the environment.

        Args:
            obs (ObsType): Current observation.

        Returns:
            action (ActType): Action taken.
            obs (ObsType): Observation after the action.
            reward (float): Reward obtained after the action.
            terminated (bool): Whether the episode has terminated after the action.
            truncated (bool): Whether the episode has been truncated after the action.
            info (dict[str, Any]): Additional information about the action.
        """
        action = self(obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.callback.on_step(obs, float(reward), terminated, truncated, info)
        return action, obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Close the agent."""
        self.env.close()
        self.callback.on_close()

    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment and return the initial observation.

        Args:
            seed (int, optional): Seed to use for the environment. Defaults to None.

        Returns:
            obs (ObsType): Initial observation.
            info (dict[str, Any]): Additional information about the initial state of the environment.
        """
        return self.env.reset(seed=seed)


class RandomAgent[ObsType, ActType](BaseAgent[ObsType, ActType]):
    """A random agent."""

    def __init__(self, env: BaseAI2THOREnv[ObsType, ActType], callback: BaseCallback | None = None) -> None:
        """
        Initialize the agent.

        Args:
            env (BaseAI2THOREnv): Environment to interact with.
            callback (BaseCallback, optional): Callback to use. Defaults to None.
        """
        super().__init__(env, callback)

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """
        Return a random action.

        Args:
            obs (ObsType, optional): Current observation. Defaults to None.

        Returns:
            ActType: Random action.
        """
        action = self.env.action_space.sample()
        return action
