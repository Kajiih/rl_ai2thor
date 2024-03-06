"""Agents in AI2THOR RL environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from rl_ai2thor.agents.callbacks import BaseCallback

if TYPE_CHECKING:
    import gymnasium as gym


# TODO: Improve type hints with more specific types
# %% === Agents ===
class BaseAgent(ABC):
    """Base class for agents in the AI2THOR RL environment."""

    def __init__(self, env: gym.Env, callback: BaseCallback | None = None) -> None:
        """Initialize the agent."""
        self.env = env
        if callback is None:
            callback = BaseCallback()
        self.callback = callback

    @abstractmethod
    def __call__(self, obs: Any = None) -> Any:
        """Return the action to take given the current observation."""

    def continue_episode(
        self,
        obs: Any,
        max_steps: int | None = None,
    ) -> tuple[float, int, Any, float, bool, bool, dict[str, Any]]:
        """Continue the episode and return the final state of the environment and the number of steps taken."""
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
    ) -> tuple[float, int, Any, float, bool, bool, dict[str, Any]]:
        """Start and run a certain number of episodes and return the final state of the environment and the number of steps taken."""
        total_reward = 0
        total_nb_steps = 0

        for _ep in range(nb_episodes):
            obs = self.reset()
            remaining_steps = total_max_steps - total_nb_steps if total_max_steps is not None else None
            sum_reward, nb_steps, obs, reward, terminated, truncated, info = self.continue_episode(obs, remaining_steps)
            total_reward += sum_reward
            total_nb_steps += nb_steps
            if total_max_steps is not None and total_nb_steps >= total_max_steps:
                break
        return total_reward, total_nb_steps, obs, reward, terminated, truncated, info

    def step(self, obs: Any) -> tuple[Any, Any, float, bool, bool, dict[str, Any]]:
        """Return the next action to take and the resulting state of the environment."""
        action = self(obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.callback.on_step(obs, float(reward), terminated, truncated, info)
        return action, obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Close the agent."""
        self.env.close()
        self.callback.on_close()

    def reset(self, seed: int | None = None) -> Any:
        """Reset the environment and return the initial observation."""
        return self.env.reset(seed=seed)


class RandomAgent(BaseAgent):
    """A random agent."""

    def __init__(self, env: gym.Env, callback: BaseCallback | None = None) -> None:
        """Initialize the agent."""
        super().__init__(env, callback)

    def __call__(self, obs: Any = None) -> Any:  # noqa: ARG002
        """Return a random action."""
        action = self.env.action_space.sample()
        return action
