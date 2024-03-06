"""Agents in AI2THOR RL environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, SupportsFloat

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

    def run_episode(
        self,
        obs: Any,
        max_steps: int | None = None,
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any], int]:
        """Run an episode with the agent and return the final state of the environment and the number of steps taken."""
        terminated = False
        nb_steps = 0
        while not terminated and (max_steps is None or nb_steps < max_steps):
            action, obs, reward, terminated, truncated, info = self.step(obs)
            nb_steps += 1
        return obs, reward, terminated, truncated, info, nb_steps

    def step(self, obs: Any) -> tuple[Any, Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Return the next action to take and the resulting state of the environment."""
        action = self(obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.callback.on_step(obs, reward, terminated, truncated, info)
        return action, obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the agent."""
        self.env.close()
        self.callback.on_close()

    def reset(self, seed: int | None = None) -> Any:
        """Reset the environment and return the initial observation."""
        return self.env.reset(seed=seed)


class RandomAgent(BaseAgent):
    """A random agent."""

    def __init__(self, env: gym.Env, callback: BaseCallback) -> None:
        """Initialize the agent."""
        super().__init__(env, callback)

    def __call__(self, obs: Any = None) -> Any:  # noqa: ARG002
        """Return a random action."""
        action = self.env.action_space.sample()
        return action
