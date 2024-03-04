"""
Module for defining reward functions for AI2-THOR environments.

TODO: Finish module docstring.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from rl_ai2thor.utils.ai2thor_types import EventLike


class BaseRewardHandler(ABC):
    """Base class for reward handlers for AI2-THOR environments."""

    @abstractmethod
    def get_reward(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the reward, task completion and additional information for the given event.

        Args:
            event (Any): Event to calculate the reward for.

        Returns:
            reward (float): Reward for the event.
            terminated (bool, Optional): Whether the episode has terminated.
            info (dict[str, Any]): Additional information.
        """

    @abstractmethod
    def reset(self, event: EventLike) -> tuple[bool, dict[str, Any]]:
        """
        Reset the reward handler.

        Args:
            event (Any): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            terminated (bool): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """


class MultiRewardHandler(BaseRewardHandler):
    """Reward handler for AI2-THOR environments."""

    def __init__(self, reward_handlers: Iterable[BaseRewardHandler]) -> None:
        """
        Initialize the reward handler.

        Args:
            reward_handlers (Iterable[BaseRewardHandler]): Reward handlers to use.
        """
        self.reward_handlers = reward_handlers

    def get_reward(self, event: EventLike) -> tuple[float, bool, dict[str, dict[str, Any]]]:
        """
        Return the sum of the rewards from the reward handlers.

        Args:
            event (Any): Event to calculate the reward for.

        Returns:
            reward (float): Sum of the rewards from the reward handlers for the event.
            terminated (bool): Whether one of the reward handlers has terminated the episode.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        rewards, task_completions, infos = zip(
            *(reward_handler.get_reward(event) for reward_handler in self.reward_handlers), strict=True
        )
        combined_info = {
            handler.__class__.__name__: info for handler, info in zip(self.reward_handlers, infos, strict=True)
        }
        return sum(rewards), any(task_completions), combined_info

    def reset(self, event: EventLike) -> tuple[bool, dict[str, dict[str, Any]]]:
        """
        Reset the reward handlers.

        Args:
            event (Any): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            terminated (bool): Whether one of the reward handlers has terminated the episode.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        task_completions, infos = zip(
            *(reward_handler.reset(event) for reward_handler in self.reward_handlers), strict=True
        )
        combined_info = {
            handler.__class__.__name__: info for handler, info in zip(self.reward_handlers, infos, strict=True)
        }
        return any(task_completions), combined_info
