"""
Module for defining reward functions for AI2THOR RL environments.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ai2thor.controller import Controller

    from rl_ai2thor.utils.ai2thor_types import EventLike


class BaseRewardHandler(ABC):
    """Base class for reward handlers."""

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
    def reset(self, controller: Controller) -> tuple[bool, dict[str, Any]]:
        """
        Reset the reward handler.

        Args:
            controller (Controller): AI2THOR controller at the beginning of the episode.

        Returns:
            terminated (bool): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """


class MultiRewardHandler(BaseRewardHandler):
    """Reward handler for that combines multiple reward handlers."""

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

    def reset(self, controller: Controller) -> tuple[bool, dict[str, dict[str, Any]]]:
        """
        Reset the reward handlers.

        Args:
            controller (Controller): AI2THOR controller at the beginning of the episode.

        Returns:
            terminated (bool): Whether one of the reward handlers has terminated the episode.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        task_completions, infos = zip(
            *(reward_handler.reset(controller) for reward_handler in self.reward_handlers), strict=True
        )
        combined_info = {
            handler.__class__.__name__: info for handler, info in zip(self.reward_handlers, infos, strict=True)
        }
        return any(task_completions), combined_info
