"""
Module for defining reward functions for AI2-THOR RL environments.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ai2thor.controller import Controller
    from ai2thor.server import Event


class BaseRewardHandler(ABC):
    """Base class for reward handlers."""

    @abstractmethod
    def get_reward(
        self,
        event: Event,
        controller_action: dict[str, Any],
    ) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the reward, task completion and additional information for the given event.

        Args:
            event (Event): Event to calculate the reward for.
            controller_action (dict[str, Any]): Dictionary containing the information about the
                action executed by the controller, obtained with controller.last_action.

        Returns:
            reward (float): Reward for the event.
            terminated (bool, Optional): Whether the episode has terminated.
            info (dict[str, Any]): Additional information.
        """

    @abstractmethod
    def reset(self, controller: Controller) -> tuple[bool, bool, dict[str, Any]]:
        """
        Reset the reward handler.

        In some cases, the reset can fail (e.g. if the task and the scene are incompatible).

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the reset is successful.
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

    def get_reward(
        self,
        event: Event,
        controller_action: dict[str, Any],
    ) -> tuple[float, bool, dict[str, dict[str, Any]]]:
        """
        Return the sum of the rewards from the reward handlers.

        Args:
            event (Event): Event to calculate the reward for.
            controller_action (dict[str, Any]): Dictionary containing the information about the
                action executed by the controller.

        Returns:
            reward (float): Sum of the rewards from the reward handlers for the event.
            terminated (bool): Whether one of the reward handlers has terminated the episode.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        rewards, task_completions, infos = zip(
            *(reward_handler.get_reward(event, controller_action) for reward_handler in self.reward_handlers),
            strict=True,
        )
        combined_info = {
            handler.__class__.__name__: info for handler, info in zip(self.reward_handlers, infos, strict=True)
        }
        return sum(rewards), any(task_completions), combined_info

    def reset(self, controller: Controller) -> tuple[bool, bool, dict[str, dict[str, Any]]]:
        """
        Reset the reward handlers.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the reset is successful.
            terminated (bool): Whether one of the reward handlers has terminated the episode.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        resets_successful, task_completions, infos = zip(
            *(reward_handler.reset(controller) for reward_handler in self.reward_handlers), strict=True
        )
        combined_info = {
            handler.__class__.__name__: info for handler, info in zip(self.reward_handlers, infos, strict=True)
        }
        return all(resets_successful), any(task_completions), combined_info
