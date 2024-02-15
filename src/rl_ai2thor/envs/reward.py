"""
Module for defining reward functions for AI2-THOR environments.

TODO: Finish module docstring.
"""

from typing import Any

from rl_ai2thor.envs.tasks import GraphTask
from rl_ai2thor.utils.ai2thor_types import EventLike


# TODO: Make more general for non-graph tasks
# TODO: Add more options
class GraphTaskRewardHandler:
    """
    Reward handler for AI2-THOR environments.

    TODO: Finish docstring
    """

    def __init__(self, task: GraphTask) -> None:
        """
        Initialize the reward handler.

        Args:
            task (GraphTask): Task to calculate rewards for.
        """
        self.task = task
        self.last_step_advancement = 0

    # TODO: Add shortcut when the action failed or similar special cases
    def get_reward(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the reward for the given event.

        Args:
            event (Any): Event to calculate the reward for.

        Returns:
            reward (float): Reward for the event.
            task_completion (bool): Whether the task has been completed.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        task_advancement, task_completion, info = self.task.get_task_advancement(event)
        reward = task_advancement - self.last_step_advancement
        self.last_step_advancement = task_advancement

        return reward, task_completion, info

    def reset(self, event: EventLike) -> tuple[bool, dict[str, Any]]:
        """
        Reset the reward handler.

        Args:
            event (Any): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            task_completion (bool): Whether the task is already completed.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        # Reset the task
        task_advancement, task_completion, info = self.task.get_task_advancement(event)
        # Initialize the last step advancement
        self.last_step_advancement = task_advancement

        return task_completion, info
