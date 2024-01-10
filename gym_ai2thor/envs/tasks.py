"""
Tasks for the AI2THOR RL environment.

TODO: Finish module docstrings.
"""

from abc import abstractmethod
from dataclasses import dataclass

import ai2thor.server


# %% Task definitions
@dataclass
class BaseTask:
    """
    Base class for tasks in the environment.

    Methods:
        get_reward(event): Returns the reward corresponding to the event.
    """

    @abstractmethod
    def get_reward(self, event: ai2thor.server.Event) -> tuple[float, bool]:
        """
        Returns the reward corresponding to the event.

        Returns:
            reward (float): Reward obtained at the step.
            done (bool): Whether the episode finished at this step.
        """
        raise NotImplementedError


class UndefinedTask(BaseTask):
    """
    Undefined task raising an error when used.
    """

    def get_reward(self, event):
        raise NotImplementedError(
            "Task is undefined. This is an unexpected behavior, maybe you forgot to reset the environment?"
        )


@dataclass
class DummyTask(BaseTask):
    """
    Dummy task for testing purposes.
    A reward of 0 is returned at each step and the episode is never terminated.
    """

    def get_reward(self, event):
        return 0, False
