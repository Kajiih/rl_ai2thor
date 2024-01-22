"""Module for types for the AI2THOR RL environment.

"""

from ai2thor.server import MultiAgentEvent, Event
from typing import Union


EventLike = Union[MultiAgentEvent, Event]
