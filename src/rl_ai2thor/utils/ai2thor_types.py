"""Module for custom types for the AI2THOR RL environment."""

from ai2thor.server import Event, MultiAgentEvent

type EventLike = MultiAgentEvent | Event
