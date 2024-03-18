"""
Scenes in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from enum import StrEnum
from typing import NewType

# type SceneId = str
SceneId = NewType("SceneId", str)


class SceneGroup(StrEnum):
    """Scene groups in AI2-THOR environment."""

    KITCHEN = "Kitchen"
    LIVING_ROOM = "LivingRoom"
    BEDROOM = "Bedroom"
    BATHROOM = "Bathroom"


SCENE_IDS: dict[SceneGroup, list[SceneId]] = {
    SceneGroup.KITCHEN: [SceneId(f"FloorPlan{i}") for i in range(1, 31)],
    SceneGroup.LIVING_ROOM: [SceneId(f"FloorPlan{200 + i}") for i in range(1, 31)],
    SceneGroup.BEDROOM: [SceneId(f"FloorPlan{300 + i}") for i in range(1, 31)],
    SceneGroup.BATHROOM: [SceneId(f"FloorPlan{400 + i}") for i in range(1, 31)],
}
