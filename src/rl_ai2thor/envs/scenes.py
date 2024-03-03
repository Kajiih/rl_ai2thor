"""
Scenes in AI2THOR RL environment.

TODO: Finish module docstring.
"""

from enum import StrEnum

type SceneId = str


class SceneGroup(StrEnum):
    """Scene groups in AI2THOR environment."""

    KITCHEN = "Kitchen"
    LIVING_ROOM = "LivingRoom"
    BEDROOM = "Bedroom"
    BATHROOM = "Bathroom"


SCENE_IDS = {
    SceneGroup.KITCHEN: [f"FloorPlan{i}" for i in range(1, 31)],
    SceneGroup.LIVING_ROOM: [f"FloorPlan{200 + i}" for i in range(1, 31)],
    SceneGroup.BEDROOM: [f"FloorPlan{300 + i}" for i in range(1, 31)],
    SceneGroup.BATHROOM: [f"FloorPlan{400 + i}" for i in range(1, 31)],
}
