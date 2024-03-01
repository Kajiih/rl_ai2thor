"""
Scenes in AI2THOR RL environment.

TODO: Finish module docstring.
"""

SCENE_IDS = {
    "Kitchen": [f"FloorPlan{i}" for i in range(1, 31)],
    "LivingRoom": [f"FloorPlan{200 + i}" for i in range(1, 31)],
    "Bedroom": [f"FloorPlan{300 + i}" for i in range(1, 31)],
    "Bathroom": [f"FloorPlan{400 + i}" for i in range(1, 31)],
}
