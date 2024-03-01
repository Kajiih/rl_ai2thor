"""Data for the RL AI2THOR package."""

import json
from pathlib import Path

with Path("src/rl_ai2thor/data/object_types.json").open("r") as f:
    OBJECT_TYPES_DATA = json.load(f)
