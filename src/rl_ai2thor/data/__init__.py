"""Data for the RL AI2THOR package."""

import json
from pathlib import Path

script_path = Path(__file__).parent.resolve()
with Path(script_path / "object_types.json").open("r") as f:
    OBJECT_TYPES_DATA = json.load(f)
