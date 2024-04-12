"""Data for the RL AI2-THOR package."""

import json
from pathlib import Path

script_path = Path(__file__).parent.resolve()
with Path(script_path / "object_types_data.json").open("r") as f:
    _OBJECT_TYPES_DICT = json.load(f)
