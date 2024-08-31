"""Data for the RL AI2-THOR package."""

import json
from pathlib import Path

script_path = Path(__file__).parent.resolve()
object_types_data_path = Path(script_path / "object_types_data.json")
with object_types_data_path.open("r") as f:
    _OBJECT_TYPES_DICT = json.load(f)
