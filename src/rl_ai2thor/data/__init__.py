"""Data for the RL AI2-THOR package."""

import json
from pathlib import Path

script_path = Path(__file__).parent.resolve()
with Path(script_path / "object_types_data.json").open("r") as f:
    OBJECT_TYPES_DATA = json.load(f)
    # Change all lists to frozensets
    for sim_object_type in OBJECT_TYPES_DATA:
        for key, value in OBJECT_TYPES_DATA[sim_object_type].items():
            if isinstance(value, list):
                OBJECT_TYPES_DATA[sim_object_type][key] = frozenset(value)
