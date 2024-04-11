"""Data for the RL AI2-THOR package."""

import json
from dataclasses import dataclass
from pathlib import Path

from rl_ai2thor.envs.sim_objects import SimObjectType, SimObjFixedProp


@dataclass(frozen=True)
class ObjTypeData:
    """Data for a sim object type in AI2-THOR."""

    scenes: str
    actionable_properties: frozenset[SimObjFixedProp]
    materials_properties: frozenset[SimObjFixedProp]
    compatible_receptacles: frozenset[SimObjectType]
    contextual_interactions: str


script_path = Path(__file__).parent.resolve()
with Path(script_path / "object_types_data.json").open("r") as f:
    object_types_dict = json.load(f)
    OBJECT_TYPES_DATA = {
        SimObjectType(sim_object_type): ObjTypeData(
            scenes=object_type_data["scenes"],
            actionable_properties=frozenset(object_type_data["actionable_properties"]),
            materials_properties=frozenset(object_type_data["materials_properties"]),
            compatible_receptacles=frozenset(object_type_data["compatible_receptacles"]),
            contextual_interactions=object_type_data["contextual_interactions"],
        )
        for sim_object_type, object_type_data in object_types_dict.items()
    }
