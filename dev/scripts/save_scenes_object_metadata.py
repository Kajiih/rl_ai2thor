"""Script to save a sample from objects metadata and a list the objects for each scene in AI2-THOR."""

import json
from pathlib import Path

from ai2thor.controller import Controller

objects_metadata_path = Path(__file__).parent.parent / "data"

controller = Controller()
scene_list = controller.ithor_scenes(
    include_kitchens=True, include_living_rooms=True, include_bedrooms=True, include_bathrooms=True
)

scene_objects_dict = {}
for scene in scene_list:
    event = controller.reset(scene)
    objects_metadata = event.metadata["objects"]
    scene_objects_metadata = {obj["objectId"]: obj for obj in objects_metadata}
    scene_objects_dict[scene] = [obj["objectType"] for obj in objects_metadata]

    with (objects_metadata_path / "objects_metadata_samples" / f"{scene}.json").open("w") as f:
        json.dump(scene_objects_metadata, f, indent=4)

with (objects_metadata_path / f"scene_objects_list.json").open("w") as f:
    json.dump(scene_objects_dict, f, indent=4)

print(f"Objects metadata samples saved to {objects_metadata_path}")
controller.stop()
