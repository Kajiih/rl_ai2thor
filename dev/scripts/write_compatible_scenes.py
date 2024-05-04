"""Script to compute and write the list of scenes that are compatible with some task."""
# TODO: Make this usable and parametrizable with command-line arguments.

from pathlib import Path

from ai2thor.controller import Controller

from rl_thor.envs.scenes import SCENE_IDS
from rl_thor.envs.tasks.tasks import PrepareMealTask

dump_path = Path("../data/compatible_scenes.txt")
dump_path.parent.mkdir(parents=True, exist_ok=True)

scenes = []
for scene_list in SCENE_IDS.values():
    scenes += scene_list

task = PrepareMealTask()
controller = Controller()

compatible_scenes = []
for scene in scenes:
    controller.reset(scene)
    reset_successful, _, _, _ = task.preprocess_and_reset(controller)
    if reset_successful:
        compatible_scenes.append(scene)
controller.stop()

with dump_path.open("w") as f:
    for scene in compatible_scenes:
        f.write(scene + "\n")


print(f"\n{compatible_scenes = }")
