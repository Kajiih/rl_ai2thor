"""Generate data for the tasks tests."""

# %% === Setup the environment for the test tasks ===
import pickle as pkl  # noqa: S403
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ai2thor.controller import Controller

from rl_thor.envs.tasks._item_prop_variable import RECEPTACLE_MAX_OBJECTS_PROP_LIMIT  # noqa: PLC2701
from rl_thor.envs.tasks.tasks import (
    ArrangeCutleryTask,
    CleanToiletsTask,
    CleanUpBathroomTask,
    CleanUpBedroomTask,
    CleanUpKitchenTask,
    CleanUpLivingRoomTask,
    ClearDiningTable,
    DoHomeworkTask,
    PileUpDishes,
    PrepareGoingToBedTask,
    WashCutleryTask,
)
from rl_thor.envs.tasks.tasks_interface import BaseTask

if TYPE_CHECKING:
    from ai2thor.server import Event

test_task_data_dir = Path("../data/test_tasks")


def main():
    """Generate data for the tasks tests."""
    controller = Controller()

    # generate_pickup_mug_data(controller)
    # generate_open_fridge_data(controller)
    # generate_place_cooled_in_apple_counter_top_data(controller)
    # generate_look_in_light_book_data(controller)
    # generate_prepare_meal_data(controller)
    # generate_prepare_going_to_bed_data(controller)
    # generate_WashCutleryTask_data(controller)
    # generate_ClearDiningTable_data(controller)
    # TODO: test generate_ClearDiningTable_data when fixed
    # generate_DoHomework_data(controller)
    # generate_CleanToilets_data(controller)
    # generate_clean_up_kitchen_data(controller)
    # generate_clean_up_living_room_data(controller)
    # generate_clean_up_bedroom_data(controller)
    # generate_clean_up_bathroom_data(controller)

    controller.stop()


class TaskDataRecorder:
    """Class to record the task data."""

    def __init__(  # noqa: PLR0917, PLR0913
        self,
        task_name: str,
        controller: Controller,
        scene_name: str,
        test_task_data_dir: Path | str,
        task: BaseTask | None = None,
        init_advancement: int = 0,
        init_terminated: bool = False,
        reset_args: dict[str, Any] | None = None,
    ):
        """Initialize the task data recorder."""
        self.task_name = task_name
        if reset_args is None:
            reset_args = {}

        controller.reset(scene_name, **reset_args)
        if task is not None:
            task.preprocess_and_reset(controller)
        self.controller = controller
        self.step_number = 0
        self.event_list = [controller.last_event]  # type: ignore
        self.controller_action_list = [controller.last_action]
        self.advancement_list = [init_advancement]
        self.terminated_list = [init_terminated]
        self.error_dict = {}
        self.task_data_dir = Path(test_task_data_dir) / task_name

        # === Type Annotations ===
        self.task_name: str
        self.controller: Controller
        self.step_number: int
        self.event_list: list[Event]
        self.controller_action_list: list[dict[str, Any]]
        self.advancement_list: list[int]
        self.terminated_list: list[bool]
        self.error_dict: dict[str, dict[str, Any]]
        self.task_data_dir: Path

    def record_step(self, action_args: dict[str, Any], advancement: int, terminated: bool = False) -> None:
        """Record the step data."""
        self.step_number += 1
        event = self.controller.step(**action_args)
        self.event_list.append(event)  # type: ignore
        self.controller_action_list.append(self.controller.last_action)
        self.advancement_list.append(advancement)
        self.terminated_list.append(terminated)

        if event.metadata["lastActionSuccess"] is False:
            self.error_dict[f"step_{self.step_number}"] = {
                "error_message": event.metadata["errorMessage"],
                "action_args": action_args,
            }
            print(f"Error in step {self.step_number} of task {self.task_name}: {event.metadata["errorMessage"]}")

    def write_data(self) -> None:
        """Write the recorded data to files."""
        self.task_data_dir.mkdir(parents=True, exist_ok=True)

        event_list_path = self.task_data_dir / "event_list.pkl"
        controller_action_list_path = self.task_data_dir / "controller_action_list.pkl"
        advancement_list_path = self.task_data_dir / "advancement_list.pkl"
        terminated_list_path = self.task_data_dir / "terminated_list.pkl"
        error_dict_path = self.task_data_dir / "error_dict.yaml"

        with (
            event_list_path.open("wb") as f,
            controller_action_list_path.open("wb") as g,
            advancement_list_path.open("wb") as h,
            terminated_list_path.open("wb") as i,
        ):
            pkl.dump(self.event_list, f)
            pkl.dump(self.controller_action_list, g)
            pkl.dump(self.advancement_list, h)
            pkl.dump(self.terminated_list, i)
        if self.error_dict:
            with error_dict_path.open("w") as f:
                yaml.dump(self.error_dict, f)
        elif error_dict_path.exists():
            error_dict_path.unlink()


# TODO: Update when implementing order in task advancement computation
def generate_prepare_meal_data(controller: Controller) -> None:
    """Generate data for the PrepareMeal task."""
    data_recorder = TaskDataRecorder("prepare_meal", controller, "FloorPlan1", test_task_data_dir)

    # === Event 1: Pick up the potato ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Potato|-01.66|+00.93|-02.15",
            "forceAction": True,
        },
        advancement=2,  # potato(isCooked-IsPickedUp) 1 + potato(containedIn:plate) 1
    )

    # === Event 2: Put the potato on the pan ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Pan|+00.72|+00.90|-02.42",
            "forceAction": True,
        },
        advancement=0,
    )

    # === Event 3: Pick up the pan ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Pan|+00.72|+00.90|-02.42",
            "forceAction": True,
        },
        advancement=0,
    )

    # === Event 4: Put the pan on the stove ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "StoveBurner|-00.47|+00.92|-02.37",
            "forceAction": True,
        },
        advancement=0,
    )

    # === Event 5: Pick up the knife to slice the potato ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Knife|-01.70|+00.79|-00.22",
            "forceAction": True,
        },
        advancement=2,  # potato:aux:knife(IsPickedUp) 1 + knife/counterTop(containedIn) 1
    )

    # === Event 6: Slice the potato ===
    data_recorder.record_step(
        action_args={
            "action": "SliceObject",
            "objectId": "Potato|-01.66|+00.93|-02.15",
            "forceAction": True,
        },
        advancement=5,  # potato(isSliced) 2 + knife/counterTop(containedIn) 1 + potato(isCooked) 2 # TODO: Check why it is considered to be on the stove only now -> probably because when a receptacle is moved, the object it contains lose the information that they are contained in it in AI2THOR.. + it sees only the pan to be in the stove and not the potato while the potato sees the stove as one of its parent
    )

    # === Event 7: Put the knife in the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=8,  # potato(isSliced) 2 + knife/counterTop(containedIn) 4 + potato(isCooked) 2
    )

    # === Event 8: Toggle the stove on ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOn",
            "objectId": "StoveKnob|-00.48|+00.88|-02.19",
            "forceAction": True,
        },
        advancement=12,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4
    )

    # === Event 9: Toggle the stove off ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": "StoveKnob|-00.48|+00.88|-02.19",
            "forceAction": True,
        },
        advancement=12,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4
    )

    # === Event 10: Pick up the potato slice ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Potato|-01.66|+00.93|-02.15|PotatoSliced_0",
            "forceAction": True,
        },
        advancement=13,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4 + potato/plate(containedIn) 1
    )

    # === Event 11: Put the potato slice on a plate ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=16,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4 + potato/plate(containedIn) 4
    )

    # === Event 12: Pickup the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=13,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4 + plate/countertop(containedIn) 1 # TODO: Check why the potato slice is not in the plate at this moment -> Probably ai2thor bug
    )

    # === Event 13: Put the plate on the counter ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=20,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4 + potato/plate(containedIn) 4 + plate/countertop(containedIn) 4
    )

    # === Event 14: Pick up the fork ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Fork|+00.95|+00.77|-02.37",
            "forceAction": True,
        },
        advancement=21,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4 + potato/plate(containedIn) 4 + plate/countertop(containedIn) 4 + fork/counterTop(containedIn) 1
    )

    # === Event 15: Put the fork on the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=24,  # potato(isSliced) 2 + potato(isCooked) 6 (containedIn 4 + cooking_source turned on 1 + isCooked 1) + knife/counterTop(containedIn) 4 + potato/plate(containedIn) 4 + plate/countertop(containedIn) 4 + fork/counterTop(containedIn) 4
        terminated=True,
    )

    data_recorder.write_data()


def generate_prepare_going_to_bed_data(controller: Controller) -> None:
    """Generate data for the PrepareGoingToBed task."""
    task = PrepareGoingToBedTask()
    data_recorder = TaskDataRecorder(
        "prepare_going_to_bed",
        controller,
        "FloorPlan301",
        test_task_data_dir,
        task=task,
        reset_args={"gridSize": 0.05},
    )
    # Note: Desk lamps are toggled off

    # === Event 1: Toggle off light switch ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": "LightSwitch|+02.66|+01.28|+01.90",
            "forceAction": True,
        },
        advancement=1,
    )
    # light_switch IsToggled=False) 1/1 (

    # === Event 2: Pick up the book ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Book|-00.90|+00.56|+01.18",
            "forceAction": True,
        },
        advancement=3,
    )
    # light_switch (IsToggled=False) 1/1
    # book (IsPickedUp 1 + IsCloseTo 1) 2/4

    # === Event 3: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 1.5,
            "forceAction": False,
        },
        advancement=3,
    )
    # light_switch (IsToggled=False) 1/1
    # book (IsPickedUp 1 + IsCloseTo 1) 2/4

    # === Event 4: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveLeft",
            "moveMagnitude": 1.45,
            "forceAction": False,
        },
        advancement=3,
    )
    # light_switch (IsToggled=False) 1/1
    # book (IsPickedUp 1 + IsCloseTo 1) 2/4

    # === Event 5: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "RotateLeft",
            "degrees": 5,
            "forceAction": False,
        },
        advancement=3,
    )
    # light_switch (IsToggled=False) 1/1
    # book (IsPickedUp 1 + IsCloseTo 1) 2/4

    # === Event 6: Put down the book ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Bed|-00.64|+00.01|+00.87",
            "forceAction": True,
        },
        advancement=1,
    )
    # light_switch (IsToggled=False) 1/1

    # === Event 7: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 1.2,
            "forceAction": False,
        },
        advancement=1,
    )
    # light_switch (IsToggled=False) 1/1

    # === Event 8: Toggle the desk lamp on ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOn",
            "objectId": "DeskLamp|-01.32|+01.24|-00.99",
            "forceAction": True,
        },
        advancement=2,
    )
    # light_switch (IsToggled=False 1) 1/1
    # desk_lamp (sToggled=True 1) 1/3

    # === Event 9: Pick up the book ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Book|-00.90|+00.56|+01.18",
            "forceAction": True,
        },
        advancement=7,
    )
    # light_switch (IsToggled=False 1) 1/1
    # book (IsPickedUp 1 + IsCloseTo 2) 3/4
    # desk_lamp (IsCloseTo 2 + IsToggled=True 1) 3/3

    # === Event 10: Open the book ===
    data_recorder.record_step(
        action_args={
            "action": "OpenObject",
            "objectId": "Book|-00.90|+00.56|+01.18",
            "forceAction": True,
        },
        advancement=8,
        terminated=True,
    )
    # light_switch (IsToggled=False 1) 1/1
    # book (IsPickedUp 1 + IsCloseTo 2 + IsOpen 1) 4/4
    # desk_lamp (IsCloseTo 2 + IsToggled=True 1) 3/3

    # === Event 11: Switch on the light switch ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOn",
            "objectId": "LightSwitch|+02.66|+01.28|+01.90",
            "forceAction": True,
        },
        advancement=7,
    )
    # book (IsPickedUp 1 + IsCloseTo 2 + IsOpen 1) 4/4
    # desk_lamp (IsCloseTo 2 + IsToggled=True 1) 3/3

    data_recorder.write_data()


# Broken because bowl can't go in mug
def generate_PileUpDishes_data(controller: Controller) -> None:
    """Generate data for the PileUpDishes task."""
    task = PileUpDishes()
    data_recorder = TaskDataRecorder(
        "PileUpDishes",
        controller,
        "FloorPlan1",
        test_task_data_dir,
        task=task,
    )

    # === Event 1: Pick up the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=1,
    )
    # plate/counter_top (plate:is_picked_up 1) 1/4

    # === Event 2: Put the plate on the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=4,
    )
    # plate/counter_top:contained_in 4/4

    # === Event 3: Pick up the bowl ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Bowl|+00.27|+01.10|-00.75",
            "forceAction": True,
        },
        advancement=5,
    )
    # plate/counter_top:contained_in 4/4 + bowl/plate (bowl:is_picked_up 1) 1/4

    # === Event 4: Put the bowl on the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=8,
    )
    # plate/counter_top:contained_in 4/4 + bowl/plate:contained_in 4/4

    # === Event 5: Pick up the spoon ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Spoon|+00.98|+00.77|-02.29",
            "forceAction": True,
        },
        advancement=9,
    )
    # plate/counter_top:contained_in 4/4  + bowl/plate:contained_in 4/4 + spoon/bowl (spoon:is_picked_up 1) 1/4

    # === Event 6: Put the spoon in the bowl ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Bowl|+00.27|+01.10|-00.75",
            "forceAction": True,
        },
        advancement=12,
        terminated=True,
    )
    # plate/counter_top:contained_in 4/4 + bowl/plate:contained_in 4/4 + spoon/bowl:contained_in 4/4

    data_recorder.write_data()


def generate_ArrangeCutleryTask_data(controller: Controller) -> None:
    """Generate data for the ArrangeCutleryTask task."""
    task = ArrangeCutleryTask()
    data_recorder = TaskDataRecorder(
        "ArrangeCutleryTask",
        controller,
        "FloorPlan1",
        test_task_data_dir,
        task=task,
    )

    # === Event 1: Pick up the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=1,
    )
    # plate/counter_top (plate:is_picked_up 1) 1/4

    # === Event 2: Put the plate on the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Sink|-01.90|+00.97|-01.50|SinkBasin",
            "forceAction": True,
        },
        advancement=4,
    )
    # plate/counter_top:contained_in 4/4

    # === Event 3: Pick up the knife ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Knife|-01.70|+00.79|-00.22",
            "forceAction": True,
        },
        advancement=5,
    )
    # plate/counter_top:contained_in 4/4 + knife/plate (knife:is_picked_up 1) 1/4

    # === Event 4: Put the knife on the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=8,
    )
    # plate/counter_top:contained_in 4/4 + knife/plate:contained_in 4/4

    # === Event 5: Pick up the fork ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Fork|+00.95|+00.77|-02.37",
            "forceAction": True,
        },
        advancement=9,
    )
    # plate/counter_top:contained_in 4/4 + knife/plate:contained_in 4/4 + fork/plate (fork:is_picked_up 1) 1/4

    # === Event 6: Put the fork on the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=12,
    )
    # plate/counter_top:contained_in 4/4 + knife/plate:contained_in 4/4 + fork/plate:contained_in 4/4

    # === Event 7: Pick up the spoon ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Spoon|+00.98|+00.77|-02.29",
            "forceAction": True,
        },
        advancement=13,
    )
    # plate/counter_top:contained_in 4/4 + knife/plate:contained_in 4/4 + fork/plate:contained_in 4/4 + spoon/plate (spoon:is_picked_up 1) 1/4

    # === Event 8: Put the spoon on the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=16,
        terminated=True,
    )
    # plate/counter_top:contained_in 4/4 + knife/plate:contained_in 4/4 + fork/plate:contained_in 4/4 + spoon/plate:contained_in 4/4

    # TEMP:

    # === Event 8: Put the spoon on the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=16,
        terminated=True,
    )
    # plate/counter_top:contained_in 4/4 + knife/plate:contained_in 4/4 + fork/plate:contained_in 4/4 + spoon/plate:contained_in 4/4

    data_recorder.write_data()


def generate_WashCutleryTask_data(controller: Controller) -> None:
    """Generate data for the WashCutleryTask task."""
    task = WashCutleryTask()
    data_recorder = TaskDataRecorder(
        "WashCutleryTask",
        controller,
        "FloorPlan1",
        test_task_data_dir,
        task=task,
    )

    # === Event 1: Pick up the knife ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Knife|-01.70|+00.79|-00.22",
            "forceAction": True,
        },
        advancement=1,
    )
    # knife/sink_basin (knife:is_picked_up 1) 1/4

    # === Event 2: Put the knife in the sink basin ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Sink|-01.90|+00.97|-01.50|SinkBasin",
            "forceAction": True,
        },
        advancement=4,
    )
    # knife/sink_basin:contained_in 4/4

    # === Event 3: Pick up the fork ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Fork|+00.95|+00.77|-02.37",
            "forceAction": True,
        },
        advancement=5,
    )
    # knife/sink_basin:contained_in 4/4 + fork/sink_basin (fork:is_picked_up 1) 1/4

    # === Event 4: Put the fork in the sink basin ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Sink|-01.90|+00.97|-01.50|SinkBasin",
            "forceAction": True,
        },
        advancement=8,
    )
    # knife/sink_basin:contained_in 4/4 + fork/sink_basin:contained_in 4/4

    # === Event 5: Pick up the spoon ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Spoon|+00.98|+00.77|-02.29",
            "forceAction": True,
        },
        advancement=9,
    )
    # knife/sink_basin:contained_in 4/4 + fork/sink_basin:contained_in 4/4 + spoon/sink_basin (spoon:is_picked_up 1) 1/4

    # === Event 6: Put the spoon in the sink basin ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Sink|-01.90|+00.97|-01.50|SinkBasin",
            "forceAction": True,
        },
        advancement=12,
    )
    # knife/sink_basin:contained_in 4/4 + fork/sink_basin:contained_in 4/4 + spoon/sink_basin:contained_in 4/4

    # === Event 7: Toggle the faucet on ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOn",
            "objectId": "Faucet|-02.15|+00.91|-01.50",
            "forceAction": True,
        },
        advancement=13,
        terminated=True,
    )
    # knife/sink_basin:contained_in 4/4 + fork/sink_basin:contained_in 4/4 + spoon/sink_basin:contained_in 4/4 + faucet:is_toggled_on 1/1

    data_recorder.write_data()


def generate_ClearDiningTable_data(controller: Controller) -> None:
    """Generate data for the CleanUpKitchen task."""
    dining_table_receptacleObjectIds = [
        "Laptop|-01.70|+00.68|+01.66",
        "CreditCard|-01.94|+00.68|+01.80",
        "Plate|-02.40|+00.68|+01.68",
        "Book|-01.87|+00.68|+01.17",
        "Newspaper|-02.76|+00.68|+01.18",
        "Pencil|-02.83|+00.68|+01.51",
        "Pen|-03.00|+00.69|+01.46",
    ]
    init_advancement = RECEPTACLE_MAX_OBJECTS_PROP_LIMIT - len(dining_table_receptacleObjectIds)
    task = ClearDiningTable()
    data_recorder = TaskDataRecorder(
        "ClearDiningTable",
        controller,
        "FloorPlan201",
        test_task_data_dir,
        task=task,
        init_advancement=init_advancement,
    )

    for object_id in dining_table_receptacleObjectIds:
        terminated = object_id == dining_table_receptacleObjectIds[-1]  # Terminate on last object pickup
        # === Pick up the object ===
        data_recorder.record_step(
            action_args={
                "action": "PickupObject",
                "objectId": object_id,
                "forceAction": True,
            },
            advancement=advancement,
            terminated=terminated,
        )
        # Increment advancement by 1 for each pickup action
        advancement += 1

        # === Drop the object ===
        data_recorder.record_step(
            action_args={
                "action": "DropHandObject",
                "forceAction": True,
            },
            advancement=advancement,
            terminated=terminated,
        )

    data_recorder.write_data()


def generate_DoHomework_data(controller: Controller) -> None:
    """Generate data for the DoHomework task."""
    drawer_id = "Drawer|-01.31|+00.45|-00.58"
    pencil_id = "Pencil|+02.01|+00.81|-01.17"
    laptop_id = "Laptop|-00.50|+00.56|+00.29"
    cellphone_id = "CellPhone|-00.26|+00.56|+00.52"

    task = DoHomeworkTask()
    advancement = 2  # drawer(isOpen=False 1) cellphone/drawer:contained_in (cellphone:isPickedUp=True 1)
    data_recorder = TaskDataRecorder(
        "DoHomework",
        controller,
        "FloorPlan301",
        test_task_data_dir,
        task=task,
        init_advancement=advancement,
    )

    # === Event 1: Toggle off the cellphone ===
    # cellphone (isToggled=False)
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": cellphone_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 2: Open the drawer ===
    # drawer(isOpen=False -1) + cellphone/drawer:contained_in (drawer:isOpen=True 1)
    advancement += 0
    data_recorder.record_step(
        action_args={
            "action": "OpenObject",
            "objectId": drawer_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 3: Put the cellphone in the drawer ===
    # drawer:contained_in +2
    advancement += 2
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": drawer_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 4: Close the drawer ===
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "CloseObject",
            "objectId": drawer_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 5: Toggle off the laptop ===
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": laptop_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 6: Close the laptop ===
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "CloseObject",
            "objectId": laptop_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 7: Pick up the pencil ===
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": pencil_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 8: Rotate left to look at the desk ===
    data_recorder.record_step(
        action_args={
            "action": "RotateLeft",
            "forceAction": True,
            "degrees": 90,
        },
        advancement=advancement,
    )

    # === Event 9: Move forward to have the desk visible ===
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 1,
            "forceAction": True,
        },
        advancement=advancement,
        terminated=True,  # Task ends after the desk is visible
    )

    data_recorder.write_data()


def generate_CleanToilets_data(controller: Controller) -> None:
    """Generate data for the CleanToilets task."""
    toilet_paper_hanger_id = "ToiletPaperHanger|-00.43|+00.78|+02.28"
    toilet_paper_id = "ToiletPaper|-02.45|+01.03|+03.95"
    toilet_id = "Toilet|+00.06|+00.00|+03.10"
    spray_bottle_id = "SprayBottle|-03.09|+00.23|+00.29"
    scrub_brush_id = "ScrubBrush|+00.40|+00.00|+02.69"

    task = CleanToiletsTask()
    advancement = 0
    data_recorder = TaskDataRecorder(
        "CleanToilets",
        controller,
        "FloorPlan401",
        test_task_data_dir,
        task=task,
        init_advancement=advancement,
    )

    # === Event 1: Pick up the toilet paper roll ===
    # toilet_paper_roll (isPickedUp=True)
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": toilet_paper_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 2: Put the toilet paper roll on the hanger ===
    # toilet_paper_hanger (toilet_paper_roll:contained_in)
    advancement += 3
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": toilet_paper_hanger_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 3: Pick up the spray bottle ===
    # spray_bottle (isPickedUp=True)
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": spray_bottle_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 4: Put the spray bottle on the toilet ===
    # toilet (spray_bottle:contained_in)
    advancement += 3
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": toilet_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 5: Open the toilet lid ===
    # toilet (isOpen=True)
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "OpenObject",
            "objectId": toilet_id,
            "forceAction": True,
        },
        advancement=advancement,
    )

    # === Event 6: Pick up the scrub brush ===
    # scrub_brush (isPickedUp=True)
    advancement += 1
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": scrub_brush_id,
            "forceAction": True,
        },
        advancement=advancement,
        terminated=True,  # Task ends after picking up the scrub brush
    )

    data_recorder.write_data()


def generate_clean_up_kitchen_data(controller: Controller) -> None:
    """Generate data for the CleanUpKitchen task."""
    task = CleanUpKitchenTask()
    data_recorder = TaskDataRecorder(
        "clean_up_kitchen",
        controller,
        "FloorPlan1",
        test_task_data_dir,
        task=task,
        reset_args={"gridSize": 0.05},
    )

    # === Event 1: Pick up the apple ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "forceAction": True,
        },
        advancement=1,
    )
    # ..

    # === Event 2: Put the apple in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|-01.94|00.00|+02.03",
            "forceAction": True,
        },
        advancement=4,
    )
    # ..

    # === Event 3: Pick up the tomato ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Tomato|-00.39|+01.14|-00.81",
            "forceAction": True,
        },
        advancement=5,
    )
    # ..

    # === Event 4: Put the tomato in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|-01.94|00.00|+02.03",
            "forceAction": True,
        },
        advancement=8,
    )
    # ..

    # === Event 5: Pick up the potato ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Potato|-01.66|+00.93|-02.15",
            "forceAction": True,
        },
        advancement=9,
    )
    # ..

    # === Event 6: Put the potato in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|-01.94|00.00|+02.03",
            "forceAction": True,
        },
        advancement=12,
    )
    # ..

    # === Event 7: Pick up the egg ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Egg|-02.04|+00.81|+01.24",
            "forceAction": True,
        },
        advancement=13,
    )
    # ..

    # === Event 8: Put the egg in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|-01.94|00.00|+02.03",
            "forceAction": True,
        },
        advancement=16,
        terminated=True,
    )
    # ..

    data_recorder.write_data()


def generate_clean_up_living_room_data(controller: Controller) -> None:
    """Generate data for the CleanUpLivingRoom task."""
    task = CleanUpLivingRoomTask()
    data_recorder = TaskDataRecorder(
        "clean_up_living_room",
        controller,
        "FloorPlan201",
        test_task_data_dir,
        task=task,
        init_advancement=4,
        reset_args={"gridSize": 0.05},
    )

    # === Event 1: Pick up the watch ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Watch|-02.10|+00.73|-00.06",
            "forceAction": True,
        },
        advancement=5,
    )
    # ...

    # === Event 2: Put the watch in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|-03.36|+00.19|+06.43",
            "forceAction": True,
        },
        advancement=7,
    )
    # ...

    # === Event 3: Pick up the credit card ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "CreditCard|-01.94|+00.68|+01.80",
            "forceAction": True,
        },
        advancement=8,
    )
    # ...

    # === Event 4: Put the credit card in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|-03.36|+00.19|+06.43",
            "forceAction": True,
        },
        advancement=10,
    )
    # ...

    # === Event 5: Pick up the key chain ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "KeyChain|-00.27|+00.70|+03.13",
            "forceAction": True,
        },
        advancement=11,
    )
    # ...

    # === Event 6: Put the key chain in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|-03.36|+00.19|+06.43",
            "forceAction": True,
        },
        advancement=13,
    )
    # ...

    # === Event 7: Pick up the remote control ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "RemoteControl|-02.58|+00.74|-00.15",
            "forceAction": True,
        },
        advancement=14,
    )
    # ...

    # === Event 8: Put the remote control in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|-03.36|+00.19|+06.43",
            "forceAction": True,
        },
        advancement=16,
        terminated=True,
    )
    # ...

    data_recorder.write_data()


def generate_clean_up_bedroom_data(controller: Controller) -> None:
    """Generate data for the CleanUpBedroom task."""
    task = CleanUpBedroomTask()
    data_recorder = TaskDataRecorder(
        "clean_up_bedroom",
        controller,
        "FloorPlan301",
        test_task_data_dir,
        task=task,
        init_advancement=5,
        reset_args={"gridSize": 0.05},
    )

    # === Event 1: Pick up the alarm clock ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "KeyChain|+01.74|+00.80|-01.29",
            "forceAction": True,
        },
        advancement=6,
    )
    # alarm_clock (IsPickedUp 1) 1/6

    # === Event 2: Put the alarm clock in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|+02.73|+00.20|+00.89",
            "forceAction": True,
        },
        advancement=8,
    )
    # alarm_clock (IsContainedIn 1) 2/6

    # === Event 3: Pick up the CD ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "CD|+01.62|+00.80|-01.19",
            "forceAction": True,
        },
        advancement=9,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsPickedUp 1) 3/6

    # === Event 4: Put the CD in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|+02.73|+00.20|+00.89",
            "forceAction": True,
        },
        advancement=11,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsContainedIn 1) 4/6

    # === Event 5: Pick up the cell phone ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "CellPhone|-00.26|+00.56|+00.52",
            "forceAction": True,
        },
        advancement=12,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsContainedIn 1) 4/6 + cell_phone (IsPickedUp 1) 5/6

    # === Event 6: Put the cell phone in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|+02.73|+00.20|+00.89",
            "forceAction": True,
        },
        advancement=14,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsContainedIn 1) 4/6 + cell_phone (IsContainedIn 1) 6/6

    # === Event 7: Pick up the pencil ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Pencil|+02.01|+00.81|-01.17",
            "forceAction": True,
        },
        advancement=15,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsContainedIn 1) 4/6 + cell_phone (IsContainedIn 1) 6/6 + pencil (IsPickedUp 1) 7/6

    # === Event 8: Put the pencil in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|+02.73|+00.20|+00.89",
            "forceAction": True,
        },
        advancement=17,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsContainedIn 1) 4/6 + cell_phone (IsContainedIn 1) 6/6 + pencil (IsContainedIn 1) 8/6

    # === Event 9: Pick up the pen ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Pen|+01.93|+00.81|-01.18",
            "forceAction": True,
        },
        advancement=18,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsContainedIn 1) 4/6 + cell_phone (IsContainedIn 1) 6/6 + pencil (IsContainedIn 1) 8/6 + pillow (IsPickedUp 1) 9/6

    # === Event 10: Put the pen in the box ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Box|+02.73|+00.20|+00.89",
            "forceAction": True,
        },
        advancement=20,
        terminated=True,
    )
    # alarm_clock (IsContainedIn 1) 2/6 + CD (IsContainedIn 1) 4/6 + cell_phone (IsContainedIn 1) 6/6 + pencil (IsContainedIn 1) 8/6 + pillow (IsContainedIn 1) 10/6

    data_recorder.write_data()


def generate_clean_up_bathroom_data(controller: Controller) -> None:
    """Generate data for the CleanUpBathroom task."""
    task = CleanUpBathroomTask()
    data_recorder = TaskDataRecorder(
        "clean_up_bathroom",
        controller,
        "FloorPlan401",
        test_task_data_dir,
        task=task,
        reset_args={"gridSize": 0.05},
    )
    # === Event 1: Pick up the cloth ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Cloth|-00.27|+00.04|+01.02",
            "forceAction": True,
        },
        advancement=1,
    )
    # cloth/garbage_can (cloth:is_picked_up 1) 1/4

    # === Event 2: Put the cloth in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|+00.05|00.00|+03.88",
            "forceAction": True,
        },
        advancement=4,
    )
    # cloth/garbage_can:IsContainedIn 4/4

    # === Event 3: Pick up the soap bar ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "SoapBar|-00.69|+00.62|+01.99",
            "forceAction": True,
        },
        advancement=5,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    # === Event 4: Put the soap bar in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|+00.05|00.00|+03.88",
            "forceAction": True,
        },
        advancement=8,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    # === Event 5: Pick up the soap bottle ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "SoapBottle|-03.41|+01.02|+01.29",
            "forceAction": True,
        },
        advancement=9,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    # === Event 6: Put the soap bottle in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|+00.05|00.00|+03.88",
            "forceAction": True,
        },
        advancement=12,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    # === Event 7: Pick up the spray bottle ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "SprayBottle|-03.09|+00.23|+00.29",
            "forceAction": True,
        },
        advancement=13,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    # === Event 8: Put the spray bottle in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|+00.05|00.00|+03.88",
            "forceAction": True,
        },
        advancement=16,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    # === Event 9: Pick up the toilet paper ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "ToiletPaper|-02.45|+01.03|+03.95",
            "forceAction": True,
        },
        advancement=17,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    # === Event 10: Put the toilet paper in the garbage can ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "GarbageCan|+00.05|00.00|+03.88",
            "forceAction": True,
        },
        advancement=20,
        terminated=True,
    )
    # cloth/garbage_can:IsContainedIn 4/4 + ...

    data_recorder.write_data()


def generate_pickup_mug_data(controller: Controller) -> None:
    """Generate data for the Pickup task with a mug."""
    data_recorder = TaskDataRecorder("pickup_mug", controller, "FloorPlan1", test_task_data_dir)

    # === Event 1: Pick up wrong object ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "forceAction": True,
        },
        advancement=0,
    )
    # === Event 2: Drop the wrong object ===
    data_recorder.record_step(
        action_args={
            "action": "DropHandObject",
            "forceAction": True,
        },
        advancement=0,
    )
    # === Event 3: Pick up the mug ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Mug|-01.76|+00.90|-00.62",
            "forceAction": True,
        },
        advancement=1,  # IsPickedUp 1
        terminated=True,
    )
    # === Event 4: Drop the mug ===
    data_recorder.record_step(
        action_args={
            "action": "DropHandObject",
            "forceAction": True,
        },
        advancement=0,
    )
    data_recorder.write_data()


def generate_open_fridge_data(controller: Controller) -> None:
    """Generate data for the Open task with a Fridge."""
    data_recorder = TaskDataRecorder("open_fridge", controller, "FloorPlan1", test_task_data_dir)

    # === Event 1: Open the drawer ===
    data_recorder.record_step(
        action_args={
            "action": "OpenObject",
            "objectId": "Drawer|-01.56|+00.66|-00.20",
            "forceAction": True,
        },
        advancement=0,
    )
    # === Event 2: Close the drawer ===
    data_recorder.record_step(
        action_args={
            "action": "CloseObject",
            "objectId": "Drawer|-01.56|+00.66|-00.20",
            "forceAction": True,
        },
        advancement=0,
    )
    # === Event 3: Open the fridge ===
    data_recorder.record_step(
        action_args={
            "action": "OpenObject",
            "objectId": "Fridge|-02.10|+00.00|+01.07",
            "forceAction": True,
        },
        advancement=1,  # IsOpen 1
        terminated=True,
    )
    # === Event 4: Close the fridge ===
    data_recorder.record_step(
        action_args={
            "action": "CloseObject",
            "objectId": "Fridge|-02.10|+00.00|+01.07",
            "forceAction": True,
        },
        advancement=0,
    )
    data_recorder.write_data()


# TODO: Update when implementing order in task advancement computation
def generate_place_cooled_in_apple_counter_top_data(controller: Controller) -> None:
    """Generate data for the PlaceCooledIn task with an apple on a counter top."""
    data_recorder = TaskDataRecorder(
        task_name="place_cooled_in_apple_counter_top",
        controller=controller,
        scene_name="FloorPlan1",
        test_task_data_dir=test_task_data_dir,
        init_advancement=4,  # apple/plate(containedIn) 4
    )

    # === Event 1: Open the fridge ===
    data_recorder.record_step(
        action_args={
            "action": "OpenObject",
            "objectId": "Fridge|-02.10|+00.00|+01.07",
            "forceAction": True,
        },
        advancement=4,  # apple/plate(containedIn) 4
    )
    # === Event 2: Pick up the apple ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "forceAction": True,
        },
        advancement=2,  # ContainedIn 1 + Temperature(Cold) 1
    )
    # === Event 3: Put the apple in the fridge ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Fridge|-02.10|+00.00|+01.07",
            "forceAction": True,
        },
        advancement=5,  # apple(Temperature=Cold) 5 (1 + 4)
    )
    # === Event 4: Pick up the apple ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "forceAction": True,
        },
        advancement=6,  # apple(Temperature=Cold) 5 + apple/plate(containedIn) 1
    )
    # === Event 5: Put the apple on the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=9,  # apple(Temperature=Cold) 5 + apple/plate(containedIn) 4
        terminated=True,
    )
    # === Event 6: Close the fridge ===
    data_recorder.record_step(
        action_args={
            "action": "CloseObject",
            "objectId": "Fridge|-02.10|+00.00|+01.07",
            "forceAction": True,
        },
        advancement=9,  # apple(Temperature=Cold) 5 + apple/plate(containedIn) 4
        terminated=True,
    )
    # === Event 7: Pick up the apple ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "forceAction": True,
        },
        advancement=6,  # apple(Temperature=Cold) 5 + apple/plate(containedIn) 1
    )
    data_recorder.write_data()


def generate_look_in_light_book_data(controller: Controller) -> None:
    """Generate data for the LookIn task with a desk lamp and a book."""
    data_recorder = TaskDataRecorder(
        task_name="look_in_light_book",
        controller=controller,
        scene_name="FloorPlan301",
        test_task_data_dir=test_task_data_dir,
        init_advancement=1,
        reset_args={"gridSize": 0.1},
    )

    # === Event 1: Toggle the desk lamp off ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": "DeskLamp|-01.32|+01.24|-00.99",
            "forceAction": True,
        },
        advancement=0,
    )
    # === Event 2: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 1.5,
            "forceAction": False,
        },
        advancement=0,
    )
    # === Event 3: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveLeft",
            "moveMagnitude": 1,
            "forceAction": False,
        },
        advancement=0,
    )
    # === Event 4: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "RotateLeft",
            "degrees": 90,
            "forceAction": False,
        },
        advancement=0,
    )
    # === Event 5: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 0.45,
            "forceAction": False,
        },
        advancement=0,
    )
    # === Event 6: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "RotateRight",
            "degrees": 60,
            "forceAction": False,
        },
        advancement=0,
    )
    # === Event 7: Pick up the book ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Book|-00.90|+00.56|+01.18",
            "forceAction": True,
        },
        advancement=2,  # book(IsPickedUp) 1 + book/desk_lamp(IsCloseTo) 1'
    )
    # === Event 8: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 1.2,
            "forceAction": False,
        },
        advancement=5,  # book(IsPickedUp) 1 + 2 book/desk_lamp(IsCloseTo) 2
    )
    # === Event 9: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "RotateRight",
            "degrees": 30,
            "forceAction": False,
        },
        advancement=5,  # book(IsPickedUp) 1 + 2 book/desk_lamp(IsCloseTo) 2
    )
    # === Event 10: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 0.4,
            "forceAction": False,
        },
        advancement=5,  # book(IsPickedUp) 1 + 2 book/desk_lamp(IsCloseTo) 2
    )
    # === Event 11: Toggle the desk lamp on ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOn",
            "objectId": "DeskLamp|-01.32|+01.24|-00.99",
            "forceAction": True,
        },
        advancement=6,  # book(IsPickedUp) 1 + 2 book/desk_lamp(IsCloseTo) 2 + DeskLamp(isToggled) 1
        terminated=True,
    )
    # === Event 12: Toggle the desk lamp off ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": "DeskLamp|-01.32|+01.24|-00.99",
            "forceAction": True,
        },
        advancement=5,  # book(IsPickedUp) 1 + 2 book/desk_lamp(IsCloseTo) 2
    )
    data_recorder.write_data()


if __name__ == "__main__":
    main()

# %%
