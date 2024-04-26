"""Generate data for the tasks tests."""

# %% === Setup the environment for the test tasks ===
import pickle as pkl  # noqa: S403
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ai2thor.controller import Controller

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

    controller.stop()


class TaskDataRecorder:
    """Class to record the task data."""

    def __init__(  # noqa: PLR0917, PLR0913
        self,
        task_name: str,
        controller: Controller,
        scene_name: str,
        test_task_data_dir: Path | str,
        init_advancement: int = 0,
        init_terminated: bool = False,
        reset_args: dict[str, Any] | None = None,
    ):
        """Initialize the task data recorder."""
        self.task_name = task_name
        if reset_args is None:
            reset_args = {}
        self.controller = controller
        self.step_number = 0
        event = controller.reset(scene_name, **reset_args)
        self.event_list = [event]  # type: ignore
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


# TODO: Update when improving task advancement computation and aux properties/items
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

    # === Event 2: Put the egg on the pan ===
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
        advancement=3,  # potato(isSliced) 2 + knife/counterTop(containedIn) 1
    )

    # === Event 7: Put the knife in the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=5,  # potato(isSliced) 2 + knife/counterTop(containedIn) 3
    )

    # === Event 8: Toggle the stove on ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOn",
            "objectId": "StoveKnob|-00.48|+00.88|-02.19",
            "forceAction": True,
        },
        advancement=8,  # potato(isSliced) 2 + potato(isCooked) 3 + knife/counterTop(containedIn) 3
    )

    # === Event 9: Toggle the stove off ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": "StoveKnob|-00.48|+00.88|-02.19",
            "forceAction": True,
        },
        advancement=8,  # potato(isSliced) 2 + potato(isCooked) 3 + potato/plate(containedIn) 1 + knife/counterTop(containedIn) 3
    )

    # === Event 10: Pick up the potato slice ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Potato|-01.66|+00.93|-02.15|PotatoSliced_0",
            "forceAction": True,
        },
        advancement=9,  # potato(isSliced) 2 + potato(isCooked) 3 + potato/plate(containedIn) 1 + knife/counterTop(containedIn) 3
    )

    # === Event 11: Put the potato slice on a plate ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=11,  # potato(isSliced) 2 + potato(isCooked) 3 + potato/plate(containedIn) 3 + knife/counterTop(containedIn) 3
    )

    # === Event 12: Pickup the plate ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Plate|+00.96|+01.65|-02.61",
            "forceAction": True,
        },
        advancement=9,  # potato(isSliced) 2 + potato(isCooked) 3 + potato/plate(containedIn) 3 + knife/counterTop(containedIn) 3 + plate/countertop(containedIn) 1 # TODO: Check why the potato slice is not in the plate at this moment
    )

    # === Event 13: Put the plate on the counter ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=14,  # potato(isSliced) 2 + potato(isCooked) 3 + potato/plate(containedIn) 3 + knife/counterTop(containedIn) 3 + plate/countertop(containedIn) 3
    )

    # === Event 14: Pick up the fork ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Fork|+00.95|+00.77|-02.37",
            "forceAction": True,
        },
        advancement=15,  # potato(isSliced) 2 + potato(isCooked) 3 + potato/plate(containedIn) 3 + knife/counterTop(containedIn) 3 + plate/countertop(containedIn) 3 + fork/counterTop(containedIn) 1
    )

    # === Event 15: Put the fork on the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=17,  # potato(isSliced) 2 + potato(isCooked) 3 + potato/plate(containedIn) 3 + knife/counterTop(containedIn) 3 + plate/countertop(containedIn) 3 + fork/counterTop(containedIn) 3
        terminated=True,
    )

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


# TODO: Update when improving task advancement computation
def generate_place_cooled_in_apple_counter_top_data(controller: Controller) -> None:
    """Generate data for the PlaceCooledIn task with an apple on a counter top."""
    data_recorder = TaskDataRecorder(
        task_name="place_cooled_in_apple_counter_top",
        controller=controller,
        scene_name="FloorPlan1",
        test_task_data_dir=test_task_data_dir,
        init_advancement=3,
    )

    # === Event 1: Open the fridge ===
    data_recorder.record_step(
        action_args={
            "action": "OpenObject",
            "objectId": "Fridge|-02.10|+00.00|+01.07",
            "forceAction": True,
        },
        advancement=3,
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
        advancement=2,  # Temperature(Cold) 2
    )
    # === Event 4: Pick up the apple ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "forceAction": True,
        },
        advancement=3,  # ContainedIn 1 + Temperature(Cold) 2
    )
    # === Event 5: Put the apple on the counter top ===
    data_recorder.record_step(
        action_args={
            "action": "PutObject",
            "objectId": "CounterTop|+00.69|+00.95|-02.48",
            "forceAction": True,
        },
        advancement=5,  # ContainedIn 2 + ReceptacleOf 1 + Temperature(Cold) 2
        terminated=True,
    )
    # === Event 6: Close the fridge ===
    data_recorder.record_step(
        action_args={
            "action": "CloseObject",
            "objectId": "Fridge|-02.10|+00.00|+01.07",
            "forceAction": True,
        },
        advancement=5,  # ContainedIn 2 + ReceptacleOf 1 + Temperature(Cold) 2
        terminated=True,
    )
    # === Event 7: Pick up the apple ===
    data_recorder.record_step(
        action_args={
            "action": "PickupObject",
            "objectId": "Apple|-00.47|+01.15|+00.48",
            "forceAction": True,
        },
        advancement=3,  # ContainedIn 1 + Temperature(Cold) 2
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
        advancement=1,  # IsPickedUp(Book) 1
    )
    # === Event 8: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 1.2,
            "forceAction": False,
        },
        advancement=3,  # IsPickedUp(Book, True) 1 + 2 IsCloseTo 1
    )
    # === Event 9: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "RotateRight",
            "degrees": 30,
            "forceAction": False,
        },
        advancement=3,  # IsPickedUp(Book, True) 1 + 2 IsCloseTo 1
    )
    # === Event 10: Move to the desk lamp ===
    data_recorder.record_step(
        action_args={
            "action": "MoveAhead",
            "moveMagnitude": 0.4,
            "forceAction": False,
        },
        advancement=3,  # IsPickedUp(Book, True) 1 + 2 IsCloseTo 1
    )
    # === Event 11: Toggle the desk lamp on ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOn",
            "objectId": "DeskLamp|-01.32|+01.24|-00.99",
            "forceAction": True,
        },
        advancement=4,  # isPickedUp(Book, True) 1 + 2 IsCloseTo 1 + IsToggled(DeskLamp, True) 1
        terminated=True,
    )
    # === Event 12: Toggle the desk lamp off ===
    data_recorder.record_step(
        action_args={
            "action": "ToggleObjectOff",
            "objectId": "DeskLamp|-01.32|+01.24|-00.99",
            "forceAction": True,
        },
        advancement=3,  # isPickedUp(Book, True) 1 + 2 IsCloseTo 1
    )
    data_recorder.write_data()


if __name__ == "__main__":
    main()

# %%
