"""Generate data for the tasks tests."""

# %% === Setup the environment for the test tasks ===
import pickle as pkl  # noqa: S403
from pathlib import Path

from ai2thor.controller import Controller

test_task_data_dir = Path("../data/test_tasks")


def main():
    """Generate data for the tasks tests."""
    controller = Controller()

    # generate_test_pickup_mug_data(controller)
    # generate_test_open_fridge_data(controller)
    # generate_test_place_cooled_in_apple_counter_top_data(controller)
    # generate_test_look_in_light_book_data(controller)

    controller.stop()


def generate_test_pickup_mug_data(controller: Controller) -> None:
    """Generate data for the Pickup task with a mug."""
    test_data_dir = test_task_data_dir / "pickup_mug"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    event_list_path = test_data_dir / "event_list.pkl"
    controller_action_list_path = test_data_dir / "controller_action_list.pkl"
    advancement_list_path = test_data_dir / "advancement_list.pkl"
    terminated_list_path = test_data_dir / "terminated_list.pkl"
    event_list = []
    controller_action_list = []
    advancement_list = []
    terminated_list = []

    # === Event 0 ===
    event = controller.reset("FloorPlan1")
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    # === Event 1 ===
    event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    event = controller.step(action="DropHandObject", forceAction=True)

    # === Event 2 ===
    event = controller.step(action="PickupObject", objectId="Mug|-01.76|+00.90|-00.62", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(1)  # IsPickedUp 1
    terminated_list.append(True)

    # === Event 3 ===
    event = controller.step(action="DropHandObject", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    with (
        event_list_path.open("wb") as f,
        advancement_list_path.open("wb") as g,
        terminated_list_path.open("wb") as h,
        controller_action_list_path.open("wb") as i,
    ):
        pkl.dump(event_list, f)
        pkl.dump(advancement_list, g)
        pkl.dump(terminated_list, h)
        pkl.dump(controller_action_list, i)


def generate_test_open_fridge_data(controller: Controller) -> None:
    """Generate data for the OpenFridge task."""
    test_data_dir = test_task_data_dir / "open_fridge"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    event_list_path = test_data_dir / "event_list.pkl"
    controller_action_list_path = test_data_dir / "controller_action_list.pkl"
    advancement_list_path = test_data_dir / "advancement_list.pkl"
    terminated_list_path = test_data_dir / "terminated_list.pkl"
    event_list = []
    controller_action_list = []
    advancement_list = []
    terminated_list = []

    # === Event 0 ===
    event = controller.reset("FloorPlan1")
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    # === Event 1 ===
    event = controller.step(action="OpenObject", objectId="Drawer|-01.56|+00.66|-00.20", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    event = controller.step(action="CloseObject", objectId="Drawer|-01.56|+00.66|-00.20", forceAction=True)

    # === Event 2 ===
    event = controller.step(action="OpenObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(1)  # IsOpen 1
    terminated_list.append(True)

    # === Event 3 ===
    event = controller.step(action="CloseObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    with (
        event_list_path.open("wb") as f,
        advancement_list_path.open("wb") as g,
        terminated_list_path.open("wb") as h,
        controller_action_list_path.open("wb") as i,
    ):
        pkl.dump(event_list, f)
        pkl.dump(advancement_list, g)
        pkl.dump(terminated_list, h)
        pkl.dump(controller_action_list, i)


# TODO: Update when improving task advancement computation
def generate_test_place_cooled_in_apple_counter_top_data(controller: Controller) -> None:
    """Generate data for the PlaceCooledIn task with an apple and a counter top."""
    test_data_dir = test_task_data_dir / "place_cooled_in_apple_counter_top"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    event_list_path = test_data_dir / "event_list.pkl"
    controller_action_list_path = test_data_dir / "controller_action_list.pkl"
    advancement_list_path = test_data_dir / "advancement_list.pkl"
    terminated_list_path = test_data_dir / "terminated_list.pkl"
    event_list = []
    controller_action_list = []
    advancement_list = []
    terminated_list = []

    # === Event 0 ===
    event = controller.reset("FloorPlan1")
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(3)  # ContainedIn 2 + ReceptacleOf 1
    terminated_list.append(False)

    # === Event 1 ===
    event = controller.step(action="OpenObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(3)  # ContainedIn 2 + ReceptacleOf 1
    terminated_list.append(False)

    # === Event 2 ===
    event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(2)  # ContainedIn 1 + Temperature(Cold) 1
    terminated_list.append(False)

    # === Event 3 ===
    event = controller.step(
        action="PutObject",
        objectId="Fridge|-02.10|+00.00|+01.07",
        forceAction=True,
    )
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(2)  # Temperature(Cold) 2
    terminated_list.append(False)

    # === Event 4 ===
    event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(3)  # ContainedIn 1 + Temperature(Cold) 2
    terminated_list.append(False)

    # === Event 5 ===
    event = controller.step(
        action="PutObject",
        objectId="CounterTop|+00.69|+00.95|-02.48",
        forceAction=True,
    )
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(5)  # ContainedIn 2 + ReceptacleOf 1 + Temperature(Cold) 2
    terminated_list.append(True)

    # === Event 6 ===
    event = controller.step(action="CloseObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(5)  # ContainedIn 2 + ReceptacleOf 1 + Temperature(Cold) 2
    terminated_list.append(True)

    # === Event 7 ===
    event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(3)  # ContainedIn 1 + Temperature(Cold) 2
    terminated_list.append(False)

    with (
        event_list_path.open("wb") as f,
        advancement_list_path.open("wb") as g,
        terminated_list_path.open("wb") as h,
        controller_action_list_path.open("wb") as i,
    ):
        pkl.dump(event_list, f)
        pkl.dump(advancement_list, g)
        pkl.dump(terminated_list, h)
        pkl.dump(controller_action_list, i)


def generate_test_look_in_light_book_data(controller: Controller) -> None:
    """Generate data for the LookIn task with a light and a book."""
    test_data_dir = test_task_data_dir / "look_in_light_book"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    event_list_path = test_data_dir / "event_list.pkl"
    controller_action_list_path = test_data_dir / "controller_action_list.pkl"
    advancement_list_path = test_data_dir / "advancement_list.pkl"
    terminated_list_path = test_data_dir / "terminated_list.pkl"
    event_list = []
    controller_action_list = []
    advancement_list = []
    terminated_list = []

    event = controller.reset("FloorPlan301", gridSize=0.1)

    # === Event 0 ===
    event = controller.step(action="ToggleObjectOff", objectId="DeskLamp|-01.32|+01.24|-00.99", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    event = controller.step(
        action="MoveAhead",
        moveMagnitude=1.5,
        forceAction=False,
    )

    event = controller.step(
        action="MoveLeft",
        moveMagnitude=1,
        forceAction=False,
    )

    event = controller.step(
        action="RotateLeft",
        degrees=90,
        forceAction=False,
    )

    event = controller.step(
        action="MoveAhead",
        moveMagnitude=0.45,
        forceAction=False,
    )

    # === Event 1 ===
    event = controller.step(
        action="RotateRight",
        degrees=60,
        forceAction=False,
    )
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(0)
    terminated_list.append(False)

    # === Event 2 ===
    event = controller.step(action="PickupObject", objectId="Book|-00.90|+00.56|+01.18", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(1)  # IsPickedUp(Book) 1
    terminated_list.append(False)

    # === Event 3 ===
    event = controller.step(
        action="MoveAhead",
        moveMagnitude=1.2,
        forceAction=False,
    )
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(3)  # IsPickedUp(Book, True) 1 + 2 IsCloseTo 1
    terminated_list.append(False)

    event = controller.step(
        action="RotateRight",
        degrees=30,
        forceAction=False,
    )

    # === Event 4 ===
    event = controller.step(
        action="MoveAhead",
        moveMagnitude=0.4,
        forceAction=False,
    )
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(3)  # IsPickedUp(Book, True) 1 + 2 IsCloseTo 1
    terminated_list.append(False)

    # === Event 5 ===
    event = controller.step(action="ToggleObjectOn", objectId="DeskLamp|-01.32|+01.24|-00.99", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(4)  # isPickedUp(Book, True) 1 + 2 IsCloseTo 1 + IsToggled(DeskLamp, True) 1
    terminated_list.append(True)

    # === Event 6 ===
    event = controller.step(action="ToggleObjectOff", objectId="DeskLamp|-01.32|+01.24|-00.99", forceAction=True)
    event_list.append(event)
    controller_action_list.append(controller.last_action)
    advancement_list.append(3)  # isPickedUp(Book, True) 1 + 2 IsCloseTo 1
    terminated_list.append(False)

    with (
        event_list_path.open("wb") as f,
        advancement_list_path.open("wb") as g,
        terminated_list_path.open("wb") as h,
        controller_action_list_path.open("wb") as i,
    ):
        pkl.dump(event_list, f)
        pkl.dump(advancement_list, g)
        pkl.dump(terminated_list, h)
        pkl.dump(controller_action_list, i)


if __name__ == "__main__":
    main()

# %%
