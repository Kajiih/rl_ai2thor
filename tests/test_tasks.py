"""Tests for the tasks module."""

import pickle as pkl  # noqa: S403
from pathlib import Path
from typing import Any

import pytest
import yaml
from ai2thor.controller import Controller
from ai2thor.server import Event
from PIL import Image

from rl_thor.envs.sim_objects import SimObjectType
from rl_thor.envs.tasks.tasks import (
    CleanToiletsTask,
    CleanUpBathroomTask,
    CleanUpBedroomTask,
    CleanUpKitchenTask,
    CleanUpLivingRoomTask,
    ClearDiningTable,
    DoHomeworkTask,
    LookInLight,
    Open,
    Pickup,
    PlaceCooledIn,
    PrepareGoingToBedTask,
    PrepareMealTask,
    WashCutleryTask,
)
from rl_thor.envs.tasks.tasks_interface import BaseTask

data_dir = Path(__file__).parent / "data"
test_task_data_dir = data_dir / "test_tasks"


def generate_task_tests_from_saved_data(task: BaseTask, task_data_dir: Path) -> None:  # noqa: PLR0914
    """Generate tests for a task from saved data."""
    event_list_path = task_data_dir / "event_list.pkl"
    controller_action_list_path = task_data_dir / "controller_action_list.pkl"
    advancement_list_path = task_data_dir / "advancement_list.pkl"
    terminated_list_path = task_data_dir / "terminated_list.pkl"
    test_info_path = task_data_dir / "test_info.yaml"
    test_info = {}
    with (
        event_list_path.open("rb") as f,
        advancement_list_path.open("rb") as g,
        terminated_list_path.open("rb") as h,
        controller_action_list_path.open("rb") as i,
    ):
        event_list = pkl.load(f)  # noqa: S301
        controller_action_list = pkl.load(i)  # noqa: S301
        advancement_list = pkl.load(g)  # noqa: S301
        terminated_list = pkl.load(h)  # noqa: S301

    initial_event = event_list[0]
    mock_controller = MockController(last_event=initial_event, last_action=controller_action_list[0])
    # reset_successful, task_advancement, is_terminated, _ = task.preprocess_and_reset(mock_controller)
    reset_successful, task_advancement, is_terminated, _ = task.reset(mock_controller)

    test_info[f"event_0"] = {
        "reset_successful": reset_successful,
        "task_advancement": {
            "expected": advancement_list[0],
            "actual": task_advancement,
        },
        "is_terminated": {
            "expected": terminated_list[0],
            "actual": is_terminated,
        },
    }
    try:
        assert reset_successful
        assert task_advancement == advancement_list[0]
        assert is_terminated == terminated_list[0]
    except AssertionError as e:
        Image.fromarray(initial_event.frame).save(task_data_dir / "last_frame.png")
        with test_info_path.open("w") as f:
            yaml.dump(test_info, f)
        raise e from None

    for i in range(1, len(event_list)):
        event = event_list[i]
        controller_action = controller_action_list[i]
        expected_advancement = advancement_list[i]
        expected_terminated = terminated_list[i]
        task_advancement, is_terminated, _ = task.compute_task_advancement(event, controller_action)

        test_info[f"event_{i}"] = {
            "task_advancement": {
                "expected": expected_advancement,
                "actual": task_advancement,
            },
            "is_terminated": {
                "expected": expected_terminated,
                "actual": is_terminated,
            },
        }
        try:
            assert task_advancement == expected_advancement
            assert is_terminated == expected_terminated
        except AssertionError as e:
            Image.fromarray(event.frame).save(task_data_dir / "last_frame.png")
            with test_info_path.open("w") as f:
                yaml.dump(test_info, f)
            raise e from None

    # === Clean up test info and last frame when all tests pass ===
    if (task_data_dir / "last_frame.png").exists():
        (task_data_dir / "last_frame.png").unlink()
    if test_info_path.exists():
        test_info_path.unlink()


# Mock ai2thor controller
@pytest.fixture
def ai2thor_controller():
    """Create a mock ai2thor controller."""
    controller = Controller()
    yield controller
    controller.stop()


class MockController(Controller):
    """Mock controller for testing, with a last_event attribute."""

    def __init__(self, last_event: Event, last_action: dict[str, Any]):
        """Initialize the MockController."""
        self.last_event = last_event
        self.last_action = last_action


def create_mock_event(object_list: list | None = None) -> Event:
    """Create a mock event for testing."""
    if object_list is None:
        object_list = []
    dummy_metadata = {
        "screenWidth": 300,
        "screenHeight": 300,
        "objects": object_list,
    }
    event = Event(dummy_metadata)

    return event


def test_pickup_task() -> None:
    """Test the Pickup task with a Mug object."""
    task = Pickup(picked_up_object_type=SimObjectType.MUG)
    task_data_dir = test_task_data_dir / "pickup_mug"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_open_task() -> None:
    """Test the Open task with a Fridge object."""
    task = Open(opened_object_type=SimObjectType.FRIDGE)
    task_data_dir = test_task_data_dir / "open_fridge"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_place_cooled_in() -> None:
    """Test the PlaceCooledIn task with a Fridge object."""
    task = PlaceCooledIn(placed_object_type=SimObjectType.APPLE, receptacle_type=SimObjectType.COUNTER_TOP)
    task_data_dir = test_task_data_dir / "place_cooled_in_apple_counter_top"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_look_in_light_book() -> None:
    """Test the LookIn task with a Book object."""
    task = LookInLight(looked_at_object_type=SimObjectType.BOOK)
    task_data_dir = test_task_data_dir / "look_in_light_book"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_prepare_meal() -> None:
    """Test the PrepareMeal task."""
    task = PrepareMealTask()
    task_data_dir = test_task_data_dir / "prepare_meal"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_prepare_going_to_bed() -> None:
    """Test the PrepareGoingToBed task."""
    task = PrepareGoingToBedTask()
    task_data_dir = test_task_data_dir / "prepare_going_to_bed"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_WashCutleryTask() -> None:
    """Test the WashCutleryTask task."""
    task = WashCutleryTask()
    task_data_dir = test_task_data_dir / "WashCutleryTask"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_ClearDiningTable() -> None:
    """Test the ClearDiningTable task."""
    task = ClearDiningTable()
    task_data_dir = test_task_data_dir / "ClearDiningTable"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_DoHomeworkTask() -> None:
    """Test the DoHomework task."""
    task = DoHomeworkTask()
    task_data_dir = test_task_data_dir / "DoHomework"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_CleanToilets() -> None:
    """Test the CleanToilets task."""
    task = CleanToiletsTask()
    task_data_dir = test_task_data_dir / "CleanToilets"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_clean_up_kitchen() -> None:
    """Test the CleanUpBathroom task."""
    task = CleanUpKitchenTask()
    task_data_dir = test_task_data_dir / "clean_up_kitchen"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_clean_up_living_room() -> None:
    """Test the CleanUpBathroom task."""
    task = CleanUpLivingRoomTask()
    task_data_dir = test_task_data_dir / "clean_up_living_room"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_clean_up_bedroom() -> None:
    """Test the CleanUpBathroom task."""
    task = CleanUpBedroomTask()
    task_data_dir = test_task_data_dir / "clean_up_bedroom"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_clean_up_bathroom() -> None:
    """Test the CleanUpBathroom task."""
    task = CleanUpBathroomTask()
    task_data_dir = test_task_data_dir / "clean_up_bathroom"
    generate_task_tests_from_saved_data(task, task_data_dir)
