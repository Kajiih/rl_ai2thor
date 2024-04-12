"""Tests for the tasks module."""

import pickle as pkl  # noqa: S403
from pathlib import Path

import pytest
from ai2thor.controller import Controller
from ai2thor.server import Event
from PIL import Image

from rl_ai2thor.envs.sim_objects import SimObjectType
from rl_ai2thor.envs.tasks.tasks import BaseTask, LookInLight, Open, Pickup, PlaceCooledIn

data_dir = Path(__file__).parent / "data"


def generate_task_tests_from_saved_data(task: BaseTask, task_data_dir: Path) -> None:
    """Generate tests for a task from saved data."""
    event_list_path = task_data_dir / "event_list.pkl"
    advancement_list_path = task_data_dir / "advancement_list.pkl"
    terminated_list_path = task_data_dir / "terminated_list.pkl"
    with event_list_path.open("rb") as f, advancement_list_path.open("rb") as g, terminated_list_path.open("rb") as h:
        event_list = pkl.load(f)  # noqa: S301
        advancement_list = pkl.load(g)  # noqa: S301
        terminated_list = pkl.load(h)  # noqa: S301

    mock_controller = MockController(last_event=event_list[0])
    reset_successful, task_advancement, is_terminated, _ = task.reset(mock_controller)
    assert reset_successful
    assert task_advancement == advancement_list[0]
    assert is_terminated == terminated_list[0]

    for i in range(1, len(event_list)):
        event = event_list[i]
        expected_advancement = advancement_list[i]
        expected_terminated = terminated_list[i]
        task_advancement, is_terminated, _ = task.compute_task_advancement(event)
        try:
            assert task_advancement == expected_advancement
            assert is_terminated == expected_terminated
        except AssertionError as e:
            Image.fromarray(event.frame).save("tests/last_frame.png")
            raise e from None


# Mock ai2thor controller
@pytest.fixture()
def ai2thor_controller():
    """Create a mock ai2thor controller."""
    controller = Controller()
    yield controller
    controller.stop()


class MockController(Controller):
    """Mock controller for testing, with a last_event attribute."""

    def __init__(self, last_event: Event):
        """Initialize the MockController."""
        self.last_event = last_event


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
    task_data_dir = data_dir / "test_pickup_mug"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_open_task() -> None:
    """Test the Open task with a Fridge object."""
    task = Open(opened_object_type=SimObjectType.FRIDGE)
    task_data_dir = data_dir / "test_open_fridge"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_place_cooled_in() -> None:
    """Test the PlaceCooledIn task with a Fridge object."""
    task = PlaceCooledIn(placed_object_type=SimObjectType.APPLE, receptacle_type=SimObjectType.COUNTER_TOP)
    task_data_dir = data_dir / "test_place_cooled_in_apple_counter_top"
    generate_task_tests_from_saved_data(task, task_data_dir)


def test_look_in_light_book() -> None:
    """Test the LookIn task with a Book object."""
    task = LookInLight(looked_at_object_type=SimObjectType.BOOK)
    task_data_dir = data_dir / "test_look_in_light_book"
    generate_task_tests_from_saved_data(task, task_data_dir)
