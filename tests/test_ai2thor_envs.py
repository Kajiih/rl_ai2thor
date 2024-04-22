"""Tests for the ai2thor_envs module."""

import pickle as pkl  # noqa: S403
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import call, patch

import gymnasium as gym
import pytest
import yaml
from PIL import Image

from rl_ai2thor.envs.actions import EnvActionName, UnknownActionCategoryError
from rl_ai2thor.envs.ai2thor_envs import (
    ITHOREnv,
)
from rl_ai2thor.envs.sim_objects import SimObjectType
from rl_ai2thor.envs.tasks.tasks import PlaceIn, PlaceNSameIn, UnknownTaskTypeError
from rl_ai2thor.envs.tasks.tasks_interface import NoTaskBlueprintError

if TYPE_CHECKING:
    from numpy.typing import NDArray

# %% === Constants ===
abs_tolerance = 4
rel_tolerance = 5e-2

seed = 42

test_media_path = Path("tests/media")
test_media_path.mkdir(exist_ok=True)


# %% === Fixtures ===
# TODO: Change fixture to have a specific base config
@pytest.fixture()
def ithor_env():
    env = ITHOREnv()
    yield env
    env.close()


@pytest.fixture()
def ithor_env_2():
    env = ITHOREnv()
    yield env
    env.close()


# %% === Init tests ===
def test__load_and_override_config():
    base_config = {"environment_mode": "default", "key0": "value0", "key1": "value1", "key2": "value2"}
    env_mode_config = {"key2": "new_value2", "key3": "new_value3"}
    override_config = {"key1": "overridden_value1", "key2": "overridden_value2"}

    # Mock the 'read_text' method to return the base and environment mode configs
    with (
        patch(
            "pathlib.Path.read_text",
            side_effect=[
                yaml.dump(base_config),
                yaml.dump(env_mode_config),
            ],
        ) as mock_read_text,
        patch("pathlib.Path.is_file", return_value=True),
    ):
        # Call the _load_and_override_config method with the override config
        config = ITHOREnv._load_and_override_config("config", override_config)

        # Assert the expected configuration values
        assert config == {
            "environment_mode": "default",
            "key0": "value0",
            "key1": "overridden_value1",
            "key2": "overridden_value2",
            "key3": "new_value3",
        }

    # Check that the 'read_text' method was called with the expected arguments
    expected_calls = [
        call(encoding="utf-8"),
        call(encoding="utf-8"),
    ]
    mock_read_text.assert_has_calls(expected_calls, any_order=False)


partial_config = {
    "action_categories": {
        "movement_actions": True,
        "body_rotation_actions": True,
        "camera_rotation_actions": True,
        "crouch_actions": False,
        "open_close_actions": True,
        "pickup_put_actions": True,
        "toggle_actions": True,
    },
    "use_done_action": False,
    "partial_openness": True,
    "discrete_actions": False,
    "simple_movement_actions": True,
    "target_closest_object": False,
}


def test__compute_action_availabilities():
    expected_availabilities = {
        EnvActionName.MOVE_AHEAD: True,
        EnvActionName.MOVE_BACK: False,
        EnvActionName.MOVE_LEFT: False,
        EnvActionName.MOVE_RIGHT: False,
        EnvActionName.LOOK_UP: True,
        EnvActionName.LOOK_DOWN: True,
        EnvActionName.ROTATE_LEFT: True,
        EnvActionName.ROTATE_RIGHT: True,
        EnvActionName.OPEN_OBJECT: False,
        EnvActionName.CLOSE_OBJECT: False,
        EnvActionName.PARTIAL_OPEN_OBJECT: True,
        EnvActionName.PICKUP_OBJECT: True,
        EnvActionName.PUT_OBJECT: True,
        EnvActionName.TOGGLE_OBJECT_ON: True,
        EnvActionName.TOGGLE_OBJECT_OFF: True,
        EnvActionName.DONE: False,
        EnvActionName.CROUCH: False,
        EnvActionName.STAND: False,
        EnvActionName.MOVE_HELD_OBJECT_AHEAD_BACK: False,
        EnvActionName.MOVE_HELD_OBJECT_RIGHT_LEFT: False,
        EnvActionName.MOVE_HELD_OBJECT_UP_DOWN: False,
        EnvActionName.ROTATE_HELD_OBJECT_ROLL: False,
        EnvActionName.ROTATE_HELD_OBJECT_PITCH: False,
        EnvActionName.ROTATE_HELD_OBJECT_YAW: False,
        EnvActionName.DROP_HAND_OBJECT: False,
        EnvActionName.THROW_OBJECT: False,
        EnvActionName.PUSH_OBJECT: False,
        EnvActionName.PULL_OBJECT: False,
        EnvActionName.FILL_OBJECT_WITH_LIQUID: False,
        EnvActionName.EMPTY_LIQUID_FROM_OBJECT: False,
        EnvActionName.BREAK_OBJECT: False,
        EnvActionName.SLICE_OBJECT: False,
        EnvActionName.USE_UP_OBJECT: False,
        EnvActionName.CLEAN_OBJECT: False,
        EnvActionName.DIRTY_OBJECT: False,
    }

    action_availabilities = ITHOREnv._compute_action_availabilities(partial_config)

    assert action_availabilities == expected_availabilities


def test_compute_action_availabilities_unknown_action_category():
    config = {
        "action_categories": {
            "_unknown_action_category": None,
        }
    }

    with pytest.raises(UnknownActionCategoryError) as exc_info:
        ITHOREnv._compute_action_availabilities(config)

    assert exc_info.value.action_category == "_unknown_action_category"


def test__action_space(ithor_env: ITHOREnv):
    ithor_env.config = partial_config

    ithor_env._create_action_space()

    assert isinstance(ithor_env.action_space, gym.spaces.Dict)
    assert "action_index" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_index"], gym.spaces.Discrete)
    assert "action_parameter" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_parameter"], gym.spaces.Box)
    assert "target_object_coordinates" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["target_object_coordinates"], gym.spaces.Box)


def test__create_observation_space(ithor_env: ITHOREnv):
    ithor_env.config = {
        "controller_parameters": {
            "height": 84,
            "width": 44,
        },
        "grayscale": False,
    }

    ithor_env._create_observation_space()

    assert isinstance(ithor_env.observation_space, gym.spaces.Dict)
    env_observation = ithor_env.observation_space.spaces["env_obs"]
    task_observation = ithor_env.observation_space.spaces["task_obs"]

    assert isinstance(env_observation, gym.spaces.Box)
    assert env_observation.shape == (84, 44, 3)
    assert isinstance(task_observation, gym.spaces.Text)
    # TODO: Need to change this when the task observation can change


def test__create_observation_space_grayscale(ithor_env: ITHOREnv):
    ithor_env.config = {
        "controller_parameters": {
            "height": 84,
            "width": 44,
        },
        "grayscale": True,
    }

    ithor_env._create_observation_space()

    assert isinstance(ithor_env.observation_space, gym.spaces.Dict)
    env_observation = ithor_env.observation_space.spaces["env_obs"]

    assert isinstance(env_observation, gym.spaces.Box)
    assert env_observation.shape == (84, 44, 1)


def test__compute_available_scenes():
    scenes = ["Kitchen", "FloorPlan201", "FloorPlan301"]
    excluded_scenes = {"FloorPlan1", "FloorPlan301", "FloorPlan401"}

    expected_available_scenes = {
        "FloorPlan10",
        "FloorPlan11",
        "FloorPlan12",
        "FloorPlan13",
        "FloorPlan14",
        "FloorPlan15",
        "FloorPlan16",
        "FloorPlan17",
        "FloorPlan18",
        "FloorPlan19",
        "FloorPlan2",
        "FloorPlan20",
        "FloorPlan201",
        "FloorPlan21",
        "FloorPlan22",
        "FloorPlan23",
        "FloorPlan24",
        "FloorPlan25",
        "FloorPlan26",
        "FloorPlan27",
        "FloorPlan28",
        "FloorPlan29",
        "FloorPlan3",
        "FloorPlan30",
        "FloorPlan4",
        "FloorPlan5",
        "FloorPlan6",
        "FloorPlan7",
        "FloorPlan8",
        "FloorPlan9",
    }

    available_scenes = ITHOREnv._compute_config_available_scenes(scenes, excluded_scenes)

    assert available_scenes == expected_available_scenes


def test__create_task_blueprints():
    config = {
        "globally_excluded_scenes": ["FloorPlan1"],
        "tasks": [
            {
                "type": "PlaceIn",
                "args": {
                    "placed_object_type": "Knife",
                    "receptacle_type": "Sink",
                },
                "scenes": ["FloorPlan1", "FloorPlan2"],
            },
            {
                "type": "PlaceNSameIn",
                "args": {"placed_object_type": "Apple", "receptacle_type": "Plate", "n": 2},
                "scenes": "FloorPlan3",
            },
        ],
    }

    task_blueprints = ITHOREnv._create_task_blueprints(config)

    assert len(task_blueprints) == len(config["tasks"])

    # Check task blueprint 1
    task_blueprint_1 = task_blueprints[0]
    assert task_blueprint_1.task_type == PlaceIn
    assert task_blueprint_1.scenes == {"FloorPlan2"}
    assert task_blueprint_1.task_args == {
        "placed_object_type": SimObjectType("Knife"),
        "receptacle_type": SimObjectType("Sink"),
    }

    # Check task blueprint 2
    task_blueprint_2 = task_blueprints[1]
    assert task_blueprint_2.task_type == PlaceNSameIn
    assert task_blueprint_2.scenes == {"FloorPlan3"}
    assert task_blueprint_2.task_args == {
        "placed_object_type": SimObjectType("Apple"),
        "receptacle_type": SimObjectType("Plate"),
        "n": 2,
    }


def test__create_task_blueprints_unknown_task():
    config = {
        "globally_excluded_scenes": [],
        "tasks": [
            {
                "type": "_unknown_task",
                "args": {},
                "scenes": [],
            },
        ],
    }

    with pytest.raises(UnknownTaskTypeError) as exc_info:
        ITHOREnv._create_task_blueprints(config)

    assert exc_info.value.task_type == "_unknown_task"


# More with empty task config
def test__create_task_blueprints_empty_task_config():
    config = {
        "globally_excluded_scenes": [],
        "tasks": [],
    }

    with pytest.raises(NoTaskBlueprintError) as exc_info:
        ITHOREnv._create_task_blueprints(config)

    assert exc_info.value.config == {"globally_excluded_scenes": [], "tasks": []}


# %% === Reproducibility tests ===
@pytest.mark.xfail(reason="Rendering in ai2thor is not deterministic")
def test_reset_exact_observation_reproducibility(ithor_env: ITHOREnv):
    obs1, info1 = ithor_env.reset(seed=seed)
    obs2, info2 = ithor_env.reset(seed=seed)

    assert obs1 == pytest.approx(obs2, abs=abs_tolerance, rel=rel_tolerance)
    assert info1 == info2


# This test fails sometimes because AI2-THOR is not deterministic
# ! Sometimes 'Pen' and 'Pencil' are switched...?
def test_reset_same_runtime_reproducible(ithor_env: ITHOREnv, ithor_env_2: ITHOREnv):  # noqa: PLR0914
    media_path = test_media_path / "reset_same_runtime_reproducible"
    media_path.mkdir(exist_ok=True)

    obs1, info1 = ithor_env.reset(seed=seed)
    env_obs1: NDArray = obs1["env_obs"]  # type: ignore
    task_obs1 = obs1["task_obs"]
    obs2, info2 = ithor_env_2.reset(seed=seed)
    env_obs2: NDArray = obs2["env_obs"]  # type: ignore
    task_obs2 = obs2["task_obs"]
    assert ithor_env.current_task_type == ithor_env_2.current_task_type
    assert task_obs1 == task_obs2

    obs1_2, info1_2 = ithor_env.reset(seed=seed)
    env_obs1_2: NDArray = obs1_2["env_obs"]  # type: ignore
    task_obs1_2 = obs1_2["task_obs"]
    obs2_2, info2_2 = ithor_env_2.reset(seed=seed)
    env_obs2_2: NDArray = obs2_2["env_obs"]  # type: ignore
    task_obs2_2 = obs2_2["task_obs"]
    assert ithor_env.current_task_type == ithor_env_2.current_task_type
    assert task_obs1_2 == task_obs2_2

    # Check if the scene are identical
    split_assert_dicts(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)
    split_assert_dicts(info1_2["metadata"], info2_2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)

    # Check if the observations are identical
    try:
        assert env_obs1 == pytest.approx(env_obs2, abs=rel_tolerance * 255, rel=rel_tolerance)
    except AssertionError:
        Image.fromarray(env_obs1).save(media_path / "obs1.png")
        Image.fromarray(env_obs2).save(media_path / "obs2.png")
        Image.fromarray(env_obs1 - env_obs2).save(media_path / "diff.png")
        assert env_obs1 == pytest.approx(env_obs2, abs=rel_tolerance * 255, rel=rel_tolerance)

    try:
        assert env_obs1_2 == pytest.approx(env_obs2_2, abs=rel_tolerance * 255, rel=rel_tolerance)
    except AssertionError:
        Image.fromarray(env_obs1_2).save(media_path / "obs1_2.png")
        Image.fromarray(env_obs1_2).save(media_path / "obs2_2.png")
        Image.fromarray(env_obs1_2 - env_obs2_2).save(media_path / "diff_2.png")
        assert env_obs1_2 == pytest.approx(env_obs2_2, abs=rel_tolerance * 255, rel=rel_tolerance)

    assert are_close_dict(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)
    assert are_close_dict(info1_2["metadata"], info2_2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)


def test_reset_different_runtime_reproducible(ithor_env: ITHOREnv):
    media_path = test_media_path / "reset_different_runtime_reproducible"
    media_path.mkdir(exist_ok=True)

    obs1, info1 = ithor_env.reset(seed=seed)
    task_type = ithor_env.current_task_type
    data_path = Path("tests/data/test_reset_different_runtime_reproducible_obs_info.pkl")

    # Run the following only once to save the data
    # to_serialize_data = (obs1, info1, task_type)
    # with data_path.open("wb") as f:
    #     pkl.dump(to_serialize_data, f)

    with data_path.open("rb") as f:
        obs2, info2, task_type2 = pkl.load(f)  # noqa: S301

    assert task_type == task_type2

    # Check if the scene are identical
    split_assert_dicts(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)

    # Check if the observations are identical
    env_obs1: NDArray = obs1["env_obs"]  # type: ignore
    print(f"{env_obs1.shape = }")
    print(f"{env_obs1.shape = }")
    print(f"{env_obs1.shape = }")
    print(f"{env_obs1.shape = }")
    task_obs1 = obs1["task_obs"]
    env_obs2: NDArray = obs2["env_obs"]  # type: ignore
    task_obs2 = obs2["task_obs"]
    assert task_obs1 == task_obs2
    try:
        assert env_obs1 == pytest.approx(env_obs2, abs=rel_tolerance * 255, rel=rel_tolerance * 255)
    except AssertionError:
        Image.fromarray(env_obs1).save(media_path / "obs1.png")
        Image.fromarray(env_obs2).save(media_path / "obs2.png")
        Image.fromarray(env_obs1 - env_obs2).save(media_path / "diff.png")
        assert env_obs1 == pytest.approx(env_obs2, abs=rel_tolerance * 255, rel=rel_tolerance * 255)

    assert are_close_dict(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)


def test_reset_not_same_scene(ithor_env: ITHOREnv):
    _, info1 = ithor_env.reset(seed=seed)
    _, info2 = ithor_env.reset(seed=seed + 1)

    assert not are_close_dict(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)


# %% === Utils ===
def split_assert_dicts(d1, d2, abs_tol=None, rel_tol=None, nan_ok=False):
    keys_or_indices = range(len(d1)) if isinstance(d1, list) else d1.keys()
    for k in keys_or_indices:
        if isinstance(d1[k], Mapping | list):
            split_assert_dicts(d1[k], d2[k], abs_tol=abs_tol, rel_tol=rel_tol, nan_ok=nan_ok)
        elif isinstance(d1[k], float | int) and not isinstance(d1[k], bool):
            maxi, mini = max(d1[k], d2[k]), min(d1[k], d2[k])
            new_abs_tol = max(abs_tol, rel_tol * 360) if abs_tol is not None and rel_tol is not None else None
            # Handle the degrees case
            if mini != pytest.approx(maxi - 360, abs=new_abs_tol, rel=rel_tol, nan_ok=nan_ok):
                assert d1[k] == pytest.approx(d2[k], abs=abs_tol, rel=rel_tol, nan_ok=nan_ok)
        elif k != "isMoving":  # Special case for isMoving
            try:
                assert d1[k] == pytest.approx(d2[k], abs=abs_tol, rel=rel_tol, nan_ok=nan_ok)
            except AssertionError:
                print(f"Key: {k}, d1: {d1[k]}, d2: {d2[k]}")


def are_close_dict(d1, d2, abs_tol=None, rel_tol=None, nan_ok=False):
    keys_or_indices = range(len(d1)) if isinstance(d1, list) else d1.keys()
    for k in keys_or_indices:
        if isinstance(d1[k], Mapping | list):
            if not are_close_dict(d1[k], d2[k], abs_tol=abs_tol, rel_tol=rel_tol, nan_ok=nan_ok):
                return False
        elif isinstance(d1[k], float | int) and not isinstance(d1[k], bool):
            maxi, mini = max(d1[k], d2[k]), min(d1[k], d2[k])
            new_abs_tol = max(abs_tol, rel_tol * 360) if abs_tol is not None and rel_tol is not None else None
            # Handle the degrees case
            if not (
                mini == pytest.approx(maxi - 360, abs=new_abs_tol, rel=rel_tol, nan_ok=nan_ok)
                or d1[k] == pytest.approx(d2[k], abs=abs_tol, rel=rel_tol, nan_ok=nan_ok)
            ):
                return False
        elif (
            d1[k] != pytest.approx(d2[k], abs=abs_tol, rel=rel_tol, nan_ok=nan_ok) and k != "isMoving"
        ):  # Special case for isMoving
            return False

    return True
