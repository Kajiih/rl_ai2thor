"""Tests for the ai2thor_envs module."""

import pickle as pkl  # noqa: S403
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from unittest.mock import call, patch

import gymnasium as gym
import pytest
from _pytest.python_api import ApproxMapping  # noqa: PLC2701
from PIL import Image

from rl_ai2thor.envs.actions import EnvActionName
from rl_ai2thor.envs.ai2thor_envs import (
    ITHOREnv,
    NoTaskBlueprintError,
    UnknownActionCategoryError,
    UnknownTaskTypeError,
)
from rl_ai2thor.envs.sim_objects import ALL_OBJECT_GROUPS, SimObjectType
from rl_ai2thor.envs.tasks.tasks import ALL_TASKS

# %% === Constants ===
abs_tolerance = 4
rel_tolerance = 5e-2

seed = 42


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


def test__compute_action_availabilities(ithor_env: ITHOREnv):
    ithor_env.config = partial_config

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

    action_availabilities = ithor_env._compute_action_availabilities()

    assert action_availabilities == expected_availabilities


def test_compute_action_availabilities_unknown_action_category(ithor_env: ITHOREnv):
    ithor_env.config = {
        "action_categories": {
            "_unknown_action_category": None,
        }
    }

    with pytest.raises(UnknownActionCategoryError) as exc_info:
        ithor_env._compute_action_availabilities()

    assert exc_info.value.action_category == "_unknown_action_category"


def test__action_space(ithor_env: ITHOREnv):
    ithor_env.config = partial_config

    ithor_env._create_action_space()

    assert isinstance(ithor_env.action_space, gym.spaces.Dict)
    assert "action_index" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_index"], gym.spaces.Discrete)
    assert "action_parameter" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_parameter"], gym.spaces.Box)
    assert "target_object_position" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["target_object_position"], gym.spaces.Box)


def test__create_observation_space(ithor_env: ITHOREnv):
    ithor_env.config = {
        "controller_parameters": {
            "height": 84,
            "width": 44,
        },
        "grayscale": False,
    }

    ithor_env._create_observation_space()

    assert isinstance(ithor_env.observation_space, gym.spaces.Box)
    assert ithor_env.observation_space.shape == (84, 44, 3)


def test__create_observation_space_grayscale(ithor_env: ITHOREnv):
    ithor_env.config = {
        "controller_parameters": {
            "height": 84,
            "width": 44,
        },
        "grayscale": True,
    }

    ithor_env._create_observation_space()

    assert isinstance(ithor_env.observation_space, gym.spaces.Box)
    assert ithor_env.observation_space.shape == (84, 44, 1)


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

    available_scenes = ITHOREnv._compute_available_scenes(scenes, excluded_scenes)

    assert available_scenes == expected_available_scenes


def test__develop_config_task_args_single_arg():
    task_args = "Mug"
    developed_task_arg = ITHOREnv._develop_config_task_args(task_args)
    assert developed_task_arg == frozenset(["Mug"])


def test__develop_config_task_args_single_arg_from_group():
    task_args = "_PICKUPABLES"
    developed_task_arg = ITHOREnv._develop_config_task_args(task_args)
    expected_developed_task_arg = frozenset(ALL_OBJECT_GROUPS["_PICKUPABLES"])
    assert developed_task_arg == expected_developed_task_arg


def test__develop_config_task_args_multiple_args():
    task_args = ["Mug", "Knife", "_PICKUPABLES"]
    developed_task_arg = ITHOREnv._develop_config_task_args(task_args)
    expected_developed_task_arg = frozenset(["Mug", "Knife", *ALL_OBJECT_GROUPS["_PICKUPABLES"]])
    assert developed_task_arg == expected_developed_task_arg


def test__develop_config_task_args_multiple_args_with_duplicates():
    task_args = ["Mug", "Knife", "_PICKUPABLES", "Mug"]
    developed_task_arg = ITHOREnv._develop_config_task_args(task_args)
    expected_developed_task_arg = frozenset(["Mug", "Knife", *ALL_OBJECT_GROUPS["_PICKUPABLES"]])
    assert developed_task_arg == expected_developed_task_arg


def test__develop_config_task_args_empty_arg():
    task_args = []
    developed_task_arg = ITHOREnv._develop_config_task_args(task_args)
    assert developed_task_arg == frozenset()


def test__create_task_blueprints(ithor_env: ITHOREnv):
    ithor_env.config = {
        "globally_excluded_scenes": ["FloorPlan1"],
        "tasks": [
            {
                "type": "PlaceIn",
                "args": {
                    "placed_object_type": ["Mug", "Knife"],
                    "receptacle_type": ["Sink", "Pot", "_PICKUPABLE_RECEPTACLES"],
                },
                "scenes": ["FloorPlan1", "FloorPlan2"],
            },
            {
                "type": "PlaceNSameIn",
                "args": {"placed_object_type": "Apple", "receptacle_type": "Plate", "n": 2},
                "scenes": ["FloorPlan3", "FloorPlan4"],
            },
        ],
    }

    task_blueprints = ithor_env._create_task_blueprints()

    assert len(task_blueprints) == 2  # noqa: PLR2004

    # Check task blueprint 1
    task_blueprint_1 = task_blueprints[0]
    assert task_blueprint_1.task_type == ALL_TASKS["PlaceIn"]
    assert task_blueprint_1.scenes == {"FloorPlan2"}
    assert task_blueprint_1.task_args == {
        "placed_object_type": frozenset([SimObjectType("Mug"), SimObjectType("Knife")]),
        "receptacle_type": frozenset([
            SimObjectType("Sink"),
            SimObjectType("Pot"),
            *list(ALL_OBJECT_GROUPS["_PICKUPABLE_RECEPTACLES"]),
        ]),
    }

    # Check task blueprint 2
    task_blueprint_2 = task_blueprints[1]
    assert task_blueprint_2.task_type == ALL_TASKS["PlaceNSameIn"]
    assert task_blueprint_2.scenes == {"FloorPlan3", "FloorPlan4"}
    assert task_blueprint_2.task_args == {
        "placed_object_type": frozenset([SimObjectType("Apple")]),
        "receptacle_type": frozenset([SimObjectType("Plate")]),
        "n": frozenset({2}),
    }


def test__create_task_blueprints_unknown_task(ithor_env: ITHOREnv):
    ithor_env.config = {
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
        ithor_env._create_task_blueprints()

    assert exc_info.value.task_type == "_unknown_task"


# More with empty task config
def test__create_task_blueprints_empty_task_config(ithor_env: ITHOREnv):
    env_config = {
        "globally_excluded_scenes": [],
        "tasks": [],
    }
    ithor_env.config = deepcopy(env_config)

    with pytest.raises(NoTaskBlueprintError) as exc_info:
        ithor_env._create_task_blueprints()

    assert exc_info.value.config == {"globally_excluded_scenes": [], "tasks": []}


# %% === Reproducibility tests ===
@pytest.mark.xfail(reason="Rendering in ai2thor is not deterministic")
def test_reset_exact_observation_reproducibility(ithor_env: ITHOREnv):
    obs1, info1 = ithor_env.reset(seed=seed)
    obs2, info2 = ithor_env.reset(seed=seed)

    assert obs1 == pytest.approx(obs2, abs=abs_tolerance, rel=rel_tolerance)
    assert info1 == info2


# This test fails sometimes because AI2THOR is not deterministic
# ! Sometimes 'Pen' and 'Pencil' are switched...?
def test_reset_same_scene_reproducibility(ithor_env: ITHOREnv, ithor_env_2: ITHOREnv):
    obs1, info1 = ithor_env.reset(seed=seed)
    obs2, info2 = ithor_env_2.reset(seed=seed)
    assert ithor_env.current_task_type == ithor_env_2.current_task_type
    assert ithor_env.current_task_args == ithor_env_2.current_task_args

    obs1_2, info1_2 = ithor_env.reset(seed=seed)
    obs2_2, info2_2 = ithor_env_2.reset(seed=seed)
    assert ithor_env.current_task_type == ithor_env_2.current_task_type
    assert ithor_env.current_task_args == ithor_env_2.current_task_args

    # Check if the scene are identical
    split_assert_dicts(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)
    split_assert_dicts(info1_2["metadata"], info2_2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)

    # Check if the observations are identical
    try:
        assert obs1 == pytest.approx(obs2, abs=rel_tolerance * 255, rel=rel_tolerance)
    except AssertionError:
        Image.fromarray(obs1).save("obs1.png")
        Image.fromarray(obs2).save("obs2.png")
        Image.fromarray(obs1 - obs2).save("diff.png")
        assert obs1 == pytest.approx(obs2, abs=0, rel=0)

    try:
        assert obs1_2 == pytest.approx(obs2_2, abs=rel_tolerance * 255, rel=rel_tolerance)
    except AssertionError:
        Image.fromarray(obs1_2).save("obs1_2.png")
        Image.fromarray(obs2_2).save("obs2_2.png")
        Image.fromarray(obs1_2 - obs2_2).save("diff_2.png")
        assert obs1_2 == pytest.approx(obs2_2, abs=0, rel=0)

    assert are_close_dict(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)
    assert are_close_dict(info1_2["metadata"], info2_2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)


def test_reset_separate_runs_reproducibility(ithor_env: ITHOREnv):
    obs1, info1 = ithor_env.reset(seed=seed)
    task_type = ithor_env.current_task_type
    task_args = ithor_env.current_task_args
    data_path = Path("tests/data/reset_separate_runs_reproducibility_obs_info.pkl")
    # to_serialize_data = (obs1, info1, task_type, task_args)
    # with data_path.open("wb") as f:
    #     pkl.dump(to_serialize_data, f)
    obs2, info2, task_type2, task_args2 = pkl.load(data_path.open("rb"))  # noqa: S301

    assert task_type == task_type2
    assert task_args == task_args2

    # Check if the scene are identical
    split_assert_dicts(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)

    # Check if the observations are identical
    try:
        assert obs1 == pytest.approx(obs2, abs=rel_tolerance * 255, rel=rel_tolerance)
    except AssertionError:
        Image.fromarray(obs1).save("obs1.png")
        Image.fromarray(obs2).save("obs2.png")
        Image.fromarray(obs1 - obs2).save("diff.png")
        assert obs1 == pytest.approx(obs2, abs=0, rel=0)

    assert are_close_dict(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)


def test_reset_not_same_scene(ithor_env: ITHOREnv):
    _, info1 = ithor_env.reset(seed=seed)
    _, info2 = ithor_env.reset(seed=seed + 1)

    assert not are_close_dict(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)


# %% Utils
def nested_dict_approx(expected, rel=None, abs=None, nan_ok=False):  # noqa: A002
    if isinstance(expected, Mapping):
        return ApproxNestedMapping(expected, rel, abs, nan_ok)
    return pytest.approx(expected, rel, abs, nan_ok)


# ** Broken
class ApproxNestedMapping(ApproxMapping):
    def _yield_comparisons(self, actual):
        for k in self.expected:
            if isinstance(actual[k], type(self.expected)):
                yield from ApproxNestedMapping(
                    self.expected[k],
                    rel=self.rel,
                    abs=self.abs,
                    nan_ok=self.nan_ok,
                )._yield_comparisons(actual[k])
            else:
                yield actual[k], self.expected[k]

    def _check_type(self):
        for value in self.expected.values():
            if not isinstance(value, type(self.expected)) and not isinstance(self.expected, Mapping):
                super()._check_type()


def nested_list_to_dict(nested_dict):
    if isinstance(nested_dict, list):
        return {i: nested_list_to_dict(nested_dict[i]) for i in range(len(nested_dict))}
    if isinstance(nested_dict, dict):
        return {key: nested_list_to_dict(nested_dict[key]) for key in nested_dict}
    return nested_dict


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
