from collections.abc import Mapping
from unittest.mock import call, patch

import gymnasium as gym
import pytest
import yaml
from _pytest.python_api import ApproxMapping  # noqa: PLC2701

from rl_ai2thor.envs.actions import EnvActionName
from rl_ai2thor.envs.ai2thor_envs import ITHOREnv

# %% === Constants ===
abs_tolerance = 1
rel_tolerance = 2e-1

seed = 42


# %% === Fixtures ===
@pytest.fixture()
def ithor_env():
    return ITHOREnv()


# %% === Init tests ===
def test_load_and_override_config(ithor_env: ITHOREnv):
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
        config = ithor_env._load_and_override_config(override_config)

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


def test_compute_action_availabilities(ithor_env: ITHOREnv):
    # Set the environment mode config
    ithor_env.config = partial_config

    # Define the expected action availabilities based on the environment mode config
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

    # Call the _compute_action_availabilities method
    action_availabilities = ithor_env._compute_action_availabilities()

    # Assert the expected action availabilities
    assert action_availabilities == expected_availabilities


def test_action_space(ithor_env: ITHOREnv):
    # Set the environment mode config
    ithor_env.config = partial_config

    # Call the _create_action_space method
    ithor_env._create_action_space()

    # Assert the action space dictionary
    assert isinstance(ithor_env.action_space, gym.spaces.Dict)
    assert "action_index" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_index"], gym.spaces.Discrete)
    assert "action_parameter" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_parameter"], gym.spaces.Box)
    assert "target_object_position" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["target_object_position"], gym.spaces.Box)


def test_create_observation_space(ithor_env: ITHOREnv):
    # Set the environment mode config
    ithor_env.config = {
        "controller_parameters": {
            "height": 84,
            "width": 44,
        },
        "grayscale": False,
    }

    # Call the _create_observation_space method
    ithor_env._create_observation_space()

    # Assert the observation space
    assert isinstance(ithor_env.observation_space, gym.spaces.Box)
    assert ithor_env.observation_space.shape == (84, 44, 3)


def test_create_observation_space_grayscale(ithor_env: ITHOREnv):
    # Set the environment mode config
    ithor_env.config = {
        "controller_parameters": {
            "height": 84,
            "width": 44,
        },
        "grayscale": True,
    }

    # Call the _create_observation_space method
    ithor_env._create_observation_space()

    # Assert the observation space
    assert isinstance(ithor_env.observation_space, gym.spaces.Box)
    assert ithor_env.observation_space.shape == (84, 44, 1)


# %% === Reproducibility tests ===
@pytest.mark.xfail(reason="Rendering in ai2thor is not deterministic")
def test_reset_exact_observation_reproducibility(ithor_env: ITHOREnv):
    # Initialize the environment with the seed
    obs1, info1 = ithor_env.reset(seed=seed)

    # Reinitialize the environment with the same seed
    obs2, info2 = ithor_env.reset(seed=seed)

    # Check if the observations are identical
    assert obs1 == pytest.approx(obs2, abs=2)
    assert info1 == info2


def test_reset_same_scene_reproducibility(ithor_env: ITHOREnv):
    # Initialize the environment with the seed
    _, info1 = ithor_env.reset(seed=seed)

    # Reinitialize the environment with the same seed
    _, info2 = ithor_env.reset(seed=seed)

    # Check if the scene are identical
    split_assert_dicts(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)
    assert are_close_dict(info1["metadata"], info2["metadata"], abs_tol=abs_tolerance, rel_tol=rel_tolerance)


def test_reset_not_same_scene(ithor_env: ITHOREnv):
    # Initialize the environment with the seed
    _, info1 = ithor_env.reset(seed=seed)

    # Reset the environment to get a different scene
    _, info2 = ithor_env.reset()

    # Check that the scenes are different
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
            assert d1[k] == pytest.approx(d2[k], abs=abs_tol, rel=rel_tol, nan_ok=nan_ok)


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


# %%
