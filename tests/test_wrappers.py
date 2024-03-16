"""Tests for the wrappers module."""

import numpy as np
import pytest

from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
from rl_ai2thor.envs.tasks.tasks import TaskBlueprint
from rl_ai2thor.envs.wrappers import (
    MoreThanOneArgumentValueError,
    MoreThanOneTaskBlueprintError,
    SingleTaskWrapper,
)


# %% === Fixtures ===
@pytest.fixture()
def ithor_env():
    env = ITHOREnv()
    yield env
    env.close()


# %% === SingleTaskWrapper tests ===
single_task_override_config = {
    "tasks": [
        {
            "type": "PlaceIn",
            "args": {"placed_object_type": "Apple", "receptacle_type": "Bowl"},
            "scenes": ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
        }
    ]
}


@pytest.fixture()
def single_task_ithor_env():
    env = ITHOREnv(override_config=single_task_override_config)
    yield env
    env.close()


def test_single_task_wrapper_observation_space(single_task_ithor_env):
    wrapped_env = SingleTaskWrapper(single_task_ithor_env)
    assert wrapped_env.observation_space == single_task_ithor_env.observation_space.spaces["env_obs"]


def test_single_task_wrapper_more_than_one_task_blueprint_error():
    multi_task_config = {
        "tasks": [
            {
                "type": "PlaceIn",
                "args": {"placed_object_type": "Apple", "receptacle_type": "Bowl"},
                "scenes": ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
            },
            {
                "type": "PlaceIn",
                "args": {"placed_object_type": "Banana", "receptacle_type": "Bowl"},
                "scenes": ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
            },
        ]
    }
    env = ITHOREnv(override_config=multi_task_config)
    with pytest.raises(MoreThanOneTaskBlueprintError) as exc_info:
        SingleTaskWrapper(env)
    assert exc_info.value.config == env.config


def test_single_task_wrapper_more_than_one_argument_value_error():
    multi_arg_values_config = {
        "tasks": [
            {
                "type": "PlaceIn",
                "args": {"placed_object_type": ["Apple", "Banana"], "receptacle_type": "Bowl"},
                "scenes": ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
            },
        ]
    }
    env = ITHOREnv(override_config=multi_arg_values_config)
    with pytest.raises(MoreThanOneArgumentValueError) as exc_info:
        SingleTaskWrapper(env)
    assert exc_info.value.config == env.config


def test_single_task_wrapper_reset(single_task_ithor_env):
    wrapped_env = SingleTaskWrapper(single_task_ithor_env)
    observation, info = wrapped_env.reset()
    assert isinstance(observation, np.ndarray)
    assert observation.shape == single_task_ithor_env.observation_space.spaces["env_obs"].shape
    assert isinstance(info, dict)


def test_single_task_wrapper_step(single_task_ithor_env):
    wrapped_env = SingleTaskWrapper(single_task_ithor_env)
    action = {"action_index": 0, "action_parameter": 1, "target_object_position": [0.5, 0.5]}
    wrapped_env.reset()
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == single_task_ithor_env.observation_space.spaces["env_obs"].shape
    assert isinstance(reward, float | int)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
