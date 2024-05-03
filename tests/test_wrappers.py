"""Tests for the wrappers module."""

import gymnasium as gym
import numpy as np
import pytest

from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
from rl_ai2thor.envs.wrappers import (
    ChannelFirstObservationWrapper,
    MoreThanOneTaskBlueprintError,
    NormalizeActionWrapper,
    NotSimpleActionEnvironmentMode,
    SimpleActionSpaceWrapper,
    SingleTaskWrapper,
)


# %% === Fixtures ===
@pytest.fixture()
def ithor_env():
    env = ITHOREnv()
    yield env
    env.close()


# %% === SingleTaskWrapper tests ===
single_task_override_config: dict = {
    "tasks": {
        "task_blueprints": [
            {
                "task_type": "PlaceIn",
                "args": {"placed_object_type": "Apple", "receptacle_type": "Bowl"},
                "scenes": ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
            }
        ]
    }
}


@pytest.fixture()
def single_task_ithor_env():
    env = ITHOREnv(config_override=single_task_override_config)
    yield env
    env.close()


def test_single_task_wrapper_observation_space(single_task_ithor_env):
    wrapped_env = SingleTaskWrapper(single_task_ithor_env)
    assert wrapped_env.observation_space == single_task_ithor_env.observation_space.spaces["env_obs"]


def test_single_task_wrapper_more_than_one_task_blueprint_error():
    multi_task_config = {
        "tasks": {
            "task_blueprints": [
                {
                    "task_type": "PlaceIn",
                    "args": {"placed_object_type": "Apple", "receptacle_type": "Bowl"},
                    "scenes": ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
                },
                {
                    "task_type": "PlaceIn",
                    "args": {"placed_object_type": "Banana", "receptacle_type": "Bowl"},
                    "scenes": ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
                },
            ]
        }
    }
    env = ITHOREnv(config_override=multi_task_config)
    with pytest.raises(MoreThanOneTaskBlueprintError) as exc_info:
        SingleTaskWrapper(env)
    assert exc_info.value.config == env.config


def test_single_task_wrapper_reset(single_task_ithor_env):
    wrapped_env = SingleTaskWrapper(single_task_ithor_env)
    observation, info = wrapped_env.reset()
    assert isinstance(observation, np.ndarray)
    assert observation.shape == single_task_ithor_env.observation_space.spaces["env_obs"].shape
    assert isinstance(info, dict)


def test_single_task_wrapper_step():
    override_config = single_task_override_config.copy()
    override_config["action_modifiers"] = {"discrete_actions": False}

    single_task_ithor_env = ITHOREnv(config_override=override_config)
    wrapped_env = SingleTaskWrapper(single_task_ithor_env)
    action = {"action_index": 0, "action_parameter": 1, "target_object_coordinates": [0.5, 0.5]}
    wrapped_env.reset()
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == single_task_ithor_env.observation_space.spaces["env_obs"].shape
    assert isinstance(reward, float | int)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# %% === FirstChannelObservationWrapper tests ===
base_config = {
    "controller_parameters": {
        "frame_width": 300,
        "frame_height": 300,
    }
}


@pytest.fixture()
def channel_last_ithor_env():
    env = ITHOREnv(config_override=override_config)
    yield env
    env.close()


def test_channel_first_observation_wrapper_observation_space(channel_last_ithor_env):
    wrapped_env = ChannelFirstObservationWrapper(channel_last_ithor_env)
    env_obs_space = channel_last_ithor_env.observation_space.spaces["env_obs"]
    assert env_obs_space == gym.spaces.Box(low=0, high=255, shape=(3, 300, 300), dtype=np.uint8)


def test_channel_first_observation_wrapper_reset(channel_last_ithor_env):
    wrapped_env = ChannelFirstObservationWrapper(channel_last_ithor_env)
    observation, _ = wrapped_env.reset()
    environment_obs = observation["env_obs"]
    assert isinstance(environment_obs, np.ndarray)
    assert environment_obs.shape == (3, 300, 300)


def test_channel_first_observation_wrapper_step():
    override_config = base_config.copy()
    override_config["action_modifiers"] = {"discrete_actions": False}
    channel_last_ithor_env = ITHOREnv(config_override=override_config)
    wrapped_env = ChannelFirstObservationWrapper(channel_last_ithor_env)
    action = {"action_index": 0, "action_parameter": 1, "target_object_coordinates": [0.5, 0.5]}
    wrapped_env.reset()
    observation, _, _, _, _ = wrapped_env.step(action)
    environment_obs = observation["env_obs"]
    assert isinstance(environment_obs, np.ndarray)
    assert environment_obs.shape == (3, 300, 300)


# %% === NormalizeActionWrapper tests ===#
def test_normalize_action_wrapper_action_space():
    env = ITHOREnv()
    wrapped_env = NormalizeActionWrapper(env)
    assert wrapped_env.action_space.spaces["target_object_coordinates"] == gym.spaces.Box(low=-1, high=1, shape=(2,))


def test_normalize_action_wrapper_action():
    env = ITHOREnv()
    wrapped_env = NormalizeActionWrapper(env)
    action = {
        "action_index": 0,
        "action_parameter": 1,
        "target_object_coordinates": [0.5, 0.5],
    }
    normalized_action = wrapped_env.action(action)
    assert normalized_action["target_object_coordinates"] == [0.75, 0.75]


# %% === SimpleActionSpaceWrapper tests ===
@pytest.fixture()
def simple_action_space_ithor_env():
    override_config = {
        "action_modifiers": {
            "discrete_actions": True,
            "target_closest_object": True,
        }
    }
    env = ITHOREnv(config_override=override_config)
    yield env
    env.close()


def test_simple_action_space_wrapper_not_simple_action_space_error():
    override_config = {
        "action_modifiers": {
            "discrete_actions": False,
            "target_closest_object": True,
        }
    }
    env = ITHOREnv(config_override=override_config)
    with pytest.raises(NotSimpleActionEnvironmentMode) as exc_info:
        SimpleActionSpaceWrapper(env)
    assert exc_info.value.config == env.config


def test_simple_action_space_wrapper_action_space(simple_action_space_ithor_env):
    wrapped_env = SimpleActionSpaceWrapper(simple_action_space_ithor_env)
    assert wrapped_env.action_space == simple_action_space_ithor_env.action_space.spaces["action_index"]


def test_simple_action_space_wrapper_action(simple_action_space_ithor_env):
    wrapped_env = SimpleActionSpaceWrapper(simple_action_space_ithor_env)
    action = 0
    converted_action = wrapped_env.action(action)
    assert converted_action == {"action_index": action}


# TODO: Add tests for combined wrappers
