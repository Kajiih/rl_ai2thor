"""Tests for the ai2thor_envs module."""

import pickle as pkl  # noqa: S403
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import pytest
from PIL import Image

from rl_ai2thor.envs._config import EnvConfig, InvalidActionGroupError  # noqa: PLC2701
from rl_ai2thor.envs.actions import ActionGroup, EnvActionName
from rl_ai2thor.envs.ai2thor_envs import (
    ITHOREnv,
    NoTaskBlueprintError,
)
from rl_ai2thor.envs.sim_objects import SimObjectType
from rl_ai2thor.envs.tasks.tasks import PlaceIn, PlaceNSameIn, UnknownTaskTypeError

if TYPE_CHECKING:
    from ai2thor.server import Event
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
partial_config_dict = {
    "action_groups": {
        ActionGroup.MOVEMENT_ACTIONS: True,
        ActionGroup.ROTATION_ACTIONS: True,
        ActionGroup.HEAD_MOVEMENT_ACTIONS: True,
        ActionGroup.CROUCH_ACTIONS: False,
        ActionGroup.OPEN_CLOSE_ACTIONS: True,
        ActionGroup.PICKUP_PUT_ACTIONS: True,
        ActionGroup.TOGGLE_ACTIONS: True,
    },
    "action_modifiers": {
        "partial_openness": True,
        "discrete_actions": False,
        "simple_movement_actions": True,
        "target_closest_object": False,
    },
}
partial_config = EnvConfig.init_from_dict(partial_config_dict)


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
        # EnvActionName.CLEAN_OBJECT: False,
        # EnvActionName.DIRTY_OBJECT: False,
    }
    action_availabilities = ITHOREnv._compute_action_availabilities(partial_config)

    assert action_availabilities == expected_availabilities


def test_compute_action_availabilities_unknown_action_category():
    config_dict = {
        "action_groups": {
            "_unknown_action_category": None,
        }
    }

    with pytest.raises(InvalidActionGroupError) as exc_info:
        EnvConfig.init_from_dict(config_dict)

    assert exc_info.value.action_group == "_unknown_action_category"


def test__action_space():
    ithor_env = ITHOREnv(override_dict=partial_config_dict)

    ithor_env._initialize_action_space()

    assert isinstance(ithor_env.action_space, gym.spaces.Dict)
    assert "action_index" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_index"], gym.spaces.Discrete)
    assert "action_parameter" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["action_parameter"], gym.spaces.Box)
    assert "target_object_coordinates" in ithor_env.action_space.spaces
    assert isinstance(ithor_env.action_space.spaces["target_object_coordinates"], gym.spaces.Box)


def test__create_observation_space():
    ithor_env = ITHOREnv(
        override_dict={
            "controller_parameters": {
                "frame_height": 84,
                "frame_width": 44,
            },
        }
    )

    ithor_env._initialize_observation_space()

    assert isinstance(ithor_env.observation_space, gym.spaces.Dict)
    env_observation = ithor_env.observation_space.spaces["env_obs"]
    task_observation = ithor_env.observation_space.spaces["task_obs"]
    scene_observation = ithor_env.observation_space.spaces["scene_obs"]

    assert isinstance(env_observation, gym.spaces.Box)
    assert env_observation.shape == (84, 44, 3)
    assert isinstance(task_observation, gym.spaces.Discrete)
    assert isinstance(scene_observation, gym.spaces.Discrete)
    # TODO: Need to change this when the task observation can change


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
    config_dict = {
        "tasks": {
            "globally_excluded_scenes": ["FloorPlan1"],
            "task_blueprints": [
                {
                    "task_type": "PlaceIn",
                    "args": {
                        "placed_object_type": "Knife",
                        "receptacle_type": "Sink",
                    },
                    "scenes": ["FloorPlan1", "FloorPlan2"],
                },
                {
                    "task_type": "PlaceNSameIn",
                    "args": {"placed_object_type": "Apple", "receptacle_type": "Plate", "n": 2},
                    "scenes": "FloorPlan3",
                },
            ],
        }
    }
    config = EnvConfig.init_from_dict(config_dict)

    task_blueprints = ITHOREnv._create_task_blueprints(config)

    assert len(task_blueprints) == len(config_dict["tasks"])

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
    config_dict = {
        "tasks": {
            "globally_excluded_scenes": [],
            "task_blueprints": [
                {
                    "task_type": "_unknown_task",
                    "args": {},
                    "scenes": [],
                },
            ],
        }
    }
    config = EnvConfig.init_from_dict(config_dict)

    with pytest.raises(UnknownTaskTypeError) as exc_info:
        ITHOREnv._create_task_blueprints(config)

    assert exc_info.value.task_type == "_unknown_task"


# More with empty task config
def test__create_task_blueprints_empty_task_config():
    config = {
        "tasks": {
            "task_blueprints": [],
            "globally_excluded_scenes": [],
        }
    }
    config = EnvConfig.init_from_dict(config)

    with pytest.raises(NoTaskBlueprintError) as exc_info:
        ITHOREnv._create_task_blueprints(config)

    assert isinstance(exc_info.value, NoTaskBlueprintError)


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


# === Test randomize_scene ===
randomize_scene_media_path = test_media_path / "_randomize_scene"


def test__randomize_scene_random_agent_spawn():  # noqa: PLR0914
    image_path = randomize_scene_media_path / "randomize_scene_random_agent_spawn"
    image_path.mkdir(exist_ok=True, parents=True)

    config_dict = {
        "scene_randomization": {
            "random_agent_spawn": True,
            "random_object_spawn": False,
            "random_object_materials": False,
            "random_lighting": False,
            "random_object_colors": False,
        }
    }
    ithor_env = ITHOREnv(override_dict=config_dict)
    controller = ithor_env.controller
    ithor_env.last_event = controller.reset()  # type: ignore

    initial_event: Event = ithor_env.last_event  # type: ignore
    metadata_1 = initial_event.metadata
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_0.png")
    ithor_env._randomize_scene(controller)
    event_1: Event = ithor_env.last_event  # type: ignore
    metadata_2 = event_1.metadata
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_1.png")
    ithor_env._randomize_scene(controller)
    event_2: Event = ithor_env.last_event  # type: ignore
    metadata_3 = event_2.metadata
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_2.png")

    # Check cameraOrthSize
    cameraOrthSize_1 = metadata_1["cameraOrthSize"]
    cameraOrthSize_2 = metadata_2["cameraOrthSize"]
    cameraOrthSize_3 = metadata_3["cameraOrthSize"]
    assert cameraOrthSize_1 != cameraOrthSize_2 or cameraOrthSize_2 != cameraOrthSize_3

    # Check cameraPosition
    cameraPosition_1 = metadata_1["cameraPosition"]
    cameraPosition_2 = metadata_2["cameraPosition"]
    cameraPosition_3 = metadata_3["cameraPosition"]
    assert cameraPosition_1 != cameraPosition_2
    assert cameraPosition_2 != cameraPosition_3
    assert cameraPosition_1 != cameraPosition_3


def test__randomize_scene_random_object_spawn():
    image_path = randomize_scene_media_path / "randomize_scene_random_object_spawn"
    image_path.mkdir(exist_ok=True, parents=True)

    config_dict = {
        "scene_randomization": {
            "random_agent_spawn": False,
            "random_object_spawn": True,
            "random_object_materials": False,
            "random_lighting": False,
            "random_object_colors": False,
        }
    }
    ithor_env = ITHOREnv(override_dict=config_dict)
    controller = ithor_env.controller
    ithor_env.last_event = controller.reset()  # type: ignore

    initial_event: Event = ithor_env.last_event  # type: ignore
    metadata_1 = initial_event.metadata
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_0.png")
    ithor_env._randomize_scene(controller)
    event_1: Event = ithor_env.last_event  # type: ignore
    metadata_2 = event_1.metadata
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_1.png")
    ithor_env._randomize_scene(controller)
    event_2: Event = ithor_env.last_event  # type: ignore
    metadata_3 = event_2.metadata
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_2.png")

    # Check object positions
    objects_1 = {obj["objectId"] for obj in metadata_1["objects"]}
    objects_2 = {obj["objectId"] for obj in metadata_2["objects"]}
    objects_3 = {obj["objectId"] for obj in metadata_3["objects"]}
    assert objects_1 != objects_2
    assert objects_2 != objects_3
    assert objects_1 != objects_3


def test__randomize_scene_random_object_materials():
    image_path = randomize_scene_media_path / "randomize_scene_object_materials"
    image_path.mkdir(exist_ok=True, parents=True)

    config_dict = {
        "scene_randomization": {
            "random_agent_spawn": False,
            "random_object_spawn": False,
            "random_object_materials": True,
            "random_lighting": False,
            "random_object_colors": False,
        }
    }
    ithor_env = ITHOREnv(override_dict=config_dict)
    controller = ithor_env.controller
    ithor_env.last_event = controller.reset()  # type: ignore

    initial_event: Event = ithor_env.last_event  # type: ignore
    metadata_1 = initial_event.metadata
    frame_1 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_0.png")
    ithor_env._randomize_scene(controller)
    event_1: Event = ithor_env.last_event  # type: ignore
    metadata_2 = event_1.metadata
    frame_2 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_1.png")
    ithor_env._randomize_scene(controller)
    event_2: Event = ithor_env.last_event  # type: ignore
    metadata_3 = event_2.metadata
    frame_3 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_2.png")

    # Check frames
    assert not np.allclose(frame_1, frame_2)
    assert not np.allclose(frame_2, frame_3)
    assert not np.allclose(frame_1, frame_3)


def test__randomize_scene_random_lighting():
    image_path = randomize_scene_media_path / "randomize_scene_lighting"
    image_path.mkdir(exist_ok=True, parents=True)

    config_dict = {
        "scene_randomization": {
            "random_agent_spawn": False,
            "random_object_spawn": False,
            "random_object_materials": False,
            "random_lighting": True,
            "random_object_colors": False,
        }
    }
    ithor_env = ITHOREnv(override_dict=config_dict)
    controller = ithor_env.controller
    ithor_env.last_event = controller.reset()  # type: ignore

    initial_event: Event = ithor_env.last_event  # type: ignore
    metadata_1 = initial_event.metadata
    frame_1 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_0.png")
    ithor_env._randomize_scene(controller)
    event_1: Event = ithor_env.last_event  # type: ignore
    metadata_2 = event_1.metadata
    frame_2 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_1.png")
    ithor_env._randomize_scene(controller)
    event_2: Event = ithor_env.last_event  # type: ignore
    metadata_3 = event_2.metadata
    frame_3 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_2.png")

    # Check frames
    assert not np.allclose(frame_1, frame_2)
    assert not np.allclose(frame_2, frame_3)
    assert not np.allclose(frame_1, frame_3)


def test__randomize_scene_random_object_colors():
    image_path = randomize_scene_media_path / "randomize_scene_object_colors"
    image_path.mkdir(exist_ok=True, parents=True)

    config_dict = {
        "scene_randomization": {
            "random_agent_spawn": False,
            "random_object_spawn": False,
            "random_object_materials": False,
            "random_lighting": False,
            "random_object_colors": True,
        }
    }
    ithor_env = ITHOREnv(override_dict=config_dict)
    controller = ithor_env.controller
    ithor_env.last_event = controller.reset()  # type: ignore

    initial_event: Event = ithor_env.last_event  # type: ignore
    metadata_1 = initial_event.metadata
    frame_1 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_0.png")
    ithor_env._randomize_scene(controller)
    event_1: Event = ithor_env.last_event  # type: ignore
    metadata_2 = event_1.metadata
    frame_2 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_1.png")
    ithor_env._randomize_scene(controller)
    event_2: Event = ithor_env.last_event  # type: ignore
    metadata_3 = event_2.metadata
    frame_3 = ithor_env.last_frame
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_2.png")

    # Check frames
    assert not np.allclose(frame_1, frame_2)
    assert not np.allclose(frame_2, frame_3)
    assert not np.allclose(frame_1, frame_3)


def test_reset_scene_randomization_materials():
    image_path = randomize_scene_media_path / "reset_scene_randomization_materials"
    image_path.mkdir(exist_ok=True, parents=True)

    config_dict = {
        "scene_randomization": {
            "random_agent_spawn": True,
            "random_object_spawn": True,
            "random_object_materials": True,
            "random_lighting": True,
            "random_object_colors": True,
        },
        "tasks": {
            "task_blueprints": [
                {
                    "task_type": "Open",
                    "args": {"opened_object_type": "Fridge"},
                    "scenes": ["FloorPlan1"],
                }
            ],
        },
    }
    ithor_env = ITHOREnv(override_dict=config_dict)
    obs1, _info1 = ithor_env.reset()
    frame_1 = obs1["env_obs"]
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_0.png")

    obs2, _info2 = ithor_env.reset()
    frame_2 = obs2["env_obs"]
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_1.png")

    obs3, _info3 = ithor_env.reset()
    frame_3 = obs3["env_obs"]
    Image.fromarray(ithor_env.last_frame).save(image_path / "frame_2.png")

    assert not np.allclose(frame_1, frame_2)
    assert not np.allclose(frame_2, frame_3)
    assert not np.allclose(frame_1, frame_3)


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
