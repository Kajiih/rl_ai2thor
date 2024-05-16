"""
Gymnasium interface for RL-THOR environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import operator
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
import yaml
from ai2thor.controller import Controller
from numpy.typing import NDArray

from rl_thor.envs._config import EnvConfig
from rl_thor.envs.actions import (
    ACTIONS_BY_GROUP,
    ACTIONS_BY_NAME,
    ALL_ACTIONS,
    EnvActionName,
    EnvironmentAction,
)
from rl_thor.envs.scenes import ALL_SCENES, SCENE_ID_TO_INDEX_MAP, SCENE_IDS, SceneGroup, SceneId
from rl_thor.envs.sim_objects import SimObjMetadata, SimObjVariableProp
from rl_thor.envs.tasks.tasks import ALL_TASKS, UnknownTaskTypeError
from rl_thor.envs.tasks.tasks_interface import (
    BaseTask,
    TaskBlueprint,
)
from rl_thor.utils.general_utils import ROOT_DIR, update_nested_dict

if TYPE_CHECKING:
    from ai2thor.server import Event

    from rl_thor.envs.reward import BaseRewardHandler
    from rl_thor.envs.sim_objects import SimObjId


# %% === Environment definitions ===
class BaseAI2THOREnv[ObsType, ActType](gym.Env, ABC):
    """Base class for AI2-THOR environment."""

    def __init__(self) -> None:
        """Initialize the environment."""

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (ActType): Action to take in the environment.

        Returns:
            observation (ObsType): Observation of the environment.
            reward (float): Reward of the action.
            terminated (bool): Whether the agent reaches a terminal state (realized the task).
            truncated (bool): Whether the limit of steps per episode has been reached.
            info (dict): Additional information about the environment.
        """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """
        Reset the environment.

        Reinitialize the environment random number generator if a seed is given.
        """
        super().reset(seed=seed, options=options)

    def close(self) -> None:
        """Close the environment."""


class ITHOREnv(
    BaseAI2THOREnv[
        dict[str, NDArray[np.uint8] | str],
        dict[str, Any],
    ]
):
    """
    Wrapper base class for iTHOR environment.

    It is a multi-task environment that follows MTEnv interface with the observation
    being returned as a dictionary with a env_obs and task_obs keys corresponding to
    the environment observation and the task observation respectively.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 15}
    render_mode: str = "rgb_array"
    # TODO: Change render fps with config

    def __init__(
        self,
        config_path: str | Path = Path("config/environment_config.yaml"),
        config_override: dict | None = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            config_path (str | Path): Relative path to the environment config file. Default is
                "config/environment_config.yaml".
            config_override (dict, Optional): Dictionary whose keys will override the given config.
        """
        self.config = self._load_config(config_path, config_override)
        # Initialize gymnasium seed
        super().reset(seed=self.config.seed)
        # TODO: Add the possibility to add task blueprints directly instead of going through the config
        self.task_blueprints = self._create_task_blueprints(self.config)
        self._initialize_action_space()
        self._initialize_observation_space()
        self.controller = self._create_ai2thor_controller(self.config)

        # === Type Annotations ===
        self.config: EnvConfig
        self.action_availabilities: dict[EnvActionName, bool]
        self.action_idx_to_name: dict[int, EnvActionName]
        self.action_space: gym.spaces.Dict
        self.observation_space: gym.spaces.Dict
        self.controller: Controller
        self.current_scene: SceneId
        self.current_task_type: type[BaseTask]
        self.task: BaseTask
        self.step_count: int
        self.reward_handler: BaseRewardHandler
        self.np_random: np.random.Generator
        self.task_blueprints: list[TaskBlueprint]
        self.last_info: dict[str, Any]

    @property
    def last_event(self) -> Event:
        """Return the last event of the environment."""
        return self.controller.last_event  # type: ignore

    @property
    def last_frame(self) -> NDArray[np.uint8]:
        """Return the last frame of the environment."""
        return self.last_event.frame  # type: ignore

    def render(self) -> NDArray[np.uint8]:
        """Return the last frame of the environment."""
        return self.last_frame

    @staticmethod
    def _load_config(config_path: str | Path, config_override: dict | None = None) -> EnvConfig:
        """
        Load the environment config from a yaml file.

        Args:
            config_path (str | Path): Relative path to the environment config file.
            config_override (dict, Optional): Dictionary whose keys will override the given config.

        Returns:
            dict: Environment config.
        """
        config_path = Path(ROOT_DIR) / config_path
        with config_path.open("r") as file:
            config = yaml.safe_load(file)
        if config_override is not None:
            update_nested_dict(config, config_override)
        return EnvConfig.init_from_dict(config)

    @staticmethod
    def _compute_action_availabilities(config: EnvConfig) -> dict[EnvActionName, bool]:
        """
        Compute the action availabilities based on the environment mode config.

        Args:
            config (EnvConfig): Environment config.

        Returns:
            dict[EnvActionName, bool]: Dictionary indicating which actions are available.
        """
        action_availabilities = {action.name: False for action in ALL_ACTIONS}

        for action_group in config.action_groups:
            if config.action_groups[action_group]:
                # Enable all actions in the action group
                for action in ACTIONS_BY_GROUP[action_group]:
                    action_availabilities[action.name] = True

        # Handle specific cases
        if config.action_modifiers.simple_movement_actions:
            for action_name in [EnvActionName.MOVE_BACK, EnvActionName.MOVE_LEFT, EnvActionName.MOVE_RIGHT]:
                action_availabilities[action_name] = False

        if config.action_modifiers.partial_openness:
            for action_name in [EnvActionName.OPEN_OBJECT, EnvActionName.CLOSE_OBJECT]:
                action_availabilities[action_name] = False
            action_availabilities[EnvActionName.PARTIAL_OPEN_OBJECT] = True

        return action_availabilities

    def _initialize_action_space(self) -> None:
        """Create the action space according to the available action groups in the environment mode config."""
        self.action_availabilities = self._compute_action_availabilities(self.config)

        available_actions = [action_name for action_name, available in self.action_availabilities.items() if available]
        self.action_idx_to_name = dict(enumerate(available_actions))

        # Create the action space dictionary
        action_space_dict: dict[str, gym.Space] = {"action_index": gym.spaces.Discrete(len(self.action_idx_to_name))}

        if not self.config.action_modifiers.discrete_actions:
            action_space_dict["action_parameter"] = gym.spaces.Box(low=0, high=1, shape=())

        if not self.config.action_modifiers.target_closest_object:
            action_space_dict["target_object_coordinates"] = gym.spaces.Box(low=0, high=1, shape=(2,))

        self.action_space = gym.spaces.Dict(action_space_dict)

    def _initialize_observation_space(self) -> None:
        resolution = (
            self.config.controller_parameters.frame_height,
            self.config.controller_parameters.frame_width,
        )
        env_obs_space = gym.spaces.Box(low=0, high=255, shape=(resolution[0], resolution[1], 3), dtype=np.uint8)
        # task_desc_obs_space = gym.spaces.Text(
        #     max_length=1000, charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,. "
        # )
        task_obs_space = gym.spaces.Discrete(len(self.task_blueprints))

        scene_obs_space = gym.spaces.Discrete(
            len(ALL_SCENES)
        )  # TODO? Replace by only the number of scenes in the task blueprints?

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict({
            "env_obs": env_obs_space,
            # "task_desc_obs": task_desc_obs_space,
            "task_obs": task_obs_space,
            "scene_obs": scene_obs_space,
        })

    @staticmethod
    def _compute_config_available_scenes(
        scenes: list[str] | str,
        excluded_scenes: set[str] | None = None,
    ) -> set[SceneId]:
        """
        Compute the available scenes based on the environment mode config.

        Args:
            scenes (list[str] | str): Scene names to consider.
            excluded_scenes (set[str], Optional): Set of scene names to exclude.

        Returns:
            set[SceneId]: Set of available scene names.
        """
        if not isinstance(scenes, list):
            scenes = [scenes]
        available_scenes = set()
        for scene in scenes:
            if scene in SceneGroup:
                available_scenes.update(SCENE_IDS[SceneGroup(scene)])
            else:
                available_scenes.add(scene)

        if excluded_scenes is not None:
            available_scenes -= excluded_scenes

        return available_scenes

    @staticmethod
    def _create_task_blueprints(config: EnvConfig) -> list[TaskBlueprint]:
        """
        Create the task blueprints based on the environment mode config.

        Args:
            config (EnvConfig): Environment config.

        Returns:
            list[TaskBlueprint]: List of task blueprints.
        """
        task_blueprints_config = config.tasks.task_blueprints
        task_blueprints = []
        globally_excluded_scenes = set(config.tasks.globally_excluded_scenes)
        for task_blueprint_config in task_blueprints_config:
            task_type = task_blueprint_config.task_type
            if task_type not in ALL_TASKS:
                raise UnknownTaskTypeError(task_type)

            task_blueprints.append(
                TaskBlueprint(
                    task_type=ALL_TASKS[task_type],
                    scenes=ITHOREnv._compute_config_available_scenes(
                        task_blueprint_config.scenes, excluded_scenes=globally_excluded_scenes
                    ),
                    task_args=task_blueprint_config.args,
                )
            )

        if not task_blueprints:
            raise NoTaskBlueprintError()

        return task_blueprints

    @staticmethod
    def _create_ai2thor_controller(config: EnvConfig) -> Controller:
        """
        Initialize the AI2THOR controller.

        Args:
            config (EnvConfig): Environment config.

        Returns:
            controller (Controller): AI2THOR controller.
        """
        # Hardcoded parameters
        controller_parameters = {
            "agentMode": "default",
            "snapToGrid": False,
            "rotateStepDegrees": config.action_discrete_param_values.rotation_degrees,
            "gridSize": config.action_discrete_param_values.movement_magnitude,
        }
        # Parameters from the config
        controller_parameters.update(config.controller_parameters.get_controller_parameters())

        return Controller(**controller_parameters)

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, NDArray[np.uint8] | str], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (dict): Action to take in the environment.

        Returns:
            observation(dict[str, NDArray[np.uint8] | str]): Observation of the environment.
            reward (float): Reward of the action.
            terminated (bool): Whether the agent reaches a terminal state (realized the task).
            truncated (bool): Whether the limit of steps per episode has been reached.
            info (dict): Additional information about the environment.
        """
        # === Get action name, parameters and ai2thor action ===
        action_name = self.action_idx_to_name[action["action_index"]]
        env_action = ACTIONS_BY_NAME[action_name]
        target_object_coordinates: tuple[float, float] | None = action.get("target_object_coordinates")
        action_parameter: float | None = action.get("action_parameter")
        scene_objects_dict = {obj["objectId"]: obj for obj in self.last_event.metadata["objects"]}

        # === Identify the target object if needed for the action ===
        target_object_id, failed_action_event = self._identify_target_object(
            env_action, target_object_coordinates, scene_objects_dict
        )

        # === Perform the action ===
        action_execution_time = None
        if failed_action_event is None:
            start_time = time.perf_counter()

            new_event = env_action.perform(
                env=self,
                action_parameter=action_parameter,
                target_object_id=target_object_id,  # TODO: Create NoObject object
            )

            end_time = time.perf_counter()
            action_execution_time = end_time - start_time
        else:
            new_event = failed_action_event
        new_event.metadata["target_object_id"] = target_object_id
        # TODO: Add logging of the event, especially when the action fails

        self.step_count += 1

        environment_obs: NDArray = new_event.frame  # type: ignore # TODO: Check how to fix this type issue
        observation = self._get_full_observation(environment_obs)

        # Start the timer before the computation
        start_time = time.perf_counter()
        reward, terminated, task_info = self.reward_handler.get_reward(new_event, self.controller.last_action)
        # End the timer after the computation
        end_time = time.perf_counter()
        reward_computation_time = end_time - start_time

        truncated = self.step_count >= self.config.max_episode_steps
        info = {
            "metadata": new_event.metadata,
            "task_info": task_info,
            "is_success": terminated,
            "task_advancement": task_info.get("task_advancement", None),
            # Performance logging
            "speed_performance": {
                "reward_computation_time": reward_computation_time,
                "action_execution_time": action_execution_time,
            },
        }
        self.last_info = info

        return observation, reward, terminated, truncated, info

    def _identify_target_object(
        self,
        env_action: EnvironmentAction,
        target_object_coordinates: tuple[float, float] | None,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> tuple[SimObjId | None, Event | None]:
        """
        Identify the target object the given action (if any) and a failed action event if their is no valid target object.

        Args:
            env_action (EnvironmentAction): Action to perform.
            target_object_coordinates (tuple[float, float] | None): Coordinates of the target object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping object ids to
                their metadata.

        Returns:
            target_object_id (SimObjId | None): Id of the target object.
            failed_action_event (Event | None): Event corresponding to the failed action.
        """
        # No target object case
        if not env_action.has_target_object:
            return None, None

        # Closest object case
        if self.config.action_modifiers.target_closest_object:
            # Look for the closest operable object for the action
            visible_objects_metadata = [
                obj_metadata
                for obj_metadata in scene_objects_dict.values()
                if obj_metadata[SimObjVariableProp.IS_INTERACTABLE]
            ]
            closest_operable_object = min(
                (
                    obj_metadata
                    for obj_metadata in visible_objects_metadata
                    if env_action.is_object_operable(obj_metadata)
                ),
                key=operator.itemgetter("distance"),
                default=None,
            )
            if closest_operable_object is not None:
                return closest_operable_object["objectId"], None
            return None, env_action.fail_perform(
                env=self,
                error_message=f"No operable object found to perform action {env_action.name} in the agent's field of view.",
            )

        # Target coordinate case
        assert target_object_coordinates is not None
        query = self.controller.step(
            action="GetObjectInFrame",
            x=target_object_coordinates[0],
            y=target_object_coordinates[1],
            checkVisible=False,  # TODO: Check if the behavior is correct (object not detected if not visible)
        )
        if bool(query):
            selected_object_id = query.metadata["actionReturn"]
            if env_action.is_object_operable(scene_objects_dict[selected_object_id]):
                return selected_object_id, None

        failed_action = env_action.fail_perform(
            env=self,
            error_message=f"No operable object found at position {target_object_coordinates} to perform action {env_action.name}.",
        )
        return None, failed_action

    # TODO: Adapt this with general task and reward handling
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[np.uint8] | str], dict]:
        """
        Reset the environment.

        New scene is sampled and new task and reward handlers are initialized.
        """
        print("Resetting environment.")
        super().reset(seed=seed, options=options)

        # Sample a task blueprint
        self.task_idx = self.np_random.integers(len(self.task_blueprints))
        task_blueprint = self.task_blueprints[self.task_idx]
        self.current_task_type = task_blueprint.task_type

        self.task = task_blueprint.task_type(**task_blueprint.task_args)
        # TODO: Support more than one reward handler
        self.reward_handler = self.task.get_reward_handler(self.config.no_task_advancement_reward)

        # Reset the controller, task and reward handler
        task_completion, task_info, scene_initialization_time = self._reset_controller_task_reward(task_blueprint)

        self.step_count = 0
        info = {
            "metadata": self.last_event.metadata,
            "task_info": task_info,
            "is_success": task_completion,
            "task_advancement": task_info.get("task_advancement", None),
            "speed_performance": {"scene_initialization_time": scene_initialization_time},
        }

        obs_env: NDArray = self.last_event.frame  # type: ignore
        observation = self._get_full_observation(obs_env)
        print(f"Starting new episode in {self.current_scene} with task {self.current_task_type}.")

        self.last_info = info
        return observation, info

    def _get_full_observation(self, environment_obs: NDArray[np.uint8]) -> dict[str, Any]:
        """
        Get the full observation of the environment.

        Args:
            environment_obs (NDArray[np.uint8]): Observation of the environment (frame of the
                agent's view).

        Returns:
            dict[str, Any]: Full observation of the environment.
        """
        return {
            "env_obs": environment_obs,
            # "task_desc_obs": self.task.text_description(),
            "task_obs": self.task_idx,
            "scene_obs": SCENE_ID_TO_INDEX_MAP[self.current_scene],
        }

    def _reset_controller_task_reward(self, task_blueprint: TaskBlueprint) -> tuple[bool, dict[str, Any], float]:
        """
        Sample a task from the task blueprint compatible with the given event.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint to sample from.

        Returns:
            task_completion (bool): Whether the task is completed.
            task_info (dict[str, Any]): Additional information about the task.
            scene_initialization_time (float): Time taken to initialize the scene.

        """
        successful_reset = False
        # Repeat until a compatible scene is found and remove incompatible ones from the task blueprint
        while not successful_reset:
            print(f"Sampling a scene from the task blueprint {task_blueprint.task_type.__name__}.")
            sorted_scenes = sorted(task_blueprint.scenes)
            sampled_scene = self.np_random.choice(sorted_scenes)
            print(f"Sampled scene: {sampled_scene}.")

            # Instantiate the scene
            start_time = time.perf_counter()

            self.controller.reset(scene=sampled_scene)  # type: ignore
            self._randomize_scene(self.controller)

            end_time = time.perf_counter()
            scene_initialization_time = end_time - start_time
            print(f"Scene {sampled_scene} initialized in {scene_initialization_time:.4f} seconds.")

            successful_reset, task_completion, task_info = self.reward_handler.reset(self.controller)
            if not successful_reset:  # TODO: Fix this for tasks with 0 arguments to work
                print(
                    f"Scene {sampled_scene} is not compatible with the task blueprint {task_blueprint.task_type} ({task_blueprint.task_args}). Removing it from the task blueprint."
                )
                task_blueprint.scenes.remove(sampled_scene)
            elif task_completion:
                print(
                    f"Task is already completed in scene {sampled_scene}. Removing the scene from the task blueprint and sampling a new one."
                )
                task_blueprint.scenes.remove(sampled_scene)
            if not task_blueprint.scenes:
                raise NoCompatibleSceneError(task_blueprint)

        self.current_scene = sampled_scene

        return task_completion, task_info, scene_initialization_time

    # TODO: Check why this is used nowhere
    def _randomize_scene(
        self,
        controller: Controller,
    ) -> None:
        """
        Randomize the scene according to the environment config.

        Args:
            controller (Controller): AI2THOR controller after initializing the scene.
        """
        randomization_config = self.config.scene_randomization
        if randomization_config.random_agent_spawn:
            positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
            sampled_position = self.np_random.choice(positions)
            # Sample int from 0 to 11 and multiply by 30 to get a random rotation
            random_rotation = self.np_random.integers(12) * 30
            controller.step(
                action="Teleport",
                position=sampled_position,
                rotation=random_rotation,
                horizon=0,
                standing=True,
            )
        if randomization_config.random_object_spawn:
            controller.step(
                action="InitialRandomSpawn",
                randomSeed=self.np_random.integers(0, 1000),  # TODO? Add a parameter for the number of different seeds?
                forceVisible=True,  # TODO: Force object to be visible even without randomizing object spawn.
                numPlacementAttempts=15,
                placeStationary=True,
            )
        if randomization_config.random_object_materials:
            controller.step(action="RandomizeMaterials")
        if randomization_config.random_lighting:
            controller.step(
                action="RandomizeLighting",
                synchronized=False,  # TODO: Check we keep this to False
            )
        if randomization_config.random_object_colors:
            controller.step(action="RandomizeColors")

    def close(self) -> None:
        """
        Close the environment.

        In particular, stop the ai2thor controller.
        """
        self.controller.stop()


# %% === Exceptions ===
class NoCompatibleSceneError(ValueError):
    """
    Exception raised when no compatible scene is found to instantiate a task from a task blueprint.

    Either the available scenes are not compatible with the task blueprint or the task is
    simply impossible in the environment.
    """

    def __init__(self, task_blueprint: TaskBlueprint) -> None:
        self.task_blueprint = task_blueprint

    def __str__(self) -> str:
        return f"No compatible scene found to instantiate a task from the task blueprint {self.task_blueprint}.\n Make sure that the task is possible in the environment and that the available scenes are compatible with the task blueprint."


class NoTaskBlueprintError(Exception):
    """Exception raised when no task blueprint is found in the environment mode config."""

    def __str__(self) -> str:
        return f"No task blueprint found in the environment mode config. Task blueprints should be defined in the section 'tasks/task_blueprints' of the config."
