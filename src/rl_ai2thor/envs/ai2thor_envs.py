"""
Gymnasium interface for AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ai2thor.controller
import gymnasium as gym
import numpy as np
import yaml
from ai2thor.server import Event
from numpy.typing import NDArray

from rl_ai2thor.envs.actions import (
    ACTIONS_BY_CATEGORY,
    ACTIONS_BY_NAME,
    ALL_ACTIONS,
    ActionCategory,
    EnvActionName,
    EnvironmentAction,
)
from rl_ai2thor.envs.scenes import SCENE_IDS, SceneGroup, SceneId
from rl_ai2thor.envs.tasks.tasks import ALL_TASKS, BaseTask, TaskBlueprint, UndefinableTask
from rl_ai2thor.utils.general_utils import ROOT_DIR, update_nested_dict

if TYPE_CHECKING:
    from rl_ai2thor.envs.sim_objects import SimObjId


# %% Environment definitions
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

    def __init__(
        self,
        config_folder_path: str | Path = "config",
        override_config: dict | None = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            config_folder_path (str | Path): Relative path to the folder containing the configs.
                The folder should contain a general.yaml file and a folder named environment_modes containing the environment mode configs.
            override_config (dict, Optional): Dictionary whose keys will override the default config.
        """
        self.config = self._load_and_override_config(config_folder_path, override_config)
        self._create_action_space()
        self._create_observation_space()
        self._initialize_ai2thor_controller()
        self._initialize_other_attributes()
        self.task_blueprints = self._create_task_blueprints()

    @staticmethod
    def _load_and_override_config(
        config_folder_path: str | Path, override_config: dict | None = None
    ) -> dict[str, Any]:
        """
        Load and update the config of the environment according to the override config.

        The environment mode config is added to the base config and given keys are overridden.

        Args:
            config_folder_path (str | Path): Relative path to the folder containing the configs.
            override_config (dict, Optional): Dictionary whose keys will override the default config.
        """
        config_dir = Path(ROOT_DIR, config_folder_path)
        general_config_path = config_dir / "general.yaml"
        config = yaml.safe_load(general_config_path.read_text(encoding="utf-8"))

        env_mode_config_path = config_dir / "environment_modes" / f"{config["environment_mode"]}.yaml"
        env_mode_config = yaml.safe_load(env_mode_config_path.read_text(encoding="utf-8"))

        update_nested_dict(config, env_mode_config)

        if override_config is not None:
            update_nested_dict(config, override_config)

        return config

    def _compute_action_availabilities(self) -> dict[EnvActionName, bool]:
        """
        Compute the action availabilities based on the environment mode config.

        Returns:
            dict[EnvActionName, bool]: Dictionary indicating which actions are available.
        """
        action_availabilities = {action.name: False for action in ALL_ACTIONS}

        for action_category in self.config["action_categories"]:
            if action_category not in ActionCategory:
                raise UnknownActionCategoryError(action_category)

            if self.config["action_categories"][action_category]:
                # Enable all actions in the category
                for action in ACTIONS_BY_CATEGORY[action_category]:
                    action_availabilities[action.name] = True

        # Handle specific cases
        if self.config["simple_movement_actions"]:
            for action_name in [EnvActionName.MOVE_BACK, EnvActionName.MOVE_LEFT, EnvActionName.MOVE_RIGHT]:
                action_availabilities[action_name] = False

        if self.config["use_done_action"]:
            action_availabilities[EnvActionName.DONE] = True

        if (
            self.config["partial_openness"]
            and self.config["action_categories"]["open_close_actions"]
            and not self.config["discrete_actions"]
        ):
            for action_name in [EnvActionName.OPEN_OBJECT, EnvActionName.CLOSE_OBJECT]:
                action_availabilities[action_name] = False
            action_availabilities[EnvActionName.PARTIAL_OPEN_OBJECT] = True

        return action_availabilities

    def _create_action_space(self) -> None:
        """Create the action space according to the available action groups in the environment mode config."""
        self.action_availabilities = self._compute_action_availabilities()

        available_actions = [action_name for action_name, available in self.action_availabilities.items() if available]
        self.action_idx_to_name = dict(enumerate(available_actions))

        # Create the action space dictionary
        action_space_dict: dict[str, gym.Space] = {"action_index": gym.spaces.Discrete(len(self.action_idx_to_name))}

        if not self.config["discrete_actions"]:
            action_space_dict["action_parameter"] = gym.spaces.Box(low=0, high=1, shape=())

        if not self.config["target_closest_object"]:
            action_space_dict["target_object_coordinates"] = gym.spaces.Box(low=0, high=1, shape=(2,))

        self.action_space = gym.spaces.Dict(action_space_dict)

    def _create_observation_space(self) -> None:
        controller_parameters = self.config["controller_parameters"]
        resolution = (
            controller_parameters["height"],
            controller_parameters["width"],
        )
        nb_channels = 1 if self.config["grayscale"] else 3
        env_obs_space = gym.spaces.Box(
            low=0, high=255, shape=(resolution[0], resolution[1], nb_channels), dtype=np.uint8
        )
        task_obs_space = gym.spaces.Text(
            max_length=1000, charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,. "
        )
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict({
            "env_obs": env_obs_space,
            "task_obs": task_obs_space,
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

    def _create_task_blueprints(self) -> list[TaskBlueprint]:
        """
        Create the task blueprints based on the environment mode config.

        Returns:
            list[TaskBlueprint]: List of task blueprints.
        """
        tasks_config = self.config["tasks"]
        if not isinstance(tasks_config, list):
            tasks_config = [tasks_config]
        task_blueprints = []
        globally_excluded_scenes = set(self.config["globally_excluded_scenes"])
        for task_description in tasks_config:
            task_type = task_description["type"]
            if task_type not in ALL_TASKS:
                raise UnknownTaskTypeError(task_type)
            task_args = task_description.get("args", {})

            task_blueprints.append(
                TaskBlueprint(
                    task_type=ALL_TASKS[task_type],
                    scenes=self._compute_config_available_scenes(
                        task_description["scenes"], excluded_scenes=globally_excluded_scenes
                    ),
                    task_args=task_args,
                )
            )

        if not task_blueprints:
            raise NoTaskBlueprintError(self.config)

        return task_blueprints

    def _initialize_ai2thor_controller(self) -> None:
        self.config["controller_parameters"]["agentMode"] = "default"
        self.controller = ai2thor.controller.Controller(
            **self.config["controller_parameters"],
        )

    def _initialize_other_attributes(self) -> None:
        dummy_metadata = {
            "screenWidth": self.config["controller_parameters"]["width"],
            "screenHeight": self.config["controller_parameters"]["height"],
        }
        self.last_event = Event(dummy_metadata)
        # TODO: Check if this is correct ^
        self.current_scene = None  # TODO: Replace with an undefined scene of type SceneId
        self.current_task_type = UndefinableTask
        self.task = UndefinableTask()
        self.step_count = 0
        # Initialize gymnasium seed
        super().reset(seed=self.config["seed"])

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

        # === Identify the target object if needed for the action ===
        target_object_id, failed_action_event = self._identify_target_object(env_action, target_object_coordinates)

        # === Perform the action ===
        if failed_action_event is None:
            new_event = env_action.perform(
                env=self,
                action_parameter=action_parameter,
                target_object_id=target_object_id,  # TODO: Create NoObject object
            )
        else:
            new_event = failed_action_event
        new_event.metadata["target_object_id"] = target_object_id
        # TODO: Add logging of the event, especially when the action fails

        self.step_count += 1

        environment_obs: NDArray = new_event.frame  # type: ignore # TODO: Check how to fix this type issue
        observation = {"env_obs": environment_obs, "task_obs": self.task.text_description()}
        reward, terminated, task_info = self.reward_handler.get_reward(new_event)

        truncated = self.step_count >= self.config["max_episode_steps"]
        info = {"metadata": new_event.metadata, "task_info": task_info}

        self.last_event = new_event

        return observation, reward, terminated, truncated, info

    def _identify_target_object(
        self, env_action: EnvironmentAction, target_object_coordinates: tuple[float, float] | None
    ) -> tuple[SimObjId | None, Event | None]:
        """
        Identify the target object the given action (if any) and a failed action event if their is no valid target object.

        Args:
            env_action (EnvironmentAction): Action to perform.
            target_object_coordinates (tuple[float, float] | None): Coordinates of the target object.

        Returns:
            target_object_id (SimObjId | None): Id of the target object.
            failed_action_event (Event | None): Event corresponding to the failed action.
        """
        # No target object case
        if not env_action.has_target_object:
            return None, None

        # Closest object case
        if self.config["target_closest_object"]:
            # Look for the closest operable object for the action
            visible_objects = [obj for obj in self.last_event.metadata["objects"] if obj["visible"]]
            closest_operable_object = min(
                (obj for obj in visible_objects if obj[env_action.object_required_property]),
                key=lambda obj: obj["distance"],
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
            return query.metadata["actionReturn"], None
        return None, env_action.fail_perform(
            env=self,
            error_message=f"No object found at position {target_object_coordinates} to perform action {env_action.name}.",
        )

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
        task_blueprint = self.task_blueprints[self.np_random.choice(len(self.task_blueprints))]
        self.current_task_type = task_blueprint.task_type

        # Initialize controller and sample task
        self.last_event, self.task = self._initialize_controller_and_task(task_blueprint)

        # TODO: Support more than one reward handler
        self.reward_handler = self.task.get_reward_handler()
        task_completion, task_info = self.reward_handler.reset(self.controller)

        # TODO: Check if this is correct
        if task_completion:
            self.reset()

        self.step_count = 0
        info = {"metadata": self.last_event.metadata, "task_info": task_info}

        obs_env: NDArray = self.last_event.frame  # type: ignore
        observation = {"env_obs": obs_env, "task_obs": self.task.text_description()}
        print(
            f"Resetting environment and starting new episode in {self.current_scene} with task {self.current_task_type}."
        )

        return observation, info

    def _initialize_controller_and_task(self, task_blueprint: TaskBlueprint) -> tuple[Event, BaseTask]:
        """
        Sample a task from the task blueprint compatible with the given event.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint to sample from.

        Returns:
            initial_event (Event): Initial event of the environment.
            task (BaseTask): Sampled task.
        """
        compatible_arguments = []
        # Repeat until a compatible scene is found and remove incompatible ones from the task blueprint
        while not compatible_arguments:
            print(f"Sampling a scene from the task blueprint {task_blueprint.task_type.__name__}.")
            sorted_scenes = sorted(task_blueprint.scenes)
            sampled_scene = self.np_random.choice(sorted_scenes)
            print(f"Sampled scene: {sampled_scene}.")
            # Instantiate the scene
            controller_parameters = self.config["controller_parameters"]
            controller_parameters["scene"] = sampled_scene
            initial_event: Event = self.controller.reset(sampled_scene)  # type: ignore

            compatible_arguments = task_blueprint.compute_compatible_task_args(event=initial_event)
            if not compatible_arguments:  # TODO: Fix this for tasks with 0 arguments to work
                print(f"No compatible arguments found for scene {sampled_scene}. Removing it from the task blueprint.")
                task_blueprint.scenes.remove(sampled_scene)
                if not task_blueprint.scenes:
                    raise NoCompatibleSceneError(task_blueprint)
        sorted_compatible_arguments = sorted(compatible_arguments)
        sampled_task_args = sorted_compatible_arguments[self.np_random.choice(len(compatible_arguments))]
        print(f"Sampled task arguments: {sampled_task_args}.")

        self.current_scene = sampled_scene
        self.current_task_args = sampled_task_args

        return initial_event, task_blueprint.task_type(*sampled_task_args)

    def close(self) -> None:
        """
        Close the environment.

        In particular, stop the ai2thor controller.
        """
        self.controller.stop()


# %% Exceptions
class UnknownActionCategoryError(ValueError):
    """Exception raised for unknown action categories in environment mode config."""

    def __init__(self, action_category: str) -> None:
        self.action_category = action_category
        super().__init__(
            f"Unknown action category '{action_category}' in environment mode config. "
            f"Available action categories are {[category.value for category in ActionCategory]}."
        )


class UnknownTaskTypeError(ValueError):
    """Exception raised for unknown task types in environment mode config."""

    def __init__(self, task_type: str) -> None:
        self.task_type = task_type
        super().__init__(
            f"Unknown task type '{task_type}' in environment mode config."
            f"Available tasks are {list(ALL_TASKS.keys())}."
            f"If you have defined a new task, make sure to add it to the ALL_TASKS dictionary of the envs.tasks.tasks module."
        )


class NoTaskBlueprintError(Exception):
    """Exception raised when no task blueprint is found in the environment mode config."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def __str__(self) -> str:
        return f"No task blueprint found in the environment mode config. Task blueprints should be defined in config['tasks']. Current config: {self.config}."


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
