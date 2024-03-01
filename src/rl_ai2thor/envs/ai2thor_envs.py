"""
Gymnasium interface for AI2THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

import ai2thor.controller
import gymnasium as gym
import numpy as np
import yaml
from ai2thor.server import Event

from rl_ai2thor.envs.actions import (
    ACTIONS_BY_CATEGORY,
    ACTIONS_BY_NAME,
    ALL_ACTIONS,
    ActionCategory,
    EnvActionName,
    EnvironmentAction,
)
from rl_ai2thor.envs.reward import GraphTaskRewardHandler
from rl_ai2thor.envs.sim_objects import SimObjFixedProp
from rl_ai2thor.envs.tasks.tasks import GraphTask, PlaceIn, UndefinableTask
from rl_ai2thor.utils.general_utils import ROOT_DIR, update_nested_dict

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from rl_ai2thor.utils.ai2thor_types import EventLike


# %% Environment definitions
class ITHOREnv(gym.Env):
    """Wrapper base class for iTHOR environment."""

    metadata: ClassVar[dict] = {
        "render_modes": ["human"],
        "render_fps": 30,
    }
    # TODO: Check if we keep this

    def __init__(
        self,
        override_config: dict | None = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            override_config (dict, Optional): Dictionary whose keys will override the default config.
        """
        self.config = self._load_and_override_config(override_config)
        self._create_action_space()
        self._create_observation_space()
        self._initialize_ai2thor_controller()
        self._initialize_other_attributes()

    @staticmethod
    def _load_and_override_config(override_config: dict | None = None) -> dict[str, Any]:
        """
        Load and update the config of the environment according to the override config.

        The environment mode config is added to the base config and given keys are overridden.

        Args:
            override_config (dict, Optional): Dictionary whose keys will override the default config.
        """
        config_dir = pathlib.Path(ROOT_DIR, "config")
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
            action_space_dict["target_object_position"] = gym.spaces.Box(low=0, high=1, shape=(2,))

        self.action_space = gym.spaces.Dict(action_space_dict)

    def _create_observation_space(self) -> None:
        controller_parameters = self.config["controller_parameters"]
        resolution = (
            controller_parameters["height"],
            controller_parameters["width"],
        )
        nb_channels = 1 if self.config["grayscale"] else 3
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(resolution[0], resolution[1], nb_channels))

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
        self.task = UndefinableTask()
        self.step_count = 0
        self.np_random = np.random.default_rng(self.config["seed"])

    def step(self, action: dict) -> tuple[ArrayLike, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action (dict): Action to take in the environment.

        Returns:
            observation(ArrayLike): Observation of the environment.
            reward (float): Reward of the action.
            terminated (bool): Whether the agent reaches a terminal state (realized the task).
            truncated (bool): Whether the limit of steps per episode has been reached.
            info (dict): Additional information about the environment.
        """
        # === Get action name, parameters and ai2thor action ===
        action_name = self.action_idx_to_name[action["action_index"]]
        env_action = ACTIONS_BY_NAME[action_name]
        target_object_coordinates: tuple[float, float] | None = action.get("target_object_position")
        action_parameter: float | None = action.get("action_parameter")

        # === Identify the target object if needed for the action ===
        target_object_id, failed_action_event = self._identify_target_object(env_action, target_object_coordinates)

        # === Perform the action ===
        if failed_action_event is None:
            new_event = env_action.perform(
                env=self,
                action_parameter=action_parameter,
                target_object_id=target_object_id,
            )
        else:
            new_event = failed_action_event
        new_event.metadata["target_object_id"] = target_object_id
        # TODO: Add logging of the event, especially when the action fails

        self.step_count += 1

        observation: ArrayLike = new_event.frame  # TODO: Check how to fix this type issue
        reward, terminated, task_info = self.reward_handler.get_reward(new_event)

        truncated = self.step_count >= self.config["max_episode_steps"]
        info = {"metadata": new_event.metadata, "task_info": task_info}

        self.last_event = new_event

        return observation, reward, terminated, truncated, info

    def _identify_target_object(
        self, env_action: EnvironmentAction, target_object_coordinates: tuple[float, float] | None
    ) -> tuple[str | None, EventLike | None]:
        """
        Identify the target object the given action (if any) and a failed action event if their is no valid target object.

        Args:
            env_action (EnvironmentAction): Action to perform.
            target_object_coordinates (tuple[float, float] | None): Coordinates of the target object.

        Returns:
            target_object_id (str | None): Id of the target object.
            failed_action_event (EventLike | None): Event corresponding to the failed action.
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
    def reset(self, seed: int | None = None) -> tuple[ArrayLike, dict]:
        """
        Reset the environment.

        New scene is sampled and new task and reward handlers are initialized.
        """
        print("Resetting environment and starting new episode")
        # TODO: Check that the seed is used correctly
        super().reset(seed=seed)

        # TODO: Add scene id handling
        scenes_list = self.controller.ithor_scenes(
            include_kitchens=True, include_living_rooms=True, include_bedrooms=True, include_bathrooms=True
        )
        sampled_scene = self.np_random.choice(scenes_list)

        # Setup the scene
        self.last_event = self.controller.reset(sampled_scene)
        observation = self.last_event.frame  # type: ignore

        # Initialize the task and reward handler
        self.task = self._sample_task(self.last_event)
        self.reward_handler = GraphTaskRewardHandler(self.task)
        task_completion, task_info = self.reward_handler.reset(self.last_event)

        # TODO: Check if this is correct
        if task_completion:
            self.reset(seed=seed)

        self.step_count = 0
        info = {"metadata": self.last_event.metadata, "task_info": task_info}

        return observation, info

    def close(self) -> None:
        """
        Close the environment.

        In particular, stop the ai2thor controller.
        """
        self.controller.stop()

    # TODO: Implement appropriate task sampling
    def _sample_task(self, event: EventLike) -> GraphTask:
        """
        Sample a task for the environment.

        # TODO: Make it dependant on the scene..?
        """
        # Temporarily return only a PlaceObject task
        # Sample a receptacle and an object to place
        scene_pickupable_objects = [
            obj[SimObjFixedProp.OBJECT_TYPE] for obj in event.metadata["objects"] if obj[SimObjFixedProp.PICKUPABLE]
        ]
        scene_receptacles = [
            obj[SimObjFixedProp.OBJECT_TYPE] for obj in event.metadata["objects"] if obj[SimObjFixedProp.RECEPTACLE]
        ]

        object_to_place = self.np_random.choice(scene_pickupable_objects)
        receptacle = self.np_random.choice(scene_receptacles)

        return PlaceIn(
            placed_object_type=object_to_place,
            receptacle_type=receptacle,
        )


# %% Exceptions
class UnknownActionCategoryError(ValueError):
    """Exception raised for unknown action categories in environment mode config."""

    def __init__(self, action_category: str) -> None:
        self.action_category = action_category
        super().__init__(
            f"Unknown action category {action_category} in environment mode config. "
            "Available action categories are ACTION_CATEGORIES."
        )
