"""
Gymnasium interface for AI2THOR RL environment.

TODO: Finish module docstrings.
"""

import pathlib
from typing import ClassVar

import ai2thor.controller
import gymnasium as gym
import numpy as np
import yaml
from ai2thor.server import Event
from numpy.typing import ArrayLike

from rl_ai2thor.envs.actions import (
    ACTION_CATEGORIES,
    ACTIONS_BY_CATEGORY,
    ACTIONS_BY_NAME,
    ALL_ACTIONS,
    EnvironmentAction,
)
from rl_ai2thor.envs.reward import GraphTaskRewardHandler
from rl_ai2thor.envs.tasks import GraphTask, PlaceObject
from rl_ai2thor.utils.ai2thor_types import EventLike
from rl_ai2thor.utils.general_utils import ROOT_DIR, update_nested_dict


# %% Exceptions
class UnknownActionCategoryError(ValueError):
    """Exception raised for unknown action categories in environment mode config."""

    def __init__(self, action_category: str) -> None:
        self.action_category = action_category
        super().__init__(
            f"Unknown action category {action_category} in environment mode config. "
            f"Available action categories are ACTION_CATEGORIES."
        )


# TODO: Check the covariance problems in the type hints
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
        self._load_full_config(override_config)
        self._create_action_space()
        self._create_observation_space()
        self._initialize_ai2thor_controller()
        self._initialize_other_attributes()

    def _load_full_config(self, override_config: dict | None = None) -> None:
        """
        Load and update the config of the environment according to the override config.

        The environment mode config is added to the base config and given keys are overridden.

        Args:
            override_config (dict, Optional): Dictionary whose keys will override the default config.
        """
        config_dir = pathlib.Path(ROOT_DIR, "config")
        with (config_dir / "general.yaml").open(encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        with (config_dir / "environment_modes" / f"{self.config['environment_mode']}.yaml").open(encoding="utf-8") as f:
            enviroment_mode_config = yaml.safe_load(f)
        update_nested_dict(self.config, enviroment_mode_config)

        if override_config is not None:
            update_nested_dict(self.config, override_config)

    def _create_action_space(self) -> None:
        """Create the action space accoding to the available action groups in the environment mode config."""
        self.action_availablities = {action.name: False for action in ALL_ACTIONS}

        # Get the available actions from the environment mode config
        for action_category in self.config["action_categories"]:
            if action_category in ACTION_CATEGORIES:
                if self.config["action_categories"][action_category]:
                    # Enable all actions in the category
                    for action in ACTIONS_BY_CATEGORY[action_category]:
                        self.action_availablities[action.name] = True
            else:
                raise UnknownActionCategoryError(action_category)

        # Handle specific cases
        # Simple movement actions
        if self.config["simple_movement_actions"]:
            self.action_availablities["MoveBack"] = False
            self.action_availablities["MoveLeft"] = False
            self.action_availablities["MoveRight"] = False
        # Done actions
        if self.config["use_done_action"]:
            self.action_availablities["Done"] = True
        # Partial openness
        if (
            self.config["partial_openness"]
            and self.config["action_categories"]["open_close_actions"]
            and not self.config["discrete_actions"]
        ):
            self.action_availablities["OpenObject"] = False
            self.action_availablities["CloseObject"] = False

        available_actions = [action_name for action_name, available in self.action_availablities.items() if available]
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
        self.task = None
        self.step_count = 0

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
        target_object_id = None
        failed_action_event = None
        if env_action.has_target_object:
            if self.config["target_closest_obejct"]:
                # Look for the closest operable object for the action
                visible_objects = [obj for obj in self.last_event.metadata["objects"] if obj["visible"]]
                object_required_property = env_action.object_required_property
                closest_operable_object, search_distance = None, np.inf
                for obj in visible_objects:
                    if obj[object_required_property] and obj["distance"] < search_distance:
                        closest_operable_object = obj
                        search_distance = obj["distance"]
                if closest_operable_object is not None:
                    target_object_id = closest_operable_object["objectId"]
                else:
                    failed_action_event = env_action.fail_perform(
                        env=self,
                        error_message=f"No operable object found to perform action {env_action.name} in the agent's field of view.",
                    )
            else:
                assert target_object_coordinates is not None
                query = self.controller.step(
                    action="GetObjectInFrame",
                    x=target_object_coordinates[0],
                    y=target_object_coordinates[1],
                    checkVisible=False,  # TODO: Check if the behavior is correct (object not detected if not visible)
                )
                if bool(query):
                    target_object_id = query.metadata["actionReturn"]
                else:
                    failed_action_event = env_action.fail_perform(
                        env=self,
                        error_message=f"No object found at position {target_object_coordinates} to perform action {env_action.name}.",
                    )
                    # TODO: Implement a range of tolerance

        return target_object_id, failed_action_event

    # TODO: Adapt this with general task and reward handling
    def reset(self, seed: int | None = None) -> tuple[ArrayLike, dict]:
        """
        Reset the environment.

        New scene is sampled and new task and reward handlers are initialized.
        """
        print("Resetting environment and starting new episode")
        super().reset(seed=seed)

        # Setup the scene
        self.last_event = self.controller.reset("FloorPlan301")
        observation = self.last_event.frame  # type: ignore

        # TODO: Add scene id handling

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

    # TODO: Implement approprate task sampling
    def _sample_task(self, event: EventLike) -> GraphTask:
        """
        Sample a task for the environment.

        # TODO: Make it dependant on the scene..?
        """
        # Temporarily return only a PlaceObject task
        # Sample a receptacle and an object to place
        scene_pickupable_objects = [obj["objectType"] for obj in event.metadata["objects"] if obj["pickupable"]]
        scene_receptacles = [obj["objectType"] for obj in event.metadata["objects"] if obj["receptacle"]]

        np_rng: np.random.Generator = self._np_random  # type: ignore
        object_to_place = np_rng.choice(scene_pickupable_objects)
        receptacle = np_rng.choice(scene_receptacles)

        return PlaceObject(
            placed_object_type=object_to_place,
            receptacle_type=receptacle,
        )


# %%
