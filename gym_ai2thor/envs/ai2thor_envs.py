"""
Gymnasium interface for ai2thor environment.

Based on the code from cups-rl: https://github.com/TheMTank/cups-rl (MIT License)
# TODO: Check if we keep this

# TODO: Check and add type annotations
"""
from shutil import move
from typing import Optional, Any, Callable
from numpy.typing import ArrayLike

from abc import abstractmethod
from dataclasses import dataclass, field
import dataclasses

import numpy as np

import ai2thor.controller
import ai2thor.server
import gymnasium as gym
import yaml

from utils import update_nested_dict


# %% Action definitions
ACTION_CATEGORIES = set(
    [
        "movement_actions",
        "body_rotation_actions",
        "camera_rotation_actions",
        "crouch_actions",
        "done_actions",
        "hand_movement_actions",
        "pickup_put_actions",
        "drop_actions",
        "throw_actions",
        "push_pull_actions",
        "open_close_actions",
        "toggle_actions",
        "liquid_manipulation_actions",
        "break_actions",
        "slice_actions",
        "use_up_actions",
        "clean_dirty_actions",
    ]
)
ALL_ACTIONS = {
    # Navigation actions (see: https://ai2thor.allenai.org/ithor/documentation/navigation)
    "movement_actions": ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"],
    "body_rotation_actions": ["RotateLeft", "RotateRight"],
    "camera_rotation_actions": ["LookUp", "LookDown"],
    "crouch_actions": ["Crouch", "Stand"],
    # note: "Teleport", "TeleportFull" are not available to the agent
    "done_actions": ["Done"],
    # Object manipulation actions (see: https://ai2thor.allenai.org/ithor/documentation/interactive-physics)
    "hand_movement_actions": [
        "MoveHeldObjectAheadBack",
        "MoveHeldObjectRightLeft",
        "MoveHeldObjectUpDown",
        "RotateHeldObjectRoll",  # Around forward-back axis
        "RotateHeldObjectPitch",  # Around left-right axis
        "RotateHeldObjectYaw",  # Around up-down axis
    ],
    "pickup_put_actions": ["PickupObject", "PutObject"],
    "drop_actions": [
        "DropHandObject"
    ],  # Like throwing but with 0 force, meant to be used in tandem with the Move/Rotate hand movement actions
    "throw_actions": ["ThrowObject"],
    "push_pull_actions": [
        "PushObject",
        "PullObject",
    ],  # Syntactic sugar for DirectionalPush with a pushAngle of 0 and 180 degrees
    # note: "DirectionalPush", "TouchThenApplyForce" are not available because we keep only actions with a single parameter
    # Object interaction actions (see: https://ai2thor.allenai.org/ithor/documentation/object-state-changes)
    "open_close_actions": ["OpenObject", "CloseObject"],
    # ? Keep only OpenObject when action space is continuous?
    "toggle_actions": ["ToggleObject"],
    "liquid_manipulation_actions": ["FillObjectWithLiquid", "EmptyLiquidFromObject"],
    "break_actions": ["BreakObject"],
    "slice_actions": ["SliceObject"],
    "use_up_actions": ["UseUpObject"],
    "clean_dirty_actions": ["DirtyObject", "CleanObject"],
    # note: "CookObject" is not used because it has "magical" effects instead of having contextual effects (like using a toaster to cook bread)
}
PARAMETERIZED_ACTIONS = set(
    [
        "MoveAhead",
        "MoveBack",
        "MoveLeft",
        "MoveRight",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "MoveHeldObjectAheadBack",
        "MoveHeldObjectRightLeft",
        "MoveHeldObjectUpDown",
        "RotateHeldObjectRoll",
        "RotateHeldObjectPitch",
        "RotateHeldObjectYaw",
        "ThrowObject",
        "PushObject",
        "PullObject",
        "OpenObject",
    ]
)
ACTION_TO_REQUIRED_PROPERTY = {
    "PickupObject": "pickupable",
    "PutObject": "receptacle",
    "PushObject": "moveable",
    "PullObject": "moveable",
    "OpenObject": "openable",
    "CloseObject": "openable",
    "BreakObject": "breakable",
    "SliceObject": "sliceable",
    "ToggleObject": "toggleable",
    "FillObjectWithLiquid": "canFillWithLiquid",
    "EmptyLiquidFromObject": "canFillWithLiquid",
    "UseUpObject": "canBeUsedUp",
    "DirtyObject": "dirtyable",
    "CleanObject": "dirtyable",
}
OPENABLE_OBJECTS = {}  # TODO: Add openable objects
BREAKABLE_OBJECTS = {}  # TODO: Add breakable objects
SLICEABLE_OBJECTS = {}  # TODO: Add sliceable objects
TOGGLEABLE_OBJECTS = {}  # TODO: Add toggleable objects
CAN_BE_FILLED_OBJECTS = {}  # TODO: Add fillable objects
CAN_BE_USED_UP_OBJECTS = {}  # TODO: Add usable objects
DIRTYABLE_OBJECTS = {}  # TODO: Add cleanable objects

OBJECTS_BY_PROPERTY = {
    "pickupable": {},
    "receptacle": {},
    "moveable": {},
    "openable": OPENABLE_OBJECTS,
    "breakable": BREAKABLE_OBJECTS,
    "sliceable": SLICEABLE_OBJECTS,
    "toggleable": TOGGLEABLE_OBJECTS,
    "canFillWithLiquid": CAN_BE_FILLED_OBJECTS,
    "canBeUsedUp": CAN_BE_USED_UP_OBJECTS,
    "dirtyable": DIRTYABLE_OBJECTS,
}
# TODO: Delete - Unused


# %% Environment definitions
class ITHOREnv(gym.Env):
    """
    Wrapper base class for iTHOR enviroment.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }
    # TODO: Check if we keep this

    def __init__(
        self,
        custom_config: Optional[dict] = None,  # TODO: Check if we keep this like this
    ) -> None:
        """
        Initialize the environment.

        Args:
            custom_config (dict): Dictionary whose keys will override the default config.
        """
        # === Get full config ===
        with open("config/general.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        # Update config with user config
        if custom_config is not None:
            update_nested_dict(self.config, custom_config)

        with open(f"{self.config['enviroment_mode']}.yaml", "r", encoding="utf-8") as f:
            self.enviroment_mode_config = yaml.safe_load(f)

        # === Get action space ===
        self.action_availablities = {
            action_name: False
            for action_category in ACTION_CATEGORIES
            for action_name in ALL_ACTIONS[action_category]
        }
        # TODO: Check if we keep this or replace with a simple set of available actions
        # Update the available actions with the environment mode config
        for action_category in self.enviroment_mode_config["action_categories"]:
            if action_category in ACTION_CATEGORIES:
                if self.enviroment_mode_config["action_categories"][action_category]:
                    self.action_availablities[action_category] = True
            else:
                raise ValueError(
                    f"Unknown action category in environment mode config: {action_category}"
                )

        if self.enviroment_mode_config["simple_movement_actions"]:
            self.action_availablities["MoveBack"] = False
            self.action_availablities["MoveLeft"] = False
            self.action_availablities["MoveRight"] = False

        # Create action space dictionary
        available_actions = [
            action_name
            for action_name, available in self.action_availablities.items()
            if available
        ]
        self.action_idx_to_name = dict(enumerate(available_actions))
        action_space_dict: dict[str, gym.Space] = {
            "action_index": gym.spaces.Discrete(len(self.action_idx_to_name))
        }
        if not self.config["discrete_actions"]:
            action_space_dict["action_parameter"] = gym.spaces.Box(
                low=0, high=1, shape=(1,)
            )
        if not self.config["target_closest_object"]:
            action_space_dict["target_object_position"] = gym.spaces.Box(
                low=0, high=1, shape=(2,)
            )

        self.action_space = gym.spaces.Dict(action_space_dict)

        # === Get observation space ===
        controller_parameters = self.config["controller_parameters"]
        resolution = (
            controller_parameters["height"],
            controller_parameters["width"],
        )
        nb_channels = 1 if self.config["grayscale"] else 3
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(resolution[0], resolution[1], nb_channels)
        )

        # === Initialize ai2thor controller ===
        self.config["controller_parameters"]["agentMode"] = "default"
        self.controller = ai2thor.controller.Controller(
            *self.config["controller_parameters"],
        )

        # === Other attributes ===
        dummy_metadata = {
            "screenWidth": self.config["controller_parameters"]["width"],
            "screenHeight": self.config["controller_parameters"]["height"],
        }
        self.last_event = ai2thor.server.Event(dummy_metadata)
        # TODO: Check if this is correct ^
        self.held_object = None

    def step(self, action: dict) -> tuple[ArrayLike, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action (dict): Action to take in the environment.

        Returns:
            observation(ArrayLike): Observation of the environment.
            reward (float): Reward of the action.
            done (bool): Whether the episode is done.
            info (dict): Additional information about the environment.
        """
        # === Get action name and parameters ===
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(
                f"Action {action} is not contained in the action space"
            )
        action_name = self.action_idx_to_name[action["action_index"]]
        target_object_coordinates = action.get("target_object_position", None)
        action_parameter = action.get("action_parameter", None)

        # === Identify the target object if needed for the action ===
        if action_name in ACTION_TO_REQUIRED_PROPERTY:
            if self.config["target_closest_object"]:
                # Look for the closest operable object for the action
                visible_objects = [
                    obj for obj in self.last_event.metadata["objects"] if obj["visible"]
                ]
                object_required_property = ACTION_TO_REQUIRED_PROPERTY[action_name]
                closest_operable_object, search_distance = None, np.inf
                for obj in visible_objects:
                    if (
                        obj[object_required_property]
                        and obj["distance"] < search_distance
                    ):
                        closest_operable_object = obj
                        search_distance = obj["distance"]
                if closest_operable_object is not None:
                    target_object_id = closest_operable_object["objectId"]
                else:
                    raise ValueError(
                        f"No object found with property {object_required_property} to perform action {action_name}"
                    )
                    # TODO: Log this event, make action fail and don't raise an error
            else:
                query = self.controller.step(
                    action="GetObjectsInFrame",
                    x=target_object_coordinates[0],
                    y=target_object_coordinates[1],
                    checkVisible=False,  # TODO: Check if the behavior is correct (object not detected if not visible)
                )
                if bool(query):
                    target_object_id = query.metadata["actionReturn"]
                else:
                    raise ValueError(
                        f"No object found at position {target_object_coordinates}"
                    )
                    # TODO: Implement a range of tolerance
                    # TODO: Log this event, make action fail and don't raise an error
        else:
            target_object_id = None
        # === Perform the action ===

    def reset(self, seed: Optional[int] = None) -> tuple[ArrayLike, dict]:
        """
        Reset the environment.

        TODO: Finish this method
        """
        print("Resetting environment and starting new episode")
        super().reset(seed=seed)

        self.last_event = self.controller.reset()
        # TODO: Check if the event is correct
        # TODO: Add scene id handling

        # Setup the scene
        # Chose the task

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def close(self):
        self.controller.stop()


@dataclass
class EnvironmentAction:
    """
    Base class for complex environment actions that correspond to ai2thor actions.

    Attributes:
        name (str): Name of the action in the RL environment.
        ai2thor_action (str): Name of the ai2thor action corresponding to the environment's action.
        has_target_object (bool, optional): Whether the action requires a target object.
        parameter_name (str, optional): Name of the quantitative parameter of the action (if any).
        other_ai2thor_parameters (dict[str, Any], optional): Other ai2thor parameters of the action that take
            a fixed value (e.g. "up" and "right" for MoveHeldObject) and their value.
        config_dependent_parameters (set[str], optional): Set of parameters that depend on the environment config.

    Methods:
        perform():
            Perform the action in the environment and return the event.

    """

    name: str
    ai2thor_action: str
    _: dataclasses.KW_ONLY  # Following arguments are keyword-only
    has_target_object: bool = False
    parameter_name: Optional[str] = None
    other_ai2thor_parameters: dict[str, Any] = field(default_factory=dict)
    config_dependent_parameters: set[str] = field(default_factory=set)

    # TODO: Check if this is correct
    def perform(
        self,
        env: ITHOREnv,
        action_parameter: Optional[float] = None,
        target_object_id: Optional[str] = None,
    ) -> ai2thor.server.Event:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.

        Returns:
            event (ai2thor.controller.Event): Event returned by the controller.
        """

        action_parameters = self.other_ai2thor_parameters.copy()
        if self.parameter_name is not None:
            action_parameters[self.parameter_name] = action_parameter
        if self.has_target_object:
            action_parameters["objectId"] = target_object_id
        for parameter_name in self.config_dependent_parameters:
            action_parameters[parameter_name] = env.config["action_parameters"][
                parameter_name
            ]
        event = env.controller.step(
            action=self.ai2thor_action,
            **action_parameters,
        )
        return event  # type: ignore


@dataclass
class BaseActionCondition:
    """
    Base class for conditions that can be used to determine whether an action
    can be performed in the environment.

    Attributes:
        overriding_message (str, optional): Message to display when the condition
            is not met. If None, a default message can be defined in the
            _base_error_message method.
    """

    overriding_message: Optional[str] = field(default=None, kw_only=True)

    @abstractmethod
    def __call__(self, env: ITHOREnv) -> bool:
        """
        Check whether the condition is met in the environment.

        Args:
            env (ITHOREnv): Environment in which to check the condition.

        Returns:
            condition_met (bool): Whether the condition is met.
        """
        raise NotImplementedError

    def error_message(self, action: EnvironmentAction) -> str:
        """
        Return a message to display when the condition is not met.

        Returns:
            message (str): Message to display.
        """
        if self.overriding_message is not None:
            return self.overriding_message
        return self._base_error_message(action)

    def _base_error_message(self, action: EnvironmentAction) -> str:
        return f"Condition {self.__class__.__name__} not met for action {action.ai2thor_action}!"


@dataclass
class HoldingObjectTypeCondition(BaseActionCondition):
    """
    Condition for actions that require the agent to be holding an object of a
    specific type (e.g. SliceObject requires the agent to hold a knife).

    Attributes:
        object_type (str): Type of object that the agent needs to hold.
    """

    object_type: str

    def __call__(self, env: ITHOREnv) -> bool:
        """
        Check whether the agent is holding an object of the required type.

        Args:
            env (ITHOREnv): Environment in which to check the condition.

        Returns:
            bool: Whether the agent is holding an object of the required type.
        """
        return (
            env.last_event.metadata["inventoryObjects"][0]["objectType"]
            == self.object_type
        )

    def _base_error_message(self, action: EnvironmentAction) -> str:
        return f"Agent needs to hold an object of type {self.object_type} to perform action {action.ai2thor_action}!"


slice_object_condition = HoldingObjectTypeCondition(
    object_type="Knife",
    overriding_message="Agent needs to hold a knife to slice an object!",
)


class VisibleWaterCondition(BaseActionCondition):
    """
    Condition for actions that require the agent to have running water in its
    field of view (e.g. FillObjectWithLiquid).
    """

    def __call__(self, env: ITHOREnv) -> bool:
        """
        Check whether the agent has visible running water in its field of view.

        Args:
            env (ITHOREnv): Environment in which to check the condition.

        Returns:
            bool: Whether the agent has visible running water in its field of view.
        """
        for obj in env.last_event.metadata["objects"]:
            if (
                obj["visible"]
                and obj["isToggledOn"]
                and obj["objectType"] in ["Faucet", "ShowerHead"]
            ):
                return True
        return False

    def _base_error_message(self, action: EnvironmentAction) -> str:
        return f"Agent needs to have visible running water to perform action {action.ai2thor_action}!"


fill_object_with_liquid_condition = VisibleWaterCondition(
    overriding_message="Agent needs to have visible running water to fill an object with liquid!"
)

clean_object_condition = VisibleWaterCondition(
    overriding_message="Agent needs to have visible running water to clean an object!"
)


@dataclass
class ConditionalExecutionAction(EnvironmentAction):
    """
    Class for actions that can only be performed under certain conditions that
    are not natively handled by ai2thor (e.g. SliceObject can only be performed
    if the agent is holding a knife).

    Attributes:
        condition_function (Callable): Function that takes the environment as input
            and returns a boolean indicating whether the action can be performed.
    """

    action_condition: BaseActionCondition

    def perform(
        self,
        env: ITHOREnv,
        action_parameter: Optional[float] = None,
        target_object_id: Optional[str] = None,
    ) -> ai2thor.server.Event:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.

        Returns:
            event (ai2thor.controller.Event): Event returned by the controller.
        """
        if self.action_condition(env):
            event = super().perform(env, action_parameter, target_object_id)
        else:
            event = env.controller.step(action="Done")
            event.metadata["lastAction"] = self.ai2thor_action
            event.metadata["lastActionSuccess"] = False
            event.metadata["errorMessage"] = self.action_condition.error_message(self)

        return event  # type: ignore


move_ahead_action = EnvironmentAction(
    name="MoveAhead",
    ai2thor_action="MoveAhead",
    parameter_name="moveMagnitude",
)
move_back_action = EnvironmentAction(
    name="MoveBack",
    ai2thor_action="MoveBack",
    parameter_name="moveMagnitude",
)
move_left_action = EnvironmentAction(
    name="MoveLeft",
    ai2thor_action="MoveLeft",
    parameter_name="moveMagnitude",
)
move_right_action = EnvironmentAction(
    name="MoveRight",
    ai2thor_action="MoveRight",
    parameter_name="moveMagnitude",
)
rotate_left_action = EnvironmentAction(
    name="RotateLeft",
    ai2thor_action="RotateLeft",
    parameter_name="degrees",
)
rotate_right_action = EnvironmentAction(
    name="RotateRight",
    ai2thor_action="RotateRight",
    parameter_name="degrees",
)
look_up_action = EnvironmentAction(
    name="LookUp",
    ai2thor_action="LookUp",
    parameter_name="degrees",
)
look_down_action = EnvironmentAction(
    name="LookDown",
    ai2thor_action="LookDown",
    parameter_name="degrees",
)
crouch_action = EnvironmentAction(
    name="Crouch",
    ai2thor_action="Crouch",
)
stand_action = EnvironmentAction(
    name="Stand",
    ai2thor_action="Stand",
)
done_action = EnvironmentAction(
    name="Done",
    ai2thor_action="Done",
)
move_held_object_ahead_back_action = EnvironmentAction(
    name="MoveHeldObjectAheadBack",
    ai2thor_action="MoveHeldObject",
    parameter_name="ahead",
    other_ai2thor_parameters={"right": 0, "up": 0},
    config_dependent_parameters={"forceVisible"},
)
move_held_object_right_left_action = EnvironmentAction(
    name="MoveHeldObjectRightLeft",
    ai2thor_action="MoveHeldObject",
    parameter_name="right",
    other_ai2thor_parameters={"ahead": 0, "up": 0},
    config_dependent_parameters={"forceVisible"},
)
move_held_object_up_down_action = EnvironmentAction(
    name="MoveHeldObjectUpDown",
    ai2thor_action="MoveHeldObject",
    parameter_name="up",
    other_ai2thor_parameters={"ahead": 0, "right": 0},
    config_dependent_parameters={"forceVisible"},
)
rotate_held_object_roll_action = EnvironmentAction(
    name="RotateHeldObjectRoll",
    ai2thor_action="RotateHeldObject",
    parameter_name="roll",
    other_ai2thor_parameters={"pitch": 0, "yaw": 0},
)
rotate_held_object_pitch_action = EnvironmentAction(
    name="RotateHeldObjectPitch",
    ai2thor_action="RotateHeldObject",
    parameter_name="pitch",
    other_ai2thor_parameters={"roll": 0, "yaw": 0},
)
rotate_held_object_yaw_action = EnvironmentAction(
    name="RotateHeldObjectYaw",
    ai2thor_action="RotateHeldObject",
    parameter_name="yaw",
    other_ai2thor_parameters={"roll": 0, "pitch": 0},
)
pickup_object_action = EnvironmentAction(
    name="PickupObject",
    ai2thor_action="PickupObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
put_object_action = EnvironmentAction(
    name="PutObject",
    ai2thor_action="PutObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction, manualInteract"},
)
drop_hand_object_action = EnvironmentAction(
    name="DropHandObject",
    ai2thor_action="DropHandObject",
    config_dependent_parameters={"forceAction, placeStationary"},
)
throw_object_action = EnvironmentAction(
    name="ThrowObject",
    ai2thor_action="ThrowObject",
    parameter_name="moveMagnitude",
    config_dependent_parameters={"forceAction"},
)
push_object_action = EnvironmentAction(
    name="PushObject",
    ai2thor_action="PushObject",
    parameter_name="moveMagnitude",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
pull_object_action = EnvironmentAction(
    name="PullObject",
    ai2thor_action="PullObject",
    parameter_name="moveMagnitude",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
open_object_action = EnvironmentAction(
    name="OpenObject",
    ai2thor_action="OpenObject",
    parameter_name="openness",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
close_object_action = EnvironmentAction(
    name="CloseObject",
    ai2thor_action="CloseObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
toggle_object_on_action = EnvironmentAction(
    name="ToggleObjectOn",
    ai2thor_action="ToggleObjectOn",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
toggle_object_off_action = EnvironmentAction(
    name="ToggleObjectOff",
    ai2thor_action="ToggleObjectOff",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
fill_object_with_liquid_action = ConditionalExecutionAction(
    name="FillObjectWithLiquid",
    ai2thor_action="FillObjectWithLiquid",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
    action_condition=fill_object_with_liquid_condition,
)
empty_liquid_from_object_action = EnvironmentAction(
    name="EmptyLiquidFromObject",
    ai2thor_action="EmptyLiquidFromObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
break_object_action = EnvironmentAction(
    name="BreakObject",
    ai2thor_action="BreakObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
slice_object_action = ConditionalExecutionAction(
    name="SliceObject",
    ai2thor_action="SliceObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
    action_condition=slice_object_condition,
)
use_up_object_action = EnvironmentAction(
    name="UseUpObject",
    ai2thor_action="UseUpObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
dirty_object_action = EnvironmentAction(
    name="DirtyObject",
    ai2thor_action="DirtyObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
)
clean_object_action = ConditionalExecutionAction(
    name="CleanObject",
    ai2thor_action="CleanObject",
    has_target_object=True,
    config_dependent_parameters={"forceAction"},
    action_condition=clean_object_condition,
)
