"""
Module for actions for AI2-THOR RL Environment, interfaces between the agent and the ai2thor controller.

This module provides classes and definitions for handling actions, conditions, and interactions within the AI2-THOR simulated environment

Classes:
- EnvironmentAction: Base class for complex environment actions.
- BaseActionCondition: Base class for conditions determining if an action can be performed.
- ConditionalExecutionAction: Class for actions that can only be performed under certain conditions.
- VisibleWaterCondition: Condition for actions requiring visible running water.
- HoldingObjectTypeCondition: Condition for actions requiring the agent to hold a specific object type.


Constants:
- ALL_ACTIONS: List of all defined actions.
- ACTION_CATEGORIES: Set of unique action categories.
- ACTIONS_BY_CATEGORY: Dictionary mapping action categories to corresponding actions.
- ACTIONS_BY_NAME: Dictionary mapping action names to their corresponding definitions.

TODO: Update the module docstring
"""

# %% === Imports ===
from __future__ import annotations

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_ai2thor.envs.sim_objects import (
    OPENABLES,
    WATER_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjMetadata,
    SimObjVariableProp,
)
from rl_ai2thor.utils.general_utils import nested_dict_get

if TYPE_CHECKING:
    from ai2thor.server import Event

    from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
    from rl_ai2thor.envs.sim_objects import SimObjId


# %% == Enums ==
class EnvActionName(StrEnum):
    """Enum for environment actions."""

    MOVE_AHEAD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    MOVE_LEFT = "MoveLeft"
    MOVE_RIGHT = "MoveRight"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    LOOK_UP = "LookUp"
    LOOK_DOWN = "LookDown"
    CROUCH = "Crouch"
    STAND = "Stand"
    DONE = "Done"  # TODO: Check if we keep this action
    MOVE_HELD_OBJECT_AHEAD_BACK = "MoveHeldObjectAheadBack"
    MOVE_HELD_OBJECT_RIGHT_LEFT = "MoveHeldObjectRightLeft"
    MOVE_HELD_OBJECT_UP_DOWN = "MoveHeldObjectUpDown"
    ROTATE_HELD_OBJECT_ROLL = "RotateHeldObjectRoll"
    ROTATE_HELD_OBJECT_PITCH = "RotateHeldObjectPitch"
    ROTATE_HELD_OBJECT_YAW = "RotateHeldObjectYaw"
    PICKUP_OBJECT = "PickupObject"
    PUT_OBJECT = "PutObject"
    DROP_HAND_OBJECT = "DropHandObject"
    THROW_OBJECT = "ThrowObject"
    PUSH_OBJECT = "PushObject"
    PULL_OBJECT = "PullObject"
    OPEN_OBJECT = "OpenObject"
    CLOSE_OBJECT = "CloseObject"
    PARTIAL_OPEN_OBJECT = "PartialOpenObject"
    TOGGLE_OBJECT_ON = "ToggleObjectOn"
    TOGGLE_OBJECT_OFF = "ToggleObjectOff"
    FILL_OBJECT_WITH_LIQUID = "FillObjectWithLiquid"
    EMPTY_LIQUID_FROM_OBJECT = "EmptyLiquidFromObject"
    BREAK_OBJECT = "BreakObject"
    SLICE_OBJECT = "SliceObject"
    USE_UP_OBJECT = "UseUpObject"
    DIRTY_OBJECT = "DirtyObject"
    CLEAN_OBJECT = "CleanObject"


class Ai2thorAction(StrEnum):
    """Enum for ai2thor actions."""

    MOVE_AHEAD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    MOVE_LEFT = "MoveLeft"
    MOVE_RIGHT = "MoveRight"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    LOOK_UP = "LookUp"
    LOOK_DOWN = "LookDown"
    CROUCH = "Crouch"
    STAND = "Stand"
    DONE = "Done"  # TODO: Check if we keep this action
    MOVE_HELD_OBJECT = "MoveHeldObject"
    ROTATE_HELD_OBJECT = "RotateHeldObject"
    PICKUP_OBJECT = "PickupObject"
    PUT_OBJECT = "PutObject"
    DROP_HAND_OBJECT = "DropHandObject"
    THROW_OBJECT = "ThrowObject"
    PUSH_OBJECT = "PushObject"
    PULL_OBJECT = "PullObject"
    OPEN_OBJECT = "OpenObject"
    CLOSE_OBJECT = "CloseObject"
    TOGGLE_OBJECT_ON = "ToggleObjectOn"
    TOGGLE_OBJECT_OFF = "ToggleObjectOff"
    FILL_OBJECT_WITH_LIQUID = "FillObjectWithLiquid"
    EMPTY_LIQUID_FROM_OBJECT = "EmptyLiquidFromObject"
    BREAK_OBJECT = "BreakObject"
    SLICE_OBJECT = "SliceObject"
    USE_UP_OBJECT = "UseUpObject"
    DIRTY_OBJECT = "DirtyObject"
    CLEAN_OBJECT = "CleanObject"

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class ActionCategory(StrEnum):
    """Enum for action categories."""

    MOVEMENT_ACTIONS = "movement_actions"
    BODY_ROTATION_ACTIONS = "body_rotation_actions"
    CAMERA_ROTATION_ACTIONS = "camera_rotation_actions"
    CROUCH_ACTIONS = "crouch_actions"
    DONE_ACTIONS = "done_actions"  # TODO: Check if we keep this action category
    HAND_MOVEMENT_ACTIONS = "hand_movement_actions"
    PICKUP_PUT_ACTIONS = "pickup_put_actions"
    DROP_ACTIONS = "drop_actions"
    THROW_ACTIONS = "throw_actions"
    PUSH_PULL_ACTIONS = "push_pull_actions"
    OPEN_CLOSE_ACTIONS = "open_close_actions"
    TOGGLE_ACTIONS = "toggle_actions"
    LIQUID_MANIPULATION_ACTIONS = "liquid_manipulation_actions"
    BREAK_ACTIONS = "break_actions"
    SLICE_ACTIONS = "slice_actions"
    USE_UP_ACTIONS = "use_up_actions"
    CLEAN_DIRTY_ACTIONS = "clean_dirty_actions"
    SPECIAL = "_special"  # For categories that shouldn't be directly enabled from config


# === Action Classes ===
# TODO? Change perform to not need the environment?
# TODO? Add target value to object required property?
# TODO: Create a separate class for action with target objects (and then object_required_property too)
@dataclass(frozen=True)
class EnvironmentAction:
    """
    Base class for complex environment actions that correspond to ai2thor actions.

    Attributes:
        name (EnvActionName): Name of the action in the RL environment.
        ai2thor_action (Ai2thorAction): Name of the ai2thor action corresponding to the
            environment's action.
        action_category (ActionCategory): Category of the action (e.g. movement_actions
            for MoveAhead).
        has_target_object (bool, optional): Whether the action requires a target
            object.
        _object_required_property (SimObjFixedProp, optional): Name of the required property
            of the target object.
        _parameter_name (str, optional): Name of the quantitative parameter of
            the action.
        _parameter_range (tuple[float, float], optional): Range of the quantitative
            parameter of the action. Can be overridden by the config.
        _parameter_discrete_value (float, optional): Value of the quantitative
            parameter of the action in discrete environment mode. Can be
            overridden by the config.
        _other_ai2thor_parameters (dict[str, Any], optional): Other ai2thor
            parameters of the action that take a fixed value (e.g. "up" and
            "right" for MoveHeldObject) and their value.
        _config_dependent_parameters (set[str], optional): Set of parameters
            that depend on the environment config.

    Methods:
        perform(
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.
        ) -> Event:
            Perform the action in the environment and return the event.

        fail_perform(
            env (ITHOREnv): Environment in which the action was performed.
            error_message (str): Error message to log in the event.
        ) -> Event:
            Generate an event corresponding to the failure of the action.
    """

    name: EnvActionName
    ai2thor_action: Ai2thorAction
    action_category: ActionCategory
    _: dataclasses.KW_ONLY  # Following arguments are keyword-only
    has_target_object: bool = False
    _object_required_property: SimObjFixedProp | None = None
    _parameter_name: str | None = None
    _parameter_range: tuple[float, float] | None = None
    _parameter_discrete_value: float | None = None
    _other_ai2thor_parameters: dict[str, Any] = field(default_factory=dict)
    _config_dependent_parameters: set[str] = field(default_factory=set)

    def is_object_operable(self, obj_metadata: SimObjMetadata) -> bool:
        """
        Return whether the object is operable by the action.

        Args:
            obj_metadata (SimObjMetadata): Metadata of the object to check.

        Returns:
            is_operable (bool): Whether the object is operable by the action.
        """
        if self._object_required_property is None:
            return True
        return obj_metadata[self._object_required_property]

    def perform(
        self,
        env: ITHOREnv,
        action_parameter: float | None = None,
        target_object_id: SimObjId | None = None,
    ) -> Event:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (SimObjId, optional): ID of the target object for the action.

        Returns:
            event (Event): Event returned by the controller.
        """
        action_parameters = self._other_ai2thor_parameters.copy()
        if self._parameter_name is not None:
            # Deal with the discrete/continuous environment mode
            if action_parameter is None:
                assert self._parameter_discrete_value is not None
                assert env.config["discrete_actions"]
                # Override the action parameter with the value from the config
                action_parameter = nested_dict_get(
                    d=env.config,
                    keys=["action_parameter_data", self.name, "discrete_value"],
                    default=self._parameter_discrete_value,
                )
            elif self._parameter_range is not None:
                # Override the range with the value from the config
                parameter_range = nested_dict_get(
                    d=env.config,
                    keys=["action_parameter_data", self.name, "range"],
                    default=self._parameter_range,
                )
                action_parameter = parameter_range[0] + action_parameter * (parameter_range[1] - parameter_range[0])
            else:
                raise MissingParameterRangeError(self)

            action_parameters[self._parameter_name] = action_parameter
        if self.has_target_object:
            action_parameters["objectId"] = target_object_id
        for parameter_name in self._config_dependent_parameters:
            action_parameters[parameter_name] = env.config["action_parameters"][parameter_name]

        event: Event = env.controller.step(action=self.ai2thor_action, **action_parameters)  # type: ignore
        return event

    def fail_perform(
        self,
        env: ITHOREnv,
        error_message: str,
    ) -> Event:
        """
        Generate an event corresponding to the failure of the action.

        Args:
            env (ITHOREnv): Environment in which the action was performed.
            error_message (str): Error message to log in the event.

        Returns:
            event (Event): Event for the failed action.
        """
        event: Event = env.controller.step(action="Done")  # type: ignore
        event.metadata["lastAction"] = self.ai2thor_action
        event.metadata["lastActionSuccess"] = False
        event.metadata["errorMessage"] = error_message
        return event

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class BaseActionCondition:
    """
    Base class for action conditions.

    Action conditions can be used to determine whether an action can
    be performed according to the current state of the environment.

    Attributes:
        overriding_message (str, optional): Message to display when the condition
            is not met. If None, a default message can be defined in the
            _base_error_message method.
    """

    overriding_message: str | None = field(default=None, kw_only=True)

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


@dataclass(frozen=True)
class ConditionalExecutionAction(EnvironmentAction):
    """
    Base class for actions that can only be performed under certain conditions.

    Actions that inherit from this class add conditions that are not natively
    handled by ai2thor (e.g. SliceObject can only be performed
    if the agent is holding a knife).

    Attributes:
        condition_function (Callable): Function that takes the environment as input
            and returns a boolean indicating whether the action can be successfully
            performed.
    """

    action_condition: BaseActionCondition

    def perform(
        self,
        env: ITHOREnv,
        action_parameter: float | None = None,
        target_object_id: SimObjId | None = None,
    ) -> Event:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (SimObjId, optional): ID of the target object for the action.

        Returns:
            event (Event): Event returned by the controller.
        """
        event = (
            super().perform(env, action_parameter, target_object_id)
            if self.action_condition(env)
            else self.fail_perform(env, error_message=self.action_condition.error_message(self))
        )
        return event

    # We need to redefine the hash manually because of dataclass behavior
    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class VisibleWaterCondition(BaseActionCondition):
    """
    Check whether the agent has visible running water in its field of view.

    Used for FillObjectWithLiquid and CleanObject.
    """

    def __call__(self, env: ITHOREnv) -> bool:
        """
        Check whether the agent has visible running water in its field of view.

        Args:
            env (ITHOREnv): Environment in which to check the condition.

        Returns:
            water_is_visible (bool): Whether the agent has visible running water in its field of view.
        """
        water_is_visible = any(
            (
                obj[SimObjVariableProp.VISIBLE]
                and obj[SimObjVariableProp.IS_TOGGLED]
                and obj[SimObjFixedProp.OBJECT_TYPE] in WATER_SOURCES
            )
            for obj in env.last_event.metadata["objects"]
        )
        return water_is_visible

    def _base_error_message(self, action: EnvironmentAction) -> str:  # noqa: PLR6301
        """Return the default error message for the condition."""
        return f"Agent needs to have visible running water to perform action {action.ai2thor_action}!"


@dataclass
class HoldingObjectTypeCondition(BaseActionCondition):
    """
    Check whether the agent is holding an object of a specific type.

    Used for SliceObject.

    Attributes:
        object_type (SimObjectType): Type of object that the agent needs to hold.
    """

    object_type: SimObjectType

    def __call__(self, env: ITHOREnv) -> bool:
        """
        Check whether the agent is holding an object of the required type.

        Args:
            env (ITHOREnv): Environment in which to check the condition.

        Returns:
            object_is_held (bool): Whether the agent is holding an object of the required type.
        """
        object_is_held = (
            len(env.last_event.metadata["inventoryObjects"]) > 0
            and env.last_event.metadata["inventoryObjects"][0][SimObjFixedProp.OBJECT_TYPE] == self.object_type
        )
        return object_is_held

    def _base_error_message(self, action: EnvironmentAction) -> str:
        """Return the default error message for the condition."""
        return f"Agent needs to hold an object of type {self.object_type} to perform action {action.name} ({action.ai2thor_action} in ai2thor)!"


fill_object_with_liquid_condition = VisibleWaterCondition(
    overriding_message="Agent needs to have visible running water to fill an object with liquid!"
)
clean_object_condition = VisibleWaterCondition(
    overriding_message="Agent needs to have visible running water to clean an object!"
)
slice_object_condition = HoldingObjectTypeCondition(
    object_type=SimObjectType.KNIFE,
    overriding_message="Agent needs to hold a knife to slice an object!",
)
# === Type Annotations ===
fill_object_with_liquid_condition: VisibleWaterCondition
clean_object_condition: VisibleWaterCondition
slice_object_condition: HoldingObjectTypeCondition


# %% === Specific action classes ===
# TODO: Create a specific class for each action group?
class OpenCloseEnvAction(EnvironmentAction):
    """
    Base class for "OpenObject" and "CloseObject" actions.

    We need a specific class for this action because the object type needs to be in the OPENABLES
    list to avoid TimeoutErrors when trying to open or close objects like `Blinds`.
    """

    def __init__(self, name: EnvActionName, ai2thor_action: Ai2thorAction) -> None:
        super().__init__(
            name=name,
            ai2thor_action=ai2thor_action,
            action_category=ActionCategory.OPEN_CLOSE_ACTIONS,
            has_target_object=True,
            _object_required_property=SimObjFixedProp.OPENABLE,
            _config_dependent_parameters={"forceAction"},
        )

    def is_object_operable(self, obj_metadata: SimObjMetadata) -> bool:
        """
        Return whether the object is operable by the "OpenObject" and "CloseObject" actions in ai2thor.

        We need to add the check for the object type to be in the OPENABLES list because `Blinds`
        are considered openable by ai2thor but they cause a TimeoutError when trying to open or
        close them.

        Args:
            obj_metadata (SimObjMetadata): Metadata of the object to check.

        Returns:
            is_operable (bool): Whether the object is operable by the action.
        """
        obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
        return obj_type in OPENABLES and super().is_object_operable(obj_metadata)


# %% == Actions definitions ==
# TODO: Use task enums to define object required properties
# Navigation actions (see: https://ai2thor.allenai.org/ithor/documentation/navigation)
move_ahead_action = EnvironmentAction(
    name=EnvActionName.MOVE_AHEAD,
    ai2thor_action=Ai2thorAction.MOVE_AHEAD,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    _parameter_name="moveMagnitude",
    _parameter_range=(0, 1),
    _parameter_discrete_value=0.25,
)
move_back_action = EnvironmentAction(
    name=EnvActionName.MOVE_BACK,
    ai2thor_action=Ai2thorAction.MOVE_BACK,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    _parameter_name="moveMagnitude",
    _parameter_range=(0, 1),
    _parameter_discrete_value=0.25,
)
move_left_action = EnvironmentAction(
    name=EnvActionName.MOVE_LEFT,
    ai2thor_action=Ai2thorAction.MOVE_LEFT,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    _parameter_name="moveMagnitude",
    _parameter_range=(0, 1),
    _parameter_discrete_value=0.25,
)
move_right_action = EnvironmentAction(
    name=EnvActionName.MOVE_RIGHT,
    ai2thor_action=Ai2thorAction.MOVE_RIGHT,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    _parameter_name="moveMagnitude",
    _parameter_range=(0, 1),
    _parameter_discrete_value=0.25,
)
rotate_left_action = EnvironmentAction(
    name=EnvActionName.ROTATE_LEFT,
    ai2thor_action=Ai2thorAction.ROTATE_LEFT,
    action_category=ActionCategory.BODY_ROTATION_ACTIONS,
    _parameter_name="degrees",
    _parameter_range=(0, 180),
    _parameter_discrete_value=45,
)
rotate_right_action = EnvironmentAction(
    name=EnvActionName.ROTATE_RIGHT,
    ai2thor_action=Ai2thorAction.ROTATE_RIGHT,
    action_category=ActionCategory.BODY_ROTATION_ACTIONS,
    _parameter_name="degrees",
    _parameter_range=(0, 180),
    _parameter_discrete_value=45,
)
look_up_action = EnvironmentAction(
    name=EnvActionName.LOOK_UP,
    ai2thor_action=Ai2thorAction.LOOK_UP,
    action_category=ActionCategory.CAMERA_ROTATION_ACTIONS,
    _parameter_name="degrees",
    _parameter_range=(0, 90),
    _parameter_discrete_value=30,
)
look_down_action = EnvironmentAction(
    name=EnvActionName.LOOK_DOWN,
    ai2thor_action=Ai2thorAction.LOOK_DOWN,
    action_category=ActionCategory.CAMERA_ROTATION_ACTIONS,
    _parameter_name="degrees",
    _parameter_range=(0, 90),
    _parameter_discrete_value=30,
)
crouch_action = EnvironmentAction(
    name=EnvActionName.CROUCH,
    ai2thor_action=Ai2thorAction.CROUCH,
    action_category=ActionCategory.CROUCH_ACTIONS,
)
stand_action = EnvironmentAction(
    name=EnvActionName.STAND,
    ai2thor_action=Ai2thorAction.STAND,
    action_category=ActionCategory.CROUCH_ACTIONS,
)
done_action = EnvironmentAction(
    name=EnvActionName.DONE,
    ai2thor_action=Ai2thorAction.DONE,
    action_category=ActionCategory.DONE_ACTIONS,
)
# Note: "Teleport", "TeleportFull" are not available to the agent
# Object manipulation actions (see: https://ai2thor.allenai.org/ithor/documentation/interactive-physics)
move_held_object_ahead_back_action = EnvironmentAction(
    name=EnvActionName.MOVE_HELD_OBJECT_AHEAD_BACK,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    _parameter_name="ahead",
    _parameter_range=(-0.5, 0.5),
    _other_ai2thor_parameters={"right": 0, "up": 0},
    _config_dependent_parameters={"forceVisible"},
)
move_held_object_right_left_action = EnvironmentAction(
    name=EnvActionName.MOVE_HELD_OBJECT_RIGHT_LEFT,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    _parameter_name="right",
    _parameter_range=(-0.5, 0.5),
    _other_ai2thor_parameters={"ahead": 0, "up": 0},
    _config_dependent_parameters={"forceVisible"},
)
move_held_object_up_down_action = EnvironmentAction(
    name=EnvActionName.MOVE_HELD_OBJECT_UP_DOWN,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    _parameter_name="up",
    _parameter_range=(-0.5, 0.5),
    _other_ai2thor_parameters={"ahead": 0, "right": 0},
    _config_dependent_parameters={"forceVisible"},
)
rotate_held_object_roll_action = EnvironmentAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_ROLL,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    _parameter_name="roll",
    _parameter_range=(-180, 180),
    _other_ai2thor_parameters={"pitch": 0, "yaw": 0},
)
rotate_held_object_pitch_action = EnvironmentAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_PITCH,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    _parameter_name="pitch",
    _parameter_range=(-180, 180),
    _other_ai2thor_parameters={"roll": 0, "yaw": 0},
)
rotate_held_object_yaw_action = EnvironmentAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_YAW,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    _parameter_name="yaw",
    _parameter_range=(-180, 180),
    _other_ai2thor_parameters={"roll": 0, "pitch": 0},
)
pickup_object_action = EnvironmentAction(
    name=EnvActionName.PICKUP_OBJECT,
    ai2thor_action=Ai2thorAction.PICKUP_OBJECT,
    action_category=ActionCategory.PICKUP_PUT_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.PICKUPABLE,
    _config_dependent_parameters={"forceAction", "manualInteract"},
)
put_object_action = EnvironmentAction(
    name=EnvActionName.PUT_OBJECT,
    ai2thor_action=Ai2thorAction.PUT_OBJECT,
    action_category=ActionCategory.PICKUP_PUT_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.RECEPTACLE,
    _config_dependent_parameters={"forceAction", "placeStationary"},
)
drop_hand_object_action = EnvironmentAction(
    name=EnvActionName.DROP_HAND_OBJECT,
    ai2thor_action=Ai2thorAction.DROP_HAND_OBJECT,
    action_category=ActionCategory.DROP_ACTIONS,
    _config_dependent_parameters={"forceAction"},
)
throw_object_action = EnvironmentAction(
    name=EnvActionName.THROW_OBJECT,
    ai2thor_action=Ai2thorAction.THROW_OBJECT,
    action_category=ActionCategory.THROW_ACTIONS,
    _parameter_name="moveMagnitude",
    _parameter_range=(0, 100),
    _parameter_discrete_value=50,
    _config_dependent_parameters={"forceAction"},
)
push_object_action = EnvironmentAction(
    name=EnvActionName.PUSH_OBJECT,
    ai2thor_action=Ai2thorAction.PUSH_OBJECT,
    action_category=ActionCategory.PUSH_PULL_ACTIONS,
    _parameter_name="moveMagnitude",
    _parameter_range=(0, 200),
    _parameter_discrete_value=100,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.MOVEABLE,
    _config_dependent_parameters={"forceAction"},
)
pull_object_action = EnvironmentAction(
    name=EnvActionName.PULL_OBJECT,
    ai2thor_action=Ai2thorAction.PULL_OBJECT,
    action_category=ActionCategory.PUSH_PULL_ACTIONS,
    _parameter_name="moveMagnitude",
    _parameter_range=(0, 200),
    _parameter_discrete_value=100,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.MOVEABLE,
    _config_dependent_parameters={"forceAction"},
)
# Note: "DirectionalPush", "TouchThenApplyForce" are not available because we keep only actions with a single parameter
# Object interaction actions (see: https://ai2thor.allenai.org/ithor/documentation/object-state-changes)
open_object_action = OpenCloseEnvAction(
    name=EnvActionName.OPEN_OBJECT,
    ai2thor_action=Ai2thorAction.OPEN_OBJECT,
)
close_object_action = OpenCloseEnvAction(
    name=EnvActionName.CLOSE_OBJECT,
    ai2thor_action=Ai2thorAction.CLOSE_OBJECT,
)
partial_open_object_action = EnvironmentAction(
    name=EnvActionName.PARTIAL_OPEN_OBJECT,
    ai2thor_action=Ai2thorAction.OPEN_OBJECT,
    action_category=ActionCategory.SPECIAL,
    _parameter_name="openness",
    _parameter_range=(0, 1),
    has_target_object=True,
    _object_required_property=SimObjFixedProp.OPENABLE,
    _config_dependent_parameters={"forceAction"},
)
toggle_object_on_action = EnvironmentAction(
    name=EnvActionName.TOGGLE_OBJECT_ON,
    ai2thor_action=Ai2thorAction.TOGGLE_OBJECT_ON,
    action_category=ActionCategory.TOGGLE_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.TOGGLEABLE,
    _config_dependent_parameters={"forceAction"},
)
toggle_object_off_action = EnvironmentAction(
    name=EnvActionName.TOGGLE_OBJECT_OFF,
    ai2thor_action=Ai2thorAction.TOGGLE_OBJECT_OFF,
    action_category=ActionCategory.TOGGLE_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.TOGGLEABLE,
    _config_dependent_parameters={"forceAction"},
)
fill_object_with_liquid_action = ConditionalExecutionAction(
    name=EnvActionName.FILL_OBJECT_WITH_LIQUID,
    ai2thor_action=Ai2thorAction.FILL_OBJECT_WITH_LIQUID,
    action_category=ActionCategory.LIQUID_MANIPULATION_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    _other_ai2thor_parameters={"fillLiquid": "water"},
    _config_dependent_parameters={"forceAction"},
    action_condition=fill_object_with_liquid_condition,
)
empty_liquid_from_object_action = EnvironmentAction(
    name=EnvActionName.EMPTY_LIQUID_FROM_OBJECT,
    ai2thor_action=Ai2thorAction.EMPTY_LIQUID_FROM_OBJECT,
    action_category=ActionCategory.LIQUID_MANIPULATION_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    _config_dependent_parameters={"forceAction"},
)
break_object_action = EnvironmentAction(
    name=EnvActionName.BREAK_OBJECT,
    ai2thor_action=Ai2thorAction.BREAK_OBJECT,
    action_category=ActionCategory.BREAK_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.BREAKABLE,
    _config_dependent_parameters={"forceAction"},
)
slice_object_action = ConditionalExecutionAction(
    name=EnvActionName.SLICE_OBJECT,
    ai2thor_action=Ai2thorAction.SLICE_OBJECT,
    action_category=ActionCategory.SLICE_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.SLICEABLE,
    _config_dependent_parameters={"forceAction"},
    action_condition=slice_object_condition,
)
use_up_object_action = EnvironmentAction(
    name=EnvActionName.USE_UP_OBJECT,
    ai2thor_action=Ai2thorAction.USE_UP_OBJECT,
    action_category=ActionCategory.USE_UP_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.SLICEABLE,
    _config_dependent_parameters={"forceAction"},
)
dirty_object_action = EnvironmentAction(
    name=EnvActionName.DIRTY_OBJECT,
    ai2thor_action=Ai2thorAction.DIRTY_OBJECT,
    action_category=ActionCategory.CLEAN_DIRTY_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.DIRTYABLE,
    _config_dependent_parameters={"forceAction"},
)
clean_object_action = ConditionalExecutionAction(
    name=EnvActionName.CLEAN_OBJECT,
    ai2thor_action=Ai2thorAction.CLEAN_OBJECT,
    action_category=ActionCategory.CLEAN_DIRTY_ACTIONS,
    has_target_object=True,
    _object_required_property=SimObjFixedProp.DIRTYABLE,
    _config_dependent_parameters={"forceAction"},
    action_condition=clean_object_condition,
)
# Note: "CookObject" is not used because it has "magical" effects instead of having contextual effects (like using a toaster to cook bread)
# === Type Annotations ===
move_ahead_action: EnvironmentAction
move_back_action: EnvironmentAction
move_left_action: EnvironmentAction
move_right_action: EnvironmentAction
rotate_left_action: EnvironmentAction
rotate_right_action: EnvironmentAction
look_up_action: EnvironmentAction
look_down_action: EnvironmentAction
crouch_action: EnvironmentAction
stand_action: EnvironmentAction
done_action: EnvironmentAction
move_held_object_ahead_back_action: EnvironmentAction
move_held_object_right_left_action: EnvironmentAction
move_held_object_up_down_action: EnvironmentAction
rotate_held_object_roll_action: EnvironmentAction
rotate_held_object_pitch_action: EnvironmentAction
rotate_held_object_yaw_action: EnvironmentAction
pickup_object_action: EnvironmentAction
put_object_action: EnvironmentAction
drop_hand_object_action: EnvironmentAction
throw_object_action: EnvironmentAction
push_object_action: EnvironmentAction
pull_object_action: EnvironmentAction
open_object_action: EnvironmentAction
close_object_action: EnvironmentAction
partial_open_object_action: EnvironmentAction
toggle_object_on_action: EnvironmentAction
toggle_object_off_action: EnvironmentAction
fill_object_with_liquid_action: ConditionalExecutionAction
empty_liquid_from_object_action: EnvironmentAction
break_object_action: EnvironmentAction
slice_object_action: ConditionalExecutionAction
use_up_object_action: EnvironmentAction
dirty_object_action: EnvironmentAction
clean_object_action: ConditionalExecutionAction

# %% === Constants ===
ALL_ACTIONS: set[EnvironmentAction] = {
    move_ahead_action,
    move_back_action,
    move_left_action,
    move_right_action,
    rotate_left_action,
    rotate_right_action,
    look_up_action,
    look_down_action,
    crouch_action,
    stand_action,
    done_action,
    move_held_object_ahead_back_action,
    move_held_object_right_left_action,
    move_held_object_up_down_action,
    rotate_held_object_roll_action,
    rotate_held_object_pitch_action,
    rotate_held_object_yaw_action,
    pickup_object_action,
    put_object_action,
    drop_hand_object_action,
    throw_object_action,
    push_object_action,
    pull_object_action,
    close_object_action,
    open_object_action,
    partial_open_object_action,
    toggle_object_on_action,
    toggle_object_off_action,
    fill_object_with_liquid_action,
    empty_liquid_from_object_action,
    break_object_action,
    slice_object_action,
    use_up_object_action,
    dirty_object_action,
    clean_object_action,
}
ACTIONS_BY_CATEGORY: dict[ActionCategory, list[EnvironmentAction]]
ACTIONS_BY_CATEGORY = {category: [] for category in ActionCategory}
for action in ALL_ACTIONS:
    category = action.action_category
    ACTIONS_BY_CATEGORY[category].append(action)

ACTIONS_BY_NAME: dict[EnvActionName, EnvironmentAction]
ACTIONS_BY_NAME = {action.name: action for action in ALL_ACTIONS}


# %% === Exceptions ===
class MissingParameterRangeError(ValueError):
    """
    Exception raised when an action requires a parameter but parameter range has been defined for the action.

    Either the action should not require a parameter, or the action has been incorrectly defined in the environment.
    """

    def __init__(self, environment_action: EnvironmentAction) -> None:
        self.environment_action = environment_action
        super().__init__(f"Action {self.environment_action} requires a parameter but no parameter range is defined.")


class UnknownActionCategoryError(ValueError):
    """Exception raised for unknown action categories in environment mode config."""

    def __init__(self, action_category: str) -> None:
        self.action_category = action_category
        super().__init__(
            f"Unknown action category '{action_category}' in environment mode config. "
            f"Available action categories are {[category.value for category in ActionCategory]}."
        )
