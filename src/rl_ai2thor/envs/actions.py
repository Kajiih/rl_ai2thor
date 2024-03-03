"""
Module for actions for AI2THOR RL Environment, interfaces between the agent and the ai2thor controller.

This module provides classes and definitions for handling actions, conditions, and interactions within the AI2THOR simulated environment

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
"""

# %% === Imports ===
from __future__ import annotations

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_ai2thor.envs.tasks.items import SimObjFixedProp, SimObjVariableProp
from rl_ai2thor.utils.general_utils import nested_dict_get

if TYPE_CHECKING:
    from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
    from rl_ai2thor.utils.ai2thor_types import EventLike


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
# TODO: Change perform to not need the environment
# TODO: Add target value to object required property
@dataclass(frozen=True)
class EnvironmentAction:
    """
    Base class for complex environment actions that correspond to ai2thor actions.

    Attributes:
        name (str): Name of the action in the RL environment.
        ai2thor_action (str): Name of the ai2thor action corresponding to the
            environment's action.
        action_category (str): Category of the action (e.g. movement_actions
            for MoveAhead).
        has_target_object (bool, optional): Whether the action requires a target
            object.
        object_required_property (str, optional): Name of the required property
            of the target object.
        parameter_name (str, optional): Name of the quantitative parameter of
            the action.
        parameter_range (tuple[float, float], optional): Range of the quantitative
            parameter of the action. Can be overridden by the config.
        parameter_discrete_value (float, optional): Value of the quantitative
            parameter of the action in discrete environment mode. Can be
            overridden by the config.
        other_ai2thor_parameters (dict[str, Any], optional): Other ai2thor
            parameters of the action that take a fixed value (e.g. "up" and
            "right" for MoveHeldObject) and their value.
        config_dependent_parameters (set[str], optional): Set of parameters
            that depend on the environment config.

    Methods:
        perform(
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.
        ) -> EventLike:
            Perform the action in the environment and return the event.

        fail_perform(
            env (ITHOREnv): Environment in which the action was performed.
            error_message (str): Error message to log in the event.
        ) -> EventLike:
            Generate an event corresponding to the failure of the action.
    """

    name: EnvActionName
    ai2thor_action: Ai2thorAction
    action_category: ActionCategory
    _: dataclasses.KW_ONLY  # Following arguments are keyword-only
    has_target_object: bool = False
    object_required_property: SimObjFixedProp | None = None
    parameter_name: str | None = None
    parameter_range: tuple[float, float] | None = None
    parameter_discrete_value: float | None = None
    other_ai2thor_parameters: dict[str, Any] = field(default_factory=dict)
    config_dependent_parameters: set[str] = field(default_factory=set)

    def perform(
        self,
        env: ITHOREnv,
        action_parameter: float | None = None,
        target_object_id: str | None = None,
    ) -> EventLike:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.

        Returns:
            event (EventLike): Event returned by the controller.
        """
        action_parameters = self.other_ai2thor_parameters.copy()
        if self.parameter_name is not None:
            # Deal with the discrete/continuous environment mode
            if action_parameter is None:
                assert self.parameter_discrete_value is not None
                assert env.config["discrete_actions"]
                # Override the action parameter with the value from the config
                action_parameter = nested_dict_get(
                    d=env.config,
                    keys=["action_parameter_data", self.name, "discrete_value"],
                    default=self.parameter_discrete_value,
                )
            elif self.parameter_range is not None:
                # Override the range with the value from the config
                parameter_range = nested_dict_get(
                    d=env.config,
                    keys=["action_parameter_data", self.name, "range"],
                    default=self.parameter_range,
                )
                action_parameter = parameter_range[0] + action_parameter * (parameter_range[1] - parameter_range[0])
            else:
                raise MissingParameterRangeError(self.ai2thor_action)

            action_parameters[self.parameter_name] = action_parameter
        if self.has_target_object:
            action_parameters["objectId"] = target_object_id
        for parameter_name in self.config_dependent_parameters:
            action_parameters[parameter_name] = env.config["action_parameters"][parameter_name]

        return env.controller.step(
            action=self.ai2thor_action,
            **action_parameters,
        )

    def fail_perform(
        self,
        env: ITHOREnv,
        error_message: str,
    ) -> EventLike:
        """
        Generate an event corresponding to the failure of the action.

        Args:
            env (ITHOREnv): Environment in which the action was performed.
            error_message (str): Error message to log in the event.

        Returns:
            event (EventLike): Event for the failed action.
        """
        event = env.controller.step(action="Done")
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
        target_object_id: str | None = None,
    ) -> EventLike:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.

        Returns:
            event (EventLike): Event returned by the controller.
        """
        return (
            super().perform(env, action_parameter, target_object_id)
            if self.action_condition(env)
            else self.fail_perform(env, error_message=self.action_condition.error_message(self))
        )

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
            bool: Whether the agent has visible running water in its field of view.
        """
        return any(
            (
                obj[SimObjVariableProp.VISIBLE]
                and obj[SimObjVariableProp.IS_TOGGLED]
                and obj[SimObjFixedProp.OBJECT_TYPE] in {"Faucet", "ShowerHead"}
            )
            for obj in env.last_event.metadata["objects"]
        )

    @staticmethod
    def _base_error_message(action: EnvironmentAction) -> str:
        """Return the default error message for the condition."""
        return f"Agent needs to have visible running water to perform action {action.ai2thor_action}!"


@dataclass
class HoldingObjectTypeCondition(BaseActionCondition):
    """
    Check whether the agent is holding an object of a specific type.

    Used for SliceObject.

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
            len(env.last_event.metadata["inventoryObjects"]) > 0
            and env.last_event.metadata["inventoryObjects"][0][SimObjFixedProp.OBJECT_TYPE] == self.object_type
        )

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
    object_type="Knife",
    overriding_message="Agent needs to hold a knife to slice an object!",
)


# %% Exceptions
class MissingParameterRangeError(ValueError):
    """
    Exception raised when an action requires a parameter but parameter range has been defined for the action.

    Either the action should not require a parameter, or the action has been badly defined in the environment.
    """

    def __init__(self, ai2thor_action: str) -> None:
        self.ai2thor_action = ai2thor_action
        super().__init__(f"Action {self.ai2thor_action} requires a parameter but no parameter range is defined.")


# %% == Actions definitions ==
# TODO: Use task enums to define object required properties
# Navigation actions (see: https://ai2thor.allenai.org/ithor/documentation/navigation)
move_ahead_action = EnvironmentAction(
    name=EnvActionName.MOVE_AHEAD,
    ai2thor_action=Ai2thorAction.MOVE_AHEAD,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
move_back_action = EnvironmentAction(
    name=EnvActionName.MOVE_BACK,
    ai2thor_action=Ai2thorAction.MOVE_BACK,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
move_left_action = EnvironmentAction(
    name=EnvActionName.MOVE_LEFT,
    ai2thor_action=Ai2thorAction.MOVE_LEFT,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
move_right_action = EnvironmentAction(
    name=EnvActionName.MOVE_RIGHT,
    ai2thor_action=Ai2thorAction.MOVE_RIGHT,
    action_category=ActionCategory.MOVEMENT_ACTIONS,
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
rotate_left_action = EnvironmentAction(
    name=EnvActionName.ROTATE_LEFT,
    ai2thor_action=Ai2thorAction.ROTATE_LEFT,
    action_category=ActionCategory.BODY_ROTATION_ACTIONS,
    parameter_name="degrees",
    parameter_range=(0, 180),
    parameter_discrete_value=90,
)
rotate_right_action = EnvironmentAction(
    name=EnvActionName.ROTATE_RIGHT,
    ai2thor_action=Ai2thorAction.ROTATE_RIGHT,
    action_category=ActionCategory.BODY_ROTATION_ACTIONS,
    parameter_name="degrees",
    parameter_range=(0, 180),
    parameter_discrete_value=90,
)
look_up_action = EnvironmentAction(
    name=EnvActionName.LOOK_UP,
    ai2thor_action=Ai2thorAction.LOOK_UP,
    action_category=ActionCategory.CAMERA_ROTATION_ACTIONS,
    parameter_name="degrees",
    parameter_range=(0, 90),
    parameter_discrete_value=30,
)
look_down_action = EnvironmentAction(
    name=EnvActionName.LOOK_DOWN,
    ai2thor_action=Ai2thorAction.LOOK_DOWN,
    action_category=ActionCategory.CAMERA_ROTATION_ACTIONS,
    parameter_name="degrees",
    parameter_range=(0, 90),
    parameter_discrete_value=30,
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
    parameter_name="ahead",
    parameter_range=(-0.5, 0.5),
    other_ai2thor_parameters={"right": 0, "up": 0},
    config_dependent_parameters={"forceVisible"},
)
move_held_object_right_left_action = EnvironmentAction(
    name=EnvActionName.MOVE_HELD_OBJECT_RIGHT_LEFT,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    parameter_name="right",
    parameter_range=(-0.5, 0.5),
    other_ai2thor_parameters={"ahead": 0, "up": 0},
    config_dependent_parameters={"forceVisible"},
)
move_held_object_up_down_action = EnvironmentAction(
    name=EnvActionName.MOVE_HELD_OBJECT_UP_DOWN,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    parameter_name="up",
    parameter_range=(-0.5, 0.5),
    other_ai2thor_parameters={"ahead": 0, "right": 0},
    config_dependent_parameters={"forceVisible"},
)
rotate_held_object_roll_action = EnvironmentAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_ROLL,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    parameter_name="roll",
    parameter_range=(-180, 180),
    other_ai2thor_parameters={"pitch": 0, "yaw": 0},
)
rotate_held_object_pitch_action = EnvironmentAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_PITCH,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    parameter_name="pitch",
    parameter_range=(-180, 180),
    other_ai2thor_parameters={"roll": 0, "yaw": 0},
)
rotate_held_object_yaw_action = EnvironmentAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_YAW,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    action_category=ActionCategory.HAND_MOVEMENT_ACTIONS,
    parameter_name="yaw",
    parameter_range=(-180, 180),
    other_ai2thor_parameters={"roll": 0, "pitch": 0},
)
pickup_object_action = EnvironmentAction(
    name=EnvActionName.PICKUP_OBJECT,
    ai2thor_action=Ai2thorAction.PICKUP_OBJECT,
    action_category=ActionCategory.PICKUP_PUT_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.PICKUPABLE,
    config_dependent_parameters={"forceAction", "manualInteract"},
)
put_object_action = EnvironmentAction(
    name=EnvActionName.PUT_OBJECT,
    ai2thor_action=Ai2thorAction.PUT_OBJECT,
    action_category=ActionCategory.PICKUP_PUT_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.RECEPTACLE,
    config_dependent_parameters={"forceAction", "placeStationary"},
)
drop_hand_object_action = EnvironmentAction(
    name=EnvActionName.DROP_HAND_OBJECT,
    ai2thor_action=Ai2thorAction.DROP_HAND_OBJECT,
    action_category=ActionCategory.DROP_ACTIONS,
    config_dependent_parameters={"forceAction"},
)
throw_object_action = EnvironmentAction(
    name=EnvActionName.THROW_OBJECT,
    ai2thor_action=Ai2thorAction.THROW_OBJECT,
    action_category=ActionCategory.THROW_ACTIONS,
    parameter_name="moveMagnitude",
    parameter_range=(0, 100),
    parameter_discrete_value=50,
    config_dependent_parameters={"forceAction"},
)
push_object_action = EnvironmentAction(
    name=EnvActionName.PUSH_OBJECT,
    ai2thor_action=Ai2thorAction.PUSH_OBJECT,
    action_category=ActionCategory.PUSH_PULL_ACTIONS,
    parameter_name="moveMagnitude",
    parameter_range=(0, 200),
    parameter_discrete_value=100,
    has_target_object=True,
    object_required_property=SimObjFixedProp.MOVEABLE,
    config_dependent_parameters={"forceAction"},
)
pull_object_action = EnvironmentAction(
    name=EnvActionName.PULL_OBJECT,
    ai2thor_action=Ai2thorAction.PULL_OBJECT,
    action_category=ActionCategory.PUSH_PULL_ACTIONS,
    parameter_name="moveMagnitude",
    parameter_range=(0, 200),
    parameter_discrete_value=100,
    has_target_object=True,
    object_required_property=SimObjFixedProp.MOVEABLE,
    config_dependent_parameters={"forceAction"},
)
# Note: "DirectionalPush", "TouchThenApplyForce" are not available because we keep only actions with a single parameter
# Object interaction actions (see: https://ai2thor.allenai.org/ithor/documentation/object-state-changes)
open_object_action = EnvironmentAction(
    name=EnvActionName.OPEN_OBJECT,
    ai2thor_action=Ai2thorAction.OPEN_OBJECT,
    action_category=ActionCategory.OPEN_CLOSE_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.OPENABLE,
    config_dependent_parameters={"forceAction"},
)
close_object_action = EnvironmentAction(
    name=EnvActionName.CLOSE_OBJECT,
    ai2thor_action=Ai2thorAction.CLOSE_OBJECT,
    action_category=ActionCategory.OPEN_CLOSE_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.OPENABLE,
    config_dependent_parameters={"forceAction"},
)
partial_open_object_action = EnvironmentAction(
    name=EnvActionName.PARTIAL_OPEN_OBJECT,
    ai2thor_action=Ai2thorAction.OPEN_OBJECT,
    action_category=ActionCategory.SPECIAL,
    parameter_name="openness",
    parameter_range=(0, 1),
    has_target_object=True,
    object_required_property=SimObjFixedProp.OPENABLE,
    config_dependent_parameters={"forceAction"},
)
toggle_object_on_action = EnvironmentAction(
    name=EnvActionName.TOGGLE_OBJECT_ON,
    ai2thor_action=Ai2thorAction.TOGGLE_OBJECT_ON,
    action_category=ActionCategory.TOGGLE_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.TOGGLEABLE,
    config_dependent_parameters={"forceAction"},
)
toggle_object_off_action = EnvironmentAction(
    name=EnvActionName.TOGGLE_OBJECT_OFF,
    ai2thor_action=Ai2thorAction.TOGGLE_OBJECT_OFF,
    action_category=ActionCategory.TOGGLE_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.TOGGLEABLE,
    config_dependent_parameters={"forceAction"},
)
fill_object_with_liquid_action = ConditionalExecutionAction(
    name=EnvActionName.FILL_OBJECT_WITH_LIQUID,
    ai2thor_action=Ai2thorAction.FILL_OBJECT_WITH_LIQUID,
    action_category=ActionCategory.LIQUID_MANIPULATION_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    other_ai2thor_parameters={"fillLiquid": "water"},
    config_dependent_parameters={"forceAction"},
    action_condition=fill_object_with_liquid_condition,
)
empty_liquid_from_object_action = EnvironmentAction(
    name=EnvActionName.EMPTY_LIQUID_FROM_OBJECT,
    ai2thor_action=Ai2thorAction.EMPTY_LIQUID_FROM_OBJECT,
    action_category=ActionCategory.LIQUID_MANIPULATION_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    config_dependent_parameters={"forceAction"},
)
break_object_action = EnvironmentAction(
    name=EnvActionName.BREAK_OBJECT,
    ai2thor_action=Ai2thorAction.BREAK_OBJECT,
    action_category=ActionCategory.BREAK_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.BREAKABLE,
    config_dependent_parameters={"forceAction"},
)
slice_object_action = ConditionalExecutionAction(
    name=EnvActionName.SLICE_OBJECT,
    ai2thor_action=Ai2thorAction.SLICE_OBJECT,
    action_category=ActionCategory.SLICE_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.SLICEABLE,
    config_dependent_parameters={"forceAction"},
    action_condition=slice_object_condition,
)
use_up_object_action = EnvironmentAction(
    name=EnvActionName.USE_UP_OBJECT,
    ai2thor_action=Ai2thorAction.USE_UP_OBJECT,
    action_category=ActionCategory.USE_UP_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.SLICEABLE,
    config_dependent_parameters={"forceAction"},
)
dirty_object_action = EnvironmentAction(
    name=EnvActionName.DIRTY_OBJECT,
    ai2thor_action=Ai2thorAction.DIRTY_OBJECT,
    action_category=ActionCategory.CLEAN_DIRTY_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.DIRTYABLE,
    config_dependent_parameters={"forceAction"},
)
clean_object_action = ConditionalExecutionAction(
    name=EnvActionName.CLEAN_OBJECT,
    ai2thor_action=Ai2thorAction.CLEAN_OBJECT,
    action_category=ActionCategory.CLEAN_DIRTY_ACTIONS,
    has_target_object=True,
    object_required_property=SimObjFixedProp.DIRTYABLE,
    config_dependent_parameters={"forceAction"},
    action_condition=clean_object_condition,
)
# Note: "CookObject" is not used because it has "magical" effects instead of having contextual effects (like using a toaster to cook bread)


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
ACTIONS_BY_CATEGORY: dict[ActionCategory, list[EnvironmentAction]] = {category: [] for category in ActionCategory}
for action in ALL_ACTIONS:
    category = action.action_category
    ACTIONS_BY_CATEGORY[category].append(action)

ACTIONS_BY_NAME = {action.name: action for action in ALL_ACTIONS}
