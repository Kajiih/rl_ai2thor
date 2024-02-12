"""
Module for actions for AI2THOR RL Environment, interfaces between the agent and the ai2thor controller.

This module provides classes and definitions for handling actions, conditions, and interactions within the AI2THOR simulated environment

Classes:
- EnvironmentAction: Base class for complex environment actions.
- BaseActionCondition: Base class for conditions determining if an action can be performed.
- ConditionalExecutionAction: Class for actions that can only be performed under certain conditions.
- VisibleWaterCondition: Condition for actions requiring visible running water.
- HoldingObjectTypeCondition: Condition for actions requiring the agent to hold a specific object type.

Actions:
- MoveAhead
- MoveBack
- MoveLeft
- MoveRight
- RotateLeft
- RotateRight
- LookUp
- LookDown
- Crouch
- Stand
- Done
- MoveHeldObjectAheadBack
- MoveHeldObjectRightLeft
- MoveHeldObjectUpDown
- RotateHeldObjectRoll
- RotateHeldObjectPitch
- RotateHeldObjectYaw
- PickupObject
- PutObject
- DropHandObject
- ThrowObject
- PushObject
- PullObject
- CloseObject
- OpenObject
- PartialOpenObject
- ToggleObjectOn
- ToggleObjectOff
- FillObjectWithLiquid
- EmptyLiquidFromObject
- BreakObject
- SliceObject
- UseUpObject
- DirtyObject
- CleanObject


Constants:
- ALL_ACTIONS: List of all defined actions.
- ACTION_CATEGORIES: Set of unique action categories.
- ACTIONS_BY_CATEGORY: Dictionary mapping action categories to corresponding actions.
- ACTIONS_BY_NAME: Dictionary mapping action names to their corresponding definitions.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType, TypeVar

from rl_ai2thor.utils.general_utils import nested_dict_get

if TYPE_CHECKING:
    from ai2thor_envs import ITHOREnv

    from rl_ai2thor.utils.ai2thor_types import EventLike


# %% Exceptions
class MissingParameteRangeError(ValueError):
    """
    Exception raised when an action requires a parameter but parameter range has been defined for the action.

    Either the action should not require a parameter, or the action has been badly defined in the environment.
    """

    def __init__(self, ai2thor_action: str) -> None:
        self.ai2thor_action = ai2thor_action
        super().__init__(f"Action {self.ai2thor_action} requires a parameter but no parameter range is defined.")


# TODO: Define enums for ai2thor actions and other literals.
# === Action Classes ===
# TODO: Change perform to not need the environment
@dataclass
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
            parameter of the action. Can be overriden by the config.
        parameter_discrete_value (float, optional): Value of the quantitative
            parameter of the action in discrete environment mode. Can be
            overriden by the config.
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

    name: str
    ai2thor_action: str
    action_category: str
    _: dataclasses.KW_ONLY  # Following arguments are keyword-only
    has_target_object: bool = False
    object_required_property: str | None = None
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
                raise MissingParameteRangeError(self.ai2thor_action)

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


@dataclass
class ConditionalExecutionAction(EnvironmentAction):
    """
    Base class for actions that can only be performed under certain conditions.

    Actions that inherit from this class add conditions that are not natively
    handled by ai2thor (e.g. SliceObject can only be performed
    if the agent is holding a knife).

    Attributes:
        condition_function (Callable): Function that takes the environment as input
            and returns a boolean indicating whether the action can be performed.
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
        if self.action_condition(env):
            event = super().perform(env, action_parameter, target_object_id)
        else:
            event = self.fail_perform(env, error_message=self.action_condition.error_message(self))

        return event


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
        for obj in env.last_event.metadata["objects"]:
            if obj["visible"] and obj["isToggled"] and obj["objectType"] in {"Faucet", "ShowerHead"}:
                return True
        return False

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
            and env.last_event.metadata["inventoryObjects"][0]["objectType"] == self.object_type
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


# == Actions definitions ==
# Navigation actions (see: https://ai2thor.allenai.org/ithor/documentation/navigation)
move_ahead_action = EnvironmentAction(
    name="MoveAhead",
    ai2thor_action="MoveAhead",
    action_category="movement_actions",
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
move_back_action = EnvironmentAction(
    name="MoveBack",
    ai2thor_action="MoveBack",
    action_category="movement_actions",
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
move_left_action = EnvironmentAction(
    name="MoveLeft",
    ai2thor_action="MoveLeft",
    action_category="movement_actions",
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
move_right_action = EnvironmentAction(
    name="MoveRight",
    ai2thor_action="MoveRight",
    action_category="movement_actions",
    parameter_name="moveMagnitude",
    parameter_range=(0, 1),
    parameter_discrete_value=0.25,
)
rotate_left_action = EnvironmentAction(
    name="RotateLeft",
    ai2thor_action="RotateLeft",
    action_category="body_rotation_actions",
    parameter_name="degrees",
    parameter_range=(0, 180),
    parameter_discrete_value=90,
)
rotate_right_action = EnvironmentAction(
    name="RotateRight",
    ai2thor_action="RotateRight",
    action_category="body_rotation_actions",
    parameter_name="degrees",
    parameter_range=(0, 180),
    parameter_discrete_value=90,
)
look_up_action = EnvironmentAction(
    name="LookUp",
    ai2thor_action="LookUp",
    action_category="camera_rotation_actions",
    parameter_name="degrees",
    parameter_range=(0, 90),
    parameter_discrete_value=30,
)
look_down_action = EnvironmentAction(
    name="LookDown",
    ai2thor_action="LookDown",
    action_category="camera_rotation_actions",
    parameter_name="degrees",
    parameter_range=(0, 90),
    parameter_discrete_value=30,
)
crouch_action = EnvironmentAction(
    name="Crouch",
    ai2thor_action="Crouch",
    action_category="crouch_actions",
)
stand_action = EnvironmentAction(
    name="Stand",
    ai2thor_action="Stand",
    action_category="crouch_actions",
)
done_action = EnvironmentAction(
    name="Done",
    ai2thor_action="Done",
    action_category="done_actions",
)
# Note: "Teleport", "TeleportFull" are not available to the agent
# Object manipulation actions (see: https://ai2thor.allenai.org/ithor/documentation/interactive-physics)
move_held_object_ahead_back_action = EnvironmentAction(
    name="MoveHeldObjectAheadBack",
    ai2thor_action="MoveHeldObject",
    action_category="hand_movement_actions",
    parameter_name="ahead",
    parameter_range=(-0.5, 0.5),
    # parameter_discrete_value=0.25,  # ! Should not be used in discrete environment mode
    other_ai2thor_parameters={"right": 0, "up": 0},
    config_dependent_parameters={"forceVisible"},
)
move_held_object_right_left_action = EnvironmentAction(
    name="MoveHeldObjectRightLeft",
    ai2thor_action="MoveHeldObject",
    action_category="hand_movement_actions",
    parameter_name="right",
    parameter_range=(-0.5, 0.5),
    # parameter_discrete_value=0.25,  # ! Should not be used in discrete environment mode
    other_ai2thor_parameters={"ahead": 0, "up": 0},
    config_dependent_parameters={"forceVisible"},
)
move_held_object_up_down_action = EnvironmentAction(
    name="MoveHeldObjectUpDown",
    ai2thor_action="MoveHeldObject",
    action_category="hand_movement_actions",
    parameter_name="up",
    parameter_range=(-0.5, 0.5),
    # parameter_discrete_value=0.25,  # ! Should not be used in discrete environment mode
    other_ai2thor_parameters={"ahead": 0, "right": 0},
    config_dependent_parameters={"forceVisible"},
)
rotate_held_object_roll_action = EnvironmentAction(
    name="RotateHeldObjectRoll",
    ai2thor_action="RotateHeldObject",
    action_category="hand_movement_actions",
    parameter_name="roll",
    parameter_range=(-180, 180),
    # parameter_discrete_value=90,  # ! Should not be used in discrete environment mode
    other_ai2thor_parameters={"pitch": 0, "yaw": 0},
)  # Around forward-back axis
rotate_held_object_pitch_action = EnvironmentAction(
    name="RotateHeldObjectPitch",
    ai2thor_action="RotateHeldObject",
    action_category="hand_movement_actions",
    parameter_name="pitch",
    parameter_range=(-180, 180),
    # parameter_discrete_value=90,  # ! Should not be used in discrete environment mode
    other_ai2thor_parameters={"roll": 0, "yaw": 0},
)  # Around left-right axis
rotate_held_object_yaw_action = EnvironmentAction(
    name="RotateHeldObjectYaw",
    ai2thor_action="RotateHeldObject",
    action_category="hand_movement_actions",
    parameter_name="yaw",
    parameter_range=(-180, 180),
    # parameter_discrete_value=90,  # ! Should not be used in discrete environment mode
    other_ai2thor_parameters={"roll": 0, "pitch": 0},
)  # Around up-down axis
pickup_object_action = EnvironmentAction(
    name="PickupObject",
    ai2thor_action="PickupObject",
    action_category="pickup_put_actions",
    has_target_object=True,
    object_required_property="pickupable",
    config_dependent_parameters={"forceAction", "manualInteract"},
)
put_object_action = EnvironmentAction(
    name="PutObject",
    ai2thor_action="PutObject",
    action_category="pickup_put_actions",
    has_target_object=True,
    object_required_property="receptacle",
    config_dependent_parameters={"forceAction", "placeStationary"},
)
drop_hand_object_action = EnvironmentAction(
    name="DropHandObject",
    ai2thor_action="DropHandObject",
    action_category="drop_actions",
    config_dependent_parameters={"forceAction"},
)  # Like throwing but with 0 force, meant to be used in tandem with the Move/Rotate hand movement actions
throw_object_action = EnvironmentAction(
    name="ThrowObject",
    ai2thor_action="ThrowObject",
    action_category="throw_actions",
    parameter_name="moveMagnitude",
    parameter_range=(0, 100),
    parameter_discrete_value=50,
    config_dependent_parameters={"forceAction"},
)
push_object_action = EnvironmentAction(
    name="PushObject",
    ai2thor_action="PushObject",
    action_category="push_pull_actions",
    parameter_name="moveMagnitude",
    parameter_range=(0, 200),
    parameter_discrete_value=100,
    has_target_object=True,
    object_required_property="moveable",
    config_dependent_parameters={"forceAction"},
)
pull_object_action = EnvironmentAction(
    name="PullObject",
    ai2thor_action="PullObject",
    action_category="push_pull_actions",
    parameter_name="moveMagnitude",
    parameter_range=(0, 200),
    parameter_discrete_value=100,
    has_target_object=True,
    object_required_property="moveable",
    config_dependent_parameters={"forceAction"},
)
# Note: "DirectionalPush", "TouchThenApplyForce" are not available because we keep only actions with a single parameter
# Object interaction actions (see: https://ai2thor.allenai.org/ithor/documentation/object-state-changes)
open_object_action = EnvironmentAction(
    name="OpenObject",
    ai2thor_action="OpenObject",
    action_category="open_close_actions",
    has_target_object=True,
    object_required_property="openable",
    config_dependent_parameters={"forceAction"},
)
close_object_action = EnvironmentAction(
    name="CloseObject",
    ai2thor_action="CloseObject",
    action_category="open_close_actions",
    has_target_object=True,
    object_required_property="openable",
    config_dependent_parameters={"forceAction"},
)
partial_open_object_action = EnvironmentAction(
    name="PartialOpenObject",
    ai2thor_action="OpenObject",
    action_category="_special",
    parameter_name="openness",
    parameter_range=(0, 1),
    # parameter_discrete_value=1,  # ! Should not be used in discrete environment mode
    has_target_object=True,
    object_required_property="openable",
    config_dependent_parameters={"forceAction"},
)
toggle_object_on_action = EnvironmentAction(
    name="ToggleObjectOn",
    ai2thor_action="ToggleObjectOn",
    action_category="toggle_actions",
    has_target_object=True,
    object_required_property="toggleable",
    config_dependent_parameters={"forceAction"},
)
toggle_object_off_action = EnvironmentAction(
    name="ToggleObjectOff",
    ai2thor_action="ToggleObjectOff",
    action_category="toggle_actions",
    has_target_object=True,
    object_required_property="toggleable",
    config_dependent_parameters={"forceAction"},
)
fill_object_with_liquid_action = ConditionalExecutionAction(
    name="FillObjectWithLiquid",
    ai2thor_action="FillObjectWithLiquid",
    action_category="liquid_manipulation_actions",
    has_target_object=True,
    object_required_property="canFillWithLiquid",
    other_ai2thor_parameters={"fillLiquid": "water"},
    config_dependent_parameters={"forceAction"},
    action_condition=fill_object_with_liquid_condition,
)
empty_liquid_from_object_action = EnvironmentAction(
    name="EmptyLiquidFromObject",
    ai2thor_action="EmptyLiquidFromObject",
    action_category="liquid_manipulation_actions",
    has_target_object=True,
    object_required_property="canFillWithLiquid",
    config_dependent_parameters={"forceAction"},
)
break_object_action = EnvironmentAction(
    name="BreakObject",
    ai2thor_action="BreakObject",
    action_category="break_actions",
    has_target_object=True,
    object_required_property="breakable",
    config_dependent_parameters={"forceAction"},
)
slice_object_action = ConditionalExecutionAction(
    name="SliceObject",
    ai2thor_action="SliceObject",
    action_category="slice_actions",
    has_target_object=True,
    object_required_property="sliceable",
    config_dependent_parameters={"forceAction"},
    action_condition=slice_object_condition,
)
use_up_object_action = EnvironmentAction(
    name="UseUpObject",
    ai2thor_action="UseUpObject",
    action_category="use_up_actions",
    has_target_object=True,
    object_required_property="canBeUsedUp",
    config_dependent_parameters={"forceAction"},
)
dirty_object_action = EnvironmentAction(
    name="DirtyObject",
    ai2thor_action="DirtyObject",
    action_category="clean_dirty_actions",
    has_target_object=True,
    object_required_property="dirtyable",
    config_dependent_parameters={"forceAction"},
)
clean_object_action = ConditionalExecutionAction(
    name="CleanObject",
    ai2thor_action="CleanObject",
    action_category="clean_dirty_actions",
    has_target_object=True,
    object_required_property="dirtyable",
    config_dependent_parameters={"forceAction"},
    action_condition=clean_object_condition,
)
# Note: "CookObject" is not used because it has "magical" effects instead of having contextual effects (like using a toaster to cook bread)


# === Constants ===
ALL_ACTIONS: list[EnvironmentAction] = [
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
]

ACTION_CATEGORIES = {action.action_category for action in ALL_ACTIONS}
ACTIONS_BY_CATEGORY = {category: [] for category in ACTION_CATEGORIES}
for action in ALL_ACTIONS:
    category = action.action_category
    ACTIONS_BY_CATEGORY[category].append(action)
ACTIONS_BY_NAME = {action.name: action for action in ALL_ACTIONS}
