"""
Module for actions for RL-THOR Environment, interfaces between the agent and the ai2thor controller.

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

from rl_thor.envs.sim_objects import (
    OPENABLES,
    WATER_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjMetadata,
    SimObjVariableProp,
)

if TYPE_CHECKING:
    from ai2thor.server import Event

    from rl_thor.envs._config import ActionDiscreteParamValuesConfig
    from rl_thor.envs.ai2thor_envs import ITHOREnv
    from rl_thor.envs.sim_objects import SimObjId


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
    # DONE = "Done"
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
    # DONE = "Done"  # TODO: Check if we keep this action
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


class ActionGroup(StrEnum):
    """Enum for action groups."""

    # === Navigation actions ===
    MOVEMENT_ACTIONS = "movement_actions"
    ROTATION_ACTIONS = "rotation_actions"
    HEAD_MOVEMENT_ACTIONS = "head_movement_actions"
    CROUCH_ACTIONS = "crouch_actions"
    # DONE_ACTIONS = "done_actions"  # Not supported in tasks yet
    # === Object manipulation actions ===
    PICKUP_PUT_ACTIONS = "pickup_put_actions"
    DROP_ACTIONS = "drop_actions"
    THROW_ACTIONS = "throw_actions"
    PUSH_PULL_ACTIONS = "push_pull_actions"
    HAND_CONTROL_ACTIONS = "hand_control_actions"
    # === Object interaction actions ===
    OPEN_CLOSE_ACTIONS = "open_close_actions"
    TOGGLE_ACTIONS = "toggle_actions"
    LIQUID_MANIPULATION_ACTIONS = "liquid_manipulation_actions"
    BREAK_ACTIONS = "break_actions"
    SLICE_ACTIONS = "slice_actions"
    USE_UP_ACTIONS = "use_up_actions"
    CLEAN_DIRTY_ACTIONS = "_clean_dirty_actions"  # Not well integrated in the environment yet
    SPECIAL = "_special"  # For actions that shouldn't be directly enabled from config


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
        action_group (ActionGroup): Group of the action (e.g. movement_actions
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
    action_group: ActionGroup
    _: dataclasses.KW_ONLY  # Following arguments are keyword-only
    has_target_object: bool = False
    _object_required_property: SimObjFixedProp | None = None
    _parameter_name: str | None = None
    _parameter_range: tuple[float, float] | None = None
    _parameter_discrete_value: float | None = None
    _other_ai2thor_parameters: dict[str, Any] = field(default_factory=dict)
    _config_dependent_parameters: frozenset[str] = field(default_factory=frozenset)

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
            # Find the main parameter value depending on the environment discrete/continuous mode
            if action_parameter is None:
                # Discrete environment mode
                assert env.config.action_modifiers.discrete_actions
                action_parameter = self._get_discrete_param_value(env.config.action_discrete_param_values)
            else:
                # Continuous environment mode
                assert not env.config.action_modifiers.discrete_actions
                if self._parameter_range is None:
                    raise MissingParameterRangeError(self)
                # Override the range with the value from the config
                parameter_range = self._parameter_range
                action_parameter = parameter_range[0] + action_parameter * (parameter_range[1] - parameter_range[0])

            # Add the main parameter to the action parameters
            action_parameters[self._parameter_name] = action_parameter
        if self.has_target_object:
            action_parameters["objectId"] = target_object_id
        for parameter_name in self._config_dependent_parameters:
            if parameter_name == "forceAction":
                action_parameters[parameter_name] = env.config.action_modifiers.force_action
            elif parameter_name == "forceVisible":
                action_parameters[parameter_name] = env.config.action_modifiers.force_visible
            elif parameter_name == "manualInteract":
                action_parameters[parameter_name] = env.config.action_modifiers.static_pickup
            elif parameter_name == "placeStationary":
                action_parameters[parameter_name] = env.config.action_modifiers.stationary_placement
            else:
                raise UnknownConfigDependentParameterError(self, parameter_name)

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

    def _get_discrete_param_value(self, config: ActionDiscreteParamValuesConfig) -> float:  # noqa: ARG002
        """
        Return the discrete parameter value for the action.

        Args:
            config (ActionDiscreteParamValuesConfig): Configuration for the discrete parameters
                values of the actions.

        Returns:
            discrete_param_value (float): Discrete parameter value for the action.
        """
        raise NotDiscreteCompatibleActionTypeError(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, ai2thor_action={self.ai2thor_action!r})"

    def __str__(self) -> str:
        return f"{self.name}"


@dataclass(frozen=True)
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


# TODO: Add this feature to the base class?
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


# %% === Action groups definition ===
# TODO: Use task enums to define object required properties
# === Navigation actions === (see: https://ai2thor.allenai.org/ithor/documentation/navigation)
@dataclass(frozen=True)
class MovementEnvAction(EnvironmentAction):
    """Base class for movement actions."""

    action_group: ActionGroup = ActionGroup.MOVEMENT_ACTIONS
    _parameter_name: str = "moveMagnitude"
    _parameter_range: tuple[float, float] = (0, 1)

    @classmethod
    def _get_discrete_param_value(cls, config: ActionDiscreteParamValuesConfig) -> float:
        """
        Return the discrete movement magnitude value for the action.

        Args:
            config (ActionDiscreteParamValuesConfig): Configuration for the discrete parameters
                values of the actions.

        Returns:
            discrete_param_value (float): Discrete movement magnitude value for the action.
        """
        return config.movement_magnitude


@dataclass(frozen=True)
class RotationEnvAction(EnvironmentAction):
    """Base class for rotation actions."""

    action_group: ActionGroup = ActionGroup.ROTATION_ACTIONS
    _parameter_name: str = "degrees"
    _parameter_range: tuple[float, float] = (0, 180)

    @classmethod
    def _get_discrete_param_value(cls, config: ActionDiscreteParamValuesConfig) -> float:
        """
        Return the discrete rotation degrees value for the action.

        Args:
            config (ActionDiscreteParamValuesConfig): Configuration for the discrete parameters
                values of the actions.

        Returns:
            discrete_param_value (float): Discrete rotation degrees value for the action.
        """
        return config.rotation_degrees


@dataclass(frozen=True)
class HeadMovementEnvAction(EnvironmentAction):
    """Base class for head movement actions."""

    action_group: ActionGroup = ActionGroup.HEAD_MOVEMENT_ACTIONS
    _parameter_name: str = "degrees"
    _parameter_range: tuple[float, float] = (0, 90)

    @classmethod
    def _get_discrete_param_value(cls, config: ActionDiscreteParamValuesConfig) -> float:
        """
        Return the discrete head movement degrees value for the action.

        Args:
            config (ActionDiscreteParamValuesConfig): Configuration for the discrete parameters
                values of the actions.

        Returns:
            discrete_param_value (float): Discrete head movement degrees value for the action.
        """
        return config.head_movement_degrees


@dataclass(frozen=True)
class CrouchStandEnvAction(EnvironmentAction):
    """Base class for crouch and stand actions."""

    action_group: ActionGroup = ActionGroup.CROUCH_ACTIONS


# Note: "Teleport", "TeleportFull" are not available to the agent


# === Object manipulation actions === (see: https://ai2thor.allenai.org/ithor/documentation/interactive-physics)
@dataclass(frozen=True)
class PickupPutEnvAction(EnvironmentAction):
    """Base class for pickup and put actions."""

    action_group: ActionGroup = ActionGroup.PICKUP_PUT_ACTIONS
    has_target_object: bool = True


@dataclass(frozen=True)
class DropHandObjectEnvAction(EnvironmentAction):
    """Base class for drop hand object action."""

    name: EnvActionName = EnvActionName.DROP_HAND_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.DROP_HAND_OBJECT
    action_group: ActionGroup = ActionGroup.DROP_ACTIONS
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})


@dataclass(frozen=True)
class ThrowEnvAction(EnvironmentAction):
    """Base class for throw object action."""

    name: EnvActionName = EnvActionName.THROW_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.THROW_OBJECT
    action_group: ActionGroup = ActionGroup.THROW_ACTIONS
    _parameter_name: str = "moveMagnitude"
    _parameter_range: tuple[float, float] = (0, 100)
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})

    @classmethod
    def _get_discrete_param_value(cls, config: ActionDiscreteParamValuesConfig) -> float:
        """
        Return the discrete throw strength value for the action.

        Args:
            config (ActionDiscreteParamValuesConfig): Configuration for the discrete parameters
                values of the actions.

        Returns:
            discrete_param_value (float): Discrete throw strength value for the action.
        """
        return config.throw_strength


@dataclass(frozen=True)
class PushPullEnvAction(EnvironmentAction):
    """Base class for push and pull object actions."""

    action_group: ActionGroup = ActionGroup.PUSH_PULL_ACTIONS
    _parameter_name: str = "moveMagnitude"
    _parameter_range: tuple[float, float] = (0, 200)
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.MOVEABLE
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})

    @classmethod
    def _get_discrete_param_value(cls, config: ActionDiscreteParamValuesConfig) -> float:
        """
        Return the discrete push/pull strength value for the action.

        Args:
            config (ActionDiscreteParamValuesConfig): Configuration for the discrete parameters
                values of the actions.

        Returns:
            discrete_param_value (float): Discrete push/pull strength value for the action.
        """
        return config.push_pull_strength


@dataclass(frozen=True)
class HandMovementEnvAction(EnvironmentAction):
    """Base class for hand movement actions."""

    action_group: ActionGroup = ActionGroup.HAND_CONTROL_ACTIONS
    _parameter_range: tuple[float, float] = (-0.5, 0.5)
    _config_dependent_parameters: frozenset[str] = frozenset({"forceVisible"})


@dataclass(frozen=True)
class HandRotationEnvAction(EnvironmentAction):
    """Base class for hand rotation actions."""

    action_group: ActionGroup = ActionGroup.HAND_CONTROL_ACTIONS
    _parameter_range: tuple[float, float] = (-180, 180)


# Note: "DirectionalPush", "TouchThenApplyForce" are not available because we keep only actions with a single parameter

# === Object interaction actions === (see: https://ai2thor.allenai.org/ithor/documentation/object-state-changes)


@dataclass(frozen=True)
class OpenCloseEnvAction(EnvironmentAction):
    """
    Base class for "OpenObject" and "CloseObject" actions.

    We need a specific class for this action because the object type needs to be in the OPENABLES
    list to avoid TimeoutErrors when trying to open or close objects like `Blinds`.
    """

    action_group: ActionGroup = ActionGroup.OPEN_CLOSE_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.OPENABLE
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})

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


@dataclass(frozen=True)
class ToggleEnvAction(EnvironmentAction):
    """Base class for toggle object actions."""

    action_group: ActionGroup = ActionGroup.TOGGLE_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.TOGGLEABLE
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})


@dataclass(frozen=True)
class FillObjectWithLiquidEnvAction(ConditionalExecutionAction):
    """Base class for fill object with liquid action."""

    name: EnvActionName = EnvActionName.FILL_OBJECT_WITH_LIQUID
    ai2thor_action: Ai2thorAction = Ai2thorAction.FILL_OBJECT_WITH_LIQUID
    action_group: ActionGroup = ActionGroup.LIQUID_MANIPULATION_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.CAN_FILL_WITH_LIQUID
    _other_ai2thor_parameters: dict[str, Any] = field(default_factory=dict)
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})
    action_condition: BaseActionCondition = fill_object_with_liquid_condition

    def __post_init__(self) -> None:
        if "fillLiquid" not in self._other_ai2thor_parameters:
            self._other_ai2thor_parameters["fillLiquid"] = "water"


@dataclass(frozen=True)
class EmptyLiquidFromObjectEnvAction(EnvironmentAction):
    """Base class for empty liquid from object action."""

    name: EnvActionName = EnvActionName.EMPTY_LIQUID_FROM_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.EMPTY_LIQUID_FROM_OBJECT
    action_group: ActionGroup = ActionGroup.LIQUID_MANIPULATION_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.CAN_FILL_WITH_LIQUID
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})


@dataclass(frozen=True)
class BreakObjectEnvAction(EnvironmentAction):
    """Base class for break object action."""

    name: EnvActionName = EnvActionName.BREAK_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.BREAK_OBJECT
    action_group: ActionGroup = ActionGroup.BREAK_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.BREAKABLE
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})


@dataclass(frozen=True)
class SliceObjectEnvAction(ConditionalExecutionAction):
    """Base class for slice object action."""

    name: EnvActionName = EnvActionName.SLICE_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.SLICE_OBJECT
    action_group: ActionGroup = ActionGroup.SLICE_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.SLICEABLE
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})
    action_condition: HoldingObjectTypeCondition = slice_object_condition


@dataclass(frozen=True)
class UseUpObjectEnvAction(EnvironmentAction):
    """Base class for use up object action."""

    name: EnvActionName = EnvActionName.USE_UP_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.USE_UP_OBJECT
    action_group: ActionGroup = ActionGroup.USE_UP_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.CAN_BE_USED_UP
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})


# TODO: Make every action a conditional execution action and merge dirty and clean actions classes
@dataclass(frozen=True)
# * Unused
class DirtyObjectEnvAction(EnvironmentAction):
    """Base class for dirty object action."""

    name: EnvActionName = EnvActionName.DIRTY_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.DIRTY_OBJECT
    action_group: ActionGroup = ActionGroup.CLEAN_DIRTY_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.DIRTYABLE
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})


@dataclass(frozen=True)
# * Unused
class CleanObjectEnvAction(ConditionalExecutionAction):
    """Base class for clean object action."""

    name: EnvActionName = EnvActionName.CLEAN_OBJECT
    ai2thor_action: Ai2thorAction = Ai2thorAction.CLEAN_OBJECT
    action_group: ActionGroup = ActionGroup.CLEAN_DIRTY_ACTIONS
    has_target_object: bool = True
    _object_required_property: SimObjFixedProp = SimObjFixedProp.DIRTYABLE
    _config_dependent_parameters: frozenset[str] = frozenset({"forceAction"})
    action_condition: BaseActionCondition = clean_object_condition


# %% === Action definitions ===
# === Navigation actions ===
move_ahead_action = MovementEnvAction(
    name=EnvActionName.MOVE_AHEAD,
    ai2thor_action=Ai2thorAction.MOVE_AHEAD,
)
move_back_action = MovementEnvAction(
    name=EnvActionName.MOVE_BACK,
    ai2thor_action=Ai2thorAction.MOVE_BACK,
)
move_left_action = MovementEnvAction(
    name=EnvActionName.MOVE_LEFT,
    ai2thor_action=Ai2thorAction.MOVE_LEFT,
)
move_right_action = MovementEnvAction(
    name=EnvActionName.MOVE_RIGHT,
    ai2thor_action=Ai2thorAction.MOVE_RIGHT,
)
rotate_left_action = RotationEnvAction(
    name=EnvActionName.ROTATE_LEFT,
    ai2thor_action=Ai2thorAction.ROTATE_LEFT,
)
rotate_right_action = RotationEnvAction(
    name=EnvActionName.ROTATE_RIGHT,
    ai2thor_action=Ai2thorAction.ROTATE_RIGHT,
)
look_up_action = HeadMovementEnvAction(
    name=EnvActionName.LOOK_UP,
    ai2thor_action=Ai2thorAction.LOOK_UP,
)
look_down_action = HeadMovementEnvAction(
    name=EnvActionName.LOOK_DOWN,
    ai2thor_action=Ai2thorAction.LOOK_DOWN,
)
crouch_action = CrouchStandEnvAction(
    name=EnvActionName.CROUCH,
    ai2thor_action=Ai2thorAction.CROUCH,
)
stand_action = CrouchStandEnvAction(
    name=EnvActionName.STAND,
    ai2thor_action=Ai2thorAction.STAND,
)
# done_action = EnvironmentAction(
#     name=EnvActionName.DONE,
#     ai2thor_action=Ai2thorAction.DONE,
#     action_group=ActionGroup.DONE_ACTIONS,
# )
# === Object manipulation actions ===
pickup_object_action = PickupPutEnvAction(
    name=EnvActionName.PICKUP_OBJECT,
    ai2thor_action=Ai2thorAction.PICKUP_OBJECT,
    _object_required_property=SimObjFixedProp.PICKUPABLE,
    _config_dependent_parameters=frozenset({"forceAction", "manualInteract"}),
)
put_object_action = PickupPutEnvAction(
    name=EnvActionName.PUT_OBJECT,
    ai2thor_action=Ai2thorAction.PUT_OBJECT,
    _object_required_property=SimObjFixedProp.RECEPTACLE,
    _config_dependent_parameters=frozenset({"forceAction", "placeStationary"}),
)
drop_hand_object_action = DropHandObjectEnvAction()
throw_object_action = ThrowEnvAction()
push_object_action = PushPullEnvAction(
    name=EnvActionName.PUSH_OBJECT,
    ai2thor_action=Ai2thorAction.PUSH_OBJECT,
)
pull_object_action = PushPullEnvAction(
    name=EnvActionName.PULL_OBJECT,
    ai2thor_action=Ai2thorAction.PULL_OBJECT,
)
move_held_object_ahead_back_action = HandMovementEnvAction(
    name=EnvActionName.MOVE_HELD_OBJECT_AHEAD_BACK,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    _parameter_name="ahead",
    _other_ai2thor_parameters={"right": 0, "up": 0},
)
move_held_object_right_left_action = HandMovementEnvAction(
    name=EnvActionName.MOVE_HELD_OBJECT_RIGHT_LEFT,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    _parameter_name="right",
    _other_ai2thor_parameters={"ahead": 0, "up": 0},
)
move_held_object_up_down_action = HandMovementEnvAction(
    name=EnvActionName.MOVE_HELD_OBJECT_UP_DOWN,
    ai2thor_action=Ai2thorAction.MOVE_HELD_OBJECT,
    _parameter_name="up",
    _other_ai2thor_parameters={"ahead": 0, "right": 0},
)
rotate_held_object_roll_action = HandRotationEnvAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_ROLL,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    _parameter_name="roll",
    _other_ai2thor_parameters={"pitch": 0, "yaw": 0},
)
rotate_held_object_pitch_action = HandRotationEnvAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_PITCH,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    _parameter_name="pitch",
    _other_ai2thor_parameters={"roll": 0, "yaw": 0},
)
rotate_held_object_yaw_action = HandRotationEnvAction(
    name=EnvActionName.ROTATE_HELD_OBJECT_YAW,
    ai2thor_action=Ai2thorAction.ROTATE_HELD_OBJECT,
    _parameter_name="yaw",
    _other_ai2thor_parameters={"roll": 0, "pitch": 0},
)
# === Object interaction actions ===
partial_open_object_action = EnvironmentAction(
    name=EnvActionName.PARTIAL_OPEN_OBJECT,
    ai2thor_action=Ai2thorAction.OPEN_OBJECT,
    action_group=ActionGroup.SPECIAL,
    _parameter_name="openness",
    _parameter_range=(0, 1),
    has_target_object=True,
    _object_required_property=SimObjFixedProp.OPENABLE,
    _config_dependent_parameters=frozenset({"forceAction"}),
)
open_object_action = OpenCloseEnvAction(
    name=EnvActionName.OPEN_OBJECT,
    ai2thor_action=Ai2thorAction.OPEN_OBJECT,
)
close_object_action = OpenCloseEnvAction(
    name=EnvActionName.CLOSE_OBJECT,
    ai2thor_action=Ai2thorAction.CLOSE_OBJECT,
)
toggle_object_on_action = ToggleEnvAction(
    name=EnvActionName.TOGGLE_OBJECT_ON,
    ai2thor_action=Ai2thorAction.TOGGLE_OBJECT_ON,
)
toggle_object_off_action = ToggleEnvAction(
    name=EnvActionName.TOGGLE_OBJECT_OFF,
    ai2thor_action=Ai2thorAction.TOGGLE_OBJECT_OFF,
)
fill_object_with_liquid_action = FillObjectWithLiquidEnvAction()
empty_liquid_from_object_action = EmptyLiquidFromObjectEnvAction()
break_object_action = BreakObjectEnvAction()
slice_object_action = SliceObjectEnvAction()
use_up_object_action = UseUpObjectEnvAction()
# * Unused
dirty_object_action = DirtyObjectEnvAction()
# * Unused
clean_object_action = CleanObjectEnvAction()

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
# done_action: EnvironmentAction
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
    # done_action, # Not supported in tasks yet
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
    # dirty_object_action,
    # clean_object_action,
]
ACTIONS_BY_GROUP: dict[ActionGroup, list[EnvironmentAction]]
ACTIONS_BY_GROUP = {category: [] for category in ActionGroup}
for action in ALL_ACTIONS:
    category = action.action_group
    ACTIONS_BY_GROUP[category].append(action)

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
        super().__init__(
            f"Action {self.environment_action.name} requires a parameter but no parameter range is defined."
        )


class NotDiscreteCompatibleActionTypeError(ValueError):
    """Exception raised when an action is not compatible with the discrete environment mode."""

    def __init__(self, environment_action: EnvironmentAction) -> None:
        self.environment_action = environment_action
        super().__init__(f"Action {self.environment_action.name} is not compatible with the discrete environment mode.")


class UnknownConfigDependentParameterError(ValueError):
    """
    Exception raised when an unknown config-dependent parameter is found in the action definition.

    This should not happen and there is a problem in the action definition in the environment.
    """

    def __init__(self, environment_action: EnvironmentAction, parameter_name: str) -> None:
        self.environment_action = environment_action
        self.parameter_name = parameter_name
        super().__init__(f"Unknown config-dependent parameter {parameter_name} in action {self.environment_action}!")
