"""
Gymnasium interface for ai2thor environment.
"""
import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import ai2thor.controller
import ai2thor.server
import gymnasium as gym
import numpy as np
import yaml
from numpy.typing import ArrayLike

from utils import update_nested_dict, nested_dict_get


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
        custom_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            custom_config (dict): Dictionary whose keys will override the default config.
        """
        # === Get full config ===
        with open("config/general.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        # Merge enviroment mode config with general config
        with open(f"{self.config['enviroment_mode']}.yaml", "r", encoding="utf-8") as f:
            enviroment_mode_config = yaml.safe_load(f)
        update_nested_dict(self.config, enviroment_mode_config)
        # Update config with user config
        if custom_config is not None:
            update_nested_dict(self.config, custom_config)

        # === Get action space ===
        self.action_availablities = {action.name: False for action in ALL_ACTIONS}
        # Update the available actions with the environment mode config
        for action_category in self.config["action_categories"]:
            if action_category in ACTION_CATEGORIES:
                if self.config["action_categories"][action_category]:
                    self.action_availablities[action_category] = True
            else:
                raise ValueError(
                    f"Unknown action category {action_category} in environment mode config."
                )
        # Handle specific action cases
        if self.config["simple_movement_actions"]:
            self.action_availablities["MoveBack"] = False
            self.action_availablities["MoveLeft"] = False
            self.action_availablities["MoveRight"] = False
        if self.config["use_done_action"]:
            self.action_availablities["Done"] = True
        if (
            self.config["partial_openness"]
            and self.config["open_close_actions"]
            and not self.config["discrete_actions"]
        ):
            self.action_availablities["OpenObject"] = False
            self.action_availablities["CloseObject"] = False

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
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(
                f"Action {action} is not contained in the action space"
            )
        action_name = self.action_idx_to_name[action["action_index"]]
        env_action = ACTIONS_BY_NAME[action_name]
        target_object_coordinates = action.get("target_object_position", None)
        action_parameter = action.get("action_parameter", None)

        # === Identify the target object if needed for the action ===
        failed_action_event = None
        if env_action.has_target_object:
            if self.config["target_closest_object"]:
                # Look for the closest operable object for the action
                visible_objects = [
                    obj for obj in self.last_event.metadata["objects"] if obj["visible"]
                ]
                object_required_property = env_action.object_required_property
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
                    failed_action_event = env_action.fail_perform(
                        env=self,
                        error_message=f"No operable object found to perform action {action_name} in the agent's field of view.",
                    )
                    target_object_id = None
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
                    failed_action_event = env_action.fail_perform(
                        env=self,
                        error_message=f"No object found at position {target_object_coordinates} to perform action {action_name}.",
                    )
                    target_object_id = None
                    # TODO: Implement a range of tolerance
        else:
            target_object_id = None
        # === Perform the action ===
        if failed_action_event is None:
            new_event = env_action.perform(
                self,
                action_parameter=action_parameter,
                target_object_id=target_object_id,
            )
        else:
            new_event = failed_action_event
        # TODO: Add logging of the event, especially when the action fails

        self.step_count += 1

        observation = new_event.frame
        reward = 0
        # TODO: Implement reward
        terminated = False
        truncated = self.step_count >= self.config["max_episode_steps"]
        info = new_event.metadata

        self.last_event = new_event

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> tuple[ArrayLike, dict]:
        """
        Reset the environment.

        TODO: Finish this method
        """
        print("Resetting environment and starting new episode")
        super().reset(seed=seed)

        self.last_event = self.controller.reset()
        observation = self.last_event.frame  # type: ignore
        info = self.last_event.metadata

        # TODO: Add scene id handling

        # Setup the scene
        # Chose the task
        self.step_count = 0

        return observation, info

    def close(self):
        self.controller.stop()


# %% Action Classes
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
        ) -> ai2thor.server.MultiAgentEvent:
            Perform the action in the environment and return the event.

        fail_perform(
            env (ITHOREnv): Environment in which the action was performed.
            error_message (str): Error message to log in the event.
        ) -> ai2thor.server.MultiAgentEvent:
            Generate an event corresponding to the failure of the action.

    """

    name: str
    ai2thor_action: str
    action_category: str
    _: dataclasses.KW_ONLY  # Following arguments are keyword-only
    has_target_object: bool = False
    object_required_property: Optional[str] = None
    parameter_name: Optional[str] = None
    parameter_range: Optional[tuple[float, float]] = None
    parameter_discrete_value: Optional[float] = None
    other_ai2thor_parameters: dict[str, Any] = field(default_factory=dict)
    config_dependent_parameters: set[str] = field(default_factory=set)

    def perform(
        self,
        env: ITHOREnv,
        action_parameter: Optional[float] = None,
        target_object_id: Optional[str] = None,
    ) -> ai2thor.server.MultiAgentEvent:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.

        Returns:
            event (ai2thor.controller.MultiAgentEvent): Event returned by the controller.
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
            else:
                # Rescale the action parameter
                if self.parameter_range is not None:
                    # Override the range with the value from the config
                    parameter_range = nested_dict_get(
                        d=env.config,
                        keys=["action_parameter_data", self.name, "range"],
                        default=self.parameter_range,
                    )
                    action_parameter = parameter_range[0] + action_parameter * (
                        parameter_range[1] - parameter_range[0]
                    )
                else:
                    raise ValueError(
                        f"Action {self.ai2thor_action} requires a parameter but no parameter range is defined."
                    )
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
        return event

    def fail_perform(
        self,
        env: ITHOREnv,
        error_message: str,
    ) -> ai2thor.server.MultiAgentEvent:
        """
        Generate an event corresponding to the failure of the action.

        Args:
            env (ITHOREnv): Environment in which the action was performed.
            error_message (str): Error message to log in the event.

        Returns:
            event (ai2thor.server.MultiAgentEvent): Event for the failed action.
        """
        event = env.controller.step(action="Done")
        event.metadata["lastAction"] = self.ai2thor_action
        event.metadata["lastActionSuccess"] = False
        event.metadata["errorMessage"] = error_message
        return event


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
        """Default error message for the condition."""
        return f"Agent needs to have visible running water to perform action {action.ai2thor_action}!"


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
        """Default error message for the condition."""
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
    ) -> ai2thor.server.MultiAgentEvent:
        """
        Perform the action in the environment.

        Args:
            env (ITHOREnv): Environment in which to perform the action.
            action_parameter (float, optional): Quantitative parameter of the action.
            target_object_id (str, optional): ID of the target object for the action.

        Returns:
            event (ai2thor.controller.MultiAgentEvent): Event returned by the controller.
        """
        if self.action_condition(env):
            event = super().perform(env, action_parameter, target_object_id)
        else:
            event = self.fail_perform(
                env, error_message=self.action_condition.error_message(self)
            )

        return event


# %% Action definitions
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
    config_dependent_parameters={"forceAction"},
)
put_object_action = EnvironmentAction(
    name="PutObject",
    ai2thor_action="PutObject",
    action_category="pickup_put_actions",
    has_target_object=True,
    object_required_property="receptacle",
    config_dependent_parameters={"forceAction, manualInteract"},
)
drop_hand_object_action = EnvironmentAction(
    name="DropHandObject",
    ai2thor_action="DropHandObject",
    action_category="drop_actions",
    config_dependent_parameters={"forceAction, placeStationary"},
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

ALL_ACTIONS = [
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

ACTION_CATEGORIES = set(action.action_category for action in ALL_ACTIONS)
ACTIONS_BY_CATEGORY = {category: [] for category in ACTION_CATEGORIES}
for action in ALL_ACTIONS:
    category = action.action_category
    ACTIONS_BY_CATEGORY[category].append(action)
ACTIONS_BY_NAME = {action.name: action for action in ALL_ACTIONS}


# %% Task definitions
@dataclass
class BaseTask:
    """
    Base class for tasks in the environment.
    """

    @abstractmethod
    def get_reward(self, event):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError
