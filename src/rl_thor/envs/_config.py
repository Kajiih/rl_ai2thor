"""Configuration module for the RL-THOR environment."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from rl_thor.envs.actions import ActionGroup

if TYPE_CHECKING:
    from rl_thor.envs.tasks.tasks import TaskType
    from rl_thor.envs.tasks.tasks_interface import TaskArgValue


# %% === Keys enums ===
class EnvConfigKeys(StrEnum):
    """Keys for the environment configuration."""

    SEED = "seed"
    MAX_EPISODE_STEPS = "max_episode_steps"
    SCENE_RANDOMIZATION = "scene_randomization"
    CONTROLLER_PARAMETERS = "controller_parameters"
    ACTION_GROUPS = "action_groups"
    ACTION_DISCRETE_PARAM_VALUES = "action_discrete_param_values"
    ACTION_MODIFIERS = "action_modifiers"
    TASKS = "tasks"


# * Unused
class ControllerParametersConfigKeys(StrEnum):
    """Keys for the AI2THOR controller parameters configuration."""

    PLATFORM = "platform"
    VISIBILITY_DISTANCE = "visibility_distance"
    FRAME_WIDTH = "frame_width"
    FRAME_HEIGHT = "frame_height"
    FIELD_OF_VIEW = "field_of_view"


# * Unused
class SceneRandomizationConfigKeys(StrEnum):
    """Keys for the scene randomization configuration."""

    RANDOM_AGENT_SPAWN = "random_agent_spawn"
    RANDOM_OBJECT_SPAWN = "random_object_spawn"
    RANDOM_OBJECT_MATERIALS = "random_object_materials"
    RANDOM_OBJECT_COLORS = "random_object_colors"
    RANDOM_LIGHTING = "random_lighting"


# * Unused
class ActionModifiersConfigKeys(StrEnum):
    """Keys for the action modifiers configuration."""

    DISCRETE_ACTIONS = "discrete_actions"
    TARGET_CLOSEST_OBJECT = "target_closest_object"
    SIMPLE_MOVEMENT_ACTIONS = "simple_movement_actions"
    STATIC_PICKUP = "static_pickup"
    STATIONARY_PLACEMENT = "stationary_placement"
    PARTIAL_OPENNESS = "partial_openness"


# * Unused
class ActionDiscreteParamValuesConfigKeys(StrEnum):
    """Keys for the action discrete parameters configuration."""

    MOVEMENT_MAGNITUDE = "movement_magnitude"
    ROTATION_DEGREES = "rotation_degrees"
    HEAD_MOVEMENT_DEGREES = "head_movement_degrees"
    THROW_STRENGTH = "throw_strength"
    PUSH_PULL_STRENGTH = "push_pull_strength"


class TaskConfigKeys(StrEnum):
    """Keys for the task configuration."""

    GLOBALLY_EXCLUDED_SCENES = "globally_excluded_scenes"
    TASK_BLUEPRINTS = "task_blueprints"


# * Unused
class TaskBlueprintConfigKeys(StrEnum):
    """Keys for the task blueprint configuration."""

    TASK_TYPE = "task_type"
    ARGS = "args"
    SCENES = "scenes"


# %% === Simulator Configuration ===
@dataclass(frozen=True)
class ControllerParametersConfig:
    """Configuration class for the AI2THOR controller parameters."""

    # === General parameters ===
    platform: Literal["CloudRendering"] | None = None
    visibility_distance: float = 1.5
    # === Rendering parameters ===
    # render_depth_image: bool = False # Not supported yet
    # render_instance_segmentation: bool = False # Not supported yet
    frame_width: int = 300
    frame_height: int = 300
    field_of_view: int = 90

    def get_controller_parameters(self) -> dict[str, str | float | bool | None]:
        """Return the controller parameters as a dictionary."""
        return {
            "platform": self.platform,
            "visibilityDistance": self.visibility_distance,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "frameWidth": self.frame_width,
            "frameHeight": self.frame_height,
            "fieldOfView": self.field_of_view,
        }


@dataclass(frozen=True)
class SceneRandomizationConfig:
    """Configuration class for the randomization of the scenes."""

    random_agent_spawn: bool = False
    random_object_spawn: bool = False
    random_lighting: bool = False
    random_object_materials: bool = False
    random_object_colors: bool = False


# %% === Actions Configuration ===


@dataclass(frozen=True)
class ActionModifiersConfig:
    """Configuration class for the RL-THOR environment action modifiers."""

    discrete_actions: bool = True  # TODO: Add checks for not compatible action groups
    target_closest_object: bool = True
    simple_movement_actions: bool = False  # TODO: Add checks for rotation_actions and movement_actions to be True
    static_pickup: bool = False  # TODO: Add checks for pickup_put_actions to be True
    stationary_placement: bool = False  # TODO: Add checks for pickup_put_actions to be True
    partial_openness: bool = (
        False  # TODO: Add checks for open_close_actions to be True and discrete_actions to be False
    )
    force_action: bool = False  # Not supposed to be changed
    force_visible: bool = True  # Not supposed to be changed


# @dataclass
# class ActionGroupsConfig()):
#     """Configuration class for the RL-THOR environment action groups."""

#     # === Navigation actions ===
#     movement_actions: bool = True
#     rotation_actions: bool = True
#     head_movement_actions: bool = True
#     crouch_actions: bool = False
#     # === Object manipulation actions ===
#     pickup_put_actions: bool = True
#     drop_actions: bool = False
#     throw_actions: bool = False
#     push_pull_actions: bool = False
#     hand_control_actions: bool = False
#     # === Object interaction actions ===
#     open_close_actions: bool = True
#     toggle_actions: bool = True
#     slice_actions: bool = False
#     use_up_actions: bool = False
#     liquid_manipulation_actions: bool = False
#     break_actions: bool = False
#     # clean_dirty_actions: bool = False  # Not supported

#     def __post_init__(self) -> None:
#         """Check for incompatible action groups."""
#         if not self.pickup_put_actions:
#             if self.drop_actions:
#                 warnings.warn(
#                     MissingRequiredActionGroupsWarning(ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.DROP_ACTIONS),
#                     stacklevel=2,
#                 )
#                 self.drop_actions = False
#             if self.throw_actions:
#                 warnings.warn(
#                     MissingRequiredActionGroupsWarning(ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.THROW_ACTIONS),
#                     stacklevel=2,
#                 )
#                 self.throw_actions = False
#             if self.hand_control_actions:
#                 warnings.warn(
#                     MissingRequiredActionGroupsWarning(
#                         ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.HAND_CONTROL_ACTIONS
#                     ),
#                     stacklevel=2,
#                 )


class ActionGroupsConfig(dict[ActionGroup, bool]):
    """Configuration class for the RL-THOR environment action groups."""

    @staticmethod
    def _default_action_groups() -> dict[ActionGroup, bool]:
        """Return the default action groups."""
        return {
            ActionGroup.MOVEMENT_ACTIONS: True,
            ActionGroup.ROTATION_ACTIONS: True,
            ActionGroup.HEAD_MOVEMENT_ACTIONS: True,
            ActionGroup.CROUCH_ACTIONS: False,
            ActionGroup.PICKUP_PUT_ACTIONS: True,
            ActionGroup.DROP_ACTIONS: False,
            ActionGroup.THROW_ACTIONS: False,
            ActionGroup.PUSH_PULL_ACTIONS: False,
            ActionGroup.HAND_CONTROL_ACTIONS: False,
            ActionGroup.OPEN_CLOSE_ACTIONS: True,
            ActionGroup.TOGGLE_ACTIONS: True,
            ActionGroup.SLICE_ACTIONS: False,
            ActionGroup.USE_UP_ACTIONS: False,
            ActionGroup.LIQUID_MANIPULATION_ACTIONS: False,
            ActionGroup.BREAK_ACTIONS: False,
        }

    def __init__(self, **action_group_availabilities: bool) -> None:
        """Initialize the ActionGroupsConfig dictionary."""
        for action_group_name in action_group_availabilities:
            if action_group_name.startswith("_") or action_group_name not in ActionGroup:
                raise InvalidActionGroupError(action_group_name)

        action_groups = self._default_action_groups()
        action_groups.update(**action_group_availabilities)
        super().__init__(**action_groups)
        self._verify_coherence()

    def _verify_coherence(self) -> None:
        """Check for incompatible action groups."""
        if not self.get(ActionGroup.PICKUP_PUT_ACTIONS, False):
            if self.get(ActionGroup.DROP_ACTIONS, False):
                warnings.warn(
                    MissingRequiredActionGroupsWarning(ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.DROP_ACTIONS),
                    stacklevel=2,
                )
                self[ActionGroup.DROP_ACTIONS] = False
            if self.get(ActionGroup.THROW_ACTIONS, False):
                warnings.warn(
                    MissingRequiredActionGroupsWarning(ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.THROW_ACTIONS),
                    stacklevel=2,
                )
                self[ActionGroup.THROW_ACTIONS] = False
            if self.get(ActionGroup.HAND_CONTROL_ACTIONS, False):
                warnings.warn(
                    MissingRequiredActionGroupsWarning(
                        ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.HAND_CONTROL_ACTIONS
                    ),
                    stacklevel=2,
                )
                self[ActionGroup.HAND_CONTROL_ACTIONS] = False


@dataclass(frozen=True)
class ActionDiscreteParamValuesConfig:
    """Configuration class for the RL-THOR environment action parameters."""

    # === Navigation actions ===
    movement_magnitude: float = 0.25
    rotation_degrees: float = 45
    head_movement_degrees: float = 30
    # === Object manipulation actions ===
    throw_strength: float = 50
    push_pull_strength: float = 100


# %% === Tasks Configuration ===


@dataclass
class TaskBlueprintConfig:
    """Configuration class for the RL-THOR environment task blueprints."""

    task_type: type[TaskType]
    args: dict[str, TaskArgValue]
    scenes: list[str]

    def __init__(self, task_type: type[TaskType], args: dict[str, TaskArgValue], scenes: list[str] | str) -> None:
        """Initialize the TaskBlueprintConfig object."""
        self.task_type = task_type
        self.args = args
        self.scenes = scenes if isinstance(scenes, list) else [scenes]


@dataclass(frozen=True)
class TaskConfig:
    """Configuration class for the RL-THOR environment tasks."""

    globally_excluded_scenes: list[str] = field(default_factory=list)
    task_blueprints: list[TaskBlueprintConfig] = field(default_factory=list)

    @staticmethod
    def init_from_dict(task_config_dict: dict[str, Any]) -> TaskConfig:
        """Initialize the TaskConfig object from a dictionary."""
        dict_copy = task_config_dict.copy()
        task_blueprints = task_config_dict.get(TaskConfigKeys.TASK_BLUEPRINTS, [])
        if not isinstance(task_blueprints, list):
            task_blueprints = [task_blueprints]
        dict_copy.update({
            TaskConfigKeys.TASK_BLUEPRINTS: [
                TaskBlueprintConfig(**task_blueprint)
                for task_blueprint in task_config_dict.get(TaskConfigKeys.TASK_BLUEPRINTS, [])
            ]
        })
        return TaskConfig(**dict_copy)


# %% === Main Configuration ===
@dataclass(frozen=True)
class EnvConfig:
    """Configuration class for the RL-THOR environment."""

    # === General environment configuration ===
    seed: int = 0
    max_episode_steps: int = 1000
    no_task_advancement_reward: bool = False

    # === Simulator configuration ===
    controller_parameters: ControllerParametersConfig = field(default_factory=ControllerParametersConfig)
    scene_randomization: SceneRandomizationConfig = field(default_factory=SceneRandomizationConfig)
    # === Actions configuration ===
    action_groups: ActionGroupsConfig = field(default_factory=ActionGroupsConfig)
    action_modifiers: ActionModifiersConfig = field(default_factory=ActionModifiersConfig)
    action_discrete_param_values: ActionDiscreteParamValuesConfig = field(
        default_factory=ActionDiscreteParamValuesConfig
    )
    # === Tasks configuration ===
    tasks: TaskConfig = field(default_factory=TaskConfig)

    def __post_init__(self) -> None:
        """Check the coherence of the configuration."""
        self._verify_coherence()

    @staticmethod
    def init_from_dict(env_dict: dict[str, Any]) -> EnvConfig:
        """Initialize the EnvConfig object from a dictionary."""
        dict_copy = env_dict.copy()
        dict_copy.update(
            {
                EnvConfigKeys.CONTROLLER_PARAMETERS: ControllerParametersConfig(
                    **env_dict.get(EnvConfigKeys.CONTROLLER_PARAMETERS, {})
                ),
                EnvConfigKeys.SCENE_RANDOMIZATION: SceneRandomizationConfig(
                    **env_dict.get(EnvConfigKeys.SCENE_RANDOMIZATION, {})
                ),
                EnvConfigKeys.ACTION_GROUPS: ActionGroupsConfig(**env_dict.get(EnvConfigKeys.ACTION_GROUPS, {})),
                EnvConfigKeys.ACTION_MODIFIERS: ActionModifiersConfig(
                    **env_dict.get(EnvConfigKeys.ACTION_MODIFIERS, {})
                ),
                EnvConfigKeys.ACTION_DISCRETE_PARAM_VALUES: ActionDiscreteParamValuesConfig(
                    **env_dict.get(EnvConfigKeys.ACTION_DISCRETE_PARAM_VALUES, {})
                ),
                EnvConfigKeys.TASKS: TaskConfig.init_from_dict(env_dict.get(EnvConfigKeys.TASKS, {})),
            },
        )
        return EnvConfig(**dict_copy)

    # TODO: Create a general method to check implications between configuration values
    def _verify_coherence(self) -> None:
        """Check the coherence of the configuration."""


# %% === Warnings ===
class ConfigWarning(Warning):
    """Base class for configuration warnings."""


class MissingRequiredActionGroupsWarning(ConfigWarning):
    """Warning for missing required action groups."""

    def __init__(self, required_action_group: str, action_group: str) -> None:
        """Initialize the MissingRequiredActionGroupsWarning object."""
        self.required_action_group = required_action_group
        self.action_group = action_group

    def __str__(self) -> str:
        """Return the warning message."""
        return f"Action group '{self.required_action_group}' requires action group '{self.action_group}' to be True. Setting '{self.action_group}' to False."


# %% Exceptions
class InvalidActionGroupError(Exception):
    """Exception for invalid action groups."""

    def __init__(self, action_group: str) -> None:
        """Initialize the InvalidActionGroupError object."""
        self.action_group = action_group

    def __str__(self) -> str:
        """Return the error message."""
        return f"Invalid action group '{self.action_group}', must be one of {[action_group_name for action_group_name in ActionGroup if not action_group_name.startswith("_")]}."
