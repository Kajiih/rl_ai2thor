"""Configuration module for the RL-THOR environment."""

import warnings
from dataclasses import dataclass

from rl_ai2thor.envs.actions import ActionGroup


@dataclass
class Config:
    """Configuration class for the RL-THOR environment."""

    # === Environment Configuration ===


@dataclass
class ActionGroupsConfig:
    """Configuration class for the RL-THOR environment action groups."""

    # === Navigation actions ===
    movement_actions: bool = True
    rotation_actions: bool = True
    head_movement_actions: bool = True
    crouch_actions: bool = False
    # === Object manipulation actions ===
    pickup_put_actions: bool = True
    drop_actions: bool = False
    throw_actions: bool = False
    push_pull_actions: bool = False
    hand_control_actions: bool = False
    # === Object interaction actions ===
    open_close_actions: bool = True
    toggle_actions: bool = True
    slice_actions: bool = False
    use_up_actions: bool = False
    liquid_manipulation_actions: bool = False
    break_actions: bool = False
    # clean_dirty_actions: bool = False  # Not supported

    def __post_init__(self) -> None:
        """Check for incompatible action groups."""
        if not self.pickup_put_actions:
            if self.drop_actions:
                warnings.warn(
                    MissingRequiredActionGroupsWarning(ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.DROP_ACTIONS),
                    stacklevel=2,
                )
                self.drop_actions = False
            if self.throw_actions:
                warnings.warn(
                    MissingRequiredActionGroupsWarning(ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.THROW_ACTIONS),
                    stacklevel=2,
                )
                self.throw_actions = False
            if self.hand_control_actions:
                warnings.warn(
                    MissingRequiredActionGroupsWarning(
                        ActionGroup.PICKUP_PUT_ACTIONS, ActionGroup.HAND_CONTROL_ACTIONS
                    ),
                    stacklevel=2,
                )


@dataclass
class ActionParametersConfig:
    """Configuration class for the RL-THOR environment action parameters."""

    # === Navigation actions ===
    movement_speed: float = 1.0
    rotation_speed: float = 90.0
    head_rotation_speed: float = 90.0
    crouch_speed: float = 1.0
    # === Object manipulation actions ===
    pickup_distance: float = 0.5
    throw_distance: float = 1.0
    push_pull_distance: float = 0.5
    hand_control_distance: float = 0.5
    # === Object interaction actions ===
    open_close_distance: float = 0.5
    toggle_distance: float = 0.5
    slice_distance: float = 0.5
    use_up_distance: float = 0.5
    liquid_manipulation_distance: float = 0.5
    break_distance: float = 0.5
    # clean_dirty_distance: float = 0.5  # Not supported


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
