"""
Variable item properties module for RL-THOR environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from typing import Any

from rl_ai2thor.envs.sim_objects import (
    SimObjVariableProp,
)
from rl_ai2thor.envs.tasks._item_prop_fixed import (
    BreakableProp,
    CanBeUsedUpProp,
    OpenableProp,
    PickupableProp,
    ToggleableProp,
)
from rl_ai2thor.envs.tasks.item_prop_interface import (
    ItemVariableProp,
    PropAuxProp,
)


# %% === Property Definitions ===
class VisibleProp(ItemVariableProp[bool, Any]):
    """Visible item property."""

    target_ai2thor_property = SimObjVariableProp.VISIBLE


class IsInteractableProp(ItemVariableProp[bool, Any]):
    """Is interactable item property."""

    target_ai2thor_property = SimObjVariableProp.IS_INTERACTABLE


class IsPickedUpProp(ItemVariableProp[bool, bool]):
    """Is picked up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_PICKED_UP
    candidate_required_prop = PickupableProp(True)


class IsToggledProp(ItemVariableProp[bool, bool]):
    """Is toggled item property."""

    target_ai2thor_property = SimObjVariableProp.IS_TOGGLED
    candidate_required_prop = ToggleableProp(True)


class IndirectIsToggledProp(IsToggledProp):
    """
    Extension of IsToggledProp that works with objects that are indirectly toggled.

    Indirectly toggled objects (e.g. StoveBurner), don't have the Toggleable fixed property.
    """

    candidate_required_prop = None


class IsUsedUpProp(ItemVariableProp[bool, bool]):
    """Is used up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_USED_UP
    candidate_required_prop = CanBeUsedUpProp(True)


class IsOpenProp(ItemVariableProp[bool, bool]):
    """Is open item property."""

    target_ai2thor_property = SimObjVariableProp.IS_OPEN
    candidate_required_prop = OpenableProp(True)


class OpennessProp(ItemVariableProp[float, bool]):
    """Openness item property."""

    target_ai2thor_property = SimObjVariableProp.OPENNESS
    candidate_required_prop = OpenableProp(True)


class IsBrokenProp(ItemVariableProp[bool, bool]):
    """Is broken item property."""

    target_ai2thor_property = SimObjVariableProp.IS_BROKEN
    candidate_required_prop = BreakableProp(True)
    auxiliary_properties = frozenset({PropAuxProp(IsPickedUpProp, True)})