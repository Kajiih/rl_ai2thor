"""
Fixed item properties module for RL-THOR environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from rl_ai2thor.envs.sim_objects import (
    OPENABLES,
    SimObjectType,
    SimObjFixedProp,
    SimObjMetadata,
)
from rl_ai2thor.envs.tasks.item_prop_interface import (
    ItemFixedProp,
    SingleValuePSF,
)


# %% === Property Definitions ===
class ObjectTypeProp(ItemFixedProp[SimObjectType]):
    """Object type item property."""

    target_ai2thor_property = SimObjFixedProp.OBJECT_TYPE


class ReceptacleProp(ItemFixedProp[bool]):
    """Is receptacle item property."""

    target_ai2thor_property = SimObjFixedProp.RECEPTACLE


class ToggleableProp(ItemFixedProp[bool]):
    """Is toggleable item property."""

    target_ai2thor_property = SimObjFixedProp.TOGGLEABLE


class BreakableProp(ItemFixedProp[bool]):
    """Is breakable item property."""

    target_ai2thor_property = SimObjFixedProp.BREAKABLE


class CanFillWithLiquidProp(ItemFixedProp[bool]):
    """Can fill with liquid item property."""

    target_ai2thor_property = SimObjFixedProp.CAN_FILL_WITH_LIQUID


class DirtyableProp(ItemFixedProp[bool]):
    """Is dirtyable item property."""

    target_ai2thor_property = SimObjFixedProp.DIRTYABLE


class CanBeUsedUpProp(ItemFixedProp[bool]):
    """Can be used up item property."""

    target_ai2thor_property = SimObjFixedProp.CAN_BE_USED_UP


class BaseCookableProp(ItemFixedProp[bool]):
    """Is cookable item property."""

    target_ai2thor_property = SimObjFixedProp.COOKABLE


# TODO: Implement separately the True and False cases (in BoolItemProp class?) to simplify the is_object_satisfying method.
class CookableProp(BaseCookableProp):
    """Extension of BaseCookableProp that works with items whose sliced versions are cookable."""

    def __init__(
        self,
        target_satisfaction_function: SingleValuePSF[bool] | bool,
    ) -> None:
        """
        Initialize the CookableProp object.

        target_satisfaction_function attribute can only be a SingleValuePSF.
        """
        super().__init__(target_satisfaction_function)
        self.target_satisfaction_function: SingleValuePSF[bool]

    def is_object_satisfying(self, obj_metadata: SimObjMetadata) -> bool:
        """Return true if the property is satisfied by the object."""
        target_bool = self.target_satisfaction_function.target_value
        obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
        # Add `Bread` and `Egg` objects as objects considered cookable.
        if obj_type in {SimObjectType.BREAD, SimObjectType.EGG}:
            return target_bool
        return super().is_object_satisfying(obj_metadata)


class IsHeatSourceProp(ItemFixedProp[bool]):
    """Is heat source item property."""

    target_ai2thor_property = SimObjFixedProp.IS_HEAT_SOURCE


class IsColdSourceProp(ItemFixedProp[bool]):
    """Is cold source item property."""

    target_ai2thor_property = SimObjFixedProp.IS_COLD_SOURCE


# TODO? Add extension for sliced objects to be sliceable..?
class SliceableProp(ItemFixedProp[bool]):
    """Is sliceable item property."""

    target_ai2thor_property = SimObjFixedProp.SLICEABLE


# TODO: Implement separately the True and False cases (in BoolItemProp class?) to simplify the is_object_satisfying method.
class OpenableProp(ItemFixedProp[bool]):
    """Is openable item property."""

    target_ai2thor_property = SimObjFixedProp.OPENABLE

    def __init__(
        self,
        target_satisfaction_function: SingleValuePSF[bool] | bool,
    ) -> None:
        """
        Initialize the OpenableProp object.

        target_satisfaction_function attribute can only be a SingleValuePSF.
        """
        super().__init__(target_satisfaction_function)
        self.target_satisfaction_function: SingleValuePSF[bool]

    def is_object_satisfying(self, obj_metadata: SimObjMetadata) -> bool:
        """
        Return true if the property is satisfied by the object.

        We need to add the check for the object type to be in the OPENABLES list because `Blinds`
        are openable but they cause a `TimeoutError` when trying to open or close them.
        """
        target_bool = self.target_satisfaction_function.target_value
        object_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
        # Add `Blinds` object as not openable.
        if object_type not in OPENABLES:
            return not target_bool
        return super().is_object_satisfying(obj_metadata)


class PickupableProp(ItemFixedProp[bool]):
    """Is pickupable item property."""

    target_ai2thor_property = SimObjFixedProp.PICKUPABLE


class MoveableProp(ItemFixedProp[bool]):
    """Is moveable item property."""

    target_ai2thor_property = SimObjFixedProp.MOVEABLE
