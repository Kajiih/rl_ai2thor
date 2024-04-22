"""
Item Properties for AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from typing import Any

from rl_ai2thor.envs.sim_objects import (
    COOKING_SOURCES,
    HEAT_SOURCES,
    WATER_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjProp,
    SimObjVariableProp,
)
from rl_ai2thor.envs.tasks.item_prop_interface import (
    FillableLiquid,
    ItemFixedProp,
    ItemProp,
    ItemVariableProp,
    MultiValuePSF,
    PropAuxProp,
    SingleValuePSF,
    TemperatureValue,
)
from rl_ai2thor.envs.tasks.items import AuxItem


# %% === Fixed Item Properties ===
class ObjectTypeProp(ItemFixedProp[SimObjectType]):
    """Object type item property."""

    target_ai2thor_property = SimObjFixedProp.OBJECT_TYPE


class IsInteractableProp(ItemFixedProp[bool]):
    """Is interactable item property."""

    target_ai2thor_property = SimObjFixedProp.IS_INTERACTABLE


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


class CookableProp(ItemFixedProp[bool]):
    """Is cookable item property."""

    target_ai2thor_property = SimObjFixedProp.COOKABLE


class IsHeatSourceProp(ItemFixedProp[bool]):
    """Is heat source item property."""

    target_ai2thor_property = SimObjFixedProp.IS_HEAT_SOURCE


class IsColdSourceProp(ItemFixedProp[bool]):
    """Is cold source item property."""

    target_ai2thor_property = SimObjFixedProp.IS_COLD_SOURCE


class SliceableProp(ItemFixedProp[bool]):
    """Is sliceable item property."""

    target_ai2thor_property = SimObjFixedProp.SLICEABLE


class OpenableProp(ItemFixedProp[bool]):
    """Is openable item property."""

    target_ai2thor_property = SimObjFixedProp.OPENABLE


class PickupableProp(ItemFixedProp[bool]):
    """Is pickupable item property."""

    target_ai2thor_property = SimObjFixedProp.PICKUPABLE


class MoveableProp(ItemFixedProp[bool]):
    """Is moveable item property."""

    target_ai2thor_property = SimObjFixedProp.MOVEABLE


# %% === Variable Item Properties ===
class VisibleProp(ItemVariableProp[bool, Any]):
    """Visible item property."""

    target_ai2thor_property = SimObjVariableProp.VISIBLE


class IsPickedUpProp(ItemVariableProp[bool, bool]):
    """Is picked up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_PICKED_UP
    candidate_required_prop = PickupableProp(True)


class IsToggledProp(ItemVariableProp[bool, bool]):
    """Is toggled item property."""

    target_ai2thor_property = SimObjVariableProp.IS_TOGGLED
    candidate_required_prop = ToggleableProp(True)


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


# TODO: Support filling with other liquids and contextual interactions
# TODO: Add sink as an auxiliary item and a contained_in relation
class IsFilledWithLiquidProp(ItemVariableProp[bool, bool]):
    """Is filled with liquid item property."""

    target_ai2thor_property = SimObjVariableProp.IS_FILLED_WITH_LIQUID
    candidate_required_prop = CanFillWithLiquidProp(True)
    auxiliary_items = frozenset({
        AuxItem(
            t_id="water_source",
            properties={
                ObjectTypeProp(MultiValuePSF(WATER_SOURCES)),
                IsToggledProp(True),
            },
        )
    })


# TODO: Support filling with other liquids and contextual interactions
# TODO: Add IsFilledWithLiquidProp as auxiliary property
class FillLiquidProp(ItemVariableProp[FillableLiquid, bool]):
    """Fill liquid item property."""

    target_ai2thor_property = SimObjVariableProp.FILL_LIQUID
    candidate_required_prop = CanFillWithLiquidProp(True)


# TODO: Add sink as an auxiliary item and a contained_in relation
class IsDirtyProp(ItemVariableProp[bool, bool]):
    """Is dirty item property."""

    target_ai2thor_property = SimObjVariableProp.IS_DIRTY
    candidate_required_prop = DirtyableProp(True)
    auxiliary_items = frozenset({
        AuxItem(
            t_id="water_source",
            properties={
                ObjectTypeProp(MultiValuePSF(WATER_SOURCES)),
                IsToggledProp(True),
            },
        )
    })


# TODO: Implement contextual cooking interactions (e.g. toaster...)
# TODO: Implement cooking with Microwave that requires to be open to put the object inside first.
# TODO: Add a contained_in relation with cooking_source instead of the auxiliary property
class IsCookedProp(ItemVariableProp[bool, bool]):
    """
    Property for cooked items.

    Currently only supports cooking with a StoveBurner.
    """

    target_ai2thor_property = SimObjVariableProp.IS_COOKED
    candidate_required_prop = CookableProp(True)
    auxiliary_properties = frozenset({PropAuxProp(IsPickedUpProp, True)})
    auxiliary_items = frozenset({
        AuxItem(
            t_id="cooking_source",
            properties={
                ObjectTypeProp(MultiValuePSF(COOKING_SOURCES)),
                IsToggledProp(True),
            },
        )
    })


# TODO: Implement contextual temperature interactions (e.g. coffee machine and mugs...)
# TODO: Add the fact that the Microwave has to be open to put the object inside first then closed then turned on.
# TODO: Add the fact that the Fridge has to be open to put the object inside first then closed.
# TODO: Add a contained_in relation with heat_source or cold_source instead of the auxiliary property
class TemperatureProp(ItemVariableProp[TemperatureValue, Any]):
    """
    Property for items with a certain temperature.

    Currently only supports StoveBurner and Microwave as heat sources and Fridge as a cold source.

    Note: can currently only be used with a target property_function that is a SingleValuePSF.
    """

    target_ai2thor_property = SimObjVariableProp.TEMPERATURE
    auxiliary_properties = frozenset({PropAuxProp(IsPickedUpProp, True)})

    def __init__(self, target_satisfaction_function: SingleValuePSF[TemperatureValue] | TemperatureValue) -> None:
        """Initialize the Property object."""
        super().__init__(target_satisfaction_function)
        self.target_satisfaction_function: SingleValuePSF[TemperatureValue]

        if self.target_satisfaction_function.target_value == TemperatureValue.HOT:
            self.auxiliary_items = frozenset({
                AuxItem(
                    t_id="heat_source",
                    properties={
                        ObjectTypeProp(MultiValuePSF(HEAT_SOURCES)),
                        IsToggledProp(True),
                    },
                )
            })
        elif self.target_satisfaction_function.target_value == TemperatureValue.COLD:
            self.auxiliary_items = frozenset({
                AuxItem(
                    t_id="cold_source",
                    properties={
                        ObjectTypeProp(MultiValuePSF([SimObjectType.FRIDGE])),
                    },
                )
            })


# TODO: Implement the fact that Eggs can be sliced (cracked) without a knife.
class IsSlicedProp(ItemVariableProp[bool, bool]):
    """Is sliced item property."""

    target_ai2thor_property = SimObjVariableProp.IS_SLICED
    candidate_required_prop = SliceableProp(True)
    auxiliary_items = frozenset({
        AuxItem(
            t_id="knife",
            properties={
                ObjectTypeProp(SimObjectType.KNIFE),
                IsPickedUpProp(True),
            },
        )
    })


## %% === Item property mapping ===
obj_prop_id_to_item_prop: dict[SimObjProp | str, type[ItemProp]]
obj_prop_id_to_item_prop = {
    SimObjFixedProp.OBJECT_TYPE: ObjectTypeProp,
    SimObjFixedProp.IS_INTERACTABLE: IsInteractableProp,
    SimObjFixedProp.RECEPTACLE: ReceptacleProp,
    SimObjFixedProp.TOGGLEABLE: ToggleableProp,
    SimObjFixedProp.BREAKABLE: BreakableProp,
    SimObjFixedProp.CAN_FILL_WITH_LIQUID: CanFillWithLiquidProp,
    SimObjFixedProp.DIRTYABLE: DirtyableProp,
    SimObjFixedProp.CAN_BE_USED_UP: CanBeUsedUpProp,
    SimObjFixedProp.COOKABLE: CookableProp,
    SimObjFixedProp.IS_HEAT_SOURCE: IsHeatSourceProp,
    SimObjFixedProp.IS_COLD_SOURCE: IsColdSourceProp,
    SimObjFixedProp.SLICEABLE: SliceableProp,
    SimObjFixedProp.OPENABLE: OpenableProp,
    SimObjFixedProp.PICKUPABLE: PickupableProp,
    SimObjFixedProp.MOVEABLE: MoveableProp,
    SimObjVariableProp.VISIBLE: VisibleProp,
    SimObjVariableProp.IS_TOGGLED: IsToggledProp,
    SimObjVariableProp.IS_BROKEN: IsBrokenProp,
    SimObjVariableProp.IS_FILLED_WITH_LIQUID: IsFilledWithLiquidProp,
    SimObjVariableProp.FILL_LIQUID: FillLiquidProp,
    SimObjVariableProp.IS_DIRTY: IsDirtyProp,
    SimObjVariableProp.IS_USED_UP: IsUsedUpProp,
    SimObjVariableProp.IS_COOKED: IsCookedProp,
    SimObjVariableProp.TEMPERATURE: TemperatureProp,
    SimObjVariableProp.IS_SLICED: IsSlicedProp,
    SimObjVariableProp.IS_OPEN: IsOpenProp,
    SimObjVariableProp.OPENNESS: OpennessProp,
    SimObjVariableProp.IS_PICKED_UP: IsPickedUpProp,
}


# %% === Exceptions ===
class UndefinedPSFCalledError(Exception):
    """Exception raised when an UndefinedPSF is called."""

    def __init__(self, prop_value: Any) -> None:
        self.prop_value = prop_value

    def __str__(self) -> str:
        return f"UndefinedPSF should not be called, if the candidate_required_prop attribute of an item property is not None, a property satisfaction function should be properly defined. UndefinedPSF called with value: {self.prop_value}"
