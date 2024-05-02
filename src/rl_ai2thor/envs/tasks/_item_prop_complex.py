"""
Complex item properties module for RL-THOR environment.

Complex item properties are item properties that contain auxiliary items or properties in their
definition.
In particular, they currently can't be used as auxiliary properties for relations because it would
cause a circular coupling between the item properties and the relations.

TODO: Finish module docstring.
"""

from __future__ import annotations

from typing import Any

from rl_ai2thor.envs.sim_objects import (
    COOKING_SOURCES,
    HEAT_SOURCES,
    SLICED_FORMS,
    WATER_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjMetadata,
    SimObjVariableProp,
)
from rl_ai2thor.envs.tasks._item_prop_fixed import (
    CanFillWithLiquidProp,
    CookableProp,
    DirtyableProp,
    ObjectTypeProp,
    SliceableProp,
)
from rl_ai2thor.envs.tasks._item_prop_variable import IndirectIsToggledProp, IsPickedUpProp, IsToggledProp
from rl_ai2thor.envs.tasks.item_prop_interface import (
    FillableLiquid,
    ItemVariableProp,
    MultiValuePSF,
    SingleValuePSF,
    TemperatureValue,
)
from rl_ai2thor.envs.tasks.items import AuxItem
from rl_ai2thor.envs.tasks.relations import ContainedInRelation, ReceptacleOfRelation


# %% === Property Definitions ===
# TODO: Support filling with other liquids and contextual interactions
# TODO: Add sink as an auxiliary item and a contained_in relation -> doesn't work, we need "fill_liquid" action
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


# TODO: Add sink as an auxiliary item and a contained_in relation -> Check if that works
class IsDirtyProp(ItemVariableProp[bool, bool]):
    """Is dirty item property."""

    target_ai2thor_property = SimObjVariableProp.IS_DIRTY
    candidate_required_prop = DirtyableProp(True)
    auxiliary_items = frozenset({
        AuxItem(
            t_id="water_source",
            properties={
                ObjectTypeProp(MultiValuePSF(WATER_SOURCES)),
                IndirectIsToggledProp(True),
            },
            relation_descriptions={ReceptacleOfRelation: {}},
        )
    })


# TODO: Implement better handling for StoveBurner not being directly toggleable.
# TODO: Implement contextual cooking interactions (e.g. toaster...)
# TODO: Implement cooking with Microwave that requires to be open to put the object inside first.
class IsCookedProp(ItemVariableProp[bool, bool]):
    """
    Property for cooked items.

    Currently only supports cooking with a StoveBurner.
    """

    target_ai2thor_property = SimObjVariableProp.IS_COOKED
    candidate_required_prop = CookableProp(True)
    auxiliary_items = frozenset({
        AuxItem(
            t_id="cooking_source",
            properties={
                ObjectTypeProp(MultiValuePSF(COOKING_SOURCES)),
                IndirectIsToggledProp(True),
            },
            relation_descriptions={ReceptacleOfRelation: {}},
        )
    })


# TODO: Implement better handling for StoveBurner not being directly toggleable.
# TODO: Implement contextual temperature interactions (e.g. coffee machine and mugs...)
# TODO: Add the fact that the Microwave has to be open to put the object inside first then closed then turned on.
# TODO: Add the fact that the Fridge has to be open to put the object inside first then closed.
class TemperatureProp(ItemVariableProp[TemperatureValue, Any]):
    """
    Property for items with a certain temperature.

    Currently only supports StoveBurner and Microwave as heat sources and Fridge as a cold source.

    Note: can currently only be used with a target property_function that is a SingleValuePSF.
    """

    target_ai2thor_property = SimObjVariableProp.TEMPERATURE

    def __init__(self, target_satisfaction_function: SingleValuePSF[TemperatureValue] | TemperatureValue) -> None:
        """Initialize the Property object."""
        if isinstance(target_satisfaction_function, TemperatureValue):
            target_satisfaction_function = SingleValuePSF(target_satisfaction_function)

        if target_satisfaction_function.target_value == TemperatureValue.HOT:
            self.auxiliary_items = frozenset({
                AuxItem(
                    t_id="heat_source",
                    properties={
                        ObjectTypeProp(MultiValuePSF(HEAT_SOURCES)),
                        IsToggledProp(True),
                    },
                    relation_descriptions={ReceptacleOfRelation: {}},
                )
            })
        elif target_satisfaction_function.target_value == TemperatureValue.COLD:
            self.auxiliary_items = frozenset({
                AuxItem(
                    t_id="cold_source",
                    properties={
                        ObjectTypeProp(MultiValuePSF([SimObjectType.FRIDGE])),
                    },
                    relation_descriptions={ReceptacleOfRelation: {}},
                )
            })

        super().__init__(target_satisfaction_function)
        self.target_satisfaction_function: SingleValuePSF[TemperatureValue]


class BaseIsSlicedProp(ItemVariableProp[bool, bool]):
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


class IsSlicedProp(BaseIsSlicedProp):
    """Extension of BaseIsSlicedProp that counts the slice themselves as sliced."""

    def is_object_satisfying(self, obj_metadata: SimObjMetadata) -> bool:
        """Return true if the object is sliced or if it is an Egg object."""
        return super().is_object_satisfying(obj_metadata) or obj_metadata[SimObjFixedProp.OBJECT_TYPE] in SLICED_FORMS
