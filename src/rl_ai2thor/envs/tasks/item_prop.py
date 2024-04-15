"""
Item Properties for AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Container
from enum import StrEnum
from typing import Any

from rl_ai2thor.envs.sim_objects import (
    COOKING_SOURCES,
    HEAT_SOURCES,
    WATER_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjMetadata,
    SimObjProp,
    SimObjVariableProp,
)
from rl_ai2thor.envs.tasks.items import TaskItem


# %% === Property value enums ==
class TemperatureValue(StrEnum):
    """Temperature values."""

    HOT = "Hot"
    COLD = "Cold"
    ROOM_TEMP = "RoomTemp"


class FillableLiquid(StrEnum):
    """Liquid types."""

    WATER = "water"
    # COFFEE = "coffee"
    # WINE = "wine"
    # coffee and wine are not supported yet


ItemPropValue = int | float | bool | TemperatureValue | SimObjectType | FillableLiquid
# type AuxItems = frozenset[TaskItem[str]]
# type AuxProps = frozenset[ItemVariableProp]


# %% === Property satisfaction functions ===
class BasePSF[T: ItemPropValue](ABC):
    """
    Base class for functions used to define the set of acceptable values for a property to be satisfied.

    We call those functions *property satisfaction functions* (PSF).

    T is the type that the property value can take.
    """

    def __init__(self, *args: Any) -> None:
        self._init_args = args

    @abstractmethod
    def __call__(self, prop_value: T) -> bool:
        """Return True if the value satisfies the property."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self._init_args}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._init_args == other._init_args

    def __hash__(self) -> int:
        return hash(self._init_args)


class SingleValuePSF[T: ItemPropValue](BasePSF[T]):
    """Defines a property satisfaction function that only accepts a single value."""

    def __init__(self, target_value: T) -> None:
        """Initialize the target value."""
        super().__init__(target_value)
        self.target_value = target_value

    def __call__(self, prop_value: T) -> bool:
        """Return True if the value is equal to the target value."""
        return prop_value == self.target_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_value})"


class MultiValuePSF[T: ItemPropValue](BasePSF[T]):
    """Defines a property satisfaction function that accepts a set of values."""

    def __init__(self, target_values: Container[T]) -> None:
        """Initialize the target values."""
        super().__init__(target_values)
        self.target_values = target_values

    def __call__(self, prop_value: T) -> bool:
        """Return True if the value is in the target values."""
        return prop_value in self.target_values


class RangePSF(BasePSF[float | int]):
    """Defines a property satisfaction function that accepts a range of values."""

    def __init__(self, min_value: float | int, max_value: float | int) -> None:
        """Initialize the range."""
        super().__init__(min_value, max_value)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, prop_value: float | int) -> bool:
        """Return True if the value is in the range."""
        return self.min_value <= prop_value <= self.max_value


class GenericPSF[T: ItemPropValue](BasePSF[T]):
    """Defines a property satisfaction function with a custom function."""

    def __init__(self, func: Callable[[T], bool]) -> None:
        """Initialize the property satisfaction function."""
        super().__init__(func)
        self.func = func

    def __call__(self, prop_value: T) -> bool:
        """Return the result of the custom function."""
        return self.func(prop_value)


# Unused
class UndefinedPSF(BasePSF[Any]):
    """Defines a property satisfaction function that always returns False."""

    def __call__(self, prop_value: Any) -> bool:
        """Return False."""
        raise UndefinedPSFCalledError(prop_value)


type PropSatFunction[T: ItemPropValue] = BasePSF[T] | Callable[[T], bool]


# %% === Item properties  ===
# TODO? Add action validity checking (action group, etc)
# TODO: Check if we need to add a hash
# TODO: Support multiple candidate required properties
class ItemProp[T1: ItemPropValue, T2: ItemPropValue](ABC):
    """
    Base class for item properties in the definition of a task.

    If the property is fixed (cannot be changed by the agent), the candidate_required_prop
    attribute is the instance itself. If the property is variable (can be changed by the agent),
    the candidate_required_prop attribute has to be defined in the subclass.

    T1 is the type that the property value can take and T2 is the type that the candidate required
    property value can take.

    auxiliary_properties and auxiliary_items are used to define auxiliary goals: the conditions that
    should be satisfied either by the same item this property belongs to or by any other items in
    order to satisfy this property.

    Examples of auxiliary goals include:
    - A knife should be picked up to slice an object
    - The item should be picked up before cooking or cleaning it
    Similar auxiliary goals for relations:
    - The item should be picked up before placing it in a receptacle

    Attributes:
        target_ai2thor_property (SimObjProp): The target AI2-THOR property.
        target_satisfaction_function (PropSatFunction[T1]): The target property satisfaction
            function.
        candidate_required_prop (ItemFixedProp[T2] | None): The candidate required property.
        auxiliary_properties (set[ItemVariableProp] | None): The set of auxiliary properties that
            should be first satisfied in order to satisfy the main property.
        auxiliary_items (set[TaskItem[str]] | None): The set of auxiliary items whose properties
            should be first satisfied by any object in the scene in order to satisfy the main
            property. Those items are not considered in the item-candidates assignments of the task
            since they don't represent a unique task item but only an auxiliary item for a property.

    """

    target_ai2thor_property: SimObjProp
    candidate_required_prop: ItemFixedProp[T2] | None = None
    auxiliary_properties: frozenset[ItemVariableProp] | None = None
    auxiliary_items: frozenset[TaskItem[str]] | None = None

    def __init__(self, target_satisfaction_function: PropSatFunction[T1] | ItemPropValue) -> None:
        """Initialize the Property object."""
        if isinstance(target_satisfaction_function, ItemPropValue):
            target_satisfaction_function = SingleValuePSF(target_satisfaction_function)
        self.target_satisfaction_function = target_satisfaction_function

    def __call__(self, prop_value: T1) -> bool:
        """Return True if the value satisfies the property."""
        return self.target_satisfaction_function(prop_value)

    def is_object_satisfying(self, obj_metadata: SimObjMetadata) -> bool:
        """Return True if the object satisfies the property."""
        return self(obj_metadata[self.target_ai2thor_property])

    def __str__(self) -> str:
        return f"{self.target_ai2thor_property}({self.target_satisfaction_function})"

    def __repr__(self) -> str:
        return f"ItemProp({self.target_ai2thor_property}={self.target_satisfaction_function})"

    # def __eq__(self, other: Any) -> bool:
    #     return (
    #         isinstance(other, ItemProp)
    #         and self.target_ai2thor_property == other.target_ai2thor_property
    #         and self.target_satisfaction_function == other.target_satisfaction_function
    #     )

    # def __hash__(self) -> int:
    #     return hash((self.target_ai2thor_property, self.target_satisfaction_function))


class ItemFixedProp[T: ItemPropValue](ItemProp[T, T]):
    """
    Base class for fixed item properties in the definition of a task.

    Fixed properties are properties that cannot be changed by the agent.

    The candidate_required_prop attribute is the instance itself.
    """

    target_ai2thor_property: SimObjFixedProp

    def __init__(
        self,
        target_satisfaction_function: PropSatFunction[T] | ItemPropValue,
    ) -> None:
        """Initialize the candidate_required_prop attribute with self."""
        super().__init__(target_satisfaction_function)
        self.candidate_required_prop = self


class ItemVariableProp[T1: ItemPropValue, T2: ItemPropValue](ItemProp[T1, T2]):
    """
    Base class for variable item properties in the definition of a task.

    Variable properties are properties that can be changed by the agent.

    The candidate_required_prop attribute has to be defined in the subclass.
    """

    target_ai2thor_property: SimObjVariableProp
    # candidate_required_prop: ItemFixedProp[T2]  # TODO: Delete?


# %% === Item property definitions ===
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
    auxiliary_properties = frozenset({IsPickedUpProp(True)})


# TODO: Support filling with other liquids and contextual interactions
class IsFilledWithLiquidProp(ItemVariableProp[bool, bool]):
    """Is filled with liquid item property."""

    target_ai2thor_property = SimObjVariableProp.IS_FILLED_WITH_LIQUID
    candidate_required_prop = CanFillWithLiquidProp(True)
    auxiliary_items = frozenset({
        TaskItem(
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


class IsDirtyProp(ItemVariableProp[bool, bool]):
    """Is dirty item property."""

    target_ai2thor_property = SimObjVariableProp.IS_DIRTY
    candidate_required_prop = DirtyableProp(True)
    auxiliary_items = frozenset({
        TaskItem(
            t_id="water_source",
            properties={
                ObjectTypeProp(MultiValuePSF(WATER_SOURCES)),
                IsToggledProp(True),
            },
        )
    })


# TODO: Implement contextual cooking interactions (e.g. toaster...)
# TODO: Implement cooking with Microwave that requires to be open to put the object inside first.
class IsCookedProp(ItemVariableProp[bool, bool]):
    """
    Property for cooked items.

    Currently only supports cooking with a StoveBurner.
    """

    target_ai2thor_property = SimObjVariableProp.IS_COOKED
    candidate_required_prop = CookableProp(True)
    auxiliary_properties = frozenset({IsPickedUpProp(True)})
    auxiliary_items = frozenset({
        TaskItem(
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
class TemperatureProp(ItemVariableProp[TemperatureValue, Any]):
    """
    Property for items with a certain temperature.

    Currently only supports StoveBurner and Microwave as heat sources and Fridge as a cold source.

    Note: can currently only be used with a target property_function that is a SingleValuePSF.
    """

    target_ai2thor_property = SimObjVariableProp.TEMPERATURE
    auxiliary_properties = frozenset({IsPickedUpProp(True)})

    def __init__(self, target_satisfaction_function: SingleValuePSF[TemperatureValue] | TemperatureValue) -> None:
        """Initialize the Property object."""
        super().__init__(target_satisfaction_function)
        self.target_satisfaction_function: SingleValuePSF[TemperatureValue]

        if self.target_satisfaction_function.target_value == TemperatureValue.HOT:
            self.auxiliary_items = frozenset({
                TaskItem(
                    t_id="heat_source",
                    properties={
                        ObjectTypeProp(MultiValuePSF(HEAT_SOURCES)),
                        IsToggledProp(True),
                    },
                )
            })
        elif self.target_satisfaction_function.target_value == TemperatureValue.COLD:
            self.auxiliary_items = frozenset({
                TaskItem(
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
        TaskItem(
            t_id="knife",
            properties={
                ObjectTypeProp(SimObjectType.KNIFE),
                IsPickedUpProp(True),
            },
        )
    })


## %% === Item property mapping ===
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
