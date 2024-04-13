"""
Item Properties for AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Container
from enum import StrEnum
from typing import Any

from rl_ai2thor.envs.sim_objects import SimObjectType, SimObjFixedProp, SimObjProp, SimObjVariableProp


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


type ItemPropValue = int | float | bool | TemperatureValue | SimObjectType | FillableLiquid


# %% === Property satisfaction functions ===
class BasePSF[T: ItemPropValue](ABC):
    """
    Base class for functions used to define the set of acceptable values for a property to be satisfied.

    We call those functions *property satisfaction functions* (PSF).

    T is the type that the property value can take.
    """

    @abstractmethod
    def __call__(self, prop_value: T) -> bool:
        """Return True if the value satisfies the property."""


class SingleValuePSF[T: ItemPropValue](BasePSF[T]):
    """Defines a property satisfaction function that only accepts a single value."""

    def __init__(self, target_value: T) -> None:
        """Initialize the target value."""
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
        self.target_values = target_values

    def __call__(self, prop_value: T) -> bool:
        """Return True if the value is in the target values."""
        return prop_value in self.target_values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_values})"


class RangePSF(BasePSF[float | int]):
    """Defines a property satisfaction function that accepts a range of values."""

    def __init__(self, min_value: float | int, max_value: float | int) -> None:
        """Initialize the range."""
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, prop_value: float | int) -> bool:
        """Return True if the value is in the range."""
        return self.min_value <= prop_value <= self.max_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.min_value}, {self.max_value})"


class GenericPSF[T: ItemPropValue](BasePSF[T]):
    """Defines a property satisfaction function with a custom function."""

    def __init__(self, func: Callable[[T], bool]) -> None:
        """Initialize the property satisfaction function."""
        self.func = func

    def __call__(self, prop_value: T) -> bool:
        """Return the result of the custom function."""
        return self.func(prop_value)


type PropSatFunction[T: ItemPropValue] = BasePSF[T] | Callable[[T], bool]


# %% === Item properties  ===
# TODO: Add action validity checking (action group, etc)
# TODO: Check if we need to add a hash
class ItemPropOld:
    """
    Property of an item in the definition of a task.

    If the property is fixed (cannot be changed by the agent), the candidate_required_property
    attribute is set to the property itself and the candidate_required_property_value is set to
    the value of the property.
    """

    def __init__(
        self,
        target_ai2thor_property: SimObjProp,
        value_type: type,
        is_fixed: bool = False,
        candidate_required_property: SimObjFixedProp | None = None,
        candidate_required_prop_sat_function: PropSatFunction | None = None,
    ) -> None:
        """Initialize the Property object."""
        self.target_ai2thor_property = target_ai2thor_property
        self.value_type = value_type
        self.is_fixed = is_fixed
        self.candidate_required_prop = target_ai2thor_property if is_fixed else candidate_required_property
        self.candidate_required_prop_sat_function = candidate_required_prop_sat_function

    def __str__(self) -> str:
        return f"{self.target_ai2thor_property}"

    def __repr__(self) -> str:
        return f"ItemProp({self.target_ai2thor_property})"


class ItemProp[T1: ItemPropValue, T2: ItemPropValue](ABC):
    """
    Base class for item properties in the definition of a task.

    If the property is fixed (cannot be changed by the agent), the candidate_required_prop
    attribute is set to the property itself and the candidate_required_prop_sat_function is set to
    the target satisfaction function.

    T is the type that the property value can take.
    """

    target_ai2thor_property: SimObjProp
    candidate_required_prop: SimObjFixedProp | None = None

    def __init__(
        self,
        target_satisfaction_function: PropSatFunction[T1],
    ) -> None:
        """Initialize the Property object."""
        self.target_satisfaction_function = target_satisfaction_function
        self.candidate_required_prop_sat_function: PropSatFunction[T2] | None = None

    def __call__(self, prop_value: T1) -> bool:
        """Return True if the value satisfies the property."""
        return self.target_satisfaction_function(prop_value)

    def __str__(self) -> str:
        return f"{self.target_ai2thor_property}({self.target_satisfaction_function})"

    def __repr__(self) -> str:
        return f"ItemProp({self.target_ai2thor_property}={self.target_satisfaction_function})"


class ItemFixedProp[T: ItemPropValue](ItemProp[T, T]):
    """
    Base class for fixed item properties in the definition of a task.

    Fixed properties are properties that cannot be changed by the agent.

    For fixed properties, the candidate_required_prop is set to the property itself and the
    candidate_required_prop_sat_function is set to the target satisfaction function.
    """

    target_ai2thor_property: SimObjFixedProp

    def __init__(
        self,
        target_satisfaction_function: PropSatFunction[T],
    ) -> None:
        """Initialize the FixedProperty object."""
        super().__init__(target_satisfaction_function)
        self.candidate_required_prop = self.target_ai2thor_property
        self.candidate_required_prop_sat_function = self.target_satisfaction_function


class ItemVariableProp[T1: ItemPropValue, T2: ItemPropValue](ItemProp[T1, T2]):
    """
    Base class for variable item properties in the definition of a task.

    Variable properties are properties that can be changed by the agent.
    """

    target_ai2thor_property: SimObjVariableProp
    candidate_required_prop: SimObjFixedProp
    candidate_required_prop_sat_function: PropSatFunction[T2]

    def __init__(
        self,
        target_satisfaction_function: PropSatFunction[T1],
    ) -> None:
        """Initialize the VariableProperty object."""
        super().__init__(target_satisfaction_function)

        # Delete the instance attribute since it is now a class attribute
        del self.candidate_required_prop_sat_function


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


class IsToggledProp(ItemVariableProp[bool, bool]):
    """Is toggled item property."""

    target_ai2thor_property = SimObjVariableProp.IS_TOGGLED
    candidate_required_prop = SimObjFixedProp.TOGGLEABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


class IsBrokenProp(ItemVariableProp[bool, bool]):
    """Is broken item property."""

    target_ai2thor_property = SimObjVariableProp.IS_BROKEN
    candidate_required_prop = SimObjFixedProp.BREAKABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


class IsFilledWithLiquidProp(ItemVariableProp[bool, bool]):
    """Is filled with liquid item property."""

    target_ai2thor_property = SimObjVariableProp.IS_FILLED_WITH_LIQUID
    candidate_required_prop = SimObjFixedProp.CAN_FILL_WITH_LIQUID
    candidate_required_prop_sat_function = SingleValuePSF(True)


class FillLiquidProp(ItemVariableProp[FillableLiquid, bool]):
    """Fill liquid item property."""

    target_ai2thor_property = SimObjVariableProp.FILL_LIQUID
    candidate_required_prop = SimObjFixedProp.CAN_FILL_WITH_LIQUID
    candidate_required_prop_sat_function = SingleValuePSF(True)


class IsDirtyProp(ItemVariableProp[bool, bool]):
    """Is dirty item property."""

    target_ai2thor_property = SimObjVariableProp.IS_DIRTY
    candidate_required_prop = SimObjFixedProp.DIRTYABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


class IsUsedUpProp(ItemVariableProp[bool, bool]):
    """Is used up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_USED_UP
    candidate_required_prop = SimObjFixedProp.CAN_BE_USED_UP
    candidate_required_prop_sat_function = SingleValuePSF(True)


class IsCookedProp(ItemVariableProp[bool, bool]):
    """Is cooked item property."""

    target_ai2thor_property = SimObjVariableProp.IS_COOKED
    candidate_required_prop = SimObjFixedProp.COOKABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


class TemperatureProp(ItemVariableProp[TemperatureValue, Any]):
    """Temperature item property."""

    target_ai2thor_property = SimObjVariableProp.TEMPERATURE


class IsSlicedProp(ItemVariableProp[bool, bool]):
    """Is sliced item property."""

    target_ai2thor_property = SimObjVariableProp.IS_SLICED
    candidate_required_prop = SimObjFixedProp.SLICEABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


class IsOpenProp(ItemVariableProp[bool, bool]):
    """Is open item property."""

    target_ai2thor_property = SimObjVariableProp.IS_OPEN
    candidate_required_prop = SimObjFixedProp.OPENABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


class OpennessProp(ItemVariableProp[float, bool]):
    """Openness item property."""

    target_ai2thor_property = SimObjVariableProp.OPENNESS
    candidate_required_prop = SimObjFixedProp.OPENABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


class IsPickedUpProp(ItemVariableProp[bool, bool]):
    """Is picked up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_PICKED_UP
    candidate_required_prop = SimObjFixedProp.PICKUPABLE
    candidate_required_prop_sat_function = SingleValuePSF(True)


# %% === Item property mapping ===
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
