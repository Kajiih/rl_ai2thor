"""
Abstract classes for item properties for RL-THOR environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Container, Sized
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_thor.envs.sim_objects import (
    SimObjectType,
    SimObjFixedProp,
    SimObjId,
    SimObjMetadata,
    SimObjProp,
    SimObjVariableProp,
)

if TYPE_CHECKING:
    from rl_thor.envs.tasks.items import AuxItem
    from rl_thor.envs.tasks.relations import Relation


# %% === Property value enums ==
# TODO? Move to a separate module?
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


ItemPropValue = int | float | bool | TemperatureValue | SimObjectType | FillableLiquid | str | list | None


# %% === Property Satisfaction Functions ===
class BasePSF[T](ABC):
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
    """Property satisfaction function that only accepts a single value."""

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
    """Property satisfaction function that accepts a set of values."""

    def __init__(self, target_values: Container[T]) -> None:
        """Initialize the target values."""
        super().__init__(target_values)
        self.target_values = target_values

    def __call__(self, prop_value: T) -> bool:
        """Return True if the value is in the target values."""
        return prop_value in self.target_values


class RangePSF(BasePSF[float | int]):
    """Property satisfaction function that accepts a range of values."""

    def __init__(self, min_value: float | int, max_value: float | int) -> None:
        """Initialize the range."""
        super().__init__(min_value, max_value)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, prop_value: float | int) -> bool:
        """Return True if the value is in the range."""
        return self.min_value <= prop_value <= self.max_value


class SizeLimitPSF[T: Sized](BasePSF[T]):
    """Property satisfaction function that checks if a container's size meets a specified limit."""

    def __init__(self, max_elements: int, expect_less_or_equal: bool = True) -> None:
        """
        Initialize the property satisfaction function.

        Args:
            max_elements (int): The maximum number of elements allowed in the container for it to
                satisfy the property.
            expect_less_or_equal (bool): Whether to expect the container to have less or equal
                elements than `max_elements`. Set to False to expect strictly more elements.
        """
        super().__init__(max_elements, expect_less_or_equal)
        self.max_elements = max_elements
        self.expect_less_or_equal = expect_less_or_equal

    def __call__(self, prop_value: T) -> bool:
        """
        Return True if the container size meets the criteria based on `expect_less_or_equal`.

        Args:
            prop_value (T): The container to check.

        Returns:
            bool: True if the container meets the size limit criteria, False otherwise.
        """
        return (len(prop_value) <= self.max_elements) == self.expect_less_or_equal


class EmptyContainerPSF[T: Sized](SizeLimitPSF[T]):
    """Property satisfaction function that accepts any empty container."""

    def __init__(self, expect_empty: bool = True) -> None:
        """
        Initialize the property satisfaction function.

        Args:
            expect_empty (bool): Whether the container should be empty or not. Set to False to
                accept a container that is NOT empty.
        """
        super().__init__(max_elements=0, expect_less_or_equal=expect_empty)


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
# TODO: Reimplement so that the base item property doesn't rely on PSF and create a new class for AI2-THOR-properties related properties (that drectly use the value of on of the object's metadata) to simplify making simple properties (those properties need PSF to define acceptable values (we may need to give a better name than PSF)).


class BaseItemProp[Treq: ItemPropValue](ABC):
    """
    Base class for item properties in the definition of a task.

    TODO: Write docstrings
    """

    is_fixed: bool

    def __init__(self) -> None:
        """Initialize the Property object."""
        # === Type Annotations ===
        self.candidate_required_prop: ItemFixedProp[Treq] | None = None

    def __call__(
        self,
        obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,
    ) -> bool:
        """
        Return True if the object satisfies the property.

        Args:
            obj_metadata (SimObjMetadata): Metadata of the object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata] | None): Dictionary
                mapping all object ids of the scene to their metadata. Defaults to None.

        Returns:
            is_satisfying (bool): Whether the object satisfies the property.
        """
        return self.is_object_satisfying(obj_metadata, scene_objects_dict)

    @abstractmethod
    def is_object_satisfying(
        self,
        obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,
    ) -> bool:
        """
        Return True if the object satisfies the property.

        Args:
            obj_metadata (SimObjMetadata): Metadata of the object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata] | None): Dictionary
                mapping all object ids of the scene to their metadata. Defaults to None.

        Returns:
            is_satisfying (bool): Whether the object satisfies the property.
        """


class AI2ThorBasedProp[T: ItemPropValue, Treq: ItemPropValue](BaseItemProp[Treq], ABC):
    """
    Base class for properties that are based on AI2-THOR object metadata.

    This class handles fetching property values from AI2-THOR metadata and checking if they satisfy
    a given condition using a property satisfaction function.

    T is the type that the satisfaction function should receive.
    Treq is the type T of the required property.
    """

    target_ai2thor_property: SimObjProp

    def __init__(self, satisfaction_function: PropSatFunction[T] | T, *args: Any, **kwargs: Any) -> None:
        """
        Initialize an AI2ThorBasedProp with the satisfaction function.

        Args:
            satisfaction_function (PropSatFunction[T] | T): Target satisfaction function that the
                value of the target AI2-THOR property of the object should satisfy. If a simple
                value is given instead of a property satisfaction function, it is considered as a
                SingleValuePSF.
            *args (Any): Eventual arguments for instantiating ItemVariableProp
            **kwargs (Any): Eventual arguments for instantiating ItemVariableProp
        """
        super().__init__(*args, **kwargs)
        if isinstance(satisfaction_function, ItemPropValue):
            satisfaction_function = SingleValuePSF(satisfaction_function)
        self.satisfaction_function = satisfaction_function

        # === Type Annotations ===
        self.satisfaction_function: PropSatFunction[T]

    def is_object_satisfying(
        self,
        obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,  # noqa: ARG002
    ) -> bool:
        """
        Return True if the object satisfies the property.

        Args:
            obj_metadata (SimObjMetadata): Metadata of the object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata] | None): Unused.

        Returns:
            is_satisfying (bool): Whether the object satisfies the property.
        """
        return self.satisfaction_function(obj_metadata[self.target_ai2thor_property])

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.satisfaction_function})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.satisfaction_function})"


class ItemFixedProp[T: ItemPropValue](AI2ThorBasedProp[T, T], ABC):
    """
    Base class for fixed item properties in the definition of a task.

    Fixed properties are properties that cannot be changed by the agent.

    The candidate_required_prop attribute is the instance itself.
    """

    target_ai2thor_property: SimObjFixedProp
    is_fixed: bool = True

    def __init__(self, target_satisfaction_function: PropSatFunction[T] | T) -> None:
        """Initialize the candidate_required_prop attribute with self."""
        super().__init__(target_satisfaction_function)
        self.candidate_required_prop = self


# TODO: Support adding relations to auxiliary items


class ItemVariableProp[Treq: ItemPropValue](BaseItemProp[Treq], ABC):
    """
    Base class for variable item properties in the definition of a task.

    Variable properties are properties that can be changed by the agent and will be scored during
    the task advancement computation. The score describes the advancement of the property; how much
    of the auxiliary properties, auxiliary items and the main property itself are satisfied. The
    advancement is equal to the sum of the advancement of the auxiliary properties and items plus
    the advancement of the main property (1 if satisfied, 0 otherwise).

    The candidate_required_prop attribute has to be defined in the subclass.

    # !! Do not add auxiliary items to an auxiliary property, it will no be taken into account.

    Attributes:
        auxiliary_properties (frozenset[ItemVariableProp]): The set of auxiliary properties that
            should be first satisfied in order to satisfy the main property.
        auxiliary_items (frozenset[TaskItem]): The set of auxiliary items whose properties should be
            first satisfied by any object in the scene in order to satisfy the main property. Those
            items are not considered in the item-candidates assignments of the task since they don't
            represent a unique task item but only an auxiliary item for a property.
        auxiliary_relations (frozenset[Relation]): The set of auxiliary relations of the property,
            i.e. the inverse relations of the auxiliary items. They are instantiated during the
            item initialization.
        maximum_advancement (int): The maximum advancement that can be achieved by satisfying the
            property and all of its auxiliary properties and items.
    """

    is_fixed: bool = False
    auxiliary_properties_blueprint: frozenset[tuple[type[ItemVariableProp], Any]] = frozenset()
    auxiliary_items: frozenset[AuxItem] = (
        frozenset()
    )  # !! auxiliary_items is not always a class attribute (e.g. in the TemperatureProp class) # TODO: Change the implementation to make it a instance attribute for all classes

    def __init__(
        self,
        main_prop: ItemVariableProp | None = None,
        main_relation: Relation | None = None,
    ) -> None:
        """Initialize the Property object."""
        super().__init__()
        # === Handle linked object ===
        if main_prop is not None and main_relation is not None:
            raise TooManyLinkedObjectsError(self, main_prop, main_relation)
        self.main_prop = main_prop
        self.main_relation = main_relation
        self.linked_object = main_prop if main_prop is not None else self.main_relation

        self.auxiliary_properties_blueprint_list = list(self.auxiliary_properties_blueprint)
        if self.linked_object is not None:
            self.is_auxiliary = True
            self.linked_object.auxiliary_properties_blueprint_list += self.auxiliary_properties_blueprint_list
            self.auxiliary_properties = frozenset()
        else:
            self.is_auxiliary = False
            #!! Auxiliary properties of auxiliary properties are added to the blueprint list during the iteration
            self.auxiliary_properties = frozenset([
                prop_blueprint[0](*prop_blueprint[1:], main_prop=self)  # type: ignore
                for prop_blueprint in self.auxiliary_properties_blueprint_list
            ])

        # Initialize the main property of the auxiliary items
        # TODO: Check if it still works
        for aux_item in self.auxiliary_items:
            aux_item.linked_prop = self

        self.auxiliary_relations = frozenset()  # Set this before using init_maximum_advancement
        # === Type annotations ===
        self.auxiliary_properties: frozenset[ItemVariableProp]
        self.auxiliary_relations: frozenset[Relation]
        self.is_auxiliary: bool
        self.main_prop: ItemVariableProp | None
        self.main_relation: Relation | None
        self.linked_object: ItemVariableProp | Relation | None
        self.maximum_advancement: int
        self.auxiliary_properties_blueprint_list: list[tuple[type[ItemVariableProp], Any]]

    def init_maximum_advancement(self) -> None:
        """
        Recursively initialize the maximum advancement of the property and its auxiliary properties and relations.

        The maximum advancements of the auxiliary items are already initialized during the item
        initialization.
        """
        # TODO? Remove the recursive initialization since it's always 1 for auxiliary properties?
        for aux_prop in self.auxiliary_properties:
            aux_prop.init_maximum_advancement()
        for aux_relation in self.auxiliary_relations:
            aux_relation.init_maximum_advancement()
        # TODO? Replace aux_prop.maximum_advancement by 1
        self.maximum_advancement = (
            1
            + sum(aux_prop.maximum_advancement for aux_prop in self.auxiliary_properties)
            + sum(aux_item.maximum_advancement for aux_item in self.auxiliary_items)
            + sum(aux_rel.maximum_advancement for aux_rel in self.auxiliary_relations)
        )
        # TODO? Replace aux_prop.maximum_advancement by 1


type ItemProp[Treq: ItemPropValue] = ItemFixedProp[Treq] | ItemVariableProp[Treq]
type RelationAuxProp = ItemVariableProp
type PropAuxProp = ItemVariableProp
type AuxProp = RelationAuxProp | PropAuxProp


# %% === Exceptions ===
class UndefinedPSFCalledError(Exception):
    """Exception raised when an UndefinedPSF is called."""

    def __init__(self, prop_value: Any) -> None:
        self.prop_value = prop_value

    def __str__(self) -> str:
        return f"UndefinedPSF should not be called, if the candidate_required_prop attribute of an item property is not None, a property satisfaction function should be properly defined. UndefinedPSF called with value: {self.prop_value}"


class TooManyLinkedObjectsError(Exception):
    """Exception raised when a ItermVariableProp is instantied with both a linked property AND a linked item."""

    def __init__(
        self, instantied_prop: ItemVariableProp, linked_prop: ItemVariableProp, linked_relation: Relation
    ) -> None:
        self.instantied_prop = instantied_prop
        self.linked_prop = linked_prop
        self.linked_relation = linked_relation

    def __str__(self) -> str:
        return (
            f"Item property {self.instantied_prop} instantied with both a linked property ({self.linked_prop}) and linked relation ({self.linked_relation}). This case is not supported for the moment."
            ""
        )
