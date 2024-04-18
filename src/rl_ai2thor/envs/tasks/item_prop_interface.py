"""
Abstract classes for item properties for AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Container
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_ai2thor.envs.sim_objects import (
    SimObjectType,
    SimObjFixedProp,
    SimObjId,
    SimObjMetadata,
    SimObjProp,
    SimObjVariableProp,
)

if TYPE_CHECKING:
    from rl_ai2thor.envs.tasks.items import AuxItem, CandidateId
    from rl_ai2thor.envs.tasks.relations import Relation


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


# %% === Property Satisfaction Functions ===
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
# TODO: Make only variable prop having auxiliary properties and items
class BaseItemProp[T1: ItemPropValue, T2: ItemPropValue](ABC):
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
        is_fixed (bool): True if the property is fixed (cannot be changed by the agent) and False
            if the property is variable (can be changed by the agent).
    """

    target_ai2thor_property: SimObjProp
    candidate_required_prop: ItemFixedProp[T2] | None = None
    is_fixed: bool

    def __init__(self, target_satisfaction_function: PropSatFunction[T1] | T1) -> None:
        """Initialize the Property object."""
        if isinstance(target_satisfaction_function, ItemPropValue):
            target_satisfaction_function = SingleValuePSF(target_satisfaction_function)
        self.target_satisfaction_function = target_satisfaction_function

        # === Type Annotations ===
        self.target_satisfaction_function: PropSatFunction[T1]

    def __call__(self, prop_value: T1) -> bool:
        """Return True if the value satisfies the property."""
        return self.target_satisfaction_function(prop_value)

    def is_object_satisfying(self, obj_metadata: SimObjMetadata) -> bool:
        """Return True if the object satisfies the property."""
        return self(obj_metadata[self.target_ai2thor_property])

    # TODO: Delete?
    def compute_candidates_results(
        self,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
        candidates_ids: set[CandidateId],
    ) -> dict[CandidateId, bool]:
        """
        Return the results of the property satisfaction for the candidates.

        The results are stored in a dictionary where the keys are the candidates ids and the values
        are booleans indicating if the candidate satisfies the property.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
                of the objects in the scene to their metadata.
            candidates_ids (set[CandidateId]): The set of candidate ids.

        Returns:
            candidates_results (dict[CandidateId, bool]): Dictionary mapping the candidate ids to
                a boolean indicating if the candidate satisfies the property.
        """
        return {
            candidate_id: self.is_object_satisfying(scene_objects_dict[candidate_id]) for candidate_id in candidates_ids
        }

    # TODO: Delete?
    # TODO: Implement a weighted score
    @staticmethod
    def compute_candidates_scores(candidates_results: dict[CandidateId, bool]) -> dict[CandidateId, float]:
        """
        Return the scores of the candidates based on the properties results.

        The scores are stored in a dictionary where the keys are the candidates ids and the values
        are the scores of the candidates.

        Args:
            candidates_results (dict[CandidateId, bool]): Dictionary mapping the candidate ids to
                a boolean indicating if the candidate satisfies the property.

        Returns:
            candidates_scores (dict[CandidateId, float]): Dictionary mapping the candidate ids to
                their scores.
        """
        return {
            candidate_id: int(candidate_satisfies) for candidate_id, candidate_satisfies in candidates_results.items()
        }

    def __str__(self) -> str:
        return f"{self.target_ai2thor_property}({self.target_satisfaction_function})"

    def __repr__(self) -> str:
        return f"ItemProp({self.target_ai2thor_property}, {self.target_satisfaction_function})"

    # def __eq__(self, other: Any) -> bool:
    #     return (
    #         isinstance(other, ItemProp)
    #         and self.target_ai2thor_property == other.target_ai2thor_property
    #         and self.target_satisfaction_function == other.target_satisfaction_function
    #     )

    # def __hash__(self) -> int:
    #     return hash((self.target_ai2thor_property, self.target_satisfaction_function))


class ItemFixedProp[T: ItemPropValue](BaseItemProp[T, T], ABC):
    """
    Base class for fixed item properties in the definition of a task.

    Fixed properties are properties that cannot be changed by the agent.

    The candidate_required_prop attribute is the instance itself.
    """

    target_ai2thor_property: SimObjFixedProp
    is_fixed: bool = True

    def __init__(
        self,
        target_satisfaction_function: PropSatFunction[T] | T,
    ) -> None:
        """Initialize the candidate_required_prop attribute with self."""
        super().__init__(target_satisfaction_function)
        self.candidate_required_prop = self


# TODO: Support adding relations to auxiliary items
class ItemVariableProp[T1: ItemPropValue, T2: ItemPropValue](BaseItemProp[T1, T2], ABC):
    """
    Base class for variable item properties in the definition of a task.

    Variable properties are properties that can be changed by the agent and will be scored during
    the task advancement computation. The score describes the advancement of the property; how much of the auxiliary properties, auxiliary items and the main property itself are satisfied. The
    advancement is equal to the sum of the advancement of the auxiliary properties and items plus
    the advancement of the main property (1 if satisfied, 0 otherwise).

    The candidate_required_prop attribute has to be defined in the  subclass.

    Attributes:
        auxiliary_properties (frozenset[ItemVariableProp]): The set of auxiliary properties that
            should be first satisfied in order to satisfy the main property.
        auxiliary_items (frozenset[TaskItem]): The set of auxiliary items whose properties should be
            first satisfied by any object in the scene in order to satisfy the main property. Those
            items are not considered in the item-candidates assignments of the task since they don't
            represent a unique task item but only an auxiliary item for a property.
        max_advancement (int): The maximum advancement that can be achieved by satisfying the main
            property and all of its auxiliary properties and items.
    """

    target_ai2thor_property: SimObjVariableProp
    is_fixed: bool = False
    auxiliary_properties: frozenset[PropAuxProp] = frozenset()
    auxiliary_items: frozenset[AuxItem] = frozenset()

    def __init__(
        self,
        target_satisfaction_function: PropSatFunction[T1] | T1,
    ) -> None:
        """Initialize the Property object."""
        super().__init__(target_satisfaction_function)

        # Initialize the main property of the auxiliary properties and items
        for aux_item in self.auxiliary_items:
            aux_item.linked_prop = self
        for aux_prop in self.auxiliary_properties:
            aux_prop.linked_prop = self

        self.max_advancement = (
            1 + len(self.auxiliary_properties) + sum(aux_item.max_advancement for aux_item in self.auxiliary_items)
        )

        # === Type annotations ===
        self.max_advancement: int


type ItemProp[T1: ItemPropValue, T2: ItemPropValue] = ItemFixedProp[T1] | ItemVariableProp[T1, T2]


# TODO: Define this better, eventually by writing an AuxProp class for each Variable prop that is used as an auxiliary prop or by adding is_fixed and is_auxiliary attributes to the ItemProp class
class BaseAuxProp[T1: ItemPropValue, T2: ItemPropValue](ItemVariableProp[T1, T2], ABC):
    """
    Base class for auxiliary properties of an item property or a relation.

    An auxiliary property is a variable property and has no auxiliary properties or auxiliary items
    itself.

    The main point of an auxiliary property is that if its main property is satisfied, the auxiliary
    property is also considered satisfied. Also, we add the score of the auxiliary property to the
    score of the main property.

    Attributes:
        linked_object (ItemVariableProp, Relation): The main property or relation that this
            auxiliary property is linked to.
    """

    def __init__(
        self, variable_prop_type: type[ItemVariableProp[T1, T2]], target_satisfaction_function: PropSatFunction[T1] | T1
    ) -> None:
        """Initialize the Property object."""
        variable_prop_type.__init__(self, target_satisfaction_function)

        self.target_ai2thor_property = variable_prop_type.target_ai2thor_property

        # === Type annotations ===
        self.linked_object: ItemVariableProp | Relation


class RelationAuxProp[T1: ItemPropValue, T2: ItemPropValue](BaseAuxProp[T1, T2]):
    """
    Auxiliary property of a relation.

    Attributes:
        linked_relation (Relation): The relation that this auxiliary property is linked to.
    """

    def __init__(
        self, variable_prop_type: type[ItemVariableProp[T1, T2]], target_satisfaction_function: PropSatFunction[T1] | T1
    ) -> None:
        """Initialize the Property object."""
        super().__init__(variable_prop_type, target_satisfaction_function)

        self.target_ai2thor_property = variable_prop_type.target_ai2thor_property

        # === Type annotations ===
        self.linked_relation: Relation

    @property
    def linked_object(self) -> Relation:
        """Return the linked object."""
        return self.linked_relation


class PropAuxProp[T1: ItemPropValue, T2: ItemPropValue](BaseAuxProp[T1, T2]):
    """
    Auxiliary property of an item property.

    Attributes:
        linked_prop (ItemVariableProp): The main property that this auxiliary property is linked to.
    """

    def __init__(
        self, variable_prop_type: type[ItemVariableProp[T1, T2]], target_satisfaction_function: PropSatFunction[T1] | T1
    ) -> None:
        """Initialize the Property object."""
        super().__init__(variable_prop_type, target_satisfaction_function)

        self.target_ai2thor_property = variable_prop_type.target_ai2thor_property

        # === Type annotations ===
        self.linked_prop: ItemVariableProp

    @property
    def linked_object(self) -> ItemVariableProp:
        """Return the linked object."""
        return self.linked_prop


type AuxProp = RelationAuxProp | PropAuxProp


# %% === Exceptions ===
class UndefinedPSFCalledError(Exception):
    """Exception raised when an UndefinedPSF is called."""

    def __init__(self, prop_value: Any) -> None:
        self.prop_value = prop_value

    def __str__(self) -> str:
        return f"UndefinedPSF should not be called, if the candidate_required_prop attribute of an item property is not None, a property satisfaction function should be properly defined. UndefinedPSF called with value: {self.prop_value}"
