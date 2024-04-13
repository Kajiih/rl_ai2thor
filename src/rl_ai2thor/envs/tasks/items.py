"""
Task items in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Container, Hashable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from rl_ai2thor.envs.sim_objects import (
    SimObjectType,
    SimObjFixedProp,
    SimObjId,
    SimObjMetadata,
    SimObjProp,
    SimObjVariableProp,
)

if TYPE_CHECKING:
    from rl_ai2thor.envs.tasks.relations import Relation, RelationTypeId


# %% === Enums ==
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


type PropValue = int | float | bool | TemperatureValue | SimObjectType | FillableLiquid
type Assignment[T: Hashable] = dict[TaskItem[T], SimObjId]


# %% === Property satisfaction functions ===
class BasePSF[T: PropValue](ABC):
    """
    Base class for functions used to define the set of acceptable values for a property to be satisfied.

    We call those functions *property satisfaction functions* (PSF).

    T is the type that the property value can take.
    """

    @abstractmethod
    def __call__(self, prop_value: T) -> bool:
        """Return True if the value satisfies the property."""


class SingleValuePSF[T: PropValue](BasePSF[T]):
    """Defines a property satisfaction function that only accepts a single value."""

    def __init__(self, target_value: T) -> None:
        """Initialize the target value."""
        self.target_value = target_value

    def __call__(self, prop_value: T) -> bool:
        """Return True if the value is equal to the target value."""
        return prop_value == self.target_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_value})"


class MultiValuePSF[T: PropValue](BasePSF[T]):
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


class GenericPSF[T: PropValue](BasePSF[T]):
    """Defines a property satisfaction function with a custom function."""

    def __init__(self, func: Callable[[T], bool]) -> None:
        """Initialize the property satisfaction function."""
        self.func = func

    def __call__(self, prop_value: T) -> bool:
        """Return the result of the custom function."""
        return self.func(prop_value)


type PropSatFunction[T: PropValue] = BasePSF[T] | Callable[[T], bool]


# %% === Properties  ===
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


class ItemProp[T1: PropValue, T2: PropValue](ABC):
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


class ItemFixedProp[T: PropValue](ItemProp[T, T]):
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


class ItemVariableProp[T1: PropValue, T2: PropValue](ItemProp[T1, T2]):
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


# %% === Items ===
# TODO? Add support for giving some score for semi satisfied relations and using this info in the selection of interesting objects/assignments
# TODO: Store relation in a list and store the results using the id of the relation to simplify the code
# TODO: Store the results in the class and write methods to return views of the results to simplify the code
class TaskItem[T: Hashable]:
    """
    An item in the definition of a task.

    TODO: Finish docstring.
    """

    def __init__(
        self,
        t_id: T,
        properties: dict[ItemPropOld, PropSatFunction],
    ) -> None:
        """
        Initialize the TaskItem object.

        Args:
            t_id (T): The ID of the item as defined in the task description.
            properties (dict[ItemProp, PropSatFunction]): The properties of the item.
        """
        self.id = t_id
        self.properties = properties

        # Infer the candidate required properties from the item properties
        self._prop_candidate_required_properties: dict[  # TODO: Check why there is a reportAttributeAccessIssue
            SimObjFixedProp, PropSatFunction
        ] = {
            prop.candidate_required_prop: (
                prop_satisfaction_function if prop.is_fixed else prop.candidate_required_prop_sat_function
            )
            for prop, prop_satisfaction_function in self.properties.items()
            if prop.candidate_required_prop is not None
        }

        # Other attributes
        self._rel_candidate_required_properties: dict[SimObjFixedProp, PropSatFunction] = {}
        self.organized_relations: dict[T, dict[RelationTypeId, Relation]] = {}
        self.candidate_ids: set[SimObjId] = set()

    @property
    def relations(self) -> set[Relation]:
        """
        Get the set of relations of the item.

        Returns:
            set[Relation]: The set of relations.
        """
        return self._relations

    @relations.setter
    def relations(self, relations: set[Relation]) -> None:
        """
        Setter for the relations of the item.

        Automatically update the organized_relations and candidate_required_properties
        attributes.
        """
        self.organized_relations.update({
            relation.related_item.id: {
                relation.type_id: relation,
            }
            for relation in relations
        })
        self._rel_candidate_required_properties.update({  # TODO: Check why there is a reportCallIssue
            relation.candidate_required_prop: relation.candidate_required_prop_sat_function
            for relation in relations
            if relation.candidate_required_prop is not None
        })

        # Delete duplicate relations if any
        self._relations = {
            relation for relation_set in self.organized_relations.values() for relation in relation_set.values()
        }

    @property
    def candidate_required_properties(self) -> dict[SimObjProp, PropSatFunction]:
        """
        Return a dictionary containing the properties required for an object to be a candidate for the item.

        Returns:
            candidate_properties (dict[ObjPropId, PropSatFunction]): Dictionary containing the candidate required properties.
        """
        return {
            **self._prop_candidate_required_properties,
            **self._rel_candidate_required_properties,
        }

    def is_candidate(self, obj_metadata: SimObjMetadata) -> bool:
        """
        Return True if the given object is a valid candidate for the item.

        Args:
            obj_metadata (SimObjMetadata): Object metadata.

        Returns:
            is_candidate (bool): True if the given object is a valid candidate for the item.
        """
        return all(
            prop_sat_function(obj_metadata[prop_id])
            for prop_id, prop_sat_function in self.candidate_required_properties.items()
        )

    def _get_properties_satisfaction(self, obj_metadata: SimObjMetadata) -> dict[SimObjProp, bool]:
        """
        Return a dictionary indicating which properties are satisfied by the given object.

        Args:
            obj_metadata (SimObjMetadata): Object metadata.

        Returns:
            prop_satisfaction (dict[SimObjProp, bool]): Dictionary indicating which properties are satisfied by the given object.
        """
        return {
            prop.target_ai2thor_property: prop_sat_function(obj_metadata[prop.target_ai2thor_property])
            for prop, prop_sat_function in self.properties.items()
        }

    def _get_relations_semi_satisfying_objects(
        self,
        candidate_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[T, dict[RelationTypeId, set[SimObjId]]]:
        """
        Return the dictionary of satisfying objects with the given candidate for each relations.

        The relations are organized by related item id.

        Args:
            candidate_metadata (SimObjMetadata): Metadata of the candidate.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
                of the objects in the scene to their metadata.

        Returns:
            semi_satisfying_objects (dict[T, dict[RelationTypeId, set[SimObjId]]]): Dictionary indicating which objects are semi-satisfying the relations with the given object.
        """
        return {
            related_item_id: {
                relation.type_id: relation.get_satisfying_related_object_ids(candidate_metadata, scene_objects_dict)
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations
        }

    # TODO? Remove; unused
    def _compute_obj_results(
        self,
        obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[
        Literal["properties", "relations"],
        dict[SimObjProp, bool] | dict[T, dict[RelationTypeId, set[SimObjId]]],
    ]:  # TODO: Simplify this big type after finish the implementation
        """
        Return the results dictionary of the object for the item.

        The results are the satisfaction of each property and the satisfying
        objects of each relation of the item.

        Args:
            obj_metadata (SimObjMetadata): Object metadata.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
                of the objects in the scene to their metadata.

        Returns:
            results (dict[Literal["properties", "relations"], dict[SimObjProp, bool] | dict[T, dict[RelationTypeId, set[SimObjId]]]]): Results of the object for the item.
        """
        results: dict[
            Literal["properties", "relations"],
            dict[SimObjProp, bool] | dict[T, dict[RelationTypeId, set[SimObjId]]],
        ] = {
            "properties": self._get_properties_satisfaction(obj_metadata),
            "relations": self._get_relations_semi_satisfying_objects(obj_metadata, scene_objects_dict),
        }
        return results

    def _compute_all_candidates_results(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> tuple[
        dict[SimObjProp, dict[SimObjId, bool]],
        dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
    ]:
        """
        Return the results dictionary with the results of each candidate of the item.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
            of the objects in the scene to their metadata.

        Returns:
            candidates_properties_results (dict[SimObjProp, dict[SimObjId, bool]]):
                Results of each object for the item properties.
            candidates_relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
        """
        candidates_properties_results = {
            prop.target_ai2thor_property: {
                obj_id: prop_sat_function(scene_objects_dict[obj_id][prop.target_ai2thor_property])
                for obj_id in self.candidate_ids
            }
            for prop, prop_sat_function in self.properties.items()
        }

        candidates_relations_results = {
            related_item_id: {
                relation.type_id: {
                    obj_id: relation.get_satisfying_related_object_ids(scene_objects_dict[obj_id], scene_objects_dict)
                    for obj_id in self.candidate_ids
                }
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations
        }

        return candidates_properties_results, candidates_relations_results

    def _compute_all_candidates_scores(
        self,
        candidates_properties_results: dict[SimObjProp, dict[SimObjId, bool]],
        candidates_relations_results: dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
    ) -> tuple[
        dict[SimObjId, int],
        dict[SimObjId, int],
    ]:
        """
        Return the property and relation scores of each candidate of the item.

        Args:
            candidates_properties_results (dict[SimObjProp, dict[SimObjId, bool]]):
                Results of each object for the item properties.
            candidates_relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.

        Returns:
            candidates_properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.
            candidates_relations_scores (dict[SimObjId, int]):
                Relation scores of each object for the item.
        """
        candidates_properties_scores: dict[SimObjId, int] = {
            obj_id: sum(
                bool(candidates_properties_results[prop_id][obj_id]) for prop_id in candidates_properties_results
            )
            for obj_id in self.candidate_ids
        }
        candidates_relations_scores = {
            obj_id: sum(
                1
                for related_item_id in candidates_relations_results
                for relation_type_id in candidates_relations_results[related_item_id]
                if candidates_relations_results[related_item_id][relation_type_id][obj_id]
            )
            for obj_id in self.candidate_ids
        }

        return candidates_properties_scores, candidates_relations_scores

    def compute_interesting_candidates(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> tuple[
        set[SimObjId],
        dict[SimObjProp, dict[SimObjId, bool]],
        dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
        dict[SimObjId, int],
        dict[SimObjId, int],
    ]:
        """
        Return the set of interesting candidates and the results and scores of each candidate of the item.

        The interesting candidates are those that can lead to a maximum of task advancement
        depending on the assignment of objects to the other items.

        A candidate is *strong* if it has no strictly *stronger* candidate among the other
        candidates, where the stronger relation is defined in the get_stronger_candidate
        method.

        The set of interesting candidates is the set of strong candidates where we keep
        only one candidate of same strength and we add the candidate with the higher
        property score (not counting the semi satisfied relations score) if none of those
        are already added (because we don't know which relation will effectively be
        satisfied in the assignment).

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            interesting_candidates (set[SimObjId]): Set of interesting candidates for the item.
            candidates_properties_results (dict[SimObjProp, dict[SimObjId, bool]]):
                Results of each object for the item properties.
            candidates_relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
            candidates_properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.
            candidates_relations_scores (dict[SimObjId, int]):
                Relation scores of each object for the item.
        """
        # Compute the results of each object for the item
        candidates_properties_results, candidates_relations_results = self._compute_all_candidates_results(
            scene_objects_dict
        )

        # Compute the scores of each object for the item
        candidates_properties_scores, candidates_relations_scores = self._compute_all_candidates_scores(
            candidates_properties_results, candidates_relations_results
        )

        # Remove the candidates that have a stronger alternative
        interesting_candidates = list(self.candidate_ids)
        for i, candidate_id in enumerate(interesting_candidates):
            for j, other_candidate_id in enumerate(interesting_candidates[i + 1 :]):
                stronger_candidate = self._get_stronger_candidate(
                    candidate_id,
                    other_candidate_id,
                    candidates_relations_results,
                    candidates_properties_scores,
                    candidates_relations_scores,
                )
                if stronger_candidate in {candidate_id, "equal"}:
                    # In the equal case, we can keep any of the two candidates
                    # Remove the other candidate
                    interesting_candidates.pop(i + j + 1)
                elif stronger_candidate == other_candidate_id:
                    # Remove the candidate
                    interesting_candidates.pop(i)
                    break

        # Add a candidate with the highest property score if none of those are already added
        max_prop_score = max(candidates_properties_scores[candidate_id] for candidate_id in interesting_candidates)
        # Check if there is a candidate with the highest property score
        if max_prop_score not in [
            candidates_properties_scores[candidate_id] for candidate_id in interesting_candidates
        ]:
            # Add the candidate with the highest property score
            for candidate_id in self.candidate_ids:
                if candidates_properties_scores[candidate_id] == max_prop_score:
                    interesting_candidates.append(candidate_id)
                    break

        return (
            set(interesting_candidates),
            candidates_properties_results,
            candidates_relations_results,
            candidates_properties_scores,
            candidates_relations_scores,
        )

    def _get_stronger_candidate(
        self,
        obj_1_id: SimObjId,
        obj_2_id: SimObjId,
        candidates_relations_results: dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
        candidates_properties_scores: dict[SimObjId, int],
        candidates_relations_scores: dict[SimObjId, int],
    ) -> SimObjId | Literal["equal", "incomparable"]:
        """
        Return the stronger candidate between the two given candidates.

        A candidate x is stronger than a candidate y if some relations, the sets
        of satisfying objects of y are included in the set of satisfying objects of x
        and the difference Sp(x) - (Sp(y) +Sr(y)) + d[x,y] > 0 where Sp(z) is the sum
        of the property scores of z, Sr(z) is the sum of the relation scores of z and
        d[x,y] is the number of relations such that the set of satisfying objects of
        y is included in the set of satisfying objects of x. This represent the worst
        case for x compared to y.

        In particular, if Sp(y) + Sr(y) > Sp(x) + Sr(x), x cannot be stronger than y.

        Two candidates x and y have the same strength if x is stronger than y and y is
        stronger than x, otherwise they are either incomparable if none of them is
        stronger than the other or one is stronger than the other.
        The equal case can only happen if all relations have the same set of satisfying
        objects for both candidates.

        Args:
            obj_1_id (SimObjId): First candidate object id.
            obj_2_id (SimObjId): Second candidate object id.
            candidates_relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
            candidates_properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.
            candidates_relations_scores (dict[SimObjId, int]):
                Relation scores of each object for the item.

        Returns:
            stronger_candidate (SimObjId | Literal["equal", "incomparable"]): The stronger candidate
                between the two given candidates or "equal" if they have same strength or
                "incomparable" if they cannot be compared.

        """
        obj1_stronger = True
        obj2_stronger = True
        # Pre check: Sp(obj_1_id) + Sr(obj_1_id) < Sp(obj_2_id) + Sr(obj_2_id)
        if (
            candidates_properties_scores[obj_1_id] + candidates_relations_scores[obj_1_id]
            < candidates_properties_scores[obj_2_id] + candidates_relations_scores[obj_2_id]
        ):
            obj1_stronger = False
        else:
            obj1_stronger = self._is_stronger_candidate_than(
                obj_1_id, obj_2_id, candidates_relations_results, candidates_properties_scores
            )

        # Pre check: Sp(obj_1_id) + Sr(obj_1_id) > Sp(obj_2_id) + Sr(obj_2_id)
        if (
            candidates_properties_scores[obj_1_id] + candidates_relations_scores[obj_1_id]
            > candidates_properties_scores[obj_2_id] + candidates_relations_scores[obj_2_id]
        ):
            obj2_stronger = False
        else:
            obj2_stronger = self._is_stronger_candidate_than(
                obj_2_id, obj_1_id, candidates_relations_results, candidates_properties_scores
            )

        if obj1_stronger:
            return "equal" if obj2_stronger else obj_1_id
        return obj_2_id if obj2_stronger else "incomparable"

    @staticmethod
    def _is_stronger_candidate_than(
        obj_1_id: SimObjId,
        obj_2_id: SimObjId,
        candidates_relations_results: dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
        candidates_properties_scores: dict[SimObjId, int],
    ) -> bool:
        """
        Return True if the first candidate is stronger than the second candidate.

        A candidate x is stronger than a candidate y if some relations, the sets
        of satisfying objects of y are included in the set of satisfying objects of x
        and the difference Sp(x) - (Sp(y) +Sr(y)) + d[x,y] > 0 where Sp(z) is the sum
        of the property scores of z, Sr(z) is the sum of the relation scores of z and
        d[x,y] is the number of relations such that the set of satisfying objects of
        y is included in the set of satisfying objects of x. This represent the worst
        case for x compared to y.

        See the get_stronger_candidate method for more details about the "is stronger than" relation.

        Args:
            obj_1_id (SimObjId): First candidate object id.
            obj_2_id (SimObjId): Second candidate object id.
            candidates_relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
            candidates_properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.

        Returns:
            is_stronger (bool): True if the first candidate is stronger than the second candidate.
        """
        sp_x = candidates_properties_scores[obj_1_id]
        sp_y = candidates_properties_scores[obj_2_id]
        sr_y = candidates_properties_scores[obj_2_id]

        # Calculate d[x,y]
        d_xy = 0
        for related_item_id in candidates_relations_results:
            for relation_type_id in candidates_relations_results[related_item_id]:
                x_sat_obj_ids = candidates_relations_results[related_item_id][relation_type_id][obj_1_id]
                y_sat_obj_ids = candidates_relations_results[related_item_id][relation_type_id][obj_2_id]
                if y_sat_obj_ids.issubset(x_sat_obj_ids):
                    d_xy += 1

        return sp_x - (sp_y + sr_y) + d_xy > 0

    def __str__(self) -> str:
        return f"{self.id}"

    def __repr__(self) -> str:
        return f"TaskItem({self.id})\n  properties={self.properties})\n  relations={self.relations})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TaskItem):
            return False
        return self.id == other.id and self.properties == other.properties and self.relations == other.relations


class ItemOverlapClass[T: Hashable]:
    """A group of items whose sets of candidates overlap."""

    def __init__(
        self,
        items: list[TaskItem[T]],
        candidate_ids: list[SimObjId],
    ) -> None:
        """
        Initialize the overlap class with the given items and candidate ids.

        Args:
            items (list[TaskItem[T]]): The items in the overlap class.
            candidate_ids (list[SimObjId]): The candidate ids of candidates in the overlap class.
        """
        self.items = items
        self.candidate_ids = candidate_ids

        # Compute all valid assignments of objects to the items in the overlap class
        # One permutation is represented by a dictionary mapping the item to the assigned object id
        candidate_permutations = [
            dict(zip(self.items, permutation, strict=False))
            for permutation in itertools.permutations(self.candidate_ids, len(self.items))
            # TODO?: Replace candidate ids by their index in the list to make it more efficient? Probably need this kind of optimizations
        ]
        # Filter the permutations where the assigned objects are not candidates of the items
        self.valid_assignments: list[Assignment[T]] = [
            permutation
            for permutation in candidate_permutations
            if all(obj_id in item.candidate_ids for item, obj_id in permutation.items())
        ]

        if not self.valid_assignments:
            # raise NoValidAssignmentError(self)
            print(f"No valid assignment for overlap class {self}")

    def prune_assignments(self, compatible_global_assignments: list[Assignment[T]]) -> None:
        """
        Prune the valid assignments to keep only those that are part of the given compatible assignments.

        Valid assignments are assignments where each item is associated with a candidate
        that has all correct candidate_required_properties (without taking into account the
        relations between the items) and compatible assignments are valid assignment where the
        candidates are compatible when taking into account the relations between the items.

        Args:
            compatible_global_assignments (list[Assignment[T]]): List of global
                compatible (for the whole task and not only this overlap class).
        """
        compatible_global_assignments_set = {
            tuple(global_assignment[item] for item in self.items) for global_assignment in compatible_global_assignments
        }

        self.valid_assignments = [
            dict(zip(self.items, assignment_tuple, strict=True))
            for assignment_tuple in compatible_global_assignments_set
        ]

    def compute_interesting_assignments(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> tuple[
        list[Assignment[T]],
        dict[TaskItem[T], dict[SimObjProp, dict[SimObjId, bool]]],
        dict[TaskItem[T], dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]],
        dict[TaskItem[T], dict[SimObjId, int]],
        dict[TaskItem[T], dict[SimObjId, int]],
    ]:
        """
        Return the interesting assignments of objects to the items in the overlap class, the items results and items scores.

        The interesting assignments are the ones that can lead to a maximum of task advancement
        depending on the assignment of objects in the other overlap classes.
        Interesting assignments are the ones where each all assigned objects are interesting
        candidates for their item.

        For more details about the definition of interesting candidates, see the
        compute_interesting_candidates method of the TaskItem class.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            interesting_assignments (list[Assignment[T]]):
                List of the interesting assignments of objects to the items in the overlap class.
            all_properties_results (dict[TaskItem[T], dict[SimObjProp, dict[SimObjId, bool]]]):
                Results of each object for each property of each item in the overlap class.
            all_relation_results (dict[TaskItem[T], dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]]):
                Results of each object for the relation of each item in the overlap class.
            all_properties_scores (dict[TaskItem[T], dict[SimObjId, int]]):
                Property scores of each object for each item in the overlap class.
            all_relations_scores (dict[TaskItem[T], dict[SimObjId, int]]):
                Relation scores of each object for each item in the overlap class.
        """
        interesting_candidates_data = {
            item: item.compute_interesting_candidates(scene_objects_dict) for item in self.items
        }
        # Extract the interesting candidates, results and scores
        interesting_candidates = {item: data[0] for item, data in interesting_candidates_data.items()}
        all_properties_results = {item: data[1] for item, data in interesting_candidates_data.items()}
        all_relation_results = {item: data[2] for item, data in interesting_candidates_data.items()}
        all_properties_scores = {item: data[3] for item, data in interesting_candidates_data.items()}
        all_relations_scores = {item: data[4] for item, data in interesting_candidates_data.items()}

        # Filter the valid assignments to keep only the ones with interesting candidates
        interesting_assignments = [
            assignment
            for assignment in self.valid_assignments
            if all(assignment[item] in interesting_candidates[item] for item in self.items)
        ]

        return (
            interesting_assignments,
            all_properties_results,
            all_relation_results,
            all_properties_scores,
            all_relations_scores,
        )

    def __str__(self) -> str:
        return f"{{self.items}}"

    def __repr__(self) -> str:
        return f"OverlapClass({self.items})"


# %% Exceptions
class NoCandidateError(Exception):
    """
    Exception raised when no candidate is found for an item.

    The task cannot be completed if no candidate is found for an item, and this
    is probably a sign that the task and the scene are not compatible are that the task is impossible.
    """

    def __init__(self, item: TaskItem) -> None:
        self.item = item

    def __str__(self) -> str:
        return f"Item '{self.item}' has no candidate with the required properties {self.item.candidate_required_properties}"


# TODO: Unused; Replace by a warning?
class NoValidAssignmentError(Exception):
    """
    Exception raised when no valid assignment is found during the initialization of an overlap class.

    This means that it is not possible to assign a different candidates to all
    items in the overlap class, making the task impossible to complete.

    Either the task and the scene are not compatible or the task is impossible because
    at least one item has no valid candidate (a NoCandidateError should be automatically
    raised in this case).
    """

    def __init__(self, overlap_class: ItemOverlapClass) -> None:
        self.overlap_class = overlap_class

        # Check that the items have at least one valid candidate
        for item in self.overlap_class.items:
            if not item.candidate_ids:
                raise NoCandidateError(item)

    def __str__(self) -> str:
        return f"Overlap class '{self.overlap_class}' has no valid assignment."


# %% === Property Definitions ===
object_type_prop = ItemPropOld(
    SimObjFixedProp.OBJECT_TYPE,
    value_type=str,
    is_fixed=True,
)
is_interactable_prop = ItemPropOld(
    SimObjFixedProp.IS_INTERACTABLE,
    value_type=bool,
    is_fixed=True,
)
receptacle_prop = ItemPropOld(
    SimObjFixedProp.RECEPTACLE,
    value_type=bool,
    is_fixed=True,
)
toggleable_prop = ItemPropOld(
    SimObjFixedProp.TOGGLEABLE,
    value_type=bool,
    is_fixed=True,
)
breakable_prop = ItemPropOld(
    SimObjFixedProp.BREAKABLE,
    value_type=bool,
    is_fixed=True,
)
can_fill_with_liquid_prop = ItemPropOld(
    SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    value_type=bool,
    is_fixed=True,
)
dirtyable_prop = ItemPropOld(
    SimObjFixedProp.DIRTYABLE,
    value_type=bool,
    is_fixed=True,
)
can_be_used_up_prop = ItemPropOld(
    SimObjFixedProp.CAN_BE_USED_UP,
    value_type=bool,
    is_fixed=True,
)
cookable_prop = ItemPropOld(
    SimObjFixedProp.COOKABLE,
    value_type=bool,
    is_fixed=True,
)
is_heat_source_prop = ItemPropOld(
    SimObjFixedProp.IS_HEAT_SOURCE,
    value_type=bool,
    is_fixed=True,
)
is_cold_source_prop = ItemPropOld(
    SimObjFixedProp.IS_COLD_SOURCE,
    value_type=bool,
    is_fixed=True,
)
sliceable_prop = ItemPropOld(
    SimObjFixedProp.SLICEABLE,
    value_type=bool,
    is_fixed=True,
)
openable_prop = ItemPropOld(
    SimObjFixedProp.OPENABLE,
    value_type=bool,
    is_fixed=True,
)
pickupable_prop = ItemPropOld(
    SimObjFixedProp.PICKUPABLE,
    value_type=bool,
    is_fixed=True,
)
moveable_prop = ItemPropOld(
    SimObjFixedProp.MOVEABLE,
    value_type=bool,
    is_fixed=True,
)
visible_prop = ItemPropOld(
    SimObjVariableProp.VISIBLE,
    value_type=bool,
)
is_toggled_prop = ItemPropOld(
    SimObjVariableProp.IS_TOGGLED,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.TOGGLEABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
is_broken_prop = ItemPropOld(
    SimObjVariableProp.IS_BROKEN,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.BREAKABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
is_filled_with_liquid_prop = ItemPropOld(
    SimObjVariableProp.IS_FILLED_WITH_LIQUID,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
fill_liquid_prop = ItemPropOld(
    SimObjVariableProp.FILL_LIQUID,
    value_type=FillableLiquid,
    candidate_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
is_dirty_prop = ItemPropOld(
    SimObjVariableProp.IS_DIRTY,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.DIRTYABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
is_used_up_prop = ItemPropOld(
    SimObjVariableProp.IS_USED_UP,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.CAN_BE_USED_UP,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
is_cooked_prop = ItemPropOld(
    SimObjVariableProp.IS_COOKED,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.COOKABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
temperature_prop = ItemPropOld(
    SimObjVariableProp.TEMPERATURE,
    value_type=TemperatureValue,
)
is_sliced_prop = ItemPropOld(
    SimObjVariableProp.IS_SLICED,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.SLICEABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
is_open_prop = ItemPropOld(
    SimObjVariableProp.IS_OPEN,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.OPENABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
openness_prop = ItemPropOld(
    SimObjVariableProp.OPENNESS,
    value_type=float,
    candidate_required_property=SimObjFixedProp.OPENABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)
is_picked_up_prop = ItemPropOld(
    SimObjVariableProp.IS_PICKED_UP,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.PICKUPABLE,
    candidate_required_prop_sat_function=SingleValuePSF(True),
)

# %% === Mappings ===
obj_prop_id_to_item_prop = {
    SimObjFixedProp.OBJECT_TYPE: object_type_prop,
    SimObjFixedProp.IS_INTERACTABLE: is_interactable_prop,
    SimObjFixedProp.RECEPTACLE: receptacle_prop,
    SimObjFixedProp.TOGGLEABLE: toggleable_prop,
    SimObjFixedProp.BREAKABLE: breakable_prop,
    SimObjFixedProp.CAN_FILL_WITH_LIQUID: can_fill_with_liquid_prop,
    SimObjFixedProp.DIRTYABLE: dirtyable_prop,
    SimObjFixedProp.CAN_BE_USED_UP: can_be_used_up_prop,
    SimObjFixedProp.COOKABLE: cookable_prop,
    SimObjFixedProp.IS_HEAT_SOURCE: is_heat_source_prop,
    SimObjFixedProp.IS_COLD_SOURCE: is_cold_source_prop,
    SimObjFixedProp.SLICEABLE: sliceable_prop,
    SimObjFixedProp.OPENABLE: openable_prop,
    SimObjFixedProp.PICKUPABLE: pickupable_prop,
    SimObjFixedProp.MOVEABLE: moveable_prop,
    SimObjVariableProp.VISIBLE: visible_prop,
    SimObjVariableProp.IS_TOGGLED: is_toggled_prop,
    SimObjVariableProp.IS_BROKEN: is_broken_prop,
    SimObjVariableProp.IS_FILLED_WITH_LIQUID: is_filled_with_liquid_prop,
    SimObjVariableProp.FILL_LIQUID: fill_liquid_prop,
    SimObjVariableProp.IS_DIRTY: is_dirty_prop,
    SimObjVariableProp.IS_USED_UP: is_used_up_prop,
    SimObjVariableProp.IS_COOKED: is_cooked_prop,
    SimObjVariableProp.TEMPERATURE: temperature_prop,
    SimObjVariableProp.IS_SLICED: is_sliced_prop,
    SimObjVariableProp.IS_OPEN: is_open_prop,
    SimObjVariableProp.OPENNESS: openness_prop,
    SimObjVariableProp.IS_PICKED_UP: is_picked_up_prop,
}
