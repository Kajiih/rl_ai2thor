"""
Task items in AI2THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable
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


type PropValue = int | float | bool | TemperatureValue | SimObjectType


# %% === Properties  ===
# TODO: Add support for automatic scene validity and action validity checking (action group, etc)
# TODO: Add support for allowing property checking with other ways than equality.
# TODO: Check if we need to add a hash
class ItemProp:
    """Property of an item in the definition of a task."""

    def __init__(
        self,
        target_ai2thor_property: SimObjProp,
        value_type: type,
        is_fixed: bool = False,
        candidate_required_property: SimObjFixedProp | None = None,
        candidate_required_property_value: Any | None = None,
    ) -> None:
        """Initialize the Property object."""
        self.target_ai2thor_property = target_ai2thor_property
        self.value_type = value_type
        self.is_fixed = is_fixed
        self.candidate_required_property = target_ai2thor_property if is_fixed else candidate_required_property
        self.candidate_required_property_value = candidate_required_property_value

    def __str__(self) -> str:
        return f"{self.target_ai2thor_property}"

    def __repr__(self) -> str:
        return f"ItemProp({self.target_ai2thor_property})"


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
        properties: dict[ItemProp, PropValue],
    ) -> None:
        """
        Initialize the TaskItem object.

        Args:
            t_id (T): The ID of the item as defined in the task description.
            properties (dict[ItemProp, PropValue]): The properties of the item.
        """
        self.id = t_id
        self.properties = properties

        # Infer the candidate required properties from the item properties
        self._candidate_required_properties_prop = {
            prop.candidate_required_property: (value if prop.is_fixed else prop.candidate_required_property_value)
            for prop, value in self.properties.items()
            if prop.candidate_required_property is not None
        }

        # Other attributes
        self._candidate_required_properties_rel: dict[SimObjProp, PropValue] = {}
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
        self._candidate_required_properties_rel.update({
            relation.candidate_required_prop: relation.candidate_required_prop_value
            for relation in relations
            if relation.candidate_required_prop is not None
        })

        # Delete duplicate relations if any
        self._relations = {
            relation for relation_set in self.organized_relations.values() for relation in relation_set.values()
        }

    @property
    def candidate_required_properties(self) -> dict[SimObjProp, Any]:
        """
        Return a dictionary containing the properties required for an object to be a candidate for the item.

        Returns:
            candidate_properties (dict[ObjPropId, Any]): Dictionary containing the candidate required properties.
        """
        return {
            **self._candidate_required_properties_prop,
            **self._candidate_required_properties_rel,
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
            obj_metadata[prop_id] == prop_value for prop_id, prop_value in self.candidate_required_properties.items()
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
            prop.target_ai2thor_property: obj_metadata[prop.target_ai2thor_property] == prop_value
            for prop, prop_value in self.properties.items()
        }

    def _get_relations_semi_satisfying_objects(
        self, candidate_metadata: SimObjMetadata
    ) -> dict[T, dict[RelationTypeId, set[SimObjId]]]:
        """
        Return the dictionary of satisfying objects with the given candidate for each relations.

        The relations are organized by related item id.

        Args:
            candidate_metadata (SimObjMetadata): Metadata of the candidate.

        Returns:
            semi_satisfying_objects (dict[T, dict[RelationTypeId, set[SimObjId]]]): Dictionary indicating which objects are semi-satisfying the relations with the given object.
        """
        return {
            related_item_id: {
                relation.type_id: relation.get_satisfying_related_object_ids(candidate_metadata)
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations
        }

    def _compute_obj_results(
        self, obj_metadata: SimObjMetadata
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

        Returns:
            results (dict[Literal["properties", "relations"], dict[SimObjProp, bool] | dict[T, dict[RelationTypeId, set[SimObjId]]]]): Results of the object for the item.
        """
        results: dict[
            Literal["properties", "relations"],
            dict[SimObjProp, bool] | dict[T, dict[RelationTypeId, set[SimObjId]]],
        ] = {
            "properties": self._get_properties_satisfaction(obj_metadata),
            "relations": self._get_relations_semi_satisfying_objects(obj_metadata),
        }
        return results

    def _compute_all_obj_results(
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
            properties_results (dict[SimObjProp, dict[SimObjId, bool]]):
                Results of each object for the item properties.
            relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
        """
        properties_results = {
            prop.target_ai2thor_property: {
                obj_id: scene_objects_dict[obj_id][prop.target_ai2thor_property] == prop_value
                for obj_id in self.candidate_ids
            }
            for prop, prop_value in self.properties.items()
        }

        relations_results = {
            related_item_id: {
                relation.type_id: {
                    obj_id: relation.get_satisfying_related_object_ids(scene_objects_dict[obj_id])
                    # obj_id: obj_id in relation.get_satisfying_related_object_ids(scene_objects_dict[obj_id])  #TODO: Delete
                    for obj_id in self.candidate_ids
                }
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations
        }

        return properties_results, relations_results

    def _compute_all_obj_scores(
        self,
        properties_results: dict[SimObjProp, dict[SimObjId, bool]],
        relations_results: dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
    ) -> tuple[
        dict[SimObjId, int],
        dict[SimObjId, int],
    ]:
        """
        Return the property and relation scores of each candidate of the item.

        Args:
            properties_results (dict[SimObjProp, dict[SimObjId, bool]]):
                Results of each object for the item properties.
            relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.

        Returns:
            properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.
            relations_scores (dict[SimObjId, int]):
                Relation scores of each object for the item.
        """
        properties_scores: dict[SimObjId, int] = {
            obj_id: sum(bool(properties_results[prop_id][obj_id]) for prop_id in properties_results)
            for obj_id in self.candidate_ids
        }
        relations_scores = {
            obj_id: sum(
                1
                for related_item_id in relations_results
                for relation_type_id in relations_results[related_item_id]
                if relations_results[related_item_id][relation_type_id][obj_id]
            )
            for obj_id in self.candidate_ids
        }

        return properties_scores, relations_scores

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
            properties_results (dict[SimObjProp, dict[SimObjId, bool]]):
                Results of each object for the item properties.
            relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
            properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.
            relations_scores (dict[SimObjId, int]):
                Relation scores of each object for the item.
        """
        # Compute the results of each object for the item
        properties_results, relations_results = self._compute_all_obj_results(scene_objects_dict)

        # Compute the scores of each object for the item
        properties_scores, relations_scores = self._compute_all_obj_scores(properties_results, relations_results)

        # Remove the candidates that have a stronger alternative
        interesting_candidates = list(self.candidate_ids)
        for i, candidate_id in enumerate(interesting_candidates):
            for j, other_candidate_id in enumerate(interesting_candidates[i + 1 :]):
                stronger_candidate = self._get_stronger_candidate(
                    candidate_id, other_candidate_id, relations_results, properties_scores, relations_scores
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
        max_prop_score = max(properties_scores[candidate_id] for candidate_id in interesting_candidates)
        # Check if there is a candidate with the highest property score
        if max_prop_score not in [properties_scores[candidate_id] for candidate_id in interesting_candidates]:
            # Add the candidate with the highest property score
            for candidate_id in self.candidate_ids:
                if properties_scores[candidate_id] == max_prop_score:
                    interesting_candidates.append(candidate_id)
                    break

        return set(interesting_candidates), properties_results, relations_results, properties_scores, relations_scores

    def _get_stronger_candidate(
        self,
        obj_1_id: SimObjId,
        obj_2_id: SimObjId,
        relations_results: dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
        properties_scores: dict[SimObjId, int],
        relations_scores: dict[SimObjId, int],
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
            relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
            properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.
            relations_scores (dict[SimObjId, int]):
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
            properties_scores[obj_1_id] + relations_scores[obj_1_id]
            < properties_scores[obj_2_id] + relations_scores[obj_2_id]
        ):
            obj1_stronger = False
        else:
            obj1_stronger = self._is_stronger_candidate_than(obj_1_id, obj_2_id, relations_results, properties_scores)

        # Pre check: Sp(obj_1_id) + Sr(obj_1_id) > Sp(obj_2_id) + Sr(obj_2_id)
        if (
            properties_scores[obj_1_id] + relations_scores[obj_1_id]
            > properties_scores[obj_2_id] + relations_scores[obj_2_id]
        ):
            obj2_stronger = False
        else:
            obj2_stronger = self._is_stronger_candidate_than(obj_2_id, obj_1_id, relations_results, properties_scores)

        if obj1_stronger:
            return "equal" if obj2_stronger else obj_1_id
        return obj_2_id if obj2_stronger else "incomparable"

    @staticmethod
    def _is_stronger_candidate_than(
        obj_1_id: SimObjId,
        obj_2_id: SimObjId,
        relations_results: dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]],
        properties_scores: dict[SimObjId, int],
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
            relations_results (dict[T, dict[RelationTypeId, dict[SimObjId, set[SimObjId]]]]):
                Results of each object for the item relations.
            properties_scores (dict[SimObjId, int]):
                Property scores of each object for the item.

        Returns:
            is_stronger (bool): True if the first candidate is stronger than the second candidate.
        """
        sp_x = properties_scores[obj_1_id]
        sp_y = properties_scores[obj_2_id]
        sr_y = properties_scores[obj_2_id]

        # Calculate d[x,y]
        d_xy = 0
        for related_item_id in relations_results:
            for relation_type_id in relations_results[related_item_id]:
                x_sat_obj_ids = relations_results[related_item_id][relation_type_id][obj_1_id]
                y_sat_obj_ids = relations_results[related_item_id][relation_type_id][obj_2_id]
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
        self.valid_assignments = [
            permutation
            for permutation in candidate_permutations
            if all(obj_id in item.candidate_ids for item, obj_id in permutation.items())
        ]

        if not self.valid_assignments:
            raise NoValidAssignmentError(self)

    def compute_interesting_assignments(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> tuple[
        list[dict[TaskItem[T], SimObjId]],
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
            interesting_assignments (list[dict[TaskItem[T], SimObjId]]):
                List of the interesting assignments of objects to the items in the overlap class.
            all_properties_results (dict[TaskItem[T], dict[SimObjProp, bool]]):
                Results of each object for the item properties.
            all_relation_results (dict[TaskItem[T], dict[T, dict[RelationTypeId, set[SimObjId]]]]):
                Results of each object for the item relations.
            all_properties_scores (dict[TaskItem[T], dict[SimObjProp, int]]):
                Property scores of each object for the item.
            all_relations_scores (dict[TaskItem[T], dict[T, int]]):
                Relation scores of each object for the item.
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
        """Initialize the NoCandidateError object."""
        self.item = item

    def __str__(self) -> str:
        return f"Item '{self.item}' has no candidate with the required properties {self.item.candidate_required_properties}"


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
        """Initialize the NoValidAssignmentError object."""
        self.overlap_class = overlap_class

        # Check that the items have at least one valid candidate
        for item in self.overlap_class.items:
            if not item.candidate_ids:
                raise NoCandidateError(item)

    def __str__(self) -> str:
        return f"Overlap class '{self.overlap_class}' has no valid assignment."


# %% === Property Definitions ===
object_type_prop = ItemProp(
    SimObjFixedProp.OBJECT_TYPE,
    value_type=str,
    is_fixed=True,
)
is_interactable_prop = ItemProp(
    SimObjFixedProp.IS_INTERACTABLE,
    value_type=bool,
    is_fixed=True,
)
receptacle_prop = ItemProp(
    SimObjFixedProp.RECEPTACLE,
    value_type=bool,
    is_fixed=True,
)
toggleable_prop = ItemProp(
    SimObjFixedProp.TOGGLEABLE,
    value_type=bool,
    is_fixed=True,
)
breakable_prop = ItemProp(
    SimObjFixedProp.BREAKABLE,
    value_type=bool,
    is_fixed=True,
)
can_fill_with_liquid_prop = ItemProp(
    SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    value_type=bool,
    is_fixed=True,
)
dirtyable_prop = ItemProp(
    SimObjFixedProp.DIRTYABLE,
    value_type=bool,
    is_fixed=True,
)
can_be_used_up_prop = ItemProp(
    SimObjFixedProp.CAN_BE_USED_UP,
    value_type=bool,
    is_fixed=True,
)
cookable_prop = ItemProp(
    SimObjFixedProp.COOKABLE,
    value_type=bool,
    is_fixed=True,
)
is_heat_source_prop = ItemProp(
    SimObjFixedProp.IS_HEAT_SOURCE,
    value_type=bool,
    is_fixed=True,
)
is_cold_source_prop = ItemProp(
    SimObjFixedProp.IS_COLD_SOURCE,
    value_type=bool,
    is_fixed=True,
)
sliceable_prop = ItemProp(
    SimObjFixedProp.SLICEABLE,
    value_type=bool,
    is_fixed=True,
)
openable_prop = ItemProp(
    SimObjFixedProp.OPENABLE,
    value_type=bool,
    is_fixed=True,
)
pickupable_prop = ItemProp(
    SimObjFixedProp.PICKUPABLE,
    value_type=bool,
    is_fixed=True,
)
moveable_prop = ItemProp(
    SimObjFixedProp.MOVEABLE,
    value_type=bool,
    is_fixed=True,
)
visible_prop = ItemProp(
    SimObjVariableProp.VISIBLE,
    value_type=bool,
)
is_toggled_prop = ItemProp(
    SimObjVariableProp.IS_TOGGLED,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.TOGGLEABLE,
    candidate_required_property_value=True,
)
is_broken_prop = ItemProp(
    SimObjVariableProp.IS_BROKEN,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.BREAKABLE,
    candidate_required_property_value=True,
)
is_filled_with_liquid_prop = ItemProp(
    SimObjVariableProp.IS_FILLED_WITH_LIQUID,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    candidate_required_property_value=True,
)
fill_liquid_prop = ItemProp(
    SimObjVariableProp.FILL_LIQUID,
    value_type=FillableLiquid,
    candidate_required_property=SimObjFixedProp.CAN_FILL_WITH_LIQUID,
    candidate_required_property_value=True,
)
is_dirty_prop = ItemProp(
    SimObjVariableProp.IS_DIRTY,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.DIRTYABLE,
    candidate_required_property_value=True,
)
is_used_up_prop = ItemProp(
    SimObjVariableProp.IS_USED_UP,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.CAN_BE_USED_UP,
    candidate_required_property_value=True,
)
is_cooked_prop = ItemProp(
    SimObjVariableProp.IS_COOKED,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.COOKABLE,
    candidate_required_property_value=True,
)
temperature_prop = ItemProp(
    SimObjVariableProp.TEMPERATURE,
    value_type=TemperatureValue,
)
is_sliced_prop = ItemProp(
    SimObjVariableProp.IS_SLICED,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.SLICEABLE,
    candidate_required_property_value=True,
)
is_open_prop = ItemProp(
    SimObjVariableProp.IS_OPEN,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.OPENABLE,
    candidate_required_property_value=True,
)
openness_prop = ItemProp(
    SimObjVariableProp.OPENNESS,
    value_type=float,
    candidate_required_property=SimObjFixedProp.OPENABLE,
    candidate_required_property_value=True,
)
is_picked_up_prop = ItemProp(
    SimObjVariableProp.IS_PICKED_UP,
    value_type=bool,
    candidate_required_property=SimObjFixedProp.PICKUPABLE,
    candidate_required_property_value=True,
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
