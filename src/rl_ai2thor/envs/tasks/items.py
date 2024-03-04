"""
Task items in AI2THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from rl_ai2thor.envs.sim_objects import SimObjectType, SimObjFixedProp, SimObjMetadata, SimObjProp, SimObjVariableProp

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


type PropValue = float | bool | TemperatureValue


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
        self.candidate_ids: set[SimObjectType] = set()

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

    def is_candidate(self, obj_metadata: dict[SimObjMetadata, Any]) -> bool:
        """
        Return True if the given object is a valid candidate for the item.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:
            is_candidate (bool): True if the given object is a valid candidate for the item.
        """
        return all(
            obj_metadata[prop_id] == prop_value for prop_id, prop_value in self.candidate_required_properties.items()
        )

    def _get_properties_satisfaction(self, obj_metadata: dict[SimObjMetadata, Any]) -> dict[SimObjProp, bool]:
        """
        Return a dictionary indicating which properties are satisfied by the given object.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:
            prop_satisfaction (dict[ObjPropId, bool]): Dictionary indicating which properties are satisfied by the given object.
        """
        return {
            prop.target_ai2thor_property: obj_metadata[prop.target_ai2thor_property] == prop_value
            for prop, prop_value in self.properties.items()
        }

    def _get_relations_semi_satisfying_objects(
        self, candidate_metadata: dict[SimObjMetadata, Any]
    ) -> dict[T, dict[RelationTypeId, set[SimObjectType]]]:
        """
        Return the dictionary of satisfying objects with the given candidate for each relations.

        The relations are organized by related item id.

        Args:
            candidate_metadata (dict[ObjMetadataPropId, Any]): Metadata of the candidate.

        Returns:
            semi_satisfying_objects (dict[T, dict[RelationTypeId, set[ObjId]]]): Dictionary indicating which objects are semi-satisfying the relations with the given object.
        """
        return {
            related_item_id: {
                relation.type_id: relation.get_satisfying_related_object_ids(candidate_metadata)
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations
        }

    def _compute_obj_results(
        self, obj_metadata: dict[SimObjMetadata, Any]
    ) -> dict[
        Literal["properties", "relations"],
        dict[SimObjProp, bool] | dict[T, dict[RelationTypeId, set[SimObjectType]]],
    ]:  # TODO: Simplify this big type after finish the implementation
        """
        Return the results dictionary of the object for the item.

        The results are the satisfaction of each property and the satisfying
        objects of each relation of the item.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:
            results (dict[Literal["properties", "relations"], dict]): Results of the object for the item.
        """
        results: dict[
            Literal["properties", "relations"],
            dict[SimObjProp, bool] | dict[T, dict[RelationTypeId, set[SimObjectType]]],
        ] = {
            "properties": self._get_properties_satisfaction(obj_metadata),
            "relations": self._get_relations_semi_satisfying_objects(obj_metadata),
        }
        return results

    def _compute_all_obj_results(
        self, scene_objects_dict: dict[SimObjectType, Any]
    ) -> dict[
        Literal["properties", "relations"],
        dict[SimObjProp, dict[SimObjectType, bool]]
        | dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]],
    ]:
        """
        Return the results dictionary with the results of each candidate of the item.

        Args:
            scene_objects_dict (dict[ObjId, Any]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            results (dict[ObjId, dict]): Results of each object for the item.
        """
        results: dict[
            Literal["properties", "relations"],
            dict[SimObjProp, dict[SimObjectType, bool]]
            | dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]],
        ] = {
            "properties": {
                prop.target_ai2thor_property: {
                    obj_id: scene_objects_dict[obj_id][prop.target_ai2thor_property] == prop_value
                    for obj_id in self.candidate_ids
                }
                for prop, prop_value in self.properties.items()
            },
            "relations": {
                # use get_satisfying_related_object_ids instead of is_semi_satisfied
                related_item_id: {
                    relation.type_id: {
                        obj_id: obj_id in relation.get_satisfying_related_object_ids(scene_objects_dict[obj_id])
                        for obj_id in self.candidate_ids
                    }
                    for relation in self.organized_relations[related_item_id].values()
                }
                for related_item_id in self.organized_relations
            },  # type: ignore  # TODO: Delete type ignore after simplifying the type
        }

        return results

    def _compute_all_obj_scores(
        self,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[SimObjProp, dict[SimObjectType, bool]]
            | dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]],
        ],
    ) -> dict[
        SimObjectType,
        dict[Literal["sum_property_scores", "sum_relation_scores"], float],
    ]:
        """
        Return the property and relation scores of each candidate of the item.

        Args:
            objects_results (dict[ObjId, dict[Literal["properties", "relations"], dict]]):
                Results of each object for the item.

        Returns:
            scores (dict[ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]]):
                Scores of each object for the item.
        """
        scores: dict[
            SimObjectType,
            dict[Literal["sum_property_scores", "sum_relation_scores"], float],
        ] = {
            obj_id: {
                "sum_property_scores": sum(
                    bool(objects_results["properties"][prop_id][obj_id])  # type: ignore  # TODO: Delete type ignore after simplifying the type
                    for prop_id in objects_results["properties"]
                ),
                "sum_relation_scores": sum(
                    len(objects_results["relations"][relation_type_id][obj_id]) > 0  # type: ignore  # TODO: Delete type ignore after simplifying the type
                    for relation_type_id in objects_results["relations"]
                ),
            }
            for obj_id in self.candidate_ids
        }
        return scores

    def compute_interesting_candidates(
        self, scene_objects_dict: dict[SimObjectType, Any]
    ) -> tuple[
        set[SimObjectType],
        dict[
            Literal["properties", "relations"],
            dict[SimObjProp, dict[SimObjectType, bool]]
            | dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]],
        ],
        dict[SimObjectType, dict[Literal["sum_property_scores", "sum_relation_scores"], float]],
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
            scene_objects_dict (dict[ObjId, Any]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            interesting_candidates (set[ObjId]): Set of interesting candidates for the item.
            objects_results (dict[ObjId, dict[Literal["properties", "relations"], dict]]):
                Results of each object for the item.
            objects_scores (dict[ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]]):
                Scores of each object for the item.
        """
        # Compute the results of each object for the item
        objects_results = self._compute_all_obj_results(scene_objects_dict)

        # Compute the scores of each object for the item
        objects_scores = self._compute_all_obj_scores(objects_results)

        # Remove the candidates that have a stronger alternative
        interesting_candidates = list(self.candidate_ids)
        for i, candidate_id in enumerate(interesting_candidates):
            for j, other_candidate_id in enumerate(interesting_candidates[i + 1 :]):
                stronger_candidate = self._get_stronger_candidate(
                    candidate_id, other_candidate_id, objects_results, objects_scores
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
        max_prop_score = max(
            objects_scores[candidate_id]["sum_property_scores"] for candidate_id in interesting_candidates
        )
        # Check if there is a candidate with the highest property score
        if max_prop_score not in [
            objects_scores[candidate_id]["sum_property_scores"] for candidate_id in interesting_candidates
        ]:
            # Add the candidate with the highest property score

            for candidate_id in self.candidate_ids:
                if objects_scores[candidate_id]["sum_property_scores"] == max_prop_score:
                    interesting_candidates.append(candidate_id)
                    break

        return set(interesting_candidates), objects_results, objects_scores

    def _get_stronger_candidate(
        self,
        obj_1_id: SimObjectType,
        obj_2_id: SimObjectType,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[SimObjProp, dict[SimObjectType, bool]]
            | dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]],
        ],
        objects_scores: dict[SimObjectType, dict[Literal["sum_property_scores", "sum_relation_scores"], float]],
    ) -> SimObjectType | Literal["equal", "incomparable"]:
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
            obj_1_id (ObjId): First candidate object id.
            obj_2_id (ObjId): Second candidate object id.
            objects_results (dict[ObjId, dict[Literal["properties", "relations"], dict]]): Results of
                each object for the item.
            objects_scores (dict[ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]]):
                Scores of each object for the item.

        Returns:
            stronger_candidate (ObjId|Literal["equal", "incomparable"]): The stronger candidate
                between the two given candidates or "equal" if they have same strength or
                "incomparable" if they cannot be compared.

        """
        obj1_stronger = True
        obj2_stronger = True
        # Pre check: Sp(obj_1_id) + Sr(obj_1_id) < Sp(obj_2_id) + Sr(obj_2_id)
        if (
            objects_scores[obj_1_id]["sum_property_scores"] + objects_scores[obj_1_id]["sum_relation_scores"]
            < objects_scores[obj_2_id]["sum_property_scores"] + objects_scores[obj_2_id]["sum_relation_scores"]
        ):
            obj1_stronger = False
        else:
            obj1_stronger = self._is_stronger_candidate_than(obj_1_id, obj_2_id, objects_results, objects_scores)

        # Pre check: Sp(obj_1_id) + Sr(obj_1_id) > Sp(obj_2_id) + Sr(obj_2_id)
        if (
            objects_scores[obj_1_id]["sum_property_scores"] + objects_scores[obj_1_id]["sum_relation_scores"]
            > objects_scores[obj_2_id]["sum_property_scores"] + objects_scores[obj_2_id]["sum_relation_scores"]
        ):
            obj2_stronger = False
        else:
            obj2_stronger = self._is_stronger_candidate_than(obj_2_id, obj_1_id, objects_results, objects_scores)

        if obj1_stronger:
            return "equal" if obj2_stronger else obj_1_id
        return obj_2_id if obj2_stronger else "incomparable"

    @staticmethod
    def _is_stronger_candidate_than(
        obj_1_id: SimObjectType,
        obj_2_id: SimObjectType,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[SimObjProp, dict[SimObjectType, bool]]
            | dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]],
        ],
        objects_scores: dict[
            SimObjectType,
            dict[Literal["sum_property_scores", "sum_relation_scores"], float],
        ],
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
            obj_1_id (ObjId): First candidate object id.
            obj_2_id (ObjId): Second candidate object id.
            objects_results (dict[Literal["properties", "relations"], dict]): Results of
                each object for the item.
            objects_scores (dict[ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]]):
                Scores of each object for the item.

        Returns:
            is_stronger (bool): True if the first candidate is stronger than the second candidate.
        """
        sp_x = objects_scores[obj_1_id]["sum_property_scores"]
        sp_y = objects_scores[obj_2_id]["sum_property_scores"]
        sr_y = objects_scores[obj_2_id]["sum_relation_scores"]

        # Calculate d[x,y]
        d_xy = 0
        relations_results: dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]] = objects_results[
            "relations"
        ]  # type: ignore  # TODO: Delete type ignore after simplifying the type
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
        return f"TaskItem({self.id})"

    def __hash__(self) -> int:
        return hash(self.id)


class ItemOverlapClass[T: Hashable]:
    """A group of items whose sets of candidates overlap."""

    def __init__(
        self,
        items: list[TaskItem[T]],
        candidate_ids: list[SimObjectType],
    ) -> None:
        """
        Initialize the overlap class with the given items and candidate ids.

        Args:
            items (list[TaskItem[T]]): The items in the overlap class.
            candidate_ids (list[ObjId]): The candidate ids of candidates in the overlap class.
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
        self, scene_objects_dict: dict[SimObjectType, Any]
    ) -> tuple[
        list[dict[TaskItem[T], SimObjectType]],
        dict[
            TaskItem[T],
            dict[
                Literal["properties", "relations"],
                dict[SimObjProp, dict[SimObjectType, bool]]
                | dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]],
            ],
        ],
        dict[
            TaskItem[T],
            dict[
                SimObjectType,
                dict[Literal["sum_property_scores", "sum_relation_scores"], float],
            ],
        ],
    ]:
        """
        Return the interesting assignments of objects to the items in the overlap class.

        The interesting assignments are the ones that can lead to a maximum of task advancement
        depending on the assignment of objects in the other overlap classes.
        Interesting assignments are the ones where each all assigned objects are interesting
        candidates for their item.

        For more details about the definition of interesting candidates, see the
        compute_interesting_candidates method of the TaskItem class.

        Args:
            scene_objects_dict (dict[ObjId, Any]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            interesting_assignments (list[dict[TaskItem[T], ObjId]]):
                List of the interesting assignments of objects to the items in the overlap class.
        """
        interesting_candidates_data = {
            item: item.compute_interesting_candidates(scene_objects_dict) for item in self.items
        }
        # Extract the interesting candidates, results and scores
        interesting_candidates = {item: data[0] for item, data in interesting_candidates_data.items()}
        items_results = {item: data[1] for item, data in interesting_candidates_data.items()}
        items_scores: dict[
            TaskItem[T],
            dict[
                SimObjectType,
                dict[Literal["sum_property_scores", "sum_relation_scores"], float],
            ],
        ] = {item: data[2] for item, data in interesting_candidates_data.items()}

        # Filter the valid assignments to keep only the ones with interesting candidates
        interesting_assignments = [
            assignment
            for assignment in self.valid_assignments
            if all(assignment[item] in interesting_candidates[item] for item in self.items)
        ]

        return interesting_assignments, items_results, items_scores

    def __str__(self) -> str:
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
