"""
Module for defining tasks for the AI2THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Hashable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, NewType

import networkx as nx

if TYPE_CHECKING:
    from rl_ai2thor.utils.ai2thor_types import EventLike


# %% === Enums ===
# TODO: Add more relations
class RelationTypeId(StrEnum):
    """Relations between items."""

    RECEPTACLE_OF = "receptacle_of"
    CONTAINED_IN = "contained_in"
    # CLOSE_TO = "close_to"


# TODO: Add support for more mass and salient materials.
class ObjFixedPropId(StrEnum):
    """Fixed properties of objects in AI2THOR."""

    OBJECT_TYPE = "objectType"
    IS_INTERACTABLE = "isInteractable"
    RECEPTACLE = "receptacle"
    TOGGLEABLE = "toggleable"
    BREAKABLE = "breakable"
    CAN_FILL_WITH_LIQUID = "canFillWithLiquid"
    DIRTYABLE = "dirtyable"
    CAN_BE_USED_UP = "canBeUsedUp"
    COOKABLE = "cookable"
    IS_HEAT_SOURCE = "isHeatSource"
    IS_COLD_SOURCE = "isColdSource"
    SLICEABLE = "sliceable"
    OPENABLE = "openable"
    PICKUPABLE = "pickupable"
    MOVEABLE = "moveable"
    # MASS = "mass"
    # SALIENT_MATERIALS = "salientMaterials"


# TODO: Add support for position, rotation and distance.
class ObjVariablePropId(StrEnum):
    """Variable properties of objects in AI2THOR."""

    VISIBLE = "visible"
    IS_TOGGLED = "isToggled"
    IS_BROKEN = "isBroken"
    IS_FILLED_WITH_LIQUID = "isFilledWithLiquid"
    FILL_LIQUID = "fillLiquid"
    IS_DIRTY = "isDirty"
    IS_USED_UP = "isUsedUp"
    IS_COOKED = "isCooked"
    TEMPERATURE = "temperature"
    IS_SLICED = "isSliced"
    IS_OPEN = "isOpen"
    OPENNESS = "openness"
    IS_PICKED_UP = "isPickedUp"
    # POSITION = "position"
    # ROTATION = "rotation"
    # DISTANCE = "distance"


# TODO: Change this to a union of enums instead of type alias.
type ObjPropId = ObjFixedPropId | ObjVariablePropId
type ObjMetadataId = ObjPropId | str
type PropValue = float | bool | TemperatureValue


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


ObjId = NewType("ObjId", str)


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
        self._candidate_required_properties_rel = {}
        self.organized_relations: dict[T, dict[RelationTypeId, Relation]] = {}
        self.candidate_ids: set[ObjId] = set()

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
    def candidate_required_properties(self) -> dict[ObjPropId, Any]:
        """
        Return a dictionary containing the properties required for an object to be a candidate for the item.

        Returns:
            candidate_properties (dict[ObjPropId, Any]): Dictionary containing the candidate required properties.
        """
        return {
            **self._candidate_required_properties_prop,
            **self._candidate_required_properties_rel,
        }

    def is_obj_candidate(self, obj_metadata: dict[ObjMetadataId, Any]) -> bool:
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

    def _get_properties_satisfaction(self, obj_metadata: dict[ObjMetadataId, Any]) -> dict[ObjPropId, bool]:
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
        self, candidate_metadata: dict[ObjMetadataId, Any]
    ) -> dict[T, dict[RelationTypeId, set[ObjId]]]:
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
        self, obj_metadata: dict[ObjMetadataId, Any]
    ) -> dict[
        Literal["properties", "relations"],
        dict[ObjPropId, bool] | dict[T, dict[RelationTypeId, set[ObjId]]],
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
            dict[ObjPropId, bool] | dict[T, dict[RelationTypeId, set[ObjId]]],
        ] = {
            "properties": self._get_properties_satisfaction(obj_metadata),
            "relations": self._get_relations_semi_satisfying_objects(obj_metadata),
        }
        return results

    def _compute_all_obj_results(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> dict[
        Literal["properties", "relations"],
        dict[ObjPropId, dict[ObjId, bool]] | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
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
            dict[ObjPropId, dict[ObjId, bool]] | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
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
            dict[ObjPropId, dict[ObjId, bool]] | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ],
    ) -> dict[
        ObjId,
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
            ObjId,
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
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> tuple[
        set[ObjId],
        dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]] | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ],
        dict[ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]],
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
        obj_1_id: ObjId,
        obj_2_id: ObjId,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]] | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ],
        objects_scores: dict[ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]],
    ) -> ObjId | Literal["equal", "incomparable"]:
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
        obj_1_id: ObjId,
        obj_2_id: ObjId,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]] | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ],
        objects_scores: dict[
            ObjId,
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
        relations_results: dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]] = objects_results["relations"]  # type: ignore  # TODO: Delete type ignore after simplifying the type
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
        candidate_ids: list[ObjId],
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

    def compute_interesting_assignments(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> tuple[
        list[dict[TaskItem[T], ObjId]],
        dict[
            TaskItem[T],
            dict[
                Literal["properties", "relations"],
                dict[ObjPropId, dict[ObjId, bool]] | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
            ],
        ],
        dict[
            TaskItem[T],
            dict[
                ObjId,
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
                ObjId,
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


# === Properties and Relations ===
# TODO: Add support for automatic scene validity and action validity checking (action group, etc)
# TODO: Add support for allowing property checking with other ways than equality.
# TODO: Check if we need to add a hash
class ItemProp:
    """Property of an item in the definition of a task."""

    def __init__(
        self,
        target_ai2thor_property: ObjPropId,
        value_type: type,
        is_fixed: bool = False,
        candidate_required_property: ObjFixedPropId | None = None,
        candidate_required_property_value: Any | None = None,
    ) -> None:
        """Initialize the Property object."""
        self.target_ai2thor_property = target_ai2thor_property
        self.value_type = value_type
        self.is_fixed = is_fixed
        self.candidate_required_property = target_ai2thor_property if is_fixed else candidate_required_property
        self.candidate_required_property_value = candidate_required_property_value


# TODO: Add support to parameterize the relations (e.g. distance in CLOSE_TO)
class Relation(ABC):
    """A relation between two items in the definition of a task."""

    type_id: RelationTypeId
    inverse_relation_type_id: RelationTypeId
    candidate_required_prop: ObjFixedPropId | None = None
    candidate_required_prop_value: Any | None = None

    def __init__(self, main_item: TaskItem, related_item: TaskItem) -> None:
        """
        Initialize the main and related objects of the relation.

        Args:
            main_item (TaskItem): The main item in the relation.
            related_item (TaskItem): The related item to which the main item is related.
        """
        self.main_item = main_item
        self.related_item = related_item

    def __str__(self) -> str:
        return f"{self.main_item} is {self.type_id} {self.related_item}"

    def __repr__(self) -> str:
        return f"Relation({self.type_id}, {self.main_item}, {self.related_item})"

    def __hash__(self) -> int:
        return hash((self.type_id, self.main_item.id, self.related_item.id))

    @abstractmethod
    def _extract_related_object_ids(self, main_obj_metadata: dict[ObjMetadataId, Any]) -> list[ObjId]:
        """Return the list of the ids of the main object's related objects according to the relation."""

    def is_semi_satisfied(self, main_obj_metadata: dict[ObjMetadataId, Any]) -> bool:
        """
        Return True if the relation is semi satisfied.

        A relation is semi satisfied if the main object is correctly
        related to a candidate of the related item (but no related
        object might be assigned to the related item).
        """
        return any(
            related_object_id in self.related_item.candidate_ids
            for related_object_id in self._extract_related_object_ids(main_obj_metadata)
        )

    def get_satisfying_related_object_ids(self, main_obj_metadata: dict[ObjMetadataId, Any]) -> set[ObjId]:
        """Return related item's candidate's ids that satisfy the relation with the given main object."""
        return {
            related_object_id
            for related_object_id in self._extract_related_object_ids(main_obj_metadata)
            if related_object_id in self.related_item.candidate_ids
        }


class ReceptacleOfRelation(Relation):
    """
    A relation of the form "main_item is a receptacle of related_item".

    The inverse relation is ContainedInRelation.

    Args:
        main_item (TaskItem): The main item in the relation.
        related_item (TaskItem): The related item that is contained in the main item.
    """

    type_id = RelationTypeId.RECEPTACLE_OF
    inverse_relation_type_id = RelationTypeId.CONTAINED_IN
    candidate_required_prop = ObjFixedPropId.RECEPTACLE
    candidate_required_prop_value = True

    def __init__(self, main_item: TaskItem, related_item: TaskItem) -> None:
        """
        Initialize the main and related objects of the relation.

        Args:
            main_item (TaskItem): The main item in the relation.
            related_item (TaskItem): The related item to which the main item is related.
        """
        super().__init__(main_item, related_item)

    @staticmethod
    def _extract_related_object_ids(main_obj_metadata: dict[ObjMetadataId, Any]) -> list[ObjId]:
        """Return the ids of the objects contained in the main object."""
        return main_obj_metadata["receptacleObjectIds"]


class ContainedInRelation(Relation):
    """
    A relation of the form "main_item is_contained_in related_item".

    The inverse relation is ReceptacleOfRelation.
    """

    type_id = RelationTypeId.CONTAINED_IN
    inverse_relation_type_id = RelationTypeId.RECEPTACLE_OF
    candidate_required_prop = ObjFixedPropId.PICKUPABLE
    candidate_required_prop_value = True

    def __init__(self, main_item: TaskItem, related_item: TaskItem) -> None:
        """
        Initialize the main and related objects of the relation.

        Args:
            main_item (TaskItem): The main item in the relation.
            related_item (TaskItem): The related item to which the main item is related.
        """
        super().__init__(main_item, related_item)

    @staticmethod
    def _extract_related_object_ids(main_obj_metadata: dict[ObjMetadataId, Any]) -> list[ObjId]:
        """Return the ids of the objects containing the main object."""
        return main_obj_metadata["parentReceptacles"]


# === Tasks ===
type TaskDict[T: Hashable] = dict[T, dict[Literal["properties", "relations"], dict]]


# TODO: Add support for weighted properties and relations
# TODO: Add support for agent properties
# TODO: Remove networkx dependency
class GraphTask[T: Hashable]:
    """
    Base class for graph tasks.

    Graph tasks are tasks that can be represented as a graph representing the
    relations between objects in the scene in the terminal states.

    The vertices of the graph are the "items", corresponding to unique objects in the scene
    and the edges are the relations between them (e.g. "receptacle_of" if the item is
    supposed to contain another item).
    The items are represented by the properties required for satisfying the task (e.g.
    "objectType", "visible", "isSliced", "temperature",...).


    Graph tasks are defined using a task description dictionary representing each task
    item, its properties and relations.

    Example of task description dictionary for the task of placing an apple in a plate:
    target_objects = {
            "receptacle_plate": {
                "properties": {"objectType": "Plate"},
            },
            "placed_apple": {
                "properties": {"objectType": "Apple"},
                "relations": {"receptacle_plate": ["contained_in"]},
            },
        }

    This task contains 2 items defined by their unique identifiers (of generic type "T")
    in the description dictionary; "receptacle_plate" and "placed_apple". Both items have
    a required type (others available properties are available in ObjFixedPropId and
    ObjVariablePropId enums) and placed_apple is related to receptacle_plate one relation
    by the "contained_in relation (other relations available in RelationTypeId enum).
    Note: Inverse relations are automatically added to the graph, so it is not necessary
    to add them manually when creating the task.

    Attributes:
        items (List[T]): List of items representing unique objects in the scene.
        task_graph (nx.DiGraph): Directed graph representing the relations between items.
        overlap_classes (List[ItemOverlapClass]): List of overlap classes containing items
            with overlapping candidates.

    Methods:
        reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
            Reset the task with the information of the event.
        get_task_advancement(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
            Return the task advancement and whether the task is completed.
        full_initialize_items_and_relations_from_dict(task_description_dict: TaskDict) -> list[TaskItem[T]]
            Create and initialize TaskItems for the graph task.

    """

    def __init__(
        self,
        task_description_dict: TaskDict[T],
    ) -> None:
        """
        Initialize the task graph as defined in the task description dictionary.

        Args:
            task_description_dict (dict[T, dict[Literal["properties", "relations"], dict]]):
                Dictionary describing the items and their properties and relations.
        """
        self.items = self.full_initialize_items_and_relations_from_dict(task_description_dict)

        # Initialize the task graph
        self.task_graph = nx.DiGraph()

        # TODO: Check if we create ids and mappings for the items
        for item in self.items:
            self.task_graph.add_node(item)
            for relation in item.relations:
                self.task_graph.add_edge(item, relation.related_item)
                self.task_graph[item][relation.related_item][relation.type_id] = relation
        # TODO: Check if we keep the graph (unused for now)

        self.overlap_classes: list[ItemOverlapClass] = []

    # TODO: Finish this method
    def reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        Initialize the candidates of the items with the objects
        in the scene and compute the overlap classes.

        Args:
            event (EventLike): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Initialize the candidates of the items
        for item in self.items:
            for obj_metadata in event.metadata["objects"]:
                if item.is_obj_candidate(obj_metadata):
                    item.candidate_ids.add(obj_metadata["objectId"])

        # Compute the overlap classes
        overlap_classes = {}
        for item in self.items:
            item_class_idx = None
            remaining_candidates_ids = item.candidate_ids.copy()
            for class_idx, overlap_class in overlap_classes.items():
                if not remaining_candidates_ids.isdisjoint(overlap_class["candidate_ids"]):
                    # Item belongs to the overlap class
                    # Keep only new candidates
                    remaining_candidates_ids -= overlap_class["candidate_ids"]
                    if item_class_idx is None:
                        # Add the item to the overlap class
                        item_class_idx = class_idx
                        overlap_class["items"].append(item)
                        overlap_class["candidate_ids"] |= item.candidate_ids
                    else:
                        # Merge the overlap classes
                        overlap_classes[item_class_idx]["items"].extend(overlap_class["items"])
                        overlap_classes[item_class_idx]["candidate_ids"] |= overlap_class["candidate_ids"]
                        del overlap_classes[class_idx]

            if item_class_idx is None:
                # Create a new overlap class
                overlap_classes[len(overlap_classes)] = {
                    "items": [item],
                    "candidate_ids": item.candidate_ids,
                }

        self.overlap_classes = [
            ItemOverlapClass(
                items=overlap_class["items"],
                candidate_ids=list(overlap_class["candidate_ids"]),
            )
            for overlap_class in overlap_classes.values()
        ]

        # Compute max task advancement
        # Total number of properties and relations of the items
        self.max_task_advancement = sum(len(item.properties) + len(item.relations) for item in self.items)

        # Return initial task advancement
        return self.get_task_advancement(event)

    # TODO: Add trying only the top k interesting assignments according to the maximum possible score (need to order the list of interesting candidates then the list of interesting assignments for each overlap class)
    # TODO: Implement this method
    def get_task_advancement(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the task advancement and whether the task is completed.

        To compute the task advancement, we consider every interesting global assignment
        of objects to the items. A global assignment is a dictionary mapping each item of
        the task to an object in the scene (as opposed to a a overlap class assignment).
        To construct the set of global assignments, we take the cartesian product of the
        assignments of the overlap classes. Interesting global assignments are the ones
        constructed with only interesting overlap class assignments.

        For a given global assignment, the task advancement is the sum of the property
        scores of the assigned objects for each item and the sum of their relations scores
        for relations that have a satisfying object assigned to the related item (i.e. we
        consider strictly satisfied relations and not semi satisfied relations).

        Args:
            event (EventLike): Event corresponding to the state of the scene.

        Returns:
            task_advancement (float): Task advancement.
            is_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Compute the interesting assignments for each overlap class and the results and scores of each candidate for each item
        overlap_classes_assignment_data = [
            overlap_class.compute_interesting_assignments(event.metadata["objects"])
            for overlap_class in self.overlap_classes
        ]
        # Extract the interesting assignments, results and scores
        interesting_assignments = [data[0] for data in overlap_classes_assignment_data]
        # Merge the results and scores of the items
        items_results = {
            item: item_result
            for overlap_class_assignment_data in overlap_classes_assignment_data
            for item, item_result in overlap_class_assignment_data[1].items()
        }
        items_scores = {
            item: item_score
            for overlap_class_assignment_data in overlap_classes_assignment_data
            for item, item_score in overlap_class_assignment_data[2].items()
        }

        # Construct a generator of the cartesian product of the interesting assignments
        assignment_products = itertools.product(*interesting_assignments)

        max_task_advancement = 0
        best_assignment = {}
        is_terminated = False

        # Compute the task advancement for each global assignment
        for assignment_product in assignment_products:
            # Merge the assignments of the overlap classes
            global_assignment = {
                item: obj_id
                for overlap_class_assignment in assignment_product
                for item, obj_id in overlap_class_assignment.items()
            }
            # Property scores
            task_advancement = sum(
                items_scores[item][obj_id]["sum_property_scores"] for item, obj_id in global_assignment.items()
            )
            # Strictly satisfied relation scores
            for item, obj_id in global_assignment.items():
                item_relations_results: dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]] = items_results[item][
                    "relations"
                ]  # type: ignore  # TODO: Delete type ignore after simplifying the type
                for related_item, relations in item_relations_results.items():
                    related_item_assigned_obj_id = global_assignment[related_item]
                    for relations_by_obj_id in relations.values():
                        satisfying_obj_ids = relations_by_obj_id[obj_id]
                        if related_item_assigned_obj_id in satisfying_obj_ids:
                            task_advancement += 1
            if task_advancement > max_task_advancement:
                max_task_advancement = task_advancement
                best_assignment = global_assignment
                if max_task_advancement == self.max_task_advancement:
                    is_terminated = True
                    break

        # Add info about the task advancement
        info = {
            # Add best assignment, mapping between item ids and the assigned object ids
            "best_assignment": {item.id: obj_id for item, obj_id in best_assignment.items()},
            "task_advancement": max_task_advancement,
        }
        # TODO: Add other info

        return max_task_advancement, is_terminated, info

    # TODO: Check if we keep the relation set too (might not be necessary)
    # TODO: Change to only return a plain list of items
    # TODO: Add support for overriding relations and keep the most restrictive one
    @staticmethod
    def full_initialize_items_and_relations_from_dict(
        task_description_dict: TaskDict[T],
    ) -> list[TaskItem[T]]:
        """
        Create and initialize TaskItems for the graph task.

        TaskItems are created as defined in the task description
        dictionary representing the items and their properties and relations.
        The items fully initialized with their relations and the inverse
        relations are also added.

        Args:
            task_description_dict (dict[T, dict[Literal["properties", "relations"], dict]]):
                Dictionary describing the items and their properties and relations.

        Returns:
            items (list[TaskItem]): List of the items of the task.
        """
        items = {
            item_id: TaskItem(
                item_id,
                {obj_prop_id_to_item_prop[prop]: value for prop, value in item_dict["properties"].items()},
            )
            for item_id, item_dict in task_description_dict.items()
        }
        organized_relations = {
            main_item_id: {
                related_item_id: {
                    relation_type_id: relation_type_id_to_relation[relation_type_id](
                        items[main_item_id], items[related_item_id]
                    )
                    for relation_type_id in relation_type_ids
                }
                for related_item_id, relation_type_ids in main_item_dict["relations"].items()
            }
            for main_item_id, main_item_dict in task_description_dict.items()
        }

        # Add inverse relations
        for main_item_id, main_item_relations in organized_relations.items():
            for related_item_id, relations_dict in main_item_relations.items():
                for relation in relations_dict.values():
                    inverse_relation_type_id = relation.inverse_relation_type_id
                    if inverse_relation_type_id not in organized_relations[related_item_id][main_item_id]:
                        organized_relations[related_item_id][main_item_id][inverse_relation_type_id] = (
                            relation_type_id_to_relation[
                                inverse_relation_type_id
                            ](items[related_item_id], items[main_item_id])
                        )

        # Set item relations
        for item_id, item in items.items():
            item.relations = {
                relation
                for relations_dict in organized_relations[item_id].values()
                for relation in relations_dict.values()
            }

        return list(items.values())

    # TODO: Improve this
    def __repr__(self) -> str:
        return f"GraphTask({self.task_graph})"


class PlaceObject(GraphTask):
    """
    Task for placing a target object in a given receptacle.

    This is equivalent to the pick_and_place_simple task from Alfred.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.
        """
        self.placed_object_type = placed_object_type
        self.receptacle_type = receptacle_type

        target_objects: TaskDict[str] = {
            "receptacle": {
                "properties": {"objectType": self.receptacle_type},
                "relations": {"placed_object": ["receptacle_of"]},
            },
            "placed_object": {
                "properties": {"objectType": self.placed_object_type},
                "relations": {"receptacle": ["contained_in"]},
            },
        }
        super().__init__(target_objects)

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.placed_object_type} in {self.receptacle_type}"


# %% === Item properties ===
object_type_prop = ItemProp(
    ObjFixedPropId.OBJECT_TYPE,
    value_type=str,
    is_fixed=True,
)
is_interactable_prop = ItemProp(
    ObjFixedPropId.IS_INTERACTABLE,
    value_type=bool,
    is_fixed=True,
)
receptacle_prop = ItemProp(
    ObjFixedPropId.RECEPTACLE,
    value_type=bool,
    is_fixed=True,
)
toggleable_prop = ItemProp(
    ObjFixedPropId.TOGGLEABLE,
    value_type=bool,
    is_fixed=True,
)
breakable_prop = ItemProp(
    ObjFixedPropId.BREAKABLE,
    value_type=bool,
    is_fixed=True,
)
can_fill_with_liquid_prop = ItemProp(
    ObjFixedPropId.CAN_FILL_WITH_LIQUID,
    value_type=bool,
    is_fixed=True,
)
dirtyable_prop = ItemProp(
    ObjFixedPropId.DIRTYABLE,
    value_type=bool,
    is_fixed=True,
)
can_be_used_up_prop = ItemProp(
    ObjFixedPropId.CAN_BE_USED_UP,
    value_type=bool,
    is_fixed=True,
)
cookable_prop = ItemProp(
    ObjFixedPropId.COOKABLE,
    value_type=bool,
    is_fixed=True,
)
is_heat_source_prop = ItemProp(
    ObjFixedPropId.IS_HEAT_SOURCE,
    value_type=bool,
    is_fixed=True,
)
is_cold_source_prop = ItemProp(
    ObjFixedPropId.IS_COLD_SOURCE,
    value_type=bool,
    is_fixed=True,
)
sliceable_prop = ItemProp(
    ObjFixedPropId.SLICEABLE,
    value_type=bool,
    is_fixed=True,
)
openable_prop = ItemProp(
    ObjFixedPropId.OPENABLE,
    value_type=bool,
    is_fixed=True,
)
pickupable_prop = ItemProp(
    ObjFixedPropId.PICKUPABLE,
    value_type=bool,
    is_fixed=True,
)
moveable_prop = ItemProp(
    ObjFixedPropId.MOVEABLE,
    value_type=bool,
    is_fixed=True,
)
visible_prop = ItemProp(
    ObjVariablePropId.VISIBLE,
    value_type=bool,
)
is_toggled_prop = ItemProp(
    ObjVariablePropId.IS_TOGGLED,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.TOGGLEABLE,
    candidate_required_property_value=True,
)
is_broken_prop = ItemProp(
    ObjVariablePropId.IS_BROKEN,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.BREAKABLE,
    candidate_required_property_value=True,
)
is_filled_with_liquid_prop = ItemProp(
    ObjVariablePropId.IS_FILLED_WITH_LIQUID,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.CAN_FILL_WITH_LIQUID,
    candidate_required_property_value=True,
)
fill_liquid_prop = ItemProp(
    ObjVariablePropId.FILL_LIQUID,
    value_type=FillableLiquid,
    candidate_required_property=ObjFixedPropId.CAN_FILL_WITH_LIQUID,
    candidate_required_property_value=True,
)
is_dirty_prop = ItemProp(
    ObjVariablePropId.IS_DIRTY,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.DIRTYABLE,
    candidate_required_property_value=True,
)
is_used_up_prop = ItemProp(
    ObjVariablePropId.IS_USED_UP,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.CAN_BE_USED_UP,
    candidate_required_property_value=True,
)
is_cooked_prop = ItemProp(
    ObjVariablePropId.IS_COOKED,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.COOKABLE,
    candidate_required_property_value=True,
)
temperature_prop = ItemProp(
    ObjVariablePropId.TEMPERATURE,
    value_type=TemperatureValue,
)
is_sliced_prop = ItemProp(
    ObjVariablePropId.IS_SLICED,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.SLICEABLE,
    candidate_required_property_value=True,
)
is_open_prop = ItemProp(
    ObjVariablePropId.IS_OPEN,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.OPENABLE,
    candidate_required_property_value=True,
)
openness_prop = ItemProp(
    ObjVariablePropId.OPENNESS,
    value_type=float,
    candidate_required_property=ObjFixedPropId.OPENABLE,
    candidate_required_property_value=True,
)
is_picked_up_prop = ItemProp(
    ObjVariablePropId.IS_PICKED_UP,
    value_type=bool,
    candidate_required_property=ObjFixedPropId.PICKUPABLE,
    candidate_required_property_value=True,
)


# %% === Property and relation ids mapping ===
obj_prop_id_to_item_prop = {
    ObjFixedPropId.OBJECT_TYPE: object_type_prop,
    ObjFixedPropId.IS_INTERACTABLE: is_interactable_prop,
    ObjFixedPropId.RECEPTACLE: receptacle_prop,
    ObjFixedPropId.TOGGLEABLE: toggleable_prop,
    ObjFixedPropId.BREAKABLE: breakable_prop,
    ObjFixedPropId.CAN_FILL_WITH_LIQUID: can_fill_with_liquid_prop,
    ObjFixedPropId.DIRTYABLE: dirtyable_prop,
    ObjFixedPropId.CAN_BE_USED_UP: can_be_used_up_prop,
    ObjFixedPropId.COOKABLE: cookable_prop,
    ObjFixedPropId.IS_HEAT_SOURCE: is_heat_source_prop,
    ObjFixedPropId.IS_COLD_SOURCE: is_cold_source_prop,
    ObjFixedPropId.SLICEABLE: sliceable_prop,
    ObjFixedPropId.OPENABLE: openable_prop,
    ObjFixedPropId.PICKUPABLE: pickupable_prop,
    ObjFixedPropId.MOVEABLE: moveable_prop,
    ObjVariablePropId.VISIBLE: visible_prop,
    ObjVariablePropId.IS_TOGGLED: is_toggled_prop,
    ObjVariablePropId.IS_BROKEN: is_broken_prop,
    ObjVariablePropId.IS_FILLED_WITH_LIQUID: is_filled_with_liquid_prop,
    ObjVariablePropId.FILL_LIQUID: fill_liquid_prop,
    ObjVariablePropId.IS_DIRTY: is_dirty_prop,
    ObjVariablePropId.IS_USED_UP: is_used_up_prop,
    ObjVariablePropId.IS_COOKED: is_cooked_prop,
    ObjVariablePropId.TEMPERATURE: temperature_prop,
    ObjVariablePropId.IS_SLICED: is_sliced_prop,
    ObjVariablePropId.IS_OPEN: is_open_prop,
    ObjVariablePropId.OPENNESS: openness_prop,
    ObjVariablePropId.IS_PICKED_UP: is_picked_up_prop,
}

relation_type_id_to_relation = {
    RelationTypeId.RECEPTACLE_OF: ReceptacleOfRelation,
    RelationTypeId.CONTAINED_IN: ContainedInRelation,
}
