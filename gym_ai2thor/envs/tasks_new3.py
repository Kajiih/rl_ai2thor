"""
Module for defining tasks for the AI2THOR RL environment.

TODO: Finish module docstrings.
"""

from __future__ import annotations

import itertools
from enum import StrEnum
from abc import ABC, abstractmethod
from typing import Any, Hashable, Literal, Optional, Type, NewType

import networkx as nx

from ai2thor_types import EventLike


# %% === Enums ===
# TODO: Add more relations
class RelationTypeId(StrEnum):
    """
    Relations between items.
    """

    RECEPTACLE_OF = "receptacle_of"
    CONTAINED_IN = "contained_in"
    # CLOSE_TO = "close_to"


# TODO: Add support for more mass and salient materials.
class ObjFixedPropId(StrEnum):
    """
    Fixed properties of objects in AI2THOR.
    """

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
    """
    Variable properties of objects in AI2THOR.
    """

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
type PropValue = float | bool | Literal["Hot", "Cold", "RoomTemp", "water"]

ObjId = NewType("ObjId", str)


# %% === Items ===
class TaskItem[T: Hashable]:
    """
    An item in the definition of a task.

    TODO: Finish docstring.
    """

    def __init__(
        self,
        id: T,
        properties: dict[ItemProp, PropValue],
    ) -> None:
        self.id = id
        self.properties = properties

        # Infer the candidate required properties from the item properties
        self._candidate_required_properties_prop = {
            prop.candidate_required_property: (
                value if prop.is_fixed else prop.candidate_required_property_value
            )
            for prop, value in self.properties.items()
            if prop.candidate_required_property is not None
        }

        # Other attributes
        self._candidate_required_properties_rel = {}
        self.organized_relations: dict[T, dict[RelationTypeId, Relation]] = {}
        self.candidate_ids: set[ObjId] = set()

    @property
    def relations(self) -> set[Relation]:
        return self._relations

    @relations.setter
    def relations(self, relations: set[Relation]) -> None:
        """
        Set the relations of the item.

        Automatically update the organized_relations and candidate_required_properties
        attributes.
        """
        self.organized_relations.update(
            {
                relation.related_item.id: {
                    relation.type_id: relation,
                }
                for relation in relations
            }
        )
        self._candidate_required_properties_rel.update(
            {
                relation.candidate_required_prop: relation.candidate_required_prop_value
                for relation in relations
                if relation.candidate_required_prop is not None
            }
        )

        # Delete duplicate relations if any
        self._relations = {
            relation
            for relation_set in self.organized_relations.values()
            for relation in relation_set.values()
        }

    @property
    def candidate_required_properties(self) -> dict[ObjPropId, Any]:
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
        for prop_id, prop_value in self.candidate_required_properties.items():
            if obj_metadata[prop_id] != prop_value:
                return False
        return True

    # TODO? Delete: Unused
    def _nb_satisfied_prop(self, obj_metadata: dict[ObjMetadataId, Any]) -> int:
        """
        Return the number of properties satisfied by the given object.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:
            nb_satisfied_prop (int): Number of properties satisfied by the given object.
        """
        nb_satisfied_prop = sum(
            1
            for prop, prop_value in self.properties.items()
            if obj_metadata[prop.target_ai2thor_property] == prop_value
        )
        return nb_satisfied_prop

    def _get_properties_satisfaction(
        self, obj_metadata: dict[ObjMetadataId, Any]
    ) -> dict[ObjPropId, bool]:
        """
        Return a dictionary indicating which properties are satisfied by the given object.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:
            prop_satisfaction (dict[ObjPropId, bool]): Dictionary indicating which properties are satisfied by the given object.
        """

        prop_satisfaction = {
            prop.target_ai2thor_property: obj_metadata[prop.target_ai2thor_property]
            == prop_value
            for prop, prop_value in self.properties.items()
        }
        return prop_satisfaction

    # TODO? Delete: Unused
    def _nb_semi_satisfied_relations(
        self, obj_metadata: dict[ObjMetadataId, Any]
    ) -> int:
        """
        Return the number of relations semi satisfied by the given object.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:
            nb_semi_satisfied_relations (int): Number of relations semi satisfied by the given object.
        """
        nb_semi_satisfied_relations = sum(
            1 for relation in self.relations if relation.is_semi_satisfied(obj_metadata)
        )
        return nb_semi_satisfied_relations

    # TODO? Delete: Unused
    def _get_relations_semi_satisfaction(
        self, obj_metadata: dict[ObjMetadataId, Any]
    ) -> dict[T, dict[RelationTypeId, bool]]:
        """
        Return a dictionary indicating which relations are semi satisfied by the given object.

        The relations are organized by related item.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:
            relation_semi_satisfaction (dict[T, dict[RelationTypeId, bool]]): Dictionary
                indicating which relations are semi satisfied by the given object.
        """
        relation_semi_satisfaction = {
            related_item_id: {
                relation.type_id: relation.is_semi_satisfied(obj_metadata)
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations.keys()
        }
        return relation_semi_satisfaction

    def _get_relations_semi_satisfying_objects(
        self, obj_metadata: dict[ObjMetadataId, Any]
    ) -> dict[T, dict[RelationTypeId, set[ObjId]]]:
        """
        Return a dictionary indicating which objects are semi-satisfying the relations
        with the given object.

        The relations are organized by related item.

        Args:
            obj_metadata (dict[ObjMetadataPropId, Any]): Object metadata.

        Returns:

        """
        relation_semi_satisfying_objects = {
            related_item_id: {
                relation.type_id: relation.get_satisfying_related_object_ids(
                    obj_metadata
                )
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations.keys()
        }
        return relation_semi_satisfying_objects

    def _compute_obj_results(
        self, obj_metadata: dict[ObjMetadataId, Any]
    ) -> dict[
        Literal["properties", "relations"],
        dict[ObjPropId, bool] | dict[T, dict[RelationTypeId, set[ObjId]]],
    ]:  # TODO: Simplify this big type after finish the implementation
        """
        Return the results dictionnary of the object for the item.

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

    # TODO? Delete?
    def _compute_all_obj_results_old(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> dict[
        ObjId,
        dict[
            Literal["properties", "relations"],
            dict[ObjPropId, bool]
            | dict[
                T,
                dict[RelationTypeId, set[ObjId]],
            ],
        ],
    ]:
        """
        Return the results dictionnary of each object for the item.

        Args:
            scene_objects_dict (dict[ObjId, Any]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            results (dict[ObjId, dict]): Results of each object for the item.
        """
        results = {
            obj_id: self._compute_obj_results(scene_objects_dict[obj_id])
            for obj_id in self.candidate_ids
        }
        return results

    # TODO? Delete? May bot be correct
    def _compute_all_obj_results_slow(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> dict[
        Literal["properties", "relations"],
        dict[ObjPropId, dict[ObjId, bool]]
        | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
    ]:
        """
        Return the results dictionnary with the results of each object for the item.

        Args:
            scene_objects_dict (dict[ObjId, Any]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            results (dict[ObjId, dict]): Results of each object for the item.
        """
        results_by_obj_id: dict[
            ObjId,
            dict[
                Literal["properties", "relations"],
                dict[ObjPropId, bool] | dict[T, dict[RelationTypeId, set[ObjId]]],
            ],
        ] = {
            obj_id: self._compute_obj_results(scene_objects_dict[obj_id])
            for obj_id in self.candidate_ids
        }

        # Reorganize the results with the object_ids at the deepest level
        results: dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]]
            | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ] = {
            "properties": {
                prop_id: {
                    obj_id: obj_results["properties"][prop_id.target_ai2thor_property]
                    for obj_id, obj_results in results_by_obj_id.items()
                }
                for prop_id in self.properties.keys()
            },
            "relations": {
                related_item_id: {
                    relation_type_id: {
                        obj_id: obj_results["relations"][related_item_id][
                            relation_type_id
                        ]
                        for obj_id, obj_results in results_by_obj_id.items()
                    }
                    for relation_type_id in self.organized_relations[
                        related_item_id
                    ].keys()
                }
                for related_item_id in self.organized_relations.keys()
            },
        }

    # TODO: Check if we keep this one
    def _compute_all_obj_results(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> dict[
        Literal["properties", "relations"],
        dict[ObjPropId, dict[ObjId, bool]]
        | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
    ]:
        """
        Return the results dictionnary with the results of each object for the item.

        Args:
            scene_objects_dict (dict[ObjId, Any]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            results (dict[ObjId, dict]): Results of each object for the item.
        """
        results: dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]]
            | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ] = {
            "properties": {
                prop.target_ai2thor_property: {
                    obj_id: scene_objects_dict[obj_id][prop.target_ai2thor_property]
                    == prop_value
                    for obj_id in self.candidate_ids
                }
                for prop, prop_value in self.properties.items()
            },
            "relations": {
                # use get_satisfying_related_object_ids instead of is_semi_satisfied
                related_item_id: {
                    relation.type_id: {
                        obj_id: obj_id
                        in relation.get_satisfying_related_object_ids(
                            scene_objects_dict[obj_id]
                        )
                        for obj_id in self.candidate_ids
                    }
                    for relation in self.organized_relations[related_item_id].values()
                }
                for related_item_id in self.organized_relations.keys()
            },  # type: ignore  # TODO: Delete type ignore after simplyfing the type
        }

        return results

    # TODO? Delete?
    def _compute_all_obj_scores_old(
        self,
        objects_results: dict[ObjId, dict[Literal["properties", "relations"], dict]],
    ) -> dict[
        ObjId,
        dict[Literal["sum_property_scores", "sum_relation_scores"], float],
    ]:
        """
        Return the property and relation scores of each object for the item.

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
                    1
                    for prop_satisfaction in obj_results["properties"].values()
                    if prop_satisfaction
                ),
                "sum_relation_scores": sum(
                    1
                    for satisfying_object_ids in obj_results["relations"].values()
                    if len(satisfying_object_ids) > 0  # type: ignore  # TODO: Delete type ignore after simplyfing the type
                ),
            }
            for obj_id, obj_results in objects_results.items()
        }
        return scores

    def _compute_all_obj_scores(
        self,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]]
            | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ],
    ) -> dict[
        ObjId,
        dict[Literal["sum_property_scores", "sum_relation_scores"], float],
    ]:
        """
        Return the property and relation scores of each object for the item.

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
                    1
                    for prop_id in objects_results["properties"]
                    if objects_results["properties"][prop_id][obj_id]  # type: ignore  # TODO: Delete type ignore after simplyfing the type
                ),
                "sum_relation_scores": sum(
                    1
                    for relation_type_id in objects_results["relations"]
                    if len(objects_results["relations"][relation_type_id][obj_id]) > 0  # type: ignore  # TODO: Delete type ignore after simplyfing the type
                ),
            }
            for obj_id in self.candidate_ids
        }
        return scores

    def filter_interesting_candidates_old(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> set[ObjId]:
        """
        Return the set of interesting candidates for the item.

        The interesting candidates are those that can lead to a maximum of task advancement
        depending on the assignment of objects to the other items.

        A candidate is *strong* if it has no stricly *stronger* candidate among the other
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
        """

        # Compute the results of each object for the item
        objects_results = self._compute_all_obj_results_old(scene_objects_dict)

        # Compute the scores of each object for the item
        objects_scores = self._compute_all_obj_scores_old(objects_results)

        # Remove the candidates that have a stronger alternative
        interesting_candidates = list(self.candidate_ids)
        for i, candidate_id in enumerate(interesting_candidates):
            for j, other_candidate_id in enumerate(interesting_candidates[i + 1 :]):
                stronger_candidate = self.get_stronger_candidate_old(
                    candidate_id, other_candidate_id, objects_results, objects_scores
                )
                if stronger_candidate == candidate_id or stronger_candidate == "equal":
                    # In the equal case, we can keep any of the two candidates
                    # Remove the other candidate
                    interesting_candidates.pop(i + j + 1)
                elif stronger_candidate == other_candidate_id:
                    # Remove the candidate
                    interesting_candidates.pop(i)
                    break

        # Add a candidate with the highest property score if none of those are already added
        max_prop_score = max(
            objects_scores[candidate_id]["sum_property_scores"]
            for candidate_id in interesting_candidates
        )
        # Check if there is a candidate with the highest property score
        if max_prop_score not in [
            objects_scores[candidate_id]["sum_property_scores"]
            for candidate_id in interesting_candidates
        ]:
            # Add the candidate with the highest property score

            for candidate_id in self.candidate_ids:
                if (
                    objects_scores[candidate_id]["sum_property_scores"]
                    == max_prop_score
                ):
                    interesting_candidates.append(candidate_id)
                    break

        return set(interesting_candidates)

        return interesting_candidates

    def filter_interesting_candidates(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> set[ObjId]:
        """
        Return the set of interesting candidates for the item.

        The interesting candidates are those that can lead to a maximum of task advancement
        depending on the assignment of objects to the other items.

        A candidate is *strong* if it has no stricly *stronger* candidate among the other
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
        """

        # Compute the results of each object for the item
        objects_results = self._compute_all_obj_results(scene_objects_dict)

        # Compute the scores of each object for the item
        objects_scores = self._compute_all_obj_scores(objects_results)

        # Remove the candidates that have a stronger alternative
        interesting_candidates = list(self.candidate_ids)
        for i, candidate_id in enumerate(interesting_candidates):
            for j, other_candidate_id in enumerate(interesting_candidates[i + 1 :]):
                stronger_candidate = self.get_stronger_candidate(
                    candidate_id, other_candidate_id, objects_results, objects_scores
                )
                if stronger_candidate == candidate_id or stronger_candidate == "equal":
                    # In the equal case, we can keep any of the two candidates
                    # Remove the other candidate
                    interesting_candidates.pop(i + j + 1)
                elif stronger_candidate == other_candidate_id:
                    # Remove the candidate
                    interesting_candidates.pop(i)
                    break

        # Add a candidate with the highest property score if none of those are already added
        max_prop_score = max(
            objects_scores[candidate_id]["sum_property_scores"]
            for candidate_id in interesting_candidates
        )
        # Check if there is a candidate with the highest property score
        if max_prop_score not in [
            objects_scores[candidate_id]["sum_property_scores"]
            for candidate_id in interesting_candidates
        ]:
            # Add the candidate with the highest property score

            for candidate_id in self.candidate_ids:
                if (
                    objects_scores[candidate_id]["sum_property_scores"]
                    == max_prop_score
                ):
                    interesting_candidates.append(candidate_id)
                    break

        return set(interesting_candidates)

        return interesting_candidates

    def __str__(self) -> str:
        return f"{self.id}"

    def __repr__(self) -> str:
        return f"TaskItem({self.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def get_stronger_candidate_old(
        self,
        obj_1_id: ObjId,
        obj_2_id: ObjId,
        objects_results: dict[ObjId, dict[Literal["properties", "relations"], dict]],
        objects_scores: dict[
            ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]
        ],
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
            objects_scores[obj_1_id]["sum_property_scores"]
            + objects_scores[obj_1_id]["sum_relation_scores"]
            < objects_scores[obj_2_id]["sum_property_scores"]
            + objects_scores[obj_2_id]["sum_relation_scores"]
        ):
            obj1_stronger = False
        else:
            obj1_stronger = self._is_stronger_candidate_than_old(
                obj_1_id, obj_2_id, objects_results, objects_scores
            )

        # Pre check: Sp(obj_1_id) + Sr(obj_1_id) > Sp(obj_2_id) + Sr(obj_2_id)
        if (
            objects_scores[obj_1_id]["sum_property_scores"]
            + objects_scores[obj_1_id]["sum_relation_scores"]
            > objects_scores[obj_2_id]["sum_property_scores"]
            + objects_scores[obj_2_id]["sum_relation_scores"]
        ):
            obj2_stronger = False
        else:
            obj2_stronger = self._is_stronger_candidate_than_old(
                obj_2_id, obj_1_id, objects_results, objects_scores
            )

        if obj1_stronger and obj2_stronger:
            return "equal"
        elif obj1_stronger:
            return obj_1_id
        elif obj2_stronger:
            return obj_2_id
        else:
            return "incomparable"

    def get_stronger_candidate(
        self,
        obj_1_id: ObjId,
        obj_2_id: ObjId,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]]
            | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
        ],
        objects_scores: dict[
            ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]
        ],
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
            objects_scores[obj_1_id]["sum_property_scores"]
            + objects_scores[obj_1_id]["sum_relation_scores"]
            < objects_scores[obj_2_id]["sum_property_scores"]
            + objects_scores[obj_2_id]["sum_relation_scores"]
        ):
            obj1_stronger = False
        else:
            obj1_stronger = self._is_stronger_candidate_than(
                obj_1_id, obj_2_id, objects_results, objects_scores
            )

        # Pre check: Sp(obj_1_id) + Sr(obj_1_id) > Sp(obj_2_id) + Sr(obj_2_id)
        if (
            objects_scores[obj_1_id]["sum_property_scores"]
            + objects_scores[obj_1_id]["sum_relation_scores"]
            > objects_scores[obj_2_id]["sum_property_scores"]
            + objects_scores[obj_2_id]["sum_relation_scores"]
        ):
            obj2_stronger = False
        else:
            obj2_stronger = self._is_stronger_candidate_than(
                obj_2_id, obj_1_id, objects_results, objects_scores
            )

        if obj1_stronger and obj2_stronger:
            return "equal"
        elif obj1_stronger:
            return obj_1_id
        elif obj2_stronger:
            return obj_2_id
        else:
            return "incomparable"

    def _is_stronger_candidate_than_old(
        self,
        obj_1_id: ObjId,
        obj_2_id: ObjId,
        objects_results: dict[
            ObjId,
            dict[
                Literal["properties", "relations"],
                dict[ObjPropId, bool]
                | dict[
                    T,
                    dict[RelationTypeId, set[ObjId]],
                ],
            ],
        ],
        objects_scores: dict[
            ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]
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
            objects_results (dict[ObjId, dict[Literal["properties", "relations"], dict]]): Results of
                each object for the item.
            objects_scores (dict[ObjId, dict[Literal["sum_property_scores", "sum_relation_scores"], float]]):
                Scores of each object for the item.

        Returns:
            is_stronger (bool): True if the first candidate is stronger than the second candidate.
        """
        sp_x = objects_scores[obj_1_id]["sum_property_scores"]
        sp_y = objects_scores[obj_2_id]["sum_property_scores"]
        sr_x = objects_scores[obj_1_id]["sum_relation_scores"]
        sr_y = objects_scores[obj_2_id]["sum_relation_scores"]

        # Calculate d[x,y]
        d_xy = 0
        x_rel_results: dict[T, dict[RelationTypeId, set[ObjId]]] = objects_results[
            obj_1_id
        ][
            "relations"
        ]  # type: ignore  # TODO: Delete type ignore after simplyfing the type
        y__rel_results: dict[T, dict[RelationTypeId, set[ObjId]]] = objects_results[
            obj_2_id
        ][
            "relations"
        ]  # type: ignore  # TODO: Delete type ignore after simplyfing the type
        for related_item_id in x_rel_results.keys():
            for relation_type_id in x_rel_results[related_item_id].keys():
                x_sat_obj_ids = x_rel_results[related_item_id][relation_type_id]
                y_sat_obj_ids = y__rel_results[related_item_id][relation_type_id]
                if y_sat_obj_ids.issubset(x_sat_obj_ids):
                    d_xy += 1

        return sp_x - (sp_y + sr_y) + d_xy > 0

    def _is_stronger_candidate_than(
        self,
        obj_1_id: ObjId,
        obj_2_id: ObjId,
        objects_results: dict[
            Literal["properties", "relations"],
            dict[ObjPropId, dict[ObjId, bool]]
            | dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]],
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
        sr_x = objects_scores[obj_1_id]["sum_relation_scores"]
        sr_y = objects_scores[obj_2_id]["sum_relation_scores"]

        # Calculate d[x,y]
        d_xy = 0
        relations_results: dict[T, dict[RelationTypeId, dict[ObjId, set[ObjId]]]] = objects_results["relations"]  # type: ignore  # TODO: Delete type ignore after simplyfing the type
        for related_item_id in relations_results.keys():
            for relation_type_id in relations_results[related_item_id].keys():
                x_sat_obj_ids = relations_results[related_item_id][relation_type_id][
                    obj_1_id
                ]
                y_sat_obj_ids = relations_results[related_item_id][relation_type_id][
                    obj_2_id
                ]
                if y_sat_obj_ids.issubset(x_sat_obj_ids):
                    d_xy += 1

        return sp_x - (sp_y + sr_y) + d_xy > 0


# TODO? Add support for giving some score for semi satisfied relations and using this info in the selection of interesting objects/assignments
# TODO: Implement ItemOverlapClass
class ItemOverlapClass[T: Hashable]:
    """
    A group of items whose candidates can overlap.
    """

    def __init__(
        self,
        items: list[TaskItem[T]],
        candidate_ids: list[ObjId],
    ) -> None:
        self.items = items
        self.candidate_ids = candidate_ids

        # Compute all vald assignments of objects to the items in the overlap class
        # One permuation is represented by a dictionary mapping the item ids to the assigned object ids
        candidate_permutations = [
            dict(zip(self.items, permutation))
            for permutation in itertools.permutations(
                self.candidate_ids, len(self.items)
            )
            # TODO?: Replace candidate ids by their index in the list to make it more efficient? Probably need this kind of optimizations
        ]
        valid_assignments = []
        for permutation in candidate_permutations:
            if all(
                obj_id in item.candidate_ids for item, obj_id in permutation.items()
            ):
                valid_assignments.append(permutation)

        self.valid_assignments: list[dict[TaskItem, ObjId]] = valid_assignments

        # Compute interesting properties of the overlap class
        self.intesting_ai2thor_properties: set[ObjPropId] = {
            prop.target_ai2thor_property
            for item in self.items
            for prop in item.properties
        }

    # TODO: Implement
    def compute_optimal_assignments(
        self, scene_objects_dict: dict[ObjId, Any]
    ) -> tuple[list[dict[TaskItem[T], ObjId]], float]:
        """
        Return the optimal assignments of objects to the items in the
        overlap class and the optimal score.

        Currently, the optimal assignments are the ones that maximize the
        number of satisfied properties and semi satisfied relations.

        Args:
            scene_objects_dict (dict[ObjId, Any]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            optimal_assignments (list[dict[TaskItem, ObjId]]): List of the optimal
                assignments of objects to the items in the overlap class.
            best_score (float): Score of the optimal assignments.
        """

        # TODO: Update this function
        # Compute the scores of each assignment
        assignment_scores = [
            sum(item_scores[item][obj_id] for item, obj_id in assignment.items())
            for assignment in self.valid_assignments
        ]

        # Compute the optimal assignments
        best_score = max(assignment_scores)
        optimal_assignments = [
            assignment
            for assignment, score in zip(self.valid_assignments, assignment_scores)
            if score == best_score
        ]

        return optimal_assignments, best_score


# === Properties and Relations ===
# TODO: Add support for automatic scene validity and action validity checking.
# TODO: Add support for allowing property checking with other ways than equality.
# TODO: Check if we need to add a hash
class ItemProp:
    """
    Property of an item in the definition of a task.

    TODO: Finish docstring.
    """

    def __init__(
        self,
        target_ai2thor_property: ObjPropId,
        value_type: Type,
        is_fixed: bool = False,
        candidate_required_property: Optional[ObjFixedPropId] = None,
        candidate_required_property_value: Optional[Any] = None,
    ) -> None:
        self.target_ai2thor_property = target_ai2thor_property
        self.value_type = value_type
        self.is_fixed = is_fixed
        self.candidate_required_property = (
            target_ai2thor_property if is_fixed else candidate_required_property
        )
        self.candidate_required_property_value = candidate_required_property_value


# TODO: Add support to parameterize the relations (e.g. distance in CLOSE_TO)
class Relation(ABC):
    """
    A relation between two items in the definition of a task.
    """

    type_id: RelationTypeId
    inverse_relation_type_id: RelationTypeId

    def __init__(self, main_item: TaskItem, related_item: TaskItem) -> None:
        self.main_item = main_item
        self.related_item = related_item
        self.candidate_required_prop: Optional[ObjFixedPropId] = None
        self.candidate_required_prop_value: Optional[Any] = None

    def __str__(self) -> str:
        return f"{self.main_item} is {self.type_id} {self.related_item}"

    def __repr__(self) -> str:
        return f"Relation({self.type_id}, {self.main_item}, {self.related_item})"

    def __hash__(self) -> int:
        return hash((self.type_id, self.main_item.id, self.related_item.id))

    @abstractmethod
    def extract_related_object_ids(
        self, main_obj_metadata: dict[ObjMetadataId, Any]
    ) -> list[ObjId]:
        """
        Return the list of the ids of the main object's related objects according to
        the relation.
        """

    def is_semi_satisfied(self, main_obj_metadata: dict[ObjMetadataId, Any]) -> bool:
        """
        Return True if the relation is semi satisfied.

        A relation is semi satisfied if the main object is correctly
        related to a candidate of the related item (but no related
        object might be assigned to the related item).
        """
        for related_object_id in self.extract_related_object_ids(main_obj_metadata):
            if related_object_id in self.related_item.candidate_ids:
                return True
        return False

    def get_satisfying_related_object_ids(
        self, main_obj_metadata: dict[ObjMetadataId, Any]
    ) -> set[ObjId]:
        """
        Return the set of the ids of the candidates of the related item that satisfy
        the relation with the main object.
        """
        return {
            related_object_id
            for related_object_id in self.extract_related_object_ids(main_obj_metadata)
            if related_object_id in self.related_item.candidate_ids
        }


class ReceptacleOfRelation(Relation):
    """
    A relation of the form "main_item is a receptacle of related_item".

    The inverse relation is ContainedInRelation.
    """

    type_id = RelationTypeId.RECEPTACLE_OF
    inverse_relation_type_id = RelationTypeId.CONTAINED_IN

    def __init__(self, main_item: TaskItem, related_item: TaskItem) -> None:
        super().__init__(main_item, related_item)
        self.candidate_required_prop = ObjFixedPropId.RECEPTACLE
        self.candidate_required_prop_value = True

    def extract_related_object_ids(
        self, main_obj_metadata: dict[ObjMetadataId, Any]
    ) -> list[ObjId]:
        """
        Return the ids of the objects contained in the main object.
        """
        return main_obj_metadata["receptacleObjectIds"]


class ContainedInRelation(Relation):
    """
    A relation of the form "main_item is_contained_in related_item".

    The inverse relation is ReceptacleOfRelation.
    """

    type_id = RelationTypeId.CONTAINED_IN
    inverse_relation_type_id = RelationTypeId.RECEPTACLE_OF

    def __init__(self, main_item: TaskItem, related_item: TaskItem) -> None:
        super().__init__(main_item, related_item)
        self.candidate_required_prop = ObjFixedPropId.PICKUPABLE
        self.candidate_required_prop_value = True

    def extract_related_object_ids(
        self, main_obj_metadata: dict[ObjMetadataId, Any]
    ) -> list[ObjId]:
        """
        Return the ids of the objects containing the main object.
        """
        return main_obj_metadata["parentReceptacles"]


# === Tasks ===
type TaskDict[T: Hashable] = dict[T, dict[Literal["properties", "relations"], dict]]


# TODO: Add support for weighted properties and relations
# TODO: Add support for agent properties
class GraphTask[T: Hashable]:
    """
    Base class for tasks that can be represented as a state graph representing the
    relations between objects in the scene (and the agent).
    For clarity purpose, we call the objects of the task "items", to avoid confusion
    with the real scene's objects.

    The vertices of the graph are the "items", corresponding to objects in the scene
    and the edges are the relations between them (e.g. "receptacle_of" if the item is
    supposed to contain another item).
    The items are represented by the properties required for the task (e.g.
    "objectType", "visible", "isSliced", "temperature",...).

    Note: Inverse relations are automatically added to the graph, so it is not necessary to add them manually when creating the task.

    We use networkx DiGraphs to represent the task graph.

    # TODO: Update docstring

    Attributes:
        TODO: Add attributes
        TODO: Add methods
    """

    def __init__(
        self,
        task_description_dict: TaskDict[T],
    ) -> None:
        """
        Initialize the task graph with the items and their relations as
        defined in the task description dictionary.

        Args:
            task_description_dict (dict[T, dict[Literal["properties", "relations"], dict]]):
                Dictionary describing the items and their properties and relations.
        """
        self.items = full_initialize_items_and_relations_from_dict(
            task_description_dict
        )

        # Initialize the task graph
        self.task_graph = nx.DiGraph()

        # TODO: Check if we create ids and mappings for the items
        for item in self.items:
            self.task_graph.add_node(item)
            for relation in item.relations:
                self.task_graph.add_edge(item, relation.related_item)
                self.task_graph[item][relation.related_item][
                    relation.type_id
                ] = relation

        self.overlap_classes: list[ItemOverlapClass] = []

    def reset(self, event: EventLike) -> None:
        """
        Reset the task with the information of the event.

        Initialize the candidates of the items with the objects
        in the scene and compute the overlap classes.

        Args:
            event (EventLike): Event corresponding to the state of the scene
            at the beginning of the episode.
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
                if not remaining_candidates_ids.isdisjoint(
                    overlap_class["candidate_ids"]
                ):
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
                        overlap_classes[item_class_idx]["items"].extend(
                            overlap_class["items"]
                        )
                        overlap_classes[item_class_idx][
                            "candidate_ids"
                        ] |= overlap_class["candidate_ids"]
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

    # TODO: Improve this
    def __repr__(self) -> str:
        return f"GraphTask({self.task_graph})"

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return self.__repr__()


class PlaceObject(GraphTask):
    """
    Task for placing a target object in a given receptacle.

    This is equivalent to the pick_and_place_simple task from Alfred.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str):
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


# %% == Auxiliary functions ==
# TODO: Check if we keep the relation set too (might not be necessary)
# TODO: Change to only return a plain lsit of items
# TODO: Add support for overriding relations and keep the most restrictive one
def full_initialize_items_and_relations_from_dict[
    T: Hashable
](task_description_dict: TaskDict[T],) -> list[TaskItem[T]]:
    """
    Create the list of TaskItem as defined in the task description
    dictionary representing the items and their properties and relations.
    The items fully initialized with their relations and the inverse
    relations are also added.

    Generic type T (Hashable) is the type of the item identifiers.

    Args:
        task_description_dict (dict[T, dict[Literal["properties", "relations"], dict]]):
            Dictionary describing the items and their properties and relations.

    Returns:
        items (list[TaskItem]): List of the items of the task.
    """
    items = {
        item_id: TaskItem(
            item_id,
            {
                obj_prop_id_to_item_prop[prop]: value
                for prop, value in item_dict["properties"].items()
            },
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
            for related_item_id, relation_type_ids in main_item_dict[
                "relations"
            ].items()
        }
        for main_item_id, main_item_dict in task_description_dict.items()
    }
    # Add inverse relations
    for main_item_id, main_item_relations in organized_relations.items():
        for related_item_id, relations_dict in main_item_relations.items():
            for relation_type_id, relation in relations_dict.items():
                inverse_relation_type_id = relation.inverse_relation_type_id
                if (
                    inverse_relation_type_id
                    not in relations_dict[related_item_id][main_item_id]
                ):
                    relations_dict[related_item_id][main_item_id][
                        inverse_relation_type_id
                    ] = relation_type_id_to_relation[inverse_relation_type_id](
                        items[related_item_id], items[main_item_id]
                    )

    # Set item relations
    for item_id, item in items.items():
        item.relations = {
            relation
            for relations_dict in organized_relations[item_id].values()
            for relation in relations_dict.values()
        }

    items = list(items.values())

    return items


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
    value_type=Literal["water"],  # coffee and wine are not supported yet
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
    value_type=Literal["Hot", "Cold", "RoomTemp"],
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


# %% === Propertiy and relation ids mapping ===
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
