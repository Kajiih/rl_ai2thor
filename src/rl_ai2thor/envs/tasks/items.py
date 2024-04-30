"""
Task items in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType

from rl_ai2thor.envs.sim_objects import (
    SimObjId,
    SimObjMetadata,
)
from rl_ai2thor.envs.tasks.item_prop_interface import (
    ItemVariableProp,
    PropAuxProp,
    RelationAuxProp,
)
from rl_ai2thor.utils.global_exceptions import DuplicateRelationsError

if TYPE_CHECKING:
    from rl_ai2thor.envs.tasks.item_prop_interface import (
        ItemFixedProp,
        ItemProp,
    )
    from rl_ai2thor.envs.tasks.relations import Relation, RelationParam, RelationTypeId


ItemId = NewType("ItemId", str)
CandidateId = NewType("CandidateId", SimObjId)


# TODO? Implement simple CandidateData class for SimpleItems that have no relations and no auxiliary items or properties? ->  Wrong
class CandidateData:
    """
    Class storing the data of an item's candidate.

    Has to be updated at each step to take into account the changes in the scene and compute the
    results and scores of the candidate.

    TODO: Implement weighted properties
    TODO: Implement weighted relations
    TODO: Implement properties and relations order

    Attributes:
        item (SimpleItem): The item of the candidate.
        id (CandidateId): The ID of the candidate.
        metadata (SimObjMetadata): The metadata of the candidate.
        base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item's
            properties to the results of the property satisfaction for the candidate.
        _props_aux_properties_results (dict[ItemVariableProp, dict[PropAuxProp, bool]]):
            Dictionary mapping the item's scored properties to the results of their auxiliary
            properties satisfaction for the candidate.
        # TODO: Add _props_aux_relations_satisfying_related_candidate_ids....
        _props_aux_items_advancement (dict[ItemVariableProp, dict[AuxItem, int]]): Dictionary
            item's scored properties to the advancement of their auxiliary items for the candidate.
        properties_advancement (dict[ItemVariableProp, int]): Dictionary mapping the item's
            properties to the advancement of the candidate for the properties.
        property_advancement (int): The advancement of the candidate for the item's properties, sum
            of the advancements of the properties.
        relations_satisfying_related_candidate_ids (dict[Relation, set[CandidateId]]): Dictionary
            mapping the item's relations to the set of satisfying related item's candidate ids for
            the candidate.
        _relations_aux_properties_results (dict[Relation, dict[RelationAuxProp, bool]]): Dictionary
            mapping the item's relations to the results of the auxiliary properties satisfaction for
            the candidate.
        relations_aux_property_advancement (dict[Relation, int]): Dictionary mapping the item's
            relations to the relation's auxiliary property advancement for the candidate.
        relations_min_max_advancement (dict[Relation, tuple[int, int]]): Dictionary mapping the
            item's relations to the minimum and maximum advancement of the candidate for the
            relations.
        relation_min_advancement (int): The minimum advancement of the candidate for the item's
            relations, sum of the advancements of the relations where all related item's candidate
            are satisfying.
        relation_max_advancement (int): The maximum advancement of the candidate for the item's
            relations, sum of the advancements of the relations where there is at least one
            semi-satisfying related item's candidate (i.e. the set of satisfying objects is not
            empty but they might not be part of the assignment).
        total_min_advancement (int): The minimum advancement of the candidate for the item, sum of
            the advancements of the properties and min advancements of the relations (where we count
            each relation advancement twice to take into account the advancement of the related
            item).
        total_max_advancement (int): The maximum advancement of the candidate for the item, sum of
            the advancements of the properties and max advancements of the relations (where we count
            each relation advancement twice to take into account the advancement of the related
            item).
    """

    def __init__(self, c_id: CandidateId, item: TaskItem) -> None:
        """
        Initialize the candidate's id and item.

        Args:
            c_id (CandidateId): The ID of the candidate.
            item (TaskItem): The item of the candidate.
        """
        self.item = item
        self.id = c_id

        # === Type annotations ===
        # === Properties ===
        self.item: TaskItem
        self.id: CandidateId
        self.metadata: SimObjMetadata
        self.base_properties_results: dict[ItemVariableProp, bool]
        self.props_aux_properties_results = dict[ItemVariableProp, dict[PropAuxProp, bool]]
        self._props_aux_relations_satisfying_related_candidate_ids: dict[
            ItemVariableProp, dict[Relation, set[CandidateId]]
        ]
        self.props_aux_relations_advancement: dict[ItemVariableProp, dict[Relation, int]]
        self.props_aux_items_advancement: dict[ItemVariableProp, dict[AuxItem, int]]
        self.properties_advancement: dict[ItemVariableProp, int]
        self.property_advancement: int
        # === Relations ===
        self.relations_satisfying_related_candidate_ids: dict[Relation, set[CandidateId]]
        self.relations_aux_properties_results: dict[Relation, dict[RelationAuxProp, bool]]
        self.relations_aux_property_advancement: dict[Relation, int]
        self.relations_min_max_advancement: dict[Relation, tuple[int, int]]
        self.relation_min_advancement: int
        self.relation_max_advancement: int
        # === Final advancement ===
        self.total_min_advancement: int
        self.total_max_advancement: int

    @property
    def _scored_properties(self) -> frozenset[ItemVariableProp]:
        """
        Scored properties of the candidate's item.

        Returns:
            frozenset[ItemVariableProp]: The scored properties of the item.
        """
        return self.item.scored_properties

    @property
    def _relations(self) -> frozenset[Relation]:
        """
        Relations of the candidate's item.

        Returns:
            frozenset[Relation]: The relations of the item.
        """
        return self.item.relations

    # TODO: Double check all this
    def update(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> None:
        """
        Update the candidate data with the given scene object dictionary.

        The auxiliary items' candidates and max score have to be updated before the candidate data.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.
        """
        self.metadata = scene_objects_dict[self.id]

        # === Properties ===
        self.base_properties_results = self._compute_base_properties_results()
        self.props_aux_properties_results = self._compute_props_aux_properties_results(self.base_properties_results)
        # self._props_aux_properties_advancement = self._compute_props_aux_properties_advancement(
        #     self._props_aux_properties_results
        # ) # TODO: Delete
        self.props_aux_properties_advancement = {
            prop: {
                aux_prop: aux_prop.maximum_advancement if self.base_properties_results[prop] else 0
                for aux_prop in prop.auxiliary_properties
            }
            for prop in self._scored_properties
        }
        self.props_aux_relations_advancement = self._compute_props_aux_relations_advancement(
            self.base_properties_results,
            scene_objects_dict,
        )
        self.props_aux_items_advancement = self._compute_props_aux_items_advancement(self.base_properties_results)

        self.properties_advancement = self._compute_properties_advancement(
            self.base_properties_results,
            self.props_aux_properties_results,
            self.props_aux_relations_advancement,
            self.props_aux_items_advancement,
        )
        self.property_advancement = sum(self.properties_advancement.values())

        # === Relations ===
        # Note: We can compute a more optimal lower bound on the relation's advancement (min_advancement) by taking into account the fact that several relations might involve the same related item and then computing the relations advancement for each assignment of the related item's candidate
        # TODO? Implement this?
        self.relations_satisfying_related_candidate_ids = self._compute_relations_satisfying_related_candidate_ids(
            self._relations,
            scene_objects_dict,
        )
        self.relations_aux_properties_results = self._compute_relations_aux_properties_results(
            self.relations_satisfying_related_candidate_ids
        )
        self.relations_aux_property_advancement = self._compute_relations_aux_property_advancement(
            self.relations_aux_properties_results
        )

        self.relations_min_max_advancement = self._compute_relations_min_max_advancement(
            self.relations_satisfying_related_candidate_ids, self.relations_aux_property_advancement
        )
        self.relation_min_advancement = sum(
            min_adv for (min_adv, _max_adv) in self.relations_min_max_advancement.values()
        )
        self.relation_max_advancement = sum(
            max_adv for (_min_adv, max_adv) in self.relations_min_max_advancement.values()
        )

        # === Final advancement ===
        # TODO: Update this: For each relation that is considered satisfied, we need to add the advancement of the inverse relation too. It's not necessarily x2 for the relations because the inverse relation might not have the same max advancement (we need to implement this)
        self.total_min_advancement = self.property_advancement + 1 * self.relation_min_advancement
        self.total_max_advancement = self.property_advancement + 1 * self.relation_max_advancement

    def _compute_base_properties_results(self) -> dict[ItemVariableProp, bool]:
        """
        Return the results dictionary of each properties for the candidate.

        Returns:
            base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item properties
            to the results of the property satisfaction for the candidate.
        """
        return {prop: prop.is_object_satisfying(self.metadata) for prop in self._scored_properties}

    def _compute_props_aux_properties_results(
        self,
        base_properties_results: dict[ItemVariableProp, bool],
    ) -> dict[ItemVariableProp, dict[PropAuxProp, bool]]:
        """
        Return the results dictionary of each auxiliary properties for the candidate.

        The results are computed only for auxiliary properties of the property that are not
        already satisfied by the candidate.

        Args:
        base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item
            properties to the results of the property satisfaction for the candidate.


        Returns:
            aux_properties_results (dict[ItemVariableProp, dict[PropAuxProp, bool]]): Dictionary
                mapping the item's properties to the results of the auxiliary properties
                satisfaction for the candidate.
        """
        return {
            prop: {aux_prop: aux_prop.is_object_satisfying(self.metadata) for aux_prop in prop.auxiliary_properties}
            for prop in self._scored_properties
            if not base_properties_results[prop]
        }

    def _compute_relations_satisfying_related_candidate_ids(
        self,
        relations: frozenset[Relation],
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[Relation, set[CandidateId]]:
        """
        Return the satisfying related candidates ids for each relation of the item.

        Args:
            relations (frozenset[Relation]): The relations for which to compute the satisfying
                related candidates ids.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            relations_satisfying_related_candidate_ids (dict[Relation, set[CandidateId]]):
                Dictionary mapping the item's relations to the set of satisfying related item's
                candidate ids for the candidate.
        """
        return {
            relation: relation.compute_satisfying_related_candidate_ids(self.metadata, scene_objects_dict)
            for relation in relations
        }

    # TODO: Implement weighted relations
    def compute_relation_advancement_for_related_candidate(
        self,
        relation: Relation,
        related_candidate_id: CandidateId,
    ) -> int:
        """
        Compute the advancement of the candidate for the relation when the related item's candidate is the given related candidate.

        Args:
            relation (Relation): The relation for which to compute the advancement.
            related_candidate_id (CandidateId): The id of the related item's candidate.

        Returns:
            relation_advancement (int): The advancement of the candidate for the relation when the
                related item's candidate is the given candidate.
        """
        return (
            relation.maximum_advancement
            if related_candidate_id in self.relations_satisfying_related_candidate_ids[relation]
            else self.relations_aux_property_advancement[relation]
        )

    def _compute_props_aux_relations_advancement(
        self,
        base_properties_results: dict[ItemVariableProp, bool],
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[ItemVariableProp, dict[Relation, int]]:
        """
        Return the advancement of the candidate for each auxiliary relation of the item's properties.

        Args:
            base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item
                properties to the results of the property satisfaction for the candidate.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            props_aux_relations_advancement (dict[ItemVariableProp, dict[Relation, int]]): Dictionary
                mapping the item's properties to the dictionary mapping the item's relations to the
                advancement of the candidate for the relations.
        """
        props_aux_relations_satisfying_related_candidate_ids: dict[ItemVariableProp, dict[Relation, set[CandidateId]]]
        props_aux_relations_satisfying_related_candidate_ids = {
            prop: self._compute_relations_satisfying_related_candidate_ids(
                prop.auxiliary_relations,
                scene_objects_dict,
            )
            for prop in self._scored_properties
            if not base_properties_results[prop]
        }
        # We keep only properties that are not satisfied by the candidate

        props_aux_relations_aux_properties_results: dict[ItemVariableProp, dict[Relation, dict[RelationAuxProp, bool]]]
        props_aux_relations_aux_properties_results = {
            prop: self._compute_relations_aux_properties_results(aux_relations_satisfying_related_candidate_ids)
            for prop, aux_relations_satisfying_related_candidate_ids in props_aux_relations_satisfying_related_candidate_ids.items()
        }
        # Relations that are satisfied by all related items' candidates are dropped

        props_aux_relations_aux_property_advancement: dict[ItemVariableProp, dict[Relation, int]]
        props_aux_relations_aux_property_advancement = {
            prop: self._compute_relations_aux_property_advancement(props_aux_relations_aux_properties_results[prop])
            for prop in props_aux_relations_aux_properties_results
        }

        props_aux_relations_advancement: dict[ItemVariableProp, dict[Relation, int]]
        props_aux_relations_advancement = {
            prop: {
                relation: relation.maximum_advancement
                if satisfying_related_candidate_ids
                else props_aux_relations_aux_property_advancement[prop][relation]
                for relation, satisfying_related_candidate_ids in aux_relations_satisfying_related_candidate_ids.items()
            }
            for prop, aux_relations_satisfying_related_candidate_ids in props_aux_relations_satisfying_related_candidate_ids.items()
        }

        return props_aux_relations_advancement

    # TODO: Double and triple check this method
    # TODO? Create methods to separate the computation of the relation's auxiliary properties results and the relation's advancement?
    def _compute_props_aux_items_advancement(
        self,
        base_properties_results: dict[ItemVariableProp, bool],
    ) -> dict[ItemVariableProp, dict[AuxItem, int]]:
        """
        Return the advancement of the auxiliary items for the candidate for each property.

        Args:
            base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item
                properties to the results of the property satisfaction for the candidate.

        Returns:
            props_aux_items_advancement (dict[ItemVariableProp, dict[AuxItem, int]]): Dictionary
                mapping the item's properties to the advancement of the auxiliary items for the
                candidate.
        """
        # Extract the advancement of every candidate of the auxiliary items when the relation's
        # related items' candidate is this candidate
        aux_items_advancement: dict[ItemVariableProp, dict[AuxItem, dict[CandidateId, int]]]
        aux_items_advancement = {
            prop: {aux_item: {} for aux_item in prop.auxiliary_items}
            for prop in self._scored_properties
            if not base_properties_results[prop]
        }
        for prop in aux_items_advancement:
            for aux_item in prop.auxiliary_items:
                # TODO? Get this data from the inverse property's results instead? -> Not possible since the relation's results are not computed
                for aux_candidate_id, aux_candidate_data in aux_item.candidates_data.items():
                    # Add properties advancement
                    aux_advancement = aux_candidate_data.property_advancement
                    # Add relation advancement
                    for relation in aux_item.relations:
                        aux_advancement += aux_candidate_data.compute_relation_advancement_for_related_candidate(
                            relation, self.id
                        )
                    aux_items_advancement[prop][aux_item][aux_candidate_id] = aux_advancement

        # Keep only the maximum advancement for each auxiliary item
        return {
            prop: {
                aux_item: max(
                    aux_items_advancement[prop][aux_item][candidate_id]
                    for candidate_id in aux_items_advancement[prop][aux_item]
                )
                for aux_item in prop.auxiliary_items
            }
            for prop in aux_items_advancement
        }

    # TODO? Add support for auxiliary properties having auxiliary properties with a recursive way of computing the advancement
    # TODO: Implement weighted properties
    @staticmethod
    def _compute_properties_advancement(
        base_properties_results: dict[ItemVariableProp, bool],
        props_aux_properties_results: dict[ItemVariableProp, dict[PropAuxProp, bool]],
        props_aux_relations_advancement: dict[ItemVariableProp, dict[Relation, int]],
        props_aux_items_advancement: dict[ItemVariableProp, dict[AuxItem, int]],
    ) -> dict[ItemVariableProp, int]:
        """
        Return the scores of the candidate for the item's properties.

        If the property is satisfied by the candidate, the score is the maximum advancement of the
        property, otherwise it is the sum of the maximum advancement of the auxiliary properties and
        the advancement of the auxiliary items.

        Args:
            base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item
                properties to the results of the property satisfaction for the candidate.
            props_aux_properties_results (dict[ItemVariableProp, dict[ItemVariableProp, bool]]):
                Dictionary mapping the item's properties to the results of the auxiliary properties
                satisfaction for the candidate.
            props_aux_relations_advancement (dict[ItemVariableProp, dict[Relation, int]]):
                Dictionary mapping the item's properties to the advancement of the candidate for the
                auxiliary relations of the properties.
            props_aux_items_advancement (dict[ItemVariableProp, dict[AuxItem, int]]): Dictionary
                mapping the item's properties to the advancement of the auxiliary items for the
                candidate.

        Returns:
            properties_advancement (dict[ItemVariableProp, int]): Dictionary mapping the item
                properties to the scores of the candidate for the properties.
        """
        # TODO? Replace aux_prop.maximum_advancement by 1
        aux_prop_advancement = {
            prop: sum(
                aux_prop.maximum_advancement
                for aux_prop, result in props_aux_properties_results[prop].items()
                if result
            )
            for prop in props_aux_properties_results
        }
        aux_relations_advancement = {
            prop: sum(advancement for advancement in aux_relations_advancement.values())
            for prop, aux_relations_advancement in props_aux_relations_advancement.items()
        }

        aux_item_advancement = {
            prop: sum(advancement for advancement in aux_items_advancement.values())
            for prop, aux_items_advancement in props_aux_items_advancement.items()
        }

        return {
            prop: prop.maximum_advancement
            if base_properties_results[prop]
            else aux_prop_advancement[prop] + aux_relations_advancement[prop] + aux_item_advancement[prop]
            for prop in base_properties_results
        }

    def _compute_relations_aux_properties_results(
        self,
        relations_satisfying_related_candidate_ids: dict[Relation, set[CandidateId]],
    ) -> dict[Relation, dict[RelationAuxProp, bool]]:
        """
        Return the results dictionary of each relation's auxiliary properties for the candidate.

        The results are computed only for auxiliary properties of the relation that are not
        satisfied by every related item's candidate (we don't need to compute this information
        in that case since the relation will be satisfied whatever the assignment).

        Args:
            relations_satisfying_related_candidate_ids (dict[Relation, set[CandidateId]]):
                Dictionary mapping the item's relations to the set of satisfying related item's
                candidate ids for the candidate.

        Returns:
            relations_aux_properties_results (dict[Relation, dict[RelationAuxProp, bool]]): Dictionary
                mapping the item's relations to the results of the auxiliary properties satisfaction
                for the candidate.
        """
        return {
            relation: {
                aux_prop: aux_prop.is_object_satisfying(self.metadata) for aux_prop in relation.auxiliary_properties
            }
            for relation, satisfying_related_candidate_ids in relations_satisfying_related_candidate_ids.items()
            if len(satisfying_related_candidate_ids) != len(relation.related_item.candidate_ids)
        }

    @staticmethod
    def _compute_relations_aux_property_advancement(
        relations_aux_properties_results: dict[Relation, dict[RelationAuxProp, bool]],
    ) -> dict[Relation, int]:
        """
        Return the relation's auxiliary property advancement for the candidate.

        Args:
            relations_aux_properties_results (dict[Relation, dict[RelationAuxProp, bool]]):
                Dictionary mapping the item's relations to the results of the auxiliary properties
                satisfaction for the candidate.

        Returns:
            relations_aux_property_advancement (dict[Relation, int]): Dictionary mapping the item's
            relations to the relation's auxiliary property advancement for the candidate.
        """
        # TODO? Replace aux_prop.maximum_advancement by 1
        return {
            relation: sum(aux_prop.maximum_advancement for aux_prop, result in aux_properties_results.items() if result)
            for relation, aux_properties_results in relations_aux_properties_results.items()
        }

    # TODO: Implement weighted relations
    @staticmethod
    def _compute_relations_min_max_advancement(
        relations_satisfying_related_candidate_ids: dict[Relation, set[CandidateId]],
        relations_aux_property_advancement: dict[Relation, int],
    ) -> dict[Relation, tuple[int, int]]:
        """
        Return the minimum and the maximum advancement of the candidate for each item's relations.

        This advancement takes into account the auxiliary properties of the relations when
        necessary, ie when the relation is not satisfied by every related item's candidate.

        Minimum and maximum advancement meaning:
        - The minimum advancement of the candidate for the item's relations is the sum of the
        advancements of the relations where all related item's candidate are satisfying (so the
        relation will be satisfied whatever the assignment).
        - The maximum advancement of the candidate for the item's relations is the sum of the
        advancements of the relations where there is at least one semi-satisfying related item's
        candidate (i.e. the set of satisfying objects is not empty but they might not be part of the
        optimal global assignment).

        Args:
            relations_satisfying_related_candidate_ids (dict[Relation, set[CandidateId]]):
                Dictionary mapping the item's relations to the set of satisfying related item's
                candidate ids for the candidate.
            relations_aux_property_advancement (dict[Relation, int]): Dictionary mapping the item's
                relations to the relation's auxiliary property advancement for the candidate.

        Returns:
            relations_min_max_scores (dict[Relation, tuple[int, int]]): Dictionary mapping the
                item's relations to the minimum and maximum scores of the candidate for the
                relations in this order.
        """
        return {
            relation: (
                relation.maximum_advancement
                if len(relations_satisfying_related_candidate_ids[relation]) == len(relation.related_item.candidate_ids)
                else relations_aux_property_advancement[relation],
                relation.maximum_advancement
                if relations_satisfying_related_candidate_ids[relation]
                else relations_aux_property_advancement[relation],
            )
            for relation in relations_satisfying_related_candidate_ids
        }

    # TODO: Update to make printable for the logs?
    def make_info_dict(self) -> dict[str, Any]:
        """
        Return the information of the candidate as a dictionary.

        Returns:
            info_dict (dict[str, Any]): Dictionary containing the information of the candidate.
        """
        info_dict = {
            "id": self.id,
            "item_id": self.item.id,
            "base_properties_results": self.base_properties_results,
            "props_aux_properties_results": self.props_aux_properties_results,
            "props_aux_items_data": {
                prop: {aux_item: aux_item.candidates_data[self.id].make_info_dict() for aux_item in aux_items}
                for prop, aux_items in self.item.props_auxiliary_items.items()
            },
            "props_aux_items_advancement": self.props_aux_items_advancement,
            "properties_advancement": self.properties_advancement,
            "property_advancement": self.property_advancement,
            "relations_satisfying_related_candidate_ids": self.relations_satisfying_related_candidate_ids,
            "relations_aux_properties_results": self.relations_aux_properties_results,
            "relations_aux_property_advancement": self.relations_aux_property_advancement,
            "relations_min_max_advancement": self.relations_min_max_advancement,
            "relation_min_advancement": self.relation_min_advancement,
            "relation_max_advancement": self.relation_max_advancement,
            "total_min_advancement": self.total_min_advancement,
            "total_max_advancement": self.total_max_advancement,
        }
        return info_dict

    def __str__(self) -> str:
        return f"CandidateData({self.id})"

    def __repr__(self) -> str:
        return f"CandidateData({self.id}\n{self.make_info_dict()})"  # TODO: Check if we keep like this


@dataclass
class AdvancementDetails:
    """Class storing the advancement details of an item, property or relation."""

    current_advancement: int
    total_max_advancement: int

    def __str__(self) -> str:
        return f"{self.current_advancement}/{self.total_max_advancement}"

    def __repr__(self) -> str:
        return f"AdvancementDetails({self.current_advancement}/{self.total_max_advancement})"


class PropAdvancementDetails(AdvancementDetails):
    """Class storing the advancement details of a property."""

    def __init__(self, prop: ItemVariableProp, assigned_candidate_data: CandidateData) -> None:
        self.prop = prop
        self.base_result = assigned_candidate_data.base_properties_results[prop]

        # === Initialize auxiliary properties advancement details ===
        # TODO? Replace aux_prop.maximum_advancement by 1
        self.aux_props_advancement_details = {
            aux_prop: AdvancementDetails(
                current_advancement=aux_prop.maximum_advancement
                if self.base_result
                else int(assigned_candidate_data.props_aux_properties_results[prop][aux_prop]),
                total_max_advancement=aux_prop.maximum_advancement,
            )  # TODO: Check why pylance is complaining
            for aux_prop in prop.auxiliary_properties
        }

        # === Initialize auxiliary relations advancement details ===
        self.aux_relations_advancement_details = {
            relation: AdvancementDetails(
                current_advancement=relation.maximum_advancement
                if self.base_result
                else assigned_candidate_data.props_aux_relations_advancement[prop][relation],
                total_max_advancement=relation.maximum_advancement,
            )
            for relation in prop.auxiliary_relations
        }

        # === Initialize auxiliary items advancement details ===
        self.aux_items_advancement_details = {
            aux_item: AdvancementDetails(
                current_advancement=aux_item.maximum_advancement
                if self.base_result
                else assigned_candidate_data.props_aux_items_advancement[prop][aux_item],
                total_max_advancement=aux_item.maximum_advancement,
            )
            for aux_item in prop.auxiliary_items
        }

        # === Compute current advancement and total max advancement ===
        self.current_advancement = (
            prop.maximum_advancement
            if self.base_result
            else (
                sum(
                    aux_prop_advancement.current_advancement
                    for aux_prop_advancement in self.aux_props_advancement_details.values()
                )
                + sum(
                    rel_advancement.current_advancement
                    for rel_advancement in self.aux_relations_advancement_details.values()
                )
                + sum(
                    item_advancement.current_advancement
                    for item_advancement in self.aux_items_advancement_details.values()
                )
            )
        )
        # Temp: Delete later
        assert self.current_advancement == assigned_candidate_data.properties_advancement[prop]

        self.total_max_advancement = prop.maximum_advancement
        # Temp: Delete later
        assert self.total_max_advancement == 1 + sum(
            aux_prop.maximum_advancement for aux_prop in prop.auxiliary_properties
        ) + sum(rel.maximum_advancement for rel in prop.auxiliary_relations) + sum(
            item.maximum_advancement for item in prop.auxiliary_items
        )

        # === Type annotations ===
        self.prop: ItemVariableProp
        self.base_result: bool
        self.aux_props_advancement_details: dict[PropAuxProp, AdvancementDetails]
        self.aux_relations_advancement_details: dict[Relation, AdvancementDetails]
        self.aux_items_advancement_details: dict[AuxItem, AdvancementDetails]


# TODO: Create specific advancement details for properties and relations with auxiliary items and properties
class ItemAdvancementDetails(AdvancementDetails):
    """Class storing the advancement details of an item."""

    properties_advancement_details: dict[ItemVariableProp, AdvancementDetails]
    relations_advancement_details: dict[Relation, AdvancementDetails]
    current_advancement: int = field(init=False)
    total_max_advancement: int = field(init=False)

    def __init__(self, item: TaskItem, global_assignment: Assignment) -> None:
        self.item = item
        self.assigned_candidate_id = global_assignment[item]
        assigned_candidate_data = self.item.candidates_data[self.assigned_candidate_id]

        # === Initialize properties advancement details ===
        self.properties_advancement_details = {
            prop: PropAdvancementDetails(
                prop=prop,
                assigned_candidate_data=assigned_candidate_data,
            )
            for prop in self.item.scored_properties
        }

        # === Initialize relations advancement details ===
        self.relations_advancement_details = {
            relation: AdvancementDetails(
                current_advancement=assigned_candidate_data.compute_relation_advancement_for_related_candidate(
                    relation, global_assignment[relation.related_item]
                ),
                total_max_advancement=relation.maximum_advancement,
            )
            for relation in self.item.relations
        }

        # === Compute current advancement and total max advancement ===
        self.current_advancement = sum(
            prop_advancement.current_advancement for prop_advancement in self.properties_advancement_details.values()
        ) + sum(rel_advancement.current_advancement for rel_advancement in self.relations_advancement_details.values())
        self.total_max_advancement = self.item.maximum_advancement
        # Temp: Delete later
        assert self.total_max_advancement == sum(
            prop.maximum_advancement for prop in self.item.scored_properties
        ) + sum(rel.maximum_advancement for rel in self.item.relations)

        # === Type annotations ===
        self.item: TaskItem
        self.assigned_candidate_id: CandidateId

    def __str__(self) -> str:
        return f"{self.item.id} ({self.assigned_candidate_id}): {self.current_advancement}/{self.total_max_advancement}"

    def __repr__(self) -> str:
        return f"ItemAdvancementDetails({self.item.id} ({self.assigned_candidate_id}): {self.current_advancement}/{self.total_max_advancement})"


# %% === Items ===
class SimpleItem(ABC):
    """
    An item having properties without auxiliary items or properties and no relations.

    Attributes:
        id (ItemId): The ID of the item as defined in the task description.
        properties (frozenset[ItemProp]): Set of properties of the item (fixed and variable properties).
        relations (frozenset[Relation]): Set of relations of the item.
        candidate_required_properties (frozenset[ItemFixedProp]): Set of properties required for an object
            to be a candidate for the item.
        scored_properties (frozenset[ItemVariableProp]): Set of variable properties of the item which are
            used to compute the scores of the candidates.
        maximum_advancement (int): Maximum advancement of the item in the task: sum of the maximum
            advancement of the scored properties and relations.
        candidate_ids (set[SimObjId]): Set of candidate ids of the item.
        candidates_data (dict[CandidateId, CandidateData]): Dictionary mapping the candidate ids to
            their data.
        step_max_property_advancement (int): Maximum advancement of the item in the task at the
            current step when counting only properties, i.e. the maximum property advancement of the
            candidates. It is the "maximum" advancement of the properties, because it doesn't take
            the advancement of the other items into account and this might not be the the same
            advancement as in the optimal global task advancement.
        step_max_advancement (int): Maximum advancement of the item in the task at the current step,
            i.e. the maximum advancement of the candidates. It the "maximum" advancement, because it doesn't take the advancement of the other items into account and this might not be the the same advancement as in the
            optimal global task advancement.
    """

    def __init__(
        self,
        t_id: ItemId | str,
        properties: set[ItemProp] | frozenset[ItemProp],
    ) -> None:
        """
        Initialize the attribute of the item.

        Args:
            t_id (ItemId): The ID of the item as defined in the task description.
            properties (set[ItemProp]): Set of properties of the item.
        """
        self.id = ItemId(t_id)
        self.properties = frozenset(properties)
        self.scored_properties = frozenset(prop for prop in self.properties if isinstance(prop, ItemVariableProp))

        # Infer the candidate required properties from the item properties
        self.candidate_required_properties = frozenset(
            prop.candidate_required_prop for prop in self.properties if prop.candidate_required_prop is not None
        )

        # === Type annotations ===
        self.id: ItemId
        self.properties: frozenset[ItemProp]
        self.scored_properties: frozenset[ItemVariableProp]
        self.maximum_advancement: int
        self.candidate_required_properties: frozenset[ItemFixedProp]
        self.candidates_data: dict[CandidateId, CandidateData]
        self.step_max_property_advancement: int
        self.overlap_class: ItemOverlapClass

    @property
    def candidate_ids(self) -> set[CandidateId]:
        """
        Get the candidate ids of the item.

        Returns:
            set[CandidateId]: Set of candidate ids of the item.
        """
        return set(self.candidates_data.keys())

    @property
    def step_max_advancement(self) -> int:
        """
        Get the maximum advancement of the item in the task at the current step.

        Returns:
            int: Maximum advancement of the item in the task at the current step.
        """
        return self.step_max_property_advancement

    def _init_maximum_advancement(self) -> None:
        """
        Recursively initialize the maximum advancement of the scored properties and then the maximum advancement of the item.

        This has to be called after everything has been fully initialized.

        The maximum advancement is the sum of the maximum advancement of the scored properties.
        """
        for prop in self.scored_properties:
            prop.init_maximum_advancement()
        self.maximum_advancement = sum(prop.maximum_advancement for prop in self.scored_properties)

    def is_candidate(self, obj_metadata: SimObjMetadata) -> bool:
        """
        Return True if the given object is a valid candidate for the item.

        Args:
            obj_metadata (SimObjMetadata): Object metadata.

        Returns:
            is_candidate (bool): True if the given object is a valid candidate for the item.
        """
        return all(prop.is_object_satisfying(obj_metadata) for prop in self.candidate_required_properties)

    def _get_candidate_ids(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> set[CandidateId]:
        """
        Return the set of candidate ids of the item.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            candidate_ids (set[CandidateId]): Set of candidate ids of the item.
        """
        return {CandidateId(obj_id) for obj_id in scene_objects_dict if self.is_candidate(scene_objects_dict[obj_id])}

    @abstractmethod
    def instantiate_candidate_data(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> dict[CandidateId, CandidateData]:
        """
        Instantiate the candidate data for the item.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            candidates_data (dict[CandidateId, CandidateData]): Dictionary mapping the candidate ids to
                their data.
        """

    def update_candidates_data(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> None:
        """
        Update the data of the candidates of the item with the given scene object dictionary.

        Also update the step_max_advancement attribute.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.
        """
        for candidate_data in self.candidates_data.values():
            candidate_data.update(scene_objects_dict)

        self.step_max_property_advancement = max(
            candidate_data.property_advancement for candidate_data in self.candidates_data.values()
        )


# TODO? Add support for giving some score for semi satisfied relations and using this info in the selection of interesting objects/assignments
# TODO: Make so that the class need the relations to be instantiated like the properties instead of having to set them after
class TaskItem(SimpleItem):
    """
    An item in the definition of a task.

    The task items have properties, some of which can have auxiliary items and properties, relations
    with other task items, and they represent a unique object in the scene, so they are part of
    overlap classes to avoid assigning the same object to multiple items.

    Attributes:
        id (ItemId): The ID of the item as defined in the task description.
        properties (frozenset[ItemProp]): Set of properties of the item (fixed and variable properties).
        relations (frozenset[Relation]): Set of relations of the item.
        organized_relations (dict[ItemId, dict[RelationTypeId, Relation]]): Relations of the item
            organized by related item id and relation type id.
        candidate_required_properties (frozenset[ItemFixedProp]): Set of properties required for an object
            to be a candidate for the item.
        scored_properties (frozenset[ItemVariableProp]): Set of variable properties of the item which are
            used to compute the scores of the candidates.
        max_advancement (int): Maximum advancement of the item in the task: sum of the maximum
            advancement of the scored properties and relations.
        props_auxiliary_items (dict[ItemVariableProp, frozenset[TaskItem]]): Map of the item's properties to
            their auxiliary items.
        props_auxiliary_relations (dict[ItemVariableProp, frozenset[Relation]]): Map of the item's
            properties to their auxiliary relations (inverse relations of the auxiliary items'
            relations).
        props_auxiliary_properties (dict[ItemVariableProp, frozenset[PropAuxProp]]): Map of the item's
            properties to their auxiliary properties.
        relations_auxiliary_properties (dict[Relation, frozenset[RelationAuxProp]]): Map of the
            item's relations to their auxiliary properties.
        candidate_ids (set[SimObjId]): Set of candidate ids of the item.
        candidates_data (dict[CandidateId, CandidateData]): Dictionary mapping the candidate ids to
            their data.
        step_max_property_advancement (int): Maximum advancement of the item in the task at the
            current step when counting only properties, i.e. the maximum property advancement of the
            candidates. It is the "maximum" advancement of the properties, because it doesn't take
            the advancement of the other items into account and this might not be the the same
            advancement as in the optimal global task advancement.
        step_max_relation_advancement (int): Maximum advancement of the item in the task at the
            current step when counting only relations, i.e. the maximum relation advancement of the
            candidates. It is the "maximum" advancement of the relations, because it doesn't take
            the advancement of the other items into account and this might not be the the same
            advancement as in the optimal global task advancement.
        step_max_advancement (int): Maximum advancement of the item in the task at the current step,
            i.e. the maximum advancement of the candidates. It the "maximum" advancement, because it doesn't take the advancement of the other items into account and this might not be the the same advancement as in the
            optimal global task advancement.
    """

    def __init__(
        self,
        t_id: ItemId | str,
        properties: set[ItemProp] | frozenset[ItemProp],
        relations: set[Relation] | None = None,
    ) -> None:
        """
        Initialize the task item.

        The main item of the relations and the related item of their inverse relations are set to
        `self`.

        Args:
            t_id (ItemId): The ID of the item as defined in the task description.
            properties (set[ItemProp]): Set of properties of the item.
            relations (set[Relation], optional): Set of relations of the item.
        """
        super().__init__(t_id, properties)
        #  === Set auxiliary items and their relations with the item ===
        self.props_auxiliary_items = {prop: prop.auxiliary_items for prop in self.scored_properties}
        # Initialize the relations of the auxiliary items and and the auxiliary relations
        for prop, aux_items in self.props_auxiliary_items.items():
            for aux_item in aux_items:
                aux_item.initialize_relations(self.id)
            # Set the properties' auxiliary relations
            prop.auxiliary_relations = frozenset(
                relation.inverse_relation for aux_item in aux_items for relation in aux_item.relations
            )
        self.props_auxiliary_relations = {
            prop: frozenset(relation.inverse_relation for aux_item in aux_items for relation in aux_item.relations)
            for prop, aux_items in self.props_auxiliary_items.items()
        }

        # === Set the relations ===
        if relations is not None:
            self.relations = frozenset(relations)
        else:
            self.relations = frozenset()
        self._check_duplicate_relations(self.relations)
        # Set the main item and related item of the relations and their inverse relations
        for relation in self.relations:
            relation.main_item = self
            relation.inverse_relation.related_item = self

        relation_required_properties = frozenset(
            relation.candidate_required_prop
            for relation in self.relations
            if relation.candidate_required_prop is not None
        )
        self.candidate_required_properties |= relation_required_properties

        # === Auxiliary properties of properties and relations ===
        self.props_auxiliary_properties = {prop: prop.auxiliary_properties for prop in self.scored_properties}
        self.relations_auxiliary_properties = {relation: relation.auxiliary_properties for relation in self.relations}

        # === Initialize the maximum advancement of the item and its properties and relations ===
        self._init_maximum_advancement()

        # === Type annotations ===
        self.relations: frozenset[Relation]
        self.props_auxiliary_items: dict[ItemVariableProp, frozenset[AuxItem]]
        self.props_auxiliary_relations: dict[ItemVariableProp, frozenset[Relation]]
        self.props_auxiliary_properties: dict[ItemVariableProp, frozenset[PropAuxProp]]
        self.relations_auxiliary_properties: dict[Relation, frozenset[RelationAuxProp]]
        self.step_max_relation_advancement: int

    # TODO: Replace to hold a set of relations instead of dict of relation id -> relation
    @property
    def organized_relations(self) -> dict[ItemId, dict[RelationTypeId, Relation]]:
        """
        Get the organized relations of the item.

        Returns:
            dict[ItemId, dict[RelationTypeId, Relation]]: Dictionary containing the relations of the
            main item organized by related item id and relation type id.
        """
        return {
            relation.related_item.id: {
                relation.type_id: relation,
            }
            for relation in self.relations
        }

    @property
    def step_max_advancement(self) -> int:
        """
        Get the maximum advancement of the item in the task at the current step.

        Returns:
            int: Maximum advancement of the item in the task at the current step.
        """
        return self.step_max_property_advancement + self.step_max_relation_advancement

    def _init_maximum_advancement(self) -> None:
        """
        Recursively initialize the maximum advancement of the scored properties and relations and then the one of the item.

        This has to be called after everything has been fully initialized.

        The maximum advancement is the sum of the maximum advancement of the scored properties and
        relations.
        """
        super()._init_maximum_advancement()
        for relation in self.relations:
            relation.init_maximum_advancement()
        self.maximum_advancement += sum(relation.maximum_advancement for relation in self.relations)

    @staticmethod
    def _check_duplicate_relations(relations: frozenset[Relation]) -> None:
        """
        Check that there are no duplicate relations of the same type and main and related items.

        Args:
            relations (frozenset[Relation]): Set of relations to check.
        """
        existing_relations = {}
        for relation in relations:
            if relation.related_item_id not in existing_relations:
                existing_relations[relation.related_item_id] = {}
            elif relation.type_id in existing_relations[relation.related_item_id]:
                raise DuplicateRelationsError(relation.type_id, relation.main_item_id, relation.related_item_id)
            existing_relations[relation.related_item_id][relation.type_id] = relation

    def instantiate_candidate_data(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> dict[CandidateId, CandidateData]:
        """
        Instantiate the candidate data for the item.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            candidates_data (dict[CandidateId, CandidateData]): Dictionary mapping the candidate ids to
                their data.
        """
        candidate_ids = self._get_candidate_ids(scene_objects_dict)
        return {c_id: CandidateData(c_id, self) for c_id in candidate_ids}

    def update_candidates_data(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> None:
        """
        Update the data of the candidates of the item with the given scene object dictionary.

        Also update the auxiliary items' candidates data.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.
        """
        for auxiliary_item_set in self.props_auxiliary_items.values():
            for auxiliary_item in auxiliary_item_set:
                auxiliary_item.update_candidates_data(scene_objects_dict)
        super().update_candidates_data(scene_objects_dict)
        self.step_max_relation_advancement = max(
            candidate_data.relation_max_advancement for candidate_data in self.candidates_data.values()
        )

    # TODO: Double check the interesting candidates computation
    def compute_interesting_candidates(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> set[CandidateId]:
        """
        Return the set of interesting candidates for the item.

        The interesting candidates are those  that have to be considered for the item in the global assignment because they can lead to a maximum of task advancement depending on the
        assignment of objects to the other items.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            interesting_candidates (set[CandidateId]): Set of interesting candidates for the item.
        """
        self.update_candidates_data(scene_objects_dict)

        # TODO: Double check this
        max_of_min_advancement = max(
            candidate_data.relation_min_advancement for candidate_data in self.candidates_data.values()
        )
        interesting_candidates = {
            candidate_id
            for candidate_id in self.candidate_ids
            if self.candidates_data[candidate_id].relation_max_advancement >= max_of_min_advancement
        }
        return interesting_candidates

    # TODO: Delete?
    def compute_advancement_details(self, global_assignment: Assignment) -> ItemAdvancementDetails:
        """
        Compute the advancement details of the item.

        Args:
            global_assignment (Assignment): Global assignment of the task.

        Returns:
            advancement_details (ItemAdvancementDetails): Advancement details of the item.
        """
        return ItemAdvancementDetails(
            item=self,
            global_assignment=global_assignment,
        )

    def __str__(self) -> str:
        return f"{self.id}"

    def __repr__(self) -> str:
        # return f"TaskItem({self.id})\n  properties={self.properties})\n  relations={self.relations})"
        return f"TaskItem({self.id})"

    # def __hash__(self) -> int:
    #     return hash(self.id)

    # def __eq__(self, other: Any) -> bool:
    #     if not isinstance(other, TaskItem):
    #         return False
    #     return self.id == other.id and self.properties == other.properties and self.relations == other.relations


class AuxItem(TaskItem):
    """
    An auxiliary item in the definition of an item property.

    The relations of an auxiliary item are considered to be relations between the auxiliary item as
    main item and the linked item (item owning the linked property) as related item. The relations
    are instantiated when the linked item is known by using the initialize() method.

    Example of such auxiliary items:
    - A Knife is necessary for the property "is_sliced"
    - A Fridge is necessary for the property "temperature" with the value "cold"

    !! In contrary to the main task items, the candidates of those items never change during an episode, so in particular, they can't be sliced.

    Attributes:
        linked_prop (ItemVariableProp): The main property of the main item to which the auxiliary item is related.
    """

    def __init__(
        self,
        t_id: ItemId | str,
        properties: set[ItemProp] | frozenset[ItemProp],
        relation_descriptions: dict[type[Relation], dict[str, RelationParam]] | None = None,
    ) -> None:
        """
        Initialize the main attributes of the auxiliary item to enable the full initialization later.

        Args:
            t_id (ItemId): The ID of the item as defined in the task description.
            properties (set[ItemProp]): Set of properties of the item.
            relation_descriptions (dict[type[Relation], dict[str, RelationParam]], optional):
                Dictionary mapping the relation types of the auxiliary items to their parameter
                dictionaries. We need those to instantiate the relations when we know the linked
                item.
        """
        self.t_id = ItemId(t_id)
        self.properties = frozenset(properties)
        if relation_descriptions is None:
            relation_descriptions = {}
        self._relation_descriptions = relation_descriptions

        # === Type annotations ===
        self.linked_prop: ItemVariableProp
        self._relation_descriptions = relation_descriptions

    def initialize_relations(self, linked_item_id: ItemId) -> None:
        """
        Initialize the auxiliary item relations with the linked item id.

        This is necessary to instantiate the relations of the auxiliary item with the information
        of the linked item id.

        Args:
            linked_item_id (ItemId): The ID of the item owning the linked property.
        """
        relations = {
            relation_type(
                main_item_id=self.t_id,
                related_item_id=linked_item_id,
                _inverse_relation=None,
                **param,
            )
            for relation_type, param in self._relation_descriptions.items()
        }

        super().__init__(self.t_id, self.properties, relations)

    def __str__(self) -> str:
        return f"AuxItem({self.id})"

    def __repr__(self) -> str:
        return f"AuxItem({self.id}, {self.linked_prop})"


# %% === Overlap class ===
type Assignment = dict[TaskItem, CandidateId]


class ItemOverlapClass:
    """
    A group of items whose sets of candidates overlap.

    Attributes:
        items (list[TaskItem]): The items in the overlap class.
        candidate_ids (list[CandidateId]): The candidate ids of candidates in the overlap class.
        valid_assignments (list[Assignment]): List of valid assignments of objects to the items
            in the overlap class.
    """

    def __init__(
        self,
        items: list[TaskItem],
        candidate_ids: list[CandidateId],
    ) -> None:
        """
        Initialize the overlap class' items, candidate ids and valid assignments.

        Args:
            items (list[TaskItem]): The items in the overlap class.
            candidate_ids (list[CandidateId]): The candidate ids of candidates in the overlap class.
        """
        for item in items:
            item.overlap_class = self
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
            # raise NoValidAssignmentError(self)
            print(f"No valid assignment for overlap class {self}")

        # === Type annotations ===
        self.items: list[TaskItem]
        self.candidate_ids: list[CandidateId]
        self.valid_assignments: list[Assignment]

    def compute_valid_assignments_with_inherited_objects(
        self, main_object_id: CandidateId, inherited_object_ids: set[CandidateId]
    ) -> list[Assignment]:
        """
        Update the valid assignments with the given main object and inherited objects.

        Replace the main object by its inherited objects in the valid assignments.

        Args:
            main_object_id (CandidateId): The id of the main object from which the inherited objects
                are inherited.
            inherited_object_ids (set[CandidateId]): Set of the ids of the inherited objects.

        Returns:
            updated_valid_assignments (list[Assignment]): List of the updated valid assignments
                where the main object has been replaced by its inherited objects.
        """
        new_valid_assignments: list[Assignment] = []
        for assignment in self.valid_assignments:
            for item, candidate_id in assignment.items():
                if candidate_id == main_object_id:
                    for inherited_object_id in inherited_object_ids:
                        new_assignment = assignment.copy()
                        new_assignment[item] = inherited_object_id
                        new_valid_assignments.append(new_assignment)
                    break
            else:
                new_valid_assignments.append(assignment)

        return new_valid_assignments

    def prune_assignments(self, compatible_global_assignments: list[Assignment]) -> None:
        """
        Prune the valid assignments to keep only those that are part of the given compatible assignments.

        Valid assignments are assignments where each item is associated with a candidate
        that has all correct candidate_required_properties (without taking into account the
        relations between the items) and compatible assignments are valid assignment where the
        candidates are compatible when taking into account the relations between the items.

        Args:
            compatible_global_assignments (list[Assignment]): List of global
                compatible (for the whole task and not only this overlap class).
        """
        compatible_global_assignments_set = {
            tuple(global_assignment[item] for item in self.items) for global_assignment in compatible_global_assignments
        }

        self.valid_assignments = [
            dict(zip(self.items, assignment_tuple, strict=True))
            for assignment_tuple in compatible_global_assignments_set
        ]

    # I think there is a mistake in the computation of interesting candidates because it also
    # depends on the other items of the overlap class and not the item and its related items.
    # TODO: Check this and fix if needed
    def compute_interesting_assignments(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> list[Assignment]:
        """
        Return the interesting assignments of objects to the items in the overlap class, the items results and items scores.

        The interesting assignments are the ones that can lead to a maximum of task advancement
        depending on the assignment of objects in the other overlap classes.
        Interesting assignments are the ones where each all assigned objects are interesting
        candidates for their item.
        Note: We reduce the set of interesting assignments by considering this relations between the items in the overlap class (with the information of the assignment of the other items in the overlap class).
        # TODO? Implement this

        For more details about the definition of interesting candidates, see the
        compute_interesting_candidates method of the TaskItem class.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary containing the metadata of
                the objects in the scene. The keys are the object ids.

        Returns:
            interesting_assignments (list[Assignment]):
                List of the interesting assignments of objects to the items in the overlap class.
        """
        # Extract the interesting candidates
        items_interesting_candidates = {
            item: item.compute_interesting_candidates(scene_objects_dict) for item in self.items
        }

        # Filter the valid assignments to keep only the ones with interesting candidates
        interesting_assignments = [
            assignment
            for assignment in self.valid_assignments
            if all(assignment[item] in items_interesting_candidates[item] for item in self.items)
        ]
        # TODO? Implement the filtering of the assignments based on the relations between the items in the overlap class

        return interesting_assignments

    def __str__(self) -> str:
        return f"{{self.items}}"

    def __repr__(self) -> str:
        return f"OverlapClass({self.items})"


# %% === Exceptions ===
# TODO: Unused; Replace by a warning?
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
