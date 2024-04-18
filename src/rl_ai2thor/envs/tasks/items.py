"""
Task items in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, NewType

from rl_ai2thor.envs.sim_objects import (
    SimObjId,
    SimObjMetadata,
    SimObjProp,
)
from rl_ai2thor.envs.tasks.item_prop_interface import (
    ItemVariableProp,
)
from rl_ai2thor.utils.global_exceptions import DuplicateRelationsError

if TYPE_CHECKING:
    from rl_ai2thor.envs.tasks.item_prop_interface import (
        ItemFixedProp,
        ItemProp,
        ItemPropValue,
        PropAuxProp,
    )
    from rl_ai2thor.envs.tasks.relations import Relation, RelationTypeId


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
        props_aux_properties_results (dict[ItemVariableProp, dict[PropAuxProp, bool]]):
            Dictionary mapping the item's scored properties to the results of the auxiliary
            properties satisfaction for the candidate.
        relations_results (dict[Relation, set[CandidateId]]): Dictionary mapping the item's
        relations to the set of satisfying related item's candidate ids for the candidate.
        properties_scores (dict[ItemVariableProp, float]): Dictionary mapping the item properties to the
            scores of the candidate for the properties.
        property_score (float): The score of the candidate for the item's properties, sum of the
            scores of the properties.
        relations_min_max_scores (dict[Relation, tuple[float, float]]): Dictionary mapping the
            item's relations to the minimum and maximum scores of the candidate for the relations.
        relation_max_score (float): The maximum score of the candidate for the item's relations,
            sum of the scores of the relations where there is at least one semi-satisfying related
            item's candidate (i.e. the set of satisfying objects is not empty but they might not be
            part of the assignment).
        relation_min_score (float): The minimum score of the candidate for the item's relations,
            sum of the scores of the relations where all related item's candidate are satisfying.
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
        if TYPE_CHECKING:
            self.item: TaskItem
            self.id: CandidateId
            self.metadata: SimObjMetadata
            self.base_properties_results: dict[ItemVariableProp, bool]
            self.props_aux_properties_results = dict[ItemVariableProp, dict[PropAuxProp, bool]]
            self.relations_satisfying_related_candidates_ids: dict[Relation, set[CandidateId]]
            self.properties_scores: dict[ItemVariableProp, float]
            self.property_score: float
            self.relations_min_max_scores: dict[Relation, tuple[float, float]]
            self.relation_max_score: float
            self.relation_min_score: float

    @property
    def _scored_properties(self) -> frozenset[ItemVariableProp]:
        """
        Get the scored properties of the item.

        Returns:
            frozenset[ItemVariableProp]: The scored properties of the item.
        """
        return self.item.scored_properties

    @property
    def _relations(self) -> frozenset[Relation]:
        """
        Get the relations of the item.

        Returns:
            frozenset[Relation]: The relations of the item.
        """
        return self.item.relations

    def update(self, scene_object_dict: dict[SimObjId, SimObjMetadata]) -> None:
        """
        Update the candidate data with the given scene object dictionary.

        Args:
            scene_object_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.
        """
        # === Properties ===
        self.metadata = scene_object_dict[self.id]
        self.base_properties_results = self._compute_base_properties_results()
        self.props_aux_properties_results = self._compute_props_aux_properties_results(self.base_properties_results)
        self.props_aux_items_results = self._compute_props_aux_items_results(self.base_properties_results)

        self.properties_scores = self._compute_properties_scores(
            self.base_properties_results, self.props_aux_properties_results
        )
        self.property_score = sum(self.properties_scores.values())

        # === Relations ===
        self.relations_satisfying_related_candidates_ids = self._compute_relations_satisfying_related_candidates_ids(
            scene_object_dict
        )
        self.relations_aux_properties_results = self._compute_relations_aux_properties_results(
            self.relations_satisfying_related_candidates_ids
        )
        self.relations_min_max_scores = self._compute_relations_min_max_scores(
            self.relations_satisfying_related_candidates_ids
        )
        self.relation_min_score = sum(min_score for (min_score, _max_score) in self.relations_min_max_scores.values())
        self.relation_max_score = sum(max_score for (_min_score, max_score) in self.relations_min_max_scores.values())

        # === Final score ===
        self.min_score = self.property_score + self.relation_min_score
        self.max_score = self.property_score + self.relation_max_score

    def _compute_base_properties_results(self) -> dict[ItemVariableProp, bool]:
        """
        Return the results dictionary of each properties for the candidate.

        Returns:
            base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item properties
            to the results of the property satisfaction for the candidate.
        """
        return {prop: prop.is_object_satisfying(self.metadata) for prop in self._scored_properties}

    def _compute_props_aux_properties_results(
        self, properties_results: dict[ItemVariableProp, bool]
    ) -> dict[ItemVariableProp, dict[PropAuxProp, bool]]:
        """
        Return the results dictionary of each auxiliary properties for the candidate.

        Args:
            properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item properties to the
                results of the property satisfaction for the candidate.

        Returns:
            aux_properties_results (dict[ItemVariableProp, dict[PropAuxProp, bool]]): Dictionary
                mapping the item's properties to the results of the auxiliary properties
                satisfaction for the candidate.
        """
        return {
            prop: {
                aux_prop: properties_results[prop] or aux_prop.is_object_satisfying(self.metadata)
                for aux_prop in prop.auxiliary_properties
            }
            for prop in self._scored_properties
        }

    def _compute_relations_satisfying_related_candidates_ids(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> dict[Relation, set[CandidateId]]:
        """
        Return the satisfying related candidates ids for the candidate for each relation of the item.

        Returns:
            relations_results (dict[Relation, set[CandidateId]]): Dictionary mapping the item's relations
                to the set of satisfying related item's candidate ids for the candidate.
        """
        return {
            relation: relation.compute_satisfying_related_candidates_ids(self.metadata, scene_objects_dict)
            for relation in self._relations
        }

    # TODO: Implement weighted properties
    @staticmethod
    def _compute_properties_scores(
        base_properties_results: dict[ItemVariableProp, bool],
        props_aux_properties_results: dict[ItemVariableProp, dict[PropAuxProp, bool]],
    ) -> dict[ItemVariableProp, float]:
        """
        Return the scores of the candidate for the item's properties.

        Args:
            base_properties_results (dict[ItemVariableProp, bool]): Dictionary mapping the item properties
            to the results of the property satisfaction for the candidate.
            props_aux_properties_results (dict[ItemVariableProp, dict[ItemVariableProp, bool]]):
            Dictionary mapping the item's properties to the results of the auxiliary properties
            satisfaction for the candidate.

        Returns:
            properties_scores (dict[ItemVariableProp, float]): Dictionary mapping the item properties to the
                scores of the candidate for the properties.
        """
        base_score = {prop: int(result) for prop, result in base_properties_results.items()}
        aux_score = {
            prop: sum(int(result) for result in props_aux_properties_results[prop].values())
            for prop in props_aux_properties_results
        }
        return {prop: base_score[prop] + aux_score[prop] for prop in itertools.chain(base_score, aux_score)}

    # TODO: Implement weighted relations
    @staticmethod
    def _compute_relations_min_max_scores(
        relations_results: dict[Relation, set[CandidateId]],
    ) -> dict[Relation, tuple[float, float]]:
        """
        Return the minimum and the maximum scores of the candidate for each item's relations.

        Minimum and maximum score meaning:
        - The minimum score of the candidate for the item's relations is the sum of the scores of the
        relations where all related item's candidate are satisfying (so the relation will be
        satisfied whatever the assignment).
        - The maximum score of the candidate for the item's relations is the sum of the scores of the
        relations where there is at least one semi-satisfying related item's candidate (i.e. the set
        of satisfying objects is not empty but they might not be part of the assignment).

        Args:
            relations_results (dict[Relation, set[CandidateId]]): Dictionary mapping the item's relations
                to the set of satisfying related item's candidate ids for the candidate.

        Returns:
            relations_min_max_scores (dict[Relation, tuple[float, float]]): Dictionary mapping the
            item's relations to the minimum and maximum scores of the candidate for the relations.
        """
        return {
            relation: (
                1 if len(relations_results[relation]) == len(relation.related_item.candidate_ids) else 0,
                1 if relations_results[relation] else 0,
            )
            for relation in relations_results
        }

    def __str__(self) -> str:
        return f"CandidateData({self.id})"

    def __repr__(self) -> str:
        return f"CandidateData({self.item.id}, {self.id})"


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
        max_advancement (int): Maximum advancement of the item in the task: sum of the maximum
            advancement of the scored properties and relations.
        candidate_ids (set[SimObjId]): Set of candidate ids of the item.
        candidates_data (dict[CandidateId, CandidateData]): Dictionary mapping the candidate ids to
            their data.
    """

    def __init__(
        self,
        t_id: ItemId | str,
        properties: set[ItemProp],
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
        self.max_advancement = sum(prop.max_advancement for prop in self.scored_properties)

        # Infer the candidate required properties from the item properties
        self.candidate_required_properties = frozenset(
            prop.candidate_required_prop for prop in self.properties if prop.candidate_required_prop is not None
        )

        # === Type annotations ===
        self.id: ItemId
        self.properties: frozenset[ItemProp[ItemPropValue, ItemPropValue]]
        self.scored_properties: frozenset[ItemVariableProp[ItemPropValue, ItemPropValue]]
        self.max_advancement: int
        self.candidate_required_properties: frozenset[ItemFixedProp[ItemPropValue]]
        self.candidates_data: dict[CandidateId, CandidateData]

    @property
    def candidate_ids(self) -> set[CandidateId]:
        """
        Get the candidate ids of the item.

        Returns:
            set[CandidateId]: Set of candidate ids of the item.
        """
        return set(self.candidates_data.keys())

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

    def _update_candidates_data(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> None:
        """
        Update the data of the candidates of the item with the given scene object dictionary.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.
        """
        for candidate_id in self.candidate_ids:
            self.candidates_data[candidate_id].update(scene_objects_dict)

    # TODO: Delete
    def compute_candidates_props_scores(
        self,
        candidates_props_results: dict[ItemProp, dict[CandidateId, bool]],
    ) -> dict[CandidateId, float]:
        """
        Return the property scores of each candidate of the item.

        Args:
            candidates_props_results (dict[ItemProp, dict[CandidateId, bool]]): Dictionary mapping the
                item properties to the results of each candidates for the properties.

        Returns:
            candidates_props_scores (dict[CandidateId, float]): Dictionary mapping the candidate ids to
                their property scores.
        """
        prop_candidates_scores = {
            prop: prop.compute_candidates_scores(candidates_props_results[prop]) for prop in candidates_props_results
        }
        return {
            candidate_id: sum(prop_candidates_scores[prop][candidate_id] for prop in prop_candidates_scores)
            for candidate_id in self.candidate_ids
        }

    # TODO: Delete
    def compute_candidates_props_results(
        self,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[ItemProp[ItemPropValue, ItemPropValue], dict[CandidateId, bool]]:
        """
        Return the results dictionary of each properties for the candidates of the item.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            candidates_props_results (dict[ItemProp[ItemPropValue, ItemPropValue], dict[CandidateId, bool]]):
                Dictionary mapping the item properties to the results of each candidates for the properties.
        """
        return {
            prop: prop.compute_candidates_results(scene_objects_dict, self.candidate_ids) for prop in self.properties
        }


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
        props_auxiliary_items (dict[ItemProp, frozenset[TaskItem]]): Map of the item's properties to
            their auxiliary items.
        props_auxiliary_properties (dict[ItemProp, frozenset[ItemVariableProp]]): Map of the item's
            properties to their auxiliary properties.
        relations_auxiliary_properties (dict[Relation, frozenset[ItemVariableProp]]): Map of the
            item's relations to their auxiliary properties.
        candidate_ids (set[SimObjId]): Set of candidate ids of the item.
        candidates_data (dict[CandidateId, CandidateData]): Dictionary mapping the candidate ids to
            their data.
    """

    def __init__(
        self,
        t_id: ItemId | str,
        properties: set[ItemProp],
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

        self.max_advancement += sum(relation.max_advancement for relation in self.relations)

        relation_required_properties = frozenset(
            relation.candidate_required_prop
            for relation in self.relations
            if relation.candidate_required_prop is not None
        )
        self.candidate_required_properties |= relation_required_properties

        #  === Auxiliary items and properties ===
        self.props_auxiliary_items = {prop: prop.auxiliary_items for prop in self.scored_properties}
        self.props_auxiliary_properties = {prop: prop.auxiliary_properties for prop in self.scored_properties}
        self.relations_auxiliary_properties = {relation: relation.auxiliary_properties for relation in self.relations}

        # === Type annotations ===
        self.relations: frozenset[Relation]
        self.props_auxiliary_items: dict[ItemProp[ItemPropValue, ItemPropValue], frozenset[AuxItem]]
        self.props_auxiliary_properties: dict[ItemProp[ItemPropValue, ItemPropValue], frozenset[ItemVariableProp]]
        self.relations_auxiliary_properties: dict[Relation, frozenset[ItemVariableProp]]

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

    @staticmethod
    def _check_duplicate_relations(relations: frozenset[Relation]) -> None:
        """
        Check that there are no duplicate relations of the same type and main and related items.

        Args:
            relations (frozenset[Relation]): Set of relations to check.
        """
        existing_relations = {}
        for relation in relations:
            if relation.related_item.id not in existing_relations:
                existing_relations[relation.related_item.id] = {}
            elif relation.type_id in existing_relations[relation.related_item.id]:
                raise DuplicateRelationsError(relation.type_id, relation.main_item_id, relation.related_item_id)
            existing_relations[relation.related_item.id][relation.type_id] = relation

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

    # TODO: Replace keys by the actual properties
    # TODO: Delete; unused
    def _get_properties_satisfaction(self, obj_metadata: SimObjMetadata) -> dict[SimObjProp, bool]:
        """
        Return a dictionary indicating which properties are satisfied by the given object.

        Args:
            obj_metadata (SimObjMetadata): Object metadata.

        Returns:
            prop_satisfaction (dict[SimObjProp, bool]): Dictionary indicating which properties are
                satisfied by the given object.
        """
        return {prop.target_ai2thor_property: prop.is_object_satisfying(obj_metadata) for prop in self.properties}

    # TODO: Delete; unused
    def _get_relations_semi_satisfying_objects(
        self,
        candidate_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[ItemId, dict[RelationTypeId, set[CandidateId]]]:
        """
        Return the dictionary of satisfying objects with the given candidate for each relations.

        The relations are organized by related item id.

        Args:
            candidate_metadata (SimObjMetadata): Metadata of the candidate.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            semi_satisfying_objects (dict[ItemId, dict[RelationTypeId, set[CandidateId]]]): Dictionary
                indicating which objects are semi-satisfying the relations with the given object.
        """
        return {
            related_item_id: {
                relation.type_id: relation.compute_satisfying_related_candidates_ids(
                    candidate_metadata, scene_objects_dict
                )
                for relation in self.organized_relations[related_item_id].values()
            }
            for related_item_id in self.organized_relations
        }

    # TODO: Delete
    def compute_candidates_relations_results(
        self,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]]:
        """
        Return the results dictionary of each relations for the candidates of the item.

        The relations are organized by related item id.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id of the
                objects in the scene to their metadata.

        Returns:
            candidates_relations_results (dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]]):
                Dictionary mapping the related item ids to the results dictionary of each relation,
                    which maps the candidate ids to the set of satisfying object ids.
        """
        return {
            main_item_id: {
                relation: relation.compute_main_candidates_results(scene_objects_dict)
                for relation in self.organized_relations[main_item_id].values()
            }
            for main_item_id in self.organized_relations
        }

    # TODO: Delete
    def compute_candidates_relations_scores(
        self,
        candidates_relations_results: dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]],
    ) -> dict[CandidateId, float]:
        """
        Return the relation scores of each candidate of the item.

        Args:
            candidates_relations_results (dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]):
                Dictionary mapping the related item ids to the results dictionary of each relation,
                which maps the candidate ids to the set of satisfying object ids.

        Returns:
            candidates_relations_scores (dict[CandidateId, float]): Dictionary mapping the candidate ids
                to their relation scores.
        """
        # return {
        #     candidate_id: sum(
        #         1
        #         for related_item_id in candidates_relations_results
        #         for relation in candidates_relations_results[related_item_id]
        #         if candidates_relations_results[related_item_id][relation][candidate_id]
        #     )
        #     for candidate_id in self.candidate_ids
        # }

        relations_candidates_scores = {
            relation: relation.compute_main_candidates_scores(candidates_relations_results[related_item_id][relation])
            for related_item_id in candidates_relations_results
            for relation in candidates_relations_results[related_item_id]
        }

        return {
            candidate_id: sum([
                relations_candidates_scores[relation][candidate_id] for relation in relations_candidates_scores
            ])
            for candidate_id in self.candidate_ids
        }

    def compute_interesting_candidates(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> tuple[
        set[CandidateId],
        dict[ItemProp, dict[CandidateId, bool]],
        dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]],
        dict[CandidateId, float],
        dict[CandidateId, float],
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
            interesting_candidates (set[CandidateId]): Set of interesting candidates for the item.
            candidates_properties_results (dict[SimObjProp, dict[CandidateId, bool]]): Results of each
                object for the item properties.
            candidates_relations_results (dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]]):
                Results of each object for the item relations.
            candidates_properties_scores (dict[CandidateId, float]): Property scores of each object for
                the item.
            candidates_relations_scores (dict[CandidateId, float]): Relation scores of each object for
                the item.
        """
        self._update_auxiliary_items_data(scene_objects_dict)
        self._update_candidates_data(scene_objects_dict)
        # TODO: Update this function

        # Compute the properties and relations results of each object for the item
        candidates_properties_results = self.compute_candidates_props_results(scene_objects_dict)
        candidates_relations_results = self.compute_candidates_relations_results(scene_objects_dict)

        # Compute the scores of each object for the item
        candidates_properties_scores = self.compute_candidates_props_scores(candidates_properties_results)
        candidates_relations_scores = self.compute_candidates_relations_scores(candidates_relations_results)

        # Compute the results and scores of each auxiliary item
        aux_items_candidates_props_results = {
            prop: {aux_item: aux_item.compute_candidates_props_results(scene_objects_dict) for aux_item in aux_items}
            for prop, aux_items in self.props_auxiliary_items.items()
        }
        aux_items_scores = {
            prop: {
                aux_item: aux_item.compute_candidates_props_scores(aux_items_candidates_props_results[prop][aux_item])
                for aux_item in aux_items
            }
            for prop, aux_items in self.props_auxiliary_items.items()
        }

        # Compute the results of each auxiliary property for the items
        aux_props_results: dict[
            ItemProp,
            dict[
                ItemVariableProp,
                dict[CandidateId, bool],
            ],
        ] = {
            main_prop: {
                aux_prop: aux_prop.compute_candidates_results(scene_objects_dict, self.candidate_ids)
                for aux_prop in aux_props
            }
            for main_prop, aux_props in self.props_auxiliary_properties.items()
        }
        # Update auxiliary properties results according to their main property results
        # If the main property is satisfied, then its auxiliary properties are satisfied
        for main_prop in aux_props_results:
            for aux_prop in aux_props_results[main_prop]:
                for candidate_id in aux_props_results[main_prop][aux_prop]:
                    aux_props_results[main_prop][aux_prop][candidate_id] &= candidates_properties_results[main_prop][
                        candidate_id
                    ]
        aux_props_scores = {
            main_prop: {
                aux_prop: aux_prop.compute_candidates_scores(aux_props_results[main_prop][aux_prop])
                for aux_prop in aux_props
            }
            for main_prop, aux_props in self.props_auxiliary_properties.items()
        }
        # TODO: Finish

        # TODO: Check that it takes auxiliary properties and items into account
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
        obj_1_id: CandidateId,
        obj_2_id: CandidateId,
        candidates_relations_results: dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]],
        candidates_properties_scores: dict[CandidateId, float],
        candidates_relations_scores: dict[CandidateId, float],
    ) -> CandidateId | Literal["equal", "incomparable"]:
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
            obj_1_id (CandidateId): First candidate object id.
            obj_2_id (CandidateId): Second candidate object id.
            candidates_relations_results (dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]]):
                Results of each object for the item relations.
            candidates_properties_scores (dict[CandidateId, float]):
                Property scores of each object for the item.
            candidates_relations_scores (dict[CandidateId, float]):
                Relation scores of each object for the item.

        Returns:
            stronger_candidate (CandidateId | Literal["equal", "incomparable"]): The stronger candidate
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
        obj_1_id: CandidateId,
        obj_2_id: CandidateId,
        candidates_relations_results: dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]],
        candidates_properties_scores: dict[CandidateId, float],
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
            obj_1_id (CandidateId): First candidate object id.
            obj_2_id (CandidateId): Second candidate object id.
            candidates_relations_results (dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]]):
                Results of each object for the item relations.
            candidates_properties_scores (dict[CandidateId, float]):
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
            for relation in candidates_relations_results[related_item_id]:
                x_sat_obj_ids = candidates_relations_results[related_item_id][relation][obj_1_id]
                y_sat_obj_ids = candidates_relations_results[related_item_id][relation][obj_2_id]
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


class AuxItem(TaskItem):
    """
    An auxiliary item in the definition of an item property.

    Example of such auxiliary items:
    - A Knife is necessary for the property "is_sliced"
    - A Fridge is necessary for the property "temperature" with the value "cold"

    Attributes:
        linked_prop (ItemVariableProp): The main property of the main item to which the auxiliary item is related.
    """

    def __init__(
        self,
        t_id: ItemId | str,
        properties: set[ItemProp],
        relations: set[Relation] | None = None,
    ) -> None:
        """
        Initialize the AuxItem object.

        Args:
            t_id (ItemId): The ID of the item as defined in the task description.
            properties (set[ItemProp]): Set of properties of the item.
            relations (set[Relation], optional): Set of relations of the item.
        """
        super().__init__(t_id, properties, relations)

        # === Type annotations ===
        self.linked_prop: ItemVariableProp

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
    def compute_interesting_assignments(
        self, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> tuple[
        list[Assignment],
        dict[TaskItem, dict[ItemProp, dict[CandidateId, bool]]],
        dict[TaskItem, dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]]],
        dict[TaskItem, dict[CandidateId, float]],
        dict[TaskItem, dict[CandidateId, float]],
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
            interesting_assignments (list[Assignment]):
                List of the interesting assignments of objects to the items in the overlap class.
            all_properties_results (dict[TaskItem, dict[ItemProp, dict[CandidateId, bool]]]):
                Results of each object for each property of each item in the overlap class.
            all_relation_results (dict[TaskItem, dict[ItemId, dict[Relation, dict[CandidateId, set[CandidateId]]]]]):
                Results of each object for the relation of each item in the overlap class.
            all_properties_scores (dict[TaskItem, dict[CandidateId, float]]):
                Property scores of each object for each item in the overlap class.
            all_relations_scores (dict[TaskItem, dict[CandidateId, float]]):
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
