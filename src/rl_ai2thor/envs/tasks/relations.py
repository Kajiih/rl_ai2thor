"""
Task relations in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

from rl_ai2thor.envs.sim_objects import OBJECT_TYPES_DATA, SimObjFixedProp, SimObjId, SimObjMetadata
from rl_ai2thor.envs.tasks.item_prop import PickupableProp, ReceptacleProp
from rl_ai2thor.envs.tasks.items import ItemId
from rl_ai2thor.utils.ai2thor_utils import compute_objects_2d_distance

if TYPE_CHECKING:
    from rl_ai2thor.envs.tasks.item_prop_interface import AuxProp, ItemFixedProp
    from rl_ai2thor.envs.tasks.items import CandidateId, TaskItem


# %% === Enums ===
# TODO: Add more relations
class RelationTypeId(StrEnum):
    """Relations between items."""

    RECEPTACLE_OF = "receptacle_of"
    CONTAINED_IN = "contained_in"
    CLOSE_TO = "close_to"


type RelationParam = int | float | bool


# %% === Relations ===
# TODO? Optimize the computation of the results and the advancement by reusing the results of the inverse relation when possible?
class Relation(ABC):
    """
    A relation between two items in the definition of a task.

    The inverse relation is automatically created when the relation is instantiated, and in
    particular, the inverse relation should not be manually instantiated and instead, one should
    use the inverse_relation attribute of the relation to access it.

    The main_item and related_item are the items that are automatically set when the main item and
    teh related item respectively are instantiated with the relation and the inverse relation
    respectively.

    Attributes:
        type_id (RelationTypeId): The type of the relation.
        inverse_relation_type_id (RelationTypeId): The id of the type of the inverse relation.
        candidate_required_prop (ItemFixedProp | None): The candidate required property for the main
            item.
        auxiliary_properties (frozenset[AuxProp]): The auxiliary properties of the relation.
        main_item_id (ItemId): The id of the main item of the relation.
        related_item_id (ItemId): The id of the related item of the relation.
        inverse_relation (Relation): The inverse relation of the relation.
        main_item (TaskItem): The main item in the relation. Automatically set when the main item is
            instantiated with this relation.
        related_item (TaskItem): The related item to which the main item is related. Automatically
            set when the related item is instantiated with the inverse relation.
    """

    type_id: RelationTypeId
    inverse_relation_type_id: RelationTypeId
    candidate_required_prop: ItemFixedProp | None = None
    auxiliary_properties: frozenset[AuxProp] = frozenset()

    def __init__(
        self,
        main_item_id: ItemId | str,
        related_item_id: ItemId | str,
        _inverse_relation: Relation | None = None,
    ) -> None:
        """
        Initialize the relation and eventually the inverse relation.

        Args:
            main_item_id (ItemId): Id of the main item of the relation.
            related_item_id (ItemId): Id of the related item of the relation.
            _inverse_relation (Relation, optional): The inverse relation of the relation. Should not
                be used outside of _initialize_inverse_relation to have coherent inverse relations.
        """
        self.main_item_id = ItemId(main_item_id)
        self.related_item_id = ItemId(related_item_id)

        if _inverse_relation is not None:
            self.inverse_relation = _inverse_relation
        else:
            self.inverse_relation = self._initialize_inverse_relation()

        self.max_advancement = 1 + len(self.auxiliary_properties)

        # self.are_candidates_compatible = functools.lru_cache(maxsize=None)(self._uncached_are_candidates_compatible)

        # === Type Annotations ===
        self.main_item_id: ItemId
        self.related_item_id: ItemId
        self.inverse_relation: Relation
        self.main_item: TaskItem
        self.related_item: TaskItem

    @abstractmethod
    def _compute_inverse_relation_parameters(self) -> dict[str, RelationParam]:
        """
        Return the parameters of the inverse relation.

        Doesn't return the main and related items of the inverse relation since they don't depend on the relation.
        The output of this method is used to instantiate the inverse relation in create_inverse_relation().

        Returns:
            inverse_relation_parameters (dict[str, RelationParam]): The parameters of the inverse relation.
        """

    def _initialize_inverse_relation(self) -> Relation:
        """
        Initialize and return the inverse relation.

        Returns:
            inverse_relation (Relation): The inverse relation.
        """
        inverse_relation_parameters = self._compute_inverse_relation_parameters()
        inverse_relation_type = relation_type_id_to_relation[self.inverse_relation_type_id]
        inverse_relation = inverse_relation_type(
            **inverse_relation_parameters,
            main_item_id=self.related_item_id,
            related_item_id=self.main_item_id,
            _inverse_relation=self,
        )
        return inverse_relation

    # TODO: Find a way to implement the same thing but without the need to check every pair of candidates
    @abstractmethod
    def _are_candidates_compatible(
        self,
        main_candidate_metadata: SimObjMetadata,
        related_candidate_metadata: SimObjMetadata,
    ) -> bool:
        """
        Return True if the candidates satisfy the relation.

        It doesn't check the candidate required property since it is already used to filter the candidates.

        Args:
            main_candidate_metadata (SimObjMetadata): The metadata of a candidate of the main item.
            related_candidate_metadata (SimObjMetadata): The metadata of a candidate of the related item.

        Returns:
            compatible (bool): True if the candidates are compatible with the relation and that they can satisfy it in an assignment.
        """

    # TODO: Check if we keep this; probably unused
    def compute_compatible_related_candidates(
        self,
        main_candidate_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> set[SimObjId]:
        """
        Return the ids of the related candidates that are compatible with the main candidate.

        Args:
            main_candidate_metadata (SimObjMetadata): The metadata of a candidate of the main item.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
            of the objects in the scene to their metadata.

        Returns:
            compatible_related_candidates (set[SimObjId]): The ids of the related candidates that are compatible with the main candidate.
        """
        return {
            related_object_id
            for related_object_id in self._extract_related_object_ids(main_candidate_metadata, scene_objects_dict)
            if self._are_candidates_compatible(main_candidate_metadata, scene_objects_dict[related_object_id])
        }

    @abstractmethod
    def _extract_related_object_ids(
        self,
        main_obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> list[SimObjId]:
        """
        Return the list of the ids of the main object's related objects according to the relation.

        Args:
            main_obj_metadata (SimObjMetadata): The metadata of the main object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
            of the objects in the scene to their metadata.

        Returns:
            list[SimObjId]: The ids of the main object's related objects.
        """

    def is_semi_satisfied(
        self,
        main_obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> bool:
        """
        Return True if the relation is semi satisfied in the given main object.

        A relation is semi satisfied if the main object is correctly
        related to a candidate of the related item (but no related
        object might be assigned to the related item).

        Args:
            main_obj_metadata (SimObjMetadata): The metadata of the main object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
            of the objects in the scene to their metadata.

        Returns:
            bool: True if the relation is semi satisfied.
        """
        return any(
            related_object_id in self.related_item.candidate_ids
            for related_object_id in self._extract_related_object_ids(main_obj_metadata, scene_objects_dict)
        )

    def compute_satisfying_related_candidates_ids(
        self,
        main_obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> set[CandidateId]:
        """
        Return related item's candidate's ids that satisfy the relation with the given main item's candidate.

        Args:
            main_obj_metadata (SimObjMetadata): The metadata of the main object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
            of the objects in the scene to their metadata.

        Returns:
            set[CandidateId]: The ids of the related item's candidate's that satisfy the relation.
        """
        satisfying_related_object_ids = {
            related_object_id
            for related_object_id in self._extract_related_object_ids(main_obj_metadata, scene_objects_dict)
            if related_object_id in self.related_item.candidate_ids
        }
        return satisfying_related_object_ids

    # TODO: Delete?
    def compute_main_candidates_results(
        self,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> dict[CandidateId, set[CandidateId]]:
        """
        Return the set of related item's candidate's ids that satisfy the relation for each main object's candidate.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
                of the objects in the scene to their metadata.

        Returns:
            main_candidates_results (dict[CandidateId, set[CandidateId]]): The set of related item's
                candidate's ids that satisfy the relation for each main object's candidate.
        """
        return {
            main_candidate_id: self.compute_satisfying_related_candidates_ids(
                scene_objects_dict[main_candidate_id], scene_objects_dict
            )
            for main_candidate_id in self.main_item.candidate_ids
        }

    # TODO: Delete?
    # TODO: Implement a weighted score
    @staticmethod
    def compute_main_candidates_scores(
        main_candidates_results: dict[CandidateId, set[CandidateId]],
    ) -> dict[CandidateId, float]:
        """
        Return the score of each main object's candidate based on the related item's candidate's ids that satisfy the relation.

        The score is the number of related item's candidate's ids that satisfy the relation for the
        main object's candidate.

        Args:
            main_candidates_results (dict[SimObjId, set[SimObjId]]): Dictionary mapping the id of
                the main object's candidates to the set of related item's candidate's ids that satisfy the relation.

        Returns:
            main_candidates_scores (dict[SimObjId, float]): Dictionary mapping the id of the main
                object's candidates to their score.
        """
        return {
            main_candidate_id: 1 if related_object_ids else 0
            for main_candidate_id, related_object_ids in main_candidates_results.items()
        }

    def __str__(self) -> str:
        return f"{self.main_item} is {self.type_id} {self.related_item}"

    def __repr__(self) -> str:
        return f"Relation({self.type_id}, {self.main_item.id}, {self.related_item.id})"

    # TODO: Check if we keep this
    # def __eq__(self, other: Any) -> bool:
    #     if not isinstance(other, Relation):
    #         return False
    #     return (
    #         self.type_id == other.type_id
    #         and self.main_item.id == other.main_item.id
    #         and self.related_item.id == other.related_item.id
    #     )

    # def __hash__(self) -> int:
    #     return hash((self.type_id, self.main_item, self.related_item))


class ReceptacleOfRelation(Relation):
    """
    A relation of the form "main_item is a receptacle of related_item".

    The inverse relation is ContainedInRelation.

    Attributes:
        main_item_id (ItemId): The id of the main item of the relation.
        related_item_id (ItemId): The id of the related item that is contained in the main item.
        inverse_relation (Relation): The inverse relation of the relation.
        main_item (TaskItem): The main item in the relation.
        related_item (TaskItem): The related item that is contained in the main item.

    """

    type_id = RelationTypeId.RECEPTACLE_OF
    inverse_relation_type_id = RelationTypeId.CONTAINED_IN
    candidate_required_prop = ReceptacleProp(True)

    def __init__(
        self,
        main_item_id: ItemId | str,
        related_item_id: ItemId | str,
        _inverse_relation: Relation | None = None,
    ) -> None:
        super().__init__(main_item_id, related_item_id, _inverse_relation)

    def _compute_inverse_relation_parameters(self) -> dict[str, RelationParam]:  # noqa: PLR6301
        """Return an empty dictionary since the inverse relation doesn't have parameters."""
        return {}

    def _extract_related_object_ids(  # noqa: PLR6301
        self,
        main_obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],  # noqa: ARG002
    ) -> list[SimObjId]:
        """Return the ids of the objects contained in the main object."""
        receptacle_object_ids = main_obj_metadata["receptacleObjectIds"]
        return receptacle_object_ids if receptacle_object_ids is not None else []

    def _are_candidates_compatible(  # noqa: PLR6301
        self,
        main_candidate_metadata: SimObjMetadata,
        related_candidate_metadata: SimObjMetadata,
    ) -> bool:
        """Return True if the main candidate is a compatible receptacle of the related candidate."""
        return (
            main_candidate_metadata["objectType"]
            in OBJECT_TYPES_DATA[related_candidate_metadata["objectType"]].compatible_receptacles
        )


# TODO: Add IsPickedUp auxiliary property
class ContainedInRelation(Relation):
    """
    A relation of the form "main_item is_contained_in related_item".

    The inverse relation is ReceptacleOfRelation.

    Attributes:
        main_item_id (ItemId): The id of the main item of the relation.
        related_item_id (ItemId): The id of the related item that contains the main item.
        inverse_relation (Relation): The inverse relation of the relation.
        main_item (TaskItem): The main item in the relation.
        related_item (TaskItem): The related item that contains the main item.

    # TODO: Finish docstring.
    """

    type_id = RelationTypeId.CONTAINED_IN
    inverse_relation_type_id = RelationTypeId.RECEPTACLE_OF
    candidate_required_prop = PickupableProp(True)

    def __init__(
        self,
        main_item_id: ItemId | str,
        related_item_id: ItemId | str,
        _inverse_relation: Relation | None = None,
    ) -> None:
        super().__init__(main_item_id, related_item_id, _inverse_relation)

    def _compute_inverse_relation_parameters(self) -> dict[str, RelationParam]:  # noqa: PLR6301
        """Return an empty dictionary since the inverse relation doesn't have parameters."""
        return {}

    def _extract_related_object_ids(  # noqa: PLR6301
        self,
        main_obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],  # noqa: ARG002
    ) -> list[SimObjId]:
        """Return the ids of the objects containing the main object."""
        parent_receptacles = main_obj_metadata["parentReceptacles"]
        return parent_receptacles if parent_receptacles is not None else []

    def _are_candidates_compatible(  # noqa: PLR6301
        self,
        main_candidate_metadata: SimObjMetadata,
        related_candidate_metadata: SimObjMetadata,
    ) -> bool:
        """Return True if the related candidate is a compatible receptacle of the main candidate."""
        return (
            related_candidate_metadata["objectType"]
            in OBJECT_TYPES_DATA[main_candidate_metadata["objectType"]].compatible_receptacles
        )


# TODO: Add IsPickedUp auxiliary property
class CloseToRelation(Relation):
    """
    A relation of the form "main_item is close to related_item".

    The inverse relation is itself.

    Attributes:
        distance (float): The distance between the main item and the related item.
        main_item_id (ItemId): The id of the main item of the relation.
        related_item_id (ItemId): The id of the related item that is close to the main item.
        inverse_relation (Relation): The inverse relation of the relation.
        main_item (TaskItem): The main item in the relation.
        related_item (TaskItem): The related item that is close to the main item.

    """

    type_id = RelationTypeId.CLOSE_TO
    inverse_relation_type_id = RelationTypeId.CLOSE_TO

    def __init__(
        self,
        distance: float,
        main_item_id: ItemId | str,
        related_item_id: ItemId | str,
        _inverse_relation: Relation | None = None,
    ) -> None:
        super().__init__(main_item_id, related_item_id, _inverse_relation)
        self.distance = distance

    def _compute_inverse_relation_parameters(self) -> dict[str, float]:
        """Return a dictionary with the same parameters since the relation is symmetrical."""
        return {"distance": self.distance}

    def _extract_related_object_ids(
        self,
        main_obj_metadata: SimObjMetadata,
        scene_objects_dict: dict[SimObjId, SimObjMetadata],
    ) -> list[SimObjId]:
        """Return the ids of the objects close to the main object."""
        close_object_ids = []
        for scene_object_id, scene_object_metadata in scene_objects_dict.items():
            if (
                scene_object_id != main_obj_metadata["objectId"]
                and compute_objects_2d_distance(main_obj_metadata, scene_object_metadata) <= self.distance
            ):
                close_object_ids.append(scene_object_id)

        return close_object_ids

    def _are_candidates_compatible(  # noqa: PLR6301
        self,
        main_candidate_metadata: SimObjMetadata,
        related_candidate_metadata: SimObjMetadata,
    ) -> bool:
        """Return True if at least one of the candidates is pickupable or moveable."""
        return any(
            candidate_metadata[SimObjFixedProp.PICKUPABLE] or candidate_metadata[SimObjFixedProp.MOVEABLE]
            for candidate_metadata in [main_candidate_metadata, related_candidate_metadata]
        )


## %% === Mappings ===
relation_type_id_to_relation: dict[RelationTypeId, type[Relation]]
relation_type_id_to_relation = {
    RelationTypeId.RECEPTACLE_OF: ReceptacleOfRelation,
    RelationTypeId.CONTAINED_IN: ContainedInRelation,
    RelationTypeId.CLOSE_TO: CloseToRelation,
}
