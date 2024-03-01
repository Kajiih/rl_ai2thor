"""
Task relations in AI2THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_ai2thor.envs.sim_objects import SimObjectType, SimObjFixedProp, SimObjMetadata

if TYPE_CHECKING:
    from rl_ai2thor.envs.tasks.items import TaskItem


# %% === Enums ===
# TODO: Add more relations
class RelationTypeId(StrEnum):
    """Relations between items."""

    RECEPTACLE_OF = "receptacle_of"
    CONTAINED_IN = "contained_in"
    # CLOSE_TO = "close_to"


# %% === Relations ===
# TODO: Add support to parameterize the relations (e.g. distance in CLOSE_TO)
class Relation(ABC):
    """A relation between two items in the definition of a task."""

    type_id: RelationTypeId
    inverse_relation_type_id: RelationTypeId
    candidate_required_prop: SimObjFixedProp | None = None
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
    def _extract_related_object_ids(self, main_obj_metadata: dict[SimObjMetadata, Any]) -> list[SimObjectType]:
        """Return the list of the ids of the main object's related objects according to the relation."""

    def is_semi_satisfied(self, main_obj_metadata: dict[SimObjMetadata, Any]) -> bool:
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

    def get_satisfying_related_object_ids(self, main_obj_metadata: dict[SimObjMetadata, Any]) -> set[SimObjectType]:
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
    candidate_required_prop = SimObjFixedProp.RECEPTACLE
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
    def _extract_related_object_ids(main_obj_metadata: dict[SimObjMetadata, Any]) -> list[SimObjectType]:
        """Return the ids of the objects contained in the main object."""
        return main_obj_metadata["receptacleObjectIds"]


class ContainedInRelation(Relation):
    """
    A relation of the form "main_item is_contained_in related_item".

    The inverse relation is ReceptacleOfRelation.
    """

    type_id = RelationTypeId.CONTAINED_IN
    inverse_relation_type_id = RelationTypeId.RECEPTACLE_OF
    candidate_required_prop = SimObjFixedProp.PICKUPABLE
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
    def _extract_related_object_ids(main_obj_metadata: dict[SimObjMetadata, Any]) -> list[SimObjectType]:
        """Return the ids of the objects containing the main object."""
        return main_obj_metadata["parentReceptacles"]


# %% === Mappings ===
relation_type_id_to_relation = {
    RelationTypeId.RECEPTACLE_OF: ReceptacleOfRelation,
    RelationTypeId.CONTAINED_IN: ContainedInRelation,
}
