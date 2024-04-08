"""
Task relations in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_ai2thor.envs.sim_objects import SimObjFixedProp, SimObjId, SimObjMetadata
from rl_ai2thor.envs.tasks.items import SingleValuePSF
from rl_ai2thor.utils.ai2thor_utils import compute_objects_2d_distance

if TYPE_CHECKING:
    from rl_ai2thor.envs.tasks.items import PropSatFunction, PropValue, TaskItem


# %% === Enums ===
# TODO: Add more relations
class RelationTypeId(StrEnum):
    """Relations between items."""

    RECEPTACLE_OF = "receptacle_of"
    CONTAINED_IN = "contained_in"
    CLOSE_TO = "close_to"


type RelationParam = int | float | bool


# %% === Relations ===
@dataclass(frozen=True, repr=False, eq=False)
class Relation(ABC):
    """
    A relation between two items in the definition of a task.

    Attributes:
        main_item (TaskItem): The main item in the relation.
    related_item (TaskItem): The related item to which the main item is related.
    """

    main_item: TaskItem
    related_item: TaskItem
    type_id: RelationTypeId
    inverse_relation_type_id: RelationTypeId
    candidate_required_prop: SimObjFixedProp | None = None
    candidate_required_prop_sat_function: PropSatFunction | None = None

    @abstractmethod
    def _compute_inverse_relation_parameters(self) -> dict[str, RelationParam]:
        """
        Return the parameters of the inverse relation.

        Doesn't return the main and related items of the inverse relation since they don't depend on the relation.
        The output of this method is used to instantiate the inverse relation in create_inverse_relation().

        Returns:
            inverse_relation_parameters (dict[str, RelationParam]): The parameters of the inverse relation.
        """

    def create_inverse_relation(self) -> Relation:
        """
        Return the inverse relation.

        Returns:
            inverse_relation (Relation): The inverse relation.
        """
        inverse_relation_parameters = self._compute_inverse_relation_parameters()
        inverse_relation = relation_type_id_to_relation[self.inverse_relation_type_id](
            main_item=self.related_item,
            related_item=self.main_item,
            **inverse_relation_parameters,
        )
        return inverse_relation

    @abstractmethod
    def _extract_related_object_ids(
        self, main_obj_metadata: SimObjMetadata, scene_objects_dict: dict[SimObjId, SimObjMetadata]
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
        self, main_obj_metadata: SimObjMetadata, scene_objects_dict: dict[SimObjId, SimObjMetadata]
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

    def get_satisfying_related_object_ids(
        self, main_obj_metadata: SimObjMetadata, scene_objects_dict: dict[SimObjId, SimObjMetadata]
    ) -> set[SimObjId]:
        """
        Return related item's candidate's ids that satisfy the relation with the given main object.

        Args:
            main_obj_metadata (SimObjMetadata): The metadata of the main object.
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping the id
            of the objects in the scene to their metadata.

        Returns:
            set[SimObjId]: The ids of the related item's candidate's that satisfy the relation.
        """
        # TODO: Delete try-except block
        try:
            satisfying_related_object_ids = {
                related_object_id
                for related_object_id in self._extract_related_object_ids(main_obj_metadata, scene_objects_dict)
                if related_object_id in self.related_item.candidate_ids
            }
        except TypeError:
            print(f"main_obj_metadata: {main_obj_metadata}")
        return satisfying_related_object_ids

    def __str__(self) -> str:
        return f"{self.main_item} is {self.type_id} {self.related_item}"

    def __repr__(self) -> str:
        return f"Relation({self.type_id}, {self.main_item.id}, {self.related_item.id})"

    # TODO: Check if we keep this
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Relation):
            return False
        return (
            self.type_id == other.type_id
            and self.main_item.id == other.main_item.id
            and self.related_item.id == other.related_item.id
        )

    def __hash__(self) -> int:
        return hash((self.type_id, self.main_item, self.related_item))


@dataclass(frozen=True, repr=False, eq=False)
class ReceptacleOfRelation(Relation):
    """
    A relation of the form "main_item is a receptacle of related_item".

    The inverse relation is ContainedInRelation.

    Attributes:
        main_item (TaskItem): The main item in the relation.
        related_item (TaskItem): The related item that is contained in the main item.

    """

    type_id: RelationTypeId = field(default=RelationTypeId.RECEPTACLE_OF, init=False)
    inverse_relation_type_id: RelationTypeId = field(default=RelationTypeId.CONTAINED_IN, init=False)
    candidate_required_prop: SimObjFixedProp = field(default=SimObjFixedProp.RECEPTACLE, init=False)
    candidate_required_prop_sat_function: PropSatFunction = field(default=SingleValuePSF(True), init=False)
    main_item: TaskItem
    related_item: TaskItem

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


@dataclass(frozen=True, repr=False, eq=False)
class ContainedInRelation(Relation):
    """
    A relation of the form "main_item is_contained_in related_item".

    The inverse relation is ReceptacleOfRelation.

    Attributes:
        main_item (TaskItem): The main item in the relation.
        related_item (TaskItem): The related item that contains the main item.

    # TODO: Finish docstring.
    """

    type_id: RelationTypeId = field(default=RelationTypeId.CONTAINED_IN, init=False)
    inverse_relation_type_id: RelationTypeId = field(default=RelationTypeId.RECEPTACLE_OF, init=False)
    candidate_required_prop: SimObjFixedProp = field(default=SimObjFixedProp.PICKUPABLE, init=False)
    candidate_required_prop_sat_function: PropSatFunction = field(default=SingleValuePSF(True), init=False)
    main_item: TaskItem
    related_item: TaskItem

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


@dataclass(frozen=True, repr=False, eq=False)
class CloseToRelation(Relation):
    """
    A relation of the form "main_item is close to related_item".

    The inverse relation is itself.

    Attributes:
        main_item (TaskItem): The main item in the relation.
        related_item (TaskItem): The related item that is close to the main item.
        distance (float): The distance between the main item and the related item.

    """

    type_id: RelationTypeId = field(default=RelationTypeId.CLOSE_TO, init=False)
    inverse_relation_type_id: RelationTypeId = field(default=RelationTypeId.CLOSE_TO, init=False)
    candidate_required_prop = None
    candidate_required_prop_sat_function = None
    main_item: TaskItem
    related_item: TaskItem
    distance: float = 1.0

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


# %% === Mappings ===
relation_type_id_to_relation = {
    RelationTypeId.RECEPTACLE_OF: ReceptacleOfRelation,
    RelationTypeId.CONTAINED_IN: ContainedInRelation,
    RelationTypeId.CLOSE_TO: CloseToRelation,
}
