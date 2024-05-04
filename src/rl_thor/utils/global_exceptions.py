"""
Custom exceptions for the RL-THOR environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_thor.envs.tasks.relations import RelationTypeId


# === Relation Exceptions ===
class DuplicateRelationsError[T](Exception):
    """
    Exception raised when the two relations of the same type involving the same main and related items are detected.

    Such case is not allowed because combining relations and keeping only the most restrictive is
    not supported yet. In particular, one should not add one relation and its opposite relation
    (e.g. `receptacle is_receptacle_of object` and `object is_contained_in receptacle`) because it
    is done automatically when instantiating the task.
    """

    def __init__(
        self,
        relation_type_id: RelationTypeId,
        main_item_id: T,
        related_item_id: T,
    ) -> None:
        """
        Initialize the exception.

        Args:
            relation_type_id (RelationTypeId): The type of relation.
            main_item_id (ItemId): The id of the main item.
            related_item_id (ItemId): The id of the related item.
            task_description_dict (TaskDict[T]): Full task description dictionary.
        """
        self.relation_type_id = relation_type_id
        self.main_item_id = main_item_id
        self.related_item_id = related_item_id

        super().__init__(
            f"Two relations of the same type involving the same main and related items are detected: "
            f"{relation_type_id}({main_item_id}, {related_item_id})"
        )
