"""
Tasks in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

# %% === Imports ===
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_ai2thor.envs.actions import Ai2thorAction
from rl_ai2thor.envs.reward import BaseRewardHandler
from rl_ai2thor.envs.sim_objects import (
    LIGHT_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjVariableProp,
)
from rl_ai2thor.envs.tasks.item_prop import obj_prop_id_to_item_prop
from rl_ai2thor.envs.tasks.item_prop_interface import (
    ItemProp,
    ItemPropValue,
    MultiValuePSF,
    PropSatFunction,
    SingleValuePSF,
    TemperatureValue,
)
from rl_ai2thor.envs.tasks.items import (
    Assignment,
    AuxItem,
    CandidateId,
    ItemId,
    ItemOverlapClass,
    TaskItem,
)
from rl_ai2thor.envs.tasks.relations import (
    RelationParam,
    RelationTypeId,
    relation_type_id_to_relation,
)
from rl_ai2thor.utils.global_exceptions import DuplicateRelationsError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ai2thor.controller import Controller
    from ai2thor.server import Event

    from rl_ai2thor.envs.scenes import SceneId
    from rl_ai2thor.envs.sim_objects import SimObjId, SimObjMetadata, SimObjProp
    from rl_ai2thor.envs.tasks.relations import Relation


# %% === Reward handlers ===
# TODO: Add more options
# TODO: Make the rewards more customizable
class GraphTaskRewardHandler(BaseRewardHandler):
    """
    Reward handler for graph tasks.

    TODO: Finish docstring
    """

    def __init__(self, task: GraphTask) -> None:
        """
        Initialize the reward handler.

        Args:
            task (GraphTask): Task to calculate rewards for.
        """
        self.task = task
        self.last_step_advancement: float | int = 0

    def get_reward(self, event: Event) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the reward, task completion and additional information about the task for the given event.

        Args:
            event (Event): Event to calculate the reward for.

        Returns:
            reward (float): Reward for the event.
            terminated (bool | None): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        if not event.metadata["lastActionSuccess"]:
            return 0.0, False, {}
        task_advancement, task_completion, info = self.task.compute_task_advancement(event)
        reward = task_advancement - self.last_step_advancement
        self.last_step_advancement = task_advancement

        if task_completion:
            print("Task completed!!")
            reward += 10

        return reward, task_completion, info

    def reset(self, controller: Controller) -> tuple[bool, bool, dict[str, Any]]:
        """
        Reset the reward handler.

        The reset is considered not successful if the task and the scene are incompatible.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the task is successfully reset.
            terminated (bool): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        # Reset the task
        reset_successful, task_advancement, task_completion, info = self.task.reset(controller)
        # Initialize the last step advancement
        self.last_step_advancement = task_advancement

        return reset_successful, task_completion, info


# %% === Tasks ===
class BaseTask(ABC):
    """Base class for tasks."""

    _reward_handler_type: type[BaseRewardHandler]

    @abstractmethod
    def reset(self, controller: Controller) -> tuple[bool, float, bool, dict[str, Any]]:
        """
        Reset and initialize the task and the controller.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the task is successfully reset.
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement at the beginning of the episode.
        """

    @abstractmethod
    def compute_task_advancement(
        self,
        event: Event,
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,
    ) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the task advancement and whether the task is completed.

        Args:
            event (Event): Event corresponding to the state of the scene.
            scene_objects_dict (dict[SimObjId, SimObjMetadata], optional): Dictionary
                mapping object ids to their metadata to avoid recomputing it. Defaults to None.

        Returns:
            task_advancement (float): Task advancement.
            is_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """

    def get_reward_handler(self) -> BaseRewardHandler:
        """Return the reward handler for the task."""
        return self._reward_handler_type(self)

    @abstractmethod
    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """


class UndefinedTask(BaseTask):
    """Undefined task that is never completed and has no advancement."""

    def reset(self, controller: Controller) -> tuple[float, bool, dict[str, Any]]:
        """Reset and initialize the task and the controller."""
        raise NotImplementedError("Undefined task")

    def compute_task_advancement(
        self,
        event: Event,
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Return the task advancement and whether the task is completed."""
        raise NotImplementedError("Undefined task")

    def text_description(self) -> str:
        """Return a text description of the task."""
        raise NotImplementedError("Undefined task")


type TaskArg = ItemPropValue | int
type RelationsDict = dict[ItemId, dict[RelationTypeId, dict[str, RelationParam]]]
type PropertiesDict = dict[SimObjProp, PropSatFunction]
type TaskDict = dict[ItemId | str, TaskItemData]


@dataclass
class TaskItemData:
    """Description of a task item."""

    properties: PropertiesDict = field(default_factory=dict)
    relations: RelationsDict = field(default_factory=dict)


# TODO: Add support for weighted properties and relations
# TODO: Add support for agent properties
class GraphTask(BaseTask):
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
    task_description_dict = {
            "receptacle_plate": {
                "properties": {"objectType": "Plate"},
            },
            "placed_apple": {
                "properties": {"objectType": "Apple"},
                "relations": {"receptacle_plate": ["contained_in"]},
            },
        }

    This task contains 2 items defined by their unique identifiers in the description dictionary;
    "receptacle_plate" and "placed_apple". Both items have a required type (others available
    properties are available in ObjFixedPropId and ObjVariablePropId enums) and placed_apple is
    related to receptacle_plate one relation by the "contained_in relation (other relations
    available in RelationTypeId enum).

    Note: Inverse relation
    Inverse relations are automatically added to the graph, so you should not add them in the task
    description dictionary because it will raise an error.

    Attributes:
        task_description_dict (TaskDict): Dictionary describing the items and their
            properties and relations.
        items (List[TaskItem]): List of items of the task.
        items_by_id (Dict[ItemId, TaskItem]): Dictionary mapping item ids to their corresponding TaskItem.
        overlap_classes (List[ItemOverlapClass]): List of overlap classes containing items
            with overlapping candidates.
        auxiliary_items (FrozenSet[TaskItem]): Set of items that are not part of the task
            but that are necessary for certain item's properties or relations to be satisfied.
        maximum_advancement (int): Maximum task advancement possible for the task.

    Methods:
        reset(self, event: Event) -> tuple[float, bool, dict[str, Any]]:
            Reset the task with the information of the event.
        get_task_advancement(self, event: Event) -> tuple[float, bool, dict[str, Any]]:
            Return the task advancement and whether the task is completed.
        full_initialize_items_and_relations_from_dict(task_description_dict: TaskDict) -> list[TaskItem]
            Create and initialize TaskItems for the graph task.
    """

    _reward_handler_type = GraphTaskRewardHandler

    def __init__(
        self,
        task_description_dict: TaskDict,
    ) -> None:
        """
        Initialize the task graph as defined in the task description dictionary.

        Args:
            task_description_dict (TaskDict): Dictionary describing the items and their
                properties and relations.
        """
        self.task_description_dict = task_description_dict
        self.items = self.full_initialize_items_and_relations_from_dict(task_description_dict)
        self.items_by_id = {item.id: item for item in self.items}

        # === Type annotations ===
        self.task_description_dict: TaskDict
        self.items: list[TaskItem]
        self.items_by_id: dict[ItemId, TaskItem]
        self.overlap_classes: list[ItemOverlapClass]
        self.auxiliary_items: frozenset[AuxItem]
        self.maximum_advancement: int

    def reset(self, controller: Controller) -> tuple[bool, int, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        Initialize the candidates of the items with the objects
        in the scene and compute the overlap classes.

        Valid assignments are assignments where each item is associated with a candidate
        that has all correct candidate_required_properties (without taking into account the
        relations between the items) and compatible assignments are valid assignment where the
        candidates are compatible when taking into account the relations between the items.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the task is successfully reset.
            initial_task_advancement (int): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        event: Event = controller.last_event  # type: ignore
        scene_objects_dict: dict[SimObjId, SimObjMetadata] = {obj["objectId"]: obj for obj in event.metadata["objects"]}

        # Initialize the candidates of the items
        for item in self.items:
            item.candidates_data = item.instantiate_candidate_data(scene_objects_dict)
            if not item.candidate_ids:
                print(f"No candidate found for item {item.id}")
                return False, 0, False, {}

        # Initialize the auxiliary items
        self.auxiliary_items = frozenset().union(
            *(auxiliary_items for item in self.items for auxiliary_items in item.props_auxiliary_items.values())
        )
        for auxiliary_item in self.auxiliary_items:
            auxiliary_item.relations = frozenset()
            auxiliary_item.candidates_data = auxiliary_item.instantiate_candidate_data(scene_objects_dict)
            if not auxiliary_item.candidate_ids:
                print(f"No candidate found for auxiliary item {auxiliary_item.id}")
                return False, 0, False, {}

        # Make sure that there is at least one relation compatible assignment
        compatible_assignments = self._compute_compatible_assignments(scene_objects_dict)
        if not compatible_assignments:
            print("No compatible assignment found")
            return False, 0, False, {}

        # Keep only candidates that are in at least one compatible assignment
        for item in self.items:
            final_candidate_ids = {CandidateId(assignment[item]) for assignment in compatible_assignments}
            for candidate_id in item.candidates_data:
                if candidate_id not in final_candidate_ids:
                    del item.candidates_data[candidate_id]

        # Initialize overlap classes and keep only assignments that are part of one of the global
        # compatible assignments
        self.overlap_classes = self._compute_overlap_classes(self.items)
        # TODO: Check if this is necessary
        for overlap_class in self.overlap_classes:
            overlap_class.prune_assignments(compatible_assignments)

        # Compute max task advancement = Total number of properties and relations of the items
        # TODO: Make it compatible with weighted properties and relations
        self.maximum_advancement = sum(item.maximum_advancement for item in self.items)

        return True, *self.compute_task_advancement(event, scene_objects_dict)

    def _compute_compatible_assignments(self, scene_objects_dict: dict[SimObjId, SimObjMetadata]) -> list[Assignment]:
        """
        Compute the compatible assignments of the items in the scene.

        Args:
            scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary mapping object ids to
                their metadata.

        Returns:
            compatible_assignments (list[Assignment]): List of compatible assignments.
        """
        temp_overlap_classes = self._compute_overlap_classes(self.items)
        if not all(overlap_class.valid_assignments for overlap_class in temp_overlap_classes):
            return []

        valid_assignments_product = itertools.product(
            *(overlap_class.valid_assignments for overlap_class in temp_overlap_classes)
        )
        compatible_assignments = []
        for assignment_product in valid_assignments_product:
            global_assignment: Assignment = {
                item: candidate_id
                for overlap_class_assignment in assignment_product
                for item, candidate_id in overlap_class_assignment.items()
            }
            incompatible_assignment = False
            for i in range(len(self.items)):
                main_item = self.items[i]
                main_candidate_metadata = scene_objects_dict[global_assignment[main_item]]
                for related_item in self.items[i + 1 :]:
                    related_candidate_metadata = scene_objects_dict[global_assignment[related_item]]

                    relations = main_item.organized_relations.get(related_item.id, {})
                    for relation in relations.values():
                        if not relation._are_candidates_compatible(
                            main_candidate_metadata, related_candidate_metadata
                        ):  # TODO: Use the cached version
                            incompatible_assignment = True
                            break
                    if incompatible_assignment:
                        break
                if incompatible_assignment:
                    break
            if not incompatible_assignment:
                compatible_assignments.append(global_assignment)

        return compatible_assignments

    @staticmethod
    def _compute_overlap_classes(items: list[TaskItem]) -> list[ItemOverlapClass]:
        """
        Compute the overlap classes of the items in the scene.

        Items must have their candidates initialized before calling this method.

        Args:
            items (list[TaskItem]): List of items.

        Returns:
            overlap_classes (list[ItemOverlapClass]): List of overlap classes.
        """
        overlap_classes: dict[int, dict[str, Any]] = {}
        for item in items:
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

        return [
            ItemOverlapClass(
                items=overlap_class["items"],
                candidate_ids=list(overlap_class["candidate_ids"]),
            )
            for overlap_class in overlap_classes.values()
        ]

    # TODO: Add trying only the top k interesting assignments according to the maximum possible score (need to order the list of interesting candidates then the list of interesting assignments for each overlap class)
    def compute_task_advancement(
        self,
        event: Event,
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,
    ) -> tuple[int, bool, dict[str, Any]]:
        """
        Return the task advancement and whether the task is completed.

        To compute the task advancement, we consider every interesting global assignment
        of objects to the items. A global assignment is a dictionary mapping each item of
        the task to an object in the scene (as opposed to a an overlap class assignment).
        To construct the set of global assignments, we take the cartesian product of the
        assignments of the overlap classes. Interesting global assignments are the ones
        constructed with only interesting overlap class assignments.

        For a given global assignment, the task advancement is the sum of the property
        scores of the assigned objects for each item and the sum of their relations scores
        for relations that have a satisfying object assigned to the related item (i.e. we
        consider strictly satisfied relations and not semi satisfied relations).

        Args:
            event (Event): Event corresponding to the state of the scene.
            scene_objects_dict (dict[SimObjId, SimObjMetadata], optional): Dictionary
                mapping object ids to their metadata to avoid recomputing it. Defaults to None.

        Returns:
            task_advancement (int): Task advancement.
            is_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # TODO: Update this function
        # Compute the interesting assignments for each overlap class and the results and scores of each candidate for each item
        if scene_objects_dict is None:
            scene_objects_dict = {obj["objectId"]: obj for obj in event.metadata["objects"]}

        overlap_classes_assignments = [
            overlap_class.compute_interesting_assignments(scene_objects_dict) for overlap_class in self.overlap_classes
        ]

        # Construct a generator of the cartesian product of the interesting assignments
        assignment_products = itertools.product(*overlap_classes_assignments)

        max_task_advancement = -1
        best_assignment = {}
        is_terminated = False

        # Compute the task advancement for each global assignment
        for assignment_product in assignment_products:
            # Merge the assignments of the overlap classes
            global_assignment: Assignment = {
                item: obj_id
                for overlap_class_assignment in assignment_product
                for item, obj_id in overlap_class_assignment.items()
            }
            assignment_advancement = self.compute_assignment_advancement(global_assignment)

            if assignment_advancement > max_task_advancement:
                max_task_advancement = assignment_advancement
                best_assignment = global_assignment
                if max_task_advancement == self.maximum_advancement:
                    is_terminated = True
                    break

        # Add info about the task advancement
        info = {
            # Add best assignment, mapping between item ids and the assigned object ids
            "best_assignment": {item.id: candidate_id for item, candidate_id in best_assignment.items()},
            "candidate_data": {item.id: item.candidates_data[best_assignment[item]] for item in self.items},
            "task_advancement": max_task_advancement,
        }
        # TODO: Add other info

        return max_task_advancement, is_terminated, info

    @staticmethod
    def compute_assignment_advancement(global_assignment: Assignment) -> int:
        """
        Compute the task advancement for a given assignment.

        Args:
            global_assignment (Assignment): Assignment of objects for all items of the task.

        Returns:
            task_advancement (int): Task advancement.
        """
        task_advancement = 0

        # Add property advancement
        task_advancement += sum(
            item.candidates_data[candidate_id].property_advancement for item, candidate_id in global_assignment.items()
        )

        # Add relation advancement for the given assignment
        for item, candidate_id in global_assignment.items():
            for relation in item.relations:
                main_candidate_data = item.candidates_data[candidate_id]
                related_candidate_id = global_assignment[relation.related_item]
                task_advancement += main_candidate_data.compute_relation_advancement_for_related_candidate(
                    relation, related_candidate_id
                )

        return task_advancement

    # TODO: Add support for overriding relations and keep the most restrictive one
    @staticmethod
    def full_initialize_items_and_relations_from_dict(
        task_description_dict: TaskDict,
    ) -> list[TaskItem]:
        """
        Create and initialize TaskItems for the graph task.

        TaskItems are created as defined in the task description
        dictionary representing the items and their properties and relations.
        The items fully initialized with their relations and the inverse
        relations are also added.

        Args:
            task_description_dict (TaskDict): Dictionary describing the items and their properties
                and relations.

        Returns:
            items (list[TaskItem]): List of the items of the task.
        """
        # === Instantiate relations ===
        organized_relations: dict[ItemId | str, dict[ItemId | str, dict[RelationTypeId, Relation]]]
        organized_relations = {main_item_id: {} for main_item_id in task_description_dict}

        for main_item_id, main_item_data in task_description_dict.items():
            for related_item_id, relations_dict in main_item_data.relations.items():
                if related_item_id not in organized_relations[main_item_id]:
                    organized_relations[main_item_id][related_item_id] = {}
                    organized_relations[related_item_id][main_item_id] = {}
                for relation_type_id, relation_parameters in relations_dict.items():
                    if relation_type_id in organized_relations[main_item_id][related_item_id]:
                        raise DuplicateRelationsError(
                            relation_type_id,
                            main_item_id,
                            related_item_id,
                        )

                    # === Add direct relations ===
                    relation = relation_type_id_to_relation[relation_type_id](
                        main_item_id=main_item_id,
                        related_item_id=related_item_id,
                        _inverse_relation=None,
                        **relation_parameters,
                    )
                    organized_relations[main_item_id][related_item_id][relation_type_id] = relation

                    # === Add inverse relations ===
                    inverse_relation_type_id = relation.inverse_relation_type_id
                    if inverse_relation_type_id in organized_relations[related_item_id][main_item_id]:
                        raise DuplicateRelationsError(
                            relation_type_id=inverse_relation_type_id,
                            main_item_id=related_item_id,
                            related_item_id=main_item_id,
                        )
                    organized_relations[related_item_id][main_item_id][inverse_relation_type_id] = (
                        relation.inverse_relation
                    )

        # === Instantiate items ===
        relations_by_main_item_id = {
            main_item_id: {
                relation
                for relations_dict in organized_relations[main_item_id].values()
                for relation in relations_dict.values()
            }
            for main_item_id in organized_relations
        }
        properties_by_item_id = {
            item_id: {
                obj_prop_id_to_item_prop[prop](prop_sat_function)
                for prop, prop_sat_function in item_data.properties.items()
            }
            for item_id, item_data in task_description_dict.items()
        }
        items = [
            TaskItem(
                item_id,
                properties_by_item_id[item_id],
                relations_by_main_item_id[item_id],
            )
            for item_id in task_description_dict
        ]
        # main_item and related_item attributes of the relations are automatically set by the
        # TaskItem constructor

        return items

    # TODO: Improve this
    def __repr__(self) -> str:
        nb_items = len(self.items)
        nb_relations = sum(len(item.relations) for item in self.items)
        return f"GraphTask({nb_items} items, {nb_relations} relations)"


# %% === Task Blueprints ===
@dataclass
class TaskBlueprint:
    """Blueprint containing the information to instantiate a task in the environment."""

    task_type: type[BaseTask]
    scenes: set[SceneId]
    task_args: Mapping[str, TaskArg] = field(default_factory=dict)


# %% == Alfred tasks ==
class PlaceNSameIn(GraphTask):
    """
    Task for placing n objects of the same type in a receptacle.

    This is equivalent to the pick_two_obj_and_place task from Alfred with n=2 and
    pick_and_place_simple with n=1.
    """

    def __init__(self, placed_object_type: SimObjectType, receptacle_type: SimObjectType, n: int = 1) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.
            n (int): The number of objects to place.
        """
        self.placed_object_type = placed_object_type
        self.receptacle_type = receptacle_type
        self.n = n

        task_description_dict = self._create_task_description_dict(placed_object_type, receptacle_type, n)

        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(
        cls, placed_object_type: SimObjectType, receptacle_type: SimObjectType, n: int = 1
    ) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.
            n (int): The number of objects to place.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        receptacle_id = ItemId("receptacle")
        task_description_dict: TaskDict = {
            receptacle_id: TaskItemData(properties={SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(receptacle_type)}),
        }
        for i in range(n):
            task_description_dict[ItemId(f"placed_object_{i}")] = TaskItemData(
                properties={SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(placed_object_type)},
                relations={receptacle_id: {RelationTypeId.CONTAINED_IN: {}}},
            )

        return task_description_dict

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.n} {self.placed_object_type} in {self.receptacle_type}"


class PlaceNSameInSubclass(PlaceNSameIn, ABC):
    """Abstract subclass of PlaceNSameIn for tasks with a specific number of objects to place."""

    n: int

    def __init__(
        self,
        placed_object_type: SimObjectType,
        receptacle_type: SimObjectType,
    ) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.
        """
        super().__init__(placed_object_type, receptacle_type, self.n)

        # Replace the instance attribute with the class attribute
        del self.n


class PlaceIn(PlaceNSameInSubclass):
    """
    Task for placing a given object in a given receptacle.

    This is equivalent to the pick_and_place_simple task from Alfred.
    """

    n = 1


class PlaceWithMoveableRecepIn(GraphTask):
    """
    Task for placing an given object with a given moveable receptacle in a given receptacle.

    This is equivalent to the pick_and_place_with_movable_recep task from Alfred.
    """

    def __init__(
        self,
        placed_object_type: SimObjectType,
        pickupable_receptacle_type: SimObjectType,
        receptacle_type: SimObjectType,
    ) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            pickupable_receptacle_type (SimObjectType): The type of pickupable receptacle to place the object in.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.
        """
        self.placed_object_type = placed_object_type
        self.pickupable_receptacle_type = pickupable_receptacle_type
        self.receptacle_type = receptacle_type

        task_description_dict = self._create_task_description_dict(
            placed_object_type, pickupable_receptacle_type, receptacle_type
        )

        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(
        cls,
        placed_object_type: SimObjectType,
        pickupable_receptacle_type: SimObjectType,
        receptacle_type: SimObjectType,
    ) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            pickupable_receptacle_type (SimObjectType): The type of pickupable receptacle to place the object in.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        receptacle_id = ItemId("receptacle")
        pickupable_receptacle_id = ItemId("pickupable_receptacle")
        placed_object_id = ItemId("placed_object")
        return {
            receptacle_id: TaskItemData(
                properties={SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(receptacle_type)},
            ),
            pickupable_receptacle_id: TaskItemData(
                properties={SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(pickupable_receptacle_type)},
                relations={receptacle_id: {RelationTypeId.CONTAINED_IN: {}}},
            ),
            placed_object_id: TaskItemData(
                properties={SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(placed_object_type)},
                relations={pickupable_receptacle_id: {RelationTypeId.CONTAINED_IN: {}}},
            ),
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.placed_object_type} in {self.pickupable_receptacle_type} in {self.receptacle_type}"


class PlaceCleanedIn(PlaceIn):
    """
    Task for placing a given cleaned object in a given receptacle.

    This is equivalent to the pick_clean_then_place_in_recep task from Alfred.

    All instance of placed_object_type are made dirty during the reset of the task.
    """

    @classmethod
    def _create_task_description_dict(
        cls,
        placed_object_type: SimObjectType,
        receptacle_type: SimObjectType,
        n: int,
    ) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.
            n (int): The number of objects to place.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict(placed_object_type, receptacle_type)
        for i in range(n):
            task_description_dict[ItemId(f"placed_object_{i}")].properties[SimObjVariableProp.IS_DIRTY] = (
                SingleValuePSF(False)
            )

        return task_description_dict

    def reset(self, controller: Controller) -> tuple[bool, float, bool, dict[str, Any]]:
        """
        Make all instances of placed_object_type dirty.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            if (
                obj_metadata[SimObjFixedProp.OBJECT_TYPE] == self.placed_object_type
                and not obj_metadata[SimObjVariableProp.IS_DIRTY]
            ):
                controller.step(
                    action=Ai2thorAction.DIRTY_OBJECT,
                    objectId=obj_metadata[SimObjFixedProp.OBJECT_ID],
                    forceAction=True,
                )

        return super().reset(controller)

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place cleaned {self.placed_object_type} in {self.receptacle_type}"


class PlaceHeatedIn(PlaceIn):
    """
    Task for placing a given heated object in a given receptacle.

    This is equivalent to the pick_heat_then_place_in_recep task from Alfred.

    All sim object start at room temperature so we don't need to do anything
    during the reset of the task.

    Args:
        placed_object_type (str): The type of object to place.
        receptacle_type (str): The type of receptacle to place the object in.
    """

    @classmethod
    def _create_task_description_dict(
        cls,
        placed_object_type: SimObjectType,
        receptacle_type: SimObjectType,
        n: int,
    ) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.
            n (int): The number of objects to place.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict(placed_object_type, receptacle_type)
        for i in range(n):
            task_description_dict[ItemId(f"placed_object_{i}")].properties[SimObjVariableProp.TEMPERATURE] = (
                SingleValuePSF(TemperatureValue.HOT)
            )

        return task_description_dict

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place heated {self.placed_object_type} in {self.receptacle_type}"


class PlaceCooledIn(PlaceIn):
    """
    Task for placing a given cooled object in a given receptacle.

    This is equivalent to the pick_cool_then_place_in_recep task from Alfred.

    All sim object start at room temperature so we don't need to do anything
    during the reset of the task.

    Args:
        placed_object_type (str): The type of object to place.
        receptacle_type (str): The type of receptacle to place the object in.
    """

    @classmethod
    def _create_task_description_dict(
        cls,
        placed_object_type: SimObjectType,
        receptacle_type: SimObjectType,
        n: int,
    ) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the object in.
            n (int): The number of objects to place.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict(placed_object_type, receptacle_type)
        for i in range(n):
            task_description_dict[ItemId(f"placed_object_{i}")].properties[SimObjVariableProp.TEMPERATURE] = (
                SingleValuePSF(TemperatureValue.COLD)
            )

        return task_description_dict

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place cooled {self.placed_object_type} in {self.receptacle_type}"


class LookInLight(GraphTask):
    """
    Task for looking at a given object in light.

    More precisely, the agent has have a toggled light source visible while holding the object to look at.

    This is equivalent to the look_at_obj_in_light task from Alfred.

    All light sources are switched off during the reset of the task.
    """

    def __init__(self, looked_at_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            looked_at_object_type (SimObjectType): The type of object to look at.
        """
        self.looked_at_object_type = looked_at_object_type

        task_description_dict = self._create_task_description_dict(looked_at_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, looked_at_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            looked_at_object_type (SimObjectType): The type of object to look at.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        light_source_id = ItemId("light_source")
        looked_at_object_id = ItemId("looked_at_object")
        return {
            light_source_id: TaskItemData(
                properties={
                    SimObjFixedProp.OBJECT_TYPE: MultiValuePSF(LIGHT_SOURCES),
                    SimObjVariableProp.IS_TOGGLED: SingleValuePSF(True),
                },
            ),
            looked_at_object_id: TaskItemData(
                properties={
                    SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(looked_at_object_type),
                    SimObjVariableProp.IS_PICKED_UP: SingleValuePSF(True),
                },
                relations={light_source_id: {RelationTypeId.CLOSE_TO: {"distance": 1.0}}},
            ),
        }

    def reset(self, controller: Controller) -> tuple[bool, float, bool, dict[str, Any]]:
        """
        Switch of all light sources in the scene.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            if (
                obj_metadata[SimObjFixedProp.OBJECT_TYPE] in LIGHT_SOURCES
                and obj_metadata[SimObjVariableProp.IS_TOGGLED]
            ):
                controller.step(
                    action=Ai2thorAction.TOGGLE_OBJECT_OFF,
                    objectId=obj_metadata[SimObjFixedProp.OBJECT_ID],
                    forceAction=True,
                )

        return super().reset(controller)

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Look at {self.looked_at_object_type} in light"


# %% === Custom tasks ===
class Pickup(GraphTask):
    """Task for picking up a given object."""

    def __init__(self, picked_up_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            picked_up_object_type (str): The type of object to pick up.
        """
        self.picked_up_object_type = picked_up_object_type

        task_description_dict = self._create_task_description_dict(picked_up_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, picked_up_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            picked_up_object_type (SimObjectType): The type of object to pick up.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("picked_up_object"): TaskItemData(
                properties={
                    SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(picked_up_object_type),
                    SimObjVariableProp.IS_PICKED_UP: SingleValuePSF(True),
                },
            )
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Pick up {self.picked_up_object_type}"


# TODO: Fix this because you can't load tasks without arguments
class OpenAny(GraphTask):
    """Task for opening any object."""

    def __init__(self) -> None:
        """Initialize the task."""
        task_description_dict = self._create_task_description_dict()
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("opened_object"): TaskItemData(
                properties={SimObjVariableProp.IS_OPEN: SingleValuePSF(True)},
            )
        }

    def text_description(self) -> str:  # noqa: PLR6301
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Open any object"


class Open(GraphTask):
    """Task for opening a given object."""

    def __init__(self, opened_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            opened_object_type (SimObjectType): The type of object to open.
        """
        self.opened_object_type = opened_object_type

        task_description_dict = self._create_task_description_dict(opened_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, opened_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            opened_object_type (SimObjectType): The type of object to open.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("opened_object"): TaskItemData(
                properties={
                    SimObjFixedProp.OBJECT_TYPE: SingleValuePSF(opened_object_type),
                    SimObjVariableProp.IS_OPEN: SingleValuePSF(True),
                },
            )
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Open {self.opened_object_type}"


# %% === Constants ===
class TaskType(StrEnum):
    """Enumeration of task types."""

    # === Alfred tasks ===
    PLACE_IN = "PlaceIn"
    PLACE_N_SAME_IN = "PlaceNSameIn"
    PLACE_WITH_MOVEABLE_RECEP_IN = "PlaceWithMoveableRecepIn"
    PLACE_CLEANED_IN = "PlaceCleanedIn"
    PLACE_HEATED_IN = "PlaceHeatedIn"
    PLACE_COOLED_IN = "PlaceCooledIn"
    LOOK_IN_LIGHT = "LookInLight"
    # === Custom tasks ===
    PICKUP = "Pickup"
    OPEN = "Open"


ALL_TASKS = {
    TaskType.PLACE_IN: PlaceIn,
    TaskType.PLACE_N_SAME_IN: PlaceNSameIn,
    TaskType.PLACE_WITH_MOVEABLE_RECEP_IN: PlaceWithMoveableRecepIn,
    TaskType.PLACE_CLEANED_IN: PlaceCleanedIn,
    TaskType.PLACE_HEATED_IN: PlaceHeatedIn,
    TaskType.PLACE_COOLED_IN: PlaceCooledIn,
    TaskType.LOOK_IN_LIGHT: LookInLight,
    TaskType.PICKUP: Pickup,
    TaskType.OPEN: Open,
}


# %% === Exceptions ===
class UnknownTaskTypeError(ValueError):
    """Exception raised for unknown task types in environment mode config."""

    def __init__(self, task_type: str) -> None:
        self.task_type = task_type
        super().__init__(
            f"Unknown task type '{task_type}' in environment mode config."
            f"Available tasks are {list(ALL_TASKS.keys())}."
            f"If you have defined a new task, make sure to add it to the ALL_TASKS dictionary of the envs.tasks.tasks module."
        )


class NoTaskBlueprintError(Exception):
    """Exception raised when no task blueprint is found in the environment mode config."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def __str__(self) -> str:
        return f"No task blueprint found in the environment mode config. Task blueprints should be defined in config['tasks']. Current config: {self.config}."
