"""
Task interfaces in RL-THOR environment.

TODO: Rename task_interfaces.py

TODO: Finish module docstring.
"""

# %% === Imports ===
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rl_thor.envs.actions import Ai2thorAction
from rl_thor.envs.reward import BaseRewardHandler
from rl_thor.envs.sim_objects import SimObjectType
from rl_thor.envs.tasks.item_prop import obj_prop_id_to_item_prop
from rl_thor.envs.tasks.item_prop_interface import (
    ItemProp,
    ItemPropValue,
)
from rl_thor.envs.tasks.items import (
    Assignment,
    AuxItem,
    CandidateData,
    CandidateId,
    ItemAdvancementDetails,
    ItemId,
    ItemOverlapClass,
    TaskItem,
)
from rl_thor.envs.tasks.relations import (
    RelationParam,
    RelationTypeId,
    relation_type_id_to_relation,
)
from rl_thor.utils.global_exceptions import DuplicateRelationsError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ai2thor.controller import Controller
    from ai2thor.server import Event

    from rl_thor.envs.scenes import SceneId
    from rl_thor.envs.sim_objects import SimObjId, SimObjMetadata
    from rl_thor.envs.tasks.relations import Relation


# %% === Task Blueprints ===
@dataclass
class TaskBlueprint:
    """Blueprint containing the information to instantiate a task in the environment."""

    task_type: type[BaseTask]
    scenes: set[SceneId]
    task_args: Mapping[str, TaskArgValue] = field(default_factory=dict)


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

    def get_reward(
        self,
        event: Event,
        controller_action: dict[str, Any],
    ) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the reward, task completion and additional information about the task for the given event.

        Args:
            event (Event): Event to calculate the reward for.
            controller_action (dict[str, Any]): Dictionary containing the information about the
                action executed by the controller.

        Returns:
            reward (float): Reward for the event.
            terminated (bool | None): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        if not event.metadata["lastActionSuccess"]:
            return 0.0, False, {}
        task_advancement, task_completion, info = self.task.compute_task_advancement(event, controller_action)
        reward = task_advancement - self.last_step_advancement
        self.last_step_advancement = task_advancement

        if task_completion:
            print("Task completed!!")
            reward += 10

        return reward, task_completion, info

    def reset(self, controller: Controller) -> tuple[bool, bool, dict[str, Any]]:
        """
        Reset the reward handler and its task (after preprocessing the scene).

        The reset is considered not successful if the task and the scene are incompatible.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the task is successfully reset.
            terminated (bool): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        # Reset the task
        reset_successful, task_advancement, task_completion, info = self.task.preprocess_and_reset(controller)
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

    def _reset_preprocess(self, controller: Controller) -> bool:  # noqa: ARG002, PLR6301
        """
        Preprocess the scene before resetting the task and return whether the preprocessing was successful.

        This method is called before the task is reset and is used to preprocess the scene, for
        example to change some properties of the objects in the scene.

        By default, this method does nothing and returns True.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the task is successfully reset.
        """
        return True

    def preprocess_and_reset(self, controller: Controller) -> tuple[bool, float, bool, dict[str, Any]]:
        """
        Preprocess the scene before resetting the task and reset the task.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            reset_successful (bool): True if the task is successfully reset.
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement at the beginning of the episode.
        """
        preprocess_successful = self._reset_preprocess(controller)
        if not preprocess_successful:
            return False, 0, False, {}
        return self.reset(controller)

    @abstractmethod
    def compute_task_advancement(
        self,
        event: Event,
        controller_action: dict[str, Any],
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,
    ) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the task advancement and whether the task is completed.

        Args:
            event (Event): Event corresponding to the state of the scene.
            controller_action (dict[str, Any]): Dictionary containing the information about the
                action executed by the controller.
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

    def __str__(self) -> str:
        """Return the text description of the task."""
        return f"{self.__class__.__name__}"


class UndefinedTask(BaseTask):
    """Undefined task that is never completed and has no advancement."""

    def reset(self, controller: Controller) -> tuple[float, bool, dict[str, Any]]:
        """Reset and initialize the task and the controller."""
        raise NotImplementedError("Undefined task")

    def compute_task_advancement(
        self,
        event: Event,
        controller_action: dict[str, Any],
        scene_objects_dict: dict[SimObjId, SimObjMetadata] | None = None,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Return the task advancement and whether the task is completed."""
        raise NotImplementedError("Undefined task")

    def text_description(self) -> str:
        """Return a text description of the task."""
        raise NotImplementedError("Undefined task")


type TaskArgValue = ItemPropValue | int
type RelationsDict = dict[
    ItemId,
    dict[
        RelationTypeId,
        dict[str, RelationParam],
    ],
]  # TODO: Replace RelationTypeId by type[Relation]
type TaskDict = dict[ItemId, TaskItemData]


@dataclass
class TaskItemData:
    """Description of a task item."""

    properties: set[ItemProp] = field(default_factory=set)
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
        scene_name = event.metadata["sceneName"]
        scene_objects_dict: dict[SimObjId, SimObjMetadata] = {obj["objectId"]: obj for obj in event.metadata["objects"]}

        # Initialize the candidates of the items
        for item in self.items:
            item.candidates_data = item.instantiate_candidate_data(scene_objects_dict)
            if not item.candidate_ids:
                scene_name = event.metadata["sceneName"]
                print(f"{scene_name}: No candidate found for item '{item.id}'")
                return False, 0, False, {}

        # Initialize the auxiliary items
        self.auxiliary_items = frozenset().union(
            *(auxiliary_items for item in self.items for auxiliary_items in item.props_auxiliary_items.values())
        )
        for auxiliary_item in self.auxiliary_items:
            auxiliary_item.relations = frozenset()
            auxiliary_item.candidates_data = auxiliary_item.instantiate_candidate_data(scene_objects_dict)
            if not auxiliary_item.candidate_ids:
                print(f"{scene_name}: No candidate found for auxiliary item {auxiliary_item.id}")
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

        return True, *self.compute_task_advancement(event, controller.last_action, scene_objects_dict)

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
        controller_action: dict[str, Any],
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
            controller_action (dict[str, Any]): Dictionary containing the information about the
                action executed by the controller.
            scene_objects_dict (dict[SimObjId, SimObjMetadata], optional): Dictionary
                mapping object ids to their metadata to avoid recomputing it. Defaults to None.

        Returns:
            task_advancement (int): Task advancement.
            is_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        if scene_objects_dict is None:
            scene_objects_dict = {obj["objectId"]: obj for obj in event.metadata["objects"]}

        # === Update candidates ===
        if controller_action["action"] == Ai2thorAction.SLICE_OBJECT and event.metadata["lastActionSuccess"] == True:
            # Update the candidates of items that have the sliced object as a candidate
            sliced_object_id = controller_action["objectId"]
            # Identify inherited object
            inherited_object_ids = {
                CandidateId(obj_id) for obj_id in scene_objects_dict if obj_id.startswith(f"{sliced_object_id}|")
            }
            to_update_overlap_classes: set[ItemOverlapClass] = set()
            # Update the candidates of the items
            for item in self.items:
                if sliced_object_id in item.candidates_data:
                    item.candidates_data.update({
                        inherited_object_id: CandidateData(inherited_object_id, item)
                        for inherited_object_id in inherited_object_ids
                    })
                    del item.candidates_data[sliced_object_id]
                    to_update_overlap_classes.add(item.overlap_class)

            # Update the valid assignments of the overlap classes
            for overlap_class in to_update_overlap_classes:
                overlap_class.valid_assignments = overlap_class.compute_valid_assignments_with_inherited_objects(
                    sliced_object_id, inherited_object_ids
                )

        # Compute the interesting assignments for each overlap class
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
            "advancement_details": {item: ItemAdvancementDetails(item, best_assignment) for item in self.items},
            "task_advancement": max_task_advancement,
            "scene_objects_dict": scene_objects_dict,
        }
        # TODO: Add other info

        return max_task_advancement, is_terminated, info

    @staticmethod
    def compute_assignment_advancement(
        global_assignment: Assignment,
    ) -> int:
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
        properties_by_item_id = {item_id: item_data.properties for item_id, item_data in task_description_dict.items()}
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
        return f"{self.__class__.__name__}({self.task_description_dict})"


# %% === Utility Functions ===
# TODO: Add support for relations with parameters
# TODO: Add support for MultiValuePSF, RangePSF, etc.
def parse_task_description_dict(task_description_dict: dict[str, dict[str, Any]]) -> TaskDict:
    """
    Parse a dictionary describing the task graph and return a task description dictionary.

    Example of task description dictionary for the task of placing a hot apple in a plate:
    task_description_dict = {
        "plate_receptacle": {
            "properties": {"objectType": "Plate"},
        },
        "hot_apple": {
            "properties": {"objectType": "Apple", "temperature": "Hot"},
            "relations": {"plate_receptacle": ["contained_in"]},
        },
    }

    And it becomes the proper task description dictionary:
    task_description_dict2 = {
        ItemId("plate_receptacle"): TaskItemData(
            properties={ObjectTypeProp(SingleValuePSF(SimObjectType.PLATE))},
        ),
        ItemId("hot_apple"): TaskItemData(
            properties={
                ObjectTypeProp(SimObjectType.APPLE),
                TemperatureProp(TemperatureValue.HOT),
            },
            relations={ItemId("plate_receptacle"): {RelationTypeId.CONTAINED_IN: {}}},
        ),
    }

    Args:
        task_description_dict (dict[str, Any]): Task description dictionary.

    Returns:
        task_description_dict (TaskDict): Parsed task description dictionary.
    """
    parsed_task_description_dict: TaskDict = {}
    for item_id, item_data in task_description_dict.items():
        # === Parse properties ===
        properties = item_data.get("properties", {})
        property_dict: set[ItemProp] = {
            obj_prop_id_to_item_prop[prop](SimObjectType(prop_value) if prop == "objectType" else prop_value)
            for prop, prop_value in properties.items()
        }
        # === Parse relations ===
        relation_dict: RelationsDict = {}
        relations = item_data.get("relations", {})
        for related_item_id, relations_dict in relations.items():
            for relation_type_id in relations_dict:
                relation_dict[ItemId(related_item_id)] = {RelationTypeId(relation_type_id): {}}
        parsed_task_description_dict[ItemId(item_id)] = TaskItemData(property_dict, relation_dict)

    return parsed_task_description_dict
