"""
Tasks in AI2THOR RL environment.

TODO: Finish module docstring.
"""

# TODO: Add a way to handle the fact that not every object can be placed in every receptacle in the task advancement computation
# -> Need to add a list of object types to the receptacle required properties (its object type has to be in this list)
# -> Need to implement handling properties where the value is a list of possible values instead of a single value
# -> Then compute_compatible_args_from_blueprint can be more simply implemented using the candidates of the items

# %% === Imports ===
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx

from rl_ai2thor.data import OBJECT_TYPES_DATA
from rl_ai2thor.envs.actions import Ai2thorAction
from rl_ai2thor.envs.reward import BaseRewardHandler
from rl_ai2thor.envs.sim_objects import (
    COLD_SOURCES,
    DIRTYABLES,
    HEAT_SOURCES,
    WATER_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjVariableProp,
)
from rl_ai2thor.envs.tasks.items import (
    ItemOverlapClass,
    PropValue,
    TaskItem,
    TemperatureValue,
    obj_prop_id_to_item_prop,
)
from rl_ai2thor.envs.tasks.relations import relation_type_id_to_relation

if TYPE_CHECKING:
    from ai2thor.controller import Controller
    from ai2thor.server import Event

    from rl_ai2thor.envs.scenes import SceneId
    from rl_ai2thor.envs.sim_objects import SimObjId, SimObjMetadata
    from rl_ai2thor.envs.tasks.relations import Relation, RelationTypeId


# %% === Reward handlers ===
# TODO: Add more options
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

    # TODO: Add shortcut when the action failed or similar special cases
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

        return reward, task_completion, info

    def reset(self, controller: Controller) -> tuple[bool, dict[str, Any]]:
        """
        Reset the reward handler.

        Args:
            controller (Controller): AI2THOR controller at the beginning of the episode.

        Returns:
            terminated (bool): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        # Reset the task
        task_advancement, task_completion, info = self.task.reset(controller)
        # Initialize the last step advancement
        self.last_step_advancement = task_advancement

        return task_completion, info


# %% === Tasks ===
class BaseTask(ABC):
    """Base class for tasks."""

    _reward_handler_type: type[BaseRewardHandler]

    @abstractmethod
    def reset(self, controller: Controller) -> tuple[float, bool, dict[str, Any]]:
        """Reset and initialize the task and the controller."""

    @abstractmethod
    def compute_task_advancement(self, event: Event) -> tuple[float, bool, dict[str, Any]]:
        """Return the task advancement and whether the task is completed."""

    @staticmethod
    @abstractmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
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


class UndefinableTask(BaseTask):
    """Undefined task that is never completed and has no advancement."""

    @staticmethod
    def reset(controller: Controller) -> tuple[float, bool, dict[str, Any]]:  # noqa: ARG004
        """Reset and initialize the task and the controller."""
        return 0.0, False, {}

    @staticmethod
    def compute_task_advancement(event: Event) -> tuple[float, bool, dict[str, Any]]:  # noqa: ARG004
        """Return the task advancement and whether the task is completed."""
        return 0.0, False, {}

    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """Compute the compatible task arguments from the task blueprint and the event."""
        raise NotImplementedError

    @staticmethod
    def text_description() -> str:
        """Return a text description of the task."""
        return ""


type TaskDict[T: Hashable] = dict[T, dict[Literal["properties", "relations"], dict]]


# TODO: Add support for weighted properties and relations
# TODO: Add support for agent properties
# TODO: Remove networkx dependency
class GraphTask[T: Hashable](BaseTask):
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
        reset(self, event: Event) -> tuple[float, bool, dict[str, Any]]:
            Reset the task with the information of the event.
        get_task_advancement(self, event: Event) -> tuple[float, bool, dict[str, Any]]:
            Return the task advancement and whether the task is completed.
        full_initialize_items_and_relations_from_dict(task_description_dict: TaskDict) -> list[TaskItem[T]]
            Create and initialize TaskItems for the graph task.

    """

    _reward_handler_type = GraphTaskRewardHandler

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
        self.task_description_dict = task_description_dict
        self.items = self.full_initialize_items_and_relations_from_dict(task_description_dict)
        self._items_by_id = {item.id: item for item in self.items}

        # Initialize the task graph
        self.task_graph = nx.DiGraph()
        # TODO: Check if we keep the graph (unused for now)

        self.overlap_classes: list[ItemOverlapClass] = []

    # TODO? Add check to make sure the task is feasible?
    def reset(self, controller: Controller) -> tuple[float, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        Initialize the candidates of the items with the objects
        in the scene and compute the overlap classes.

        Args:
            controller (Controller): AI2THOR controller at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        event: Event = controller.last_event  # type: ignore
        # Initialize the candidates of the items
        for item in self.items:
            for obj_metadata in event.metadata["objects"]:
                if item.is_candidate(obj_metadata):
                    item.candidate_ids.add(obj_metadata["objectId"])

        self.overlap_classes = self._compute_overlap_classes(self.items)

        # Compute max task advancement = Total number of properties and relations of the items
        self.max_task_advancement = sum(len(item.properties) + len(item.relations) for item in self.items)

        # Return initial task advancement
        return self.compute_task_advancement(event)

    @staticmethod
    def _compute_overlap_classes(items: list[TaskItem[T]]) -> list[ItemOverlapClass[T]]:
        """
        Compute the overlap classes of the items in the scene.

        Items must have their candidates initialized before calling this method.

        Args:
            items (list[TaskItem[T]]): List of items.

        Returns:
            overlap_classes (list[ItemOverlapClass[T]]): List of overlap classes.
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
            ItemOverlapClass[T](
                items=overlap_class["items"],
                candidate_ids=list(overlap_class["candidate_ids"]),
            )
            for overlap_class in overlap_classes.values()
        ]

    # TODO: Add trying only the top k interesting assignments according to the maximum possible score (need to order the list of interesting candidates then the list of interesting assignments for each overlap class)
    def compute_task_advancement(self, event: Event) -> tuple[float, bool, dict[str, Any]]:
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
            event (Event): Event corresponding to the state of the scene.

        Returns:
            task_advancement (float): Task advancement.
            is_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Compute the interesting assignments for each overlap class and the results and scores of each candidate for each item
        scene_objects_dict: dict[SimObjId, SimObjMetadata] = {obj["objectId"]: obj for obj in event.metadata["objects"]}
        overlap_classes_assignment_data = [
            overlap_class.compute_interesting_assignments(scene_objects_dict) for overlap_class in self.overlap_classes
        ]
        # Extract the interesting assignments, results and scores
        interesting_assignments = [data[0] for data in overlap_classes_assignment_data]
        # Merge the results and scores of the items
        relation_results = {
            item: item_score
            for overlap_class_assignment_data in overlap_classes_assignment_data
            for item, item_score in overlap_class_assignment_data[2].items()
        }
        properties_scores = {
            item: item_score
            for overlap_class_assignment_data in overlap_classes_assignment_data
            for item, item_score in overlap_class_assignment_data[3].items()
        }

        # Construct a generator of the cartesian product of the interesting assignments
        assignment_products = itertools.product(*interesting_assignments)

        max_task_advancement = 0
        best_assignment = {}
        is_terminated = False

        # Compute the task advancement for each global assignment
        for assignment_product in assignment_products:
            task_advancement = 0
            # Merge the assignments of the overlap classes
            global_assignment: dict[TaskItem[T], Any] = {
                item: obj_id
                for overlap_class_assignment in assignment_product
                for item, obj_id in overlap_class_assignment.items()
            }
            # Add property scores
            task_advancement += sum(properties_scores[item][obj_id] for item, obj_id in global_assignment.items())
            # Add strictly satisfied relation scores
            for item, obj_id in global_assignment.items():
                item_relations_results = relation_results[item]
                for related_item_id, relations in item_relations_results.items():
                    related_item_assigned_obj_id = global_assignment[self._items_by_id[related_item_id]]
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
                {obj_prop_id_to_item_prop[prop]: value for prop, value in item_dict.get("properties", {}).items()},
            )
            for item_id, item_dict in task_description_dict.items()
        }
        organized_relations: dict[T, dict[T, dict[RelationTypeId, Relation]]] = {
            main_item_id: {
                related_item_id: {
                    relation_type_id: relation_type_id_to_relation[relation_type_id](
                        items[main_item_id], items[related_item_id]
                    )
                    for relation_type_id in relation_type_ids
                }
                for related_item_id, relation_type_ids in main_item_dict.get("relations", {}).items()
            }
            for main_item_id, main_item_dict in task_description_dict.items()
        }

        # Add inverse relations
        inverse_relations_type_ids = {
            related_item_id: {main_item_id: relation.inverse_relation_type_id}
            for main_item_id, relations_dict in organized_relations.items()
            for related_item_id, relations in relations_dict.items()
            for relation in relations.values()
        }

        for related_item_id in inverse_relations_type_ids:
            for main_item_id, relation_type_id in inverse_relations_type_ids[related_item_id].items():
                if main_item_id not in organized_relations[related_item_id]:
                    organized_relations[related_item_id][main_item_id] = {}
                if relation_type_id not in organized_relations[related_item_id][main_item_id]:
                    organized_relations[related_item_id][main_item_id][relation_type_id] = relation_type_id_to_relation[
                        relation_type_id
                    ](items[related_item_id], items[main_item_id])

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
        nb_items = len(self.items)
        nb_relations = sum(len(item.relations) for item in self.items)
        return f"GraphTask({nb_items} items, {nb_relations} relations)"


# %% === Task Blueprints ===
@dataclass
class TaskBlueprint:
    """Blueprint containing the information to instantiate a task in the environment."""

    task_type: type[BaseTask]
    scenes: set[SceneId]
    task_args: Mapping[str, frozenset[PropValue]] = field(default_factory=dict)

    def compute_compatible_task_args(self, event: Event) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the event.

        Args:
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        return self.task_type.compute_compatible_args_from_blueprint(self, event)


# %% == Alfred tasks ==
class PlaceNSameIn(GraphTask[str]):
    """
    Task for placing n objects of the same type in a receptacle.

    This is equivalent to the pick_two_obj_and_place task from Alfred with n=2 and
    pick_and_place_simple with n=1.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str, n: int = 1) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.
            n (int): The number of objects to place.
        """
        self.placed_object_type = placed_object_type
        self.receptacle_type = receptacle_type
        self.n = n

        task_description_dict = self._create_task_description_dict(placed_object_type, receptacle_type, n)

        super().__init__(task_description_dict)

    @staticmethod
    def _create_task_description_dict(placed_object_type: str, receptacle_type: str, n: int = 1) -> TaskDict[str]:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.
            n (int): The number of objects to place.

        Returns:
            task_description_dict (TaskDict[str]): Task description dictionary.
        """
        task_description_dict: TaskDict[str] = {
            "receptacle": {
                "properties": {"objectType": receptacle_type},
            }
        }
        for i in range(n):
            task_description_dict[f"placed_object_{i}"] = {
                "properties": {"objectType": placed_object_type},
                "relations": {"receptacle": ["contained_in"]},
            }

        return task_description_dict

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.n} {self.placed_object_type} in {self.receptacle_type}"

    # TODO: Create a generalized version of this that works for all tasks
    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}
        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        # Keep only the object types that are present in the scene for the blueprint of both 'placed_object_type' and 'receptacle_type'
        min_n = min(task_blueprint.task_args["n"])
        args_blueprints = {
            "placed_object_type": task_blueprint.task_args["placed_object_type"]
            & {obj_type for obj_type, count in scene_object_types_count.items() if count >= min_n},
            "receptacle_type": task_blueprint.task_args["receptacle_type"] & set(scene_object_types_count),
        }

        # Create a list with all the compatible combinations of placed_object_type and receptacle_types and with enough instances of placed_object_type in the scene
        compatible_args = [
            (placed_object_type, compatible_receptacle, n)
            for placed_object_type in args_blueprints["placed_object_type"]
            for compatible_receptacle in OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"]
            for n in task_blueprint.task_args["n"]
            if compatible_receptacle in args_blueprints["receptacle_type"]
            and scene_object_types_count[placed_object_type] >= n
        ]

        return compatible_args


class PlaceNSameInSubclass(PlaceNSameIn, ABC):
    """Abstract subclass of PlaceNSameIn for tasks with a specific number of objects to place."""

    n: int

    def __init__(self, placed_object_type: str, receptacle_type: str) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.
        """
        super().__init__(placed_object_type, receptacle_type, self.n)

        # Replace the instance attribute with the class attribute
        del self.n

    @classmethod
    def compute_compatible_args_from_blueprint(
        cls,
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}

        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        # Keep only the object types that are present in the scene for the blueprint of both 'placed_object_type' and 'receptacle_type'
        args_blueprints = {
            "placed_object_type": task_blueprint.task_args["placed_object_type"]
            & {obj_type for obj_type, count in scene_object_types_count.items() if count >= cls.n},
            "receptacle_type": task_blueprint.task_args["receptacle_type"] & set(scene_object_types_count),
        }

        # Create a list with all the compatible combinations of placed_object_type and receptacle_types and with enough instances of placed_object_type in the scene
        compatible_args = [
            (placed_object_type, compatible_receptacle)
            for placed_object_type in args_blueprints["placed_object_type"]
            for compatible_receptacle in OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"]
            if compatible_receptacle in args_blueprints["receptacle_type"]
        ]

        return compatible_args


class PlaceIn(PlaceNSameInSubclass):
    """
    Task for placing a given object in a given receptacle.

    This is equivalent to the pick_and_place_simple task from Alfred.
    """

    n = 1


class PlaceWithMoveableRecepIn(GraphTask[str]):
    """
    Task for placing an given object with a given moveable receptacle in a given receptacle.

    This is equivalent to the pick_and_place_with_movable_recep task from Alfred.
    """

    def __init__(self, placed_object_type: str, pickupable_receptacle_type: str, receptacle_type: str) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (str): The type of object to place.
            pickupable_receptacle_type (str): The type of pickupable receptacle to place the object in.
            receptacle_type (str): The type of receptacle to place the object in.
        """
        self.placed_object_type = placed_object_type
        self.pickupable_receptacle_type = pickupable_receptacle_type
        self.receptacle_type = receptacle_type

        task_description_dict = self._create_task_description_dict(
            placed_object_type, pickupable_receptacle_type, receptacle_type
        )

        super().__init__(task_description_dict)

    @staticmethod
    def _create_task_description_dict(
        placed_object_type: str, pickupable_receptacle_type: str, receptacle_type: str
    ) -> TaskDict[str]:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (str): The type of object to place.
            pickupable_receptacle_type (str): The type of pickupable receptacle to place the object in.
            receptacle_type (str): The type of receptacle to place the object in.

        Returns:
            task_description_dict (TaskDict[str]): Task description dictionary.
        """
        return {
            "receptacle": {
                "properties": {"objectType": receptacle_type},
            },
            "pickupable_receptacle": {
                "properties": {"objectType": pickupable_receptacle_type},
                "relations": {"receptacle": ["contained_in"]},
            },
            "placed_object": {
                "properties": {"objectType": placed_object_type},
                "relations": {"pickupable_receptacle": ["contained_in"]},
            },
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.placed_object_type} in {self.pickupable_receptacle_type} in {self.receptacle_type}"

    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}
        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        # Keep only the object types that are present in the scene for the blueprint of both 'placed_object_type', 'pickupable_receptacle_type' and 'receptacle_type'
        args_blueprints = {
            "placed_object_type": task_blueprint.task_args["placed_object_type"] & set(scene_object_types_count),
            "pickupable_receptacle_type": task_blueprint.task_args["pickupable_receptacle_type"]
            & set(scene_object_types_count),
            "receptacle_type": task_blueprint.task_args["receptacle_type"] & set(scene_object_types_count),
        }
        # Return a list with all the compatible combinations of placed_object_type, pickupable_receptacle_type and receptacle_types
        return [
            (placed_object_type, pickupable_receptacle, compatible_receptacle)
            for placed_object_type in args_blueprints["placed_object_type"]
            for pickupable_receptacle in OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"]
            if pickupable_receptacle in args_blueprints["pickupable_receptacle_type"]
            for compatible_receptacle in OBJECT_TYPES_DATA[pickupable_receptacle]["compatible_receptacles"]
            if compatible_receptacle in args_blueprints["receptacle_type"]
        ]


class PlaceCleanedIn(PlaceIn):
    """
    Task for placing a given cleaned object in a given receptacle.

    This is equivalent to the pick_clean_then_place_in_recep task from Alfred.

    All instance of placed_object_type are made dirty during the reset of the task.
    """

    def _create_task_description_dict(self, placed_object_type: str, receptacle_type: str) -> TaskDict[str]:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.

        Returns:
            task_description_dict (TaskDict[str]): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict(placed_object_type, receptacle_type)
        task_description_dict["placed_object"]["properties"][SimObjVariableProp.IS_DIRTY] = False

        return task_description_dict

    def reset(self, controller: Controller) -> tuple[float, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        All instances of placed_object_type are made dirty during the reset of the task.

        Args:
            controller (Controller): AI2THOR controller at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            if obj_metadata[SimObjFixedProp.OBJECT_TYPE] == self.placed_object_type:
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

    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:  # sourcery skip: invert-any-all
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}
        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        # Check if there is a water source in the scene
        if not any(water_source_type in scene_object_types_count for water_source_type in WATER_SOURCES):
            return []

        # Keep only the object types that are present in the scene for the blueprint of both 'placed_object_type' and 'receptacle_type'
        args_blueprints = {
            "placed_object_type": task_blueprint.task_args["placed_object_type"]
            & set(scene_object_types_count)
            & DIRTYABLES,
            "receptacle_type": task_blueprint.task_args["receptacle_type"] & set(scene_object_types_count),
        }
        # Return a list with all the compatible combinations of placed_object_type and receptacle_types
        return [
            (placed_object_type, compatible_receptacle)
            for placed_object_type in args_blueprints["placed_object_type"]
            for compatible_receptacle in OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"]
            if compatible_receptacle in args_blueprints["receptacle_type"]
        ]


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

    def _create_task_description_dict(self, placed_object_type: str, receptacle_type: str) -> TaskDict[str]:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.

        Returns:
            task_description_dict (TaskDict[str]): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict(placed_object_type, receptacle_type)
        task_description_dict["placed_object"]["properties"][SimObjVariableProp.TEMPERATURE] = TemperatureValue.HOT

        return task_description_dict

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place heated {self.placed_object_type} in {self.receptacle_type}"

    # TODO: Change this to avoid duplicating code with PlaceIn
    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}
        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        scene_heat_sources = {heat_source for heat_source in HEAT_SOURCES if heat_source in scene_object_types_count}

        # Keep only the object types that are present in the scene for the blueprint of both 'placed_object_type' and 'receptacle_type'
        args_blueprints = {
            "placed_object_type": task_blueprint.task_args["placed_object_type"] & set(scene_object_types_count),
            "receptacle_type": task_blueprint.task_args["receptacle_type"] & set(scene_object_types_count),
        }
        # Compute a list with all the compatible combinations of placed_object_type and receptacle_types
        compatible_args = [
            (placed_object_type, compatible_receptacle)
            for placed_object_type in args_blueprints["placed_object_type"]
            if OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"] & scene_heat_sources
            for compatible_receptacle in OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"]
            if compatible_receptacle in args_blueprints["receptacle_type"]
        ]
        return compatible_args


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

    def _create_task_description_dict(self, placed_object_type: str, receptacle_type: str) -> TaskDict[str]:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.

        Returns:
            task_description_dict (TaskDict[str]): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict(placed_object_type, receptacle_type)
        task_description_dict["placed_object"]["properties"][SimObjVariableProp.TEMPERATURE] = TemperatureValue.COLD

        return task_description_dict

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place cooled {self.placed_object_type} in {self.receptacle_type}"

    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}
        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        scene_cold_sources = {cold_source for cold_source in COLD_SOURCES if obj_type in scene_object_types_count}

        # Keep only the object types that are present in the scene for the blueprint of both 'placed_object_type' and 'receptacle_type'
        args_blueprints = {
            "placed_object_type": task_blueprint.task_args["placed_object_type"] & set(scene_object_types_count),
            "receptacle_type": task_blueprint.task_args["receptacle_type"] & set(scene_object_types_count),
        }
        # Compute a list with all the compatible combinations of placed_object_type and receptacle_types
        compatible_args = [
            (placed_object_type, compatible_receptacle)
            for placed_object_type in args_blueprints["placed_object_type"]
            if OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"] & scene_cold_sources
            for compatible_receptacle in OBJECT_TYPES_DATA[placed_object_type]["compatible_receptacles"]
            if compatible_receptacle in args_blueprints["receptacle_type"]
        ]
        return compatible_args


# TODO: Implement the fact that any light source can be used instead of only desk lamps
# TODO: Implement with close_to relation instead of visible for the light source
class LookInLight(GraphTask[str]):
    """
    Task for looking at a given object in light.

    This is equivalent to the look_at_obj_in_light task from Alfred.

    The light sources are listed in the LightSourcesType enum.
    """

    def __init__(self, looked_at_object_type: str) -> None:
        """
        Initialize the task.

        Args:
            looked_at_object_type (str): The type of object to look at.
        """
        self.looked_at_object_type = looked_at_object_type

        task_description_dict = self._create_task_description_dict(looked_at_object_type)
        super().__init__(task_description_dict)

    @staticmethod
    def _create_task_description_dict(looked_at_object_type: str) -> TaskDict[str]:
        """
        Create the task description dictionary for the task.

        Args:
            looked_at_object_type (str): The type of object to look at.

        Returns:
            task_description_dict (TaskDict[str]): Task description dictionary.
        """
        return {
            "light_source": {
                "properties": {
                    "objectType": SimObjectType.DESK_LAMP,  # TODO: Add support for other light sources
                    "isToggled": True,
                    "visible": True,
                },
            },
            "looked_at_object": {
                "properties": {"objectType": looked_at_object_type, "visible": True},
            },
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Look at {self.looked_at_object_type} in light"

    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}
        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        # Check that there is at least one light source in the scene
        # TODO: Add support for other light sources
        if SimObjectType.DESK_LAMP not in scene_object_types_count:
            return []

        # Keep only the object types that are present in the scene for the blueprint of 'looked_at_object_type'
        args_blueprints = {
            "looked_at_object_type": task_blueprint.task_args["looked_at_object_type"] & set(scene_object_types_count),
        }
        # Return a list with all the compatible combinations of looked_at_object_type
        return [(looked_at_object_type,) for looked_at_object_type in args_blueprints["looked_at_object_type"]]


# %% === Custom tasks ===
class Pickup(GraphTask[str]):
    """Task for picking up a given object."""

    def __init__(self, picked_object_type: str) -> None:
        """
        Initialize the task.

        Args:
            picked_object_type (str): The type of object to pick up.
        """
        self.picked_object_type = picked_object_type

        task_description_dict = self._create_task_description_dict(picked_object_type)
        super().__init__(task_description_dict)

    @staticmethod
    def _create_task_description_dict(picked_object_type: str) -> TaskDict[str]:
        """
        Create the task description dictionary for the task.

        Args:
            picked_object_type (str): The type of object to pick up.

        Returns:
            task_description_dict (TaskDict[str]): Task description dictionary.
        """
        return {
            "picked_object": {
                "properties": {"objectType": picked_object_type, "isPickedUp": True},
            }
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Pick up {self.picked_object_type}"

    @staticmethod
    def compute_compatible_args_from_blueprint(
        task_blueprint: TaskBlueprint,
        event: Event,
    ) -> list[tuple[PropValue, ...]]:
        """
        Compute the compatible task arguments from the task blueprint and the event.

        Note: The order of the returned list is not deterministic.

        Args:
            task_blueprint (TaskBlueprint): Task blueprint.
            event (Event): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            compatible_args (list[tuple[PropValue, ...]]): List of compatible task arguments.
        """
        scene_object_types_count = {}
        for obj_metadata in event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            if obj_type not in scene_object_types_count:
                scene_object_types_count[obj_type] = 0
            scene_object_types_count[obj_type] += 1

        # Keep only the object types that are present in the scene for the blueprint of 'picked_object_type'
        args_blueprints = {
            "picked_object_type": task_blueprint.task_args["picked_object_type"] & set(scene_object_types_count),
        }
        # Return a list with all the compatible combinations of picked_object_type
        return [(picked_object_type,) for picked_object_type in args_blueprints["picked_object_type"]]


# %% === Constants ===
ALL_TASKS = {
    # === Alfred tasks ===
    "PlaceIn": PlaceIn,
    "PlaceNSameIn": PlaceNSameIn,
    "PlaceWithMoveableRecepIn": PlaceWithMoveableRecepIn,
    "PlaceCleanedIn": PlaceCleanedIn,
    "PlaceHeatedIn": PlaceHeatedIn,
    "PlaceCooledIn": PlaceCooledIn,
    "LookInLight": LookInLight,
    # === Custom tasks ===
    "Pickup": Pickup,
}
