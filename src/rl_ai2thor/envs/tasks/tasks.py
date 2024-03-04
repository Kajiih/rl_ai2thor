"""
Tasks in AI2THOR RL environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx

from rl_ai2thor.envs.reward import BaseRewardHandler
from rl_ai2thor.envs.sim_objects import SimObjectType
from rl_ai2thor.envs.tasks.items import (
    ItemOverlapClass,
    NoCandidateError,
    TaskItem,
    TemperatureValue,
    obj_prop_id_to_item_prop,
)
from rl_ai2thor.envs.tasks.relations import relation_type_id_to_relation

if TYPE_CHECKING:
    from rl_ai2thor.envs.scenes import SceneId
    from rl_ai2thor.envs.sim_objects import SimObjectType
    from rl_ai2thor.envs.tasks.relations import Relation, RelationTypeId
    from rl_ai2thor.utils.ai2thor_types import EventLike


# %% === Reward handlers ===
# TODO: Add more options
class GraphTaskRewardHandler(BaseRewardHandler):
    """
    Reward handler for graph tasks AI2-THOR environments.

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
    def get_reward(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Return the reward, task completion and additional information about the task for the given event.

        Args:
            event (Any): Event to calculate the reward for.

        Returns:
            reward (float): Reward for the event.
            terminated (bool, Optional): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        task_advancement, task_completion, info = self.task.compute_task_advancement(event)
        reward = task_advancement - self.last_step_advancement
        self.last_step_advancement = task_advancement

        return reward, task_completion, info

    def reset(self, event: EventLike) -> tuple[bool, dict[str, Any]]:
        """
        Reset the reward handler.

        Args:
            event (Any): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            terminated (bool): Whether the episode has terminated.
            info (dict[str, Any]): Additional information about the state of the task.
        """
        # Reset the task
        task_advancement, task_completion, info = self.task.reset(event)
        # Initialize the last step advancement
        self.last_step_advancement = task_advancement

        return task_completion, info


# %% === Tasks ===
class BaseTask(ABC):
    """Base class for tasks."""

    _reward_handler_type: type[BaseRewardHandler]

    @abstractmethod
    def reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """Reset the task with the information of the event."""

    @abstractmethod
    def compute_task_advancement(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """Return the task advancement and whether the task is completed."""

    def get_reward_handler(self) -> BaseRewardHandler:
        """Return the reward handler for the task."""
        return self._reward_handler_type(self)


class UndefinableTask(BaseTask):
    """Undefined task that is never completed and has no advancement."""

    @staticmethod
    def reset(event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """Reset the task with the information of the event."""
        return 0.0, False, {}

    @staticmethod
    def compute_task_advancement(event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """Return the task advancement and whether the task is completed."""
        return 0.0, False, {}


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
    target_objects = {
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
        reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
            Reset the task with the information of the event.
        get_task_advancement(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
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
        self.items = self.full_initialize_items_and_relations_from_dict(task_description_dict)

        # Initialize the task graph
        self.task_graph = nx.DiGraph()
        # TODO: Check if we keep the graph (unused for now)

        self.overlap_classes: list[ItemOverlapClass] = []

    # TODO? Add check to make sure the task is feasible?
    def reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        Initialize the candidates of the items with the objects
        in the scene and compute the overlap classes.

        Args:
            event (EventLike): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Initialize the candidates of the items
        for item in self.items:
            for obj_metadata in event.metadata["objects"]:
                if item.is_candidate(obj_metadata):
                    item.candidate_ids.add(obj_metadata["objectId"])

        # Check that every item has at least one candidate # TODO: Remove?
        # for item in self.items:
        #     if not item.candidate_ids:
        #         raise NoCandidateError(item)

        # TODO: Add check that every overlap class has
        # Compute the overlap classes
        overlap_classes: dict[int, dict[str, Any]] = {}
        for item in self.items:
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

        self.overlap_classes = [
            ItemOverlapClass[T](
                items=overlap_class["items"],
                candidate_ids=list(overlap_class["candidate_ids"]),
            )
            for overlap_class in overlap_classes.values()
        ]

        # Compute max task advancement
        # Total number of properties and relations of the items
        self.max_task_advancement = sum(len(item.properties) + len(item.relations) for item in self.items)

        # Return initial task advancement
        return self.compute_task_advancement(event)

    # TODO: Add trying only the top k interesting assignments according to the maximum possible score (need to order the list of interesting candidates then the list of interesting assignments for each overlap class)
    def compute_task_advancement(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
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
            event (EventLike): Event corresponding to the state of the scene.

        Returns:
            task_advancement (float): Task advancement.
            is_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Compute the interesting assignments for each overlap class and the results and scores of each candidate for each item
        overlap_classes_assignment_data = [
            overlap_class.compute_interesting_assignments(event.metadata["objects"])
            for overlap_class in self.overlap_classes
        ]
        # Extract the interesting assignments, results and scores
        interesting_assignments = [data[0] for data in overlap_classes_assignment_data]
        # Merge the results and scores of the items
        items_results = {
            item: item_result
            for overlap_class_assignment_data in overlap_classes_assignment_data
            for item, item_result in overlap_class_assignment_data[1].items()
        }
        items_scores = {
            item: item_score
            for overlap_class_assignment_data in overlap_classes_assignment_data
            for item, item_score in overlap_class_assignment_data[2].items()
        }

        # Construct a generator of the cartesian product of the interesting assignments
        assignment_products = itertools.product(*interesting_assignments)

        max_task_advancement = 0
        best_assignment = {}
        is_terminated = False

        # Compute the task advancement for each global assignment
        for assignment_product in assignment_products:
            # Merge the assignments of the overlap classes
            global_assignment = {
                item: obj_id
                for overlap_class_assignment in assignment_product
                for item, obj_id in overlap_class_assignment.items()
            }
            # Property scores
            task_advancement = sum(
                items_scores[item][obj_id]["sum_property_scores"] for item, obj_id in global_assignment.items()
            )
            # Strictly satisfied relation scores
            for item, obj_id in global_assignment.items():
                item_relations_results: dict[T, dict[RelationTypeId, dict[SimObjectType, set[SimObjectType]]]] = (
                    items_results[item]["relations"]
                )  # type: ignore  # TODO: Delete type ignore after simplifying the type
                for related_item, relations in item_relations_results.items():
                    related_item_assigned_obj_id = global_assignment[related_item]
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
    args: dict[str, list[Any]] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Return the hash of the task blueprint."""
        return hash(self.task_type)


# %% == Alfred tasks ==
class PlaceIn(GraphTask[str]):
    """
    Task for placing a given object in a given receptacle.

    This is equivalent to the pick_and_place_simple task from Alfred.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str) -> None:
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


class PlaceSameTwoIn(GraphTask[str]):
    """
    Task for placing two objects of the same given type in a given receptacle.

    This is equivalent to the pick_two_obj_and_place task from Alfred.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str) -> None:
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
            },
            "placed_object_1": {
                "properties": {"objectType": self.placed_object_type},
                "relations": {"receptacle": ["contained_in"]},
            },
            "placed_object_2": {
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
        return f"Place 2 {self.placed_object_type} in {self.receptacle_type}"


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

        target_objects: TaskDict[str] = {
            "receptacle": {
                "properties": {"objectType": self.receptacle_type},
            },
            "pickupable_receptacle": {
                "properties": {"objectType": self.pickupable_receptacle_type},
                "relations": {"receptacle": ["contained_in"]},
            },
            "placed_object": {
                "properties": {"objectType": self.placed_object_type},
                "relations": {"pickupable_receptacle": ["contained_in"]},
            },
        }
        super().__init__(target_objects)

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.placed_object_type} in {self.pickupable_receptacle_type} in {self.receptacle_type}"


# TODO: Implement task reset
class PlaceCleanedIn(GraphTask[str]):
    """
    Task for placing a given cleaned object in a given receptacle.

    This is equivalent to the pick_clean_then_place_in_recep task from Alfred.

    All instance of placed_object_type are made dirty during the reset of the task.
    # TODO: Implement this
    """

    def __init__(self, placed_object_type: str, receptacle_type: str) -> None:
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
            },
            "cleaned_object": {
                "properties": {"objectType": self.placed_object_type, "isDirty": False},
                "relations": {"receptacle": ["contained_in"]},
            },
        }
        super().__init__(target_objects)

    def reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        All instance of placed_object_type are made dirty during the reset of the task.

        Args:
            event (EventLike): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Make all instances of placed_object_type dirty
        raise NotImplementedError

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place cleaned {self.placed_object_type} in {self.receptacle_type}"


# TODO: Implement task reset
class PlaceHeatedIn(GraphTask[str]):
    """
    Task for placing a given heated object in a given receptacle.

    This is equivalent to the pick_heat_then_place_in_recep task from Alfred.

    All instance of placed_object_type are made at room temperature during the reset of the task.
    # TODO: Implement this

    Args:
        placed_object_type (str): The type of object to place.
        receptacle_type (str): The type of receptacle to place the object in.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str) -> None:
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
            },
            "heated_object": {
                "properties": {"objectType": self.placed_object_type, "temperature": TemperatureValue.HOT},
                "relations": {"receptacle": ["contained_in"]},
            },
        }
        super().__init__(target_objects)

    def reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        All instance of placed_object_type are made at room temperature during the reset of the task.

        Args:
            event (EventLike): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Make all instances of placed_object_type at room temperature
        raise NotImplementedError

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place heated {self.placed_object_type} in {self.receptacle_type}"


# TODO: Implement task reset
class PlaceCooledIn(GraphTask[str]):
    """
    Task for placing a given cooled object in a given receptacle.

    This is equivalent to the pick_cool_then_place_in_recep task from Alfred.

    All instance of placed_object_type are made at room temperature during the reset of the task.
    # TODO: Implement this

    Args:
        placed_object_type (str): The type of object to place.
        receptacle_type (str): The type of receptacle to place the object in.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str) -> None:
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
            },
            "cooled_object": {
                "properties": {"objectType": self.placed_object_type, "temperature": TemperatureValue.COLD},
                "relations": {"receptacle": ["contained_in"]},
            },
        }
        super().__init__(target_objects)

    def reset(self, event: EventLike) -> tuple[float, bool, dict[str, Any]]:
        """
        Reset the task with the information of the event.

        All instance of placed_object_type are made at room temperature during the reset of the task.

        Args:
            event (EventLike): Event corresponding to the state of the scene
                at the beginning of the episode.

        Returns:
            initial_task_advancement (float): Initial task advancement.
            is_task_completed (bool): True if the task is completed.
            info (dict[str, Any]): Additional information about the task advancement.
        """
        # Make all instances of placed_object_type at room temperature
        raise NotImplementedError

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place cooled {self.placed_object_type} in {self.receptacle_type}"


# TODO: Implement the fact that any light source can be used instead of only desk lamps
# TODO: Implement with close_t0 relation instead of visible for the light source
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

        target_objects: TaskDict[str] = {
            "light_source": {
                "properties": {
                    "objectType": LightSourcesType.DESK_LAMP,  # TODO: Add support for other light sources
                    "isToggled": True,
                    "visible": True,
                },
            },
            "looked_at_object": {
                "properties": {"objectType": self.looked_at_object_type, "visible": True},
            },
        }
        super().__init__(target_objects)

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Look st {self.looked_at_object_type} in light"


# %%  === Task object types ===
class LightSourcesType(StrEnum):
    """Types of light sources."""

    CANDLE = "Candle"
    DESK_LAMP = "DeskLamp"
    FLOOR_LAMP = "FloorLamp"
    # LIGHT_SWITCH = "LightSwitch"


# %% === Constants ===
ALL_TASKS = {
    "PlaceIn": PlaceIn,
    "PlaceSameTwoIn": PlaceSameTwoIn,
    "PlaceWithMoveableRecepIn": PlaceWithMoveableRecepIn,
    "PlaceCleanedIn": PlaceCleanedIn,
    "PlaceHeatedIn": PlaceHeatedIn,
    "PlaceCooledIn": PlaceCooledIn,
    "LookInLight": LookInLight,
}
