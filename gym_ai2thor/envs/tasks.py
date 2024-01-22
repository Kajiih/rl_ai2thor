"""
Module for tasks for the AI2THOR RL environment.

TODO: Finish module docstrings.
"""
from typing import Union, Callable, Any, Hashable
from abc import abstractmethod
from dataclasses import dataclass, field

from ai2thor_types import EventLike
from ai2thor_utils import get_object_type_from_id, get_object_data_from_id


# === Task Classes ===
# We need to define how to *Combine* tasks
# We need to define how to task's target interacts with each other
PropertyChecker = Callable[[str], bool]


@dataclass
class BaseTask:
    """
    Base class for tasks in the environment.

    Attributes:
        completed (bool): Whether the task has been completed.

    Methods:
        get_reward(event): Returns the reward corresponding to the event.
    """

    completed: bool = field(init=False, default=False)
    # TODO: Check if we keep completed attribute or if we check if the task was completed in the previous step

    @abstractmethod
    def get_reward(
        self,
        new_event: EventLike,
    ) -> tuple[float, bool]:
        """
        Return the reward corresponding to the transition between the two events and
        whether the task has been completed.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            reward (float): Reward corresponding to the transition.
            done (bool): Whether the task has been completed.
        """
        raise NotImplementedError


class UndefinedTask(BaseTask):
    """
    Undefined task raising an error when used.
    """

    def get_reward(
        self,
        new_event: EventLike,
    ) -> tuple[float, bool]:
        raise NotImplementedError(
            "Task is undefined. This is an unexpected behavior, maybe you forgot to reset the environment?"
        )


class DummyTask(BaseTask):
    """
    Dummy task for testing purposes.
    A reward of 0 is returned at each step and the episode is never terminated.
    """

    def get_reward(
        self,
        new_event: EventLike,
    ) -> tuple[float, bool]:
        """
        Return a reward of 0 and False for the done flag.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            reward (float): Reward corresponding to the transition.
            done (bool): Whether the task has been completed.
        """
        return 0, False


@dataclass
class NavigationTask(BaseTask):
    """
    Navigation task.

    Those tasks require the agent to change its state, optionally with respect to
    an auxiliary object (e.g. reach a target object).

    Attributes:
        auxiliary_object_type (str): The type of auxiliary object the agent needs to consider.
        target_distance (float): The maximum distance allowed between the agent and the target object.
    """

    auxiliary_object_type: str
    target_distance: float

    # !! Outdated !!
    # TODO: Update
    def get_reward(
        self,
        new_event: EventLike,
    ) -> tuple[float, bool]:
        """
        Return a reward of 1 if the agent reaches the target object type within the
        target distance, 0 otherwise.

        Args:
            last_event (EventLike): Event corresponding to the state before the action
                has been taken.
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            reward (float): Reward corresponding to the transition.
            done (bool): Whether the task has been completed.
        """
        if self.completed:
            return 0, True
        target_objects = [
            obj
            for obj in new_event.metadata["objects"]
            if obj["objectType"] == self.auxiliary_object_type
        ]
        for obj in target_objects:
            if obj["distance"] < self.target_distance:
                self.completed = True
                return 1, True
        return 0, False


@dataclass
class SimpleNavigationTask(BaseTask):
    """
    Base class for simple navigation tasks.

    Those tasks require the agent to change its state without involving any auxiliary
    object (e.g. crouching).

    Attributes:
        target_agent_state (dict[str, Union[str, bool, dict, float, list, None]]): The state the agent needs to reach.
    """

    agent_property_checkers: dict[str, PropertyChecker]

    def get_reward(
        self,
        new_event: EventLike,
    ) -> tuple[float, bool]:
        """
        Return a reward of 1 if the agent reaches the target object type within the
        target distance, 0 otherwise.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            reward (float): Reward corresponding to the transition.
            done (bool): Whether the task has been completed.
        """
        if self.completed:
            return 0, True
        for property_name, property_checker in self.agent_property_checkers.items():
            if not property_checker(new_event.metadata["agent"][property_name]):
                return 0, False
        self.completed = True
        return 1, True


class ManipulationTask(BaseTask):
    """
    Manipulation task.

    Those tasks require one object to reach a desired state (e.g. open a drawer),
    open a laptop), optionally with respect to an auxiliary object (e.g.
    put a knife in a drawer).
    """

    desired_state: dict[str, Union[str, bool, dict, float, list, None]]

    def get_reward(
        self,
        last_event: EventLike,
        new_event: EventLike,
    ) -> tuple[float, bool]:
        """
        Return a reward of 1 if the agent reaches the target object type within the
        target distance, 0 otherwise.

        Args:
            last_event (EventLike): Event corresponding to the state before the action
                has been taken.
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            reward (float): Reward corresponding to the transition.
            done (bool): Whether the task has been completed.
        """
        if self.completed:
            return 0, True
        target_objects = [
            obj
            for obj in new_event.metadata["objects"]
            if obj["objectType"] == self.target_object_type
        ]
        for obj in target_objects:
            if obj["state"] == self.target_state:
                self.completed = True
                return 1, True
        return 0, False


class PlaceObjectInReceptacleTask(BaseTask):
    """
    Task for placing a target object in a receptacle.

    This is equivalent to the pick_and_place_simple task from Alfred.
    """

    target_object_type: str
    target_receptacle_type: str

    def is_compled(self, new_event: EventLike) -> bool:
        """
        Return whether the task is completed in the given state.
        It could have been completed in any previous steps.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task is completed in the given state.
        """
        for obj in new_event.metadata["objects"]:
            if obj[
                "objectType"
            ] == self.target_object_type and self.target_receptacle_type in [
                get_object_type_from_id(receptacle_id)
                for receptacle_id in obj["parentReceptacles"]
            ]:
                return True
        return False

    def just_completed(self, last_event: EventLike, new_event: EventLike) -> bool:
        """
        Return whether the task was completed during the last step.

        Args:
            last_event (EventLike): Event corresponding to the state before the action
                has been taken.
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task was completed during the last step.
        """
        action_success = new_event.metadata["lastActionSuccess"]
        ai2thor_action = new_event.metadata["lastAction"]
        action_object_type = get_object_type_from_id(new_event.metadata["actionReturn"])
        try:
            last_inventory_object_type = get_object_type_from_id(
                last_event.metadata["inventoryObjects"][0]["objectType"]
            )
        except (IndexError, KeyError):
            return False
        if (
            action_success
            and ai2thor_action == "PutObject"
            and action_object_type == self.target_receptacle_type
            and last_inventory_object_type == self.target_object_type
        ):
            return True
        return False


class PlaceHotObjectInReceptacleTask(BaseTask):
    """
    Task for placing a hot target object in a receptacle.

    This is equivalent to the pick_heat_then_place_in_recep task from ALFRED.
    """

    target_object_type: str
    target_receptacle_type: str

    def is_compled(self, new_event: EventLike) -> bool:
        """
        Return whether the task is completed in the given state.
        It could have been completed in any previous steps.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task is completed in the given state.
        """
        for obj in new_event.metadata["objects"]:
            if (
                obj["objectType"] == self.target_object_type
                and self.target_receptacle_type
                in [
                    get_object_type_from_id(receptacle_id)
                    for receptacle_id in obj["parentReceptacles"]
                ]
                and obj["ObjectTemperature"] == "Hot"
            ):
                return True
        return False

    def just_completed(self, last_event: EventLike, new_event: EventLike) -> bool:
        """
        Return whether the task was completed during the last step.

        Args:
            last_event (EventLike): Event corresponding to the state before the action
                has been taken.
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task was completed during the last step.
        """

        action_success = new_event.metadata["lastActionSuccess"]
        ai2thor_action = new_event.metadata["lastAction"]
        action_object_type = get_object_type_from_id(new_event.metadata["actionReturn"])
        try:
            last_inventory_object_id = get_object_type_from_id(
                last_event.metadata["inventoryObjects"][0]["objectId"]
            )
        except (IndexError, KeyError):
            return False

        last_inventory_object_data = get_object_data_from_id(
            last_event.metadata["objects"], last_inventory_object_id
        )

        if (
            action_success
            and ai2thor_action == "PutObject"
            and action_object_type == self.target_receptacle_type
            and last_inventory_object_data["objectType"] == self.target_object_type
            and last_inventory_object_data["ObjectTemperature"] == "Hot"
        ):
            return True
        return False


class PlaceColdObjectInReceptacleTask(BaseTask):
    """
    Task for placing a cold target object in a receptacle.

    This is equivalent to the pick_cool_then_place_in_recep task from ALFRED.
    """

    target_object_type: str
    target_receptacle_type: str

    def is_compled(self, new_event: EventLike) -> bool:
        """
        Return whether the task is completed in the given state.
        It could have been completed in any previous steps.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task is completed in the given state.
        """
        for obj in new_event.metadata["objects"]:
            if (
                obj["objectType"] == self.target_object_type
                and self.target_receptacle_type
                in [
                    get_object_type_from_id(receptacle_id)
                    for receptacle_id in obj["parentReceptacles"]
                ]
                and obj["ObjectTemperature"] == "Cold"
            ):
                return True
        return False

    def just_completed(self, last_event: EventLike, new_event: EventLike) -> bool:
        """
        Return whether the task was completed during the last step.

        Args:
            last_event (EventLike): Event corresponding to the state before the action
                has been taken.
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task was completed during the last step.
        """

        action_success = new_event.metadata["lastActionSuccess"]
        ai2thor_action = new_event.metadata["lastAction"]
        action_object_type = get_object_type_from_id(new_event.metadata["actionReturn"])
        try:
            last_inventory_object_id = get_object_type_from_id(
                last_event.metadata["inventoryObjects"][0]["objectId"]
            )
        except (IndexError, KeyError):
            return False

        last_inventory_object_data = get_object_data_from_id(
            last_event.metadata["objects"], last_inventory_object_id
        )

        if (
            action_success
            and ai2thor_action == "PutObject"
            and action_object_type == self.target_receptacle_type
            and last_inventory_object_data["objectType"] == self.target_object_type
            and last_inventory_object_data["ObjectTemperature"] == "Cold"
        ):
            return True
        return False


class PlaceSameTwoInReceptacleTask(BaseTask):
    """
    Task for placing two (of the same) target objects in a receptacle.

    This is equivalent to the pick_two_then_place_in_recep task from ALFRED.
    """

    target_object_type: str
    target_receptacle_type: str

    def is_compled(self, new_event: EventLike) -> bool:
        """
        Return whether the task is completed in the given state.
        It could have been completed in any previous steps.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task is completed in the given state.
        """
        for obj in new_event.metadata["objects"]:
            if (
                obj["objectType"] == self.target_receptacle_type
                # Count occurences of target_object_type in receptacleObjectIds
                and sum(
                    1
                    for obj_type in get_object_type_from_id(obj["receptacleObjectIds"])
                    if obj_type == self.target_object_type
                )
                >= 2
            ):
                return True
        return False

    def just_completed(self, last_event: EventLike, new_event: EventLike) -> bool:
        """
        Return whether the task was completed during the last step.

        Args:
            last_event (EventLike): Event corresponding to the state before the action
                has been taken.
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task was completed during the last step.
        """

        action_success = new_event.metadata["lastActionSuccess"]
        ai2thor_action = new_event.metadata["lastAction"]
        action_object_data = get_object_data_from_id(
            new_event.metadata["objects"], new_event.metadata["actionReturn"]
        )
        action_object_type = action_object_data["objectType"]
        try:
            last_inventory_object_id = get_object_type_from_id(
                last_event.metadata["inventoryObjects"][0]["objectId"]
            )
        except (IndexError, KeyError):
            return False

        last_inventory_object_data = get_object_data_from_id(
            last_event.metadata["objects"], last_inventory_object_id
        )

        if (
            action_success
            and ai2thor_action == "PutObject"
            and action_object_type == self.target_receptacle_type
            and last_inventory_object_data["objectType"] == self.target_object_type
            and sum(
                1
                for obj_type in get_object_type_from_id(
                    action_object_data["parentReceptacles"]
                )
                if obj_type == self.target_object_type
            )
            >= 2
        ):
            return True
        return False


class PlaceMoveableReceptacleWithObjectTask(BaseTask):
    """
    Task for placing a target moveable receptacle with target object inside
    in a target receptacle.

    This is equivalent to the pick_and_place_with_moveable_recep task from ALFRED.
    """

    target_object_type: str
    target_moveable_receptacle_type: str
    target_receptacle_type: str

    def is_compled(self, new_event: EventLike) -> bool:
        """
        Return whether the task is completed in the given state.
        It could have been completed in any previous steps.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task is completed in the given state.
        """
        for obj in new_event.metadata["objects"]:
            if obj["objectType"] == self.target_receptacle_type:
                for inside_object_id in obj["receptacleObjectIds"]:
                    in_object_data = get_object_data_from_id(
                        new_event.metadata["objects"], inside_object_id
                    )
                    if in_object_data[
                        "objectType"
                    ] == self.target_moveable_receptacle_type and self.target_object_type in [
                        get_object_type_from_id(in_in_object_id)
                        for in_in_object_id in in_object_data["receptacleObjectIds"]
                    ]:
                        return True
        return False

    def just_completed(self, last_event: EventLike, new_event: EventLike) -> bool:
        """
        Return whether the task was completed during the last step.

        Args:
            last_event (EventLike): Event corresponding to the state before the action
                has been taken.
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task was completed during the last step.
        """

        action_success = new_event.metadata["lastActionSuccess"]
        ai2thor_action = new_event.metadata["lastAction"]
        action_object_type = get_object_type_from_id(new_event.metadata["actionReturn"])
        try:
            last_inventory_object_id = get_object_type_from_id(
                last_event.metadata["inventoryObjects"][0]["objectId"]
            )
        except (IndexError, KeyError):
            return False

        last_inventory_object_data = get_object_data_from_id(
            last_event.metadata["objects"], last_inventory_object_id
        )

        if (
            action_success
            and ai2thor_action == "PutObject"
            and action_object_type == self.target_receptacle_type
            and last_inventory_object_data["objectType"]
            == self.target_moveable_receptacle_type
            and self.target_object_type
            in [
                get_object_type_from_id(in_in_object_id)
                for in_in_object_id in last_inventory_object_data["receptacleObjectIds"]
            ]
        ):
            return True
        return False


class LookObjectInLightTask(BaseTask):
    """
    Task for looking at a target object in a light.

    This is equivalent to the look_at_object_in_light task from ALFRED.
    """

    target_object_type: str

    def is_compled(self, new_event: EventLike) -> bool:
        """
        Return whether the task is completed in the given state.
        It could have been completed in any previous steps.

        Args:
            new_event (EventLike): Event corresponding to the state after the action
                has been taken.

        Returns:
            done (bool): Whether the task is completed in the given state.
        """
        try:
            inventory_object_data = get_object_data_from_id(
                new_event.metadata["objects"],
                new_event.metadata["inventoryObjects"][0]["objectId"],
            )
        except (IndexError, KeyError):
            return False
        # Check that the necessary object is in the inventory
        if (
            inventory_object_data["objectType"] == self.target_object_type
            and inventory_object_data["visible"]
        ):
            # Check if a light soure is toggled and visible
            # The main light switch of the room does not count
            for obj in new_event.metadata["objects"]:
                if (
                    obj["objectType"] in ["DeskLamp", "FloorLamp", "Candle"]
                    and obj["visible"]
                    and obj["isToggled"]
                ):
                    return True
        return False


class GraphTask:
    """
    Base class for tasks that can be represented as a state graph representing the
    relations between objects in the scene (and the agent).
    For clarity purpose, we call the objects of the task "items", to avoid confusion
    with the real scene's objects.

    The vertices of the graph are the "items", corresponding to objects in the scene
    and the edges are the relations between them (e.g. "contains" ifthe item is
    supposed to contain another item).
    The items are represented by the properties required for the task (e.g.
    "objectType", "visible", "isSliced", "temperature",...).

    Attributes:
        TODO: Add attributes
    """

    def __init__(
        self,
        task_items: dict[Hashable, dict[str, dict[str, Any]]],
        agent_properties: dict[str, Any],
    ):
        """
        Initialize the task.

        Args:
            task_items dict[Hashable, dict[str, dict[str, Any]]]: Dictionary of the
                items of the task. The keys are descriptive names of the items
                (value does not matter) and the values are dictionaries containing
                the properties and relations of the items.
            agent_properties (dict[str, Any]): The required properties of the agent.
        """
        self.task_items = task_items
        self.agent_properties = agent_properties

        # Separate item properties and relations
        self.item_properties: dict[Hashable, dict[str, dict[str, Any]]] = {
            item_id: item_dict["properties"]
            for item_id, item_dict in self.task_items.items()
        }
        self.item_relations: dict[Hashable, dict[str, dict[str, list[Hashable]]]] = {
            item_id: item_dict["relations"]
            for item_id, item_dict in self.task_items.items()
        }

        # Infer properties required for candidate objects of the scene
        # Add required properties from item properties
        self.candidate_required_properties = {
            item_id: {
                prop_to_candidate_required_prop[property_name]
                for property_name in item_property_dict.keys()
                if prop_to_candidate_required_prop[property_name] is not None
            }
            for item_id, item_property_dict in self.item_properties.items()
        }
        # Add required properties from item relations
        for item_id, item_relation_dict in self.item_relations.items():
            self.candidate_required_properties[item_id].update(
                {
                    relations_to_candidate_required_prop[relation_name]
                    for relation_name in item_relation_dict.keys()
                    if relations_to_candidate_required_prop[relation_name] is not None
                }
            )

        # Dictionary containing the sets of the ids of candidate objects for each task item
        self.candidates = {item_id: set() for item_id in self.task_items.keys()}

        # Dictionaries containing the information whether the reward of the objects property or the relation have been given
        self.intermediate_property_rewards = {
            item_id: {prop_name: False for prop_name in item_property_dict.keys()}
            for item_id, item_property_dict in self.item_properties.items()
        }
        self.intermediate_relation_rewards = {
            item_id: {
                relation_name: [False] * len(related_item_ids)
                for relation_name, related_item_ids in item_relation_dict.items()
            }
            for item_id, item_relation_dict in self.item_relations.items()
        }

    def reset(self, event: EventLike) -> None:
        """
        Reset the task with the information of the scene.

        Initialize the candidates dictionary with the objects of the scene
        that can correspond to items of the task.
        Initialize the intermediate_rewards dictionary with the properties
        and relations that already exist in the scene.

        Args:
            event (EventLike): Event corresponding to the state of the scene
            at the beginning of the episode.
        """
        scene_objects = {obj["objectId"]: obj for obj in event.metadata["objects"]}

        # Initialize candidates dictionary
        # For each object in the scene, check if it can correspond to an item of the task
        for candidate_id, obj_metadata in scene_objects.items():
            for (
                item_id,
                candidate_required_property_dict,
            ) in self.candidate_required_properties.items():
                # Check if the object has all the required properties
                for property_name in candidate_required_property_dict:
                    if not obj_metadata[property_name]:
                        break
                else:
                    # If the object has all the required properties, add it to the candidates
                    self.candidates[item_id].add(candidate_id)

        # Update intermediate rewards dictionaries
        self._update_intermediate_rewards(scene_objects)

    def _update_intermediate_rewards(
        self, scene_objects: dict[str, dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Update the intermediate rewards dictionary in place with the information of
        the scene objects and return the number of properties and relations that have
        been completed during the last step and if the whole task is completed.

        Args:
            scene_objects (dict[str, dict[str, Any]]): Dictionary of the objects of
                the scene indexed by their id.

        Returns:
            nb_new_completed_properties (int): Number of properties that have been completed
                during the last step.
            nb_new_completed_relations (int): Number of relations that have been completed
                during the last step.
        """
        nb_new_completed_properties = 0
        nb_new_completed_relations = 0

        for item_id in self.task_items.keys():
            candidates = self.candidates[item_id]
            item_property_dict = self.item_properties[item_id]
            item_relation_dict = self.item_relations[item_id]
            intermediate_property_rewards = self.intermediate_property_rewards[item_id]
            intermediate_relation_rewards = self.intermediate_relation_rewards[item_id]
            for candidate_id in candidates:
                candidate_metadata = scene_objects[candidate_id]
                # Properties intermediate rewards
                for property_name in item_property_dict.keys():
                    # If a candidate object already has a property, we don't need to give the reward again
                    if (
                        not intermediate_property_rewards[property_name]
                        and candidate_metadata[property_name]
                        == item_property_dict[property_name]
                    ):
                        intermediate_property_rewards[property_name] = True
                        nb_new_completed_properties += 1
                # Relations intermediate rewards
                for relation_name, related_item_ids in item_relation_dict.items():
                    # Each relation type has a different treatment
                    if relation_name == "contains":
                        for i, related_item_id in enumerate(related_item_ids):
                            # We don't need to check if the relation has already been completed
                            if not intermediate_relation_rewards[relation_name][i]:
                                for contained_object_id in candidate_metadata[
                                    "receptacleObjectIds"
                                ]:
                                    if (
                                        contained_object_id
                                        in self.candidates[related_item_id]
                                    ):
                                        intermediate_relation_rewards[relation_name][
                                            i
                                        ] = True
                                        nb_new_completed_relations += 1
                    if relation_name == "is_contained":
                        pass
                    else:
                        raise NotImplementedError

        return nb_new_completed_properties, nb_new_completed_relations

    def get_task_state(self, event: EventLike) -> dict[str, Union[int, bool]]:
        """
        Check every item properties and relations and return the state of the task.

        The state of the task is a dictionary containing the following keys:
        - "newly_completed_properties": Number of item properties that have been completed
            during the last step.
        - "newly_completed_relations": Number of relations that have been completed
            during the last step.
        - "remaining_intermediate_properties_rewards": Number of item properties that
            have never been completed yet.
        - "remaining_intermediate_relations_rewards": Number of relations that have
            never been completed yet.
        - "missing_properties": Number of item properties that are missing in the scene.
        - "missing_relations": Number of relations that are missing in the scene.
        - "completed": Whether the task has been fully completed.

        Args:
            event (EventLike): Event corresponding to the state of the scene.

        Returns:
            task_state (dict[str, Any]): State of the task.
        """
        # Information used for intermediate rewards
        property_completion_dict = {
            item_id: {prop_name: False for prop_name in item_property_dict.keys()}
            for item_id, item_property_dict in self.item_properties.items()
        }
        relation_completion_dict = {
            item_id: {
                relation_name: [False] * len(related_item_ids)
                for relation_name, related_item_ids in item_relation_dict.items()
            }
            for item_id, item_relation_dict in self.item_relations.items()
        }

        # Information used for completion
        satisfying_objects = {item_id: None for item_id in self.task_items.keys()}

        scene_objects = {obj["objectId"]: obj for obj in event.metadata["objects"]}

        for item_id in self.task_items.keys():
            item_property_dict = self.item_properties[item_id]
            item_relation_dict = self.item_relations[item_id]
            candidates = self.candidates[item_id]
            for candidate_id in candidates:
                is_satisfying_object = True
                candidate_metadata = scene_objects[candidate_id]
                # Checking properties
                for property_name in item_property_dict.keys():
                    if (
                        candidate_metadata[property_name]
                        == item_property_dict[property_name]
                    ):
                        property_completion_dict[item_id][property_name] = True
                    else:
                        is_satisfying_object = False
                # Checking relations
                for relation_name, related_item_ids in item_relation_dict.items():
                    # Each relation type has a different treatment
                    if relation_name == "contains":
                        for i, related_item_id in enumerate(related_item_ids):
                            for contained_object_id in candidate_metadata[
                                "receptacleObjectIds"
                            ]:
                                if (
                                    contained_object_id
                                    in self.candidates[related_item_id]
                                ):
                                    relation_completion_dict[item_id][relation_name][
                                        i
                                    ] = True
                                else:
                                    is_satisfying_object = False
                    else:
                        raise NotImplementedError
                if is_satisfying_object:
                    satisfying_objects[item_id] = candidate_id
                    break  # We don't need to check the other candidates

        # Counting intermediate rewards
        nb_completed_properties = sum(
            sum(
                1
                for completed in property_completion_dict[item_id].values()
                if completed
            )
            for item_id in self.task_items.keys()
        )
        nb_completed_relations = sum(
            1
            for item_id in self.task_items.keys()
            for completed_list in relation_completion_dict[item_id].values()
            for completed in completed_list
            if completed
        )

        satisfied_items = [
            item_id
            for item_id, satisfying_object in satisfying_objects.items()
            if satisfying_object is not None
        ]
        nb_satisfied_items = len(satisfied_items)
        


        return {
            "newly_completed_properties": nb_new_completed_properties,
            "newly_completed_relations": nb_new_completed_relations,
            "remaining_intermediate_properties_rewards": None,  # TODO: Implement
            "remaining_intermediate_relations_rewards": None,  # TODO: Implement
            "missing_properties": None,  # TODO: Implement
            "missing_relations": None,  # TODO: Implement
            "satified_items": satisfied_items,
            "completed": nb_satisfied_items == len(self.task_items),
        }


class PlaceObject(GraphTask):
    """
    Task for placing a target object in a given receptacle.

    This is equivalent to the pick_and_place_simple task from Alfred.
    """

    def __init__(self, placed_object_type: str, receptacle_type: str):
        """
        Initialize the task.

        Args:
            placed_object_type (str): The type of object to place.
            receptacle_type (str): The type of receptacle to place the object in.
        """
        self.placed_object_type = placed_object_type
        self.receptacle_type = receptacle_type

        target_objects: dict[Hashable, dict[str, dict[str, Any]]] = {
            "receptacle": {
                "properties": {"objectType": self.receptacle_type},
                "relations": {"contains": ["placed_object"]},
            },
            "placed_object": {
                "properties": {"objectType": self.placed_object_type},
                "relations": {"is_contained": ["receptacle"]},
            },
        }
        agent_properties = {}
        super().__init__(target_objects, agent_properties)

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.placed_object_type} in {self.receptacle_type}"


# Task dictionary example
# !! Only equality checks are supported for now, so Callable properties are not supported !!
properties = {
    # Fixed properties
    "objectType": str,
    "isInteractable": bool,
    "receptacle": bool,
    "toggleable": bool,
    "breakable": bool,
    "canFillWithLiquid": bool,
    "dirtyable": bool,
    "canBeUsedUp": bool,
    "cookable": bool,
    "isHeatSource": bool,
    "isColdSource": bool,
    "sliceable": bool,
    "openable": bool,
    "pickupable": bool,
    "moveable": bool,
    "mass": float,
    "salientMaterials": bool,
    # Variable properties
    "position": Callable,
    "rotation": Callable,
    "distance": Callable,
    "visible": bool,
    "isToggled": bool,
    "isBroken": bool,
    "isFilledWithLiquid": bool,
    "fillLiquid": str,  # "water", "wine", "coffe"
    "isDirty": bool,
    "isUsedUp": bool,
    "isCooked": bool,
    "temperature": str,  # "Cold", "Hot", "RoomTemp"
    "isSliced": bool,
    "isOpen": bool,
    "openness": Callable,
    "isPickedUp": bool,
}
relations = {
    "contains": list[str],  # Vertices of the graph
    "is_contained": list[str],  # Vertices of the graph  # Should not use
    "close_to": list[tuple[str, float]],  # Vertices of the graph and distance
}
agent_properties = {
    "cameraHorizon": Callable,
    "position": Callable,
    "rotation": Callable,
    "isStanding": bool,
}
# It's enough to make one graph traversal for each connected component
# to save computation time

# During task initialization, we can check and save each object of the scene
# that can fit for each task so we don't have to check every object at each step

# Still need to find how to decompose the task into subtasks

# ??? Possible to give rewards each time a traversal find a new completed leaf node?
# ? Possible to give rewards each time a task object obtains a new target property?
# ? Need to make sure it's a new property and not a property that was already obtained (or already existed in the scene)
# ? Possible to give reward each time a necessary relation is created with object with target property?
# ? Is it possible to make it also with objects that don't have have all the properties yet?
# ? Yes as long as we check that the object is a correct target object that can ultimately have the good properties
# ? Additionally give bigger reward once the full task is completed

# !! Unused yet !!
property_to_action_group = {
    "agent_properties": {
        "cameraHorizon": "head_rotation_actions",
        "position": "movement_actions",  # TODO: Check if we keep this
        "rotation": "body_rotation_actions",  # TODO: Check if we keep this
        "isStanding": "crouch_actions",
    },
    "object_properties": {
        # First level list represents different sets of possible actions and
        # second level list represents action groups that have to be used together
        "position": [["pickup_put_actions"]],  # TODO: Check if we keep this
        "rotation": [["hand_movement_actions"]],  # TODO: Check if we keep this
        "distance": [["pickup_put_actions"]],  # TODO: Check if we keep this
        "visible": [[]],  # TODO: Check if we can use [[]] or it poses problems
        "isToggled": [["toggle_actions"]],
        "isBroken": [
            ["break_actions"],
            [
                "throw_actions"
            ],  # !! Some objects can be broken by other means (e.g. push/pull or drop) !!
        ],
        # break_actions OR (pickup_put_actions AND throw_actions AND push_pull_actions)
        "isFilledWithLiquid": [
            ["liquid_interaction_actions"],
            ["hand_movement_actions"],
        ],
        "fillLiquid": [
            ["liquid_interaction_actions"],
            ["hand_movement_actions"],
        ],
        "isDirty": [
            ["clean_actions"]
        ],  # !! Some objects can be cleaned by other means (e.g. putin under running water), but not made dirty !!
        "isUsedUp": [["useUp_actions"]],
        "isCooked": [["pickup_put_actions"], ["pickup_put_actions", "toggle_actions"]],
        # pickup_put_actions OR (open_close_actions AND toggle_actions)
        "temperature": [
            ["pickup_put_actions"]
        ],  # !! Sometimes require to open/close (refrigerator and sometimes toggle (microwave) !!
        "isSliced": [["slice_actions"]],
        "isOpen": [["open_close_actions"]],
        "openness": [
            ["open_close_actions"]
        ],  # !! partial_openness is also necessary here !!
        "isPickedUp": [["pickup_put_actions"]],
    },
    "relations": {
        "contains": "movement_actions",
        "is_contained": "movement_actions",
        "close_to": "movement_actions",
    },
}

prop_to_candidate_required_prop = {
    # Variable properties
    "position": "pickupable",
    "rotation": "pickupable",
    "distance": None,
    "visible": None,
    "isToggled": "toggleable",
    "isBroken": "breakable",
    "isFilledWithLiquid": "canFillWithLiquid",
    "fillLiquid": "canFillWithLiquid",
    "isDirty": "dirtyable",
    "isUsedUp": "canBeUsedUp",
    "isCooked": "cookable",
    "temperature": None,
    "isSliced": "sliceable",  # TODO: Check if there is something to be careful about with slicing
    "isOpen": "openable",
    "openness": "openable",
    "isPickedUp": "moveable",
    # Fixed properties
    "objectType": "objectType",
    "isInteractable": "isInteractable",
    "receptacle": "receptacle",
    "toggleable": "toggleable",
    "breakable": "breakable",
    "canFillWithLiquid": "canFillWithLiquid",
    "dirtyable": "dirtyable",
    "canBeUsedUp": "canBeUsedUp",
    "cookable": "cookable",
    "isHeatSource": "isHeatSource",
    "isColdSource": "isColdSource",
    "sliceable": "sliceable",
    "openable": "openable",
    "pickupable": "pickupable",
    "moveable": "moveable",
    "mass": None,
    "salientMaterials": "salientMaterials",
}

relations_to_candidate_required_prop = {
    "contains": "receptacle",
    "is_contained": "pickupable",  # !! Not supported, use contains instead !!
    "close_to": "pickupable",  # !! Not implemented yet !!
}
