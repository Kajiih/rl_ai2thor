"""
Tasks in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

# %% === Imports ===
from __future__ import annotations

from abc import ABC
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rl_ai2thor.envs.actions import Ai2thorAction
from rl_ai2thor.envs.sim_objects import (
    LIGHT_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjVariableProp,
)
from rl_ai2thor.envs.tasks.item_prop_interface import (
    MultiValuePSF,
    SingleValuePSF,
    TemperatureValue,
)
from rl_ai2thor.envs.tasks.items import (
    ItemId,
)
from rl_ai2thor.envs.tasks.relations import RelationTypeId
from rl_ai2thor.envs.tasks.tasks_interface import GraphTask, TaskDict, TaskItemData

if TYPE_CHECKING:
    from ai2thor.controller import Controller
    from ai2thor.server import Event


# === Enums ===
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


# %% === Custom Tasks ===
# === Simple Tasks
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


# === Complex Tasks ===
# TODO: Implement

# %% === Constants ===
ALL_TASKS: dict[TaskType, type[GraphTask]]
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
