"""
Tasks in RL-THOR environment.

TODO: Finish module docstring.
"""

# %% === Imports ===
from __future__ import annotations

from abc import ABC
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ai2thor.controller import Controller

from rl_thor.envs.actions import Ai2thorAction
from rl_thor.envs.sim_objects import (
    LIGHT_SOURCES,
    SimObjectType,
    SimObjFixedProp,
    SimObjVariableProp,
)
from rl_thor.envs.tasks._item_prop_variable import ReceptacleClearedProp
from rl_thor.envs.tasks.item_prop import (
    IsBrokenProp,
    IsCookedProp,
    IsDirtyProp,
    IsFilledWithLiquidProp,
    IsOpenProp,
    IsPickedUpProp,
    IsSlicedProp,
    IsToggledProp,
    ObjectTypeProp,
    TemperatureProp,
    VisibleProp,
)
from rl_thor.envs.tasks.item_prop_interface import (
    MultiValuePSF,
    TemperatureValue,
)
from rl_thor.envs.tasks.items import (
    ItemId,
)
from rl_thor.envs.tasks.relations import RelationTypeId
from rl_thor.envs.tasks.tasks_interface import GraphTask, TaskDict, TaskItemData, parse_task_description_dict

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
    # === Simple tasks ===
    PICKUP = "Pickup"
    OPEN = "Open"
    OPEN_ANY = "OpenAny"
    COOK = "Cook"
    SLICE_AND_COOK_POTATO = "SliceAndCookPotato"
    BREAK = "Break"
    TOGGLE = "Toggle"
    COOL_DOWN = "CoolDown"
    BRING_CLOSE = "BringClose"
    PLACE_TWO_IN = "PlaceTwoIn"
    POUR_COFFEE = "PourCoffee"
    WATCH_TV = "WatchTV"
    # === Benchmark tasks ===
    PREPARE_MEAL = "PrepareMeal"
    RELAX_ON_SOFA = "RelaxOnSofa"
    READ_BOOK_IN_BED = "ReadBookInBed"
    SETUP_BATH = "SetupBath"
    CLEAN_UP_KITCHEN = "CleanUpKitchen"
    CLEAN_UP_LIVING_ROOM = "CleanUpLivingRoom"
    CLEAN_UP_BEDROOM = "CleanUpBedroom"
    CLEAN_UP_BATHROOM = "CleanUpBathroom"
    PLACE_IN_FILLED_SINK = "PlaceInFilledSink"
    PLACE_3_IN_FILLED_SINK = "Place3InFilledSink"


# === Custom Graph asks ===
class CustomGraphTask(GraphTask):
    """
    Custom graph task.

    Class used to define a task graph with a plane task description dictionary describing the adjacency list of the graph.

    To create a custom graph task, simply instantiate this class with a task description dictionary of this form:
    task_description_dict = {
            "plate_receptacle": {
                "properties": {"objectType": "Plate"},
            },
            "hot_apple": {
                "properties": {"objectType": "Apple", "temperature": "Hot"},
                "relations": {"plate_receptacle": ["contained_in"]},
            },
        }
    """

    def __init__(
        self,
        task_description_dict: dict[str, dict[str, Any]],
        text_description: str | None = None,
    ) -> None:
        """
        Initialize the task.

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

        The text description is used to replace the text_description method of the task.

        Args:
            task_description_dict (TaskDict): Task description dictionary.
            text_description (str): Text description of the task.
        """
        parsed_task_description_dict = parse_task_description_dict(task_description_dict)
        super().__init__(parsed_task_description_dict)
        if text_description is not None:
            self._text_description = text_description
        else:
            self._text_description = "Custom task without text description"

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return self._text_description


# %% == Alfred Tasks ==
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
        task_description_dict: TaskDict = {
            ItemId("receptacle"): TaskItemData(
                properties={ObjectTypeProp(receptacle_type)},
            ),
        }
        for i in range(n):
            task_description_dict[ItemId(f"placed_object_{i}")] = TaskItemData(
                properties={ObjectTypeProp(placed_object_type)},
                relations={ItemId("receptacle"): {RelationTypeId.CONTAINED_IN: {}}},
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
        return {
            ItemId("receptacle"): TaskItemData(
                properties={ObjectTypeProp(receptacle_type)},
            ),
            ItemId("pickupable_receptacle"): TaskItemData(
                properties={ObjectTypeProp(pickupable_receptacle_type)},
                relations={ItemId("receptacle"): {RelationTypeId.CONTAINED_IN: {}}},
            ),
            ItemId("placed_object"): TaskItemData(
                properties={ObjectTypeProp(placed_object_type)},
                relations={ItemId("pickupable_receptacle"): {RelationTypeId.CONTAINED_IN: {}}},
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
            task_description_dict[ItemId(f"placed_object_{i}")].properties.add(IsDirtyProp(False))

        return task_description_dict

    def _reset_preprocess(self, controller: Controller) -> bool:
        """
        Make all instances of placed_object_type dirty.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.
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

        return True

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
            task_description_dict[ItemId(f"placed_object_{i}")].properties.add(TemperatureProp(TemperatureValue.HOT))
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
            task_description_dict[ItemId(f"placed_object_{i}")].properties.add(TemperatureProp(TemperatureValue.COLD))
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
        return {
            ItemId("light_source"): TaskItemData(
                properties={
                    ObjectTypeProp(MultiValuePSF(LIGHT_SOURCES)),
                    IsToggledProp(True),
                },
            ),
            ItemId("looked_at_object"): TaskItemData(
                properties={
                    ObjectTypeProp(looked_at_object_type),
                    IsPickedUpProp(True),
                },
                relations={ItemId("light_source"): {RelationTypeId.CLOSE_TO: {"distance": 1.0}}},
            ),
        }

    def _reset_preprocess(self, controller: Controller) -> bool:  # noqa: PLR6301
        """
        Switch off all light sources in the scene.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

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

        return True

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
                    ObjectTypeProp(picked_up_object_type),
                    IsPickedUpProp(True),
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
                properties={IsOpenProp(True)},
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
                    ObjectTypeProp(opened_object_type),
                    IsOpenProp(True),
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


class Cook(GraphTask):
    """Task for cooking a given object."""

    def __init__(self, cooked_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            cooked_object_type (SimObjectType): The type of object to open.
        """
        self.cooked_object_type = cooked_object_type

        task_description_dict = self._create_task_description_dict(cooked_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, cooked_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            cooked_object_type (SimObjectType): The type of object to open.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("cooked_object"): TaskItemData(
                properties={
                    ObjectTypeProp(cooked_object_type),
                    IsCookedProp(True),
                },
            )
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Cook {self.cooked_object_type}"


class PlaceTwoIn(GraphTask):
    """Task for placing two objects in a receptacle."""

    def __init__(
        self, object_type_1: SimObjectType, object_type_2: SimObjectType, receptacle_type: SimObjectType
    ) -> None:
        """
        Initialize the task.

        Args:
            object_type_1 (SimObjectType): The type of the first object to place.
            object_type_2 (SimObjectType): The type of the second object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the objects in.
        """
        self.object_type_1 = object_type_1
        self.object_type_2 = object_type_2
        self.receptacle_type = receptacle_type

        task_description_dict = self._create_task_description_dict(object_type_1, object_type_2, receptacle_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(
        cls, object_type_1: SimObjectType, object_type_2: SimObjectType, receptacle_type: SimObjectType
    ) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            object_type_1 (SimObjectType): The type of the first object to place.
            object_type_2 (SimObjectType): The type of the second object to place.
            receptacle_type (SimObjectType): The type of receptacle to place the objects in.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("receptacle"): TaskItemData(
                properties={ObjectTypeProp(receptacle_type)},
            ),
            ItemId("object_1"): TaskItemData(
                properties={ObjectTypeProp(object_type_1)},
                relations={ItemId("receptacle"): {RelationTypeId.CONTAINED_IN: {}}},
            ),
            ItemId("object_2"): TaskItemData(
                properties={ObjectTypeProp(object_type_2)},
                relations={ItemId("receptacle"): {RelationTypeId.CONTAINED_IN: {}}},
            ),
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.object_type_1} and {self.object_type_2} in {self.receptacle_type}"


class Break(GraphTask):
    """Task for breaking a given object."""

    def __init__(self, broken_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            broken_object_type (SimObjectType): The type of object to break.
        """
        self.broken_object_type = broken_object_type

        task_description_dict = self._create_task_description_dict(broken_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, broken_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            broken_object_type (SimObjectType): The type of object to break.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("broken_object"): TaskItemData(
                properties={
                    ObjectTypeProp(broken_object_type),
                    IsBrokenProp(True),
                },
            )
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Break {self.broken_object_type}"


class Toggle(GraphTask):
    """Task for toggling a given object."""

    def __init__(self, toggled_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            toggled_object_type (SimObjectType): The type of object to toggle.
        """
        self.toggled_object_type = toggled_object_type

        task_description_dict = self._create_task_description_dict(toggled_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, toggled_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            toggled_object_type (SimObjectType): The type of object to toggle.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("toggled_object"): TaskItemData(
                properties={
                    ObjectTypeProp(toggled_object_type),
                    IsToggledProp(True),
                },
            )
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Toggle {self.toggled_object_type}"


class CoolDown(GraphTask):
    """Task for cooling down a given object."""

    def __init__(self, cooled_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            cooled_object_type (SimObjectType): The type of object to cool down.
        """
        self.cooled_object_type = cooled_object_type

        task_description_dict = self._create_task_description_dict(cooled_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, cooled_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            cooled_object_type (SimObjectType): The type of object to cool down.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("cooled_object"): TaskItemData(
                properties={
                    ObjectTypeProp(cooled_object_type),
                    TemperatureProp(TemperatureValue.COLD),
                },
            )
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Cool down {self.cooled_object_type}"


class BringClose(GraphTask):
    """Task for bringing a given object close to another object."""

    def __init__(self, object_type_1: SimObjectType, object_type_2: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            object_type_1 (SimObjectType): The type of the first object.
            object_type_2 (SimObjectType): The type of the second object.
        """
        self.object_type_1 = object_type_1
        self.object_type_2 = object_type_2

        task_description_dict = self._create_task_description_dict(object_type_1, object_type_2)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, object_type_1: SimObjectType, object_type_2: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            object_type_1 (SimObjectType): The type of the first object.
            object_type_2 (SimObjectType): The type of the second object.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("object_1"): TaskItemData(
                properties={ObjectTypeProp(object_type_1)},
            ),
            ItemId("object_2"): TaskItemData(
                properties={ObjectTypeProp(object_type_2)},
                relations={ItemId("object_1"): {RelationTypeId.CLOSE_TO: {"distance": 0.5}}},
            ),
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Bring {self.object_type_1} close to {self.object_type_2}"


class SliceAndCookPotato(GraphTask):
    """Task for slicing and cooking a potato."""

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
            ItemId("cooked_potato_sliced"): TaskItemData(
                properties={
                    ObjectTypeProp(MultiValuePSF({SimObjectType.POTATO, SimObjectType.POTATO_SLICED})),
                    IsSlicedProp(True),
                    IsCookedProp(True),
                },
            )
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Cook a slice of potato"


class PourCoffee(GraphTask):
    """
    Task for pouring coffee.

    The Agent has to pickup a mug, place it in a coffee machine, and then toggle the coffee machine.
    """

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
            ItemId("mug"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.MUG)},
                relations={ItemId("coffee_machine"): {RelationTypeId.CONTAINED_IN: {}}},
            ),
            ItemId("coffee_machine"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.COFFEE_MACHINE),
                    IsToggledProp(True),
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Pour coffee"


class WatchTV(GraphTask):
    """
    Task for watching TV.

    The agent has to switch off the light switch, pick up the remote control, and switch on the TV.
    """

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
            ItemId("light_switch"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.LIGHT_SWITCH),
                    IsToggledProp(False),
                },
            ),
            ItemId("remote_control"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.REMOTE_CONTROL),
                    IsPickedUpProp(True),
                },
            ),
            ItemId("tv"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.TELEVISION),
                    IsToggledProp(True),
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """Return a text description of the task."""
        return "Watch TV"


# === Complex Tasks ===
# TODO: Add FillLiquid = Water
class PrepareMealTaskOld(GraphTask):
    """
    old task for preparing a meal.

    !! We changed it because of some impractical behavior of AI2THOR preventing from putting cooked egg in a plate !!

    The task requires to put on a counter top a plate with a cooked cracked egg inside and a fresh
    cup of water.

    This task is supposed to be used with Kitchen scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("counter_top"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.COUNTER_TOP)},
            ),
            ItemId("plate"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.PLATE)},
                relations={
                    ItemId("counter_top"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("cooked_cracked_egg"): TaskItemData(
                properties={
                    ObjectTypeProp(MultiValuePSF({SimObjectType.EGG, SimObjectType.EGG_CRACKED})),
                    IsSlicedProp(True),
                    IsCookedProp(True),
                },
                relations={
                    ItemId("plate"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("cup_of_water"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.CUP),
                    IsFilledWithLiquidProp(True),
                    TemperatureProp(TemperatureValue.COLD),
                },
                relations={
                    ItemId("counter_top"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare a meal by putting a plate with a cooked cracked egg inside and a fresh cup of water on a counter top"


class PrepareMealTask(GraphTask):
    """
    Task for preparing a meal.

    The task requires to put on a counter top a plate with a cooked potato slice with a fork and a
    knife on the same counter top.

    This task is supposed to be used with Kitchen scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("counter_top"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.COUNTER_TOP)},
            ),
            ItemId("plate"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.PLATE)},
                relations={
                    ItemId("counter_top"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("cooked_potato_sliced"): TaskItemData(
                properties={
                    ObjectTypeProp(MultiValuePSF({SimObjectType.POTATO, SimObjectType.POTATO_SLICED})),
                    IsSlicedProp(True),
                    IsCookedProp(True),
                },
                relations={
                    ItemId("plate"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("fork"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.FORK),
                },
                relations={
                    ItemId("counter_top"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("knife"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.KNIFE),
                },
                relations={
                    ItemId("counter_top"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare a meal by putting a plate with a cooked potato slice with a fork and a knife on the same counter top"


class ExtendedPrepareMealTask(PrepareMealTask):
    """Extended version of the PrepareMealTask where the plate also has to be cleaned."""

    @classmethod
    def _create_task_description_dict(cls) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict()
        task_description_dict[ItemId("plate")].properties.add(IsDirtyProp(False))

        return task_description_dict

    def _reset_preprocess(self, controller: Controller) -> bool:
        """
        Make the plate dirty.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            if obj_metadata[SimObjFixedProp.OBJECT_TYPE] == SimObjectType.PLATE:
                controller.step(
                    action=Ai2thorAction.DIRTY_OBJECT,
                    objectId=obj_metadata[SimObjFixedProp.OBJECT_ID],
                    forceAction=True,
                )
        return super()._reset_preprocess(controller)

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare a meal by putting a plate with a cooked potato slice with a fork and a knife on the same counter top"


# !! Broken because bowls doesn't have plates as a compatible receptacle
class PileUpDishes(GraphTask):
    """
    Task for piling up dishes.

    The task requires putting a spoon in a bowl, the bowl on a plate, and the plate on a counter top.

    This task is supposed to be used with Kitchen scenes.

    This task is NOT used for the RL-THOR benchmark.
    """

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
            ItemId("counter_top"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.COUNTER_TOP)},
            ),
            ItemId("plate"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.PLATE)},
                relations={
                    ItemId("counter_top"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("bowl"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.BOWL)},
                relations={
                    ItemId("plate"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("spoon"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SPOON)},
                relations={
                    ItemId("bowl"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Pile up dishes by putting a spoon in a bowl, the bowl on a plate, and the plate on a counter top "


class ArrangeCutleryTask(GraphTask):
    """
    Task for arranging cutlery on a plate.

    The task requires putting a knife, a fork, and a spoon on a plate, and placing the plate on a counter top.

    This task is supposed to be used with Kitchen scenes.

    This task is NOT used for the RL-THOR benchmark.
    """

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
            ItemId("counter_top"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.COUNTER_TOP)},
            ),
            ItemId("plate"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.PLATE)},
                relations={
                    ItemId("counter_top"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("knife"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.KNIFE)},
                relations={
                    ItemId("plate"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("fork"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.FORK)},
                relations={
                    ItemId("plate"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("spoon"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SPOON)},
                relations={
                    ItemId("plate"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return (
            "Arrange cutlery by placing a knife, a fork, and a spoon on a plate, and placing the plate on a counter top"
        )


class WashCutleryTask(GraphTask):
    """
    Task for washing cutlery.

    The task requires putting a knife, a fork, and a spoon in a sink basin and toggling a faucet.

    This task is supposed to be used with Kitchen scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("sink_basin"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SINK_BASIN)},
            ),
            ItemId("knife"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.KNIFE)},
                relations={
                    ItemId("sink_basin"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("fork"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.FORK)},
                relations={
                    ItemId("sink_basin"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("spoon"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SPOON)},
                relations={
                    ItemId("sink_basin"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("faucet"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.FAUCET),
                    IsToggledProp(True),
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Wash cutlery by putting a knife, a fork, and a spoon in a sink basin and toggling a faucet"


class PrepareWatchingTVTask(GraphTask):
    """
    Task for preparing for watching TV.

    The task requires to put a newspaper and a switched on laptop on a sofa and look at a turned on TV.

    This task is supposed to be used with Living Room scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("sofa"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SOFA)},
            ),
            ItemId("newspaper"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.NEWSPAPER)},
                relations={
                    ItemId("sofa"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("laptop"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.LAPTOP),
                    IsToggledProp(True),
                },
                relations={
                    ItemId("sofa"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("tv"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.TELEVISION),
                    IsToggledProp(True),
                    VisibleProp(True),
                },
            ),
        }

    def _reset_preprocess(self, controller: Controller) -> bool:  # noqa: PLR6301
        """
        Turn off TVs and turn off and close laptops.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
            # === Turn off TVs ===
            if obj_type == SimObjectType.TELEVISION and obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                controller.step(
                    action=Ai2thorAction.TOGGLE_OBJECT_OFF,
                    objectId=obj_id,
                    forceAction=True,
                )

            # === Turn off and close laptops ===
            if obj_type == SimObjectType.LAPTOP:
                if obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                    controller.step(
                        action=Ai2thorAction.TOGGLE_OBJECT_OFF,
                        objectId=obj_id,
                        forceAction=True,
                    )
                if obj_metadata[SimObjVariableProp.IS_OPEN]:
                    controller.step(
                        action=Ai2thorAction.CLOSE_OBJECT,
                        objectId=obj_id,
                        forceAction=True,
                    )

        return True

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare for watching TV by putting a newspaper and a switched on laptop on a sofa and looking at a turned on TV"


class ExtendedPrepareWatchingTVTask(PrepareWatchingTVTask):
    """Extended version of the PrepareWatchingTVTask where the light switch also has to be turned off."""

    @classmethod
    def _create_task_description_dict(cls) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict()
        task_description_dict[ItemId("light_switch")] = TaskItemData(
            properties={
                ObjectTypeProp(SimObjectType.LIGHT_SWITCH),
                IsToggledProp(False),
            }
        )

        return task_description_dict

    def _reset_preprocess(self, controller: Controller) -> bool:
        """
        Turn on light switches, turn off TVs and turn off and close laptops.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
            # === Turn on light switches ===
            if obj_type == SimObjectType.LIGHT_SWITCH and not obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                controller.step(
                    action=Ai2thorAction.TOGGLE_OBJECT_ON,
                    objectId=obj_id,
                    forceAction=True,
                )

        return super()._reset_preprocess(controller)

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare for watching TV by putting a newspaper and a switched on laptop on a sofa and looking at a turned on TV. Also turn off the light switch"


class ClearDiningTable(GraphTask):
    """
    Task for clearing the dining table.

    The task requires removing all items from a dining table.
    # TODO: Add the fact that no item should be held too

    This task is supposed to be used with Dining Room scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("dining_table"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.DINING_TABLE),
                    ReceptacleClearedProp(True),
                },
            ),
        }

    def _reset_preprocess(self, controller: Controller) -> bool:  # noqa: PLR6301
        """
        Put a pickupable object on the dining table if there is nothing on it.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        """
        last_event: Event = controller.last_event  # type: ignore
        # === Find empty tables ===
        empty_dining_tables = [
            obj_metadata[SimObjFixedProp.OBJECT_ID]
            for obj_metadata in last_event.metadata["objects"]
            if obj_metadata[SimObjFixedProp.OBJECT_TYPE] == SimObjectType.DINING_TABLE
            and len(obj_metadata[SimObjVariableProp.RECEPTACLE_OBJ_IDS]) == 0
        ]
        if not empty_dining_tables:
            return True

        # === Find pickupables ===
        pickupables = [
            obj_metadata[SimObjFixedProp.OBJECT_ID]
            for obj_metadata in last_event.metadata["objects"]
            if obj_metadata[SimObjFixedProp.PICKUPABLE] == True
        ]

        # === Put one pickupable on every empty table
        for i, table_id in enumerate(empty_dining_tables):
            controller.step(
                action=Ai2thorAction.PICKUP_OBJECT,
                objectId=pickupables[i],
                forceAction=True,
            )
            controller.step(
                action=Ai2thorAction.PUT_OBJECT,
                objectId=table_id,
                forceAction=True,
            )

        return True

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Clear the dining table by removing all objects from it"


class PrepareGoingToBedTask(GraphTask):
    """
    Task for preparing for going to bed.

    The task requires to turn off the light switch, turn on a desk lamp and hold an open book close to it.

    This task is supposed to be used with Bedroom scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("light_switch"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.LIGHT_SWITCH),
                    IsToggledProp(False),
                },
            ),
            ItemId("desk_lamp"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.DESK_LAMP),
                    IsToggledProp(True),
                },
            ),
            ItemId("book"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.BOOK),
                    IsOpenProp(True),
                    IsPickedUpProp(True),
                },
                relations={
                    ItemId("desk_lamp"): {RelationTypeId.CLOSE_TO: {"distance": 1.0}},
                },
            ),
        }

    def _reset_preprocess(self, controller: Controller) -> bool:  # noqa: PLR6301
        """
        Turn on light switches, turn off desk lamps and close all open books.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
            # === Turn on light switches ===
            if obj_type == SimObjectType.LIGHT_SWITCH and not obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                controller.step(
                    action=Ai2thorAction.TOGGLE_OBJECT_ON,
                    objectId=obj_id,
                    forceAction=True,
                )

            # === Turn off desk lamps ===
            if obj_type == SimObjectType.DESK_LAMP and obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                controller.step(
                    action=Ai2thorAction.TOGGLE_OBJECT_OFF,
                    objectId=obj_id,
                    forceAction=True,
                )

            # === Close all open books ===
            if obj_type == SimObjectType.BOOK and obj_metadata[SimObjVariableProp.IS_OPEN]:
                controller.step(
                    action=Ai2thorAction.CLOSE_OBJECT,
                    objectId=obj_id,
                    forceAction=True,
                )

        return True

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare for going to bed by turning off the light switch, turning on a desk lamp and holding an open book close to it"


class ExtendedPrepareGoingToBedTask(PrepareGoingToBedTask):
    """Extended version of the PrepareGoingToBedTask where the blinds also have to be closed."""

    @classmethod
    def _create_task_description_dict(cls) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict()
        task_description_dict[ItemId("blinds")] = TaskItemData(
            properties={
                ObjectTypeProp(SimObjectType.BLINDS),
                IsOpenProp(False),
            }
        )

        return task_description_dict

    def _reset_preprocess(self, controller: Controller) -> bool:
        """
        Turn on light switches, turn off desk lamps, close all open books and open blinds.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
            # === Open closed blinds ===
            if obj_type == SimObjectType.BLINDS and obj_metadata[SimObjVariableProp.IS_OPEN]:
                controller.step(
                    action=Ai2thorAction.OPEN_OBJECT,
                    objectId=obj_id,
                    forceAction=True,
                )

        return super()._reset_preprocess(controller)

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare for going to bed by turning off the light switch, turning on a desk lamp and holding an open book close to it. Also close the blinds"


class DoHomeworkOld(GraphTask):
    """
    Task for doing homework.

    The task requires putting an open book and a pencil on the desk, carrying a pen, and looking at the book.

    This task is supposed to be used with Bedroom or Study scenes.

    This task is not used for the RL THOR benchmark.
    """

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
            ItemId("desk"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.DESK)},
            ),
            ItemId("book"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.BOOK),
                    IsOpenProp(True),
                    VisibleProp(True),
                },
                relations={
                    ItemId("desk"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("pencil"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.PENCIL)},
                relations={
                    ItemId("desk"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("pen"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.PEN),
                    IsPickedUpProp(True),
                }
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Do homework by putting an open book and a pencil on the desk, carrying a pen, and looking at the book"


class DoHomework(GraphTask):
    """
    Task for doing homework.

    The task requires turning off and putting a cellphone in a closed drawer, turning off and
    closing the laptop, picking up a pencil, and having a desk visible.

    This task is supposed to be used with Bedroom or Study scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("desk"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.DESK),
                    VisibleProp(True),
                },
            ),
            ItemId("drawer"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.DRAWER),
                    IsOpenProp(False),
                },
            ),
            ItemId("cellphone"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.CELL_PHONE),
                    IsToggledProp(False),
                },
                relations={
                    ItemId("drawer"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("laptop"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.LAPTOP),
                    IsToggledProp(False),
                    IsOpenProp(False),
                }
            ),
            ItemId("pencil"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.PENCIL),
                    IsPickedUpProp(True),
                }
            ),
        }

    @classmethod
    def _reset_preprocess(cls, controller: Controller) -> bool:
        """
        Turn on and open laptops, turn on and pick up cellphones.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore
        holding_item = False

        for obj_metadata in last_event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]

            # === Turn on and open laptops ===
            if obj_type == SimObjectType.LAPTOP:
                if not obj_metadata[SimObjVariableProp.IS_OPEN]:
                    controller.step(
                        action=Ai2thorAction.OPEN_OBJECT,
                        objectId=obj_id,
                        forceAction=True,
                    )
                if not obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                    controller.step(
                        action=Ai2thorAction.TOGGLE_OBJECT_ON,
                        objectId=obj_id,
                        forceAction=True,
                    )

            # === Turn on and pick up cellphones ===
            if obj_type == SimObjectType.CELL_PHONE and not holding_item:
                if not obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                    controller.step(
                        action=Ai2thorAction.TOGGLE_OBJECT_ON,
                        objectId=obj_id,
                        forceAction=True,
                    )
                if not obj_metadata[SimObjVariableProp.IS_PICKED_UP]:
                    controller.step(
                        action=Ai2thorAction.PICKUP_OBJECT,
                        objectId=obj_id,
                        forceAction=True,
                    )
                holding_item = True

        return True

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Do homework by turning off and putting a cellphone in a closed drawer, turning off and closing the laptop, picking up a pencil, and ensuring the desk is visible."


class PrepareForShowerTask(GraphTask):
    """
    Task for preparing for a shower.

    The task requires put a towel on a towel holder, a soap bar in the bathtub and turn on the shower head.

    This task is supposed to be used with Bathroom scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("towel_holder"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.TOWEL_HOLDER)},
            ),
            ItemId("towel"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.TOWEL)},
                relations={
                    ItemId("towel_holder"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("bathtub"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.BATHTUB)},
            ),
            ItemId("soap"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SOAP_BAR)},
                relations={
                    ItemId("bathtub"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("shower_head"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.SHOWER_HEAD),
                    IsToggledProp(True),
                },
            ),
        }

    def _reset_preprocess(self, controller: Controller) -> bool:  # noqa: PLR6301
        """
        Put towels on the floor, soap bars on the sink and turn off the shower head.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore

        # === Find sink basin ===
        sink_basin_id = None
        for obj_metadata in last_event.metadata["objects"]:
            if obj_metadata[SimObjFixedProp.OBJECT_TYPE] == SimObjectType.SINK_BASIN:
                sink_basin_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
                break
        else:
            return False

        for obj_metadata in last_event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
            # === Drop towels that are on a towel holder ===
            if obj_type == SimObjectType.TOWEL and obj_metadata["parentReceptacles"]:
                controller.step(
                    action=Ai2thorAction.PICKUP_OBJECT,
                    objectId=obj_id,
                    forceAction=True,
                    manualInteract=True,
                )
                controller.step(
                    action=Ai2thorAction.DROP_HAND_OBJECT,
                    forceAction=True,
                )

            #  === Put soap bars in the sink ===
            if obj_type == SimObjectType.SOAP_BAR:
                controller.step(
                    action=Ai2thorAction.PICKUP_OBJECT,
                    objectId=obj_id,
                    forceAction=True,
                )
                controller.step(
                    action=Ai2thorAction.PUT_OBJECT,
                    objectId=sink_basin_id,
                    forceAction=True,
                )

            # === Turn off the shower head ===
            if obj_type == SimObjectType.SHOWER_HEAD and obj_metadata[SimObjVariableProp.IS_TOGGLED]:
                controller.step(
                    action=Ai2thorAction.TOGGLE_OBJECT_OFF,
                    objectId=obj_id,
                    forceAction=True,
                )

        return True

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare for a shower by putting a towel on a towel holder, a soap bar in the bathtub and turning on the shower head"


class ExtendedPrepareForShowerTask(PrepareForShowerTask):
    """Extended version of the PrepareForShowerTask where cloths also have to be put in the garbage can (laundry basket)."""

    @classmethod
    def _create_task_description_dict(cls) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = super()._create_task_description_dict()
        task_description_dict[ItemId("garbage_can")] = TaskItemData(
            properties={ObjectTypeProp(SimObjectType.GARBAGE_CAN)},
        )
        task_description_dict[ItemId("cloths")] = TaskItemData(
            properties={ObjectTypeProp(SimObjectType.CLOTH)},
            relations={
                ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
            },
        )

        return task_description_dict

    def _reset_preprocess(self, controller: Controller) -> bool:
        """
        Put towels on the floor, soap bars on the sink, turn off the shower head and put cloths on the floor in front of the agent.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore

        for obj_metadata in last_event.metadata["objects"]:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
            # === Put cloths on the floor in front of the agent ===
            if obj_type == SimObjectType.CLOTH:
                controller.step(
                    action=Ai2thorAction.PICKUP_OBJECT,
                    objectId=obj_id,
                    forceAction=True,
                )
                controller.step(
                    action=Ai2thorAction.DROP_HAND_OBJECT,
                    forceAction=True,
                )

        return super()._reset_preprocess(controller)

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Prepare for a shower by putting a towel on a towel holder, a soap bar in the bathtub and turning on the shower head. Also put cloths in the garbage can"


class CleanToilets(GraphTask):
    """
    Task for cleaning the toilets.

    The task requires putting a toilet paper roll on the toilet paper hanger, opening the toilet lid,
    putting the spray bottle on the toilets, and holding the scrub brush.

    This task is supposed to be used with Bathroom scenes.

    This task is used for the RL THOR benchmark.
    """

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
            ItemId("toilet_paper_hanger"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.TOILET_PAPER_HANGER)},
            ),
            ItemId("toilet_paper_roll"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.TOILET_PAPER)},
                relations={
                    ItemId("toilet_paper_hanger"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("toilet"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.TOILET),
                    IsOpenProp(True),
                },
            ),
            ItemId("spray_bottle"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SPRAY_BOTTLE)},
                relations={
                    ItemId("toilet"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("scrub_brush"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.SCRUB_BRUSH),
                    IsPickedUpProp(True),
                },
            ),
        }

    def _reset_preprocess(self, controller: Controller) -> bool:  # noqa: PLR6301
        """
        Drop toilet paper on the floor if it is on the toilet paper hanger.

        Args:
            controller (Controller): AI2-THOR controller at the beginning of the episode.

        Returns:
            preprocess_successful (bool): Whether the preprocess was successful.
        """
        last_event: Event = controller.last_event  # type: ignore
        objects_metadata = last_event.metadata["objects"]
        organized_metadata = {obj_metadata["objectId"]: obj_metadata for obj_metadata in objects_metadata}

        # Iterate over all objects in the scene
        for obj_metadata in objects_metadata:
            obj_type = obj_metadata[SimObjFixedProp.OBJECT_TYPE]
            obj_id = obj_metadata[SimObjFixedProp.OBJECT_ID]

            # Check if the object is toilet paper on the toilet paper hanger
            if obj_type == SimObjectType.TOILET_PAPER and obj_metadata["parentReceptacles"]:
                parent_obj_id = obj_metadata["parentReceptacles"][0]
                parent_obj_type = organized_metadata[parent_obj_id][SimObjFixedProp.OBJECT_TYPE]
                if parent_obj_type == SimObjectType.TOILET_PAPER_HANGER:
                    # Pick up the toilet paper
                    controller.step(
                        action=Ai2thorAction.PICKUP_OBJECT,
                        objectId=obj_id,
                        forceAction=True,
                        # manualInteract=True,
                    )
                    # Drop it on the floor
                    controller.step(
                        action=Ai2thorAction.DROP_HAND_OBJECT,
                        forceAction=True,
                    )

        return True

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Clean the toilets by putting a toilet paper roll on the hanger, opening the toilet lid, placing a spray bottle on the toilet, and holding a scrub brush"


class CleanUpKitchenTask(GraphTask):
    """
    Task for cleaning up a kitchen.

    The agent has to put an apple, a tomato, a potato and an egg and in the garbage can.
    """

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
            ItemId("garbage_can"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.GARBAGE_CAN)},
            ),
            ItemId("apple"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.APPLE)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("tomato"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.TOMATO)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("potato"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.POTATO)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("egg"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.EGG)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Clean up the kitchen by putting an apple, a tomato, a potato and an egg in the garbage can"


# AlarmClock, CD, CellPhone, Pen, Pencil in Box
class CleanUpBedroomTask(GraphTask):
    """
    Task for cleaning up a bedroom.

    The agent has to put an key chain, a CD, a cell phone, a pen and a pencil in a box.
    """

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
            ItemId("box"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.BOX)},
            ),
            ItemId("key_chain"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.KEY_CHAIN)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("cd"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.CD)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("cell_phone"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.CELL_PHONE)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("pen"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.PEN)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("pencil"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.PENCIL)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Clean up the bedroom by putting an key chain, a CD, a cell phone, a pen and a pencil in a box"


class CleanUpLivingRoomTask(GraphTask):
    """
    Task for cleaning up a living room.

    The agent has to put a credit card, a key chain, a remote control and a watch in a box.
    """

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
            ItemId("box"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.BOX)},
            ),
            ItemId("credit_card"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.CREDIT_CARD)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("key_chain"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.KEY_CHAIN)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("remote_control"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.REMOTE_CONTROL)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("watch"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.WATCH)},
                relations={
                    ItemId("box"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Clean up the living room by putting a credit card, a key chain, a remote control and a watch in a box"


class CleanUpBathroomTask(GraphTask):
    """
    Task for cleaning up a bathroom.

    The agent has to put a piece of cloth, soap bar, soap bottle, spray bottle and toilet paper in the garbage can.
    """

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
            ItemId("garbage_can"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.GARBAGE_CAN)},
            ),
            ItemId("cloth"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.CLOTH)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("soap_bar"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SOAP_BAR)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("soap_bottle"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SOAP_BOTTLE)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("spray_bottle"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SPRAY_BOTTLE)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
            ItemId("toilet_paper"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.TOILET_PAPER)},
                relations={
                    ItemId("garbage_can"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    @classmethod
    def text_description(cls) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return "Clean up the bathroom by putting a piece of cloth, soap bar, soap bottle, spray bottle and toilet paper in the garbage can"


# %% === Other Benchmark Tasks ===
class PlaceInFilledSink(GraphTask):
    """Task for placing a given object in a filled sink."""

    def __init__(self, placed_object_type: SimObjectType) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.
        """
        self.placed_object_type = placed_object_type

        task_description_dict = self._create_task_description_dict(placed_object_type)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, placed_object_type: SimObjectType) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_type (SimObjectType): The type of object to place.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        return {
            ItemId("sink_basin"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SINK_BASIN)},
            ),
            ItemId("faucet"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.FAUCET),
                    IsToggledProp(True),
                },
            ),
            ItemId("placed_object"): TaskItemData(
                properties={ObjectTypeProp(placed_object_type)},
                relations={
                    ItemId("sink_basin"): {RelationTypeId.CONTAINED_IN: {}},
                },
            ),
        }

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.placed_object_type} in a sink basin filled with water"


class Place3InFilledSink(GraphTask):
    """Task for placing a 3 given objects in a filled sink."""

    def __init__(
        self,
        placed_object_type_1: SimObjectType,
        placed_object_type_2: SimObjectType,
        placed_object_type_3: SimObjectType,
    ) -> None:
        """
        Initialize the task.

        Args:
            placed_object_type_1 (SimObjectType): The type of object to place.
            placed_object_type_2 (SimObjectType): The type of object to place.
            placed_object_type_3 (SimObjectType): The type of object to place.
        """
        self.placed_object_types = [placed_object_type_1, placed_object_type_2, placed_object_type_3]

        task_description_dict = self._create_task_description_dict(self.placed_object_types)
        super().__init__(task_description_dict)

    @classmethod
    def _create_task_description_dict(cls, placed_object_types: list[SimObjectType]) -> TaskDict:
        """
        Create the task description dictionary for the task.

        Args:
            placed_object_types (list[SimObjectType]): The types of objects to place.

        Returns:
            task_description_dict (TaskDict): Task description dictionary.
        """
        task_description_dict = {
            ItemId("sink_basin"): TaskItemData(
                properties={ObjectTypeProp(SimObjectType.SINK_BASIN)},
            ),
            ItemId("faucet"): TaskItemData(
                properties={
                    ObjectTypeProp(SimObjectType.FAUCET),
                    IsToggledProp(True),
                },
            ),
        }
        for i, placed_object_type in enumerate(placed_object_types):
            task_description_dict[ItemId(f"placed_object_{i}")] = TaskItemData(
                properties={ObjectTypeProp(placed_object_type)},
                relations={
                    ItemId("sink_basin"): {RelationTypeId.CONTAINED_IN: {}},
                },
            )

        return task_description_dict

    def text_description(self) -> str:
        """
        Return a text description of the task.

        Returns:
            description (str): Text description of the task.
        """
        return f"Place {self.placed_object_types[0]}, {self.placed_object_types[1]} and {self.placed_object_types[2]} in a sink filled with water"


# %% === Constants ===
ALL_TASKS: dict[TaskType, type[GraphTask]]
ALL_TASKS = {
    # === Alfred tasks ===
    TaskType.PLACE_IN: PlaceIn,
    TaskType.PLACE_N_SAME_IN: PlaceNSameIn,
    TaskType.PLACE_WITH_MOVEABLE_RECEP_IN: PlaceWithMoveableRecepIn,
    TaskType.PLACE_CLEANED_IN: PlaceCleanedIn,
    TaskType.PLACE_HEATED_IN: PlaceHeatedIn,
    TaskType.PLACE_COOLED_IN: PlaceCooledIn,
    TaskType.LOOK_IN_LIGHT: LookInLight,
    # === Simple tasks ===
    TaskType.PICKUP: Pickup,
    TaskType.OPEN: Open,
    TaskType.OPEN_ANY: OpenAny,
    TaskType.COOK: Cook,
    TaskType.SLICE_AND_COOK_POTATO: SliceAndCookPotato,
    TaskType.BREAK: Break,
    TaskType.TOGGLE: Toggle,
    TaskType.COOL_DOWN: CoolDown,
    TaskType.BRING_CLOSE: BringClose,
    TaskType.PLACE_TWO_IN: PlaceTwoIn,
    TaskType.POUR_COFFEE: PourCoffee,
    TaskType.WATCH_TV: WatchTV,
    # === Benchmark tasks ===
    TaskType.PREPARE_MEAL: PrepareMealTask,
    TaskType.RELAX_ON_SOFA: PrepareWatchingTVTask,
    TaskType.READ_BOOK_IN_BED: PrepareGoingToBedTask,
    TaskType.SETUP_BATH: PrepareForShowerTask,
    TaskType.CLEAN_UP_KITCHEN: CleanUpKitchenTask,
    TaskType.CLEAN_UP_BEDROOM: CleanUpBedroomTask,
    TaskType.CLEAN_UP_LIVING_ROOM: CleanUpLivingRoomTask,
    TaskType.CLEAN_UP_BATHROOM: CleanUpBathroomTask,
    TaskType.PLACE_IN_FILLED_SINK: PlaceInFilledSink,
    TaskType.PLACE_3_IN_FILLED_SINK: Place3InFilledSink,
}


# %% === Exceptions ===
class UnknownTaskTypeError(ValueError):
    """Exception raised for unknown task types in environment mode config."""

    def __init__(self, task_type: type[TaskType]) -> None:
        self.task_type = task_type
        super().__init__(
            f"Unknown task type '{task_type}' in environment mode config."
            f"Available tasks are {list(ALL_TASKS.keys())}."
            f"If you have defined a new task, make sure to add it to the ALL_TASKS dictionary of the envs.tasks.tasks module."
        )
