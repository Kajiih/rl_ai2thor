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
from rl_thor.envs.tasks.item_prop import (
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
    # === Benchmark tasks ===
    PREPARE_MEAL = "PrepareMeal"
    PREPARE_WATCHING_TV = "PrepareWatchingTV"
    PREPARE_GOING_TO_BED = "PrepareGoingToBed"
    PREPARE_FOR_SHOWER = "PrepareForShower"


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
        Switch of all light sources in the scene.

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

        # === Find sink ===
        sink_id = None
        for obj_metadata in last_event.metadata["objects"]:
            if obj_metadata[SimObjFixedProp.OBJECT_TYPE] == SimObjectType.SINK:
                sink_id = obj_metadata[SimObjFixedProp.OBJECT_ID]
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
                    objectId=sink_id,
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
    # === Benchmark tasks ===
    TaskType.PREPARE_MEAL: PrepareMealTask,
    TaskType.PREPARE_WATCHING_TV: PrepareWatchingTVTask,
    TaskType.PREPARE_GOING_TO_BED: PrepareGoingToBedTask,
    TaskType.PREPARE_FOR_SHOWER: PrepareForShowerTask,
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
