"""Task information for RL-THOR benchmark."""

from enum import StrEnum
from typing import Any

from rl_thor.envs.sim_objects import SimObjectType
from rl_thor.envs.tasks.tasks import TaskType


# %% List of available tasks
class AvailableTask(StrEnum):
    """Available tasks for training."""

    # Complex tasks
    CLEAN_UP_KITCHEN = TaskType.CLEAN_UP_KITCHEN
    CLEAN_UP_LIVING_ROOM = TaskType.CLEAN_UP_LIVING_ROOM
    CLEAN_UP_BEDROOM = TaskType.CLEAN_UP_BEDROOM
    CLEAN_UP_BATHROOM = TaskType.CLEAN_UP_BATHROOM
    MULTI_TASK_4 = "MultiTask4"
    PREPARE_MEAL = TaskType.PREPARE_MEAL
    WASH_CUTLERY = TaskType.WASH_CUTLERY
    RELAX_ON_SOFA = TaskType.RELAX_ON_SOFA
    CLEAR_DINING_TABLE = TaskType.CLEAR_DINING_TABLE
    READ_BOOK_IN_BED = TaskType.READ_BOOK_IN_BED
    DO_HOMEWORK = TaskType.DO_HOMEWORK
    SETUP_BATH = TaskType.SETUP_BATH
    CLEAN_TOILETS = TaskType.CLEAN_TOILETS
    MULTI_TASK_FULL = "MultiTaskFull"

    # Gradual tasks
    # 1 item
    BREAK_BOWL = "BreakBowl"
    OPEN_TOILET = "OpenToilet"
    OPEN_BOOK = "OpenBook"
    SWITCH_ON_TV = "SwitchOnTV"
    PICKUP_POTATO = "PickupPotato"
    # 2 items
    PLACE_POTATO_IN_FRIDGE = "PlacePotatoInFridge"
    PLACE_NEWSPAPER_ON_SOFA = "PlaceNewspaperOnSofa"
    BRING_TOWEL_CLOTH_CLOSE = "BringTowelClothesClose"
    POUR_COFFEE = TaskType.POUR_COFFEE
    LOOK_BOOK_IN_LIGHT = "LookBookInLight"
    # 3 items
    WATCH_TV = TaskType.WATCH_TV
    PLACE_PEN_BOOK_ON_DESK = "PlacePenBookOnDesk"
    PLACE_TOMATO_POTATO_IN_FRIDGE = "PlaceTomatoPotatoInFridge"
    SETUP_BATH_SIMPLE = "SetupBathSimple"
    READ_BOOK_IN_BED_SIMPLE = "ReadBookInBedSimple"


task_blueprints_configs = {
    # Complex tasks
    AvailableTask.PREPARE_MEAL: {
        "task_type": TaskType.PREPARE_MEAL,
        "args": {},
        "scenes": [
            "FloorPlan1",
            "FloorPlan2",
            "FloorPlan3",
            "FloorPlan4",
            "FloorPlan5",
            "FloorPlan6",
            "FloorPlan7",
            "FloorPlan8",
            "FloorPlan9",
            "FloorPlan10",
            "FloorPlan11",
            "FloorPlan12",
            "FloorPlan13",
            "FloorPlan14",
            "FloorPlan15",
            "FloorPlan16",
            "FloorPlan17",
            "FloorPlan18",
            "FloorPlan19",
            "FloorPlan20",
            "FloorPlan21",
            "FloorPlan22",
            "FloorPlan23",
            "FloorPlan24",
            "FloorPlan25",
            "FloorPlan26",
            "FloorPlan27",
            "FloorPlan28",
            "FloorPlan29",
            "FloorPlan30",
        ],
    },
    AvailableTask.WASH_CUTLERY: {
        "task_type": TaskType.WASH_CUTLERY,
        "args": {},
        "scenes": [
            "FloorPlan1",
            "FloorPlan2",
            "FloorPlan3",
            "FloorPlan4",
            "FloorPlan5",
            "FloorPlan6",
            "FloorPlan7",
            "FloorPlan8",
            "FloorPlan9",
            "FloorPlan10",
            "FloorPlan11",
            "FloorPlan12",
            "FloorPlan13",
            "FloorPlan14",
            "FloorPlan15",
            "FloorPlan16",
            "FloorPlan17",
            "FloorPlan18",
            "FloorPlan19",
            "FloorPlan20",
            "FloorPlan21",
            "FloorPlan22",
            "FloorPlan23",
            "FloorPlan24",
            "FloorPlan25",
            "FloorPlan26",
            "FloorPlan27",
            "FloorPlan28",
            "FloorPlan29",
            "FloorPlan30",
        ],
    },
    AvailableTask.RELAX_ON_SOFA: {
        "task_type": TaskType.RELAX_ON_SOFA,
        "args": {},
        "scenes": [
            "FloorPlan201",
            "FloorPlan203",
            "FloorPlan209",
            "FloorPlan210",
            "FloorPlan211",
            "FloorPlan212",
            "FloorPlan214",
            "FloorPlan215",
            "FloorPlan216",
            "FloorPlan218",
            "FloorPlan219",
            "FloorPlan222",
            "FloorPlan224",
            "FloorPlan225",
            "FloorPlan226",
            "FloorPlan227",
            "FloorPlan228",
            "FloorPlan230",
        ],
    },
    AvailableTask.CLEAR_DINING_TABLE: {
        "task_type": TaskType.CLEAR_DINING_TABLE,
        "args": {},
        "scenes": [
            "FloorPlan201",
            "FloorPlan203",
            "FloorPlan204",
            "FloorPlan205",
            "FloorPlan208",
            "FloorPlan211",
            "FloorPlan216",
            "FloorPlan218",
            "FloorPlan220",
            "FloorPlan221",
            "FloorPlan223",
            "FloorPlan227",
            "FloorPlan228",
            "FloorPlan230",
        ],
    },
    AvailableTask.READ_BOOK_IN_BED: {
        "task_type": TaskType.READ_BOOK_IN_BED,
        "args": {},
        "scenes": [
            # "FloorPlan201",
            # "FloorPlan224",
            "FloorPlan301",
            "FloorPlan302",
            "FloorPlan303",
            "FloorPlan304",
            "FloorPlan305",
            "FloorPlan306",
            "FloorPlan307",
            "FloorPlan308",
            "FloorPlan309",
            "FloorPlan310",
            "FloorPlan311",
            "FloorPlan312",
            "FloorPlan313",
            "FloorPlan314",
            "FloorPlan315",
            "FloorPlan316",
            "FloorPlan317",
            "FloorPlan318",
            "FloorPlan319",
            "FloorPlan320",
            "FloorPlan321",
            "FloorPlan322",
            "FloorPlan323",
            "FloorPlan324",
            "FloorPlan325",
            "FloorPlan326",
            "FloorPlan327",
            "FloorPlan328",
            "FloorPlan329",
            "FloorPlan330",
        ],
    },
    AvailableTask.DO_HOMEWORK: {
        "task_type": TaskType.DO_HOMEWORK,
        "args": {},
        "scenes": [
            "FloorPlan301",
            "FloorPlan302",
            "FloorPlan303",
            "FloorPlan304",
            "FloorPlan305",
            "FloorPlan306",
            "FloorPlan307",
            "FloorPlan308",
            "FloorPlan309",
            "FloorPlan310",
            "FloorPlan311",
            "FloorPlan312",
            "FloorPlan313",
            "FloorPlan314",
            "FloorPlan315",
            "FloorPlan316",
            "FloorPlan318",
            "FloorPlan321",
            "FloorPlan323",
            "FloorPlan326",
            "FloorPlan327",
            "FloorPlan328",
            "FloorPlan329",
        ],
    },
    AvailableTask.READ_BOOK_IN_BED_SIMPLE: {
        "task_type": TaskType.READ_BOOK_IN_BED,
        "args": {},
        "scenes": ["FloorPlan301"],
    },
    AvailableTask.CLEAN_TOILETS: {
        "task_type": TaskType.CLEAN_TOILETS,
        "args": {},
        "scenes": [
            "FloorPlan401",
            "FloorPlan402",
            "FloorPlan403",
            "FloorPlan404",
            "FloorPlan405",
            "FloorPlan406",
            "FloorPlan407",
            "FloorPlan408",
            "FloorPlan409",
            "FloorPlan410",
            "FloorPlan411",
            "FloorPlan412",
            "FloorPlan413",
            "FloorPlan414",
            "FloorPlan415",
            "FloorPlan416",
            "FloorPlan417",
            "FloorPlan418",
            "FloorPlan419",
            "FloorPlan420",
            "FloorPlan421",
            "FloorPlan422",
            "FloorPlan423",
            "FloorPlan424",
            "FloorPlan425",
            "FloorPlan426",
            "FloorPlan427",
            "FloorPlan428",
            "FloorPlan429",
            "FloorPlan430",
        ],
    },
    AvailableTask.SETUP_BATH: {
        "task_type": TaskType.SETUP_BATH,
        "args": {},
        "scenes": [
            "FloorPlan401",
            "FloorPlan402",
            "FloorPlan403",
            "FloorPlan404",
            "FloorPlan407",
            "FloorPlan413",
            "FloorPlan415",
            "FloorPlan419",
            "FloorPlan422",
            "FloorPlan423",
            "FloorPlan426",
            "FloorPlan427",
        ],
    },
    AvailableTask.SETUP_BATH_SIMPLE: {
        "task_type": TaskType.SETUP_BATH,
        "args": {},
        "scenes": ["FloorPlan401"],
    },
    AvailableTask.CLEAN_UP_KITCHEN: {
        "task_type": TaskType.CLEAN_UP_KITCHEN,
        "args": {},
        "scenes": [
            "FloorPlan1",
            "FloorPlan2",
            "FloorPlan3",
            "FloorPlan4",
            "FloorPlan5",
            "FloorPlan6",
            "FloorPlan7",
            "FloorPlan8",
            "FloorPlan9",
            "FloorPlan10",
            "FloorPlan11",
            "FloorPlan12",
            "FloorPlan13",
            "FloorPlan14",
            "FloorPlan15",
            "FloorPlan16",
            "FloorPlan17",
            "FloorPlan18",
            "FloorPlan19",
            "FloorPlan20",
            "FloorPlan21",
            "FloorPlan22",
            "FloorPlan23",
            "FloorPlan24",
            "FloorPlan25",
            "FloorPlan26",
            "FloorPlan27",
            "FloorPlan28",
            "FloorPlan29",
            "FloorPlan30",
        ],
    },
    AvailableTask.CLEAN_UP_LIVING_ROOM: {
        "task_type": TaskType.CLEAN_UP_LIVING_ROOM,
        "args": {},
        "scenes": [
            "FloorPlan201",
            "FloorPlan202",
            "FloorPlan203",
            "FloorPlan205",
            "FloorPlan207",
            "FloorPlan209",
            "FloorPlan210",
            "FloorPlan213",  # Doesn't work with the task?
            "FloorPlan215",
            "FloorPlan217",
            "FloorPlan219",
            "FloorPlan222",
            "FloorPlan225",
            "FloorPlan226",
            "FloorPlan228",
            "FloorPlan230",
        ],
    },
    AvailableTask.CLEAN_UP_BEDROOM: {
        "task_type": TaskType.CLEAN_UP_BEDROOM,
        "args": {},
        "scenes": [
            "FloorPlan301",
            "FloorPlan303",
            "FloorPlan304",
            "FloorPlan305",
            "FloorPlan310",
            "FloorPlan313",
            "FloorPlan314",
            "FloorPlan316",
            "FloorPlan317",
            "FloorPlan318",
            "FloorPlan319",
            "FloorPlan329",
        ],
    },
    AvailableTask.CLEAN_UP_BATHROOM: {
        "task_type": TaskType.CLEAN_UP_BATHROOM,
        "args": {},
        "scenes": [
            "FloorPlan401",
            "FloorPlan402",
            "FloorPlan403",
            "FloorPlan404",
            "FloorPlan405",
            "FloorPlan406",
            "FloorPlan407",
            "FloorPlan408",
            "FloorPlan409",
            "FloorPlan410",
            "FloorPlan411",
            "FloorPlan412",
            "FloorPlan413",
            "FloorPlan414",
            "FloorPlan415",
            "FloorPlan416",
            "FloorPlan417",
            "FloorPlan418",
            "FloorPlan419",
            "FloorPlan420",
            "FloorPlan421",
            "FloorPlan422",
            "FloorPlan423",
            "FloorPlan424",
            "FloorPlan425",
            "FloorPlan426",
            "FloorPlan427",
            "FloorPlan428",
            "FloorPlan429",
            "FloorPlan430",
        ],
    },
    # 1 item tasks
    AvailableTask.BREAK_BOWL: {
        "task_type": TaskType.BREAK,
        "args": {"broken_object_type": SimObjectType.BOWL},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.SWITCH_ON_TV: {
        "task_type": TaskType.TOGGLE,
        "args": {"toggled_object_type": SimObjectType.TELEVISION},
        "scenes": ["FloorPlan201"],
    },
    AvailableTask.OPEN_BOOK: {
        "task_type": TaskType.OPEN,
        "args": {"opened_object_type": SimObjectType.BOOK},
        "scenes": ["FloorPlan301"],
    },
    AvailableTask.OPEN_TOILET: {
        "task_type": TaskType.OPEN,
        "args": {"opened_object_type": SimObjectType.TOILET},
        "scenes": ["FloorPlan401"],
    },
    AvailableTask.PICKUP_POTATO: {
        "task_type": TaskType.PICKUP,
        "args": {"picked_up_object_type": SimObjectType.POTATO},
        "scenes": ["FloorPlan1"],
    },
    # 2 items tasks
    AvailableTask.POUR_COFFEE: {
        "task_type": TaskType.POUR_COFFEE,
        "args": {},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_POTATO_IN_FRIDGE: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.POTATO, "receptacle_type": SimObjectType.FRIDGE},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.LOOK_BOOK_IN_LIGHT: {
        "task_type": TaskType.LOOK_IN_LIGHT,
        "args": {"looked_at_object_type": SimObjectType.BOOK},
        "scenes": ["FloorPlan301"],
    },
    AvailableTask.PLACE_NEWSPAPER_ON_SOFA: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.NEWSPAPER, "receptacle_type": SimObjectType.SOFA},
        "scenes": ["FloorPlan201"],
    },
    AvailableTask.BRING_TOWEL_CLOTH_CLOSE: {
        "task_type": TaskType.BRING_CLOSE,
        "args": {"object_type_1": SimObjectType.TOWEL, "object_type_2": SimObjectType.CLOTH},
        "scenes": ["FloorPlan401"],
    },
    # 3 items tasks
    AvailableTask.PLACE_PEN_BOOK_ON_DESK: {
        "task_type": TaskType.PLACE_TWO_IN,
        "args": {
            "object_type_1": SimObjectType.PEN,
            "object_type_2": SimObjectType.BOOK,
            "receptacle_type": SimObjectType.DESK,
        },
        "scenes": ["FloorPlan301"],
    },
    AvailableTask.WATCH_TV: {
        "task_type": TaskType.WATCH_TV,
        "args": {},
        "scenes": ["FloorPlan201"],
    },
    AvailableTask.PLACE_TOMATO_POTATO_IN_FRIDGE: {
        "task_type": TaskType.PLACE_TWO_IN,
        "args": {
            "object_type_1": SimObjectType.TOMATO,
            "object_type_2": SimObjectType.POTATO,
            "receptacle_type": SimObjectType.FRIDGE,
        },
        "scenes": ["FloorPlan1"],
    },
}


def get_action_groups_override_config(task: AvailableTask) -> dict[str, Any]:
    """Return the action groups for the task."""
    action_groups = {
        "open_close_actions": False,
        "toggle_actions": False,
        "slice_actions": False,
    }
    # === Enable opening and closing ===
    if task in {
        AvailableTask.PLACE_POTATO_IN_FRIDGE,
        AvailableTask.PREPARE_MEAL,
        AvailableTask.WASH_CUTLERY,
        AvailableTask.RELAX_ON_SOFA,
        AvailableTask.CLEAR_DINING_TABLE,
        AvailableTask.READ_BOOK_IN_BED,
        AvailableTask.DO_HOMEWORK,
        AvailableTask.SETUP_BATH,
        AvailableTask.CLEAN_TOILETS,
        AvailableTask.OPEN_TOILET,
        AvailableTask.OPEN_BOOK,
        AvailableTask.PLACE_TOMATO_POTATO_IN_FRIDGE,
        AvailableTask.CLEAN_UP_KITCHEN,
        AvailableTask.CLEAN_UP_LIVING_ROOM,
        AvailableTask.CLEAN_UP_BEDROOM,
        AvailableTask.CLEAN_UP_BATHROOM,
        AvailableTask.MULTI_TASK_4,
        AvailableTask.MULTI_TASK_FULL,
    }:
        action_groups["open_close_actions"] = True

    # === Enable toggling ===
    if task in {
        AvailableTask.PREPARE_MEAL,
        AvailableTask.WASH_CUTLERY,
        AvailableTask.RELAX_ON_SOFA,
        AvailableTask.CLEAR_DINING_TABLE,
        AvailableTask.READ_BOOK_IN_BED,
        AvailableTask.DO_HOMEWORK,
        AvailableTask.SETUP_BATH,
        AvailableTask.CLEAN_TOILETS,
        AvailableTask.SWITCH_ON_TV,
        AvailableTask.LOOK_BOOK_IN_LIGHT,
        AvailableTask.SETUP_BATH_SIMPLE,
        AvailableTask.READ_BOOK_IN_BED_SIMPLE,
        AvailableTask.POUR_COFFEE,
        AvailableTask.WATCH_TV,
        AvailableTask.MULTI_TASK_FULL,
    }:
        action_groups["toggle_actions"] = True

    # === Enable slicing ===
    if task in {
        AvailableTask.PREPARE_MEAL,
        AvailableTask.WASH_CUTLERY,
        AvailableTask.RELAX_ON_SOFA,
        AvailableTask.CLEAR_DINING_TABLE,
        AvailableTask.READ_BOOK_IN_BED,
        AvailableTask.DO_HOMEWORK,
        AvailableTask.SETUP_BATH,
        AvailableTask.CLEAN_TOILETS,
        AvailableTask.MULTI_TASK_FULL,
    }:
        action_groups["slice_actions"] = True

    # === Enable throwing ===
    if task == AvailableTask.BREAK_BOWL:
        action_groups["throw_actions"] = True

    return {"action_groups": action_groups}


def keep_only_n_scenes(task_blueprint_config: dict[str, Any], nb_scenes: int) -> dict[str, Any]:
    """Return a copy of the task blueprint config with only the first n scenes."""
    task_blueprint_config = task_blueprint_config.copy()
    task_blueprint_config["scenes"] = task_blueprint_config["scenes"][:nb_scenes]
    return task_blueprint_config


def get_task_blueprint_config(task: AvailableTask, nb_scenes: int) -> list[dict[str, Any]]:
    """Return the scenes for the task."""
    match task:
        case AvailableTask.MULTI_TASK_4:
            return [
                keep_only_n_scenes(task_blueprints_configs[task], nb_scenes)
                for task in (
                    AvailableTask.CLEAN_UP_KITCHEN,
                    AvailableTask.CLEAN_UP_LIVING_ROOM,
                    AvailableTask.CLEAN_UP_BEDROOM,
                    AvailableTask.CLEAN_UP_BATHROOM,
                )
            ]
        case AvailableTask.MULTI_TASK_FULL:
            return [
                keep_only_n_scenes(task_blueprints_configs[task], nb_scenes)
                for task in (
                    AvailableTask.PREPARE_MEAL,
                    AvailableTask.WASH_CUTLERY,
                    AvailableTask.RELAX_ON_SOFA,
                    AvailableTask.CLEAR_DINING_TABLE,
                    AvailableTask.READ_BOOK_IN_BED,
                    AvailableTask.DO_HOMEWORK,
                    AvailableTask.SETUP_BATH,
                    AvailableTask.CLEAN_TOILETS,
                    AvailableTask.CLEAN_UP_KITCHEN,
                    AvailableTask.CLEAN_UP_LIVING_ROOM,
                    AvailableTask.CLEAN_UP_BEDROOM,
                    AvailableTask.CLEAN_UP_BATHROOM,
                )
            ]
        case _:
            return [keep_only_n_scenes(task_blueprints_configs[task], nb_scenes)]


# %%
