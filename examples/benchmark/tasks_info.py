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
    MULTI_TASK_8 = "MultiTask8"
    PREPARE_MEAL = TaskType.PREPARE_MEAL
    RELAX_ON_SOFA = TaskType.RELAX_ON_SOFA
    READ_BOOK_IN_BED = TaskType.READ_BOOK_IN_BED
    SETUP_BATH = TaskType.SETUP_BATH

    # Gradual tasks
    # 1 item
    BREAK_MUG = "BreakMug"
    PICKUP_KNIFE = "PickupKnife"
    TOGGLE_FAUCET = "ToggleFaucet"
    OPEN_DRAWER = "OpenDrawer"
    SWITCH_ON_TV = "SwitchOnTV"
    PICKUP_MUG = "PickupMug"
    PICKUP_POTATO = "PickupPotato"
    # 2 items
    COOL_TOMATO = "CoolTomato"
    PLACE_POTATO_IN_FRIDGE = "PlacePotatoInFridge"
    PLACE_NEWSPAPER_ON_SOFA = "PlaceNewspaperOnSofa"
    BRING_TOWEL_CLOTH_CLOSE = "BringTowelClothesClose"
    COOK_POTATO = "CookPotato"
    LOOK_BOOK_IN_LIGHT = "LookBookInLight"
    PLACE_KNIFE_IN_SINK = "PlaceKnifeInSink"
    PLACE_MUG_IN_SINK = "PlaceMugInSink"
    PLACE_KNIFE_IN_FILLED_SINK = "PlaceKnifeInFilledSink"
    PLACE_MUG_IN_FILLED_SINK = "PlaceMugInFilledSink"
    # 3 items
    PLACE_TOMATO_POTATO_IN_FRIDGE = "PlaceTomatoPotatoInFridge"
    PLACE_KNIFE_BOWL_MUG_IN_FILLED_SINK = "PlaceKnifeBowlMugInFilledSink"
    SLICE_AND_COOK_POTATO = "SliceAndCookPotato"


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
            "FloorPlan213",
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
    AvailableTask.BREAK_MUG: {
        "task_type": TaskType.BREAK,
        "args": {"broken_object_type": SimObjectType.MUG},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PICKUP_KNIFE: {
        "task_type": TaskType.PICKUP,
        "args": {"picked_up_object_type": SimObjectType.BUTTER_KNIFE},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.SWITCH_ON_TV: {
        "task_type": TaskType.TOGGLE,
        "args": {"switched_on_object_type": SimObjectType.TELEVISION},
        "scenes": ["FloorPlan201"],
    },
    AvailableTask.OPEN_DRAWER: {
        "task_type": TaskType.OPEN,
        "args": {"opened_object_type": SimObjectType.DRAWER},
        "scenes": ["FloorPlan301"],
    },
    AvailableTask.TOGGLE_FAUCET: {
        "task_type": TaskType.TOGGLE,
        "args": {"toggled_object_type": SimObjectType.FAUCET},
        "scenes": ["FloorPlan401"],
    },
    AvailableTask.PICKUP_MUG: {
        "task_type": TaskType.PICKUP,
        "args": {"picked_up_object_type": SimObjectType.MUG},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PICKUP_POTATO: {
        "task_type": TaskType.PICKUP,
        "args": {"picked_up_object_type": SimObjectType.POTATO},
        "scenes": ["FloorPlan1"],
    },
    # 2 items tasks
    AvailableTask.COOK_POTATO: {
        "task_type": TaskType.COOK,
        "args": {"cooked_object_type": SimObjectType.POTATO},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_POTATO_IN_FRIDGE: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.POTATO, "receptacle_type": SimObjectType.FRIDGE},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.COOL_TOMATO: {
        "task_type": TaskType.COOL_DOWN,
        "args": {"cooled_object_type": SimObjectType.TOMATO},
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
    AvailableTask.PLACE_KNIFE_IN_SINK: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.BUTTER_KNIFE, "receptacle_type": SimObjectType.SINK_BASIN},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_MUG_IN_SINK: {
        "task_type": TaskType.PLACE_IN,
        "args": {"placed_object_type": SimObjectType.MUG, "receptacle_type": SimObjectType.SINK_BASIN},
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_MUG_IN_FILLED_SINK: {
        "task_type": TaskType.PLACE_IN_FILLED_SINK,
        "args": {"placed_object_type": SimObjectType.MUG},
        "scenes": ["FloorPlan1"],
    },
    # 3 items tasks
    AvailableTask.PLACE_TOMATO_POTATO_IN_FRIDGE: {
        "task_type": TaskType.PLACE_TWO_IN,
        "args": {
            "object_type_1": SimObjectType.TOMATO,
            "object_type_2": SimObjectType.POTATO,
            "receptacle_type": SimObjectType.FRIDGE,
        },
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.PLACE_KNIFE_BOWL_MUG_IN_FILLED_SINK: {
        "task_type": TaskType.PLACE_IN_FILLED_SINK,
        "args": {
            "placed_object_type_1": SimObjectType.BUTTER_KNIFE,
            "placed_object_type_2": SimObjectType.BOWL,
            "placed_object_type_3": SimObjectType.MUG,
        },
        "scenes": ["FloorPlan1"],
    },
    AvailableTask.SLICE_AND_COOK_POTATO: {
        "task_type": TaskType.SLICE_AND_COOK_POTATO,
        "args": {},
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
        AvailableTask.COOK_POTATO,
        AvailableTask.SLICE_AND_COOK_POTATO,
        AvailableTask.PREPARE_MEAL,
        AvailableTask.RELAX_ON_SOFA,
        AvailableTask.READ_BOOK_IN_BED,
        AvailableTask.SETUP_BATH,
        AvailableTask.OPEN_DRAWER,
        AvailableTask.COOL_TOMATO,
        AvailableTask.PLACE_TOMATO_POTATO_IN_FRIDGE,
        AvailableTask.CLEAN_UP_KITCHEN,
        AvailableTask.CLEAN_UP_LIVING_ROOM,
        AvailableTask.CLEAN_UP_BEDROOM,
        AvailableTask.CLEAN_UP_BATHROOM,
    }:
        action_groups["open_close_actions"] = True

    # === Enable toggling ===
    if task in {
        AvailableTask.PLACE_KNIFE_IN_FILLED_SINK,
        AvailableTask.PLACE_MUG_IN_FILLED_SINK,
        AvailableTask.PLACE_KNIFE_BOWL_MUG_IN_FILLED_SINK,
        AvailableTask.COOK_POTATO,
        AvailableTask.SLICE_AND_COOK_POTATO,
        AvailableTask.PREPARE_MEAL,
        AvailableTask.RELAX_ON_SOFA,
        AvailableTask.READ_BOOK_IN_BED,
        AvailableTask.SETUP_BATH,
        AvailableTask.TOGGLE_FAUCET,
        AvailableTask.SWITCH_ON_TV,
        AvailableTask.LOOK_BOOK_IN_LIGHT,
    }:
        action_groups["toggle_actions"] = True

    # === Enable slicing ===
    if task in {
        AvailableTask.SLICE_AND_COOK_POTATO,
        AvailableTask.PREPARE_MEAL,
        AvailableTask.RELAX_ON_SOFA,
        AvailableTask.READ_BOOK_IN_BED,
        AvailableTask.SETUP_BATH,
    }:
        action_groups["slice_actions"] = True

    # === Enable dropping ===
    if task == AvailableTask.BREAK_MUG:
        action_groups["drop_actions"] = True

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
        case AvailableTask.MULTI_TASK_8:
            return [
                keep_only_n_scenes(task_blueprints_configs[task], nb_scenes)
                for task in (
                    AvailableTask.PREPARE_MEAL,
                    AvailableTask.RELAX_ON_SOFA,
                    AvailableTask.READ_BOOK_IN_BED,
                    AvailableTask.SETUP_BATH,
                    AvailableTask.CLEAN_UP_KITCHEN,
                    AvailableTask.CLEAN_UP_LIVING_ROOM,
                    AvailableTask.CLEAN_UP_BEDROOM,
                    AvailableTask.CLEAN_UP_BATHROOM,
                )
            ]
        case _:
            return [keep_only_n_scenes(task_blueprints_configs[task], nb_scenes)]
