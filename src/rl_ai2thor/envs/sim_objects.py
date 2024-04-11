"""
Sim objects and sim objects properties in AI2-THOR RL environment.

TODO: Finish module docstring.
"""

from enum import StrEnum
from typing import Any, NewType

from rl_ai2thor.data import OBJECT_TYPES_DATA


# %% === Sim Object Types ===
class SimObjectType(StrEnum):
    """All object types in AI2-THOR."""

    ALARM_CLOCK = "AlarmClock"
    ALUMINUM_FOIL = "AluminumFoil"
    APPLE = "Apple"
    APPLE_SLICED = "AppleSliced"
    ARM_CHAIR = "ArmChair"
    BASEBALL_BAT = "BaseballBat"
    BASKET_BALL = "BasketBall"
    BATHTUB = "Bathtub"
    BATHTUB_BASIN = "BathtubBasin"
    BED = "Bed"
    BLINDS = "Blinds"
    BOOK = "Book"
    BOOTS = "Boots"
    BOTTLE = "Bottle"
    BOWL = "Bowl"
    BOX = "Box"
    BREAD = "Bread"
    BREAD_SLICED = "BreadSliced"
    BUTTER_KNIFE = "ButterKnife"
    CABINET = "Cabinet"
    CANDLE = "Candle"
    CD = "CD"
    CELL_PHONE = "CellPhone"
    CHAIR = "Chair"
    CLOTH = "Cloth"
    COFFEE_MACHINE = "CoffeeMachine"
    COFFEE_TABLE = "CoffeeTable"
    COUNTER_TOP = "CounterTop"
    CREDIT_CARD = "CreditCard"
    CUP = "Cup"
    CURTAINS = "Curtains"
    DESK = "Desk"
    DESK_LAMP = "DeskLamp"
    DESKTOP = "Desktop"
    DINING_TABLE = "DiningTable"
    DISH_SPONGE = "DishSponge"
    DOG_BED = "DogBed"
    DRAWER = "Drawer"
    DRESSER = "Dresser"
    DUMBBELL = "Dumbbell"
    EGG = "Egg"
    EGG_CRACKED = "EggCracked"
    FAUCET = "Faucet"
    FLOOR = "Floor"
    FLOOR_LAMP = "FloorLamp"
    FOOTSTOOL = "Footstool"
    FORK = "Fork"
    FRIDGE = "Fridge"
    GARBAGE_BAG = "GarbageBag"
    GARBAGE_CAN = "GarbageCan"
    HAND_TOWEL = "HandTowel"
    HAND_TOWEL_HOLDER = "HandTowelHolder"
    HOUSE_PLANT = "HousePlant"
    KETTLE = "Kettle"
    KEY_CHAIN = "KeyChain"
    KNIFE = "Knife"
    LADLE = "Ladle"
    LAPTOP = "Laptop"
    LAUNDRY_HAMPER = "LaundryHamper"
    LETTUCE = "Lettuce"
    LETTUCE_SLICED = "LettuceSliced"
    LIGHT_SWITCH = "LightSwitch"
    MICROWAVE = "Microwave"
    MIRROR = "Mirror"
    MUG = "Mug"
    NEWSPAPER = "Newspaper"
    OTTOMAN = "Ottoman"
    PAINTING = "Painting"
    PAN = "Pan"
    PAPER_TOWEL_ROLL = "PaperTowelRoll"
    PEN = "Pen"
    PENCIL = "Pencil"
    PEPPER_SHAKER = "PepperShaker"
    PILLOW = "Pillow"
    PLATE = "Plate"
    PLUNGER = "Plunger"
    POSTER = "Poster"
    POT = "Pot"
    POTATO = "Potato"
    POTATO_SLICED = "PotatoSliced"
    REMOTE_CONTROL = "RemoteControl"
    ROOM_DECOR = "RoomDecor"
    SAFE = "Safe"
    SALT_SHAKER = "SaltShaker"
    SCRUB_BRUSH = "ScrubBrush"
    SHELF = "Shelf"
    SHELVING_UNIT = "ShelvingUnit"
    SHOWER_CURTAIN = "ShowerCurtain"
    SHOWER_DOOR = "ShowerDoor"
    SHOWER_GLASS = "ShowerGlass"
    SHOWER_HEAD = "ShowerHead"
    SIDE_TABLE = "SideTable"
    SINK = "Sink"
    SINK_BASIN = "SinkBasin"
    SOAP_BAR = "SoapBar"
    SOAP_BOTTLE = "SoapBottle"
    SOFA = "Sofa"
    SPATULA = "Spatula"
    SPOON = "Spoon"
    SPRAY_BOTTLE = "SprayBottle"
    STATUE = "Statue"
    STOOL = "Stool"
    STOVE_BURNER = "StoveBurner"
    STOVE_KNOB = "StoveKnob"
    TABLE_TOP_DECOR = "TableTopDecor"
    # TARGET_CIRCLE = "TargetCircle"
    TEDDY_BEAR = "TeddyBear"
    TELEVISION = "Television"
    TENNIS_RACKET = "TennisRacket"
    TISSUE_BOX = "TissueBox"
    TOASTER = "Toaster"
    TOILET = "Toilet"
    TOILET_PAPER = "ToiletPaper"
    TOILET_PAPER_HANGER = "ToiletPaperHanger"
    TOMATO = "Tomato"
    TOMATO_SLICED = "TomatoSliced"
    TOWEL = "Towel"
    TOWEL_HOLDER = "TowelHolder"
    TV_STAND = "TVStand"
    VACUUM_CLEANER = "VacuumCleaner"
    VASE = "Vase"
    WATCH = "Watch"
    WATERING_CAN = "WateringCan"
    WINDOW = "Window"
    WINE_BOTTLE = "WineBottle"


# === Sim Object Properties ===
# TODO: Add support for more mass and salient materials.
class SimObjFixedProp(StrEnum):
    """Fixed properties of sim objects in AI2-THOR."""

    OBJECT_ID = "objectId"
    OBJECT_TYPE = "objectType"
    IS_INTERACTABLE = "isInteractable"
    RECEPTACLE = "receptacle"
    TOGGLEABLE = "toggleable"
    BREAKABLE = "breakable"
    CAN_FILL_WITH_LIQUID = "canFillWithLiquid"
    DIRTYABLE = "dirtyable"
    CAN_BE_USED_UP = "canBeUsedUp"
    COOKABLE = "cookable"
    IS_HEAT_SOURCE = "isHeatSource"
    IS_COLD_SOURCE = "isColdSource"
    SLICEABLE = "sliceable"
    OPENABLE = "openable"
    PICKUPABLE = "pickupable"
    MOVEABLE = "moveable"
    MASS = "mass"
    SALIENT_MATERIALS = "salientMaterials"


# TODO: Add support for position, rotation and distance.
class SimObjVariableProp(StrEnum):
    """Variable properties of sim objects in AI2-THOR."""

    VISIBLE = "visible"
    IS_TOGGLED = "isToggled"
    IS_BROKEN = "isBroken"
    IS_FILLED_WITH_LIQUID = "isFilledWithLiquid"
    FILL_LIQUID = "fillLiquid"
    IS_DIRTY = "isDirty"
    IS_USED_UP = "isUsedUp"
    IS_COOKED = "isCooked"
    TEMPERATURE = "temperature"
    IS_SLICED = "isSliced"
    IS_OPEN = "isOpen"
    OPENNESS = "openness"
    IS_PICKED_UP = "isPickedUp"
    POSITION = "position"
    # ROTATION = "rotation"
    # DISTANCE = "distance"


# TODO: Change this to a union of enums instead of type alias.
type SimObjProp = SimObjFixedProp | SimObjVariableProp
type SimObjMetadataKey = SimObjProp | str
SimObjId = NewType("SimObjId", str)
# type SimObjId = str
type SimObjPropValue = Any
type SimObjMetadata = dict[SimObjMetadataKey, SimObjPropValue]


# %% === Sim Object Groups ===
PICKUPABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.PICKUPABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
RECEPTACLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.RECEPTACLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
MOVEABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.MOVEABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
SLICEABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.SLICEABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
OPENABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.OPENABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
TOGGLEABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.TOGGLEABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
BREAKABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.BREAKABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
FILLABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.CAN_FILL_WITH_LIQUID in OBJECT_TYPES_DATA[object_type].actionable_properties
}
DIRTYABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.DIRTYABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}
COOKABLES = {
    object_type
    for object_type in SimObjectType
    if SimObjFixedProp.COOKABLE in OBJECT_TYPES_DATA[object_type].actionable_properties
}

PICKUPABLE_RECEPTACLES = PICKUPABLES & RECEPTACLES

LIGHT_SOURCES = {
    SimObjectType.CANDLE,
    SimObjectType.DESK_LAMP,
    SimObjectType.FLOOR_LAMP,
    # SimObjectType.LIGHT_SWITCH,
}

WATER_SOURCES = {
    SimObjectType.FAUCET,
    SimObjectType.SHOWER_HEAD,
}

COLD_SOURCES = {
    SimObjectType.FRIDGE,
}

HEAT_SOURCES = {
    SimObjectType.COFFEE_MACHINE,  # Only Mug
    SimObjectType.MICROWAVE,
    SimObjectType.STOVE_BURNER,  # Turned on by StoveKnob
    # SimObjectType.TOASTER,  # Only BreadSlice // Not sure it actually heats the bread and not only cooks it
}

COOKING_SOURCES = {
    SimObjectType.COFFEE_MACHINE,  # Only Mug
    SimObjectType.MICROWAVE,
    SimObjectType.STOVE_BURNER,  # Turned on by StoveKnob
    SimObjectType.TOASTER,  # Only BreadSlice
}


ALL_OBJECT_GROUPS = {
    "_PICKUPABLES": PICKUPABLES,
    "_RECEPTACLES": RECEPTACLES,
    "_MOVEABLES": MOVEABLES,
    "_SLICEABLES": SLICEABLES,
    "_OPENABLES": OPENABLES,
    "_TOGGLEABLES": TOGGLEABLES,
    "_BREAKABLES": BREAKABLES,
    "_FILLABLES": FILLABLES,
    "_DIRTYABLES": DIRTYABLES,
    "_COOKABLES": COOKABLES,
    "_PICKUPABLE_RECEPTACLES": PICKUPABLE_RECEPTACLES,
    "_LIGHT_SOURCES": LIGHT_SOURCES,
    "_WATER_SOURCES": WATER_SOURCES,
    "_COLD_SOURCES": COLD_SOURCES,
    "_HEAT_SOURCES": HEAT_SOURCES,
    "_COOKING_SOURCES": COOKING_SOURCES,
}
