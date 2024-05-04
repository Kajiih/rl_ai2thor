"""
Module grouping all the item properties used in the tasks of RL-THOR environment.

TODO: Finish module docstring.
"""

from rl_thor.envs.sim_objects import SimObjFixedProp, SimObjProp, SimObjVariableProp
from rl_thor.envs.tasks._item_prop_complex import (
    IsCookedProp,
    # FillLiquidProp,  # TODO: Finish implementing FillLiquidProp
    IsDirtyProp,
    IsFilledWithLiquidProp,
    IsSlicedProp,
    TemperatureProp,
)
from rl_thor.envs.tasks._item_prop_fixed import (
    BreakableProp,
    CanBeUsedUpProp,
    CanFillWithLiquidProp,
    CookableProp,
    DirtyableProp,
    IsColdSourceProp,
    IsHeatSourceProp,
    MoveableProp,
    ObjectTypeProp,
    OpenableProp,
    PickupableProp,
    ReceptacleProp,
    SliceableProp,
    ToggleableProp,
)
from rl_thor.envs.tasks._item_prop_variable import (
    IsBrokenProp,
    IsInteractableProp,
    IsOpenProp,
    IsPickedUpProp,
    IsToggledProp,
    IsUsedUpProp,
    OpennessProp,
    VisibleProp,
)
from rl_thor.envs.tasks.item_prop_interface import ItemProp

# %% === Item property mapping ===
obj_prop_id_to_item_prop: dict[SimObjProp | str, type[ItemProp]]
obj_prop_id_to_item_prop = {
    # === Fixed Item Properties ===
    SimObjFixedProp.OBJECT_TYPE: ObjectTypeProp,
    SimObjFixedProp.RECEPTACLE: ReceptacleProp,
    SimObjFixedProp.TOGGLEABLE: ToggleableProp,
    SimObjFixedProp.BREAKABLE: BreakableProp,
    SimObjFixedProp.CAN_FILL_WITH_LIQUID: CanFillWithLiquidProp,
    SimObjFixedProp.DIRTYABLE: DirtyableProp,
    SimObjFixedProp.CAN_BE_USED_UP: CanBeUsedUpProp,
    SimObjFixedProp.COOKABLE: CookableProp,
    SimObjFixedProp.IS_HEAT_SOURCE: IsHeatSourceProp,
    SimObjFixedProp.IS_COLD_SOURCE: IsColdSourceProp,
    SimObjFixedProp.SLICEABLE: SliceableProp,
    SimObjFixedProp.OPENABLE: OpenableProp,
    SimObjFixedProp.PICKUPABLE: PickupableProp,
    SimObjFixedProp.MOVEABLE: MoveableProp,
    # === Variable Item Properties ===
    SimObjVariableProp.VISIBLE: VisibleProp,
    SimObjVariableProp.IS_INTERACTABLE: IsInteractableProp,
    SimObjVariableProp.IS_TOGGLED: IsToggledProp,
    SimObjVariableProp.IS_BROKEN: IsBrokenProp,
    SimObjVariableProp.IS_USED_UP: IsUsedUpProp,
    SimObjVariableProp.IS_OPEN: IsOpenProp,
    SimObjVariableProp.OPENNESS: OpennessProp,
    SimObjVariableProp.IS_PICKED_UP: IsPickedUpProp,
    # === Complex Variable Item Properties (with auxiliary items) ===
    SimObjVariableProp.IS_FILLED_WITH_LIQUID: IsFilledWithLiquidProp,
    # SimObjVariableProp.FILL_LIQUID: FillLiquidProp,
    SimObjVariableProp.IS_DIRTY: IsDirtyProp,
    SimObjVariableProp.IS_COOKED: IsCookedProp,
    SimObjVariableProp.TEMPERATURE: TemperatureProp,
    SimObjVariableProp.IS_SLICED: IsSlicedProp,
}
