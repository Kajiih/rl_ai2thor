"""
Variable item properties module for RL-THOR environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from typing import Any

from rl_thor.envs.sim_objects import (
    SimObjMetadata,
    SimObjVariableProp,
)
from rl_thor.envs.tasks._item_prop_fixed import (
    BreakableProp,
    CanBeUsedUpProp,
    OpenableProp,
    PickupableProp,
    ReceptacleProp,
    ToggleableProp,
)
from rl_thor.envs.tasks.item_prop_interface import (
    EmptyContainerPSF,
    ItemVariableProp,
    PropAuxProp,
    SizeLimitPSF,
)

# %% === Constants ===
RECEPTACLE_MAX_OBJECTS_PROP_LIMIT = 15  # Up to 15 objects on dining tables; see read_scene_objects_metadata.ipynb


# %% === Property Definitions ===
class VisibleProp(ItemVariableProp[bool, Any]):
    """Visible item property."""

    target_ai2thor_property = SimObjVariableProp.VISIBLE


class IsInteractableProp(ItemVariableProp[bool, Any]):
    """Is interactable item property."""

    target_ai2thor_property = SimObjVariableProp.IS_INTERACTABLE


class IsPickedUpProp(ItemVariableProp[bool, bool]):
    """Is picked up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_PICKED_UP
    candidate_required_prop = PickupableProp(True)


class IsPickedUpIfPossibleProp(ItemVariableProp[bool, bool]):
    """Same as IsPickedUpProp, but doesn't require the item to be pickupable."""

    target_ai2thor_property = SimObjVariableProp.IS_PICKED_UP


class IsToggledProp(ItemVariableProp[bool, bool]):
    """Is toggled item property."""

    target_ai2thor_property = SimObjVariableProp.IS_TOGGLED
    candidate_required_prop = ToggleableProp(True)


class IndirectIsToggledProp(IsToggledProp):
    """
    Extension of IsToggledProp that works with objects that are indirectly toggled.

    Indirectly toggled objects (e.g. StoveBurner), don't have the Toggleable fixed property.
    """

    candidate_required_prop = None


class IsUsedUpProp(ItemVariableProp[bool, bool]):
    """Is used up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_USED_UP
    candidate_required_prop = CanBeUsedUpProp(True)


class IsOpenProp(ItemVariableProp[bool, bool]):
    """Is open item property."""

    target_ai2thor_property = SimObjVariableProp.IS_OPEN
    candidate_required_prop = OpenableProp(True)


# TODO: Test this property
class IsOpenIfPossibleProp(ItemVariableProp[bool, bool]):
    """
    Same as IsOpenProp, but doesn't require the item to be openable.

    Used in particular as auxiliary property for ReceptacleOfRelation or IsReceptacleCleared
    because some receptacles have to be opened before placing or removing objects.
    """

    target_ai2thor_property = SimObjVariableProp.IS_OPEN


class OpennessProp(ItemVariableProp[float, bool]):
    """Openness item property."""

    target_ai2thor_property = SimObjVariableProp.OPENNESS
    candidate_required_prop = OpenableProp(True)


class IsBrokenProp(ItemVariableProp[bool, bool]):
    """Is broken item property."""

    target_ai2thor_property = SimObjVariableProp.IS_BROKEN
    candidate_required_prop = BreakableProp(True)
    auxiliary_properties = frozenset({PropAuxProp(IsPickedUpIfPossibleProp, True)})


class ReceptacleClearedProp(ItemVariableProp[bool, bool]):
    """Property of a receptacle being empty."""

    target_ai2thor_property = SimObjVariableProp.RECEPTACLE_OBJ_IDS
    candidate_required_prop = ReceptacleProp(True)

    def __init__(
        self,
        expect_clear: bool = True,
    ) -> None:
        """
        Initialize the Property object.

        Args:
            expect_clear (bool, optional): Whether the receptacle should be cleared or not.
                Defaults to False.
        """
        auxiliary_properties: list[PropAuxProp] = [PropAuxProp(IsOpenIfPossibleProp, True)]
        auxiliary_properties += [
            PropAuxProp(ReceptacleMaxObjectsProp, i) for i in range(1, RECEPTACLE_MAX_OBJECTS_PROP_LIMIT)
        ]
        # TODO: Let only PropAuxProp(ReceptacleMaxObjectsProp, RECEPTACLE_MAX_OBJECTS_PROP_LIMIT) once the auxiliary properties of auxiliary properties
        self.auxiliary_properties = frozenset(auxiliary_properties)

        super().__init__(EmptyContainerPSF(expect_clear))


class ReceptacleMaxObjectsProp(ItemVariableProp[int, bool]):
    """Property of a receptacle containing fewer than a given number of objects."""

    target_ai2thor_property = SimObjVariableProp.RECEPTACLE_OBJ_IDS
    candidate_required_prop = ReceptacleProp(True)

    def __init__(self, max_objects: int) -> None:
        """
        Initialize the Property object.

        Args:
            max_objects (int): The maximum number of objects the receptacle can hold for the
                property to be satisfied.
        """
        # auxiliary_properties: list[PropAuxProp] = [PropAuxProp(IsOpenIfPossibleProp, True)]
        # if max_objects < RECEPTACLE_MAX_OBJECTS_PROP_LIMIT:
        #     auxiliary_properties.append(PropAuxProp(ReceptacleMaxObjectsProp, max_objects + 1))

        # self.auxiliary_properties = frozenset(auxiliary_properties)
        # TODO: Uncomment this once the auxiliary properties of auxiliary properties are implemented

        # super().__init__(SizeLimitPSF(max_elements=max_objects))
        ItemVariableProp.__init__(self, SizeLimitPSF(max_elements=max_objects))
