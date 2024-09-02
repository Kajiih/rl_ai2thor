"""
Variable item properties module for RL-THOR environment.

TODO: Finish module docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_thor.envs.sim_objects import (
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
    AI2ThorBasedProp,
    EmptyContainerPSF,
    ItemVariableProp,
    SizeLimitPSF,
)

if TYPE_CHECKING:
    from rl_thor.envs.tasks.relations import Relation

# %% === Constants ===
RECEPTACLE_MAX_OBJECTS_PROP_LIMIT = 15  # Up to 15 objects on dining tables; see read_scene_objects_metadata.ipynb


# %% === Property Definitions ===
class VisibleProp(AI2ThorBasedProp, ItemVariableProp[None]):
    """Visible item property."""

    target_ai2thor_property = SimObjVariableProp.VISIBLE


class IsInteractableProp(AI2ThorBasedProp, ItemVariableProp[None]):
    """Is interactable item property."""

    target_ai2thor_property = SimObjVariableProp.IS_INTERACTABLE


class IsPickedUpProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Is picked up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_PICKED_UP
    candidate_required_prop = PickupableProp(True)


class IsPickedUpIfPossibleProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Same as IsPickedUpProp, but doesn't require the item to be pickupable."""

    target_ai2thor_property = SimObjVariableProp.IS_PICKED_UP


class IsToggledProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Is toggled item property."""

    target_ai2thor_property = SimObjVariableProp.IS_TOGGLED
    candidate_required_prop = ToggleableProp(True)


class IndirectIsToggledProp(IsToggledProp):
    """
    Extension of IsToggledProp that works with objects that are indirectly toggled.

    Indirectly toggled objects (e.g. StoveBurner), don't have the Toggleable fixed property.
    """

    candidate_required_prop = None


class IsUsedUpProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Is used up item property."""

    target_ai2thor_property = SimObjVariableProp.IS_USED_UP
    candidate_required_prop = CanBeUsedUpProp(True)


class IsOpenProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Is open item property."""

    target_ai2thor_property = SimObjVariableProp.IS_OPEN
    candidate_required_prop = OpenableProp(True)


# TODO: Test this property
class IsOpenIfPossibleProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """
    Same as IsOpenProp, but doesn't require the item to be openable.

    Used in particular as auxiliary property for ReceptacleOfRelation or IsReceptacleCleared
    because some receptacles have to be opened before placing or removing objects.
    """

    target_ai2thor_property = SimObjVariableProp.IS_OPEN


class OpennessProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Openness item property."""

    target_ai2thor_property = SimObjVariableProp.OPENNESS
    candidate_required_prop = OpenableProp(True)


class IsBrokenProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Is broken item property."""

    target_ai2thor_property = SimObjVariableProp.IS_BROKEN
    candidate_required_prop = BreakableProp(True)
    auxiliary_properties_blueprint = frozenset({(IsPickedUpIfPossibleProp, True)})


class ReceptacleClearedProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Property of a receptacle being empty."""

    target_ai2thor_property = SimObjVariableProp.RECEPTACLE_OBJ_IDS
    candidate_required_prop = ReceptacleProp(True)

    def __init__(
        self,
        expect_clear: bool = True,
        main_prop: ItemVariableProp | None = None,
        main_relation: Relation | None = None,
    ) -> None:
        """
        Initialize the Property object.

        Args:
            expect_clear (bool, optional): Whether the receptacle should be cleared or not.
                Defaults to False.
            main_prop (ItemVariableProp): For auxiliary properties.
            main_relation (Relation): For auxiliary properties.
        """
        auxiliary_properties_blueprint = [
            (IsOpenIfPossibleProp, True),
            (ReceptacleMaxObjectsProp, 1),
        ]
        # auxiliary_properties_blueprint += [
        #     (ReceptacleMaxObjectsProp, i) for i in range(1, RECEPTACLE_MAX_OBJECTS_PROP_LIMIT)
        # ]
        self.auxiliary_properties_blueprint = frozenset(auxiliary_properties_blueprint)

        super().__init__(EmptyContainerPSF(expect_clear), main_prop=main_prop, main_relation=main_relation)


class ReceptacleMaxObjectsProp(AI2ThorBasedProp, ItemVariableProp[bool]):
    """Property of a receptacle containing fewer than a given number of objects."""

    target_ai2thor_property = SimObjVariableProp.RECEPTACLE_OBJ_IDS
    candidate_required_prop = ReceptacleProp(True)

    def __init__(
        self,
        max_objects: int,
        main_prop: ItemVariableProp | None = None,
        main_relation: Relation | None = None,
    ) -> None:
        """
        Initialize the Property object.

        Args:
            max_objects (int): The maximum number of objects the receptacle can hold for the
                property to be satisfied.
            main_prop (ItemVariableProp): For auxiliary properties.
            main_relation (Relation): For auxiliary properties.
        """
        # auxiliary_properties_blueprint: list[tuple[type[ItemVariableProp], Any]] = [(IsOpenIfPossibleProp, True)]
        # TODO: Add this when elimination of duplicate auxiliary props is finished
        auxiliary_properties_blueprint = []
        if max_objects < RECEPTACLE_MAX_OBJECTS_PROP_LIMIT:
            auxiliary_properties_blueprint.append((ReceptacleMaxObjectsProp, max_objects + 1))

        self.auxiliary_properties_blueprint = frozenset(auxiliary_properties_blueprint)

        super().__init__(SizeLimitPSF(max_elements=max_objects), main_prop=main_prop, main_relation=main_relation)
        # ItemVariableProp.__init__(self, SizeLimitPSF(max_elements=max_objects))
