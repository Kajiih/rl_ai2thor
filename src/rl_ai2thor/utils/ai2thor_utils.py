"""Utility functions specific to AI2-THOR."""

from rl_ai2thor.envs.sim_objects import SimObjectType, SimObjId, SimObjMetadata


def get_object_type_from_id(object_id: str) -> SimObjectType:
    """
    Return the object type from an object id.

    Args:
        object_id (str): Object id to parse.

    Returns:
        object_type (str): Object type.
    """
    return SimObjectType(object_id.split("|")[0])


def get_scene_objects_dict(objects: list[SimObjMetadata]) -> dict[SimObjId, SimObjMetadata]:
    """
    Return a dictionary of objects indexed by object id.

    Args:
        objects (list[dict]): List of objects to index.

    Returns:
        scene_object_dict (dict[SimObjId, SimObjMetadata]): Dictionary of objects indexed by object id.
    """
    return {obj["objectId"]: obj for obj in objects}
