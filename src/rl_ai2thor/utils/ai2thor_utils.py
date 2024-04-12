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
        scene_objects_dict (dict[SimObjId, SimObjMetadata]): Dictionary of objects indexed by object id.
    """
    return {obj["objectId"]: obj for obj in objects}


def compute_objects_3d_distance(obj1: SimObjMetadata, obj2: SimObjMetadata) -> float:
    """
    Compute the 3D distance between two objects.

    Args:
        obj1 (dict): Metadata of the first object.
        obj2 (dict): Metadata of the second object.

    Returns:
        float: 3D distance between the two objects.
    """
    return (
        (obj1["position"]["x"] - obj2["position"]["x"]) ** 2
        + (obj1["position"]["y"] - obj2["position"]["y"]) ** 2
        + (obj1["position"]["z"] - obj2["position"]["z"]) ** 2
    ) ** 0.5


def compute_objects_2d_distance(obj1: SimObjMetadata, obj2: SimObjMetadata) -> float:
    """
    Compute the 2D distance between two objects.

    Args:
        obj1 (dict): Metadata of the first object.
        obj2 (dict): Metadata of the second object.

    Returns:
        float: 2D distance between the two objects.
    """
    return (
        (obj1["position"]["x"] - obj2["position"]["x"]) ** 2 + (obj1["position"]["z"] - obj2["position"]["z"]) ** 2
    ) ** 0.5
