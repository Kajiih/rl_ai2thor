"""
Utility functions specific to AI2-THOR.
"""

from typing import Any


def get_object_type_from_id(object_id: str) -> str:
    """
    Return the object type from an object id.

    Args:
        object_id (str): Object id to parse.

    Returns:
        str: Object type.
    """
    return object_id.split("|")[0]


def get_objects_dict(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Return a dictionary of objects indexed by object id.

    Args:
        objects (list[dict]): List of objects to index.

    Returns:
        dict[str, dict]: Dictionary of objects indexed by object id.
    """
    return {obj["objectId"]: obj for obj in objects}
