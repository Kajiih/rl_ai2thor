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


def get_objects_dict(objects: list[dict]) -> dict[str, dict]:
    """
    Return a dictionary of objects indexed by object id.

    Args:
        objects (list[dict]): List of objects to index.

    Returns:
        dict[str, dict]: Dictionary of objects indexed by object id.
    """
    return {obj["objectId"]: obj for obj in objects}


def get_object_data_from_id(objects: list[dict], object_id: str) -> dict:
    """
    Return the object with the specified id.

    Args:
        objects (list[dict]): List of objects to search.
        object_id (str): Object id to search for.

    Returns:
        dict: Object with the specified id.
    """
    for obj in objects:
        if obj["objectId"] == object_id:
            return obj
    raise ValueError(f"Object with id {object_id} not found.")


def get_objects_with_properties(
    objects: list[dict], properties: dict[str, Any]
) -> list[dict]:
    """
    Return a list of objects that have the specified properties.

    Args:
        objects (list[dict]): List of objects to search.
        properties (dict[str, Any]): Dictionary of properties to search for.

    Returns:
        list[dict]: List of objects that have the specified properties.
    """
    return [obj for obj in objects if all(obj[k] == v for k, v in properties.items())]
