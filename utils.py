"""
Utility functions.
"""
from typing import Any


def update_nested_dict(original_dict: dict, new_dict: dict) -> None:
    """Recursively update a nested dictionary in place."""
    for key, value in new_dict.items():
        if (
            isinstance(value, dict)
            and key in original_dict
            and isinstance(original_dict[key], dict)
        ):
            update_nested_dict(original_dict[key], value)
        else:
            original_dict[key] = value


def nested_dict_get(d: dict, keys: list, default: Any = None) -> Any:
    """
    Returns the value for keys in a nested dictionary; if not found returns a default value.

    Args:
        d (dict): Nested dictionary to search.
        keys (list): List of keys to search for.
        default (any): Default value to return if keys not found.

    Returns:
        any: Value at keys in d, or default if not found.
    """
    for key in keys:
        try:
            d = d[key]
        except KeyError:
            return default
    return d
