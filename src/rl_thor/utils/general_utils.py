"""Utility functions and variables for handling configuration files."""

import pathlib
from collections.abc import Hashable

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

type NestedDict[T: Hashable, V] = dict[T, NestedDict[T, V] | V]


def update_nested_dict[T, V](original_dict: NestedDict[T, V], new_dict: NestedDict[T, V]) -> None:
    """Recursively update a nested dictionary in place."""
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in original_dict and isinstance(original_dict[key], dict):
            update_nested_dict(original_dict[key], value)
        else:
            original_dict[key] = value


def nested_dict_get[T, V](d: NestedDict[T, V], keys: list[T], default: V = None) -> V:
    """
    Return the value for keys in a nested dictionary; if not found return a default value.

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
