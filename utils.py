"""
Utility functions.
"""
from typing import Any, TypeVar, Callable


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


T = TypeVar("T")


def nb_occurences_in_list(lst: list[V], value: V) -> int:
    """
    Returns the number of occurences of value in a list.
    """
    return sum(1 for x in lst if x == value)


def count_elements_with_property(
    data_list: list[T], property_checker: Callable[[T], bool]
) -> int:
    return sum(1 for x in data_list if property_checker(x))

filter()