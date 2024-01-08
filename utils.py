"""
Utility functions.
"""


def update_nested_dict(original_dict, new_dict):
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
