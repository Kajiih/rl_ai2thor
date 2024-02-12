"""Module for custom exceptions for the AI2THOR RL environment."""


class UnknownActionCategoryError(Exception):
    """Exception raised for unknown action categories in environment mode config."""

    def __init__(self, action_category: str) -> None:
        """
        Initialize the UnknownActionCategoryError.

        Args:
            action_category (str): The unknown action category.
        """
        self.action_category = action_category
        super().__init__(f"Unknown action category {action_category} in environment mode config.")
