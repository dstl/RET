"""Ret manager for registered lists that may be extended."""
import warnings
from typing import Callable, Generic, Optional, TypeVar  # noqa: TC002

T = TypeVar("T")


class RetRegisterManager(Generic[T]):
    """A register to serve lists of options in ret that may be extended by other modules."""

    def __init__(self):
        """Create a the register with an empty dictionary of generators."""
        self._methods: dict[str, Callable[[], T]] = dict()

    def register(self, name: str, generator: Callable[[], T]):
        """Register an item with it's generator.

        Args:
            name (str): The name of the register item.
            generator (Callable[[], T]): The generator method of the register item.
        """
        if name == "None":
            warnings.warn("Cannot register an item to 'None'")
            return

        if name in self._methods:
            warnings.warn(f"Overwriting the registered member: '{name}'")
        self._methods[name] = generator

    def get_register_items(self) -> list[str]:
        """Retrieve a list of names of registered items.

        Returns:
            list[str]: A list of the names of registered items.
        """
        return list(self._methods.keys())

    def get_register_item(self, name: str) -> Optional[T]:
        """Retrieve an item from the register using its name.

        If `name` is "None", no registered item will be returned, regardless of whether the register
        has an item registed to the "None" name.

        Args:
            name (str): The name of the item to be retrieved.

        Returns:
            Optional[T]: The item from the register matching the name. Returns None if
                the name is not registered.
        """
        if name == "None":
            return None

        if name in self._methods:
            try:
                return self._methods[name]()
            except TypeError:
                warnings.warn(f"Error in registered member: '{name}'. Returning None.")
                return None
        warnings.warn(f"'{name}' is unregistered. Returning None.")
        return None
