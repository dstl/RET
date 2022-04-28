"""Generic Template interface."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from typing import Optional

T = TypeVar("T")
L = TypeVar("L")


class Template(Generic[T]):
    """Generic interface to define objects as templates."""

    @abstractmethod
    def get_new_instance(self) -> T:  # pragma: no cover
        """Return a new instance of a functionally identical object.

        Returns:
            T: New instance of the object
        """
        pass

    @staticmethod
    def generate_object_list(template_list: Optional[list[Template[L]]]) -> list[L]:
        """Generate instances of templates from a list of templates.

        Args:
            template_list (Optional[list[Template[L]]]): List of templates to generate
                instances from.

        Returns:
            list[L]: List of template instances generated from input template list.
        """
        if template_list is None:
            template_list = []
        return [t.get_new_instance() for t in template_list]
