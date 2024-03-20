"""Ret Model Parser."""
from __future__ import annotations

import importlib
from json import load
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, Union

import pydantic

from . import v1, v2

if TYPE_CHECKING:
    from typing import Any, Optional, TextIO

T = TypeVar("T")
Schema = Union[v1.RetModelSchema, v2.RetModelSchema]


class Parser(Generic[T]):
    """RET model definition file parser."""

    def __init__(self, model_type: type, depth: int = 10):
        """Create a new Parser.

        Args:
            model_type (type): The model type to be upgraded to
            depth (int): The maximum number of allowed upgrades before a recursion
                exception is thrown. Defaults to 10.
        """
        self.depth = depth
        self.model_type = model_type

    def parse(
        self,
        path: Optional[str] = None,
        file: Optional[TextIO] = None,
    ) -> T:
        """Parse a RET model definition file.

        Args:
            path (Optional[str]): Path to (JSON) file to parse. Defaults to None.
            file (Optional[TextIO]): File handle to JSON file. Defaults to None.

        Raises:
            ParserConfigurationError: Neither `file` or `path` are defined, or both
                `path` and `file` are defined.

        Returns:
            T: Model converted into the format T
        """
        if path is not None and file is not None:
            raise ParserConfigurationError(path=path, file=file)

        if file is not None:
            obj = self._parse_file(file)
            return self.upgrade(model=obj)  # type: ignore

        if path is not None:
            with Path(path).open() as f:
                return self.parse(file=f)

        raise ParserConfigurationError(path=path, file=file)

    def upgrade(self, model: Schema) -> T:
        """Attempt to upgrade model to the latest type.

        Args:
            model (Schema): Model to be upgraded

        Raises:
            IOUpgradeError: A model that is not of type target does not have have
                an upgrade method
            RecursionError: Model has attempted to upgrade more than the allowable number of times

        Returns:
            T: Representation of model in the defined target type
        """
        depth = 0
        try:
            while not (isinstance(model, self.model_type)):
                depth += 1
                if self.depth < depth:
                    raise RecursionError()

                model = model.upgrade()  # type: ignore
            return model  # type: ignore
        except AttributeError:
            raise IOUpgradeError(model, self.model_type)

    def _parse_file(self, file: TextIO) -> Optional[Schema]:
        j = load(file)
        if "version" not in j:
            return None

        if "imports" in j:
            for imp in j["imports"]:
                importlib.import_module(imp)

        if j["version"] == 1:
            return pydantic.parse_obj_as(v1.RetModelSchema, j["data"])  # type: ignore

        if j["version"] == 2:
            return pydantic.parse_obj_as(v2.RetModelSchema, j["data"])  # type: ignore

        return None


class IOUpgradeError(Exception):
    """Exception for cases where upgrade fails."""

    def __init___(self, model: Any, target_type: type):
        """Return a new IOUpgradeException.

        Args:
            model (Any): Model that is being upgraded
            target_type (type): The target type of the upgrade process.
        """
        msg = f"Unable to upgrade {type(model)} to {target_type} - No 'upgrade' method available"
        Exception.__init__(self, msg)


class ParserConfigurationError(Exception):
    """Custom exception for invalid Parser configurations."""

    def __init__(self, path: Any, file: Any):
        """Create a new ParserConfigurationException.

        Args:
            path (Any): path setting
            file (Any): file setting
        """
        msg = f"Invalid path and file combination:\n\tPath = {path}\n\tFile = {file}"
        Exception.__init__(self, msg)
