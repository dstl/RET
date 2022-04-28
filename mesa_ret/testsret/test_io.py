"""Test cases for model definition file Parser."""

from __future__ import annotations

from pathlib import Path

from mesa_ret.io.parser import IOUpgradeError, Parser, ParserConfigurationError
from pytest import raises


class V1Model:
    """Dummy model, to be used to test upgrading."""

    def upgrade(self) -> V2Model:
        """Upgrade method.

        Returns:
            V2Model: Updated version of the model
        """
        return V2Model("From V1")


class V2Model:
    """Dummy model, to be used to test upgrading."""

    def __init__(self, msg: str):
        """Create a new V2Model, with a message indicating source.

        Args:
            msg (str): Model description
        """
        self.msg = msg

    def upgrade(self) -> V3Model:
        """Upgrade method.

        Returns:
            V3Model: Updated version of the model
        """
        return V3Model(self.msg + "\n" + "From V2")


class V3Model:
    """Dummy model, to be used to test upgrading."""

    def __init__(self, msg: str):
        """Create a new V3Model, with a message indicating source.

        Args:
            msg (str): Model description
        """
        self.msg = msg


class V4Model:
    """Dummy model, to be used to test upgrading."""

    pass


class RecursiveModel:
    """Dummy model, for testing upgrades that get caught in infinite loops."""

    def __init__(self, loop_count: int):
        """Create a recursive model.

        Args:
            loop_count (int): Loop counter
        """
        self.loop_count = loop_count

    def upgrade(self) -> RecursiveModel:
        """Upgrade to a copy of ones self with incrementing loop counter.

        Returns:
            RecursiveModel: Updated version of the model
        """
        return RecursiveModel(self.loop_count + 1)


def test_recursive_upgrade():
    """Test that upgrade in an infinite loop times out and raises recursion error."""
    parser = Parser[V2Model](V2Model)
    with raises(RecursionError):
        parser.upgrade(RecursiveModel(0))


def test_upgrade():
    """Test upgrade from a V1Model to a V2Model."""
    parser = Parser[V2Model](V2Model)
    v2_model = parser.upgrade(V1Model())

    assert v2_model.msg == "From V1"


def test_chained_upgrade():
    """Test upgrade from a V1Model to a V3Model, via a V2Model."""
    parser = Parser[V3Model](V3Model)
    v3_model = parser.upgrade(V1Model())

    assert v3_model.msg == "From V1\nFrom V2"


def test_upgrade_fail():
    """Test error trapping where upgrade fails.

    In this instance, attempting to upgrade a V1Model to a V4Model, which is not
    possible to do as there is no suitable upgrade method on the V3Model.
    """
    parser = Parser[V4Model](V4Model)

    with raises(IOUpgradeError):
        parser.upgrade(V1Model())

    pass


def test_parser_configuration_error_no_spec():
    """Test that parser throws exception if initialised incorrectly.

    No configuration provided.
    """
    with raises(ParserConfigurationError):
        Parser[V1Model](V1Model).parse()


def test_parser_configuration_error_conflicting_spec(tmp_path: Path):
    """Test that the parser throws an exception if initialised incorrectly.

    Conflicting path and file specification provided.

    Args:
        tmp_path (Path): Temporary path fixture
    """
    with raises(ParserConfigurationError):
        p = Parser[V1Model](V1Model)
        path = tmp_path / "tmp.json"
        Path(path).touch()
        with Path(path).open() as f:
            p.parse(file=f, path=str(path))
