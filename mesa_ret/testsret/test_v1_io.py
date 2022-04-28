"""Test cases for v1 model definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

# This import needs to be included in the test in order for the module to be
# discoverable, and therefore used during the model upgrade process.
from mesa_ret.io import v1, v2
from mesa_ret.io.parser import Parser

if TYPE_CHECKING:
    from pathlib import Path

    import mesa_ret.model  # noqa: F401

V1_MODEL = """
        {
            "version": 1,
            "data":
            {
                "agents": 10,
                "space":
                {
                    "x_max": 10,
                    "y_max": 10
                }
            }
        }
        """


def test_parse_path(tmp_path: Path):
    """Test parsing a model from a defined file path.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    p = tmp_path / "model.json"
    p.write_text(V1_MODEL)

    parser = Parser[v1.RetModelSchema](v1.RetModelSchema)
    model: v1.RetModelSchema = parser.parse(path=str(tmp_path / "model.json"))

    assert model.agents == 10
    assert model.space.x_max == 10
    assert model.space.y_max == 10


def test_parse_file(tmp_path: Path):
    """Test parsing a model from a defined file handle.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    p = tmp_path / "model.json"
    p.write_text(V1_MODEL)
    parser = Parser[v1.RetModelSchema](v1.RetModelSchema)
    with p.open() as f:
        model: v1.RetModelSchema = parser.parse(file=f)

    assert model.agents == 10
    assert model.space.x_max == 10
    assert model.space.y_max == 10


def test_v1_to_v2_upgrade(tmp_path: Path):
    """Test upgrading a v1 model to a v2 model.

    Args:
        tmp_path (Path): Temporary path fixture
    """
    p = tmp_path / "model.json"
    p.write_text(V1_MODEL)

    parser = Parser[v2.RetModelSchema](v2.RetModelSchema)
    model: v2.RetModelSchema = parser.parse(path=str(tmp_path / "model.json"))

    assert len(model.experimental_controls.numeric_parameters) == 1
    assert len(model.experimental_controls.categoric_parameters) == 0
    assert len(model.scenario_dependent_parameters.numeric_parameters) == 0
    assert len(model.scenario_dependent_parameters.categoric_parameters) == 0

    assert isinstance(model.time, v2.TimeSchema)
    assert isinstance(model.space, v2.SpaceSchema)

    assert model.space.x_max == 10
    assert model.space.y_max == 10
    assert model.space.x_min == 0
    assert model.space.y_min == 0
    assert model.space.dimensions == 2
    assert model.space.terrain_image_path == ""
    assert model.space.height_black == 1
    assert model.space.height_white == 0
    assert model.space.culture_image_path == ""
    assert model.space.cultures == []
    assert model.space.clutter_background_level == 0
    assert model.space.ground_clutter_height == 0
    assert model.space.ground_clutter_value == 0


def test_run_v1_model_as_v2(tmp_path: Path):
    """Test that a V1 model can be converted into a batch-runner and run.

    Args:
        tmp_path (Path): Temporary path fixture
    """
    p = tmp_path / "model.json"
    p.write_text(V1_MODEL)

    parser = Parser[v2.RetModelSchema](v2.RetModelSchema)
    model: v2.RetModelSchema = parser.parse(path=str(tmp_path / "model.json"))

    batchrunner = model.to_model()
    batchrunner.run_all()

    # Some basic checks on the properties of the batchrunner
    assert batchrunner.iterations == 10
    assert batchrunner.max_steps == 1000
