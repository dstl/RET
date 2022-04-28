"""Installation tests."""
from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import pytest
from mesa_ret import run_model_from_config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def v2_model():
    """V2 model JSON configuration data as a dictionary.

    Returns:
        dict: JSON configuration string stored as a dictionary.
    """
    return {
        "version": 2,
        "data": {
            "iterations": 5,
            "max_steps": 2000,
            "model_module_name": "testsret.test_v2_io",
            "model_name": "ParamModel",
            "space": {
                "dimensions": 2,
                "x_max": 100,
                "y_max": 90,
                "x_min": -5,
                "y_min": -10,
                "terrain_image": "",
                "height_black": 1,
                "height_white": 0,
                "culture_image_path": "",
                "cultures": [],
                "clutter_background_level": 0,
                "ground_clutter_height": 0,
                "ground_clutter_value": 0,
            },
            "time": {
                "start_time": "2021-01-01T00:00",
                "end_time": "2021-01-02T00:00",
                "time_step": "00:05:00",
            },
            "experimental_controls": {
                "numeric_parameters": {
                    "n_agents": {"min_val": 1, "max_val": 10, "name": "n_agents"}
                },
                "categoric_parameters": {},
            },
            "scenario_dependent_parameters": {"numeric_parameters": {}, "categoric_parameters": {}},
            "agent_reporters": {"id": "unique_id"},
        },
    }


def test_run_model(tmp_path: Path, capsys, v2_model):
    """Test that the full model can run.

    Args:
        tmp_path (Path): Temporary directory fixture
        capsys: Print output capture
        v2_model: dictionary of json configuration data
    """
    p = tmp_path / "test_input.json"
    p.write_text(json.dumps(v2_model))

    run_model_from_config(str(p))

    out, _ = capsys.readouterr()

    assert "Run completed" in str(out)
    assert "Running model..." in str(out)
    assert "Loading model from configuration file" in str(out)


def test_run_model_with_map_copy(tmp_path: Path, v2_model):
    """Test running the model supplying a path for a map file to copy.

    Args:
        tmp_path (Path): Temporary directory fixture
        v2_model: dictionary of json configuration data
    """
    v2_model["data"]["playback_writer"] = "JsonWriter"
    path_input = tmp_path / "test_input.json"
    path_input.write_text(json.dumps(v2_model))

    path_map = tmp_path / "test_map.png"
    path_map.write_text("map_data")

    path_map_destination = tmp_path / "./output/assets/base_map.png"

    run_model_from_config(str(path_input), str(path_map), str(path_map_destination))

    with open(path_map_destination) as map:
        assert map.read() == "map_data"


def test_run_model_with_invalid_map_filetype(tmp_path: Path, v2_model):
    """Test running the model but supplying a path with an map file extension.

    Args:
        tmp_path (Path): Temporary directory fixture
        v2_model: dictionary of json configuration data
    """
    v2_model["data"]["playback_writer"] = "JsonWriter"
    path_input = tmp_path / "test_input.json"
    path_input.write_text(json.dumps(v2_model))

    path_map = tmp_path / "test_map.file_extension"
    path_map.write_text("map_data")

    path_map_destination = tmp_path / "./output/assets/base_map.png"

    with warnings.catch_warnings(record=True) as w:
        run_model_from_config(str(path_input), str(path_map), str(path_map_destination))
        assert len(w) == 1
        assert "map file not a PNG filetype." in str(w[0].message)


def test_run_model_with_no_map_copy(tmp_path: Path, v2_model):
    """Test running the model but supplying a path that doesn't point to a map file.

    Args:
        tmp_path (Path): Temporary directory fixture
        v2_model: dictionary of json configuration data
    """
    v2_model["data"]["playback_writer"] = "JsonWriter"
    path_input = tmp_path / "test_input.json"
    path_input.write_text(json.dumps(v2_model))

    path_map = tmp_path / "missing_map.png"

    path_map_destination = tmp_path / "./output/assets/base_map.png"

    with warnings.catch_warnings(record=True) as w:
        run_model_from_config(str(path_input), str(path_map), str(path_map_destination))
        assert len(w) == 1
        assert "map file path either doesn't exist or is not a file." in str(w[0].message)
