"""RetGen integration tests."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import mesa_ret.io.v2 as v2
from dash import Dash
from mesa_ret.io.parser import Parser
from retgen.app import create_app

if TYPE_CHECKING:
    from pathlib import Path


V2_MODEL = {
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
            "numeric_parameters": {"n_agents": {"min_val": 1, "max_val": 10, "name": "n_agents"}},
            "categoric_parameters": {},
        },
        "scenario_dependent_parameters": {"numeric_parameters": {}, "categoric_parameters": {}},
        "agent_reporters": {"id": "unique_id"},
    },
}


def test_end_to_end_app_load_file(tmp_path: Path):
    """Test end to end app where loaded from file.

    Args:
        tmp_path (Path): Temporary directory fixture
    """
    p = tmp_path / "model.json"
    p.write_text(json.dumps(V2_MODEL))

    parser = Parser[v2.RetModelSchema](v2.RetModelSchema)
    schema = parser.parse(str(p))

    app = create_app(schema.model_module_name, schema.model_name, schema)
    assert isinstance(app, Dash)
