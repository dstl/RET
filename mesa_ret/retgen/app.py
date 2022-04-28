"""Launcher for RET Gen."""
from __future__ import annotations

from argparse import ArgumentParser
from typing import TYPE_CHECKING

import dash
import dash_bootstrap_components as dbc
from mesa_ret.io import v2
from mesa_ret.io.parser import Parser

from retgen.retgen_v2 import RetGenV2

if TYPE_CHECKING:
    from typing import Optional


def create_app(
    model_module: str, model_name: str, schema: Optional[v2.RetModelSchema] = None
) -> dash.Dash:
    """Create new RET Gen application.

    Args:
        model_module (str): Name of module storing the model.
        model_name (str): Name of the model to run.
        schema (Optional[v2.RetModelSchema]): Pre-existing schema, if it exists. Defaults to None.

    Returns:
        dash.Dash: Dash components representing the model
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

    retgen = RetGenV2(app, model_module, model_name, schema)
    retgen.generate_dash_interface(app)

    return app


def run_model():
    """Run RET Generation.

    Raises:
        ValueError: Model run with invalid configuration settings
    """
    parser = ArgumentParser("RET Generation")
    parser.add_argument("--file", type=str, help="Existing RET Gen configuration file.")
    parser.add_argument("--module", type=str, help="Module containing new model to create.")
    parser.add_argument("--cls", type=str, help="Name of new model to run")

    args = parser.parse_args()

    schema: Optional[v2.RetModelSchema] = None
    model = args.cls
    module = args.module

    if args.file:
        parser = Parser[v2.RetModelSchema](v2.RetModelSchema)
        schema = parser.parse(args.file)
        model = schema.model_name
        module = schema.model_module_name

    if (not args.module or not args.cls) and schema is None:
        raise ValueError("Either a file, or module and cls must be provided.")

    app = create_app(module, model, schema)
    app.run_server(debug=True)
