# -*- coding: utf-8 -*-
"""Rapid Exploratory modelling Toolset (RET).

Core Objects: Model, and Agent.

"""
from __future__ import annotations

import argparse
import datetime
import importlib
import os
import pathlib
import shutil
import warnings
from ret.io import v2
from ret.io.parser import Parser
from pyshortcuts import make_shortcut


__title__ = "ret"
__version__ = "2.0.0"
__license__ = "Apache License, Version 2.0"
__copyright__ = "Copyright %s Frazer-Nash Consultancy" % datetime.date.today().year


def create_shortcut() -> None:
    """Creates desktop shortcuts for RetPlay."""
    make_shortcut(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "retplay/__init__.py"),
        name="RetPlay",
        icon=os.path.join(os.path.dirname(os.path.dirname(__file__)), "retplay/assets/favicon.ico"),
        desktop=True,
    )


def run_model() -> None:
    """Runner script for launching Ret Model."""
    arg_parser = argparse.ArgumentParser(description="Run RET model")
    arg_parser.add_argument("config", type=str, help="RET Model File")
    arg_parser.add_argument(
        "--map-location",
        type=str,
        help="RET Map File. This needs to be the path to a PNG image file.",
    )
    args = arg_parser.parse_args()

    config = args.config
    if args.map_location:
        map_location = args.map_location
    else:
        map_location = None

    run_model_from_config(config, map_location)


def run_model_from_config(config: str, map_location: str = None, map_destination: str = None):
    """Run model from configuration file.

    If a map destination is given, without a map location, it will not be used.

    Args:
        config (str): Path to configuration file.
        map_location (Optional[str]): Path to the existing map file. Defaults to None.
        map_destination (Optional[str]): Path to the new map file to be created. Defaults to None.
    """
    print(f"Loading model from configuration file [{config}]")

    parser = Parser[v2.RetModelSchema](v2.RetModelSchema)
    schema = parser.parse(config)

    # This needs to be done in order to import the module that is described in the config file
    importlib.import_module(schema.model_module_name)

    if len(schema.playback_writer) > 0 and map_location is not None:
        copy_map_file(map_location, map_destination)

    print("Running model...")

    batchrunner = schema.to_model()
    batchrunner.run_all()

    print("Run completed")


def copy_map_file(map_location: str, map_destination: str = None) -> None:
    """Copy the map file to known location and name.

    Args:
        map_location (str): The location of the map file.
        map_destination (Optional[str]): The location the map file is coped to. Defaults
            to './output/assets/base_map.png'
    """
    if map_destination is None:
        map_destination = "./output/assets/base_map.png"
    path = pathlib.Path(map_location)
    if path.exists() and path.is_file():
        if pathlib.Path(map_location).suffix != ".png":
            warnings.warn("map file not a PNG filetype.", stacklevel=2)
            return
        copy_path = os.path.abspath(map_destination)
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        shutil.copy(path, copy_path)
    else:
        warnings.warn("map file path either doesn't exist or is not a file.", stacklevel=2)
