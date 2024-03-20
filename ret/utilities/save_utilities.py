"""Utilities for getting and creating paths."""

from __future__ import annotations
import os
from re import match
import pathlib
from typing import Union
from warnings import warn
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def validate_save_path(path: Union[str, Path]) -> Union[str, Path, None]:
    """Checks if path exists, if not get the most recent subfolder in the given directory."""
    if not os.path.exists(path):
        directory = os.path.dirname(path)
        latest_subfolder = get_latest_subfolder(directory)
        if latest_subfolder is None:
            warn(f"No path or subfolder found for: {path}", stacklevel=2)
            return None
        else:
            return os.path.join(directory, latest_subfolder)
    else:
        return path


def get_latest_subfolder(path: Union[str, Path]) -> Union[str, None]:
    """Get the most recently created subfolder in the given directory."""
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not dirs:
        return None
    latest_subfolder = max(dirs, key=lambda x: os.path.getctime(os.path.join(path, x)))
    return latest_subfolder


def add_datetime_stamp(path: str) -> str:
    """Check if the last part of the path is in the format 'YYYY-MM-DD_HH-MM-SS'.

    Args:
        path (str): The file or directory path.

    Returns:
        outpath (str): path with datetime.
    """
    if not is_timestamp_directory(path):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
        outpath = os.path.join(path, current_datetime)
        return outpath
    else:
        return path


def add_iteration_stamp(path: str, run_count: int) -> str:
    """Create run number sub directory.

    Args:
        path (str): The datetime file path.
        run_count (int): The run number.

    Returns
        outpath (str): datetime path with iteration subdirectory.
    """
    outpath = os.path.join(path, str(f"iteration_{run_count}"))
    os.makedirs(outpath, exist_ok=True)
    return outpath


def is_timestamp_directory(path: Union[str, Path]) -> bool:
    """Check if the last part of the path is in the format 'YYYY-MM-DD_HH-MM-SS'.

    Args:
        path (str): The file or directory path.

    Returns:
        bool: True if the last part of the path matches the timestamp format, False otherwise.
    """
    # Extract the directory name or last part of the path
    directory_name = pathlib.Path(path).name

    # Define the regex pattern for the timestamp
    timestamp_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"

    # Use regex to match the pattern
    return bool(match(timestamp_pattern, directory_name))
