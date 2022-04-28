"""Utility functions for creating visualisations."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.register_manager import RetRegisterManager
from mesa_ret.visualisation.json_writer import JsonWriter
from mesa_ret.visualisation.playback_writer import PlaybackWriter

if TYPE_CHECKING:
    from typing import Optional

__writer_register__: Optional[RetRegisterManager] = None


def get_playback_writer_register() -> RetRegisterManager:
    """Get the playback writer register or create it if it does not exist.

    Returns:
        RetRegisterManager: The playback writer register.
    """
    global __writer_register__

    if not __writer_register__:
        __writer_register__ = RetRegisterManager[PlaybackWriter]()
        __writer_register__.register("JsonWriter", lambda: JsonWriter())

    return __writer_register__
