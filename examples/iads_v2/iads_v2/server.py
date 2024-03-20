"""IADS server."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules.ContinuousVisulization import CanvasContinuous

from . import iads_constants as iads_constants
from .model import IADSIntegration
from .portrayal import unit_portrayal

if TYPE_CHECKING:
    from typing import Any

background_image = os.path.abspath(os.path.join(iads_constants.IMAGE_DIR, "Base_map.png"))
IADS_canvas = CanvasContinuous(unit_portrayal, 1084, 830, background_image)
model_params: dict[Any, Any] = {}

server = ModularServer(IADSIntegration, [IADS_canvas], "IADS", model_params)
