"""RetGen python class."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
from dash import html

from retgen.retgen_input_types import RetGenTitle
from retgen.retgen_ui_controller import RetGenController

if TYPE_CHECKING:
    from typing import Optional

    from dash.dash import Dash
    from mesa_ret.io.v2 import RetModelSchema


class RetGenV2:
    """RetGen class for generating and managing the RetGen dashboard."""

    def __init__(
        self,
        dash_app: Dash,
        model_module: str,
        model_class: str,
        ret_model: Optional[RetModelSchema] = None,
    ) -> None:
        """Instantiate RetGen object.

        Args:
            dash_app (Dash): Dash app to generate components for.
            model_module (str): The name of the model module.
            model_class (str): The name of the model class
            ret_model (Optional[RetModelSchema]): RetModel to generate components for, if an
                existing schema exists. If not, all parameters will be generated solely from the
                model class. Defaults to None.
        """
        import_module(model_module)
        self._dash_app = dash_app
        self._current_working_model = ret_model
        self._component_generator = RetGenController(model_module, model_class, ret_model)

    def change_current_working_model(self, ret_model: RetModelSchema) -> None:
        """Define model to be worked on by RetGen class.

        Args:
            ret_model (RetModelSchema): RetModel to generate components for.
        """
        self._current_working_model = ret_model

    def generate_dash_interface(self, app: Dash):
        """Generate dash components for the parameters of the current working model.

        Args:
            app (Dash): Dash app to configure
        """
        app.layout = html.Div(
            style={"font-family": "Arial"},
        )

        app.layout.children = [
            RetGenTitle(),
            dbc.Container(self._component_generator.components),
        ]

        for callback in self._component_generator.callbacks:
            callback(app)
