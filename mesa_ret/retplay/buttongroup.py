"""Step navigation buttons for Ret Play."""
import dash_bootstrap_components as dbc
from dash import html


class ButtonGroup(html.Div):
    """Buttons to navigate through Scenario Steps."""

    def __init__(self, max_timestep: int):
        """Create navigation buttons."""
        children = [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Back", id="Back-Button", n_clicks=0, style={"width": "100%"}),
                        width={"size": 1, "offset": 1},
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Forward", id="Forward-Button", n_clicks=0, style={"width": "100%"}
                        ),
                        width=1,
                    ),
                ],
                justify="start",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Label("Current Timestep: "),
                        width=1,
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="Timestep-Selector",
                            type="number",
                            min=0,
                            max=max_timestep,
                        ),
                        width=2,
                    ),
                ],
                justify="start",
            ),
        ]

        super().__init__(children=children)
