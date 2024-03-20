"""Step navigation buttons for Ret Play."""
import dash_bootstrap_components as dbc
from dash import html, dcc


class ButtonGroup(html.Div):
    """Buttons to navigate through Scenario Steps and loading files."""

    def __init__(self, max_timestep: int):
        """Create navigation buttons."""
        children = [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Play", id="Play-Button", n_clicks=0, style={"width": "100px"}),
                        width="auto",
                        style={"margin-right": "8px"},
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Rewind", id="Rewind-Button", n_clicks=0, style={"width": "100px"}
                        ),
                        width="auto",
                        style={"margin-right": "8px"},
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Pause", id="Pause-Button", n_clicks=0, style={"width": "100px"}
                        ),
                        width="auto",
                        style={"margin-right": "8px"},
                    ),
                    dbc.Col(
                        dbc.Button("Back", id="Back-Button", n_clicks=0, style={"width": "100px"}),
                        width="auto",
                        style={"margin-right": "8px"},
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Forward", id="Forward-Button", n_clicks=0, style={"width": "100px"}
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Open Folder",
                            id="Open-Folder-Button",
                            n_clicks=0,
                            style={"width": "120px"},
                        ),
                        width="auto",
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
                    dbc.Col(
                        dbc.Label("Current Model Time: "),
                        width="auto",
                    ),
                    dbc.Col(html.Div(id="Time-Display")),
                ],
                justify="start",
            ),
            dcc.Interval(
                id="interval-component",
                interval=1 * 1000,
                max_intervals=0,
            ),
            html.Div(id="aux", style={"display": "none"}, children=0),
            html.Div(id="interval_state", style={"display": "none"}, children=0),
        ]

        super().__init__(children=children)
