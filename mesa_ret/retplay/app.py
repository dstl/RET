"""Dash app for RetPlay."""
import glob
import os
import shutil
import warnings
from datetime import datetime
from json import load, loads
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, callback_context, html, no_update
from mesa_ret.visualisation.json_models import JsonOutObject, RetPlayAgent  # noqa: TC002
from pydantic import parse_obj_as

from retplay.agentdatapanel import AgentDataPanel
from retplay.buttongroup import ButtonGroup
from retplay.header import RetPlayHeader
from retplay.retplaymaps import RetPlayMapFromPNG


def create_app(playback: str) -> Dash:
    """Create RET Play application from playback folder.

    Args:
        playback (str): Path to playback folder

    Returns:
        Dash: Dash application
    """
    external_stylesheets = [dbc.themes.CERULEAN]

    file, folder = validate_playback_data(playback)
    assets_path = copy_playback_assets(folder)

    with open(file) as f:
        json_blob = load(f)

    print(f"Loading model from playback file in folder [{playback}]")
    model = parse_obj_as(JsonOutObject, json_blob)

    model_map = RetPlayMapFromPNG(
        map_size=model.initial_data.map_size,
        steps=model.step_data,
        assets_path=assets_path,
    )

    app = Dash(__name__, external_stylesheets=external_stylesheets)
    buttons = ButtonGroup(model_map.n_steps)
    header = RetPlayHeader()
    display_panel = AgentDataPanel(model.step_data[0].agents)
    app.layout = html.Div(
        children=[
            header,
            dbc.Row([dbc.Col(display_panel, width=3), dbc.Col(model_map, width=9)]),
            buttons,
        ],
        id="Layout",
    )

    @app.callback(
        Output("Timestep-Selector", "value"),
        Input("Back-Button", "n_clicks"),
        Input("Forward-Button", "n_clicks"),
    )
    def forward_back_clicked(back, forward):
        clicked = [p["prop_id"] for p in callback_context.triggered][0]
        if "Back-Button" in clicked:
            if model_map.step > 0:
                return model_map.step - 1
        if "Forward-Button" in clicked:
            if model_map.step < model_map.n_steps:
                return model_map.step + 1
        else:
            return model_map.step

    @app.callback(
        [Output("Agent-Dropdown", "value"), Output(str("Map from PNG"), "children")],
        Input("Timestep-Selector", "value"),
        Input({"type": "map-marker", "index": ALL}, "n_clicks"),
    )
    def timestep_changed(value, marker_clicked):
        changed_id = [p["prop_id"] for p in callback_context.triggered][0]
        current_agent = display_panel.displayed_agent_id
        if "Timestep-Selector" in changed_id:
            if value in range(0, model_map.n_steps):
                model_map.step = value
                new_markers = model_map.update_agent_positions()
            else:
                new_markers = no_update
        else:
            new_markers = no_update
            current_agent = loads(changed_id.split(".")[0])["index"]
        return current_agent, new_markers

    @app.callback(
        [
            Output("Name-Display", "children"),
            Output("ID-Display", "children"),
            Output("Type-Display", "children"),
            Output("Affiliation-Display", "children"),
            Output("X-Display", "children"),
            Output("Y-Display", "children"),
            Output("Z-Display", "children"),
            Output("Active-Order-Display", "children"),
            Output("Status-Display", "children"),
        ],
        Input("Agent-Dropdown", "value"),
    )
    def update_agent_data_callback(selected_agent_id):
        display_panel.displayed_agent_id = selected_agent_id
        outputs = update_agent_data(selected_agent_id, model.step_data[model_map.step].agents)
        return outputs

    return app


def validate_playback_data(folder: str) -> tuple[Path, Path]:
    """Validate playback data folder.

    Checks the folder passed in by the caller to ensure the expected file structure is present.

    Args:
        folder (str): The playback data folder specified by the caller.

    Raises:
        TypeError: Invalid folder properties.

    Returns:
        tuple[Path, Path]: Path to the playback JSON file. Path to the playback assets folder.
    """
    if not os.path.isdir(folder):
        raise TypeError("The folder selected does not exist.")

    playback_file = Path(folder, "playback.json")
    if not os.path.isfile(playback_file):
        raise TypeError("Playback file does not exist in selected folder.")

    assets_dir = Path(folder, "assets")
    if os.path.isdir(assets_dir):
        icons_dir = Path(assets_dir, "icons")
        if os.path.isdir(icons_dir):
            if glob.glob(str(icons_dir) + "/*.svg") == []:
                warnings.warn("No icon images present. Default icons will be used.")
        else:
            warnings.warn(
                "Icons folder does not exist in selected folder. Default icons will be used."
            )
    else:
        raise TypeError("Assets folder does not exist in selected folder.")

    map_file = Path(assets_dir, "base_map.png")
    if not os.path.isfile(map_file):
        warnings.warn("No map file present. No image will be used.")

    return playback_file, assets_dir


def copy_playback_assets(assets_dir) -> Path:
    """Copy the playback assets to a relative path.

    Copies the filetree of the playback assets folder to a folder relative to the Dash app,
    allowing Dash to access and display the assets.

    Args:
        assets_dir (str): Path to the assets directory to be copied.

    Returns:
        folder_name (Path): Path to the folder relative to the Dash app.
    """
    timestring = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = Path("assets", "." + timestring)
    print("Loading playback assets...")
    shutil.copytree(assets_dir, Path(Path(__file__).parent.resolve(), folder_name))

    return folder_name


def update_agent_data(selected_agent_id: int, agents: list[RetPlayAgent]) -> list[str]:
    """Obtain data values for selected agent.

    This method is used by the callback which is activated when a user selects a new agent to
    display in the data panel, or when the timestep displayed on the map changes.

    Args:
        selected_agent_id (int): The unique ID of the agent selected by the user.
        agents (list[RetPlayAgent]): A list of agents in the playback file.

    Returns:
        outputs (list[str]): The output data for display in RetPlay.
    """
    agent_name = "No data found for selected agent."
    agent_id = "No data found for selected agent."
    agent_type = "No data found for selected agent."
    agent_affiliation = "No data found for selected agent."
    x_pos = "No data."
    y_pos = "No data."
    z_pos = "No data."
    agent_active_order = "No data found for selected agent."
    agent_status = "No data found for selected agent."

    for agent in agents:
        if selected_agent_id == agent.id:
            agent_name = agent.name
            agent_id = str(agent.id)
            agent_type = agent.agent_type
            agent_affiliation = agent.affiliation
            x_pos = get_formatted_position(agent.pos[0])
            y_pos = get_formatted_position(agent.pos[1])
            if len(agent.pos) > 2:
                z_pos = get_formatted_position(agent.pos[2])
            else:
                z_pos = "No data available."
            agent_active_order = agent.active_order
            if agent.killed:
                agent_status = "Killed"
            else:
                agent_status = "Active"

    return [
        agent_name,
        agent_id,
        agent_type,
        agent_affiliation,
        x_pos,
        y_pos,
        z_pos,
        agent_active_order,
        agent_status,
    ]


def get_formatted_position(pos: float) -> str:
    """Get the position of an agent formatted for display in RetPlay.

    Args:
        pos (float): The position of the agent in the space.

    Returns:
        str: The position of the agent as an integer unless this less than 4 digits, then it is
            given to 4 significant figures.
    """
    if pos > 1000:
        rtn = "{:g}".format(float("{:.0f}".format(pos))) + " m,"
    else:
        rtn = "{:g}".format(float("{:.4g}".format(pos))) + " m,"
    return rtn
