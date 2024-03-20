"""Dash app for RetPlay."""
import glob
import os
import shutil
import warnings
from datetime import datetime
from json import load, loads
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory
import multiprocessing
from typing import Any, Tuple, Dict

import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, callback_context, html, no_update, dcc
from pydantic import parse_obj_as
from retplay.agentdatapanel import AgentDataPanel
from retplay.buttongroup import ButtonGroup
from retplay.header import RetPlayHeader
from retplay.retplaymaps import RetPlayMapFromPNG
from retplay.battlestatus import BattleStatus

from ret.visualisation.json_models import JsonOutObject, RetPlayAgent, RetPlaySpace  # noqa: TC002


def create_app(playback: str) -> Dash:  # noqa: C901
    """Create RET Play application from playback folder.

    Args:
        playback (str): Path to playback folder

    Returns:
        Dash: Dash application
    """
    external_stylesheets = [dbc.themes.CERULEAN]

    if playback is None:
        model, model_map, buttons, display_panel, battle_status = generate_empty_data()
    else:
        model, model_map, buttons, display_panel, battle_status = load_in_new_data(playback)

    app = Dash(__name__, external_stylesheets=external_stylesheets)
    header = RetPlayHeader()
    app.title = "RetPlay"
    app._favicon = "favicon.ico"
    app.layout = html.Div(
        children=[
            header,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="Display_Panel_Component", children=display_panel),
                            html.Div(
                                id="Battle_Status_Component",
                                style={"padding": "2rem"},
                                children=battle_status,
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col([html.Div(id="Model_Map_Component", children=model_map)], width=10),
                ]
            ),
            html.Div(id="Buttons_Component", children=buttons),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.Input(
                            id="folder-path-display",
                            placeholder="Selected folder path will appear here",
                            readonly=True,
                            style={"width": "90%"},
                        )
                    ],
                    width=10,
                ),
                align="start",
            ),
            dcc.Store(id="initial-folder-path", data={"path": playback}),
        ],
        id="Layout",
    )

    @app.callback(
        [
            Output("Timestep-Selector", "value"),
            Output("aux", "children"),
            Output("interval-component", "max_intervals"),
        ],
        [
            Input("Play-Button", "n_clicks"),
            Input("Rewind-Button", "n_clicks"),
            Input("Pause-Button", "n_clicks"),
            Input("Back-Button", "n_clicks"),
            Input("Forward-Button", "n_clicks"),
            Input("interval-component", "n_intervals"),
        ],
        [
            State("Timestep-Selector", "value"),
            State("aux", "children"),
        ],
    )
    def media_control(play, rewind, pause, back, forward, n_intervals, timestep, aux_value):
        clicked_id = [p["prop_id"] for p in callback_context.triggered][0].split(".")[0]

        if timestep is None:
            timestep = 0  # Default value

        if clicked_id == "Play-Button":
            aux_value = "play"
        elif clicked_id == "Rewind-Button":
            aux_value = "rewind"
        elif clicked_id == "Pause-Button":
            aux_value = "pause"
        elif clicked_id == "Back-Button":
            # Changes timestep backwards only when the interval is paused
            if timestep > 0 and aux_value == "pause":
                return timestep - 1, aux_value, 0
        elif clicked_id == "Forward-Button":
            # Changes timestep forwards only when the interval is paused
            if timestep < model_map.n_steps and aux_value == "pause":
                return timestep + 1, aux_value, 0

        if aux_value == "play" and timestep < model_map.n_steps:
            return timestep + 1, aux_value, -1
        elif aux_value == "rewind" and timestep > 0:
            return timestep - 1, aux_value, -1
        else:
            return timestep, aux_value, 0

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
        [Output("folder-path-display", "value")],
        [Input("Open-Folder-Button", "n_clicks"), Input("initial-folder-path", "data")],
    )
    def load_outputs(n_clicks, data):
        # Trigger the Tkinter dialog
        if not callback_context.triggered:
            if data and "path" in data:
                path = data["path"]
                if path is None:
                    return no_update
                else:
                    return [format_path(path)]
            else:
                return no_update
        if n_clicks:
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=tkinter_process, args=(queue,))
            p.start()
            p.join()
            folder_path = queue.get()
            if folder_path == "":
                return no_update
            else:
                return [format_path(folder_path)]

        return no_update

    @app.callback(
        [
            Output("Name-Display", "children"),
            Output("ID-Display", "children"),
            Output("Type-Display", "children"),
            Output("Affiliation-Display", "children"),
            Output("X-Display", "children"),
            Output("Y-Display", "children"),
            Output("Z-Display", "children"),
            Output("Formatted-Active-Order-Display", "children"),
            Output("Status-Display", "children"),
            Output("Moving-Display", "children"),
            Output("Time-Display", "children"),
        ],
        Input("Agent-Dropdown", "value"),
    )
    def update_agent_data_callback(selected_agent_id):
        display_panel.displayed_agent_id = selected_agent_id
        if playback is None:
            return [""] * 11
        else:
            outputs = update_agent_data(
                selected_agent_id,
                model.step_data[model_map.step].agents,
                model.step_data[model_map.step].model_time,
            )
            return outputs

    @app.callback(
        [
            Output("Display_Panel_Component", "children"),
            Output("Battle_Status_Component", "children"),
            Output("Model_Map_Component", "children"),
            Output("Buttons_Component", "children"),
        ],
        [Input("folder-path-display", "value")],
    )
    def update_with_new_data(folder_path):
        nonlocal model
        nonlocal model_map
        nonlocal buttons
        nonlocal display_panel
        nonlocal battle_status
        nonlocal playback

        model, model_map, buttons, display_panel, battle_status = load_in_new_data(folder_path)

        playback = folder_path

        model_map_component = html.Div(model_map, key=f"map-{folder_path}")
        # Have to return component with new id otherwise map will not update

        outputs = display_panel, battle_status, model_map_component, buttons
        return outputs

    return app


def generate_empty_data() -> (
    Tuple[Any, RetPlayMapFromPNG, ButtonGroup, AgentDataPanel, BattleStatus]
):
    """Generates empty data to allow initialisation without data.

    Returns:
        list: A list containing initialized components in the following order:
              - model (dict): an empty dict - placeholder until data is loaded.
              - model_map (RetPlayMapFromPNG): A map representation for the playback data.
              - buttons (ButtonGroup): A group of buttons for playback control.
              - display_panel (AgentDataPanel): A panel displaying data of agents.
              - battle_status (BattleStatus): A component showing the status of the battle..
    """
    print("Starting RETPlay without any initial data loaded")

    model: Dict[Any, Any] = {}

    buttons = ButtonGroup(0)

    model_map = RetPlayMapFromPNG(
        map_size=RetPlaySpace(x_min=0, x_max=1, y_min=0, y_max=1), steps=[], assets_path=Path("")
    )

    display_panel = AgentDataPanel([])
    battle_status = BattleStatus([])

    return model, model_map, buttons, display_panel, battle_status


def load_in_new_data(
    folder_path: str,
) -> Tuple[Any, RetPlayMapFromPNG, ButtonGroup, AgentDataPanel, BattleStatus]:
    """Loads in data and creates all relevant objects for RetPlay.

    Args:
        path (str): The file path to be of output folder.

    Returns:
        list: A list containing initialized components in the following order:
              - model (JsonOutObject): The parsed model object from the JSON data.
              - model_map (RetPlayMapFromPNG): A map representation for the playback data.
              - buttons (ButtonGroup): A group of buttons for playback control.
              - display_panel (AgentDataPanel): A panel displaying data of agents.
              - battle_status (BattleStatus): A component showing the status of the battle..
    """
    try:
        file, folder = validate_playback_data(folder_path)
        assets_path = copy_playback_assets(folder)

        with open(file) as f:
            json_blob = load(f)

        print(f"Loading model from playback file in folder [{folder_path}]")
        model = parse_obj_as(JsonOutObject, json_blob)

        model_map = RetPlayMapFromPNG(
            map_size=model.initial_data.map_size,
            steps=model.step_data,
            assets_path=assets_path,
        )

        buttons = ButtonGroup(model_map.n_steps)

        display_panel = AgentDataPanel(model.step_data[0].agents)
        battle_status = BattleStatus(model.step_data[0].agents)

        return model, model_map, buttons, display_panel, battle_status

    except Exception as e:
        print(f"unable to load data from {folder_path} due to error: {e}")
        return generate_empty_data()


def format_path(path: str) -> str:
    """Formats a given file path to windows absolute path.

    Args:
        path (str): The file path to be formatted. It can be an absolute path or
                    a relative path starting with '.'.

    Returns:
        str: The formatted absolute file path.
    """
    if path[0] == ".":
        path = os.getcwd() + "\\" + path[1:]
        path = os.path.normpath(path)

    return path


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
                warnings.warn("No icon images present. Default icons will be used.", stacklevel=2)
        else:
            warnings.warn(
                "Icons folder does not exist in selected folder. Default icons will be used.",
                stacklevel=2,
            )
    else:
        raise TypeError("Assets folder does not exist in selected folder.")

    map_file = Path(assets_dir, "base_map.png")
    if not os.path.isfile(map_file):
        warnings.warn("No map file present. No image will be used.", stacklevel=2)

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


def update_agent_data(selected_agent_id: int, agents: list[RetPlayAgent], time: str) -> list[str]:
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
    agent_moving = "No data found for selected agent."
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
            if "move" in agent_active_order.lower():
                agent_moving = "Moving"
            else:
                agent_moving = "Stationary"

    if agent_active_order.startswith("Compound Task:"):
        start_of_task = agent_active_order.find("(") + 1
        end_of_task = (
            agent_active_order.find(")")
            if agent_active_order.find(")") != -1
            else len(agent_active_order)
        )
        first_comma = (
            agent_active_order.find(",") if agent_active_order.find(",") != -1 else end_of_task
        )

        task_intro = agent_active_order[:start_of_task]
        first_task = agent_active_order[start_of_task:first_comma]
        remaining_order = (
            agent_active_order[first_comma:end_of_task] + agent_active_order[end_of_task:]
        )

        # Format First_Task
        formatted_active_order = html.Span(
            [
                html.Span(task_intro),
                html.Span(first_task, style={"fontWeight": "bold"}),
                html.Span(remaining_order),
            ]
        )
    else:
        formatted_active_order = agent_active_order

    return [
        agent_name,
        agent_id,
        agent_type,
        agent_affiliation,
        x_pos,
        y_pos,
        z_pos,
        formatted_active_order,
        agent_status,
        agent_moving,
        time,
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


def tkinter_process(queue: multiprocessing.Queue) -> None:
    """Opens a Tkinter dialog for directory selection and sends the chosen path to a queue.

    Args:
        queue (multiprocessing.Queue): A multiprocessing queue to which the selected directory
                                       path will be put.
    """
    root = Tk()
    root.withdraw()
    folder_path = askdirectory(title="Choose a RET simulation to visualise")
    root.destroy()
    queue.put(folder_path)
