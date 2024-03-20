"""Create agent data panel."""
from __future__ import annotations
from typing import TYPE_CHECKING
import dash_bootstrap_components as dbc
from dash import dcc, html

if TYPE_CHECKING:
    from ret.visualisation.json_models import RetPlayAgent


class AgentDataPanel(dbc.Container):
    """A class to display data for a selected agent.

    The panel is fixed in the RetPlay interface to the left of the map.
    """

    def __init__(self, agents: list[RetPlayAgent]):
        """Create panel.

        Creates the interface components and places them on the display panel.

        Args:
            agents (list[RetPlayAgent]): A list of agents in the playback file.
        """
        self.displayed_agent_id = None
        self.agent_dropdown_menu = self.create_dropdown_from_agent_list(agents)

        name_label = html.Div("Agent Name:")
        name_display = html.Div(id="Name-Display")
        name_row = dbc.Row([dbc.Col(name_label), dbc.Col(name_display)], justify="between")

        id_label = html.Div("Agent ID:")
        id_display = html.Div(id="ID-Display")
        id_row = dbc.Row([dbc.Col(id_label), dbc.Col(id_display)], justify="between")

        type_label = html.Div("Agent Type:")
        type_display = html.Div(id="Type-Display")
        type_row = dbc.Row([dbc.Col(type_label), dbc.Col(type_display)], justify="between")

        affiliation_label = html.Div("Agent Affiliation:")
        affiliation_display = html.Div(id="Affiliation-Display")
        affiliation_row = dbc.Row(
            [dbc.Col(affiliation_label), dbc.Col(affiliation_display)], justify="between"
        )

        pos_label = html.Div("Agent Position:")
        pos_label_row = dbc.Row(pos_label)

        x_label = html.Div("X: ")
        x_display = html.Div(id="X-Display")
        y_label = html.Div("Y: ")
        y_display = html.Div(id="Y-Display")
        z_label = html.Div("Z: ")
        z_display = html.Div(id="Z-Display")
        pos_row = dbc.Row(
            [
                dbc.Col(x_label, width=1),
                dbc.Col(x_display, width=3),
                dbc.Col(y_label, width=1),
                dbc.Col(y_display, width=3),
                dbc.Col(z_label, width=1),
                dbc.Col(z_display, width=3),
            ],
            justify="between",
        )

        active_order_label_row = dbc.Row(html.Div("Current Active Order:"))
        active_order_display_row = dbc.Row(html.Div(id="Formatted-Active-Order-Display"))

        status_label = html.Div("Agent Status:")
        status_display = html.Div(id="Status-Display")
        status_row = dbc.Row([dbc.Col(status_label), dbc.Col(status_display)], justify="between")

        moving_label = html.Div("Agent Moving:")
        moving_display = html.Div(id="Moving-Display")
        moving_row = dbc.Row([dbc.Col(moving_label), dbc.Col(moving_display)], justify="between")

        layout = [
            dbc.Row(self.agent_dropdown_menu),
            name_row,
            id_row,
            type_row,
            affiliation_row,
            pos_label_row,
            pos_row,
            active_order_label_row,
            active_order_display_row,
            status_row,
            moving_row,
        ]

        super().__init__(children=layout)

    def create_dropdown_from_agent_list(self, agents: list[RetPlayAgent]) -> dcc.Dropdown:
        """Create a dropdown menu.

        Creates a dropdown menu interface component populated with all agents in the playback file.

        Args:
            agents (list[RetPlayAgent]): Lit of agents to be added to the dropdown menu.

        Returns:
            dcc.Dropdown: A dropdown menu UI component populated with agents.
        """
        dropdown_options = []
        for agent in agents:
            dropdown_options.append(
                {
                    "label": agent.affiliation + " " + agent.name + " (ID: " + str(agent.id) + ")",
                    "value": agent.id,
                }
            )

        dropdown_menu = dcc.Dropdown(
            id="Agent-Dropdown",
            options=dropdown_options,
            clearable=False,
            placeholder="Select an Agent...",
        )

        return dropdown_menu
