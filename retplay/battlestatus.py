"""Create agent data panel."""
from __future__ import annotations
import dash_bootstrap_components as dbc
from dash import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ret.visualisation.json_models import RetPlayAgent  # noqa: TC002


class BattleStatus(dbc.Container):
    """A class to display data for the Battle Status (Force Deg / Total # of Units).

    The panel is fixed in the RetPlay interface to the left of the map.
    """

    def __init__(self, agents: list[RetPlayAgent]):
        """Create panel.

        Creates the interface components and places them on the display panel.

        Args:
            agents (list[RetPlayAgent]): A list of agents in the playback file.
        """
        self.displayed_agent_id = None

        total_agents = len(agents)
        friendly_agents = sum(1 for agent in agents if agent.affiliation == "FRIENDLY")
        hostile_agents = sum(1 for agent in agents if agent.affiliation == "HOSTILE")

        total_label = html.Div("Total Agents:")
        friendly_label = html.Div("Friendly Agents:")
        hostile_label = html.Div("Hostile Agents:")

        header_row = dbc.Row([dbc.Col(html.H6("Battle Status"))], justify="between")
        total_row = dbc.Row([dbc.Col(total_label), dbc.Col(total_agents)], justify="between")
        friendly_row = dbc.Row(
            [dbc.Col(friendly_label), dbc.Col(friendly_agents)], justify="between"
        )
        hostile_row = dbc.Row([dbc.Col(hostile_label), dbc.Col(hostile_agents)], justify="between")

        layout = [
            header_row,
            total_row,
            friendly_row,
            hostile_row,
        ]

        super().__init__(children=layout)
