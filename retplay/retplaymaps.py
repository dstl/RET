"""Dash graph displaying RET scenario map and agents."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import dash_leaflet as dl
from dash import html

from ret.agents.affiliation import Affiliation

if TYPE_CHECKING:
    from typing import Any

    from ret.visualisation.json_models import RetPlayAgent, RetPlaySpace, RetPlayStepData


class RetPlayMapFromPNG(dl.Map):
    """A class to display a map from a PNG Image.

    This class creates a Dash-Leaflet map using a given .png image and adds a map marker for each
    agent in the scenario. The icons for agents are read from the given agent's icon at the given
    timestep.

    This class currently represents a naïve implementation of the RetPlay map feature. This class
    currently only displays data for a single timestep. Future development is expected to have the
    map iterate through multiple timesteps of data, with marker positions and images updating
    accordingly.
    """

    def __init__(
        self,
        map_size: RetPlaySpace,
        steps: list[RetPlayStepData],
        assets_path: Path,
    ) -> None:
        """Create a RetPlay map.

        Args:
            map_size (RetPlaySpace): Object containing the map size
            steps (list[RetPlayStepData]): A step data to display on the map.
            assets_path (Path): The path to the playback assets including icons.
        """
        x_scale = 360 / (map_size.x_max - map_size.x_min)
        y_scale = 180 / (map_size.y_max - map_size.y_min)
        self.scale = min(x_scale, y_scale)
        self.x_offset = (map_size.x_max + map_size.x_min) / 2
        self.y_offset = (map_size.y_max + map_size.y_min) / 2

        min_bounds = self._convert_map_coords((map_size.x_min, map_size.y_min))
        max_bounds = self._convert_map_coords((map_size.x_max, map_size.y_max))

        self.assets_path = assets_path
        self.icons_path = Path(assets_path, "icons")
        self.retplay_root_directory = Path(__file__).parent.resolve()

        self.n_steps = len(steps)
        self.step = 0
        self.step_data = steps
        self.marker_list: list[dl.Marker] = []
        self.marker_ids: list[int] = []

        if steps == []:
            pass
        else:
            for agent in self.step_data[self.step].agents:
                self.create_agent_marker(agent)

        map_path = str(Path(assets_path, "base_map.png"))
        self.map = dl.ImageOverlay(
            opacity=1.0, url=map_path, bounds=[min_bounds, max_bounds], id="Map"
        )

        children = [self.map] + self.marker_list
        style = {"width": "100%", "height": "75vh"}

        super().__init__(
            crs="Simple",
            style=style,
            bounds=[min_bounds, max_bounds],
            maxBounds=[min_bounds, max_bounds],
            children=children,
            id="Map from PNG",
            zoom=2,
        )

    def create_agent_marker(self, agent: RetPlayAgent) -> None:
        """Create a RetPlay marker for an Agent.

        Args:
            agent (RetPlayAgent): The agent to create the map marker for.
        """
        pos = agent.pos
        if pos:
            agent_icon = {
                "iconUrl": self.get_path_to_icon(agent),
                "iconSize": [35, 35],
                "iconAnchor": [12.5, 17.5],
                "popupAnchor": [0, 0],
            }
            agent_marker = dl.Marker(
                id={"type": "map-marker", "index": agent.id},
                position=self._convert_map_coords(agent.pos),
                icon=agent_icon,
                children=[self.create_tooltip(agent)],
            )
            self.marker_list.append(agent_marker)
            self.marker_ids.append(agent.id)

    def get_path_to_icon(self, agent: RetPlayAgent) -> str:
        """Get the path to the icon for an agent.

        Args:
            agent (RetPlayAgent): The agent for which the icon will be found.

        Returns:
            str: The file path of the icon.
        """
        icon = Path(self.icons_path, agent.icon)
        icon_path = Path(self.retplay_root_directory, icon)
        if os.path.isfile(icon_path):
            return str(icon)
        else:
            return self.get_default_affiliation_icon(agent)

    def get_default_affiliation_icon(self, agent: RetPlayAgent) -> str:
        """Get the default icon for the agent based on its affiliation.

        Args:
            agent (RetPlayAgent): The agent to get a default icon for.

        Returns:
            str: The relative path to the default icon.
        """
        if agent.affiliation == str(Affiliation.FRIENDLY.name):
            return str(Path("assets", "default_icons", "genericagent_friendly.svg"))
        elif agent.affiliation == str(Affiliation.HOSTILE.name):
            return str(Path("assets", "default_icons", "genericagent_hostile.svg"))
        elif agent.affiliation == str(Affiliation.NEUTRAL.name):
            return str(Path("assets", "default_icons", "genericagent_neutral.svg"))
        elif agent.affiliation == str(Affiliation.UNKNOWN.name):
            return str(Path("assets", "default_icons", "genericagent_unknown.svg"))
        else:
            warnings.warn("Unrecognised affiliation type", stacklevel=2)
            return ""

    def _convert_map_coords(self, input: tuple[float, ...]) -> tuple[float, float]:
        x = input[0] * self.scale - self.x_offset
        y = input[1] * self.scale - self.y_offset

        return (y, x)

    def update_agent_positions(self) -> list[Any]:
        """Updates agent positions.

        Called on button press, work in progress.

        Returns:
            list[Any]: New children to be displayed
        """
        new_children = [self.map]
        for agent in self.step_data[self.step].agents:
            if agent.id not in self.marker_ids:
                self.create_agent_marker(agent)
            for marker in self.marker_list:
                if marker.id == {"type": "map-marker", "index": agent.id}:
                    if marker.position != self._convert_map_coords(agent.pos):
                        marker.position = self._convert_map_coords(agent.pos)

                    icon = self.get_path_to_icon(agent)
                    if marker.icon["iconUrl"] != icon:
                        marker.icon["iconUrl"] = icon

                    marker.children = [self.create_tooltip(agent)]
                    new_children.append(marker)

        return new_children

    @staticmethod
    def create_tooltip(agent: RetPlayAgent) -> dl.Tooltip:
        """Create the tooltip for an agent.

        Adds data to the tooltip for agent name, id,  position, and status (active or killed).

        Args:
            agent (RetPlayAgent): The agent to display data for.

        Returns:
            tooltip (dl.Tooltip): The agent tooltip data.
        """
        name_component = html.P("Name: " + str(agent.name))
        id_component = html.P("ID: " + str(agent.id))

        x_string = "{:g}".format(float("{:.4g}".format(agent.pos[0])))
        y_string = "{:g}".format(float("{:.4g}".format(agent.pos[1])))

        if len(agent.pos) > 2:
            z_string = "{:g}".format(float("{:.4g}".format(agent.pos[2])))
            pos_string = f"(X: {x_string} m, Y: {y_string} m, Z: {z_string} m)"
        else:
            pos_string = f"(X: {x_string} m, Y: {y_string} m)"

        pos_component_children = [
            html.P("Position:"),
            html.P(pos_string),
        ]

        pos_component = html.P(children=pos_component_children)

        if agent.killed:
            status = "Killed"
        else:
            status = "Active"

        status_component = html.P("Status: " + status)

        tooltip_children = [
            name_component,
            id_component,
            pos_component,
            status_component,
        ]

        tooltip = dl.Tooltip(children=tooltip_children)

        return tooltip
