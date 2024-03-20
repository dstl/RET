"""IADS model."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from mesa.time import RandomActivation
from ret.model import RetModel
from ret.space.space import ContinuousSpaceWithTerrainAndCulture3d

from . import iads_constants as iads_constants
from .friendly_agent_creator import FriendlyAgentCreator
from .hostile_agent_creator import HostileAgentCreator
from .order_controller import IADSOrderController


class IADS(RetModel):
    """A model of the IADS Scenario.

    The model runs between 00:00 to 23:55 on the 11th January 2020 with 5 minute time
    intervals.
    """

    def __init__(self, **kwargs):
        """Initialise the IADS model."""
        agent_reporters = {
            "pos": lambda a: a.pos,
        }
        tables = {"Neighbour_Numbers": ["time_step", "neighbour_number"]}

        super().__init__(
            start_time=datetime(2020, 11, 1, 0, 0),
            time_step=timedelta(minutes=5),
            end_time=datetime(2020, 11, 1, 23, 55),
            space=self.set_up_space(),
            schedule=RandomActivation(self),
            agent_reporters=agent_reporters,
            tables=tables,
            **kwargs,
        )

        friendly_agent_creator = FriendlyAgentCreator(self)
        hostile_agent_creator = HostileAgentCreator(self)

        friendly_agents = friendly_agent_creator.create_agents()
        hostile_agents = hostile_agent_creator.create_agents()

        IADSOrderController.set_friendly_orders(self, friendly_agents + hostile_agents)
        IADSOrderController.set_hostile_orders(self, friendly_agents + hostile_agents)

    def step(self):
        """Advance the model by one step and collect data."""
        self.write_table_values()
        super().step()

    def write_table_values(self):
        """Write the numbers of Amsterdam neighbours to the appropriate table."""
        neighbour_number = len(self.space.get_neighbors((400000, 200000), 150000))
        row = {"time_step": self.get_time(), "neighbour_number": neighbour_number}
        self.datacollector.add_table_row("Neighbour_Numbers", row)

    def set_up_space(self) -> ContinuousSpaceWithTerrainAndCulture3d:
        """Set up ContinuousSpaceWithTerrainAndCulture3d for IADS scenario."""
        return ContinuousSpaceWithTerrainAndCulture3d(
            x_max=542000,
            y_max=415000,
            terrain_image_path=os.path.join(iads_constants.IMAGE_DIR, "Base_map_terrain.png"),
            height_black=0.0,
            height_white=0.0,
            culture_image_path=os.path.join(iads_constants.IMAGE_DIR, "Base_map_culture.png"),
            culture_dictionary={
                (0, 255, 0): iads_constants.culture_land,
                (0, 0, 255): iads_constants.culture_sea,
            },
        )
