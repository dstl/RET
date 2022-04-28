"""Example RET Model."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mesa.time import RandomActivation
from mesa_ret.model import RetModel
from mesa_ret.scenario_independent_data import ModelMetadata
from mesa_ret.space.feature import CompoundAreaFeature
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
from mesa_ret.visualisation.json_writer import JsonWriter

from . import constants
from .friendly_agent_creator import FriendlyAgentCreator
from .hostile_agent_creator import HostileAgentCreator
from .orders import ExampleOrdersController

if TYPE_CHECKING:
    from datetime import datetime, timedelta


class ExampleModel(RetModel):
    """An example model demonstrating the major functionality of RET.

    Full details of the model can be found in the user guide, part 1 - Overview.

    In summary, Friendly forces use airborne reconnaissance assets to locate Hostile air defence
    agents which are then targeted by indirect fires.  Once the fire mission is complete, Friendly
    ground forces advance towards Hostile held territory with a view to destroying them or forcing
    them to retreat.  Hostile forces have setup an ambush for Friendly forces to which they might
    react.
    """

    def __init__(self, start_time: datetime, time_step: timedelta, end_time: datetime):
        """Initialise an Example model.

        Args:
            start_time (datetime): Simulation start time.
            time_step (timedelta): Simulation time step.
            end_time (datetime): Simulation end time.
        """
        super().__init__(
            start_time=start_time,
            time_step=time_step,
            end_time=end_time,
            space=self.setup_space(),
            schedule=RandomActivation(self),
            log_config="all",
            playback_writer=JsonWriter(),
        )

        friendly_agent_creator = FriendlyAgentCreator(self)
        hostile_agent_creator = HostileAgentCreator(self)

        hostile_agents = hostile_agent_creator.create_agents()
        friendly_agents = friendly_agent_creator.create_agents()

        ExampleOrdersController.set_friendly_orders(self, friendly_agents + hostile_agents)
        ExampleOrdersController.set_hostile_orders(self, friendly_agents + hostile_agents)

    @staticmethod
    def get_scenario_independent_metadata() -> ModelMetadata:
        """Return the model metadata for the model."""
        subtext = [
            "This is a simple example model built in RET designed to show the use of the majority "
            "of the capabilities available to use",
            "The model is not validated and shouldn't be used for analytical purposes but is a "
            "useful resource for understanding how to build a RET model.",
        ]

        return ModelMetadata(header="Example RET Model", subtext=subtext)

    def setup_space(self):
        """Setup the space and terrain for the example model.

        Space is 20km wide and just over 13km tall.

        Setting up an area to indicate the general location of the AD unit and a compound area to
        cover the defended urban area and its approach.

        Setting up a boundary for the ambush - Hostile forces will attack only when Friendly forces
        cross the boundary

        Setting up a simple road system which units could use to move if desired (multiline with
        width)
        """
        areas = [
            constants.ad_agent_area,
            CompoundAreaFeature(
                areas=[constants.urban_area, constants.urban_approach], name="Urban Defended Area"
            ),
        ]
        boundaries = [constants.ambush_line]
        return ContinuousSpaceWithTerrainAndCulture3d(
            x_max=20000,
            y_max=13167,
            terrain_image_path=os.path.join(constants.IMAGE_DIR, "elevation_map.png"),
            height_black=0.0,
            height_white=33.0,
            culture_image_path=os.path.join(constants.IMAGE_DIR, "culture_map.png"),
            culture_dictionary={
                (64, 192, 87): constants.culture_open,
                (0, 96, 0): constants.culture_forest,
                (134, 142, 150): constants.culture_urban,
                (76, 110, 245): constants.culture_water,
            },
            features=areas + boundaries,
        )
