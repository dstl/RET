"""Hostile Agent creator for Example model.

Unit icons created using: <https://spatialillusions.com/>
"""
from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

from ret.agents.agent import Affiliation
from ret.agents.agenttype import AgentType
from ret.behaviours.communicate import CommunicateWorldviewBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.move import GroundBasedMoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.creator.agents import create_agents
from ret.sensing.sensor import ContrastAcquire1dSensor, EOContrastSensorType, SensorWavelength

from . import constants, targetting, weapons

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent
    from ret.model import RetModel


class HostileAgentCreator:
    """Hostile agent creator."""

    def __init__(self, model: RetModel) -> None:
        """Initialise example firedly agent creator."""
        self.model = model

    def create_agents(self) -> list[RetAgent]:
        """Create list of agents with comms networks set up."""
        agents = self._make_agents()

        self._set_up_comms(agents)

        return agents

    def _make_agents(self):
        """Make the hsotile agents.

        - 1x Anti armour dismounts
        - 1x Air Defence detachment
        - 2x Mechanised Infantry sections
        """
        air_def = create_agents(
            number=1,
            model=self.model,
            pos=constants.ad_agent_area,
            name="Red Air Defence",
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.AIR_DEFENCE,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "hostile_AD.svg"),
            killed_icon_path=os.path.join(constants.IMAGE_DIR, "icons", "hostile_AD_dead.svg"),
            move_behaviour=GroundBasedMoveBehaviour(
                base_speed=10.0,
                gradient_speed_modifiers=constants.ground_gradient_speed_modifiers,
                culture_speed_modifiers=constants.ground_culture_movement_modifiers,
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            sense_behaviour=SenseBehaviour(
                time_before_first_sense=timedelta(seconds=5),
                time_between_senses=timedelta(seconds=60),
            ),
            sensors=[
                ContrastAcquire1dSensor(
                    magnification=1.0,
                    performance_curve=constants.contrast_performance_curve,
                    johnson_criteria=constants.johnson_criteria,
                    sensor_type=EOContrastSensorType(),
                    wavelength=SensorWavelength(600),
                )
            ],
            fire_behaviour=targetting.DirectFireBehaviour(
                aim_error=30.0,
                hostile_agent_resolver=targetting.AirTargetResolver(),
                target_selector=targetting.RandomTargetSelectorWithAim(),
            ),
            weapons=[
                weapons.TargetTypeSpecificPKillWeapon(
                    name="Air Defence",
                    time_before_first_shot=timedelta(seconds=5),
                    time_between_rounds=timedelta(seconds=0.1),
                    radius=5.0,
                    p_kill_dict={AgentType.AIR: 0.01, AgentType.INFANTRY: 0.0005},
                    min_range=100,
                    max_range=1500,
                )
            ],
        )

        mech_inf_sects = create_agents(
            number=2,
            model=self.model,
            pos=constants.urban_area,
            name=["Red Mech Inf Sect 1", "Red Mech Inf Sect 2"],
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.ARMOUR,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "hostile_mech_inf.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "hostile_mech_inf_dead.svg"
            ),
            reflectivity=0.081,
            critical_dimension=2.0,
            temperature=20.0,
            move_behaviour=GroundBasedMoveBehaviour(
                base_speed=15.0,
                gradient_speed_modifiers=constants.ground_gradient_speed_modifiers,
                culture_speed_modifiers=constants.ground_culture_movement_modifiers,
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            sense_behaviour=SenseBehaviour(
                time_before_first_sense=timedelta(seconds=5),
                time_between_senses=timedelta(seconds=60),
            ),
            sensors=[
                ContrastAcquire1dSensor(
                    magnification=1.0,
                    performance_curve=constants.contrast_performance_curve,
                    johnson_criteria=constants.johnson_criteria,
                    sensor_type=EOContrastSensorType(),
                    wavelength=SensorWavelength(600),
                )
            ],
            communicate_worldview_behaviour=CommunicateWorldviewBehaviour(),
            fire_behaviour=targetting.DirectFireBehaviour(
                aim_error=10.0,
                hostile_agent_resolver=targetting.GroundTargetResolver(),
                target_selector=targetting.RandomTargetSelectorWithAim(),
            ),
            weapons=[
                weapons.TargetTypeSpecificPKillWeapon(
                    name="Red Cannon",
                    time_before_first_shot=timedelta(seconds=5),
                    time_between_rounds=timedelta(seconds=2),
                    radius=5.0,
                    p_kill_dict={
                        AgentType.AIR_DEFENCE: 0.8,
                        AgentType.ARMOUR: 0.7,
                        AgentType.INFANTRY: 0.2,
                        AgentType.GENERIC: 0.5,
                    },
                    min_range=200,
                    max_range=1500,
                )
            ],
        )

        anti_armour = create_agents(
            number=1,
            model=self.model,
            pos=constants.pos_anti_armour_start,
            name="Red Anti Armour",
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.INFANTRY,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "hostile_anti_armour.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "hostile_anti_armour_dead.svg"
            ),
            move_behaviour=GroundBasedMoveBehaviour(
                base_speed=1.5,
                gradient_speed_modifiers=constants.ground_gradient_speed_modifiers,
                culture_speed_modifiers=constants.ground_culture_movement_modifiers,
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            sense_behaviour=SenseBehaviour(
                time_before_first_sense=timedelta(seconds=5),
                time_between_senses=timedelta(seconds=60),
            ),
            sensors=[
                ContrastAcquire1dSensor(
                    magnification=1.0,
                    performance_curve=constants.contrast_performance_curve,
                    johnson_criteria=constants.johnson_criteria,
                    sensor_type=EOContrastSensorType(),
                    wavelength=SensorWavelength(600),
                    height_of_sensor=2,
                )
            ],
            communicate_worldview_behaviour=CommunicateWorldviewBehaviour(),
            fire_behaviour=targetting.DirectFireBehaviour(
                aim_error=5.0, target_selector=targetting.RandomTargetSelectorWithAim()
            ),
            weapons=[
                weapons.TargetTypeSpecificPKillWeapon(
                    name="Red Cannon",
                    time_before_first_shot=timedelta(seconds=15),
                    time_between_rounds=timedelta(seconds=30),
                    radius=15.0,
                    p_kill_dict={AgentType.ARMOUR: 0.85, AgentType.GENERIC: 0.5},
                    min_range=400,
                    max_range=2000,
                )
            ],
        )

        hostile_agents = air_def + mech_inf_sects + anti_armour

        return hostile_agents

    def _set_up_comms(self, agents: list[RetAgent]):
        """Setup comms between hostile agents.

        Args:
            agents (list[RetAgents]): list of agents.

        - Mech inf sects talk to each other
        - AD and AA talk to Mech inf sects
        """
        mech_inf_1 = [a for a in agents if a.name == "Red Mech Inf Sect 1"][0]
        mech_inf_2 = [a for a in agents if a.name == "Red Mech Inf Sect 2"][0]
        anti_armour = [a for a in agents if a.name == "Red Anti Armour"][0]
        air_def = [a for a in agents if a.name == "Red Air Defence"][0]

        mech_inf_1.communication_network.add_recipient(mech_inf_2)
        mech_inf_2.communication_network.add_recipient(mech_inf_1)

        anti_armour.communication_network.add_recipient(mech_inf_1)
        anti_armour.communication_network.add_recipient(mech_inf_2)

        air_def.communication_network.add_recipient(mech_inf_1)
        air_def.communication_network.add_recipient(mech_inf_2)
