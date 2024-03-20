"""Friendly Agent creator for Example model.

Unit icons created using: <https://spatialillusions.com/>
"""
from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

from ret.agents.agent import Affiliation, RetAgent
from ret.agents.agenttype import AgentType
from ret.behaviours.communicate import (
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.move import AircraftMoveBehaviour, GroundBasedMoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.creator.agents import create_agents
from ret.sensing.sensor import (
    ContrastAcquire1dSensor,
    EOContrastSensorType,
    SensorWavelength,
    TemperatureAcquire1dSensor,
)

from . import constants, targetting, weapons

if TYPE_CHECKING:
    from ret.model import RetModel


class FriendlyAgentCreator:
    """Friendly agent creator."""

    def __init__(self, model: RetModel) -> None:
        """Initialise example agent creator."""
        self.model = model

    def create_agents(self) -> list[RetAgent]:
        """Create list of agents with comms networks set up."""
        agents = self._make_agents()

        self._set_up_comms(agents)

        return agents

    def _make_agents(self):
        """Make the friendly agents.

        - 1x Rocket artillery detachment
        - 2x ISTAR UAV
        - 9x Armoured Inf (AI) Sect
        - 3x AI Pl (Group agent)
        - 1x AI Coy (C2 agent)
        """
        ai_coy_hq = RetAgent(
            model=self.model,
            name="Blue AI Coy",
            pos=constants.pos_ai_coy,
            affiliation=Affiliation.FRIENDLY,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_mech_inf.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "friendly_mech_inf_dead.svg"
            ),
            critical_dimension=5.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
            behaviours=[
                WaitBehaviour(),
                CommunicateOrdersBehaviour(),
                CommunicateWorldviewBehaviour(),
            ],
        )

        uavs = create_agents(
            number=2,
            model=self.model,
            pos=constants.pos_uav_start,
            name=["Blue UAV 1", "Blue UAV 2"],
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            height_bands=constants.uav_height_bands,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_UAV.svg"),
            killed_icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_UAV_dead.svg"),
            move_behaviour=AircraftMoveBehaviour(
                base_speed=50.0, height_bands=constants.uav_height_bands
            ),
            wait_behaviour=WaitBehaviour(),
            sense_behaviour=SenseBehaviour(
                time_before_first_sense=timedelta(seconds=5),
                time_between_senses=timedelta(seconds=30),
            ),
            sensors=[
                ContrastAcquire1dSensor(
                    magnification=3.0,
                    performance_curve=constants.contrast_performance_curve,
                    johnson_criteria=constants.johnson_criteria,
                    sensor_type=EOContrastSensorType(),
                    wavelength=SensorWavelength(600),
                ),
                TemperatureAcquire1dSensor(
                    magnification=1.0,
                    performance_curve=constants.temperature_performance_curve,
                    johnson_criteria=constants.johnson_criteria,
                    wavelength=SensorWavelength(600),
                ),
            ],
            communicate_worldview_behaviour=CommunicateWorldviewBehaviour(),
        )

        rocket_arty = create_agents(
            number=1,
            model=self.model,
            pos=constants.pos_rockets_start,
            name="Blue Rocket Arty",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            reflectivity=0.5,
            critical_dimension=10,
            temperature=20.0,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_rockets.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "friendly_rockets_dead.svg"
            ),
            move_behaviour=GroundBasedMoveBehaviour(
                base_speed=10.0,
                gradient_speed_modifiers=constants.ground_gradient_speed_modifiers,
                culture_speed_modifiers=constants.ground_culture_movement_modifiers,
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            fire_behaviour=FireBehaviour(),
            weapons=[
                weapons.TargetTypeSpecificPKillWeapon(
                    name="Rocket",
                    time_before_first_shot=timedelta(seconds=120),
                    time_between_rounds=timedelta(seconds=10),
                    radius=30.0,
                    p_kill_dict={
                        AgentType.AIR_DEFENCE: 0.65,
                        AgentType.ARMOUR: 0.5,
                        AgentType.INFANTRY: 0.8,
                        AgentType.GENERIC: 0.5,
                    },
                    min_range=2000,
                    max_range=50000,
                )
            ],
        )

        ai_sects_pl_1 = create_agents(
            number=3,
            model=self.model,
            pos=constants.ai_pl_1_positions,
            name=["Blue AI Sect 1 1", "Blue AI Sect 1 2", "Blue AI Sect 1 3"],
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.ARMOUR,
            reflectivity=0.5,
            critical_dimension=10,
            temperature=20.0,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_mech_inf.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "friendly_mech_inf_dead.svg"
            ),
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
                aim_error=10.0, target_selector=targetting.RandomTargetSelectorWithAim()
            ),
            weapons=[
                weapons.TargetTypeSpecificPKillWeapon(
                    name="Blue Cannon",
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

        ai_sects_pl_2 = create_agents(
            number=3,
            model=self.model,
            pos=constants.ai_pl_2_positions,
            name=["Blue AI Sect 2 1", "Blue AI Sect 2 2", "Blue AI Sect 2 3"],
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.ARMOUR,
            reflectivity=0.5,
            critical_dimension=10,
            temperature=20.0,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_mech_inf.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "friendly_mech_inf_dead.svg"
            ),
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
            fire_behaviour=FireBehaviour(target_selector=targetting.RandomTargetSelectorWithAim()),
            weapons=[
                weapons.TargetTypeSpecificPKillWeapon(
                    name="Blue Cannon",
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

        ai_sects_pl_3 = create_agents(
            number=3,
            model=self.model,
            pos=constants.ai_pl_3_positions,
            name=["Blue AI Sect 3 1", "Blue AI Sect 3 2", "Blue AI Sect 3 3"],
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.ARMOUR,
            reflectivity=0.5,
            critical_dimension=10,
            temperature=20.0,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_mech_inf.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "friendly_mech_inf_dead.svg"
            ),
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
            fire_behaviour=FireBehaviour(target_selector=targetting.RandomTargetSelectorWithAim()),
            weapons=[
                weapons.TargetTypeSpecificPKillWeapon(
                    name="Blue Cannon",
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

        ai_pls = create_agents(
            number=3,
            model=self.model,
            pos=constants.ai_pl_positions,
            name=["Blue AI Pl 1", "Blue AI Pl 2", "Blue AI Pl 3"],
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.SENSOR_FUSION,
            icon_path=os.path.join(constants.IMAGE_DIR, "icons", "friendly_mech_inf.svg"),
            killed_icon_path=os.path.join(
                constants.IMAGE_DIR, "icons", "friendly_mech_inf_dead.svg"
            ),
            move_behaviour=GroundBasedMoveBehaviour(
                base_speed=15.0,
                gradient_speed_modifiers=constants.ground_gradient_speed_modifiers,
                culture_speed_modifiers=constants.ground_culture_movement_modifiers,
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            communicate_worldview_behaviour=CommunicateWorldviewBehaviour(),
            communicate_orders_behaviour=CommunicateOrdersBehaviour(),
        )

        friendly_agents = (
            rocket_arty
            + uavs
            + ai_pls
            + [ai_coy_hq]
            + ai_sects_pl_1
            + ai_sects_pl_2
            + ai_sects_pl_3
        )

        return friendly_agents

    def _set_up_comms(self, agents: list[RetAgent]):
        """Setup comms between friendly agents.

        Args:
            agents (list[RetAgent]): A list of agents.

        - Sects send to owning platoon
        - Platoons send to Coy HQ
        - UAVs send to Coy HQ
        - HQ sends to rockets, uavs and Platoons
        """
        ai_coy_hq = [a for a in agents if a.name == "Blue AI Coy"][0]
        ai_pl_1_sects = [a for a in agents if "Blue AI Sect 1" in a.name]
        ai_pl_2_sects = [a for a in agents if "Blue AI Sect 2" in a.name]
        ai_pl_3_sects = [a for a in agents if "Blue AI Sect 3" in a.name]
        ai_pl_1 = [a for a in agents if a.name == "Blue AI Pl 1"][0]
        ai_pl_2 = [a for a in agents if a.name == "Blue AI Pl 2"][0]
        ai_pl_3 = [a for a in agents if a.name == "Blue AI Pl 3"][0]
        rocket_arty = [a for a in agents if a.name == "Blue Rocket Arty"][0]
        uavs = [a for a in agents if "Blue UAV" in a.name]

        for section in ai_pl_1_sects:
            section.communication_network.add_recipient(ai_pl_1)
        for section in ai_pl_2_sects:
            section.communication_network.add_recipient(ai_pl_2)
        for section in ai_pl_3_sects:
            section.communication_network.add_recipient(ai_pl_3)

        ai_pl_1.communication_network.add_recipient(ai_coy_hq)
        ai_pl_2.communication_network.add_recipient(ai_coy_hq)
        ai_pl_3.communication_network.add_recipient(ai_coy_hq)

        for uav in uavs:
            uav.communication_network.add_recipient(ai_coy_hq)

        ai_coy_hq.communication_network.add_recipient(ai_pl_1)
        ai_coy_hq.communication_network.add_recipient(ai_pl_2)
        ai_coy_hq.communication_network.add_recipient(ai_pl_3)
        ai_coy_hq.communication_network.add_recipient(rocket_arty)
        ai_coy_hq.communication_network.add_recipient(uavs)
