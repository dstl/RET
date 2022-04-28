"""Utilities to generate hostile agents for IADS scenario."""
from __future__ import annotations

from datetime import timedelta
from math import inf

from mesa_ret.agents.agent import Affiliation, RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.agents.protectedassetagent import ProtectedAssetAgent
from mesa_ret.agents.sensorfusionagent import SensorFusionAgent
from mesa_ret.behaviours.communicate import (
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from mesa_ret.behaviours.fire import FireBehaviour
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import GroundBasedMoveBehaviour
from mesa_ret.behaviours.sense import SenseBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.creator.agents import create_agents
from mesa_ret.sensing.sensor import (
    LineOfSightSensor,
    SensorDistanceThresholds,
    SensorSamplingDistance,
)
from mesa_ret.weapons.weapon import BasicWeapon

from . import iads_constants as iads_constants
from .iads_agent_creator import IADSAgentCreator


class HostileAgentCreator(IADSAgentCreator):
    """Utility class for generating hostile agents and setting up comms."""

    def _make_agents(self) -> list[RetAgent]:
        """Make hostile agents.

        - 12 batteries (weapons)
        - 1 c2 agent
        - 6 sensors
        - 1 target (protected asset)
        - 1 sensor fusion

        Returns:
            list(RetAgents): List of hostile agents
        """
        hostile_battery_agents = create_agents(
            number=12,
            model=self.model,
            pos=[
                iads_constants.pos_amsterdam_north,
                iads_constants.pos_amsterdam_east,
                iads_constants.pos_amsterdam_south,
                iads_constants.pos_amsterdam_west,
                iads_constants.pos_amsterdam_north,
                iads_constants.pos_amsterdam_north,
                iads_constants.pos_amsterdam_east,
                iads_constants.pos_amsterdam_east,
                iads_constants.pos_amsterdam_south,
                iads_constants.pos_amsterdam_south,
                iads_constants.pos_amsterdam_west,
                iads_constants.pos_amsterdam_west,
            ],
            name=[
                "PAC 3 Battery (N)",
                "PAC 3 Battery (E)",
                "PAC 3 Battery (S)",
                "PAC 3 Battery (W)",
                "NASAMS 2 Battery (N)",
                "NASAMS 2 Battery (N)",
                "NASAMS 2 Battery (E)",
                "NASAMS 2 Battery (E)",
                "NASAMS 2 Battery (S)",
                "NASAMS 2 Battery (S)",
                "NASAMS 2 Battery (W)",
                "NASAMS 2 Battery (W)",
            ],
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.AIR_DEFENCE,
            icon_path=[
                iads_constants.patriot_icon_path,
                iads_constants.patriot_icon_path,
                iads_constants.patriot_icon_path,
                iads_constants.patriot_icon_path,
                iads_constants.nasams_icon_path,
                iads_constants.nasams_icon_path,
                iads_constants.nasams_icon_path,
                iads_constants.nasams_icon_path,
                iads_constants.nasams_icon_path,
                iads_constants.nasams_icon_path,
                iads_constants.nasams_icon_path,
                iads_constants.nasams_icon_path,
            ],
            # killed_icon_path=[],
            fire_behaviour=FireBehaviour(),
        )

        hostile_c2 = RetAgent(
            model=self.model,
            pos=iads_constants.pos_amsterdam_sensor_north,
            name="Amsterdam C2",
            affiliation=Affiliation.HOSTILE,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[CommunicateOrdersBehaviour()],
            weapons=[
                BasicWeapon(
                    name="C2 Weapon",
                    radius=10,
                    time_before_first_shot=timedelta(seconds=0),
                    time_between_rounds=timedelta(seconds=5),
                    kill_probability_per_round=0.05,
                )
            ],
        )

        hostile_target = ProtectedAssetAgent(
            model=self.model,
            name="Target",
            pos=iads_constants.pos_amsterdam_center,
            affiliation=Affiliation.HOSTILE,
        )

        hostile_sensor_agents = create_agents(
            number=6,
            model=self.model,
            pos=[
                iads_constants.pos_amsterdam_sensor_north,
                iads_constants.pos_amsterdam_sensor_north,
                iads_constants.pos_amsterdam_sensor_south,
                iads_constants.pos_amsterdam_sensor_south,
                iads_constants.pos_amsterdam_sensor_west,
                iads_constants.pos_amsterdam_sensor_west,
            ],
            name=[
                "GM200 MM/C (N)",
                "GM200 MM/C (N)",
                "GM200 MM/C (S)",
                "GM200 MM/C (S)",
                "GM200 MM/C (W)",
                "GM200 MM/C (W)",
            ],
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.AIR_DEFENCE,
            icon_path=iads_constants.radar_icon_path,
            # killed_icon_path="",
            move_behaviour=GroundBasedMoveBehaviour(
                base_speed=0.015,
                gradient_speed_modifiers=[
                    ((-inf, -1.1), 0.8),
                    ((-1.1, 1.1), 1),
                    ((1.1, inf), 0.8),
                ],
                culture_speed_modifiers={
                    iads_constants.culture_land: 1.0,
                    iads_constants.culture_sea: 1.0,
                },  # type: ignore
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            sense_behaviour=SenseBehaviour(
                time_before_first_sense=timedelta(seconds=0),
                time_between_senses=timedelta(seconds=5),
            ),
            communicate_orders_behaviour=CommunicateOrdersBehaviour(),
            communicate_worldview_behaviour=CommunicateWorldviewBehaviour(),
            sensors=[
                LineOfSightSensor(
                    distance_thresholds=SensorDistanceThresholds(
                        max_detect_dist=200000,
                        max_recognise_dist=150000,
                        max_identify_dist=75000,
                    ),
                    sampling_distance=SensorSamplingDistance(sampling_distance=25000),
                )
            ],
        )

        hostile_sensor_relay = SensorFusionAgent(
            model=self.model,
            pos=iads_constants.pos_amsterdam_sensor_south,
            name="Hostile Sensor Fusion",
            affiliation=Affiliation.HOSTILE,
            behaviours=[CommunicateWorldviewBehaviour()],
        )

        agents: list[RetAgent] = (
            hostile_battery_agents
            + [hostile_c2]
            + [hostile_target]
            + hostile_sensor_agents
            + [hostile_sensor_relay]
        )

        return agents

    def _set_up_comms(self, agents: list[RetAgent]) -> None:
        """Set up hostile comms network.

        - Sensors send to fusion
        - Fusion sends to C2
        - C2 sends to defence

        Args:
            agents (list[RetAgent]): Agents to set up communications between
        """
        fusion = [a for a in agents if "Fusion" in a.name][0]
        c2 = [a for a in agents if "Amsterdam C2" in a.name][0]
        sensors = [a for a in agents if "GM200" in a.name]
        defence = [a for a in agents if "Battery" in a.name]

        for s in sensors:
            s.communication_network.add_recipient(fusion)

        fusion.communication_network.add_recipient(c2)

        c2.communication_network.add_recipient(defence)
