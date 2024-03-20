"""Utilities to generate friendly agents for IADS scenario.

Unit icons created using: <https://spatialillusions.com/>
"""
from __future__ import annotations

from datetime import timedelta

from ret.agents.agent import Affiliation, RetAgent
from ret.agents.agenttype import AgentType
from ret.agents.groupagent import GroupAgent
from ret.behaviours.communicate import (
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.move import AircraftMoveBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.creator.agents import create_agents
from ret.orders.order import Order
from ret.orders.tasks.fire import FireAtTargetTask
from ret.sensing.distribution import TriangularDistribution
from ret.sensing.sensor import (
    DistanceAttenuatedSensor,
    SensorClutterAttenuator,
    SensorDetectionTimings,
    SensorDistanceThresholds,
)
from ret.space.clutter.countermeasure import SphereCountermeasure
from ret.weapons.weapon import BasicWeapon

from . import iads_constants as iads_constants
from .iads_agent_creator import IADSAgentCreator
from .iads_triggers import HostileIdentifiedAgentsTrigger
from .order_controller import MoveAndCommunicateBehaviour, SenseAndCommunicateBehaviour


class FriendlyAgentCreator(IADSAgentCreator):
    """Utility class for generating friendly agents and setting up comms."""

    def _make_agents(self) -> list[RetAgent]:
        """Make Friendly agents.

        - 1 sentinel (sensors)
        - 4 protectors (countermeasures)
        - 3 voyagers
        - 24 typhoons (weapons)
        - 1 group agent containing the 24 typhoons
        - 1 c2 agent

        Returns:
            list(RetAgents): List of friendly agents
        """
        sentinels = create_agents(
            number=1,
            model=self.model,
            pos=iads_constants.pos_coningsby,
            name="Sentinel",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            move_behaviour=MoveAndCommunicateBehaviour(
                base_speed=160, height_bands=iads_constants.height_bands
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            communicate_worldview_behaviour=CommunicateWorldviewBehaviour(),
            sense_behaviour=SenseAndCommunicateBehaviour(
                time_before_first_sense=timedelta(0), time_between_senses=timedelta(minutes=1)
            ),
            sensors=[
                DistanceAttenuatedSensor(
                    detection_timings=SensorDetectionTimings(
                        TriangularDistribution(lower_limit=0, mode=5, upper_limit=10)
                    ),
                    distance_thresholds=SensorDistanceThresholds(
                        max_detect_dist=150000,
                        max_recognise_dist=75000,
                        max_identify_dist=50000,
                    ),
                    clutter_attenuator=SensorClutterAttenuator(
                        attenuation_strength=20,
                    ),
                ),
                DistanceAttenuatedSensor(
                    detection_timings=SensorDetectionTimings(
                        TriangularDistribution(lower_limit=0, mode=5, upper_limit=10)
                    ),
                    distance_thresholds=SensorDistanceThresholds(
                        max_detect_dist=150000,
                        max_recognise_dist=75000,
                        max_identify_dist=50000,
                    ),
                    clutter_attenuator=SensorClutterAttenuator(
                        attenuation_strength=10,
                    ),
                ),
            ],
        )

        protectors = create_agents(
            number=4,
            model=self.model,
            pos=iads_constants.pos_coningsby,
            name="Protector",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            move_behaviour=AircraftMoveBehaviour(
                base_speed=160, height_bands=iads_constants.height_bands
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            countermeasures=[
                SphereCountermeasure(
                    clutter_value=20,
                    clutter_radius=1000,
                    life_time=timedelta(minutes=1),
                )
            ],
            deploy_countermeasure_behaviour=DeployCountermeasureBehaviour(),
        )

        voyagers = create_agents(
            number=3,
            model=self.model,
            pos=iads_constants.pos_coningsby,
            name="Voyager",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            move_behaviour=AircraftMoveBehaviour(
                base_speed=160, height_bands=iads_constants.height_bands
            ),
        )

        typhoons = create_agents(
            number=24,
            model=self.model,
            pos=iads_constants.pos_coningsby,
            name="Typhoon",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            move_behaviour=AircraftMoveBehaviour(
                base_speed=160, height_bands=iads_constants.height_bands
            ),
            wait_behaviour=WaitBehaviour(),
            hide_behaviour=HideBehaviour(),
            fire_behaviour=FireBehaviour(),
            orders=[
                Order(
                    trigger=HostileIdentifiedAgentsTrigger(),
                    task=FireAtTargetTask(iads_constants.pos_amsterdam_center),
                    priority=20,
                )
            ],
            weapons=[
                BasicWeapon(
                    "Typhoon Weapon",
                    radius=3000,
                    time_before_first_shot=timedelta(seconds=0),
                    time_between_rounds=timedelta(seconds=30),
                    kill_probability_per_round=0.5,
                )
            ],
        )

        typhoon_group = GroupAgent(
            model=self.model,
            name="Typhoon Group",
            affiliation=Affiliation.FRIENDLY,
            communicate_orders_behaviour=CommunicateOrdersBehaviour(),
        )

        typhoon_group.add_agents(typhoons)

        c2_agent = RetAgent(
            model=self.model,
            name="C2 Coningsby",
            pos=iads_constants.pos_coningsby,
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
            behaviours=[WaitBehaviour(), CommunicateOrdersBehaviour()],
        )

        friendly_agents: list[RetAgent] = (
            sentinels + protectors + voyagers + [typhoon_group] + [c2_agent]
        )

        return friendly_agents

    def _set_up_comms(self, agents: list[RetAgent]) -> None:
        """Set up friendly comms network.

        - C2 sends to sentinels, protectors, typhoons and voyagers
        - Sentinels sent to typhoons

        Args:
            agents (list[RetAgent]): Agents to set up communications between
        """
        c2_agent = [a for a in agents if a.name == "C2 Coningsby"][0]
        sentinels = [a for a in agents if a.name == "Sentinel"]
        typhoon_group = [a for a in agents if a.name == "Typhoon Group"][0]
        protectors = [a for a in agents if a.name == "Protector"]
        voyagers = [a for a in agents if a.name == "Voyager"]

        c2_agent.communication_network.add_recipient(sentinels)
        c2_agent.communication_network.add_recipient(protectors)
        c2_agent.communication_network.add_recipient(typhoon_group)
        c2_agent.communication_network.add_recipient(voyagers)

        for sentinel in sentinels:
            sentinel.communication_network.add_recipient(protectors)
            sentinel.communication_network.add_recipient(typhoon_group)
            sentinel.communication_network.add_recipient(voyagers)
