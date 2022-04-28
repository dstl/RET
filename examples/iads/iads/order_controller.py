"""Controller for distributing orders to friendly and hostile agents."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agentfilter import AgentFilter, FilterNot
from mesa_ret.agents.groupagent import GroupAgent
from mesa_ret.behaviours.move import AircraftMoveBehaviour
from mesa_ret.behaviours.sense import SenseBehaviour
from mesa_ret.creator.addorders import add_orders_to_agents
from mesa_ret.formations import SquareFormationRounded
from mesa_ret.orders.order import CompoundTask, GroupAgentOrder, Order
from mesa_ret.orders.tasks.communicate import CommunicateOrderTask, CommunicateWorldviewTask
from mesa_ret.orders.tasks.deploycountermeasure import DeployCountermeasureTask
from mesa_ret.orders.tasks.fire import FireAtTargetTask
from mesa_ret.orders.tasks.move import GroupFormationMoveTask, MoveInBandTask
from mesa_ret.orders.tasks.sense import SenseTask
from mesa_ret.orders.triggers.immediate import ImmediateTrigger
from mesa_ret.orders.triggers.killed import KilledAgentsAtPositionTrigger
from mesa_ret.orders.triggers.position import PositionTrigger
from mesa_ret.orders.triggers.time import TimeTrigger

from . import iads_constants as iads_constants
from .iads_triggers import FriendlyIdentifiedAgentsTrigger, HostileIdentifiedAgentsTrigger

if TYPE_CHECKING:
    from typing import Optional

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.model import RetModel
    from mesa_ret.types import Coordinate


class MoveAndCommunicateBehaviour(AircraftMoveBehaviour):
    """Aircraft movement behaviour that simultaneously communicates worldview."""

    def step(self, mover: RetAgent, destination: Coordinate) -> None:
        """Communicates world view then move steps."""
        mover.communicate_worldview_step()

        super().step(mover, destination)


class SenseAndCommunicateBehaviour(SenseBehaviour):
    """Aircraft sense behaviour that communicates worldview immediately after sense."""

    def step(self, senser: RetAgent, direction: Optional[float]) -> None:
        """Sense steps, then communicates worldview."""
        super().step(senser, direction)

        senser.communicate_worldview_step()

        return


class GroupAgentFilter(AgentFilter):
    """AgentFilter for agent type Group."""

    def run(self, agents: list[RetAgent]) -> list[RetAgent]:
        """Filter list of agents by agent type.

        Args:
            agents (list[RetAgent]): Initial list of agents

        Returns:
            list[RetAgent]: Reduced list of agents
        """
        return [a for a in agents if isinstance(a, GroupAgent)]


class TyphoonOrGroupAgentFilter(AgentFilter):
    """AgentFilter for agent type Typhoon or Group Agent."""

    def run(self, agents: list[RetAgent]) -> list[RetAgent]:
        """Filter list of agents by agent type.

        Args:
            agents (list[RetAgent]): Initial list of agents

        Returns:
            list[RetAgent]: Reduced list of agents
        """
        return [a for a in agents if a.name == "Typhoon" or isinstance(a, GroupAgent)]


class IADSOrderController:
    """Utility class for setting orders once agents have been created."""

    @staticmethod
    def set_friendly_orders(iads_model: RetModel, all_agents: list[RetAgent]) -> None:
        """Set friendly orders.

        - sentinels sense, communicate worldview if hostile targets are found at
            amsterdam center
        - typhoons fire at amsterdam center if hostiles found there
        - c2 agent sends orders to attack at 2:00, return once agent is killed
        """
        sentinels = [
            a for a in all_agents if a.name == "Sentinel" and a.affiliation is Affiliation.FRIENDLY
        ]

        for sentinel in sentinels:
            sentinel_orders = [
                Order(
                    trigger=PositionTrigger(
                        agent=sentinel,
                        position=(360000, 170000),
                        tolerance=iads_constants.pos_tolerance,
                    ),
                    task=SenseTask(duration=timedelta(days=1)),
                    priority=2,
                    persistent=True,
                ),
            ]
            add_orders_to_agents(sentinel, sentinel_orders)

        protectors = [
            a for a in all_agents if a.name == "Protector" and a.affiliation is Affiliation.FRIENDLY
        ]

        for protector in protectors:
            protector_orders = [
                Order(
                    trigger=(HostileIdentifiedAgentsTrigger()),
                    task=DeployCountermeasureTask(),
                ),
            ]
            add_orders_to_agents(protector, protector_orders)

        target = [
            a for a in all_agents if a.name == "Target" and a.affiliation == Affiliation.HOSTILE
        ][0]

        c2 = [
            a
            for a in all_agents
            if a.name == "C2 Coningsby" and a.affiliation == Affiliation.FRIENDLY
        ][0]

        to_amsterdam_orders: list[Order] = [
            Order(
                ImmediateTrigger(),
                CompoundTask(
                    [
                        MoveInBandTask(
                            (200000, 245000, "12000m"),
                            iads_constants.height_bands,
                            iads_model.space,  # type: ignore
                            iads_constants.pos_tolerance,
                        ),
                        MoveInBandTask(
                            (290000, 210000, "2000m"),
                            iads_constants.height_bands,
                            iads_model.space,  # type: ignore
                            iads_constants.pos_tolerance,
                        ),
                        MoveInBandTask(
                            (360000, 170000, "500m"),
                            iads_constants.height_bands,
                            iads_model.space,  # type: ignore
                            iads_constants.pos_tolerance,
                        ),
                    ]
                ),
            ),
            Order(
                KilledAgentsAtPositionTrigger(target.pos),
                CompoundTask(
                    [
                        MoveInBandTask(
                            (290000, 210000, "500m"),
                            iads_constants.height_bands,
                            iads_model.space,  # type: ignore
                            iads_constants.pos_tolerance,
                        ),
                        MoveInBandTask(
                            (30000, 300000, "12000m"),
                            iads_constants.height_bands,
                            iads_model.space,  # type: ignore
                            iads_constants.pos_tolerance,
                        ),
                    ]
                ),
                priority=10,
            ),
        ]

        group_to_amsterdam_orders: list[Order] = [
            GroupAgentOrder(
                ImmediateTrigger(),
                CompoundTask(
                    [
                        GroupFormationMoveTask(
                            MoveInBandTask(
                                (200000, 245000, "12000m"),
                                iads_constants.height_bands,
                                iads_model.space,  # type: ignore
                                iads_constants.pos_tolerance,
                            ),
                            SquareFormationRounded(separation=200),
                            subordinate_move_priority=10,
                        ),
                        GroupFormationMoveTask(
                            MoveInBandTask(
                                (290000, 210000, "2000m"),
                                iads_constants.height_bands,
                                iads_model.space,  # type: ignore
                                iads_constants.pos_tolerance,
                            ),
                            SquareFormationRounded(separation=200, heading=(1, 1)),
                            subordinate_move_priority=10,
                        ),
                        GroupFormationMoveTask(
                            MoveInBandTask(
                                (360000, 170000, "500m"),
                                iads_constants.height_bands,
                                iads_model.space,  # type: ignore
                                iads_constants.pos_tolerance,
                            ),
                            SquareFormationRounded(separation=200, heading=(1, 1)),
                            subordinate_move_priority=10,
                        ),
                    ]
                ),
            ),
            GroupAgentOrder(
                ImmediateTrigger(),
                CompoundTask(
                    [
                        GroupFormationMoveTask(
                            MoveInBandTask(
                                (290000, 210000, "500m"),
                                iads_constants.height_bands,
                                iads_model.space,  # type: ignore
                                iads_constants.pos_tolerance,
                            ),
                            SquareFormationRounded(separation=200),
                            subordinate_move_priority=1,
                            subordinate_move_trigger=KilledAgentsAtPositionTrigger(target.pos),
                        ),
                        GroupFormationMoveTask(
                            MoveInBandTask(
                                (30000, 300000, "12000m"),
                                iads_constants.height_bands,
                                iads_model.space,  # type: ignore
                                iads_constants.pos_tolerance,
                            ),
                            SquareFormationRounded(
                                separation=20, grid_ratio=(1, 0), include_killed=True
                            ),
                            subordinate_move_priority=1,
                            subordinate_move_trigger=KilledAgentsAtPositionTrigger(target.pos),
                        ),
                    ]
                ),
                priority=10,
            ),
        ]

        communicate_orders_to_british_agents_at_0200 = Order(
            trigger=TimeTrigger(time=datetime(2020, 11, 1, 2, 0, 0), sticky=False),
            task=CommunicateOrderTask(
                to_amsterdam_orders, recipient_filter=FilterNot(TyphoonOrGroupAgentFilter())
            ),
        )

        communicate_orders_to_british_group_agents_at_0145 = Order(
            trigger=TimeTrigger(time=datetime(2020, 11, 1, 1, 45, 0), sticky=False),
            task=CommunicateOrderTask(
                group_to_amsterdam_orders, recipient_filter=GroupAgentFilter()
            ),
        )

        add_orders_to_agents(
            c2,
            [
                communicate_orders_to_british_agents_at_0200,
                communicate_orders_to_british_group_agents_at_0145,
            ],
        )

    @staticmethod
    def set_hostile_orders(iads_model: RetModel, all_agents: list[RetAgent]) -> None:
        """Set hostile orders.

        - c2 send orders to deploy countermeasures if unknown agents perceived
        - c2 send orders to fire if hostile agents perceived
        - sensors sense and communicate world view
        """
        hostile_c2 = [
            a
            for a in all_agents
            if a.name == "Amsterdam C2" and a.affiliation == Affiliation.HOSTILE
        ][0]

        hostile_c2_orders = [
            Order(
                trigger=FriendlyIdentifiedAgentsTrigger(),
                task=CommunicateOrderTask(
                    [
                        Order(
                            trigger=ImmediateTrigger(),
                            task=FireAtTargetTask((360000, 170000, 500)),
                        )
                    ]
                ),
            ),
        ]

        add_orders_to_agents(hostile_c2, hostile_c2_orders)

        hostile_sensors = [
            a for a in all_agents if "GM200" in a.name and a.affiliation == Affiliation.HOSTILE
        ]

        hostile_sensor_orders = [
            Order(
                task=SenseTask(duration=timedelta(days=1)),
                trigger=ImmediateTrigger(),
                priority=1,
                persistent=True,
            ),
            Order(
                task=CommunicateWorldviewTask(),
                trigger=FriendlyIdentifiedAgentsTrigger(),
                priority=3,
                persistent=True,
            ),
        ]

        add_orders_to_agents(hostile_sensors, hostile_sensor_orders)
