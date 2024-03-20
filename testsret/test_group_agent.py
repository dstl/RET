"""Tests for GroupAgent."""
from __future__ import annotations

import pathlib
import unittest
import warnings
from datetime import timedelta
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest import TestCase

from mesa.time import BaseScheduler, RandomActivation, StagedActivation
from ret.agents.affiliation import Affiliation
from ret.agents.agent import AgentConstructionError, RetAgent
from ret.agents.agenttype import AgentType
from ret.agents.groupagent import GroupAgent
from ret.behaviours.behaviourpool import AlwaysAdder
from ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.disablecommunication import (
    DisableAllHostileCommsInRangeBehaviour,
    DisableCommunicationBehaviour,
)
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.move import AircraftMoveBehaviour, GroundBasedMoveBehaviour, MoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import Confidence, PerceivedAgent
from ret.testing.mocks import MockGroupAgentOrderWithId, MockModel2d, MockOrderWithId
from parameterized import parameterized

if TYPE_CHECKING:
    from ret.behaviours import Behaviour
    from ret.orders.order import Order
    from ret.template import Template


class GroupAgentInitialisationTest(TestCase):
    """Tests for initialisation of Group Agent."""

    def setUp(self):
        """Set up test model."""
        self.model = MockModel2d()

    @parameterized.expand(
        [
            [WaitBehaviour, 0],
            [MoveBehaviour, 0],
            [GroundBasedMoveBehaviour, 0],
            [FireBehaviour, 0],
            [DeployCountermeasureBehaviour, 0],
            [CommunicateWorldviewBehaviour, 0],
            [CommunicateMissionMessageBehaviour, 0],
            [CommunicateOrdersBehaviour, 0],
            [DisableCommunicationBehaviour, 0],
            [SenseBehaviour, 0],
        ]
    )
    def test_init_no_behaviour(self, behaviour_type: type, expected: int):
        """Test initialising Group Agent with no user-defined behaviour.

        Args:
            behaviour_type (type): Behaviour type to add
            expected (int): Expected number of behaviours of the behaviour type
        """
        agent = GroupAgent(self.model, "Group Agent Under Test", Affiliation.FRIENDLY)

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert len(behaviours) == expected

    def test_init_with_custom_behaviour(self):
        """Test GroupAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours
        """
        new_behaviour = CommunicateOrdersBehaviour()
        agent = GroupAgent(
            self.model,
            "Group Agent under test",
            Affiliation.FRIENDLY,
            communicate_orders_behaviour=new_behaviour,
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", CommunicateOrdersBehaviour)
        assert behaviours == [new_behaviour]

    @parameterized.expand(
        [
            [WaitBehaviour, WaitBehaviour()],
            [AircraftMoveBehaviour, AircraftMoveBehaviour(0, [])],
            [FireBehaviour, FireBehaviour()],
            [
                DisableCommunicationBehaviour,
                DisableAllHostileCommsInRangeBehaviour(range=1.0),
            ],
            [
                SenseBehaviour,
                SenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                ),
            ],
        ]
    )
    def test_adding_behaviours_to_existing_agent(
        self, behaviour_type: type, new_behaviour: Behaviour
    ):
        """Test GroupAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours

        Args:
            behaviour_type (type): Behaviour type to add
            new_behaviour (Behaviour): New behaviour to add
        """
        communicate_orders_behaviour = CommunicateOrdersBehaviour()
        agent = GroupAgent(
            self.model,
            "Group Agent Under Test",
            Affiliation.FRIENDLY,
            behaviour_adder=AlwaysAdder,
            communicate_orders_behaviour=communicate_orders_behaviour,
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == []

        agent.behaviour_pool.add_behaviour(new_behaviour)

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour]


class TestGroupAgent(unittest.TestCase):
    """Tests for group agent."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()

        self.agent1 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Subordinate 1",
            model=self.model,
        )

        self.agent2 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(2, 2),
            name="Subordinate 1",
            model=self.model,
        )

        self.group_agent = GroupAgent(
            affiliation=Affiliation.FRIENDLY,
            name="Group",
            model=self.model,
            communicate_orders_behaviour=CommunicateOrdersBehaviour(),
        )

    def test_add_agent_to_group(self):
        """Test that agents can be added to a group agent."""
        self.group_agent.add_agents(self.agent1)

        assert self.agent1 not in self.model.schedule.agents

        assert self.agent1 in self.group_agent.agents
        assert self.agent1 in self.group_agent.schedule.agents
        assert self.agent1 in self.group_agent.communication_network._recipients

        self.group_agent.check_killed()
        self.assertFalse(self.group_agent.killed)

    def test_remove_agent_from_group(self):
        """Test that agents can be removed from a group agent."""
        self.group_agent.add_agents(self.agent1)

        self.group_agent.remove_agents(self.agent1)
        assert self.agent1 in self.model.schedule.agents

        assert self.agent1 not in self.group_agent.agents
        assert self.agent1 not in self.group_agent.schedule.agents
        assert self.agent1 not in self.group_agent.communication_network._recipients

    def test_groupagent_killed_when_empty(self):
        """Test that where empty, a group agent is considered to be killed."""
        self.group_agent.check_killed()

        self.assertTrue(self.group_agent.killed)

    def test_groupagent_killed_when_killed(self):
        """Test that where all agents are killed, a group agent is killed."""
        self.group_agent.add_agents(self.agent1)
        self.agent1.kill()

        self.group_agent.check_killed()

        self.assertTrue(self.group_agent.killed)

    def test_groupagent_revives(self):
        """Test that a groupagent revives."""
        self.group_agent.step()
        self.assertTrue(self.group_agent.killed)

        self.group_agent.add_agents(self.agent1)
        self.assertFalse(self.group_agent.killed)

    def test_groupagent_base_scheduler(self):
        """Test that group agent uses BaseScheduler if model uses it."""
        self.group_agent.create_scheduler()

        assert type(self.group_agent.schedule) is BaseScheduler

    def test_groupagent_random_scheduler(self):
        """Test that group agent user RandomActivation scheduler if model uses it."""
        self.model.schedule = RandomActivation(self.model)
        self.group_agent.create_scheduler()

        assert type(self.group_agent.schedule) is RandomActivation

    def test_add_remove_multiple_agents(self):
        """Test adding and removing multiple agents."""
        new_groupagent = GroupAgent(
            affiliation=Affiliation.FRIENDLY,
            name="Group",
            model=self.model,
            communicate_orders_behaviour=CommunicateOrdersBehaviour(),
            agents=[self.agent1, self.agent2],
        )

        assert self.agent1 not in self.model.schedule.agents
        assert self.agent2 not in self.model.schedule.agents

        assert self.agent1 in new_groupagent.agents
        assert self.agent2 in new_groupagent.agents

        assert self.agent1 in new_groupagent.schedule.agents
        assert self.agent2 in new_groupagent.schedule.agents
        assert self.agent1 in new_groupagent.communication_network._recipients
        assert self.agent2 in new_groupagent.communication_network._recipients

        new_groupagent.remove_agents([self.agent1, self.agent2])
        assert self.agent1 in self.model.schedule.agents
        assert self.agent2 in self.model.schedule.agents

        assert self.agent1 not in new_groupagent.agents
        assert self.agent2 not in new_groupagent.agents

        assert self.agent1 not in new_groupagent.schedule.agents
        assert self.agent2 not in new_groupagent.schedule.agents

        assert self.agent1 not in new_groupagent.communication_network._recipients
        assert self.agent2 not in new_groupagent.communication_network._recipients

    def test_disseminate_orders(self):
        """Test dissemination of orders."""
        communicator = RetAgent(
            model=self.model,
            pos=(0, 0),
            agent_type=AgentType.GENERIC,
            name="Communicator",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[CommunicateOrdersBehaviour()],
        )

        self.group_agent.add_agents([self.agent1, self.agent2])

        orders: list[Template[Order]] = [MockOrderWithId(id=10), MockOrderWithId(id=20)]

        communicator.communication_network.add_recipient(self.group_agent)
        communicator.communicate_orders_step(orders)

        a1_order_ids = set([o.id for o in self.agent1._orders])
        a2_order_ids = set([o.id for o in self.agent2._orders])

        assert a1_order_ids == set([10, 20])
        assert a2_order_ids == set([10, 20])

    def test_group_orders(self):
        """Test group orders are sent to the group agent."""
        communicator = RetAgent(
            model=self.model,
            pos=(0, 0),
            agent_type=AgentType.GENERIC,
            name="Communicator",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[CommunicateOrdersBehaviour()],
        )

        self.group_agent.add_agents([self.agent1, self.agent2])

        orders: list[Template[Order]] = [MockGroupAgentOrderWithId(id=10), MockOrderWithId(id=20)]

        communicator.communication_network.add_recipient(self.group_agent)
        communicator.communicate_orders_step(orders)

        a1_order_ids = set([o.id for o in self.agent1._orders])
        a2_order_ids = set([o.id for o in self.agent2._orders])

        group_order_ids = set([o.id for o in self.group_agent._orders])

        assert a1_order_ids == set([20])
        assert a2_order_ids == set([20])
        assert group_order_ids == set([10])

    def test_disseminate_worldview(self):
        """Test dissemination of world view."""
        communicator = RetAgent(
            model=self.model,
            pos=(0, 0),
            agent_type=AgentType.GENERIC,
            name="Communicator",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[CommunicateWorldviewBehaviour()],
        )

        self.group_agent.add_agents([self.agent1, self.agent2])

        communicator.communication_network.add_recipient(self.group_agent)

        views = [
            PerceivedAgent(
                sense_time=self.model.get_time(),
                confidence=Confidence.KNOWN,
                unique_id=10,
                affiliation=Affiliation.UNKNOWN,
                agent_type=AgentType.UNKNOWN,
                location=(10, 10),
                casualty_state=AgentCasualtyState.ALIVE,
            ),
            PerceivedAgent(
                sense_time=self.model.get_time(),
                confidence=Confidence.KNOWN,
                unique_id=20,
                affiliation=Affiliation.UNKNOWN,
                agent_type=AgentType.UNKNOWN,
                location=(20, 20),
                casualty_state=AgentCasualtyState.ALIVE,
            ),
        ]

        communicator.perceived_world.add_acquisitions(views)
        communicator.communicate_worldview_step()

        a1_views = self.agent1.perceived_world.get_perceived_agents()
        a2_views = self.agent2.perceived_world.get_perceived_agents()

        assert all(view in a1_views for view in views)
        assert all(view in a2_views for view in views)

    def test_disseminate_mission_message(self):
        """Test dissemination of mission message."""
        communicator = RetAgent(
            model=self.model,
            pos=(0, 0),
            agent_type=AgentType.GENERIC,
            name="Communicator",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[CommunicateMissionMessageBehaviour()],
        )

        self.group_agent.add_agents([self.agent1, self.agent2])

        communicator.communication_network.add_recipient(self.group_agent)

        communicator.communicate_mission_message_step("test")

        assert len(self.agent1.mission_messages) == 1
        assert "test" in self.agent1.mission_messages
        assert len(self.agent2.mission_messages) == 1
        assert "test" in self.agent2.mission_messages

    def test_find_pos(self):
        """Test the group agent position."""
        self.group_agent.add_agents([self.agent1, self.agent2])

        assert self.group_agent.pos == (1, 1)

    def test_wrong_scheduler(self):
        """Test groupagent behaviour when scheduler is non-standard."""
        self.model.schedule = StagedActivation(self.model)
        warning_string = (
            "Unable to determine scheduler type for Group Agent Group. " + "Using BaseScheduler"
        )
        with warnings.catch_warnings(record=True) as w:
            new_groupagent = GroupAgent(
                affiliation=Affiliation.FRIENDLY,
                name="Group",
                model=self.model,
                communicate_orders_behaviour=CommunicateOrdersBehaviour(),
            )
            assert len(w) == 1
            assert warning_string in str(w[0].message)

        assert isinstance(new_groupagent.schedule, BaseScheduler)

    def test_specify_icon(self):
        """Test that the groupagent can take a specified icon different to its agents' icons."""
        with TemporaryDirectory() as td:
            p = pathlib.Path(td, "test_icon.svg")
            icon_path = pathlib.Path(p)
            icon_path.write_text("icon_info")

            p_killed = pathlib.Path(td, "test_killed_icon.svg")
            icon_path_killed = pathlib.Path(p_killed)
            icon_path_killed.write_text("killed_icon_info")

            new_groupagent = GroupAgent(
                affiliation=Affiliation.FRIENDLY,
                name="Group",
                model=self.model,
                communicate_orders_behaviour=CommunicateOrdersBehaviour(),
                agents=[self.agent1, self.agent2],
                icon_path=str(p),
                killed_icon_path=str(p_killed),
            )

            assert new_groupagent.icon == "icon_info"
            new_groupagent.kill()
            assert new_groupagent.icon == "killed_icon_info"

    def test_inherit_icon(self):
        """Test that the groupagent inherits the icon of the first agent."""
        assert self.group_agent.icon == self.agent1.icon

        agent3 = RetAgent(
            affiliation=Affiliation.HOSTILE,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(2, 2),
            name="Agent with different icon",
            model=self.model,
        )

        new_groupagent = GroupAgent(
            affiliation=Affiliation.FRIENDLY,
            name="Group",
            model=self.model,
            communicate_orders_behaviour=CommunicateOrdersBehaviour(),
            agents=[agent3, self.agent2],
        )

        assert new_groupagent.icon == agent3.icon

        new_groupagent.add_agents([self.agent1])
        new_groupagent._get_icons()

        assert new_groupagent.icon == self.agent1.icon

    def test_unused_conditions(self):
        """Test the unused conditions in the GroupAgent class."""
        self.agent1.in_group = True
        self.group_agent.add_agents(self.agent1)
        assert len(self.group_agent.agents) == 0

        self.group_agent.remove_agents(self.agent1)
        assert self.agent1.in_group is True

        self.agent2.kill()
        self.group_agent.add_agents(self.agent2)

    def test_get_reflectivity(self):
        """Test exception is thrown when trying to access reflectivity property."""
        with self.assertRaises(AgentConstructionError) as e:
            self.group_agent.reflectivity
        assert (
            f"Reflectivity must be defined for {self.group_agent.name} for this model to be valid"
            in e.exception.args[0]
        )

    def test_get_temperature(self):
        """Test exception is thrown when trying to access temperature property."""
        with self.assertRaises(AgentConstructionError) as e:
            self.group_agent.temperature
        assert (
            f"Temperature must be defined for {self.group_agent.name} for this model to be valid"
            in e.exception.args[0]
        )
