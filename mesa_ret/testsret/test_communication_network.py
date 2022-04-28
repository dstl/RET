"""Tests for communication network."""
from __future__ import annotations

import unittest
import warnings
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.behaviours.disablecommunication import (
    DisableAllHostileCommsInRangeBehaviour,
    DisableCommunicationBehaviour,
)
from mesa_ret.communication.communicationnetwork import (
    CommunicationNetwork,
    CommunicationNetworkModifier,
)
from mesa_ret.communication.network_setup_utility import network_setup_complete_graph
from mesa_ret.testing.mocks import (
    MockCommunicateMissionMessageBehaviour,
    MockCommunicateOrdersBehaviour,
    MockCommunicateWorldviewBehaviour,
    MockModel2d,
    MockOrder,
)

if TYPE_CHECKING:
    from typing import Optional

    from mesa_ret.orders.order import Order
    from mesa_ret.template import Template


class DisableCommsToRecipient(DisableCommunicationBehaviour):
    """Disable communications behaviour for disabling comms to a predefined agent."""

    def __init__(self, sender: RetAgent, recipient: Optional[list[RetAgent]]):
        """Create a new DisableCommsToRecipient.

        Args:
            sender (RetAgent): Agent who's comms are being disabled
            recipient (RetAgent, None): Agent to whom the communications are being
                relayed. If None, disables communications to all recipients.
        """
        super().__init__()
        self.sender = sender
        self.recipient = recipient

    def identify_agents_to_disable(
        self, disabler: RetAgent
    ) -> list[tuple[RetAgent, Optional[list[RetAgent]]]]:
        """Return a description of the agents to disable.

        Args:
            disabler (RetAgent): Agent doing the disabling

        Returns:
            list[tuple[RetAgent, Union[None, list[RetAgent]]]]: Agents to disable.
        """
        return [(self.sender, self.recipient)]


class TestCommunicationNetwork(unittest.TestCase):
    """Tests for communications network."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()

        self.agent = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Communicator",
            model=self.model,
            behaviours=[
                MockCommunicateOrdersBehaviour(),
                MockCommunicateWorldviewBehaviour(),
                MockCommunicateMissionMessageBehaviour(),
            ],
        )

        self.agent_no_comms = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(18, 18),
            name="Communicator",
            model=self.model,
        )

        self.agent.communication_network.add_recipient(self.agent_no_comms)

        self.agent_not_in_network = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Communicator",
            model=self.model,
        )

        self.disabler = RetAgent(
            affiliation=Affiliation.HOSTILE,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(20, 20),
            name="Disabler",
            model=self.model,
            behaviours=[
                DisableCommsToRecipient(self.agent, [self.agent_no_comms]),
            ],
        )

    def test_initialise_network(self):
        """Test network initialisation."""
        with warnings.catch_warnings(record=True) as w:
            self.agent_no_comms.communicate_orders_step(orders=[])
            assert len(w) == 1
            assert "Communicate Orders behaviour requested for agent" in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            self.agent_no_comms.communicate_worldview_step()
            assert len(w) == 1
            assert "Communicate Worldview behaviour requested for agent" in str(w[0].message)

    def test_orders_over_network(self):
        """Test communication of orders over the network."""
        self.agent.communicate_orders_step([MockOrder()])

        assert len(self.agent_no_comms._orders) == 1
        assert len(self.agent_not_in_network._orders) == 0

    def test_perceived_world_over_network(self):
        """Test communication of perceived world over the network."""
        self.agent.communicate_worldview_step()

        assert len(self.agent_no_comms.perceived_world.get_perceived_agents()) == 2
        assert len(self.agent_not_in_network.perceived_world.get_perceived_agents()) == 1

    def test_mission_message_over_network(self):
        """Test communication of mission message over the network."""
        self.agent.communicate_mission_message_step("test")

        assert len(self.agent_no_comms.mission_messages) == 1
        assert "test" in self.agent_no_comms.mission_messages
        assert len(self.agent_not_in_network.mission_messages) == 0

    def test_disabled_comms_dont_receive_worldview(self):
        """Test that with comms are disabled, recipients don't receive new worldview."""
        self.disabler.disable_communication_step()
        self.agent.communicate_worldview_step()
        assert len(self.agent_no_comms.perceived_world.get_perceived_agents()) == 1

    def test_disabled_comms_dont_receive_orders(self):
        """Test that with comms are disabled, recipients don't receive new orders."""
        self.disabler.disable_communication_step()
        orders: list[Template[Order]] = [MockOrder()]
        self.agent.communicate_orders_step(orders)
        assert len(self.agent_no_comms._orders) == 0

    def test_disabled_comms_dont_receive_mission_message(self):
        """Test that with comms are disabled, recipients don't receive new mission message."""
        self.disabler.disable_communication_step()
        self.agent.communicate_mission_message_step("test")
        assert len(self.agent_no_comms.mission_messages) == 0

    def test_network_modifier_blocking_comms_to_specific_agent(self):
        """Test that network modifiers can restrict info to specific agents."""
        network = CommunicationNetwork()
        network.add_recipient([self.agent, self.agent_no_comms])

        network.add_modifier(CommunicationNetworkModifier(target=self.agent))
        assert network.get_recipients() == [self.agent_no_comms]

        network.add_modifier(CommunicationNetworkModifier(target=self.agent_no_comms))
        assert network.get_recipients() == []

    def test_network_modifier_blocking_all_comms(self):
        """Test that network modifiers can restrict info to all agents."""
        network = CommunicationNetwork()
        network.add_recipient(self.agent)
        network.add_recipient(self.agent_no_comms)

        network.add_modifier(CommunicationNetworkModifier())
        assert network.get_recipients() == []

    def test_disable_all_hostile_comms_in_range(self):
        """Test the capability of DisableAllHostileCommsInRangeBehaviour."""
        network_modifier = DisableAllHostileCommsInRangeBehaviour(range=10)
        within_10 = network_modifier.identify_agents_to_disable(self.agent_no_comms)

        assert within_10 == [(self.disabler, None)]

        network_modifier = DisableAllHostileCommsInRangeBehaviour(range=1)
        within_1 = network_modifier.identify_agents_to_disable(self.agent_no_comms)

        # With shorter range, "disabler" is no longer within range
        assert within_1 == []

    def test_disable_hostile_ignores_friendly(self):
        """Test that DisableAllHostileCommsInRangeBehaviour ignore friendly agents."""
        network_modifier = DisableAllHostileCommsInRangeBehaviour(range=10)
        within_10 = network_modifier.identify_agents_to_disable(self.agent)

        # agent_without_comms_behaviour_not_in_network is skipped despite being at the
        # same position as agent, and agent doesn't identify it's self
        assert within_10 == []

    def add_recipient(self):
        """Test adding a recipient to a communication network."""
        network = CommunicationNetwork()
        network.add_recipient(self.agent)
        assert self.agent in network._recipients

    def test_remove_recipient(self):
        """Test removing a recipient from a communication network."""
        network = CommunicationNetwork()
        network.add_recipient(self.agent)
        assert self.agent in network._recipients
        network.remove_recipient(self.agent)
        assert self.agent not in network._recipients

    def test_add_recipients(self):
        """Test adding recipients to communication network."""
        network = CommunicationNetwork()
        network.add_recipient([self.agent, self.agent_no_comms])
        assert self.agent_no_comms in network._recipients
        assert self.agent in network._recipients

    def test_remove_recipients(self):
        """Test removing recipients from a communication network."""
        network = CommunicationNetwork()
        network.add_recipient([self.agent, self.agent_no_comms, self.agent_not_in_network])
        assert self.agent_no_comms in network._recipients
        assert self.agent in network._recipients
        assert self.agent_not_in_network in network._recipients
        network.remove_recipient([self.agent_not_in_network, self.agent_no_comms])
        assert self.agent_no_comms not in network._recipients
        assert self.agent_not_in_network not in network._recipients
        assert self.agent in network._recipients

    def test_add_duplicate_agent(self):
        """Test that a warning is raised where an agent is added twice to a network."""
        network = CommunicationNetwork()
        network.add_recipient(self.agent)

        with warnings.catch_warnings(record=True) as w:
            network.add_recipient(self.agent)
            assert len(w) == 1
            assert "is already in the communication network" in str(w[0].message)

    def test_remove_agent_not_in_network(self):
        """Test that a warning is raised if removing an agent not in a network."""
        network = CommunicationNetwork()

        with warnings.catch_warnings(record=True) as w:
            network.remove_recipient(self.agent)
            assert len(w) == 1
            assert "is not in this communication network" in str(w[0].message)

    def test_add_duplicate_agents(self):
        """Test that a warning is raised where an agent is added twice to a network.

        Testing that the warning is only raised for the duplicate agents where there
        are multiple agents added in a single action.
        """
        network = CommunicationNetwork()
        network.add_recipient([self.agent])

        with warnings.catch_warnings(record=True) as w:
            network.add_recipient([self.agent, self.agent_no_comms])
            assert len(w) == 1
            assert "is already in the communication network" in str(w[0].message)

        assert self.agent_no_comms in network._recipients

    def test_remove_agents_not_in_network(self):
        """Test that a warning is raised if removing an agent not in a network.

        Testing that the warning is only raised for the missing agents where there are
        multiple agents removed in a single action
        """
        network = CommunicationNetwork()
        network.add_recipient([self.agent])
        with warnings.catch_warnings(record=True) as w:
            network.remove_recipient([self.agent, self.agent_no_comms])
            assert len(w) == 1
            assert "is not in this communication network" in str(w[0].message)

        assert self.agent not in network._recipients

    def test_network_setup_utiltity(self):
        """Test the network setup utility connects all agents to one another's network."""
        self.agent2 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Communicator2",
            model=self.model,
            behaviours=[
                MockCommunicateOrdersBehaviour(),
                MockCommunicateWorldviewBehaviour(),
                MockCommunicateMissionMessageBehaviour(),
            ],
        )
        self.agent3 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Communicator3",
            model=self.model,
            behaviours=[
                MockCommunicateOrdersBehaviour(),
                MockCommunicateWorldviewBehaviour(),
                MockCommunicateMissionMessageBehaviour(),
            ],
        )
        self.agent.communication_network._recipients = []
        network_setup_complete_graph([self.agent, self.agent2, self.agent3])

        assert len(self.agent.communication_network.get_recipients()) == 2
        assert len(self.agent2.communication_network.get_recipients()) == 2
        assert len(self.agent3.communication_network.get_recipients()) == 2

        assert self.agent2 in self.agent.communication_network._recipients
        assert self.agent3 in self.agent.communication_network._recipients

        assert self.agent in self.agent2.communication_network._recipients
        assert self.agent3 in self.agent2.communication_network._recipients

        assert self.agent in self.agent3.communication_network._recipients
        assert self.agent2 in self.agent3.communication_network._recipients
