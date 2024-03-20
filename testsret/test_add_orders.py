"""Tests for addorders utility script."""
import unittest
from datetime import datetime, timedelta

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.creator.addorders import (
    add_background_orders_to_agents,
    add_background_orders_to_single_agent,
    add_orders_to_agents,
    add_orders_to_single_agent,
)
from ret.testing.mocks import MockBackgroundOrderWithId, MockModel2d, MockOrderWithId


class TestAddOrdersToSingleAgent(unittest.TestCase):
    """Tests for adding a set of orders to a single agent."""

    def setUp(self):
        """Set up test cases."""
        self.start_time = datetime(2020, 1, 1, 0, 0)
        self.time_step = timedelta(hours=1)
        self.end_time = datetime(2020, 1, 1, 5, 0)

        self.model = MockModel2d(self.start_time, self.time_step, self.end_time)
        self.agent1 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Test Agent",
            model=self.model,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.mock_order = MockOrderWithId(17022021)
        self.mock_background_order = MockBackgroundOrderWithId(24032022)

    def test_add_to_agent_with_no_orders(self):
        """Test adding an order to an agent with no orders."""
        assert len(self.agent1._orders) == 0
        add_orders_to_single_agent(self.agent1, [self.mock_order])
        assert self.agent1._orders[0].id == self.mock_order.id

    def test_add_to_agent_with_orders(self):
        """Test adding an order to an agent with existing orders."""
        self.agent1.add_orders([MockOrderWithId(1), MockOrderWithId(2)])
        add_orders_to_single_agent(self.agent1, [self.mock_order])
        assert self.agent1._orders[2].id == self.mock_order.id

    def test_add_background_order_to_agent_with_no_orders(self):
        """Test adding an order to an agent with no orders."""
        assert len(self.agent1._orders) == 0
        add_background_orders_to_single_agent(self.agent1, [self.mock_background_order])
        assert self.agent1._background_orders[0].id == self.mock_background_order.id

    def test_add_background_order_to_agent_with_orders(self):
        """Test adding an order to an agent with existing orders."""
        self.agent1.add_background_orders(
            [MockBackgroundOrderWithId(1), MockBackgroundOrderWithId(2)]
        )
        add_background_orders_to_single_agent(self.agent1, [self.mock_background_order])
        assert self.agent1._background_orders[2].id == self.mock_background_order.id


class TestAddOrdersToAgents(unittest.TestCase):
    """Tests for adding orders to a multiple agents."""

    def setUp(self):
        """Set up test cases."""
        self.start_time = datetime(2020, 1, 1, 0, 0)
        self.time_step = timedelta(hours=1)
        self.end_time = datetime(2020, 1, 1, 5, 0)

        self.model = MockModel2d(self.start_time, self.time_step, self.end_time)
        self.agent1 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Test Agent 1",
            model=self.model,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.agent2 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Test Agent 2",
            model=self.model,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.agent3 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Test Agent 3",
            model=self.model,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.agent_list = [self.agent1, self.agent2, self.agent3]

    def test_add_same_orders_to_mult_agents(self):
        """Test that a single set of orders can be given to multiple agents."""
        order = MockOrderWithId(1234)
        add_orders_to_agents(self.agent_list, [order])

        assert self.agent1._orders[0].id == order.id
        assert self.agent2._orders[0].id == order.id
        assert self.agent3._orders[0].id == order.id

    def test_add_diff_orders_to_mult_agents(self):
        """Test that different sets of orders can be given to different agents."""
        order1 = MockOrderWithId(123)
        order2 = MockOrderWithId(246)
        order3 = MockOrderWithId(369)
        add_orders_to_agents(self.agent_list, [[order1], [order2], [order3]])

        assert self.agent1._orders[0].id == order1.id
        assert self.agent2._orders[0].id == order2.id
        assert self.agent3._orders[0].id == order3.id

    def test_not_enough_orders(self):
        """Test for exception if wrong length list is passed in."""
        order1 = MockOrderWithId(123)
        order2 = MockOrderWithId(246)

        with self.assertRaises(IndexError) as e:
            add_orders_to_agents(self.agent_list, [[order1], [order2]])
        self.assertEqual(
            "List length incompatible with number of agents. No agents created.",
            str(e.exception),
        )

    def test_add_orders_to_single_agent(self):
        """Test that a single agent can be passed into the method."""
        order1 = MockOrderWithId(369)
        order2 = MockOrderWithId(4812)

        add_orders_to_agents(self.agent1, [order1, order2])
        assert self.agent1._orders[0].id == order1.id
        assert self.agent1._orders[1].id == order2.id

    def test_add_same_background_orders_to_mult_agents(self):
        """Test that a single set of orders can be given to multiple agents."""
        order = MockBackgroundOrderWithId(1234)
        add_background_orders_to_agents(self.agent_list, [order])

        assert self.agent1._background_orders[0].id == order.id
        assert self.agent2._background_orders[0].id == order.id
        assert self.agent3._background_orders[0].id == order.id

    def test_add_diff_background_orders_to_mult_agents(self):
        """Test that different sets of orders can be given to different agents."""
        order1 = MockBackgroundOrderWithId(123)
        order2 = MockBackgroundOrderWithId(246)
        order3 = MockBackgroundOrderWithId(369)
        add_background_orders_to_agents(self.agent_list, [[order1], [order2], [order3]])

        assert self.agent1._background_orders[0].id == order1.id
        assert self.agent2._background_orders[0].id == order2.id
        assert self.agent3._background_orders[0].id == order3.id

    def test_not_background_enough_orders(self):
        """Test for exception if wrong length list is passed in."""
        order1 = MockBackgroundOrderWithId(123)
        order2 = MockBackgroundOrderWithId(246)

        with self.assertRaises(IndexError) as e:
            add_background_orders_to_agents(self.agent_list, [[order1], [order2]])
        self.assertEqual(
            "List length incompatible with number of agents. No agents created.",
            str(e.exception),
        )

    def test_add_background_orders_to_single_agent(self):
        """Test that a single agent can be passed into the method."""
        order1 = MockBackgroundOrderWithId(369)
        order2 = MockBackgroundOrderWithId(4812)

        add_background_orders_to_agents(self.agent1, [order1, order2])
        assert self.agent1._background_orders[0].id == order1.id
        assert self.agent1._background_orders[1].id == order2.id
