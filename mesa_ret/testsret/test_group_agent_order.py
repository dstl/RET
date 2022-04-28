"""Test cases for group agent orders."""
from __future__ import annotations

import unittest
from datetime import datetime

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.orders.order import GroupAgentOrder
from mesa_ret.orders.tasks.move import MoveTask
from mesa_ret.orders.triggers.time import TimeTrigger
from mesa_ret.testing.mocks import MockModel2d, MockMoveBehaviour, MockTask


class MockCompletableTask(MockTask):
    """Mock task to give to a subordinate agent in a group."""

    agent: RetAgent

    def __init__(self, agent: RetAgent, log: bool = True):
        """Initialise a mock subordinate agent task.

        Args:
            agent (RetAgent): A reference to the subordinate agent the task is given to.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)
        self.agent = agent


class TestGroupAgentOrders(unittest.TestCase):
    """Test cases for group agent orders."""

    move_behaviour = MockMoveBehaviour()

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()
        self.agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            reflectivity=0.1,
            temperature=20.0,
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            behaviours=[self.move_behaviour],
        )

        self.group_order = GroupAgentOrder(
            TimeTrigger(datetime(2020, 1, 1, 0, 1)), MoveTask((5000, 5000), 1)
        )
        self.persistent_group_order = GroupAgentOrder(
            TimeTrigger(datetime(2020, 1, 1, 0, 1)), MoveTask((0, 0), 1), True
        )
        self.future_group_order = GroupAgentOrder(
            TimeTrigger(datetime(2021, 1, 1, 0, 1)), MoveTask((5000, 5000), 1)
        )
        self.future_persistent_group_order = GroupAgentOrder(
            TimeTrigger(datetime(2021, 1, 1, 0, 1)), MoveTask((0, 0), 1), True
        )

    def test_is_persistent(self):
        """Test order persistence."""
        assert not self.group_order.is_persistent()
        assert self.persistent_group_order.is_persistent()

    def test_is_condition_met(self):
        """Test whether an order's conditions is met."""
        assert self.group_order.is_condition_met(self.agent)
        assert self.persistent_group_order.is_condition_met(self.agent)

    def test_is_condition_not_met(self):
        """Test whether an order's conditions is not met."""
        assert not self.future_group_order.is_condition_met(self.agent)
        assert not self.future_persistent_group_order.is_condition_met(self.agent)

    def test_execute_step(self):
        """Test the outcome of executing an order."""
        self.group_order.execute_step(self.agent)
        assert self.group_order.is_complete(self.agent)

        self.persistent_group_order.execute_step(self.agent)
        assert self.persistent_group_order.is_complete(self.agent)

    def test_order_str_method(self):
        """Test the compound trigger string method."""
        text = str(self.group_order._task) + " and " + str(self.group_order._trigger)
        assert str(self.group_order) == text

    def test_order_get_new_instance(self):
        """Test the get_new_instance method."""
        group_order = GroupAgentOrder(
            TimeTrigger(datetime(2020, 1, 1, 0, 1)),
            MoveTask((5000, 5000), 1),
            persistent=True,
            priority=5,
        )
        group_order._trigger._sticky = True
        group_order._trigger._active_flag = True
        clone = group_order.get_new_instance()

        assert group_order is not clone
        assert group_order._trigger is not clone._trigger
        assert group_order._trigger._time == clone._trigger._time
        self.assertFalse(clone._trigger._active_flag)
        assert group_order._task is not clone._task
        assert group_order._task._destination == clone._task._destination
        assert group_order._persistent == clone._persistent
        assert group_order.priority == clone.priority

    def test_order_get_new_instance_sticky_status_maintained(self):
        """Test the get_new_instance_sticky_status_maintained method."""
        group_order = GroupAgentOrder(
            TimeTrigger(datetime(2020, 1, 1, 0, 1)),
            MoveTask((5000, 5000), 1),
            persistent=True,
            priority=5,
        )
        group_order._trigger._sticky = True
        group_order._trigger._active_flag = False
        clone = group_order.get_new_instance_sticky_status_maintained()

        assert group_order is not clone
        assert group_order._trigger is not clone._trigger
        assert group_order._trigger._time == clone._trigger._time
        self.assertFalse(clone._trigger._active_flag)
        assert group_order._task is not clone._task
        assert group_order._task._destination == clone._task._destination
        assert group_order._persistent == clone._persistent
        assert group_order.priority == clone.priority

        group_order._trigger._sticky = True
        group_order._trigger._active_flag = True
        clone = group_order.get_new_instance_sticky_status_maintained()

        assert group_order is not clone
        assert group_order._trigger is not clone._trigger
        assert group_order._trigger._time == clone._trigger._time
        self.assertTrue(clone._trigger._active_flag)
        assert group_order._task is not clone._task
        assert group_order._task._destination == clone._task._destination
        assert group_order._persistent == clone._persistent
        assert group_order.priority == clone.priority
