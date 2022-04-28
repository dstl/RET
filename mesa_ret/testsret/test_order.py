"""Test cases for orders, tasks and triggers."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.orders.order import Order
from mesa_ret.orders.tasks.move import MoveTask
from mesa_ret.orders.tasks.wait import WaitTask
from mesa_ret.orders.triggers.time import TimeTrigger
from mesa_ret.testing.mocks import MockModel2d, MockMoveBehaviour, MockWaitBehaviour
from parameterized import parameterized

if TYPE_CHECKING:
    from typing import Any

    from mesa_ret.template import Template


class TestOrdersSequencing(unittest.TestCase):
    """Test cases for order sequencing."""

    time_0 = datetime(2020, 1, 1, 0, 0)
    time_1 = time_0 + timedelta(hours=1)
    time_1_25 = time_1 + timedelta(hours=0.25)
    time_1_5 = time_1 + timedelta(hours=0.5)
    time_2 = time_1 + timedelta(hours=1)
    time_3 = time_2 + timedelta(hours=1)
    time_4 = time_3 + timedelta(hours=1)
    time_4_25 = time_4 + timedelta(hours=0.25)
    time_4_5 = time_4 + timedelta(hours=0.5)
    time_4_75 = time_4 + timedelta(hours=0.75)
    time_5 = time_4 + timedelta(hours=1)
    time_6 = time_5 + timedelta(hours=1)
    time_6_5 = time_6 + timedelta(hours=0.5)
    time_7 = time_6 + timedelta(hours=1)
    time_8 = time_7 + timedelta(hours=1)

    times = [time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8]

    orders_off_time_step: list[Order] = [
        Order(TimeTrigger(time_1_5, True), WaitTask(timedelta(hours=3)), priority=0),
        Order(TimeTrigger(time_6_5, True), WaitTask(timedelta(hours=1)), priority=0),
    ]

    orders_on_time_step: list[Order] = [
        Order(TimeTrigger(time_1, True), WaitTask(timedelta(hours=3)), priority=0),
        Order(TimeTrigger(time_6, True), WaitTask(timedelta(hours=1)), priority=0),
    ]

    orders_within_time_step: list[Order] = [
        Order(TimeTrigger(time_1_25, True), WaitTask(timedelta(hours=0.5)), priority=0),
    ]

    orders_overlapping_in_time_step: list[Order] = [
        Order(TimeTrigger(time_1_5, True), WaitTask(timedelta(hours=3.25)), priority=0),
        Order(TimeTrigger(time_4_25, True), WaitTask(timedelta(hours=3.25)), priority=0),
    ]

    orders_start_and_finish_in_time_step: list[Order] = [
        Order(TimeTrigger(time_1_5, True), WaitTask(timedelta(hours=2.75)), priority=0),
        Order(TimeTrigger(time_4_75, True), WaitTask(timedelta(hours=2.75)), priority=0),
    ]

    orders_start_and_finish_at_time_step: list[Order] = [
        Order(TimeTrigger(time_1_5, True), WaitTask(timedelta(hours=2.5)), priority=0),
        Order(TimeTrigger(time_4, True), WaitTask(timedelta(hours=3.5)), priority=0),
    ]

    @parameterized.expand(
        [
            [
                orders_off_time_step,
                [None, None, time_1_5, time_1_5, None, None, None, time_6_5, None],
            ],
            [
                orders_on_time_step,
                [None, None, time_1, time_1, None, None, None, time_6, None],
            ],
            [
                orders_within_time_step,
                [None, None, time_1_25, None, None, None, None, None, None],
            ],
            [
                orders_overlapping_in_time_step,
                [
                    None,
                    None,
                    time_1_5,
                    time_1_5,
                    time_1_5,
                    time_4_25,
                    time_4_25,
                    time_4_25,
                    None,
                ],
            ],
            [
                orders_start_and_finish_in_time_step,
                [
                    None,
                    None,
                    time_1_5,
                    time_1_5,
                    None,
                    time_4_75,
                    time_4_75,
                    None,
                    None,
                ],
            ],
            [
                orders_start_and_finish_at_time_step,
                [None, None, time_1_5, time_1_5, None, time_4, time_4, time_4, None],
            ],
        ]
    )
    def test_iterate(self, orders_list: list[Template[Order]], states: list[Any]):
        """Test that model time iterates as the model is stepped.

        Args:
            orders_list (list[Template[Order]]): Orders
            states (list[Any]): State following each timestep
        """
        move_behaviour = MockMoveBehaviour()
        wait_behaviour = MockWaitBehaviour()

        model = MockModel2d()
        agent = RetAgent(
            model=model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            orders=orders_list,
            behaviours=[move_behaviour, wait_behaviour],
        )

        for i in range(8):
            assert model.get_time() == self.times[i]
            if states[i] is not None:
                assert agent.active_order._trigger._time == states[i]  # type: ignore
            else:
                assert agent.active_order is None
            model.step()


class TestOrders(unittest.TestCase):
    """Test cases for orders."""

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

        self.order = Order(TimeTrigger(datetime(2020, 1, 1, 0, 1)), MoveTask((5000, 5000), 1))
        self.persistent_order = Order(
            TimeTrigger(datetime(2020, 1, 1, 0, 1)), MoveTask((0, 0), 1), True
        )
        self.future_order = Order(
            TimeTrigger(datetime(2021, 1, 1, 0, 1)), MoveTask((5000, 5000), 1)
        )
        self.future_persistent_order = Order(
            TimeTrigger(datetime(2021, 1, 1, 0, 1)), MoveTask((0, 0), 1), True
        )

    def test_is_persistent(self):
        """Test order persistence."""
        assert not self.order.is_persistent()
        assert self.persistent_order.is_persistent()

    def test_is_condition_met(self):
        """Test whether an order's conditions is met."""
        assert self.order.is_condition_met(self.agent)
        assert self.persistent_order.is_condition_met(self.agent)

    def test_is_condition_not_met(self):
        """Test whether an order's conditions is not met."""
        assert not self.future_order.is_condition_met(self.agent)
        assert not self.future_persistent_order.is_condition_met(self.agent)

    def test_execute_step(self):
        """Test the outcome of executing an order."""
        self.order.execute_step(self.agent)
        assert self.order.is_complete(self.agent)

        self.persistent_order.execute_step(self.agent)
        assert self.persistent_order.is_complete(self.agent)

    def test_order_str_method(self):
        """Test the compound trigger string method."""
        text = str(self.order._task) + " and " + str(self.order._trigger)
        assert str(self.order) == text

    def test_order_get_new_instance(self):
        """Test the get_new_instance method."""
        order = Order(
            TimeTrigger(datetime(2020, 1, 1, 0, 1)),
            MoveTask((5000, 5000), 1),
            persistent=True,
            priority=5,
        )
        order._trigger._sticky = True
        order._trigger._active_flag = True
        clone = order.get_new_instance()

        assert order is not clone
        assert order._trigger is not clone._trigger
        assert order._trigger._time == clone._trigger._time
        self.assertFalse(clone._trigger._active_flag)
        assert order._task is not clone._task
        assert order._task._destination == clone._task._destination
        assert order._persistent == clone._persistent
        assert order.priority == clone.priority

    def test_order_get_new_instance_sticky_status_maintained(self):
        """Test the get_new_instance_sticky_status_maintained method."""
        order = Order(
            TimeTrigger(datetime(2020, 1, 1, 0, 1)),
            MoveTask((5000, 5000), 1),
            persistent=True,
            priority=5,
        )
        order._trigger._sticky = True
        order._trigger._active_flag = False
        clone = order.get_new_instance_sticky_status_maintained()

        assert order is not clone
        assert order._trigger is not clone._trigger
        assert order._trigger._time == clone._trigger._time
        self.assertFalse(clone._trigger._active_flag)
        assert order._task is not clone._task
        assert order._task._destination == clone._task._destination
        assert order._persistent == clone._persistent
        assert order.priority == clone.priority

        order._trigger._sticky = True
        order._trigger._active_flag = True
        clone = order.get_new_instance_sticky_status_maintained()

        assert order is not clone
        assert order._trigger is not clone._trigger
        assert order._trigger._time == clone._trigger._time
        self.assertTrue(clone._trigger._active_flag)
        assert order._task is not clone._task
        assert order._task._destination == clone._task._destination
        assert order._persistent == clone._persistent
        assert order.priority == clone.priority
