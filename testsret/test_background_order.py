"""Tests for background orders."""

import unittest
from datetime import datetime, timedelta

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.orders.background_order import BackgroundOrder
from ret.orders.tasks.move import MoveTask
from ret.orders.triggers.time import TimeTrigger
from ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
from ret.testing.mocks import MockBasicMoveBehaviour, MockModel3d

background_move_order = BackgroundOrder(
    time_period=timedelta(seconds=60),
    task=MoveTask(destination=(500.0, 0.0, 0.0), tolerance=1.0),
)

background_move_order_with_trigger = BackgroundOrder(
    trigger=TimeTrigger(datetime(2020, 1, 1, 0, 1)),
    time_period=timedelta(seconds=60),
    task=MoveTask(destination=(500.0, 0.0, 0.0), tolerance=1.0),
)


class BackgroundOrderTestModel(MockModel3d):
    """Initialise model for background order tests."""

    def __init__(self) -> None:
        """Create a new BackgroundOrderTestModel."""
        super().__init__(time_step=timedelta(seconds=10))

        self.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=1000.0,
            y_max=1000.0,
        )


class TestBackgroundOrder(unittest.TestCase):
    """Test background order behaviour."""

    def setUp(self):
        """Set up test cases."""
        self.model = BackgroundOrderTestModel()

        self.agent_1 = RetAgent(
            model=self.model,
            pos=(0.0, 0.0, 0.0),
            name="Background Order Agent 1",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[MockBasicMoveBehaviour(10)],
            background_orders=[background_move_order],
        )

        self.agent_2 = RetAgent(
            model=self.model,
            pos=(0.0, 0.0, 0.0),
            name="Background Order Agent 1",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[MockBasicMoveBehaviour(10)],
            background_orders=[background_move_order_with_trigger],
        )

    def test_background_order(self):
        """Tests that background orders are carried out.

        Background order carried out once every 60s, model time step is 10s.
        """
        assert self.agent_1.pos == (0.0, 0.0, 0.0)

        self.model.step()  # Step once, t=10s

        assert self.agent_1.pos == (10.0, 0.0, 0.0)

        for _ in range(5):
            self.model.step()  # Step 5 times, t=60s

            assert self.agent_1.pos == (10.0, 0.0, 0.0)

        self.model.step()  # Step once, t=70s

        assert self.agent_1.pos == (20.0, 0.0, 0.0)

    def test_background_order_with_trigger(self):
        """Tests that background orders are carried out when trigger condition is met."""
        assert self.agent_2.pos == (0.0, 0.0, 0.0)

        for _ in range(6):
            self.model.step()  # Step 6 times, t=60s
            assert self.agent_2.pos == (0.0, 0.0, 0.0)

        self.model.step()  # Step once, t=70s
        assert self.agent_2.pos == (10.0, 0.0, 0.0)

        for _ in range(20):
            self.model.step()
            assert self.agent_2.pos == (10.0, 0.0, 0.0)
