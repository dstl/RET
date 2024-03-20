"""Test for CommunicationReceiver."""
from __future__ import annotations

import random
import warnings
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING
from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.communication.communicationreceiver import (
    CommunicationReceiver,
    MissionMessageHandler,
    WorldviewHandler,
)
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import Confidence, PerceivedAgent
from ret.testing.mocks import MockModel2d, MockOrderWithId

if TYPE_CHECKING:
    from ret.orders.order import Order
    from ret.template import Template


class TestCommunicationReceiver(TestCase):
    """Tests for CommunicationReceiver."""

    def setUp(self):
        """Set up test case."""
        model = MockModel2d()
        self.agent = RetAgent(
            model=model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

    def test_receive_order(self):
        """Test that a communication receiver can receive an order."""
        communication_receiver = CommunicationReceiver()
        random_int = random.randint(0, 9999)
        order = MockOrderWithId(random_int)
        payload = {"orders": order}
        communication_receiver.receive(self.agent, payload)

        assert len(self.agent._orders) == 1
        assert self.agent._orders[0].id == random_int

    def test_receive_orders(self):
        """Test that a communication receiver can receive multiple orders.

        Orders received in a single step.
        """
        communication_receiver = CommunicationReceiver()
        random_int_1 = random.randint(0, 9999)
        random_int_2 = random.randint(0, 9999)
        orders: list[Template[Order]] = [
            MockOrderWithId(random_int_1),
            MockOrderWithId(random_int_2),
        ]
        payload = {"orders": orders}
        communication_receiver.receive(self.agent, payload)

        assert len(self.agent._orders) == 2
        assert self.agent._orders[0].id == random_int_1
        assert self.agent._orders[1].id == random_int_2

    def test_receive_multiple_orders(self):
        """Test that a communication receiver can receive multiple orders.

        Orders received in multiple steps.
        """
        communication_receiver = CommunicationReceiver()
        random_int_1 = random.randint(0, 9999)
        random_int_2 = random.randint(0, 9999)
        orders: list[Template[Order]] = [
            MockOrderWithId(random_int_1),
            MockOrderWithId(random_int_2),
        ]
        payload_1 = {"orders": orders[0]}
        payload_2 = {"orders": orders[1]}
        communication_receiver.receive(self.agent, payload_1)
        communication_receiver.receive(self.agent, payload_2)

        assert len(self.agent._orders) == 2
        assert self.agent._orders[0].id == random_int_1
        assert self.agent._orders[1].id == random_int_2

    def test_unhandled_message(self):
        """Test a communication receiver throws a warning if for unhandled messages."""
        communication_receiver = CommunicationReceiver()
        payload = {"unhandled": None}

        with warnings.catch_warnings(record=True) as w:
            communication_receiver.receive(self.agent, payload)
            assert "Message component unhandled is not handled by" in str(w[0].message)

    def test_combine_worldviews(self):
        """Test that combined worldviews joins together multiple views."""
        view_1 = [
            PerceivedAgent(
                unique_id=1,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.KNOWN,
                location=(1, 1),
                casualty_state=AgentCasualtyState.ALIVE,
            )
        ]
        view_2 = [
            PerceivedAgent(
                unique_id=2,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.DETECT,
                location=(2, 2),
                casualty_state=AgentCasualtyState.ALIVE,
            ),
            PerceivedAgent(
                unique_id=3,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.RECOGNISE,
                location=(3, 3),
                casualty_state=AgentCasualtyState.ALIVE,
            ),
        ]
        worldview_handler = WorldviewHandler()
        combined_views = worldview_handler.combine_worldviews(view_1, view_2)

        assert isinstance(combined_views, Iterable)
        assert len(combined_views) == 3

    def test_pick_agent_with_highest_confidence_at_location(self):
        """Test that combining world-views filters for duplicates at location.

        Perceived agents with lower confidences are removed.
        """
        view_1 = [
            PerceivedAgent(
                unique_id=1,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.RECOGNISE,
                location=(1, 1),
                casualty_state=AgentCasualtyState.ALIVE,
            )
        ]
        view_2 = [
            PerceivedAgent(
                unique_id=2,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.DETECT,
                location=(1, 1),
                casualty_state=AgentCasualtyState.ALIVE,
            ),
            PerceivedAgent(
                unique_id=3,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.IDENTIFY,
                location=(1, 1),
                casualty_state=AgentCasualtyState.ALIVE,
            ),
        ]
        worldview_handler = WorldviewHandler()
        combined_views = worldview_handler.combine_worldviews(view_1, view_2)

        assert isinstance(combined_views, Iterable)
        assert len(combined_views) == 1
        assert combined_views[0].confidence == Confidence.IDENTIFY

    def test_mission_message_handler(self):
        """Test the mission message handler."""
        message = "mission complete"
        mission_handler = MissionMessageHandler()
        mission_handler.receive(self.agent, message)
        assert message in self.agent.mission_messages
