"""Tests for the sensor fusion receiver."""

from datetime import datetime
from unittest import TestCase

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.sensorfusionagent import SensorFusionAgent
from mesa_ret.communication.sensorfusionreceiver import SensorFusionReceiver
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.sensing.perceivedworld import Confidence, PerceivedAgent
from mesa_ret.testing.mocks import MockCommunicateWorldviewBehaviour, MockModel2d


class TestSensorFusionReceiver(TestCase):
    """Test cases for sensor fusion receiver."""

    def test_receive(self):
        """Test that the fusion receiver updates the fusion agent."""
        model = MockModel2d()
        agent = SensorFusionAgent(
            model=model,
            pos=(0, 0),
            name="Fusion",
            affiliation=Affiliation.NEUTRAL,
            behaviours=[MockCommunicateWorldviewBehaviour()],
        )
        receiver = SensorFusionReceiver()

        new_views = [
            PerceivedAgent(
                unique_id=1,
                sense_time=datetime(2020, 1, 1),
                location=(0, 0),
                confidence=Confidence.DETECT,
                casualty_state=AgentCasualtyState.ALIVE,
            )
        ]

        payload = {"worldview": new_views}

        self.assertFalse(agent.new_info)
        receiver.receive(agent, payload)

        self.assertTrue(agent.new_info)
