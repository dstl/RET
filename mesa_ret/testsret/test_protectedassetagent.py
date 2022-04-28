"""Test cases for ProtectedAssetAgent."""


from unittest import TestCase

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.protectedassetagent import ProtectedAssetAgent
from mesa_ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from mesa_ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from mesa_ret.behaviours.disablecommunication import DisableCommunicationBehaviour
from mesa_ret.behaviours.fire import FireBehaviour
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import GroundBasedMoveBehaviour, MoveBehaviour
from mesa_ret.behaviours.sense import SenseBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.testing.mocks import MockModel2d
from parameterized import parameterized


class ProtectedAssetAgentInitialisationTest(TestCase):
    """Tests for initialisation of Protected Asset Agent."""

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
            [HideBehaviour, 0],
        ]
    )
    def test_init_no_behaviour(self, behaviour_type: type, expected: int):
        """Test initialising Protected Asset Agent with no user-defined behaviour.

        Args:
            behaviour_type (type): Type of behaviour
            expected (int): Expected number of occurrences of the behaviour
        """
        agent = ProtectedAssetAgent(
            self.model,
            (0.0, 0.0),
            "Protected Asset Agent Under Test",
            Affiliation.FRIENDLY,
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert len(behaviours) == expected
