"""Test cases for ProtectedAssetAgent."""


from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.protectedassetagent import ProtectedAssetAgent
from ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.disablecommunication import DisableCommunicationBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.move import GroundBasedMoveBehaviour, MoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.testing.mocks import MockModel2d
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
