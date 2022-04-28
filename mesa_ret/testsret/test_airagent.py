"""Test cases for AirAgent."""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.airagent import AirAgent
from mesa_ret.behaviours.behaviourpool import AlwaysAdder
from mesa_ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from mesa_ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from mesa_ret.behaviours.disablecommunication import DisableCommunicationBehaviour
from mesa_ret.behaviours.fire import FireBehaviour
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import AircraftMoveBehaviour, MoveBehaviour
from mesa_ret.behaviours.sense import SenseBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.testing.mocks import MockModel2d
from parameterized import parameterized

if TYPE_CHECKING:
    from mesa_ret.behaviours import Behaviour


class AirAgentInitialisationTest(TestCase):
    """Tests for initialisation of Air Agent."""

    def setUp(self):
        """Set up test model."""
        self.model = MockModel2d()

    @parameterized.expand(
        [
            [WaitBehaviour, 1],
            [MoveBehaviour, 1],
            [AircraftMoveBehaviour, 1],
            [FireBehaviour, 1],
            [DeployCountermeasureBehaviour, 1],
            [CommunicateWorldviewBehaviour, 0],
            [CommunicateMissionMessageBehaviour, 0],
            [CommunicateOrdersBehaviour, 0],
            [DisableCommunicationBehaviour, 0],
            [SenseBehaviour, 0],
            [HideBehaviour, 1],
        ]
    )
    def test_init_no_behaviour(self, behaviour_type: type, expected: int):
        """Test initialising Air Agent with no user-defined behaviour.

        Args:
            behaviour_type (type): Behaviour type to add
            expected (int): The expected number of behaviours of the behaviour type

        """
        agent = AirAgent(self.model, (0.0, 0.0), "Air Agent Under Test", Affiliation.FRIENDLY)

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert len(behaviours) == expected

    @parameterized.expand(
        [
            [WaitBehaviour, WaitBehaviour()],
            [AircraftMoveBehaviour, AircraftMoveBehaviour(0, [])],
            [FireBehaviour, FireBehaviour()],
            [DeployCountermeasureBehaviour, DeployCountermeasureBehaviour()],
        ]
    )
    def test_init_with_custom_behaviour(self, behaviour_type: type, new_behaviour: Behaviour):
        """Test AirAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours

        Args:
            behaviour_type (type): Behaviour type to add
            new_behaviour (Behaviour): New behaviour to add
        """
        agent = AirAgent(
            self.model,
            (0, 0),
            "Air Agent Under Test",
            Affiliation.FRIENDLY,
            behaviours=[new_behaviour],
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour]

    @parameterized.expand(
        [
            [WaitBehaviour, WaitBehaviour()],
            [AircraftMoveBehaviour, AircraftMoveBehaviour(0, [])],
            [FireBehaviour, FireBehaviour()],
            [DeployCountermeasureBehaviour, DeployCountermeasureBehaviour()],
        ]
    )
    def test_adding_behaviours_to_existing_air_agent(
        self, behaviour_type: type, new_behaviour: Behaviour
    ):
        """Test AirAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours.

        Args:
            behaviour_type (type): Behaviour type to add
            new_behaviour (Behaviour): New behaviour to add
        """
        agent = AirAgent(
            self.model,
            (0, 0),
            "Air Agent Under Test",
            Affiliation.FRIENDLY,
            behaviours=[new_behaviour],
            behaviour_adder=AlwaysAdder,
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour]

        agent.behaviour_pool.add_behaviour(new_behaviour)

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour, new_behaviour]
