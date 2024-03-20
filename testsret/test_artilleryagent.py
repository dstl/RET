"""Test cases for ArtilleryAgent."""
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.artilleryagent import ArtilleryAgent
from ret.behaviours.behaviourpool import AlwaysAdder
from ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.disablecommunication import DisableCommunicationBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.move import AircraftMoveBehaviour, GroundBasedMoveBehaviour, MoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.testing.mocks import MockModel2d
from ret.weapons.weapon import BasicWeapon
from parameterized import parameterized

if TYPE_CHECKING:
    from ret.behaviours import Behaviour


class ArtilleryAgentInitialisationTest(TestCase):
    """Tests for initialisation of Artillery Agent."""

    def setUp(self):
        """Set up test model."""
        self.model = MockModel2d()

    @parameterized.expand(
        [
            [WaitBehaviour, 1],
            [MoveBehaviour, 1],
            [GroundBasedMoveBehaviour, 1],
            [FireBehaviour, 1],
            [DeployCountermeasureBehaviour, 0],
            [CommunicateWorldviewBehaviour, 0],
            [CommunicateMissionMessageBehaviour, 0],
            [CommunicateOrdersBehaviour, 0],
            [DisableCommunicationBehaviour, 0],
            [SenseBehaviour, 1],
            [HideBehaviour, 1],
        ]
    )
    def test_init_no_behaviour(self, behaviour_type: type, expected: int):
        """Test initialising Artillery Agent with no user-defined behaviour.

        Args:
            behaviour_type (type): Behaviour type
            expected (int): Expected number of behaviours present
        """
        agent = ArtilleryAgent(self.model, (0.0, 0.0), "Artillery Under Test", Affiliation.FRIENDLY)

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert len(behaviours) == expected

    @parameterized.expand(
        [
            [WaitBehaviour, WaitBehaviour()],
            [GroundBasedMoveBehaviour, GroundBasedMoveBehaviour(0, [])],
            [FireBehaviour, FireBehaviour()],
            [
                SenseBehaviour,
                SenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                ),
            ],
            [HideBehaviour, HideBehaviour()],
        ]
    )
    def test_init_with_custom_behaviour(self, behaviour_type: type, new_behaviour: Behaviour):
        """Test ArtilleryAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours

        Args:
            behaviour_type (type): Behaviour type
            new_behaviour (Behaviour): Instance of the behaviour
        """
        agent = ArtilleryAgent(
            self.model,
            (0, 0),
            "Artillery Agent Under Test",
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
            [
                SenseBehaviour,
                SenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                ),
            ],
            [HideBehaviour, HideBehaviour()],
        ]
    )
    def test_adding_behaviours_to_existing_agent(
        self, behaviour_type: type, new_behaviour: Behaviour
    ):
        """Test ArtilleryAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours

        Args:
            behaviour_type (type): Behaviour type
            new_behaviour (Behaviour): Instance of the behaviour
        """
        agent = ArtilleryAgent(
            self.model,
            (0, 0),
            "Artillery Agent Under Test",
            Affiliation.FRIENDLY,
            behaviours=[new_behaviour],
            behaviour_adder=AlwaysAdder,
            weapons=[
                BasicWeapon(
                    name="Weapon",
                    radius=10,
                    time_before_first_shot=timedelta(seconds=10),
                    time_between_rounds=timedelta(seconds=10),
                    kill_probability_per_round=0.1,
                )
            ],
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour]

        agent.behaviour_pool.add_behaviour(new_behaviour)

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour, new_behaviour]
