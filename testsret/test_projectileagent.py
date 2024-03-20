"""Test cases for ProjectileAgent."""
from __future__ import annotations

from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.projectileagent import ProjectileAgent, GuidedProjectileAgent
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.move import MoveBehaviour
from ret.testing.mocks import MockModel2d, MockAgent
from parameterized import parameterized
from ret.weapons.weapon import ProjectilePayload


class ProjectileAgentInitialisationTest(TestCase):
    """Tests for initialisation of Projectile Agent."""

    def setUp(self):
        """Set up test model."""
        self.model = MockModel2d()

        self.weapon = ProjectilePayload(
            name="Test Payload", radius=10, kill_probability_per_round=0.9
        )

    @parameterized.expand(
        [
            [MoveBehaviour, 1],
            [FireBehaviour, 1],
        ]
    )
    def test_init_no_behaviour(self, behaviour_type: type, expected: int):
        """Test initialising Air Agent with no user-defined behaviour.

        Args:
            behaviour_type (type): Behaviour type to add
            expected (int): The expected number of behaviours of the behaviour type

        """
        agent = ProjectileAgent(
            self.model,
            (0.0, 0.0),
            "Projectile Agent Under Test",
            Affiliation.FRIENDLY,
            base_speed=1,
            weapon=self.weapon,
            critical_dimension=1,
            firer=MockAgent(unique_id=5, pos=(0, 0)),
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert len(behaviours) == expected

    @parameterized.expand(
        [
            [MoveBehaviour, 1],
            [FireBehaviour, 1],
        ]
    )
    def test_init_guided_no_behaviour(self, behaviour_type: type, expected: int):
        """Test initialising Air Agent with no user-defined behaviour, guided weapon.

        Args:
            behaviour_type (type): Behaviour type to add
            expected (int): The expected number of behaviours of the behaviour type

        """
        agent = GuidedProjectileAgent(
            self.model,
            (0.0, 0.0),
            "Projectile Agent Under Test",
            Affiliation.FRIENDLY,
            base_speed=1,
            weapon=self.weapon,
            critical_dimension=1,
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert len(behaviours) == expected
