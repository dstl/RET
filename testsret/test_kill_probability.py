"""Tests for functionality around kill probabilities."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from datetime import timedelta
from ret.agents.agenttype import AgentType
from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.weapons.weapon import BasicWeapon
from ret.weapons.mutil_kill_probability import ProbabilityByType
from ret.testing.mocks import (
    MockModel2d,
    MockWeaponWithVariableKillProb,
    MockModel,
    MockFireBehaviour,
)

if TYPE_CHECKING:
    from ret.types import Coordinate2dOr3d


class TestFireBehaviours(unittest.TestCase):
    """Test ret agent fire behaviour."""

    def create_model(self) -> MockModel:
        """Return model.

        Returns:
            MockModel: 2D model
        """
        return MockModel2d()

    def get_agent_1_location(self) -> Coordinate2dOr3d:
        """Return coordinate of agent 1.

        Returns:
            Coordinate2dOr3d: 2D coordinate at (0, 0)
        """
        return (0, 0)

    def get_agent_2_location(self) -> Coordinate2dOr3d:
        """Return coordinate of agent 2.

        Returns:
            Coordinate2dOr3d: 2D coordinate at (10, 10)
        """
        return (10, 10)

    def get_agent_3_location(self) -> Coordinate2dOr3d:
        """Return coordinate of agent 2.

        Returns:
            Coordinate2dOr3d: 2D coordinate at (20, 20)
        """
        return (20, 20)

    def get_target(self) -> Coordinate2dOr3d:
        """Return coordinate of target.

        Returns:
            Coordinate2dOr3d: 2D coordinate at (9, 9)
        """
        return (9, 9)

    def setUp(self):
        """Set up test cases."""
        self.model = self.create_model()

        self.firer = RetAgent(
            model=self.model,
            pos=self.get_agent_1_location(),
            name="firer",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.targetAir = RetAgent(
            model=self.model,
            pos=self.get_agent_2_location(),
            name="air target",
            agent_type=AgentType.AIR,
            affiliation=Affiliation.UNKNOWN,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.targetArmour = RetAgent(
            model=self.model,
            pos=self.get_agent_3_location(),
            name="armour target",
            agent_type=AgentType.ARMOUR,
            affiliation=Affiliation.UNKNOWN,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

    def test_fire(self):
        """Firing test with weapon with probability of 0.0 and 1.0 for AIR and ARMOUR targets."""
        behaviour = MockFireBehaviour()
        self.assertFalse(self.targetAir.killed)
        self.assertFalse(self.targetArmour.killed)
        weapon = MockWeaponWithVariableKillProb()
        behaviour.step(firer=self.firer, rounds=1, weapon=weapon, location=self.targetAir.pos)
        self.assertTrue(self.targetAir.killed)
        behaviour.step(firer=self.firer, rounds=1, weapon=weapon, location=self.targetArmour.pos)
        self.assertFalse(self.targetArmour.killed)

    def test_kill_prob(self):
        """Test Variable kill prob returns correct kill prob for agent type."""
        weapon = MockWeaponWithVariableKillProb()

        assert weapon._kill_probability_per_round.get_probability(AgentType.ARMOUR) == 0.0
        assert weapon._kill_probability_per_round.get_probability(AgentType.GENERIC) == 0.2
        assert weapon._kill_probability_per_round.get_probability(AgentType.GROUP) == 0.4
        assert weapon._kill_probability_per_round.get_probability(AgentType.OTHER) == 0.6
        assert weapon._kill_probability_per_round.get_probability(AgentType.AIR) == 1.0
        assert weapon._kill_probability_per_round.get_probability(AgentType.PROTECTED_ASSET) == 0.5

    def test_validator_empty_prob_dict(self):
        """Test demonstrating validator will reject empty kill_probability_by_agent_type."""
        with self.assertRaises(ValueError) as e:
            BasicWeapon(
                name="Test Weapon",
                radius=3000,
                time_before_first_shot=timedelta(seconds=0),
                time_between_rounds=timedelta(seconds=30),
                kill_probability_per_round=ProbabilityByType(
                    base_kill_probability_per_round=0.5, kill_probability_by_agent_type={}
                ),
            )
        self.assertEqual(
            "kill_probability_by_agent_type is empty",
            str(e.exception),
        )

    def test_validator_too_large_base_prob(self):
        """Test demonstrating validator will reject base_kill_probability_per_round above 1."""
        with self.assertRaises(ValueError) as e:
            BasicWeapon(
                name="Test Weapon",
                radius=3000,
                time_before_first_shot=timedelta(seconds=0),
                time_between_rounds=timedelta(seconds=30),
                kill_probability_per_round=ProbabilityByType(
                    base_kill_probability_per_round=2,
                    kill_probability_by_agent_type={AgentType.ARMOUR: 1.0},
                ),
            )
        self.assertEqual(
            "Invalid base_kill_probability_per_round. Must between 0 and 1.",
            str(e.exception),
        )

    def test_validator_too_large_variable_prob(self):
        """Test demonstrating validator will reject  kill_probability_by_agent_type above 1."""
        with self.assertRaises(ValueError) as e:
            BasicWeapon(
                name="Test Weapon",
                radius=3000,
                time_before_first_shot=timedelta(seconds=0),
                time_between_rounds=timedelta(seconds=30),
                kill_probability_per_round=ProbabilityByType(
                    base_kill_probability_per_round=1,
                    kill_probability_by_agent_type={AgentType.ARMOUR: 2.0},
                ),
            )
        self.assertEqual(
            f"Invalid kill probability for {AgentType.ARMOUR}: {2.0}."
            + "Must be a float between 0 and 1.",
            str(e.exception),
        )

    def test_validator_wrong_types_variable_prob(self):
        """Test demonstrating validator will reject string probabilities."""
        with self.assertRaises(ValueError) as e:
            BasicWeapon(
                name="Test Weapon",
                radius=3000,
                time_before_first_shot=timedelta(seconds=0),
                time_between_rounds=timedelta(seconds=30),
                kill_probability_per_round=ProbabilityByType(
                    base_kill_probability_per_round=1,
                    kill_probability_by_agent_type={AgentType.ARMOUR: "2.0"},
                ),
            )
            print(e)
        self.assertEqual(
            f"Invalid kill probability for {AgentType.ARMOUR}: 2.0."
            + "Must be a float between 0 and 1.",
            str(e.exception),
        )
