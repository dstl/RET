"""Tests for ret unit portrayal."""
import unittest

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.testing.mocks import MockModel2d
from mesa_ret.visualisation.retportrayal import ret_portrayal


class TestGroupAgent(unittest.TestCase):
    """Ret unit portrayal."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()

        self.agent1 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Agent 1",
            model=self.model,
        )

        self.agent2 = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(2, 2),
            name="Agent 2",
            model=self.model,
        )

    def test_none_param(self):
        """Test that None is returned when None is passed in."""
        portrayal = ret_portrayal(None)
        assert portrayal is None

    def test_agent_in_group(self):
        """Test that None is returned when agent is in a group."""
        self.agent1.in_group = True
        portrayal = ret_portrayal(self.agent1)
        assert portrayal is None

    def test_agent_not_in_group(self):
        """Test that a portrayal is returned for a valid agent."""
        expected_coord = str([round(coord, 2) for coord in self.agent2.pos])

        expected_result = {
            "Shape": "svg",
            "svgSource": self.agent2.icon,
            "Name": self.agent2.name,
            "Coord": expected_coord,
            "Layer": 0,
        }

        portrayal = ret_portrayal(self.agent2)
        assert portrayal == expected_result
