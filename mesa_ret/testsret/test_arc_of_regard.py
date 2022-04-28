"""Tests for arc of regard."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.testing.mocks import MockModel2d
from parameterized.parameterized import parameterized

if TYPE_CHECKING:

    from mesa_ret.sensing.sensor import ArcOfRegard


class TestArcOfRegard(unittest.TestCase):
    """Test arc of regard calculations."""

    def setUp(self):
        """Set up test cases."""
        self.arc_of_regard: ArcOfRegard = {(-20, 20): 4, (-90, -20): 3, (20, 90): 3}

        self.model = MockModel2d()

        self.sensor_agent = RetAgent(
            model=self.model,
            pos=(100.0, 100.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            arc_of_regard=self.arc_of_regard,
        )

    @staticmethod
    def assert_arc_of_regard_normalised(arc_of_regard):
        """Assert that the arc of regard has been normalised compared to the setUp value.

        Args:
            arc_of_regard ([type]): The arc of regard to compare
        """
        assert arc_of_regard is not None
        assert sum(arc_of_regard.values()) == 1
        for arc in arc_of_regard.keys():
            assert 0 <= arc[0] <= 360
            assert 0 <= arc[1] <= 360
        assert (340, 20) in arc_of_regard
        assert (270, 340) in arc_of_regard
        assert (20, 90) in arc_of_regard
        assert arc_of_regard[(340, 20)] == 0.4
        assert arc_of_regard[(270, 340)] == 0.3
        assert arc_of_regard[(20, 90)] == 0.3

    def test_normalise_arc_of_regard(self):
        """Test normalising arc of regard."""
        arc_of_regard: ArcOfRegard = self.arc_of_regard
        normalised_arc_of_regard = self.sensor_agent._normalise_arc_of_regard(arc_of_regard)
        self.assert_arc_of_regard_normalised(normalised_arc_of_regard)

    def test_set_arc_of_regard(self):
        """Test that normalise method was used in agent constructor."""
        assert self.sensor_agent.arc_of_regard is not None
        self.assert_arc_of_regard_normalised(self.sensor_agent.arc_of_regard)

    @parameterized.expand(
        [
            [{(-20, 20): 4, (10, 30): 6}],
            [{(-20, 20): 4, (-10, 30): 6}],
            [{(-20, 20): 4, (20, 360): 6}],
        ]
    )
    def test_overlapping_sectors(self, arc_of_regard: ArcOfRegard):
        """Test that an error is raised if an arc of regard has overlapping sectors.

        Args:
            arc_of_regard (ArcOfRegard): Arc of regard to test
        """
        with self.assertRaises(ValueError) as e:
            RetAgent(
                model=self.model,
                pos=(100.0, 100.0),
                name="Sensor Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                arc_of_regard=arc_of_regard,
            )
        assert "Arc of Regard cannot contain overlapping sectors." == str(e.exception)
