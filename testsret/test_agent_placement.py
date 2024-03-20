"""Tests for placement of agents."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from warnings import catch_warnings

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.space.feature import BoxFeature
from ret.testing.mocks import MockModel2d, MockModel3d
from parameterized.parameterized import parameterized

if TYPE_CHECKING:
    from ret.types import Coordinate2d, Coordinate2dOr3d, Coordinate3d


class TestAgentPlacement(unittest.TestCase):
    """Test cases for agent placement."""

    @parameterized.expand([[(0, 0)], [(1, 1)]])
    def test_placement_with_2d_in_2d(self, coord: Coordinate2d):
        """Test agent placement using a 2d coord in a 2d space.

        Args:
            coord (Coordinate2d): 2D coordinate to place in 2D space
        """
        agent = RetAgent(
            model=MockModel2d(),
            pos=coord,
            name="mock",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert len(agent.pos) == 2
        assert agent.pos == coord

    @parameterized.expand([[(0, 0)], [(1, 1)]])
    def test_placement_with_2d_in_3d(self, coord: Coordinate2d):
        """Test agent placement using a 2d coord in a 3d space.

        Args:
            coord (Coordinate2d): 2D coordinate to place in 3d space
        """
        with catch_warnings(record=True) as w:
            agent = RetAgent(
                model=MockModel3d(),
                pos=coord,
                name="mock",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

        assert len(w) == 1
        assert (
            "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
            in str(w[0].message)
        )

        assert len(agent.pos) == 3
        assert agent.pos == (coord[0], coord[1], 0)

    @parameterized.expand([[(0, 0, 0)], [(1, 1, 1)]])
    def test_placement_with_3d_in_3d(self, coord: Coordinate3d):
        """Test agent placement using a 3d coord in a 3d space.

        Args:
            coord (Coordinate3d): Coordinate to place
        """
        agent = RetAgent(
            model=MockModel3d(),
            pos=coord,
            name="mock",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert len(agent.pos) == 3
        assert agent.pos == coord

    def test_placement_with_2d_list_in_2d(self):
        """Test agent placement using a list of 2d coords in a 2d space."""
        coords: list[Coordinate2dOr3d] = [
            (0, 0),
            (1, 1),
            (10, 10),
            (100, 100),
            (1000, 1000),
        ]
        for _ in range(100):
            agent = RetAgent(
                model=MockModel2d(),
                pos=coords,
                name="mock",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            assert len(agent.pos) == 2
            assert agent.pos in coords

    def test_placement_with_2d_list_in_3d(self):
        """Test agent placement using a list of 2d coords in a 3d space."""
        coords: list[Coordinate2dOr3d] = [
            (0, 0),
            (1, 1),
            (10, 10),
            (100, 100),
            (1000, 1000),
        ]
        for _ in range(100):
            with catch_warnings(record=True) as w:
                agent = RetAgent(
                    model=MockModel3d(),
                    pos=coords,
                    name="mock",
                    affiliation=Affiliation.FRIENDLY,
                    critical_dimension=2.0,
                    reflectivity=0.1,
                    temperature=20.0,
                )

            assert len(w) == 1
            assert (
                "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
                in str(w[0].message)
            )

            assert len(agent.pos) == 3
            assert (agent.pos[0], agent.pos[1]) in coords

    def test_placement_with_3d_list_in_3d(self):
        """Test agent placement using a list of 3d coords in a 3d space."""
        coords: list[Coordinate2dOr3d] = [
            (0, 0, 0),
            (1, 1, 1),
            (10, 10, 10),
            (100, 100, 100),
            (1000, 1000, 1000),
        ]
        for _ in range(100):
            agent = RetAgent(
                model=MockModel3d(),
                pos=coords,
                name="mock",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            assert len(agent.pos) == 3
            assert agent.pos in coords

    def test_placement_with_2d_area_in_2d(self):
        """Test agent placement using a 2d area in a 2d space."""
        area = BoxFeature((5, 5), (10, 10), "area")
        for _ in range(100):
            agent = RetAgent(
                model=MockModel2d(),
                pos=area,
                name="mock",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            assert len(agent.pos) == 2
            assert area.contains(agent.pos)

    def test_placement_with_2d_area_in_3d(self):
        """Test agent placement using a 2d area in a 3d space."""
        area = BoxFeature((5, 5), (10, 10), "area")
        for _ in range(100):
            with catch_warnings(record=True) as w:
                agent = RetAgent(
                    model=MockModel3d(),
                    pos=area,
                    name="mock",
                    affiliation=Affiliation.FRIENDLY,
                    critical_dimension=2.0,
                    reflectivity=0.1,
                    temperature=20.0,
                )

            assert len(w) == 1
            assert (
                "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
                in str(w[0].message)
            )

            assert len(agent.pos) == 3
            assert area.contains(agent.pos)

    def test_placement_with_3d_area_in_3d(self):
        """Test agent placement using a 3d area in a 3d space."""
        area = BoxFeature((5, 5, 5), (10, 10, 10), "area")
        for _ in range(100):
            agent = RetAgent(
                model=MockModel3d(),
                pos=area,
                name="mock",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            assert len(agent.pos) == 3
            assert area.contains(agent.pos)
