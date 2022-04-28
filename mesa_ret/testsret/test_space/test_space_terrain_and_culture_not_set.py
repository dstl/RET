"""Tests for space where terrain and culture is not set."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

from mesa_ret.space.culture import default_culture
from mesa_ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from mesa_ret.testing.mocks import MockAgent
from parameterized import parameterized_class
from tests.test_space import TEST_AGENTS
from testsret.test_space.test_space import get_test_name

if TYPE_CHECKING:
    from mesa_ret.types import Coordinate2d


@parameterized_class(
    ("space_type", "pos"),
    [
        ("2d", (-30, -50)),
        ("2d", (20, -50)),
        ("2d", (69, -50)),
        ("2d", (69, 0)),
        ("2d", (69, 49)),
        ("2d", (20, 49)),
        ("2d", (-30, 49)),
        ("2d", (-30, 0)),
        ("3d", (-30, -50, 0)),
        ("3d", (20, -50, 0)),
        ("3d", (69, -50, 0)),
        ("3d", (69, 0, 0)),
        ("3d", (69, 49, 0)),
        ("3d", (20, 49, 0)),
        ("3d", (-30, 49, 0)),
        ("3d", (-30, 0, 0)),
    ],
    class_name_func=get_test_name,
)
class TestTerrainAndCultureNotSet(unittest.TestCase):
    """Test continuous space without terrain or culture maps provided."""

    space_type: str
    pos: Coordinate2d

    def setUp(self):
        """Set up the given space."""
        if self.space_type == "2d":
            self.setup_2d()
        elif self.space_type == "3d":
            self.setup_3d()

    def setup_2d(self):
        """Create a test 2d space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=70,
            y_max=50,
            x_min=-30,
            y_min=-50,
        )
        self.agents = []
        for i, pos in enumerate(TEST_AGENTS):
            a = MockAgent(i, (0, 0))
            self.agents.append(a)
            self.space.place_agent(a, pos)

    def setup_3d(self):
        """Create a test 3d space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=70,
            y_max=50,
            x_min=-30,
            y_min=-50,
        )
        self.agents = []
        for i, pos in enumerate(TEST_AGENTS):
            a = MockAgent(i, (0, 0, 0))
            self.agents.append(a)
            pos_3d = [p for p in pos] + [0]
            self.space.place_agent(a, pos_3d)  # type: ignore

    def test_terrain_height(self):
        """Test height returns 0 when terrain not supplied."""
        assert self.space.get_terrain_height(self.pos) == 0

    def test_terrain_gradient(self):
        """Test gradient returns 0 when terrain not supplied."""
        assert self.space.get_terrain_gradient(self.pos) == (0, 0)

    def test_terrain_gradient_along_vec(self):
        """Test gradient along vector returns 0 when terrain not supplied."""
        assert self.space.get_terrain_gradient_along_vec(self.pos, (1, 0)) == 0

    def test_culture(self):
        """Test culture returns default_culture when culture not supplied."""
        assert self.space.get_culture(self.pos) is default_culture()
        assert self.space.get_culture(self.pos).name == "Default Culture"
