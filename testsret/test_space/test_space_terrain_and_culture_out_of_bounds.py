"""Tests for space with continue terrain and culture handling out of bounds requests."""
from __future__ import annotations

import unittest
from pathlib import Path
from typing import TYPE_CHECKING

from ret.space.culture import Culture
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from ret.testing.mocks import MockAgent
from parameterized import parameterized_class
from tests.test_space import TEST_AGENTS
from testsret.test_space.test_space import H_0, H_255, get_test_name

if TYPE_CHECKING:
    from ret.types import Coordinate2d

culture_red = Culture("red culture")
culture_green = Culture("green culture")
culture_blue = Culture("blue culture")
culture_yellow = Culture("yellow culture")


@parameterized_class(
    ("space_type", "pos"),
    [
        ("2d", (-31, -51)),
        ("2d", (20, -51)),
        ("2d", (71, -51)),
        ("2d", (71, 0)),
        ("2d", (71, 51)),
        ("2d", (20, 51)),
        ("2d", (-31, 51)),
        ("2d", (-31, 0)),
        ("3d", (-31, -51, 0)),
        ("3d", (20, -51, 0)),
        ("3d", (71, -51, 0)),
        ("3d", (71, 0, 0)),
        ("3d", (71, 51, 0)),
        ("3d", (20, 51, 0)),
        ("3d", (-31, 51, 0)),
        ("3d", (-31, 0, 0)),
    ],
    class_name_func=get_test_name,
)
class TestTerrainAndCultureOutOfBounds(unittest.TestCase):
    """Test continuous space with terrain and culture handles out of bounds requests."""

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
            terrain_image_path=str(Path(__file__).parent.joinpath("TestTerrain.png")),
            height_black=H_0,
            height_white=H_255,
            culture_image_path=str(Path(__file__).parent.joinpath("TestCulture.png")),
            culture_dictionary={
                (255, 0, 0): culture_red,
                (0, 255, 0): culture_green,
                (0, 0, 255): culture_blue,
                (255, 255, 0): culture_yellow,
            },
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
            terrain_image_path=str(Path(__file__).parent.joinpath("TestTerrain.png")),
            height_black=H_0,
            height_white=H_255,
            culture_image_path=str(Path(__file__).parent.joinpath("TestCulture.png")),
            culture_dictionary={
                (255, 0, 0): culture_red,
                (0, 255, 0): culture_green,
                (0, 0, 255): culture_blue,
                (255, 255, 0): culture_yellow,
            },
        )
        self.agents = []
        for i, pos in enumerate(TEST_AGENTS):
            a = MockAgent(i, (0, 0, 0))
            self.agents.append(a)
            pos_3d = [p for p in pos] + [0]
            self.space.place_agent(a, pos_3d)  # type: ignore

    def test_terrain_height(self):
        """Test height out of bounds throws an exception."""
        with self.assertRaises(Exception) as e:
            self.space.get_terrain_height(self.pos)
        self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))

    def test_terrain_gradient(self):
        """Test gradient out of bounds throws an exception."""
        with self.assertRaises(Exception) as e:
            self.space.get_terrain_gradient(self.pos)
        self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))

    def test_terrain_gradient_along_vec(self):
        """Test gradient along vector out of bounds throws an exception."""
        with self.assertRaises(Exception) as e:
            self.space.get_terrain_gradient_along_vec(self.pos, (1, 0))
        self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))

    def test_culture(self):
        """Test culture out of bounds throws an exception."""
        with self.assertRaises(Exception) as e:
            self.space.get_culture(self.pos)
        self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))
