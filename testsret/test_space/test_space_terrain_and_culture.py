"""Tests for space with terrain and culture."""

from __future__ import annotations

import math
import statistics
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ret.space.culture import Culture
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from ret.testing.mocks import MockAgent
from parameterized import parameterized_class
from tests.test_space import TEST_AGENTS
from testsret.test_space.test_space import H_0, H_127, H_191, H_255, TEST_AGENTS_3D, get_test_name

if TYPE_CHECKING:
    from ret.types import Coordinate2d

culture_red = Culture("red")
culture_green = Culture("green")
culture_blue = Culture("blue")
culture_yellow = Culture("yellow")


@parameterized_class(
    (
        "space_type",
        "pos",
        "expected_height",
        "expected_x_gradient_dir",
        "expected_y_gradient_dir",
        "expected_culture",
    ),
    [
        ("2d", (-30.0, -50.0), H_191, 0, 0, culture_green),
        ("2d", (20, -50), statistics.mean([H_191, H_127]), -1, 0, culture_red),
        ("2d", (69, -50), H_127, 0, 0, culture_red),
        ("2d", (69, 0), statistics.mean([H_127, H_255]), 0, 1, culture_blue),
        ("2d", (69.9, 49.9), H_255, 0, 0, culture_blue),
        ("2d", (20, 49), statistics.mean([H_255, H_0]), 1, 0, culture_blue),
        ("2d", (-30, 49), H_0, 0, 0, culture_yellow),
        ("2d", (-30, 0), statistics.mean([H_0, H_191]), 0, -1, culture_yellow),
        ("2d", (-5, -25), H_191, 0, 0, culture_green),
        ("2d", (20, -25), statistics.mean([H_191, H_127]), -1, 0, culture_red),
        ("2d", (45, -25), H_127, 0, 0, culture_red),
        ("2d", (45, 0), statistics.mean([H_127, H_255]), 0, 1, culture_blue),
        ("2d", (45, 25), H_255, 0, 0, culture_blue),
        ("2d", (20, 25), statistics.mean([H_255, H_0]), 1, 0, culture_blue),
        ("2d", (-5, 25), H_0, 0, 0, culture_yellow),
        ("2d", (-5, 0), statistics.mean([H_0, H_191]), 0, -1, culture_yellow),
        (
            "2d",
            (20, 0),
            statistics.mean([H_0, H_127, H_191, H_255]),
            1,
            -1,
            culture_blue,
        ),
        ("2d", (0, 0), statistics.mean([H_0, H_191]), 0, -1, culture_yellow),
        ("3d", (-30.0, -50.0), H_191, 0, 0, culture_green),
        ("3d", (20, -50), statistics.mean([H_191, H_127]), -1, 0, culture_red),
        ("3d", (69, -50), H_127, 0, 0, culture_red),
        ("3d", (69, 0), statistics.mean([H_127, H_255]), 0, 1, culture_blue),
        ("3d", (69.9, 49.9), H_255, 0, 0, culture_blue),
        ("3d", (20, 49), statistics.mean([H_255, H_0]), 1, 0, culture_blue),
        ("3d", (-30, 49), H_0, 0, 0, culture_yellow),
        ("3d", (-30, 0), statistics.mean([H_0, H_191]), 0, -1, culture_yellow),
        ("3d", (-5, -25), H_191, 0, 0, culture_green),
        ("3d", (20, -25), statistics.mean([H_191, H_127]), -1, 0, culture_red),
        ("3d", (45, -25), H_127, 0, 0, culture_red),
        ("3d", (45, 0), statistics.mean([H_127, H_255]), 0, 1, culture_blue),
        ("3d", (45, 25), H_255, 0, 0, culture_blue),
        ("3d", (20, 25), statistics.mean([H_255, H_0]), 1, 0, culture_blue),
        ("3d", (-5, 25), H_0, 0, 0, culture_yellow),
        ("3d", (-5, 0), statistics.mean([H_0, H_191]), 0, -1, culture_yellow),
        (
            "3d",
            (20, 0),
            statistics.mean([H_0, H_127, H_191, H_255]),
            1,
            -1,
            culture_blue,
        ),
        ("3d", (0, 0), statistics.mean([H_0, H_191]), 0, -1, culture_yellow),
    ],
    class_name_func=get_test_name,
)
class TestTerrainAndCulture(unittest.TestCase):
    """Testing a continuous space with terrain and culture."""

    space_type: str
    pos: Coordinate2d
    expected_height: float
    expected_x_gradient_dir: float
    expected_y_gradient_dir = float
    expected_culture: Culture
    space: ContinuousSpaceWithTerrainAndCulture2d

    def setUp(self):
        """Set up the given space."""
        if self.space_type == "2d":
            self.setup_2d()
        elif self.space_type == "3d":
            self.setup_3d()

    def setup_2d(self):
        """Create a 2d test space and populate with Mock Agents."""
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
            a = MockAgent(i, None)
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
        for i, pos in enumerate(TEST_AGENTS_3D):
            a = MockAgent(i, None)
            self.agents.append(a)
            self.space.place_agent(a, pos)

    def test_terrain_height(self):
        """Test height is as expected."""
        assert isinstance(self.space.get_terrain_height(self.pos), np.float64)
        assert math.isclose(
            self.space.get_terrain_height(self.pos),
            self.expected_height,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )

    def test_terrain_gradient(self):
        """Test gradient is as expected."""
        gradient = self.space.get_terrain_gradient(self.pos)
        x_gradient = gradient[0]
        y_gradient = gradient[1]

        if self.expected_x_gradient_dir == 0:
            assert math.isclose(x_gradient, 0, abs_tol=1e-9)
        else:
            assert np.sign(x_gradient) == np.sign(self.expected_x_gradient_dir)

        if self.expected_y_gradient_dir == 0:
            assert math.isclose(y_gradient, 0, abs_tol=1e-9)
        else:
            assert np.sign(y_gradient) == np.sign(self.expected_y_gradient_dir)

    def test_terrain_gradient_along_vec(self):
        """Test gradient along vector is as expected."""
        x_gradient = self.space.get_terrain_gradient_along_vec(self.pos, (1, 0))
        y_gradient = self.space.get_terrain_gradient_along_vec(self.pos, (0, 1))

        if self.expected_x_gradient_dir == 0:
            assert math.isclose(x_gradient, 0, abs_tol=1e-9)
        else:
            assert np.sign(x_gradient) == np.sign(self.expected_x_gradient_dir)

        if self.expected_y_gradient_dir == 0:
            assert math.isclose(y_gradient, 0, abs_tol=1e-9)
        else:
            assert np.sign(y_gradient) == np.sign(self.expected_y_gradient_dir)

    def test_culture(self):
        """Test culture is as expected."""
        assert self.space.get_culture(self.pos) == self.expected_culture
