"""Tests for space with terrain on different scales."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from parameterized import parameterized, parameterized_class

if TYPE_CHECKING:
    from typing import Any

    from ret.types import Coordinate2d


def get_test_name(cls: type, num: int, params_dict: dict[str, Any]) -> str:
    """Return a safe test name based on parametrised properties.

    Args:
        cls (type): Class under test
        num (int): Test number
        params_dict (dict[str, Any]): Parameters dictionary.

    Returns:
        str: String representation of test name
    """
    return "%s_%s_%s_(%s, %s)_%s" % (
        cls.__name__,
        num,
        parameterized.to_safe_name(str(params_dict["space_type"])),
        parameterized.to_safe_name(str(params_dict["pos"][0])),
        parameterized.to_safe_name(str(params_dict["pos"][1])),
        parameterized.to_safe_name(str(params_dict["terrain_file_path"])),
    )


@parameterized_class(
    (
        "space_type",
        "pos",
        "expected_x_gradient_value",
        "expected_y_gradient_value",
        "terrain_file_path",
    ),
    [
        (
            "2d",
            (0, 0),
            0,
            -1 * ((70 - -30) / 255),
            "terrain_pixel_grid_top_to_bottom.png",
        ),
        (
            "2d",
            (0, 0),
            1 * ((70 - -30) / 255),
            0,
            "terrain_pixel_grid_left_to_right.png",
        ),
        (
            "2d",
            (30, 50),
            0,
            -1 * ((70 - -30) / 255),
            "terrain_pixel_grid_top_to_bottom.png",
        ),
        (
            "2d",
            (30, 50),
            1 * ((70 - -30) / 255),
            0,
            "terrain_pixel_grid_left_to_right.png",
        ),
        (
            "2d",
            (128, 128),
            0,
            -1 * ((70 - -30) / 255),
            "terrain_pixel_grid_top_to_bottom.png",
        ),
        (
            "2d",
            (128, 128),
            1 * ((70 - -30) / 255),
            0,
            "terrain_pixel_grid_left_to_right.png",
        ),
        (
            "3d",
            (0, 0),
            0,
            -1 * ((70 - -30) / 255),
            "terrain_pixel_grid_top_to_bottom.png",
        ),
        (
            "3d",
            (0, 0),
            1 * ((70 - -30) / 255),
            0,
            "terrain_pixel_grid_left_to_right.png",
        ),
        (
            "3d",
            (30, 50),
            0,
            -1 * ((70 - -30) / 255),
            "terrain_pixel_grid_top_to_bottom.png",
        ),
        (
            "3d",
            (30, 50),
            1 * ((70 - -30) / 255),
            0,
            "terrain_pixel_grid_left_to_right.png",
        ),
        (
            "3d",
            (128, 128),
            0,
            -1 * ((70 - -30) / 255),
            "terrain_pixel_grid_top_to_bottom.png",
        ),
        (
            "3d",
            (128, 128),
            1 * ((70 - -30) / 255),
            0,
            "terrain_pixel_grid_left_to_right.png",
        ),
    ],
    class_name_func=get_test_name,
)
class TestTerrainOnDifferentScales(unittest.TestCase):
    """Test terrain on different scales."""

    space_type: str
    pos: Coordinate2d
    expected_x_gradient_value: float
    expected_y_gradient_value: float
    terrain_file_path: str

    max_x = 256
    max_y = 256

    def setUp(self):
        """Set up the given space."""
        if self.space_type == "2d":
            self.setup_2d()
        elif self.space_type == "3d":
            self.setup_3d()

    def setup_2d(self):
        """Create a test 2d space and populate with Mock Agents."""
        self.smallSpace = ContinuousSpaceWithTerrainAndCulture2d(
            self.max_x,
            self.max_y,
            x_min=0,
            y_min=0,
            terrain_image_path=str(Path(__file__).parent.joinpath(self.terrain_file_path)),
            height_black=-30,
            height_white=70,
        )
        self.largeSpace = ContinuousSpaceWithTerrainAndCulture2d(
            2 * self.max_x,
            2 * self.max_y,
            x_min=0,
            y_min=0,
            terrain_image_path=str(Path(__file__).parent.joinpath(self.terrain_file_path)),
            height_black=-30,
            height_white=70,
        )

    def setup_3d(self):
        """Create a test 3d space and populate with Mock Agents."""
        self.smallSpace = ContinuousSpaceWithTerrainAndCulture3d(
            self.max_x,
            self.max_y,
            x_min=0,
            y_min=0,
            terrain_image_path=str(Path(__file__).parent.joinpath(self.terrain_file_path)),
            height_black=-30,
            height_white=70,
        )
        self.largeSpace = ContinuousSpaceWithTerrainAndCulture3d(
            2 * self.max_x,
            2 * self.max_y,
            x_min=0,
            y_min=0,
            terrain_image_path=str(Path(__file__).parent.joinpath(self.terrain_file_path)),
            height_black=-30,
            height_white=70,
        )

    def test_gradient_direction(self):
        """Test whether the gradients are in the right direction for this setup."""
        assert self.smallSpace.get_terrain_gradient_along_vec(self.pos, (1, 0)) >= 0
        assert self.smallSpace.get_terrain_gradient_along_vec(self.pos, (0, 1)) <= 0
        assert self.largeSpace.get_terrain_gradient_along_vec(self.pos, (1, 0)) >= 0
        assert self.largeSpace.get_terrain_gradient_along_vec(self.pos, (0, 1)) <= 0

    def test_gradient_normal_small_space(self):
        """Test whether the gradient normals on the small space is as expected."""
        assert math.isclose(
            self.smallSpace.get_terrain_gradient(self.pos)[0],
            self.expected_x_gradient_value,
        )
        assert math.isclose(
            self.smallSpace.get_terrain_gradient(self.pos)[1],
            self.expected_y_gradient_value,
        )

    def test_gradient_normal_large_space(self):
        """Test whether the gradient normals on the large space is as expected."""
        assert math.isclose(
            self.largeSpace.get_terrain_gradient(self.pos)[0],
            self.expected_x_gradient_value / 2,
        )
        assert math.isclose(
            self.largeSpace.get_terrain_gradient(self.pos)[1],
            self.expected_y_gradient_value / 2,
        )

    def test_gradient_exactly_small_space(self):
        """Test whether the gradient on the small space is as expected."""
        assert math.isclose(
            self.smallSpace.get_terrain_gradient_along_vec(self.pos, (1, 0)),
            self.expected_x_gradient_value,
        )
        assert math.isclose(
            self.smallSpace.get_terrain_gradient_along_vec(self.pos, (0, 1)),
            self.expected_y_gradient_value,
        )

    def test_gradient_exactly_large_space(self):
        """Test whether the gradient on the large space is as expected."""
        assert math.isclose(
            self.largeSpace.get_terrain_gradient_along_vec(self.pos, (1, 0)),
            self.expected_x_gradient_value / 2,
        )
        assert math.isclose(
            self.largeSpace.get_terrain_gradient_along_vec(self.pos, (0, 1)),
            self.expected_y_gradient_value / 2,
        )

    def test_gradient_scales(self):
        """Test gradient scaling.

        Test whether the gradient scales to one half when the model is scaled up by a
        factor of two.
        """
        assert math.isclose(
            self.smallSpace.get_terrain_gradient_along_vec(self.pos, (1, 0)),
            2 * self.largeSpace.get_terrain_gradient_along_vec(self.pos, (1, 0)),
        )
        assert math.isclose(
            self.smallSpace.get_terrain_gradient_along_vec(self.pos, (0, 1)),
            2 * self.largeSpace.get_terrain_gradient_along_vec(self.pos, (0, 1)),
        )
