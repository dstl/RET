"""Tests for height layers."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

from ret.space.heightband import AbsoluteHeightBand, RelativeHeightBand
from ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
from parameterized import parameterized, parameterized_class
from testsret.test_space.test_space import H_0, H_127, H_191, H_255

if TYPE_CHECKING:
    from typing import Any

    from ret.types import Coordinate2d


def get_test_name(cls: type, num: int, params_dict: dict[str, Any]) -> str:
    """Convert parametrised test case settings to string.

    Args:
        cls (type): Class under test
        num (int): Test number
        params_dict (dict[str, Any]): Test parameters

    Returns:
        str: Test name
    """
    return "%s_%s_(%s, %s)" % (
        cls.__name__,
        num,
        parameterized.to_safe_name(str(params_dict["pos"][0])),
        parameterized.to_safe_name(str(params_dict["pos"][1])),
    )


@parameterized_class(
    (
        "pos",
        "terrain_height",
    ),
    [((-5, -25), H_191), ((45, -25), H_127), ((45, 25), H_255), ((-5, 25), H_0)],
    class_name_func=get_test_name,
)
class TestHeightBands(unittest.TestCase):
    """Tests for height bands."""

    pos: Coordinate2d
    terrain_height: float

    def setUp(self):
        """Create a test space and populate with Mock Agents."""
        terrain = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=70,
            y_max=50,
            x_min=-30,
            y_min=-50,
            terrain_image_path=str(Path(__file__).parent.joinpath("TestTerrain.png")),
            height_black=H_0,
            height_white=H_255,
        )

        self.absBand1 = AbsoluteHeightBand("0m", 0)
        self.absBand2 = AbsoluteHeightBand("-100m", -100)
        self.absBand3 = AbsoluteHeightBand("100m", 100)

        self.relBand1 = RelativeHeightBand("0m (rel)", 0, terrain)
        self.relBand2 = RelativeHeightBand("-100m (rel)", -100, terrain)
        self.relBand3 = RelativeHeightBand("100m (rel)", 100, terrain)

    def test_absolute_height_bands(self):
        """Test height is as expected."""
        assert math.isclose(
            self.absBand1.get_height(self.pos),
            0,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
        assert math.isclose(
            self.absBand2.get_height(self.pos),
            -100,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
        assert math.isclose(
            self.absBand3.get_height(self.pos),
            +100,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )

    def test_relative_height_bands(self):
        """Test height is as expected."""
        assert math.isclose(
            self.relBand1.get_height(self.pos),
            self.terrain_height,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
        assert math.isclose(
            self.relBand2.get_height(self.pos),
            self.terrain_height - 100,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
        assert math.isclose(
            self.relBand3.get_height(self.pos),
            self.terrain_height + 100,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
