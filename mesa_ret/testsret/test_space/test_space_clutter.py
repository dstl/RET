"""Tests for clutter field in space."""

import unittest

from mesa_ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from mesa_ret.testing.mocks import MockAgent


class TestSpaceClutter(unittest.TestCase):
    """Test clutter in various spaces."""

    def setUp(self):
        """Set up test cases."""
        self.mockAgent = MockAgent(1, (0, 0, 0))

    def test_no_ground_clutter_2d(self):
        """Test clutter in 2D space with no ground."""
        space = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=10, y_max=10, clutter_background_level=10
        )

        assert len(space.clutter_field._modifiers) == 0
        assert space.clutter_field._background_level == 10
        assert space.clutter_field.get_value((0, 0), self.mockAgent) == 10

    def test_no_ground_clutter_3d(self):
        """Test clutter in 3D space with no ground."""
        space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=10, y_max=10, clutter_background_level=10
        )

        assert len(space.clutter_field._modifiers) == 0
        assert space.clutter_field._background_level == 10
        assert space.clutter_field.get_value((0, 0, 0), self.mockAgent) == 10

    def test_with_ground_clutter_2d(self):
        """Test clutter in 2D space with ground terrain."""
        space = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=10,
            y_max=10,
            clutter_background_level=10,
            ground_clutter_value=10,
            ground_clutter_height=10,
        )

        assert len(space.clutter_field._modifiers) == 1
        assert space.clutter_field._background_level == 10
        assert space.clutter_field.get_value((0, 0), self.mockAgent) == 20

    def test_with_ground_clutter_3d(self):
        """Test clutter in 3D space with ground terrain."""
        space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=10,
            y_max=10,
            clutter_background_level=10,
            ground_clutter_value=10,
            ground_clutter_height=10,
        )

        assert len(space.clutter_field._modifiers) == 1
        assert space.clutter_field._background_level == 10
        assert space.clutter_field.get_value((0, 0, 0), self.mockAgent) == 20
        assert space.clutter_field.get_value((0, 0, 20), self.mockAgent) == 10

    def test_with_ground_clutter_no_height_2d(self):
        """Test clutter with no height in 2D space with ground terrain is forbidden."""
        with self.assertRaises(TypeError) as e:
            ContinuousSpaceWithTerrainAndCulture2d(
                x_max=10,
                y_max=10,
                clutter_background_level=10,
                ground_clutter_value=10,
            )
        self.assertEqual(
            "Must provide ground clutter height if providing ground clutter value",
            str(e.exception),
        )

    def test_with_ground_clutter_no_height_3d(self):
        """Test clutter with no height in 2D space with ground terrain is forbidden."""
        with self.assertRaises(TypeError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=10,
                y_max=10,
                clutter_background_level=10,
                ground_clutter_value=10,
            )
        self.assertEqual(
            "Must provide ground clutter height if providing ground clutter value",
            str(e.exception),
        )
