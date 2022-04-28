"""Test line of sight."""
import unittest
from pathlib import Path

from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture3d


class TestLineOfSightNoTerrain(unittest.TestCase):
    """Testing line of sight calculations where no Terrain map has been specified."""

    position_a = (10.0, 10.0, 1.0)
    position_b = (10.0, 20.0, 1.0)

    def setUp(self):
        """Create a test space."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            x_min=0,
            y_min=0,
        )

    def test_line_of_sight_no_terrain(self):
        """Test correct line of sight result given no terrain map has been specified."""
        self.assertTrue(self.space.check_line_of_sight(self.position_a, self.position_b, 1))


class TestLineOfSightWithTerrain(unittest.TestCase):
    """Testing line of sight calculations across a number of different terrain cases."""

    position_a = (10.0, 10.0, 1.0)
    position_b = (10.0, 20.0, 1.0)
    position_c = (90.0, 10.0, 1.0)
    position_d = (10.0, 10.0, 10.0)

    position_off_a = (-10.0, -10.0, 1.0)
    position_off_b = (110.0, -10.0, 1.0)

    def setUp(self):
        """Create a test space."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            x_min=0,
            y_min=0,
            terrain_image_path=str(Path(__file__).parent.joinpath("TestLineOfSight.png")),
        )

    def test_no_obstacle(self):
        """Test no obstacles in line of sight.

        The correct line of sight result is calculated given two positions at same
        height on a flat plane.
        """
        self.assertTrue(self.space.check_line_of_sight(self.position_a, self.position_b, 1))
        self.assertTrue(self.space.check_line_of_sight(self.position_b, self.position_a, 1))

    def test_default_sample_distance(self):
        """Test line of sight within sample distance.

        Test correct line of sight result given two positions at same height on a flat
        plane, using the default sample distance.
        """
        self.assertTrue(self.space.check_line_of_sight(self.position_a, self.position_b))
        self.assertTrue(self.space.check_line_of_sight(self.position_b, self.position_a))

    def test_terrain_obstacle(self):
        """Test obstacles in line of sight.

        Test correct line of sight result given two positions at same height with
        terrain between them that blocks line of sight.
        """
        self.assertFalse(self.space.check_line_of_sight(self.position_a, self.position_c, 1))
        self.assertFalse(self.space.check_line_of_sight(self.position_c, self.position_a, 1))

    def test_vertically_stacked_positions(self):
        """Test line of sight for vertically stacked positions.

        Test correct line of sight result given two positions that are directly above
        or below one another.
        """
        self.assertTrue(self.space.check_line_of_sight(self.position_a, self.position_d, 1))
        self.assertTrue(self.space.check_line_of_sight(self.position_d, self.position_a, 1))

    def test_distance_smaller_than_sample_distance(self):
        """Test distance below sample distance.

        Test correct line of sight result given two positions that are closer to one
        another than the sampling distance.
        """
        self.assertTrue(self.space.check_line_of_sight(self.position_a, self.position_a, 1))

    def test_off_terrain_positions(self):
        """Test line of sight exception handling."""
        with self.assertRaises(Exception) as e:
            self.space.check_line_of_sight(self.position_off_a, self.position_off_b, 1)
        self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))
