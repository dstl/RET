"""Tests for features in space."""

import unittest

from mesa_ret.space.feature import BoxFeature, LineFeature
from mesa_ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)


class TestSpaceFeatures(unittest.TestCase):
    """Test features in space."""

    def setUp(self):
        """Set up test cases."""
        self.area_1 = BoxFeature((0, 0), (1, 1), "area 1")
        self.area_2 = BoxFeature((0, 0), (1, 1), "area 2")
        self.duplicate_area_1 = BoxFeature((0, 0), (1, 1), "area 1")

        self.boundary_1 = LineFeature((0, 0), (1, 1), "boundary 1")
        self.boundary_2 = LineFeature((0, 0), (1, 1), "boundary 2")
        self.duplicate_boundary_1 = LineFeature((0, 0), (1, 1), "boundary 1")

    def test_areas_2d(self):
        """Test areas in a 2d space."""
        space = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=10, y_max=10, features=[self.area_1, self.area_2]
        )

        assert len(space.areas) == 2
        assert "area 1" in space.areas
        assert space.areas["area 1"].name == "area 1"
        assert "area 2" in space.areas
        assert space.areas["area 2"].name == "area 2"

        assert len(space.boundaries) == 2
        assert "area 1" in space.boundaries
        assert space.boundaries["area 1"].name == "area 1"
        assert "area 2" in space.boundaries
        assert space.boundaries["area 2"].name == "area 2"

    def test_boundaries_2d(self):
        """Test boundaries in a 2d space."""
        space = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=10, y_max=10, features=[self.boundary_1, self.boundary_2]
        )

        assert len(space.boundaries) == 2
        assert "boundary 1" in space.boundaries
        assert space.boundaries["boundary 1"].name == "boundary 1"
        assert "boundary 2" in space.boundaries
        assert space.boundaries["boundary 2"].name == "boundary 2"

        assert len(space.areas) == 0

    def test_invalid_areas_2d(self):
        """Test invalid areas in a 2d space."""
        with self.assertRaises(ValueError) as e:
            ContinuousSpaceWithTerrainAndCulture2d(
                x_max=10, y_max=10, features=[self.area_1, self.duplicate_area_1]
            )
        self.assertEqual("All provided areas must have unique names", str(e.exception))

    def test_invalid_boundaries_2d(self):
        """Test invalid boundaries in a 2d space."""
        with self.assertRaises(ValueError) as e:
            ContinuousSpaceWithTerrainAndCulture2d(
                x_max=10, y_max=10, features=[self.boundary_1, self.duplicate_boundary_1]
            )
        self.assertEqual("All provided boundaries must have unique names", str(e.exception))

    def test_areas_3d(self):
        """Test areas in a 3d space."""
        space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=10, y_max=10, features=[self.area_1, self.area_2]
        )

        assert len(space.areas) == 2
        assert "area 1" in space.areas
        assert space.areas["area 1"].name == "area 1"
        assert "area 2" in space.areas
        assert space.areas["area 2"].name == "area 2"

        assert len(space.boundaries) == 2
        assert "area 1" in space.boundaries
        assert space.boundaries["area 1"].name == "area 1"
        assert "area 2" in space.boundaries
        assert space.boundaries["area 2"].name == "area 2"

    def test_boundaries_3d(self):
        """Test boundaries in a 3d space."""
        space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=10, y_max=10, features=[self.boundary_1, self.boundary_2]
        )

        assert len(space.boundaries) == 2
        assert "boundary 1" in space.boundaries
        assert space.boundaries["boundary 1"].name == "boundary 1"
        assert "boundary 2" in space.boundaries
        assert space.boundaries["boundary 2"].name == "boundary 2"

        assert len(space.areas) == 0

    def test_invalid_areas_3d(self):
        """Test invalid areas in a 3d space."""
        with self.assertRaises(ValueError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=10, y_max=10, features=[self.area_1, self.duplicate_area_1]
            )
        self.assertEqual("All provided areas must have unique names", str(e.exception))

    def test_invalid_boundaries_3d(self):
        """Test invalid boundaries in a 3d space."""
        with self.assertRaises(ValueError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=10, y_max=10, features=[self.boundary_1, self.duplicate_boundary_1]
            )
        self.assertEqual("All provided boundaries must have unique names", str(e.exception))
