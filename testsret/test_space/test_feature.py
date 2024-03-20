"""Tests for features."""
from __future__ import annotations

import random
import unittest
from typing import TYPE_CHECKING

from ret.space.feature import (
    BoxFeature,
    CompoundAreaFeature,
    LineFeature,
    LineFeatureWithWidth,
    MultiLineFeature,
    MultiLineFeatureWithWidth,
    SphereFeature,
)
from parameterized import parameterized

if TYPE_CHECKING:
    from ret.types import Coordinate2d, Coordinate3d


class TestFeatures(unittest.TestCase):
    """Test cases for features."""

    @parameterized.expand(
        [
            [(0, 10), (10, 0), True],  # crossing
            [(0, 1), (0, 11), False],  # parallel
            [(11, 11), (12, 12), False],  # extension
            [(0, 10), (5, 5), True],  # touching
            [(0, 10), (0, 5), False],  # not crossing
        ]
    )
    def test_line(self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool):
        """Test line.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the crossed has been crossed
        """
        line = LineFeature((0, 0), (10, 10), "line")

        assert line.has_crossed(coord_1, coord_2) == expected

    @parameterized.expand(
        [
            [(0, 10), (10, 0), True],  # crossing
            [(0, 1), (0, 11), False],  # parallel
            [(11, 11), (12, 12), False],  # extension
            [(0, 10), (5, 5), True],  # touching
            [(0, 10), (0, 5), False],  # not crossing
            [(0, 5), (20, 5), False],  # crosses twice
            [(0, 5), (30, 5), True],  # crosses three times
        ]
    )
    def test_multi_line(self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool):
        """Test multi-line.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the crossed has been crossed an odd number of times
        """
        line = MultiLineFeature([(0, 0), (10, 10), (20, 0), (30, 10)], "line")
        assert line.has_crossed(coord_1, coord_2) == expected

    @parameterized.expand(
        [
            [(10, 10), True],
            [(10, 10), True],
            [(10, 0), True],
            [(0, 10), True],
            [(5, 5), True],
            [(0, 0), False],
            [(10, 100), False],
            [(100, 10), False],
        ]
    )
    def test_sphere_2d_contains_in_2d(self, coord: Coordinate2d, expected: bool):
        """Test a circular area in 2d.

        Args:
            coord (Coordinate2d): Coordinate to test
            expected (bool): Whether or not area contains coordinate
        """
        area = SphereFeature((10, 10), 10, "area")

        assert area.contains(coord) == expected

    @parameterized.expand(
        [
            [(0, 0), (0, 0), False],
            [(0, 0), (10, 10), True],
            [(10, 10), (0, 0), False],
            [(10, 10), (10, 10), False],
        ]
    )
    def test_sphere_2d_has_crossed_in_2d(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test a circular area in 2d.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the sphere has been crossed
        """
        area = SphereFeature((10, 10), 10, "area")

        assert area.has_crossed(coord_1, coord_2) == expected

    @parameterized.expand(
        [
            [(10, 10, -10), True],
            [(10, 10, 0), True],
            [(10, 10, 10), True],
            [(10, 10, -10), True],
            [(10, 10, 0), True],
            [(10, 10, 10), True],
            [(10, 0, -10), True],
            [(10, 0, 0), True],
            [(10, 0, 10), True],
            [(0, 10, -10), True],
            [(0, 10, 0), True],
            [(0, 10, 10), True],
            [(5, 5, -10), True],
            [(5, 5, 0), True],
            [(5, 5, 10), True],
            [(0, 0, -10), False],
            [(0, 0, 0), False],
            [(0, 0, 10), False],
            [(10, 100, -10), False],
            [(10, 100, 0), False],
            [(10, 100, 10), False],
            [(100, 10, -10), False],
            [(100, 10, 0), False],
            [(100, 10, 10), False],
        ]
    )
    def test_sphere_2d_contains_in_3d(self, coord: Coordinate3d, expected: bool):
        """Test a circular area in 3d.

        Args:
            coord (Coordinate3d): Coordinate to test
            expected (bool): Whether or not area contains coordinate
        """
        area = SphereFeature((10, 10), 10, "area")

        assert area.contains(coord) == expected

    @parameterized.expand(
        [
            [(0, 0, 0), (0, 0, 0), False],
            [(0, 0, 0), (10, 10, 0), True],
            [(10, 10, 0), (0, 0, 0), False],
            [(10, 10, 0), (10, 10, 0), False],
        ]
    )
    def test_sphere_2d_has_crossed_in_3d(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test a circular area in 3d.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the sphere has been crossed
        """
        area = SphereFeature((10, 10), 10, "area")

        assert area.has_crossed(coord_1, coord_2) == expected

    def test_sphere_2d_random_coord_inside(self):
        """Test a circular area can generate points inside itself."""
        area = SphereFeature((10, 10), 10, "area")
        random_generator = random.Random()

        for _ in range(100):
            coord = area.get_coord_inside(random_generator)
            assert len(coord) == 2
            assert area.contains(coord)

    @parameterized.expand(
        [
            [(10, 10, 10), True],
            [(10, 10, 0), True],
            [(10, 0, 10), True],
            [(0, 10, 10), True],
            [(5, 5, 5), True],
            [(0, 0, 0), False],
            [(10, 10, 100), False],
            [(10, 100, 10), False],
            [(100, 10, 10), False],
            [(10, 100, 100), False],
            [(100, 10, 100), False],
            [(100, 100, 10), False],
        ]
    )
    def test_sphere_3d_contains(self, coord: Coordinate3d, expected: bool):
        """Test a spherical area.

        Args:
            coord (Coordinate3d): Coordinate to test
            expected (bool): Whether or not area contains coordinate
        """
        area = SphereFeature((10, 10, 10), 10, "area")

        assert area.contains(coord) == expected

    @parameterized.expand(
        [
            [(0, 0, 0), (0, 0, 0), False],
            [(0, 0, 0), (10, 10, 10), True],
            [(10, 10, 10), (0, 0, 0), False],
            [(10, 10, 10), (10, 10, 10), False],
        ]
    )
    def test_sphere_3d_has_crossed(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test a circular area in 3d.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the sphere has been crossed
        """
        area = SphereFeature((10, 10, 10), 10, "area")

        assert area.has_crossed(coord_1, coord_2) == expected

    def test_sphere_3d_random_coord_inside(self):
        """Test a spherical area can generate points inside itself."""
        area = SphereFeature((10, 10, 10), 10, "area")
        random_generator = random.Random()

        for _ in range(100):
            coord = area.get_coord_inside(random_generator)
            assert len(coord) == 3
            assert area.contains(coord)

    def test_sphere_invalid(self):
        """Test that sphere can't be created with invalid inputs."""
        with self.assertRaises(ValueError) as e:
            SphereFeature((10, 10), -1, "area")
        self.assertEqual("Radius must be positive", str(e.exception))

        with self.assertRaises(ValueError) as e:
            SphereFeature((10, 10, 10), -1, "area")
        self.assertEqual("Radius must be positive", str(e.exception))

    @parameterized.expand(
        [
            [(10, 10), True],
            [(5, 10), True],
            [(5, 5), True],
            [(10, 5), True],
            [(7, 7), True],
            [(0, 0), False],
            [(0, 7), False],
            [(7, 0), False],
            [(12, 7), False],
            [(7, 12), False],
            [(12, 12), False],
        ]
    )
    def test_box_2d_contains_in_2d(self, coord: Coordinate2d, expected: bool):
        """Test a 2D box area in 2d.

        Args:
            coord (Coordinate2d): Coordinate to test
            expected (bool): Whether or not area contains coordinate
        """
        area = BoxFeature((5, 5), (10, 10), "area")

        assert area.contains(coord) == expected

    @parameterized.expand(
        [
            [(0, 0), (0, 0), False],
            [(0, 0), (7, 7), True],
            [(7, 7), (0, 0), False],
            [(7, 7), (7, 7), False],
        ]
    )
    def test_box_2d_has_crossed_in_2d(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test a 2D box area in 2d.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the box has been crossed
        """
        area = BoxFeature((5, 5), (10, 10), "area")

        assert area.has_crossed(coord_1, coord_2) == expected

    @parameterized.expand(
        [
            [(10, 10, 10), True],
            [(5, 10, 10), True],
            [(10, 5, 10), True],
            [(10, 10, 5), True],
            [(10, 5, 5), True],
            [(5, 5, 10), True],
            [(5, 5, 5), True],
            [(7, 7, 7), True],
            [(0, 0, 0), False],
            [(0, 7, 7), False],
            [(7, 0, 7), False],
            [(7, 7, 0), True],
            [(12, 7, 7), False],
            [(7, 12, 7), False],
            [(7, 7, 12), True],
            [(12, 12, 12), False],
        ]
    )
    def test_box_2d_contains_in_3d(self, coord: Coordinate3d, expected: bool):
        """Test a 2D box area in 3d.

        Args:
            coord (Coordinate3d): Coordinate to test
            expected (bool): Whether or not area contains coordinate
        """
        area = BoxFeature((5, 5), (10, 10), "area")

        assert area.contains(coord) == expected

    @parameterized.expand(
        [
            [(0, 0, 0), (0, 0, 0), False],
            [(0, 0, 0), (7, 7, 7), True],
            [(7, 7, 7), (0, 0, 0), False],
            [(7, 7, 7), (7, 7, 7), False],
        ]
    )
    def test_box_2d_has_crossed_in_3d(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test a 2D box area in 3d.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the box has been crossed
        """
        area = BoxFeature((5, 5), (10, 10), "area")

        assert area.has_crossed(coord_1, coord_2) == expected

    def test_box_2d_random_coord_inside(self):
        """Test a 2D box area can generate points inside itself."""
        area = BoxFeature((5, 5), (10, 10), "area")
        random_generator = random.Random()

        for _ in range(100):
            coord = area.get_coord_inside(random_generator)
            assert len(coord) == 2
            assert area.contains(coord)

    @parameterized.expand(
        [
            [(10, 10, 10), True],
            [(5, 10, 10), True],
            [(10, 5, 10), True],
            [(10, 10, 5), True],
            [(10, 5, 5), True],
            [(5, 5, 10), True],
            [(5, 5, 5), True],
            [(7, 7, 7), True],
            [(0, 0, 0), False],
            [(0, 7, 7), False],
            [(7, 0, 7), False],
            [(7, 7, 0), False],
            [(12, 7, 7), False],
            [(7, 12, 7), False],
            [(7, 7, 12), False],
            [(12, 12, 12), False],
        ]
    )
    def test_box_3d_contains(self, coord: Coordinate3d, expected: bool):
        """Test a 3D box area.

        Args:
            coord (Coordinate3d): Coordinate to test
            expected (bool): Whether or not area contains coordinate
        """
        area = BoxFeature((5, 5, 5), (10, 10, 10), "area")

        assert area.contains(coord) == expected

    @parameterized.expand(
        [
            [(0, 0, 0), (0, 0, 0), False],
            [(0, 0, 0), (7, 7, 7), True],
            [(7, 7, 7), (0, 0, 0), False],
            [(7, 7, 7), (7, 7, 7), False],
        ]
    )
    def test_box_3d_has_crossed(self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool):
        """Test a 3D box area in 3d.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the box has been crossed
        """
        area = BoxFeature((5, 5, 5), (10, 10, 10), "area")

        assert area.has_crossed(coord_1, coord_2) == expected

    def test_box_3d_random_coord_inside(self):
        """Test a 3D box area can generate points inside itself."""
        area = BoxFeature((5, 5, 5), (10, 10, 10), "area")
        random_generator = random.Random()

        for _ in range(100):
            coord = area.get_coord_inside(random_generator)
            assert len(coord) == 3
            assert area.contains(coord)

    def test_box_invalid(self):
        """Test that box can't be created with invalid inputs."""
        with self.assertRaises(ValueError) as e:
            BoxFeature((10, 10), (5, 5), "area")
        self.assertEqual(
            ("max_coord (5, 5) must be greater than min_coord (10, 10) in all " "directions"),
            str(e.exception),
        )

        with self.assertRaises(ValueError) as e:
            BoxFeature((10, 10, 10), (5, 5, 5), "area")
        self.assertEqual(
            (
                "max_coord (5, 5, 5) must be greater than min_coord (10, 10, 10) in "
                "all directions"
            ),
            str(e.exception),
        )

    @parameterized.expand(
        [
            [(0, 10), (10, 0), True],  # crossing
            [(0, 1), (0, 11), False],  # parallel
            [(11, 11), (12, 12), False],  # extension
            [(0, 10), (5, 5), True],  # touching
            [(0, 10), (0, 5), False],  # not crossing
        ]
    )
    def test_line_with_width_has_crossed(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test line with width has crossed method.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the line has been crossed
        """
        line = LineFeatureWithWidth((0, 0), (10, 10), 10, "line")

        assert line.has_crossed(coord_1, coord_2) == expected

    @parameterized.expand(
        [
            [(0, 0), True],
            [(5, 5), True],
            [(-1, -2), True],
            [(11, 11), True],
            [(5, 6), True],
            [(100, 100), False],
            [(5, 100), False],
        ]
    )
    def test_line_with_width_contains(self, coord: Coordinate2d, expected: bool):
        """Test line with width contains method.

        Args:
            coord (Coordinate2d): Coordinate to test
            expected (bool): Whether or not line contains coordinate
        """
        line = LineFeatureWithWidth((0, 0), (10, 10), 10, "line")

        assert line.contains(coord) == expected

    def test_line_with_width_random_coord_inside(self):
        """Test line with width contains method."""
        line = LineFeatureWithWidth((0, 0), (10, 10), 10, "line")
        random_generator = random.Random()

        for _ in range(100):
            coord = line.get_coord_inside(random_generator)
            assert len(coord) == 2
            assert line.contains(coord)

    @parameterized.expand(
        [
            [(0, 10), (10, 0), True],  # crossing
            [(0, 1), (0, 11), False],  # parallel
            [(11, 11), (12, 12), False],  # extension
            [(0, 10), (5, 5), True],  # touching
            [(0, 10), (0, 5), False],  # not crossing
            [(0, 5), (20, 5), False],  # crosses twice
            [(0, 5), (30, 5), True],  # crosses three times
        ]
    )
    def test_multi_line_with_width_has_crossed(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test multi-line with width has crossed method.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the line has been crossed
        """
        line = MultiLineFeatureWithWidth([(0, 0), (10, 10), (20, 0), (30, 10)], 10, "line")

        assert line.has_crossed(coord_1, coord_2) == expected

    @parameterized.expand(
        [
            [(0, 0), True],
            [(5, 5), True],
            [(-1, -2), True],
            [(11, 11), True],
            [(5, 6), True],
            [(100, 100), False],
            [(25, 5), True],
        ]
    )
    def test_multi_line_with_width_contains(self, coord: Coordinate2d, expected: bool):
        """Test multi-line with width contains method.

        Args:
            coord (Coordinate2d): Coordinate to test
            expected (bool): Whether or not line contains coordinate
        """
        line = MultiLineFeatureWithWidth([(0, 0), (10, 10), (20, 0), (30, 10)], 10, "line")

        assert line.contains(coord) == expected

    def test_multi_line_with_width_random_coord_inside(self):
        """Test multi-line with width contains method."""
        line = MultiLineFeatureWithWidth([(0, 0), (10, 10), (20, 0), (30, 10)], 10, "line")
        random_generator = random.Random()

        for _ in range(100):
            coord = line.get_coord_inside(random_generator)
            assert len(coord) == 2
            assert line.contains(coord)

    @parameterized.expand(
        [
            [(10, 10, 10), True],
            [(5, 10, 10), True],
            [(10, 5, 10), True],
            [(10, 10, 5), True],
            [(10, 5, 5), True],
            [(5, 5, 10), True],
            [(5, 5, 5), True],
            [(7, 7, 7), True],
            [(3, 3, 3), True],
            [(5.5, 5.5, 5.5), True],
            [(0, 0, 0), False],
            [(0, 7, 7), False],
            [(7, 0, 7), False],
            [(7, 7, 0), False],
            [(12, 7, 7), False],
            [(7, 12, 7), False],
            [(7, 7, 12), False],
            [(12, 12, 12), False],
        ]
    )
    def test_compound_area_contains(self, coord: Coordinate3d, expected: bool):
        """Test a compound area contains method.

        Args:
            coord (Coordinate3d): Coordinate to test
            expected (bool): Whether or not area contains coordinate
        """
        area_1 = BoxFeature((5, 5, 5), (10, 10, 10), "area_1")
        area_2 = BoxFeature((1, 1, 1), (6, 6, 6), "area_2")
        area = CompoundAreaFeature([area_1, area_2], "area")

        assert area.contains(coord) == expected

    @parameterized.expand(
        [
            [(0, 0, 0), (0, 0, 0), False],
            [(0, 0, 0), (7, 7, 7), True],
            [(0, 0, 0), (3, 3, 3), True],
            [(7, 7, 7), (0, 0, 0), False],
            [(7, 7, 7), (7, 7, 7), False],
            [(3, 3, 3), (7, 7, 7), False],
        ]
    )
    def test_compound_area_has_crossed(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, expected: bool
    ):
        """Test a compound area has crossed method.

        Args:
            coord_1 (Coordinate2d): Start of path
            coord_2 (Coordinate2d): End of path
            expected (bool): Whether or not the compound feature has been crossed
        """
        area_1 = BoxFeature((5, 5, 5), (10, 10, 10), "area_1")
        area_2 = BoxFeature((1, 1, 1), (6, 6, 6), "area_2")
        area = CompoundAreaFeature([area_1, area_2], "area")

        assert area.has_crossed(coord_1, coord_2) == expected

    def test_compound_area_random_coord_inside(self):
        """Test a compound area can generate points inside itself."""
        area_1 = BoxFeature((5, 5, 5), (10, 10, 10), "area_1")
        area_2 = BoxFeature((1, 1, 1), (6, 6, 6), "area_2")
        area = CompoundAreaFeature([area_1, area_2], "area")
        random_generator = random.Random()

        for _ in range(100):
            coord = area.get_coord_inside(random_generator)
            assert len(coord) == 3
            assert area.contains(coord)
