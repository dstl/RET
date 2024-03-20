"""Test cases for spatial coordinate conversion."""

import unittest
from warnings import catch_warnings
from ret.space.heightband import AbsoluteHeightBand
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from ret.testing.mocks import MockAgent


class TestSpaceCoordinateConversion2d(unittest.TestCase):
    """Test cases for coordinate conversion in 2D."""

    def setUp(self):
        """Create a test space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture2d(70, 20, -30, -30)

    def test_get_coordinate_2d(self):
        """Test that 2D coordinates can be used."""
        assert self.space.get_coordinate_2d((10, 10)) == (10, 10)
        assert self.space.get_coordinate_2d((10.5, 10.5)) == (10.5, 10.5)
        assert self.space.get_coordinate_2d((10, 10, 10)) == (10, 10)
        assert self.space.get_coordinate_2d((10, 10, "test")) == (10, 10)

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_2d(10)
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_2d((10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_2d((10, 10, 10, 10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

    def test_get_coordinate_in_correct_dimension(self):
        """Test that coordinates are returned with the correct number of dimensions."""
        assert self.space.get_coordinate_in_correct_dimension((10, 10)) == (10, 10)
        assert self.space.get_coordinate_in_correct_dimension((10.5, 10.5)) == (
            10.5,
            10.5,
        )
        assert self.space.get_coordinate_in_correct_dimension((10, 10, 10)) == (10, 10)
        assert self.space.get_coordinate_in_correct_dimension((10, 10, "test")) == (
            10,
            10,
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_in_correct_dimension(10)
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_in_correct_dimension((10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_in_correct_dimension((10, 10, 10, 10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )


class TestSpaceCoordinateConversion3d(TestSpaceCoordinateConversion2d):
    """Test cases for coordinate conversion in 3D."""

    def setUp(self):
        """Create a test space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(70, 20, -30, -30)

    def test_get_coordinate_3d(self):
        """Test that 2D coordinates can be converted into 3D coordinates."""
        assert self.space.get_coordinate_3d((10, 10)) == (10, 10, 0)

        d2_to_d3 = self.space.get_coordinate_3d((10.5, 10.5))
        assert d2_to_d3 == (10.5, 10.5, 0)

        assert self.space.get_coordinate_3d((10, 10, 10)) == (10, 10, 10)
        assert self.space.get_coordinate_3d((10, 10, "test"), [AbsoluteHeightBand("test", 10)]) == (
            10,
            10,
            10,
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_3d(10)
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_3d((10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_3d((10, 10, 10, 10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_3d((10, 10, "test"))
        self.assertEqual(
            "When converting a Coordinate3dBand to Coordinate3d you must "
            + "provide the list of bands",
            str(e.exception),
        )

    def test_get_coordinate_in_correct_dimension(self):
        """Test that coordinates are returned with the correct number of dimensions."""
        assert self.space.get_coordinate_in_correct_dimension((10, 10)) == (10, 10, 0)

        assert self.space.get_coordinate_in_correct_dimension((10.5, 10.5)) == (
            10.5,
            10.5,
            0,
        )

        assert self.space.get_coordinate_in_correct_dimension((10, 10, 10)) == (
            10,
            10,
            10,
        )
        assert self.space.get_coordinate_in_correct_dimension(
            (10, 10, "test"), [AbsoluteHeightBand("test", 10)]
        ) == (10, 10, 10)

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_in_correct_dimension(10)
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_in_correct_dimension((10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_in_correct_dimension((10, 10, 10, 10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.get_coordinate_in_correct_dimension((10, 10, "test"))
        self.assertEqual(
            "When converting a Coordinate3dBand to Coordinate3d you must "
            + "provide the list of bands",
            str(e.exception),
        )

    def test_place_2d_agent(self):
        """Test that agents can be placed using 2D coordinates."""
        a = MockAgent(1, (10, 10))
        with catch_warnings(record=True) as w:
            self.space.place_agent(a, (10, 10))
            assert a.pos == (10, 10, 0)
        assert len(w) == 1
        assert (
            "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
            in str(w[0].message)
        )

    def test_move_2d_agent(self):
        """Test that agents can be moved using 2D coordinates."""
        a = MockAgent(1, (10, 10))
        with catch_warnings(record=True) as w:
            self.space.place_agent(a, (10, 10))
        assert len(w) == 1
        assert (
            "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
            in str(w[0].message)
        )

        self.space.move_agent(a, (15, 15))
        assert a.pos == (15, 15, 0)

    def test_get_neighbors_2d(self):
        """Test that it's possible to identify neighbours using 2D coordinates."""
        a = MockAgent(1, (10, 10))
        with catch_warnings(record=True) as w:
            self.space.place_agent(a, (10, 10))
        assert len(w) == 1
        assert (
            "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
            in str(w[0].message)
        )
        neighbors = self.space.get_neighbors((10, 10), 1)
        assert len(neighbors) == 1

    def test_get_heading_2d(self):
        """Test that headings can be calculated using 2D coordinates."""
        pos_1_2d = (0, 0)
        pos_2_2d = (10, 0)
        pos_1_3d = (0, 0, 0)
        pos_2_3d = (10, 0, 0)

        d2_to_d2 = self.space.get_heading(pos_1_2d, pos_2_2d)
        d2_to_d3 = self.space.get_heading(pos_1_2d, pos_2_3d)
        d3_to_d2 = self.space.get_heading(pos_1_3d, pos_2_2d)
        d3_to_d3 = self.space.get_heading(pos_1_3d, pos_2_3d)

        assert d2_to_d2 == (10, 0, 0)
        assert d2_to_d3 == (10, 0, 0)
        assert d3_to_d2 == (10, 0, 0)
        assert d3_to_d3 == (10, 0, 0)

    def test_get_distance_2d(self):
        """Test that distances can be calculated using 2D coordinates."""
        pos_1_2d = (0, 0)
        pos_2_2d = (10, 0)
        pos_1_3d = (0, 0, 0)
        pos_2_3d = (10, 0, 0)

        distance_2d_to_2d = self.space.get_distance(pos_1_2d, pos_2_2d)
        distance_3d_to_2d = self.space.get_distance(pos_1_2d, pos_2_3d)
        distance_2d_to_3d = self.space.get_distance(pos_1_3d, pos_2_2d)

        distance_3d_to_3d = self.space.get_distance(pos_1_3d, pos_2_3d)

        assert distance_2d_to_2d == 10
        assert distance_3d_to_2d == 10
        assert distance_2d_to_3d == 10
        assert distance_3d_to_3d == 10

    def test_torus_adj_2d(self):
        """Test that space can perform torus adjustment using 2D coordinates."""
        assert self.space.torus_adj((10, 10)) == (10, 10)

        with self.assertRaises(TypeError) as e:
            self.space.torus_adj((10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )

        with self.assertRaises(TypeError) as e:
            self.space.torus_adj((10, 10, 10, 10))
        self.assertEqual(
            "Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand",
            str(e.exception),
        )
