"""Tests for coordinate utilities."""

import unittest
from math import isclose

from geopy.point import Point
from ret.space.coordinates import LatLongConverter
from parameterized import parameterized


class TestLatLongConverter(unittest.TestCase):
    """Test case for lat long conversion."""

    @parameterized.expand([[0, 0], [5, 0], [0, 5], [5, 5], [10, 10], [30, 30]])
    def test_conversion_two_way(self, lat: float, long: float):
        """Check that lat-long converts to XY and back to ~ the same value.

        Args:
            lat (float): Latitude
            long (float): Longitude
        """
        gen = LatLongConverter((10, 10))
        xy = gen.to_xy(Point(lat, long))
        lat_long: Point = gen.to_lat_long(xy)

        assert isclose(lat_long[0], lat, abs_tol=0.1)  # type: ignore
        assert isclose(lat_long[1], long, abs_tol=0.1)  # type: ignore


class TestLatLongConversionBetweenKnownPoints(unittest.TestCase):
    """Tests for conversions between known points.

    Synthetic data for these calculations is generated using
    https://www.nhc.noaa.gov/gccalc.shtml
    """

    def test_known_la0lo0_to_la10la0(self):
        """(0, 0) to (10, 0) is ~1111km (to nearest KM)."""
        gen = LatLongConverter((0, 0))
        xy = gen.to_xy(Point(10, 0))
        assert isclose(xy[0], 0, abs_tol=0.1)
        assert isclose(xy[1] / 1000, 1111, rel_tol=0.01)

    def test_known_la0lo0_to_la0lo20(self):
        """(0, 0) to (0, 20) is ~222km (to nearest KM)."""
        gen = LatLongConverter((0, 0))
        xy = gen.to_xy(Point(0, 20))
        assert isclose(xy[0] / 1000, 2222, rel_tol=0.01)
        assert isclose(xy[1], 0, abs_tol=0.1)

    def test_known_la10lo10_to_la0lo0(self):
        """(10, 10) to (0, 0) gives x ~-1096 and y as ~-111km each.

        The x-coordinate is slightly lower than for the test at the equator.
        """
        gen = LatLongConverter((10, 10))
        xy = gen.to_xy(Point(0, 0))

        assert isclose(xy[0] / 1000, -1096, rel_tol=0.01)
        assert isclose(xy[1] / 1000, -1111, rel_tol=0.01)
