"""Tests for environmental parameters in space."""

import unittest

from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture3d, Precipitation
from parameterized import parameterized


class TestSpaceEnvironmentalParameters(unittest.TestCase):
    """Test environmental parameters in space."""

    def test_environmental_parameters_defaults(self):
        """Test environmental parameters defaults are set correctly."""
        space = ContinuousSpaceWithTerrainAndCulture3d(x_max=10, y_max=10)

        assert space._relative_humidity == 0.75
        assert space._precipitation == Precipitation.CLEAR
        assert space._dust == 0.0
        assert space._meteorological_visibility_range == 320.0
        assert space._ambient_temperature == 0.0

    def test_environmental_parameters(self):
        """Test environmental parameters are set correctly."""
        relative_humidity = 0.5
        precipitation = Precipitation.DRIZZLE_HEAVY
        dust = 0.1
        meteorological_visibility_range = 50.0
        ambient_temperature = 2.0

        space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=10,
            y_max=10,
            relative_humidity=relative_humidity,
            precipitation=precipitation,
            dust=dust,
            meteorological_visibility_range=meteorological_visibility_range,
            ambient_temperature=ambient_temperature,
        )

        assert space._relative_humidity == relative_humidity
        assert space._precipitation == precipitation
        assert space._dust == dust
        assert space._meteorological_visibility_range == meteorological_visibility_range
        assert space._ambient_temperature == ambient_temperature

    @parameterized.expand([[-1.0], [0.0], [1.0], [2.0]])
    def test_relative_humidity_value_error(self, relative_humidity):
        """Test Space exception handling for invalid relative humidity values.

        Args:
            relative_humidity (float): An invalid relative humidity value
        """
        with self.assertRaises(ValueError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=10, y_max=10, relative_humidity=relative_humidity
            )
        assert (
            f"Relative humidity ({relative_humidity}) must be between 0 and 1, exclusive."
            == str(e.exception)
        )

    def test_dust_value_error(self):
        """Test Space exception handling for negative dust values."""
        dust = -1.0
        with self.assertRaises(ValueError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(x_max=10, y_max=10, dust=-1.0)
        assert f"Atmospheric dust ({dust} g/m3) cannot be negative." == str(e.exception)

    def test_meteorological_visibility_range_value_error(self):
        """Test Space exception handling for negative meteorological visibility range values."""
        meteorological_visibility_range = -1.0
        with self.assertRaises(ValueError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=10, y_max=10, meteorological_visibility_range=-1.0
            )
        assert (
            f"Meteorological visibility range ({meteorological_visibility_range} m) "
            + "cannot be negative."
            == str(e.exception)
        )
