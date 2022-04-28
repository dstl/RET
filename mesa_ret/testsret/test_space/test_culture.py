"""Tests for culture."""
import unittest

from mesa_ret.space.culture import Culture
from parameterized import parameterized


class TestCulture(unittest.TestCase):
    """Testing culture class."""

    def setUp(self):
        """Instantiate test cultures."""
        self.culture_red = Culture("red culture", height=100.0, reflectivity=0.6, temperature=5.0)
        self.culture_blue = Culture("blue culture")

    def test_get_culture_height(self):
        """Test culture height can be accessed correctly."""
        assert self.culture_red.height == 100.0

    def test_get_culture_height_default(self):
        """Test culture height set by default is set correctly."""
        assert self.culture_blue.height == 0.0

    def test_get_culture_reflectivity(self):
        """Test culture reflectivity can be accessed correctly."""
        assert self.culture_red.reflectivity == 0.6

    def test_get_culture_reflectivity_default(self):
        """Test culture reflectivity set by default is set correctly."""
        assert self.culture_blue.reflectivity == 0.174

    def test_get_culture_temperature(self):
        """Test culture temperature can be accessed correctly."""
        assert self.culture_red.temperature == 5.0

    def test_get_culture_temperature_default(self):
        """Test culture temperature set by default is set correctly."""
        assert self.culture_blue.temperature is None

    def test_attenuation_selection(self):
        """Test highest attenuation factor at wavelength dictionary band boundary."""
        wavelength_dict_1 = {
            (500.0, 600.0): 1.0,
            (600.0, 700.0): 2.0,
        }

        test_culture = Culture(
            name="test culture",
            wavelength_attenuation_factors=wavelength_dict_1,
        )

        assert test_culture.get_attenuation_factor(500.0) == 1.0
        assert test_culture.get_attenuation_factor(600.0) == 2.0
        assert test_culture.get_attenuation_factor(700.0) == 2.0

    def test_overlapping_wavelength_bands(self):
        """Test exception is raised if overlapping wavelength bands are specified."""
        invalid_wavelength_dict_1 = {
            (500.0, 600.0): 1.0,
            (650.0, 750.0): 1.0,
            (600.0, 700.0): 1.0,
        }

        with self.assertRaises(ValueError) as e:
            Culture(
                name="invalid culture 1",
                wavelength_attenuation_factors=invalid_wavelength_dict_1,
            )
        self.assertEqual(
            'Culture: "{name}" has attenuation wavelength bands that overlap',
            str(e.exception),
        )

    def test_incorrect_band_boundaries(self):
        """Test exception handling for invalid boundaries.

        An exception is raised if wavelength band minimum is greater than wavelength
        band maximum
        """
        invalid_wavelength_dict_2 = {
            (600.0, 500.0): 1.0,
        }

        with self.assertRaises(ValueError) as e:
            Culture(
                name="invalid culture 2",
                wavelength_attenuation_factors=invalid_wavelength_dict_2,
            )
        self.assertEqual(
            "Minimum wavelength band value greater than maximum wavelength band value",
            str(e.exception),
        )

    def test_invalid_attenuation_factor(self):
        """Test exception is raised if attenuation factor < 1 specified."""
        invalid_wavelength_dict_3 = {
            (500.0, 600.0): 1.0,
            (650.0, 750.0): 1.0,
            (600.0, 700.0): 0.5,
        }

        with self.assertRaises(ValueError) as e:
            Culture(
                name="invalid culture 3",
                wavelength_attenuation_factors=invalid_wavelength_dict_3,
            )
        self.assertEqual(
            "All culture attenuation factors must be greater than or equal to 1.0",
            str(e.exception),
        )

    def test_negative_culture_height(self):
        """Test exception is raised if negative height is passed to a culture."""
        with self.assertRaises(ValueError) as e:
            Culture(
                name="invalid culture 4",
                height=-100.0,
            )
        self.assertEqual("Culture height cannot be less than 0.0", str(e.exception))

    @parameterized.expand(
        [
            [-1.0],
            [0.0],
            [2.0],
        ]
    )
    def test_invalid_culture_reflectivity(self, reflectivity: float):
        """Test exception is raised if reflectivity outside of range is provided."""
        with self.assertRaises(ValueError) as e:
            Culture(name="invalid culture 5", reflectivity=reflectivity)
        self.assertEqual(
            f"Culture reflectivity ({reflectivity}) must be "
            + "greater than 0 and less than or equal to 1.",
            str(e.exception),
        )
