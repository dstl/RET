"""RetGen tests."""
from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import mesa_ret.io.v2 as v2
import pytest
from pydantic import ValidationError


class TestRetGen:
    """V2 model validation tests."""

    # color testing
    invalid_r_too_low = {"r": -1000, "g": 0, "b": 0}
    invalid_b_too_high = {"r": 0, "g": 0, "b": 1000}
    valid_color = {"r": 0, "g": 0, "b": 0}

    @pytest.mark.parametrize(
        "input, expected",
        [
            (
                invalid_r_too_low,
                pytest.raises(
                    ValidationError,
                    match="Color RGB values must be between 0 and 255.",
                ),
            ),
            (
                invalid_b_too_high,
                pytest.raises(
                    ValidationError,
                    match="Color RGB values must be between 0 and 255.",
                ),
            ),
            (valid_color, does_not_raise()),
        ],
    )
    def test_color_schema(self, input, expected):
        """Test v2 color schema.

        Args:
            input: Input arguments to culture
            expected: Expected values and context manager
        """
        with expected:
            v2.ColorSchema(**input)

    # wavelength attenuator testing
    invalid_min = {"wavelength_min": -5, "wavelength_max": 100, "attenuation": 5}
    invalid_max = {"wavelength_min": 5, "wavelength_max": 2, "attenuation": 5}
    invalid_attenutation = {"wavelength_min": 5, "wavelength_max": 10, "attenuation": 0}
    valid_attenuator = {"wavelength_min": 5, "wavelength_max": 10, "attenuation": 2}

    @pytest.mark.parametrize(
        "input, expected",
        [
            (
                invalid_min,
                pytest.raises(
                    ValidationError,
                    match="Minimum wavelength value must be greater than 0.",
                ),
            ),
            (
                invalid_max,
                pytest.raises(
                    ValidationError,
                    match="Maximum wavelength must be greater than minimum wavelength.",
                ),
            ),
            (
                invalid_attenutation,
                pytest.raises(
                    ValidationError,
                    match="Attenuation must be greater than or equal to 1.",
                ),
            ),
            (valid_attenuator, does_not_raise()),
        ],
    )
    def test_wavelength_attenuation_schema(self, input, expected):
        """Test v2 wavelength attenuation schema.

        Args:
            input: Input arguments to culture
            expected: Expected values and context manager
        """
        with expected:
            v2.CultureWavelengthAttenuationSchema(**input)

    # culture testing
    invalid_culture_height = {
        "name": "test",
        "height": -5,
        "wavelength_attenuations": [v2.CultureWavelengthAttenuationSchema(**valid_attenuator)],
    }
    valid_culture = {
        "name": "test",
        "height": 5,
        "wavelength_attenuations": [v2.CultureWavelengthAttenuationSchema(**valid_attenuator)],
    }

    @pytest.mark.parametrize(
        "input, expected",
        [
            (
                invalid_culture_height,
                pytest.raises(
                    ValidationError,
                    match="Culture height must be greater than or equal to 0.",
                ),
            ),
            (valid_culture, does_not_raise()),
        ],
    )
    def test_culture_schema(self, input, expected):
        """Test v2 culture schema.

        Args:
            input: Input arguments to culture
            expected: Expected values and context manager
        """
        with expected:
            v2.CultureSchema(**input)
