"""Map cultures."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

WavelengthFactor = dict[tuple[float, float], float]


class Culture:
    """Base class for map culture."""

    def __init__(
        self,
        name: str,
        height: float = 0.0,
        reflectivity: float = 0.174,
        temperature: Optional[float] = None,
        wavelength_attenuation_factors: Optional[WavelengthFactor] = None,
    ) -> None:
        """Class that describes culture on terrain.

        Args:
            name (str): Name of culture
            height (float): Height culture extends above terrain. Defaults to 0.0.
            reflectivity (float): Reflectivity of the culture, greater than 0 and less
                than or equal to 1. Defaults to 0.174.
            temperature (Optional[float]): Optional temperature of the culture, if different
                to the ambient temperature of the Space. Defaults to None.
            wavelength_attenuation_factors (Optional[WavelengthFactor]): Wavelength band
                attenuation factors. If wavelength attenuation factors are not provided,
                requests for mappings default to 1.0 (i.e. no attenuation).

        Raises:
            ValueError: Negative value for culture height
        """
        if height < 0.0:
            raise ValueError("Culture height cannot be less than 0.0")
        if reflectivity <= 0 or reflectivity > 1:
            raise ValueError(
                f"Culture reflectivity ({reflectivity}) must be "
                + "greater than 0 and less than or equal to 1."
            )

        if wavelength_attenuation_factors is None:
            wavelength_attenuation_factors = {}
        else:
            self._check_wavelength_factors_validity(wavelength_attenuation_factors)

        self.name = name
        self.height = height
        self.reflectivity = reflectivity
        self.temperature = temperature
        self.wavelength_attenuation_factors = wavelength_attenuation_factors

    def _check_wavelength_factors_validity(self, attenuation_factors: WavelengthFactor):
        """Check that wavelength ranges in attenuation factor dictionary do not overlap.

        Args:
            attenuation_factors (WavelengthFactor): Wavelength factors

        Raises:
            ValueError: Raised when supplied with attenuation factors less than 1.0
            ValueError: Raised when minimum wavelength is greater than maximum
                wavelength
            ValueError: Raised if wavelength ranges are found that overlap
        """
        if not all([factor >= 1 for factor in list(attenuation_factors.values())]):
            raise ValueError("All culture attenuation factors must be greater than or equal to 1.0")

        wavelength_ranges = list(attenuation_factors.keys())

        for r in wavelength_ranges:
            if r[0] > r[1]:
                raise ValueError(
                    "Minimum wavelength band value greater than maximum wavelength " + "band value"
                )

        sorted_wavelength_ranges = sorted(wavelength_ranges, key=lambda x: x[0])

        for i in range(len(sorted_wavelength_ranges) - 1):
            if sorted_wavelength_ranges[i][1] > sorted_wavelength_ranges[i + 1][0]:
                raise ValueError('Culture: "{name}" has attenuation wavelength bands that overlap')

    def get_attenuation_factor(self, wavelength: float) -> float:
        """Return attenuation factor given a wavelength.

        Args:
            wavelength (float): signal wavelength

        Returns:
            float: attenuation factor of signal at given wavelength
        """
        greatest_attenuation_factor = 1.0

        for k, v in self.wavelength_attenuation_factors.items():
            if k[0] <= wavelength <= k[1]:
                if greatest_attenuation_factor < v:
                    greatest_attenuation_factor = v

        return greatest_attenuation_factor


__culture__ = Culture(name="Default Culture")


def default_culture() -> Culture:
    """Return the default culture.

    Returns:
        Culture: Default culture.
    """
    return __culture__
