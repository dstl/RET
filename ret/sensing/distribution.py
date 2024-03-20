"""Probability distribution representation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy import random

if TYPE_CHECKING:
    from typing import Optional


class Distribution(ABC):
    """Distribution class for defining probability distributions."""

    @abstractmethod
    def __init__(self) -> None:
        """Create a distribution."""

    @abstractmethod
    def sample(self) -> float:  # pragma: no cover
        """Sample from probability distribution.

        Returns:
            float: random value from probability distribution.
        """

    pass


class SingleValueDistribution(Distribution):
    """Distribution class for defining a single value distribution.

    Always returns a single value.
    """

    def __init__(self, mean: float) -> None:
        """Initialise a single value distribution.

        Args:
            mean (float): Mean value for distribution
        """
        self.mean = mean

    def sample(self) -> float:
        """Sample from single value distribution.

        Returns:
            float: float value defined by single value distribution.
        """
        return self.mean


class TriangularDistribution(Distribution):
    """Triangular probability distribution class."""

    def __init__(
        self, lower_limit: float, upper_limit: float, mode: Optional[float] = None
    ) -> None:
        """Initialise a triangular probability distribution.

        Args:
            lower_limit (float): Minimum value returnable by triangular distribution.
            upper_limit (float): Maximum value returnable by triangular distribution.
            mode (Optional[float], optional): Mode of triangular distribution. Defaults
                to None, which results in a mode value being chosen that is the middle
                value between the lower and upper limit.

        Raises:
            ValueError: Lower limit must be less than the upper limit.
            ValueError: Mode must be equal to, or between the lower and upper limits.
        """
        if mode is None:
            mode = (upper_limit + lower_limit) / 2

        if not lower_limit < upper_limit:
            raise ValueError("Lower limit must be less than upper limit")

        if not lower_limit <= mode <= upper_limit:
            raise ValueError(
                "Mode must be greater than or equal to lower limit, "
                + "and less than or equal to upper limit"
            )

        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._mode = mode

    def sample(self) -> float:
        """Sample from triangular distribution.

        Returns:
            float: random value generated using triangular distribution
        """
        sample: float = random.triangular(
            left=self._lower_limit, right=self._upper_limit, mode=self._mode, size=1
        )[0]

        return sample
