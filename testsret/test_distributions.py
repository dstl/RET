"""Tests for distributions."""

import unittest

from ret.sensing.distribution import SingleValueDistribution, TriangularDistribution
from parameterized.parameterized import parameterized


class TestDistributions(unittest.TestCase):
    """Test distributions."""

    def test_triangular_distribution(self):
        """Test to check triangular distribution.

        Ensure sample only returns values between upper and lower limits.
        """
        triangular_distribution = TriangularDistribution(lower_limit=10, upper_limit=20, mode=15)

        for _ in range(10000):
            test_value = triangular_distribution.sample()
            assert 10 <= test_value <= 20

    def test_triangular_distribution_return_type(self):
        """Test triangular distribution returns float."""
        triangular_distribution = TriangularDistribution(lower_limit=10, upper_limit=20, mode=15)
        assert isinstance(triangular_distribution.sample(), float)

    def test_default_triangular_distribution(self):
        """Test to check triangular distribution mode defaults to correct value."""
        triangular_distribution = TriangularDistribution(lower_limit=10, upper_limit=20)

        assert triangular_distribution._mode == 15

    def test_triangular_distribution_invalid_limits(self):
        """Test to check exception is raised."""
        with self.assertRaises(ValueError) as e:
            TriangularDistribution(lower_limit=20, upper_limit=10)
        self.assertEqual("Lower limit must be less than upper limit", str(e.exception))

    @parameterized.expand([[0.0], [30.0]])
    def test_triangular_distribution_invalid_mode(self, param_mode: float):
        """Test to check exception is raised.

        Args:
            param_mode (float): invalid parameter modal value
        """
        with self.assertRaises(ValueError) as e:
            TriangularDistribution(lower_limit=10, upper_limit=20, mode=param_mode)
        self.assertEqual(
            "Mode must be greater than or equal to lower limit, "
            + "and less than or equal to upper limit",
            str(e.exception),
        )

    def test_single_value_distribution_return_type(self):
        """Test to check single value distribution returns expected type."""
        delta_distribution = SingleValueDistribution(20.0)

        assert isinstance(delta_distribution.sample(), float)

    def test_single_value_distribution(self):
        """Test to check single value distribution returns expected value."""
        delta_distribution = SingleValueDistribution(20.0)

        assert delta_distribution.sample() == 20.0
