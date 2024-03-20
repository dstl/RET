"""Tests for model parameters."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from ret.parameters import (
    CategoricModelParameter,
    NumericParameterSpecification,
    SampleableContinuousModelParameter,
    get_categoric_parameter_options,
    get_model,
    get_model_parameters,
    get_numeric_min_max_values,
)
from ret.testing.mocks import MockParametrisedModel

if TYPE_CHECKING:
    from typing import Callable


class TestModelParameters(unittest.TestCase):
    """Tests cases for model parameters."""

    def test_numeric_parameter_to_default_min_max_allowable_none(self):
        """Test the NumericParameterSpecification to_default_min method with max as None."""
        param = NumericParameterSpecification("Name", "Description", None, None)
        assert param.to_default_min() == 0

    def test_numeric_parameter_to_default_min_max_allowable_datetime(self):
        """Test the NumericParameterSpecification to_default_min method with max as datetime."""
        param = NumericParameterSpecification("Name", "Description", None, datetime(2022, 1, 2))
        assert param.to_default_min() == datetime(2022, 1, 1)

    def test_numeric_parameter_to_default_min_max_allowable_set_positive(self):
        """Test the NumericParameterSpecification to_default_min method with max as positive."""
        param = NumericParameterSpecification("Name", "Description", None, 2)
        assert param.to_default_min() == 0

    def test_numeric_parameter_to_default_min_max_allowable_set_negative(self):
        """Test the NumericParameterSpecification to_default_min method with max as negative."""
        param = NumericParameterSpecification("Name", "Description", None, -2)
        assert param.to_default_min() == -4

    def test_numeric_parameter_to_default_max_min_allowable_none(self):
        """Test the NumericParameterSpecification to_default_max method with min as None."""
        param = NumericParameterSpecification("Name", "Description", None, None)
        assert param.to_default_max() == 1

    def test_numeric_parameter_to_default_max_min_allowable_set_positive(self):
        """Test the NumericParameterSpecification to_default_max method with min as positive."""
        param = NumericParameterSpecification("Name", "Description", 2, None)
        assert param.to_default_max() == 4

    def test_numeric_parameter_to_default_max_min_allowable_set_negative(self):
        """Test the NumericParameterSpecification to_default_max method with min as negative."""
        param = NumericParameterSpecification("Name", "Description", -2, None)
        assert param.to_default_max() == 0

    def test_numeric_parameter_to_default_max_min_allowable_set_zero(self):
        """Test the NumericParameterSpecification to_default_max method with min as 0."""
        param = NumericParameterSpecification("Name", "Description", 0, None)
        assert param.to_default_max() == 1

    def test_numeric_parameter_to_default_max_min_allowable_set_timedelta_zero(self):
        """Test the NumericParameterSpecification to_default_max method with min as timedelta(0)."""
        param = NumericParameterSpecification("Name", "Description", timedelta(0), None)
        assert param.to_default_max() == timedelta(seconds=1)

    def test_numeric_parameter_to_default_max_min_allowable_datetime(self):
        """Test the NumericParameterSpecification to_default_max method with min as datetime."""
        param = NumericParameterSpecification("Name", "Description", datetime(2022, 1, 2), None)
        assert param.to_default_max() == datetime(2022, 1, 3)

    def test_custom_categorisation_description(self):
        """Test custom categorisation descriptions."""

        def custom_description(category: str) -> str:
            if category == "A":
                return "Custom category name for (A)"
            return "Category " + category

        categories = ["A", "B", "C"]

        func: Callable[[str], str] = custom_description

        param = CategoricModelParameter(
            name="test",
            options=categories,
            item_description=func,
        )

        assert param.get_descriptions() == [
            "Custom category name for (A)",
            "Category B",
            "Category C",
        ]

    def test_default_categorisation_description(self):
        """Test default categorisation description."""
        categories = ["1", "2", "3"]
        param = CategoricModelParameter(name="test", options=categories)
        assert param.get_descriptions() == ["1", "2", "3"]

    def test_categoric_model_empty_options(self):
        """Test a ValueError is thrown with empty options for CategoricModelParameters."""
        with self.assertRaises(ValueError) as e:
            CategoricModelParameter(name="test", options=[])
        self.assertEqual(
            "options cannot be empty.",
            str(e.exception),
        )

    def test_categoric_model_get_options(self):
        """Test get_options method for CategoricModelParameter."""
        categories = ["1", "2", "3"]
        param = CategoricModelParameter(name="test", options=categories)
        assert param.get_options() == ["1", "2", "3"]

    def test_categoric_model_get_default_value(self):
        """Test get_default_value method for CategoricModelParameter."""
        categories = ["1", "2", "3"]
        default_value = "TestDefault"
        param = CategoricModelParameter(
            name="test", options=categories, default_value=default_value
        )
        assert param.get_default_value() == "TestDefault"

    def test_sampleable_continuous_model_get_options(self):
        """Test get_options method for SampleableContinuousModelParameter."""
        param = SampleableContinuousModelParameter("Test", 0, 0.2, 0.1)
        assert param.get_options() == [0.0, 0.1, 0.2]

    def test_sampleable_continuous_model_get_default_value(self):
        """Test get_default_value method for SampleableContinuousModelParameter."""
        param = SampleableContinuousModelParameter("Test", 0, 10, 0.1, 5)
        assert param.get_default_value() == 5

    def test_get_model(self):
        """Get model class."""
        model = get_model("ret.testing.mocks", "MockParametrisedModel")
        assert model == MockParametrisedModel

    def test_get_model_parameters(self):
        """Get model parameters from class."""
        params = get_model_parameters("ret.testing.mocks", "MockParametrisedModel")
        assert params == MockParametrisedModel.get_parameters()

    def test_get_choices(self):
        """Test accessing choices for a categoric parameter."""
        choices = get_categoric_parameter_options(
            "ret.testing.mocks", "MockParametrisedModel", "c1"
        )

        assert choices == ["Choice 1", "Choice 2", "Choice 3"]

    def test_get_choices_error(self):
        """Test accessing invalid choices for a categoric parameter throws a ValueError."""
        with self.assertRaises(ValueError) as e:
            get_categoric_parameter_options(
                "ret.testing.mocks", "MockParametrisedModel", "InvalidParam"
            )
        self.assertEqual(
            "'InvalidParam' not defined for model.",
            str(e.exception),
        )

    def test_get_min_max(self):
        """Test accessing min/max values for a numeric parameter."""
        min_val, max_val = get_numeric_min_max_values(
            "ret.testing.mocks", "MockParametrisedModel", "n1"
        )

        assert min_val == 0
        assert max_val == 100

    def test_get_min_max_error(self):
        """Test ValueError for undefined model parameters."""
        with self.assertRaises(ValueError) as e:
            get_numeric_min_max_values("ret.testing.mocks", "MockParametrisedModel", "InvalidParam")
        self.assertEqual(
            "'InvalidParam' not defined for model.",
            str(e.exception),
        )
