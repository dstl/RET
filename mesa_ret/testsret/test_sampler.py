"""Sampler tests."""
from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

from mesa_ret.parameters import CategoricModelParameter, SampleableContinuousModelParameter
from mesa_ret.sampling import (
    GradientAdaptiveSampler,
    LatinHypercubeSampler,
    ParameterProductSampler,
    get_parameter_options_dict,
)
from parameterized import parameterized

if TYPE_CHECKING:
    from mesa_ret.parameters import SampleableParameter


def test_get_parameter_options_dict():
    """Test get parameter options dict method."""
    sampleable_parameters = [
        CategoricModelParameter(
            name="Categoric",
            options=["A", "B", "C"],
        ),
        SampleableContinuousModelParameter("Continuous", 0, 0.2, 0.1),
    ]
    options = get_parameter_options_dict(sampleable_parameters)

    assert options["Categoric"] == ["A", "B", "C"]
    assert options["Continuous"] == [0.0, 0.1, 0.2]


def test_parameter_product_sampler():
    """Test the parameter product sampler."""
    sampler = ParameterProductSampler()

    sampleable_parameters: list[SampleableParameter] = [
        CategoricModelParameter(
            name="Categoric 1",
            options=["A", "B"],
        ),
        CategoricModelParameter(
            name="Categoric 2",
            options=["C", "D"],
        ),
    ]
    samples = sampler.sample(sampleable_parameters=sampleable_parameters)

    expected_result = [
        {"Categoric 1": "A", "Categoric 2": "C"},
        {"Categoric 1": "A", "Categoric 2": "D"},
        {"Categoric 1": "B", "Categoric 2": "C"},
        {"Categoric 1": "B", "Categoric 2": "D"},
    ]

    assert samples == expected_result


@parameterized.expand([[2], [10], [20]])
def test_latin_hypercube_sampler(n: int):
    """Test the latin hypercube sampler."""
    sampler = LatinHypercubeSampler(n_experiments=n, random_state=1)

    param_1_options = ["A", "B", "C"]
    param_2_options = ["D", "E", "F"]
    sampleable_parameters: list[SampleableParameter] = [
        CategoricModelParameter(
            name="Categoric 1",
            options=param_1_options,
        ),
        CategoricModelParameter(
            name="Categoric 2",
            options=param_2_options,
        ),
    ]
    samples = sampler.sample(sampleable_parameters=sampleable_parameters)
    assert len(samples) == n
    for sample in samples:
        assert "Categoric 1" in sample
        assert sample["Categoric 1"] in param_1_options
        assert "Categoric 2" in sample
        assert sample["Categoric 2"] in param_2_options


def test_gradient_adaptive_sampler_with_categoric_parameters():
    """Test the gradient adaptive sampler with a non-continuous parameter."""
    sampler = GradientAdaptiveSampler(random=random.Random(1))
    categoric_param = CategoricModelParameter(
        name="Categoric",
        options=["A", "B", "C"],
    )
    continuous_param = SampleableContinuousModelParameter("Continuous", 0, 0.2, 0.1)
    sampleable_parameters: list[SampleableParameter] = [
        categoric_param,
        continuous_param,
    ]

    with warnings.catch_warnings(record=True) as w:
        samples = sampler.sample(sampleable_parameters=sampleable_parameters)
        assert len(w) == 1
        assert (
            "1 non-continuous sampleable parameters will be ignored by the Adaptive Sampler."
            == str(w[0].message)
        )

    assert len(samples) == 1
    sample = samples[0]
    assert categoric_param.name in sample
    assert sample[categoric_param.name] == categoric_param.get_default_value()
    assert continuous_param.name in sample
    assert continuous_param.min_val <= sample[continuous_param.name] <= continuous_param.max_val
