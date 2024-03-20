"""Unit Tests for the run_resampling algorithm loop."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pytest import raises

from ret.parameters import CategoricModelParameter
from ret.resampling import run_resampling
from ret.sampling import Sampler
from ret.testing.mocks import MockParametrisedModel

if TYPE_CHECKING:
    from typing import Any, Optional, Tuple

    from ret.resampling import SampleableParameter


class MockCategoricModelParameter(CategoricModelParameter):
    """Mock Categoric Model Parameter for testing."""

    def __init__(self) -> None:
        """Creates a new MockCategoricModelParameter."""
        super().__init__(name="Mock Categoric Parameter", options=["1", "2"])


class MockZeroExperimentSampler(Sampler):
    """Mock Sampler representation which returns no experiments."""

    def __init__(self) -> None:
        """Creates a new Mock Zero Experiment Sampler."""
        super().__init__()

    def _sample(
        self,
        sampleable_parameters: list[SampleableParameter],
        previous_results: Optional[list[Tuple[dict[str, Any], Any]]] = None,
    ) -> list[dict[str, Any]]:
        return list()


def test_run_resampling_early_breakout():
    """Tests that the resampling algorithm breaks out after 2 zero-experiment iterations."""
    n_experiments, _, n_invalid_experiments, _ = run_resampling(
        model_cls=MockParametrisedModel,
        variable_params=[MockCategoricModelParameter()],
        sampler=MockZeroExperimentSampler(),
        iterations=3,
    )

    assert len(n_experiments) == 2
    assert len(n_invalid_experiments) == 2


def test_run_resampling_missing_variable_parameters():
    """Test run resampling with no variable parameters raises an error."""
    with raises(ValueError) as e:
        run_resampling(
            model_cls=MockParametrisedModel,
            variable_params=[],
            sampler=MockZeroExperimentSampler(),
            iterations=3,
        )
    assert e.value.args[0] == "Variable Parameters must be defined"


def test_run_resampling_missing_sampler():
    """Test run resampling with no sampler raises an error."""
    with raises(ValueError) as e:
        run_resampling(
            model_cls=MockParametrisedModel,
            variable_params=[MockCategoricModelParameter()],
            sampler=None,
            iterations=3,
        )
    assert e.value.args[0] == "Sampler must be defined"
