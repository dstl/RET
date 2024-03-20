"""Model Parameter Samplers."""
from __future__ import annotations

import copy
import math
import random
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import adaptive as ad
import numpy as np
import pyDOE

from mesa.batchrunner import ParameterProduct
from ret.parameters import SampleableContinuousModelParameter

if TYPE_CHECKING:
    from typing import Any, Optional, Tuple, Union

    from ret.resampling import SampleableParameter


def get_parameter_options_dict(
    sampleable_parameters: list[SampleableParameter],
) -> dict[str, list[Any]]:
    """Convert list of sampleable parameters to dictionary of param name to options.

    Args:
        sampleable_parameters (list[SampleableParameter]): The sampleable parameters

    Returns:
        dict[str, list[Any]]: Dictionary of parameter name to options
    """
    return {parameter.name: parameter.get_options() for parameter in sampleable_parameters}


class LatinHypercubeParameterSampler(ParameterProduct):
    """Iterator to produce experiments using Latin Hypercube sampling methodology."""

    def __init__(
        self,
        variable_parameters: Optional[dict[str, Any]],
        n_experiments: int,
        random_state: Optional[int] = None,
    ):
        """Create a new LatinHypercubeParameterSampler.

        Args:
            variable_parameters (Optional[dict[str, Any]]): Dictionary of parameters to
                lists of values. The model will be run with every combination of these
                parameters.
                For example, given variable_parameters of
                    {"param_1": range(5),
                    "param_2": [1, 5, 10]}
                If None, a single experiment will be run with no variable parameters.
            n_experiments (int): number of experiments required
            random_state: (Optional[int]): pseudo-random number generate state used for
                random uniform sampling from lists of possible values. Defaults to None.
        """
        if variable_parameters is not None:
            param_names, param_lists = zip(*(copy.deepcopy(variable_parameters)).items())
            param_lists = list(param_lists)
            for i, param_list in enumerate(param_lists):
                if isinstance(param_list[0], dict):
                    param_lists[i] = param_list
                else:
                    param_lists[i] = sorted(param_list)

            if isinstance(random_state, int):
                np.random.seed(random_state)

            lhs_samples = pyDOE.lhs(len(param_names), n_experiments)

            experiment_list = []
            for sample in lhs_samples:  # type: ignore
                sample_dict = {}
                for i, param_name in enumerate(param_names):
                    ordinal_position = math.floor(sample[i] * len(param_lists[i]))
                    sample_dict[param_name] = param_lists[i][ordinal_position]
                experiment_list.append(sample_dict)

            self._experiment_iter = iter(experiment_list)
        else:
            self._experiment_iter = iter([])

    def __iter__(self):
        """Return iterable representation of self.

        Returns:
            BatchRunner: Iterable instance
        """
        return self

    def __next__(self) -> dict[str, Any]:
        """Get next item in batch runner.

        Returns:
            dict[str, Any]: Next experiment
        """
        return next(self._experiment_iter)


class Sampler(ABC):
    """Embodies a Sampling Protocol (with configured settings)."""

    @abstractmethod
    def __init__(self) -> None:  # pragma: no cover
        """Creates a new Sampler."""
        pass

    def sample(
        self,
        sampleable_parameters: list[SampleableParameter],
        previous_results: Optional[list[Tuple[dict[str, Any], Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Samples the specified parameters.

        Args:
            sampleable_parameters (list[SampleableParameter]): List of
                sampleable parameters.
            previous_results (Optional[list[Tuple[dict[str, Any], Any]]]): List of
                results in the format of Tuple[design, result] where design is a
                dictionary of parameters to value, and result is the corresponding
                result. Depending on the Sampling algorithm, this may be ignored.
                Defaults to None.

        Returns:
            list[dict[str, Any]]: A list of dict[str, Any] param-value combinations.
        """
        return self._sample(
            sampleable_parameters=sampleable_parameters, previous_results=previous_results
        )

    @abstractmethod
    def _sample(
        self,
        sampleable_parameters: list[SampleableParameter],
        previous_results: Optional[list[Tuple[dict[str, Any], Any]]] = None,
    ) -> list[dict[str, Any]]:  # pragma: no cover
        pass


class ParameterProductSampler(Sampler):
    """Sampler representation of the MESA Parameter Product."""

    def __init__(self) -> None:
        """Creates a new Parameter Product Sampler."""
        super().__init__()

    def _sample(
        self,
        sampleable_parameters: list[SampleableParameter],
        previous_results: Optional[list[Tuple[dict[str, Any], Any]]] = None,
    ) -> list[dict[str, Any]]:
        return list(
            ParameterProduct(variable_parameters=get_parameter_options_dict(sampleable_parameters))
        )


class LatinHypercubeSampler(Sampler):
    """Sampler implementation embodying the Latin Hypercube approach."""

    def __init__(self, n_experiments: int = 1, random_state: Optional[int] = None) -> None:
        """Creates a new Latin Hypercube Sampler.

        Args:
            n_experiments (int): number of experiments required. Defaults to 1.
            random_state (Optional[int]): pseudo-random number generate state used
                for random uniform sampling from lists of possible values.
                Defaults to None.
        """
        # Pass it on
        super().__init__()
        # Initialise
        self.n_experiments = n_experiments
        self.random_state = random_state

    def _sample(
        self,
        sampleable_parameters: list[SampleableParameter],
        previous_results: Optional[list[Tuple[dict[str, Any], Any]]] = None,
    ) -> list[dict[str, Any]]:
        return list(
            LatinHypercubeParameterSampler(
                variable_parameters=get_parameter_options_dict(sampleable_parameters),
                n_experiments=self.n_experiments,
                random_state=self.random_state,
            )
        )


class AdaptiveSamplerInvalidDefaultValueError(ValueError):
    """Custom error type for gradient adaptive sampler with invalid default."""

    pass


class GradientAdaptiveSampler(Sampler):
    """Sampler implementation embodying an adaptive gradient based approach.

    Ref: https://github.com/python-adaptive/adaptive
    """

    def __init__(self, random: random.Random, default_result: Optional[float] = None) -> None:
        """Create a new GradientAdaptiveSampler.

        Args:
            random (Random): A random number generator to use in the adaptive
                sampler.
            default_result (Optional[float]): Default result to apply to invalid parameter
                combinations in the adaptive sampler. This should be a non-extreme value otherwise
                sample points are taken along the boundary of invalid regions. This must be
                specified if there is a disallowed region. Defaults to None.
        """
        super().__init__()
        self.random = random
        self.default_result = default_result

    def _sample(
        self,
        sampleable_parameters: list[SampleableParameter],
        previous_results: Optional[list[Tuple[dict[str, float], Any]]] = None,
    ) -> list[dict[str, float]]:
        continuous_parameters: list[SampleableContinuousModelParameter] = []
        other_params: list[SampleableParameter] = []
        for param in sampleable_parameters:
            if isinstance(param, SampleableContinuousModelParameter):
                continuous_parameters.append(param)
            else:
                other_params.append(param)
        if len(other_params) > 0:
            warnings.warn(
                f"{len(other_params)} non-continuous sampleable parameters will be "
                "ignored by the Adaptive Sampler.",
                stacklevel=2,
            )

        bounds = [(param.min_val, param.max_val) for param in continuous_parameters]
        # Create a new learner and add any previous results to its data
        learner = ad.LearnerND(func=None, bounds=bounds)  # TODO  explain None
        learner._random = self.random
        if previous_results is not None:
            for param_set, response in previous_results:
                params = tuple(param_set.values())
                if response is not None:
                    learner.tell(params, response)
                else:
                    # give a constant response value to invalid options
                    if self.default_result is None:
                        raise AdaptiveSamplerInvalidDefaultValueError(
                            "Adaptive sampler default value cannot be None if there are invalid "
                            "regions."
                        )
                    learner.tell(params, self.default_result)

        # Get the next experiment from the learner
        next_experiment_loss_pair: tuple[
            list[tuple[Any, ...]], list[Union[Any, list[Any]]]
        ] = learner.ask(1)

        experiment: tuple[float, ...] = next_experiment_loss_pair[0][0]

        next_experiment: dict[str, float] = dict(
            zip(
                [param.name for param in continuous_parameters],
                experiment,
            )
        )
        for param in other_params:
            next_experiment[param.name] = param.get_default_value()

        return [next_experiment]
