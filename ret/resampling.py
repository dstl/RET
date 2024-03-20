"""Main entry point for iterative re-sampling analyses."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ret.batchrunner import FixedReportingBatchRunner

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, OrderedDict, Tuple

    from ret.parameters import SampleableParameter
    from ret.sampling import Sampler


def run_resampling(
    model_cls: type,
    variable_params: list[SampleableParameter],
    sampler: Sampler,
    response: Optional[Callable[[OrderedDict[Any, Any]], Any]] = None,
    iterations: int = 1,
    validity_check: Optional[Callable[[dict[str, Any]], bool]] = None,
    **kwargs,
) -> Tuple[
    list[int], list[Tuple[dict[str, Any], Any]], list[int], list[Tuple[dict[str, Any], Any]]
]:
    """Runs a batch-evaluated Resampling loop with the specified arguments.

    Args:
        model_cls (type): Model Class to use (see FixedReportingBatchRunner).
        variable_params (list[SampleableParameter]): list of Sampleable Parameters
            defining the ranges of values which can be explored in combination.
            This will be translated (via the sampler) into the FixedReportingBatchRunner
            parameters_list.
        sampler (Sampler): Sampler to use.
        response (Optional[Callable[[OrderedDict[Any, Any]], Any]]): Callable defining how to
            interpret the response function from a FixedReportingBatchRunner.model_vars. If None,
            resampling will ignore the response. Defaults to None.
        iterations (int): Number of sampling iterations to run. Defaults to 1.
        validity_check (Optional[Callable[[dict[str, Any]], bool]]): Callable defining validity
            check for variable parameters. If None, no check is run. Defaults to None.
        kwargs (kwargs): Additional keyword arguments into the SamplingFixedReportingBatchRunner.
            Included 'parameters_list' will be ignored and overridden by this algorithm.
            Included 'iterations' will be ignored and overridden by this algorithm.

    Returns:
        (list[int], list[Tuple[dict[str, Any], Any]], list[int], list[Tuple[dict[str, Any], Any]])
        A tuple containing:
            0: list[int] A list of the number of experiments evaluated in each iteration
                (this can also be used to reverse index the other return data) (sum = n_valid)
            1: list[Tuple[dict[str, Any], Any]] A list of all experiments evaluated and
                their response values as Tuples (length = n_valid)
            2: list[int] A list of the number of invalid experiments in each iteration
                (sum = n_invalid)
            3: list[Tuple[dict[str, Any], Any]] A list of all invalid experiments with default
                response values as Tuples (length = n_invalid)
    Note: the number of iterations undertaken (length of return value 0) may be less
        than 'iterations', if subsequent iterations found no new unique experiments to run or
        invalid parameter combinations were sampled.

    Raises:
        ValueError: for any unspecified or ill-shaped arguments
    """
    # Checks
    if not variable_params or len(variable_params) < 1:
        raise ValueError("Variable Parameters must be defined")
    if not sampler:
        raise ValueError("Sampler must be defined")

    # Loop
    i: int = 0
    loop: bool = i < iterations
    all_experiments: list[Tuple[dict[str, Any], Any]] = list()
    invalid_experiments: list[Tuple[dict[str, Any], Any]] = list()
    n_experiments: list[int] = list()
    n_invalid_experiments: list[int] = list()
    last_zero: bool = False
    while loop:
        # Sample the parameters
        parameters_list = sampler.sample(
            variable_params,
            all_experiments + invalid_experiments,
        )

        # Filter for new experiments only
        parameters_list = [
            experiment
            for experiment in parameters_list
            if experiment
            not in [prev_experiment for prev_experiment, _ in all_experiments + invalid_experiments]
        ]

        invalid_parameters: list[dict[str, Any]] = []
        # Filter for valid experiments only
        if validity_check is not None:
            invalid_parameters = [
                experiment for experiment in parameters_list if not validity_check(experiment)
            ]
            parameters_list = [
                experiment for experiment in parameters_list if experiment not in invalid_parameters
            ]

        # Any left?
        n: int = len(parameters_list)
        n_invalid: int = len(invalid_parameters)
        if n == 0:
            # Terminate?
            if last_zero:
                loop = False

            # Flag if no invalid params found
            if n_invalid == 0:
                last_zero = True
            br = None
        else:
            # Run the analyses
            br = FixedReportingBatchRunner(
                model_cls=model_cls, parameters_list=parameters_list, iterations=1, **kwargs
            )
            br.run_all()

        # Capture experiment data and response values
        n_experiments.append(n)
        n_invalid_experiments.append(n_invalid)
        all_experiments += [
            (
                experiment,
                None
                if not response or br is None
                else response(
                    br.model_vars[
                        tuple(experiment[param.name] for param in variable_params) + (ii,)
                    ]
                ),
            )
            for ii, experiment in enumerate(parameters_list)
        ]

        invalid_experiments += [(experiment, None) for experiment in invalid_parameters]

        # Increment
        i += 1
        if i >= iterations:
            loop = False

    # Return
    return (n_experiments, all_experiments, n_invalid_experiments, invalid_experiments)
