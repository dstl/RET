"""Model parameters."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import import_module
from sys import modules
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    from ret.scenario_independent_data import ModelMetadata

T = TypeVar("T", int, float, datetime, timedelta)


@dataclass
class ExperimentalControls:
    """Specification for experimental controls."""

    numeric_parameters: dict[str, NumericParameterSpecification]
    categoric_parameters: dict[str, CategoricParameterSpecification]


@dataclass
class ScenarioDependentData:
    """Specification for Scenario Dependent data."""

    numeric_parameters: dict[str, NumericParameterSpecification]
    categoric_parameters: dict[str, CategoricParameterSpecification]


@dataclass
class ModelParameterSpecification:
    """Base class for a set of model parameters.

    This class contains two sub-divisions. `experimental_controls` contains specifications for
    parameters for which the user can provide a range of values. `scenario_dependent_data` contains
    parameters for which the user can provide a single value per parameter.
    """

    experimental_controls: ExperimentalControls
    scenario_dependent_data: ScenarioDependentData


@dataclass
class NumericParameterSpecification(Generic[T]):
    """Specification for a numeric parameter.

    Note that for Instances of NumericParameterSpecification where T is datetime, timedelta, or any
    other variant where initialisation to '0' is not valid, it is essential for either the min value
    or the max value to be defined, else it is not possible to determine the default properties for
    a new instance of the parameter.
    """

    name: str
    description: str
    min_allowable: Optional[T] = None
    max_allowable: Optional[T] = None
    affected_fields: Optional[list[str]] = None
    default_distribution: str = "range"

    def to_default_min(self) -> T:
        """Returns a default value to be used as a minimum value.

        Returns:
            T: New default value
        """
        if self.min_allowable is not None:
            return self.min_allowable

        if self.max_allowable is None:
            # Note - This is a bug - In cases where T is a datetime or timedelta, it will not be
            # resolved correctly, and it's not possible to interpret the type from min_allowable and
            # max_allowable as both are None.
            # Furthermore, T cannot be determined at runtime directly, as it is defined for type
            # checking purposes only.
            return 0  # type: ignore

        if isinstance(self.max_allowable, datetime):
            return self.max_allowable - timedelta(days=1)

        # This syntax allows the following logic to be valid for timedeltas as well as int/float,
        # as timedeltas cannot be compared directly against a numeric
        if self.max_allowable >= 0 * self.max_allowable:
            return 0 * self.max_allowable  # type: ignore
        else:
            # As the value is negative, this return value will be lower than max_allowable
            return 2 * self.max_allowable

    def to_default_max(self) -> T:
        """Returns a default value to be used as the maximum value.

        Returns:
            T: New max value
        """
        if self.max_allowable is not None:
            return self.max_allowable

        if self.min_allowable is None:
            # Note - This is a bug - In cases where T is a datetime or timedelta, it will not be
            # resolved correctly, and it's not possible to interpret the type from min_allowable and
            # max_allowable as both are None.
            # Furthermore, T cannot be determined at runtime directly, as it is defined for type
            # checking purposes only
            return 1  # type: ignore

        if isinstance(self.min_allowable, datetime):
            return self.min_allowable + timedelta(days=1)

        # This syntax allows the following logic to be valid for timedeltas as well as int/float,
        # as timedeltas cannot be compared directly against a numeric
        if self.min_allowable < (0 * self.min_allowable):
            return 0 * self.min_allowable  # type: ignore
        if self.min_allowable == 0:
            return 1  # type: ignore
        if isinstance(self.min_allowable, timedelta):
            if self.min_allowable.total_seconds() == 0:
                return timedelta(seconds=1)  # type: ignore

        return 2 * self.min_allowable


@dataclass
class CategoricParameterSpecification:
    """Specification for a categoric parameter."""

    name: str
    description: str
    allowable_options: list[str]
    affected_fields: Optional[list[str]] = None


@dataclass
class ModelParameter:
    """Base class for model parameters."""

    name: str


class SampleableParameter(ABC, ModelParameter):
    """Base class for sampleable parameters."""

    @abstractmethod
    def get_options(self) -> list[Any]:  # pragma: no cover
        """Get discrete options for sampleable parameter.

        Returns:
            list[Any]: A list of discrete parameter options.
        """
        pass

    @abstractmethod
    def get_default_value(self) -> Any:  # pragma: no cover
        """Get default value for the parameter.

        Returns:
            Any: The default value.
        """
        pass


@dataclass
class NumericModelParameter(ModelParameter, Generic[T]):
    """Model parameter for numeric data."""

    min_val: Optional[T] = None
    max_val: Optional[T] = None


class CategoricModelParameter(SampleableParameter):
    """Model parameter for categoric data."""

    _default_value: Optional[str]
    options: list[str]
    item_description: Callable[[str], str]

    def __init__(
        self,
        name: str,
        options: list[str],
        default_value: Optional[str] = None,
        item_description: Callable[[str], str] = lambda cat: cat,
    ) -> None:
        """Initialise a new categoric model parameter.

        Args:
            name (str): Parameter name
            options (list[str]): Parameter options
            default_value (Optional[float], optional): Default value. Defaults to first option.
            item_description (Callable[[str], str]): Function to get description from options.
        """
        if len(options) == 0:
            raise ValueError("options cannot be empty.")

        super().__init__(name=name)
        self.options = options
        self._default_value = options[0] if default_value is None else default_value
        self.item_description = item_description  # type: ignore

    def get_descriptions(self) -> list[str]:
        """Get list of descriptions.

        Returns:
            list[str]: List of descriptions
        """
        return [self.item_description(g) for g in self.options]  # type: ignore

    def get_options(self) -> list[Any]:
        """Get discrete options for sampleable parameter.

        Returns:
            list[Any]: A list of discrete parameter options.
        """
        return self.options

    def get_default_value(self) -> Any:
        """Get default value for the parameter.

        Returns:
            Any: The default value.
        """
        return self._default_value


class SampleableContinuousModelParameter(NumericModelParameter[float], SampleableParameter):
    """Sampleable continuous numeric model parameter."""

    name: str
    min_val: float
    max_val: float
    sampling_distance: float
    _default_value: Optional[float]

    def __init__(
        self,
        name: str,
        min_val: float,
        max_val: float,
        sampling_distance: Optional[float] = None,
        default_value: Optional[float] = None,
    ):
        """Initialise a SampleableContinuousModelParameter.

        Args:
            name (str): Parameter name.
            min_val (float): Minimum value in parameter range.
            max_val (float): Maximum value in parameter range.
            sampling_distance (Optional[float]): Optional sampling distance for generating
                discrete parameter options.
            default_value (Optional[float], optional): Default value. Defaults to first option.
        """
        if sampling_distance is None:
            sampling_distance = max_val - min_val
        self.sampling_distance = sampling_distance
        self._default_value = min_val if default_value is None else default_value
        super().__init__(name=name, min_val=min_val, max_val=max_val)

    def get_options(self) -> list[Any]:
        """Get discrete options for sampleable parameter.

        Returns:
            list[Any]: A list of discrete parameter options.
        """
        n = math.ceil((self.max_val - self.min_val) / self.sampling_distance)
        return [self.min_val + self.sampling_distance * i for i in range(n)] + [self.max_val]

    def get_default_value(self) -> Any:
        """Get default value for the parameter.

        Returns:
            Any: The default value.
        """
        return self._default_value


def get_model(model_module: str, model_class: str) -> type:
    """Get the model class based on a module and class name.

    Args:
        model_module (str): Name of the parent module for the model class of interest.
        model_class (str): Name of the model class of interest.

    Returns:
        type: Model type.
    """
    import_module(model_module)

    model: type = getattr(modules[model_module], model_class)  # type ignore
    return model


def get_model_parameters(model_module: str, model_class: str) -> ModelParameterSpecification:
    """Gets a ModelParameterSpecification containing parameters for the requested class.

    Args:
        model_module (str): Name of the parent module for the model class of interest.
        model_class (str): Name of the model class of interest.

    Returns:
        ModelParameterSpecification: Model parameter specification
    """
    model: ModelParameterSpecification = get_model(
        model_module, model_class
    ).get_parameters()  # type: ignore
    return model


def get_model_metadata(model_module: str, model_class: str) -> ModelMetadata:
    """Gets a ModelMetadata containing information about the requested class.

    Args:
        model_module (str): Name of the parent module for the model class of interest.
        model_class (str): Name of the model class of interest.

    Returns:
        MetalMetadata: Model metadata
    """
    metadata: ModelMetadata = get_model(
        model_module, model_class
    ).get_scenario_independent_metadata()  # type: ignore

    return metadata


def get_categoric_parameter_options(
    model_module: str, model_class: str, parameter: str
) -> list[str]:
    """Gets a list of valid (string) options for a parameter.

    Where categoric information in non-string based, e.g., where it is a class, such as a list of
    sensor types, weapons, etc., the most effective implementation is to use a string to represent
    each option, and then use reflection or an equivalent resolution mechanism to translate the
    strings into object code.

    Args:
        model_module (str): Name of the parent module for the model class of interest.
        model_class (str): Name of the model class of interest.
        parameter (str): Parameter name.

    Returns:
        list[str]: Categoric list

    Raises:
        ValueError: Invalid `model_module`, `model_class` or `parameter` provided by the user.
    """
    params: ModelParameterSpecification = get_model_parameters(
        model_module, model_class
    )  # type: ignore

    try:
        param = params.experimental_controls.categoric_parameters[parameter]

        return param.allowable_options

    except KeyError:
        raise ValueError(f"'{parameter}' not defined for model.")


def get_numeric_min_max_values(
    model_module: str, model_class: str, parameter: str
) -> tuple[Optional[T], Optional[T]]:
    """Gets the minimum and maximum value allowed for a given parameter name and model.

    Args:
        model_module (str): Name of the parent module for the model class of interest.
        model_class (str): Name of the model class of interest.
        parameter (str): Parameter name.

    Returns:
        tuple[Optional[T], Optional[T]]: Min and Max values allowed for the parameter.

    Raises:
        ValueError: Invalid `model_module`, `model_class` or `parameter` provided by the user.
    """
    params: ModelParameterSpecification = get_model_parameters(
        model_module, model_class
    )  # type: ignore

    try:
        param = params.experimental_controls.numeric_parameters[parameter]

        return param.min_allowable, param.max_allowable

    except KeyError:
        raise ValueError(f"'{parameter}' not defined for model.")
