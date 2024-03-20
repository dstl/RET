"""Version 2 RET Model Definition File IO Layer."""

from __future__ import annotations

# math, numpy and scipy.stats are imported to allow automatic translation of users' distributions
# which wouldn't be possible without these libraries
import math  # noqa flake8: F401, TC002

# Any, Optional, datetime and timedelta have to be imported directly in order to support
# their use as types within pydantic models.
from datetime import datetime, timedelta  # noqa flake8: TC002
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple  # noqa flake8: TC002
from warnings import warn

import numpy  # noqa flake8: F401, TC002
import scipy.stats  # noqa flake8: F401, TC002
from ret.batchrunner import LatinHypercubeBatchRunner
from ret.parameters import get_model
from ret.space.culture import Culture
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from ret.visualisation import get_playback_writer_register
from pydantic import BaseModel, validator

if TYPE_CHECKING:
    from typing import Callable

    from ret.parameters import NumericParameterSpecification
    from ret.types import Color

PARAMETER_GETTER_ATTRIBUTE_STR = "parameter_getter"
MODEL_REPORTER_ATTRIBUTE_STR = "get_model_reporters"


class ColorSchema(BaseModel):
    """Model for V2 Colour definition."""

    r: int
    g: int
    b: int

    @validator("*")
    @classmethod
    def color_value_must_be_0_to_255(cls, v: int) -> int:
        """Validates RGB values.

        Args:
            v (int): Color value

        Returns:
            (int) Validated color value

        Raises:
            ValueError: Color value outside allowable range
        """
        if not 0 <= v <= 255:
            raise ValueError("Color RGB values must be between 0 and 255.")
        return v

    def to_model(self) -> Color:
        """Convert model to Color.

        Returns:
            Color: RGB value tuple
        """
        return (self.r, self.g, self.b)


class TimeSchema(BaseModel):
    """Model for V2 Time Definition."""

    start_time: datetime
    end_time: datetime
    time_step: timedelta


class CultureWavelengthAttenuationSchema(BaseModel):
    """Model for V2 wavelength attenuation definition."""

    wavelength_min: float
    wavelength_max: float
    attenuation: float

    @validator("wavelength_min")
    @classmethod
    def wavelength_min_above_zero(cls, v: float) -> float:
        """Validates wavelength minimum value.

        Args:
            v (float): Wavelength value to validate.

        Raises:
            ValueError: Invalid wavelength (less than zero)

        Returns:
            float: Validated minimum wavelength
        """
        if v <= 0:
            raise ValueError("Minimum wavelength value must be greater than 0.")
        return v

    @validator("wavelength_max")
    @classmethod
    def wavelength_max_rule(cls, v: float, values: dict[str, float]) -> float:
        """Validates wavelength minimum value.

        Args:
            v (float): Wavelength value to validate
            values (dict[str, float]): Possible wavelength values

        Raises:
            ValueError: Maximum wavelength is less than minimum wavelength

        Returns:
            float: Validated maximum wavelength
        """
        if "wavelength_min" in values and v <= values["wavelength_min"]:
            raise ValueError("Maximum wavelength must be greater than minimum wavelength.")
        return v

    @validator("attenuation")
    @classmethod
    def attenuation_above_one(cls, v: float) -> float:
        """Validates attenuation value.

        Args:
            v (float): Attenuation value

        Raises:
            ValueError: Invalid attenuation value

        Returns:
            float: Validated attenuation value
        """
        if v < 1:
            raise ValueError("Attenuation must be greater than or equal to 1.")
        return v

    def to_model(self) -> tuple[tuple[float, float], float]:
        """Convert to model.

        Returns:
            tuple[tuple[float, float], float]: Wavelength band
        """
        return ((self.wavelength_min, self.wavelength_max), self.attenuation)


class CultureSchema(BaseModel):
    """Model for V2 culture definition."""

    name: str
    height: float
    wavelength_attenuations: list[CultureWavelengthAttenuationSchema]

    @validator("height")
    @classmethod
    def height_above_zero(cls, v: float) -> float:
        """Validates culture height.

        Args:
            v (float): height

        Raises:
            ValueError: Invalid height value

        Returns:
            float: Validated height
        """
        if v < 0:
            raise ValueError("Culture height must be greater than or equal to 0.")
        return v

    def to_model(self) -> Culture:
        """Convert to Culture.

        Returns:
            Culture: Culture
        """
        wavelength_attenuation: dict[tuple[float, float], float] = {}

        for wavelength in self.wavelength_attenuations:
            wavelength_band, value = wavelength.to_model()
            wavelength_attenuation[wavelength_band] = value

        return Culture(
            name=self.name,
            height=self.height,
            wavelength_attenuation_factors=wavelength_attenuation,
        )


class CultureColourSchema(BaseModel):
    """Model for V2 culture/color pair definition."""

    color: ColorSchema
    culture: CultureSchema

    def to_model(self) -> tuple[Color, Culture]:
        """Convert to color/culture tuple.

        Returns:
            tuple[Color, Culture]: Color/Culture tuple
        """
        color = self.color.to_model()
        culture = self.culture.to_model()
        return (color, culture)


class SpaceSchema(BaseModel):
    """Model for V2 Space Definition."""

    dimensions: int
    x_max: float
    y_max: float
    x_min: float
    y_min: float
    terrain_image_path: Optional[str]
    height_black: float
    height_white: float
    culture_image_path: Optional[str]
    cultures: list[CultureColourSchema]
    clutter_background_level: float
    ground_clutter_value: float
    ground_clutter_height: float

    @validator("dimensions")
    @classmethod
    def invalid_dimension(cls, v: int) -> int:
        """Validate number of dimensions in model.

        Args:
            v (int): Number of dimensions

        Returns:
            int: Validated number of dimensions

        Raises:
            ValueError: Invalid number of dimensions in model
        """
        if v not in [2, 3]:
            raise ValueError(f"Invalid number of dimensions: '{v}' - 2 or 3 allowed")
        return v

    def to_model(self) -> ContinuousSpaceWithTerrainAndCulture2d:
        """Convert schema to RET model.

        Returns:
            ContinuousSpaceWithTerrainAndCulture2d: Terrain and culture model
        """
        cultures: dict[Color, Culture] = {}
        for culture_color_pair in self.cultures:
            color, culture = culture_color_pair.to_model()
            cultures[color] = culture

        culture_image_path = None
        if self.culture_image_path != "":
            culture_image_path = self.culture_image_path

        terrain_image_path = None
        if self.terrain_image_path != "":
            terrain_image_path = self.terrain_image_path

        if self.dimensions == 2:
            return ContinuousSpaceWithTerrainAndCulture2d(
                x_max=self.x_max,
                y_max=self.y_max,
                x_min=self.x_min,
                y_min=self.y_min,
                terrain_image_path=terrain_image_path,
                height_black=self.height_black,
                height_white=self.height_white,
                culture_image_path=culture_image_path,
                culture_dictionary=cultures,
                clutter_background_level=self.clutter_background_level,
                ground_clutter_height=self.ground_clutter_height,
                ground_clutter_value=self.ground_clutter_value,
            )
        else:
            return ContinuousSpaceWithTerrainAndCulture3d(
                x_max=self.x_max,
                y_max=self.y_max,
                x_min=self.x_min,
                y_min=self.y_min,
                terrain_image_path=terrain_image_path,
                height_black=self.height_black,
                height_white=self.height_white,
                culture_image_path=culture_image_path,
                culture_dictionary=cultures,
                clutter_background_level=self.clutter_background_level,
                ground_clutter_height=self.ground_clutter_height,
                ground_clutter_value=self.ground_clutter_value,
            )


class NumericParameterSchema(BaseModel):
    """Model for numeric parameters."""

    name: str
    min_val: Any
    max_val: Any
    distribution: Optional[str] = None

    @staticmethod
    def create(spec: NumericParameterSpecification) -> NumericParameterSchema:
        """Create a default NumericParameterSchema from specification."""
        min_val = spec.to_default_min()
        max_val = spec.to_default_max()

        return NumericParameterSchema(
            name=spec.name, min_val=min_val, max_val=max_val, distribution=spec.default_distribution
        )

    @validator("max_val")
    @classmethod
    def validate_max(cls, v: Any, values: dict[str, Any]) -> Any:
        """Validate settings for max val for numeric parameter.

        Args:
            v (Any): Max val
            values (dict[str, Any]): All arguments to NumericParameterSchema

        Returns:
            Any: Validated max val

        Raises:
            ValueError: Min val is greater than max val.
        """
        if (
            "min_val" in values
            and v is not None
            and values["min_val"] is not None
            and v < values["min_val"]
        ):
            raise ValueError("Maximum value must be greater or equal to Minimum value")
        return v

    @validator("distribution")
    @classmethod
    def validate_distribution(cls, v: Optional[str], values: dict[str, Any]) -> Optional[str]:
        """Validate distribution.

        Args:
            v (Optional[str]): Distribution name
            values (dict[str, Any]): All arguments to NumericParameterSchema

        Returns:
            Optional[str]: Distribution

        Raises:
            ValueError: Invalid distribution parameters
        """
        if "min_val" not in values or "max_val" not in values:
            return v

        min_val = values["min_val"]
        max_val = values["max_val"]

        try:
            cls._to_distribution(v, min_val, max_val)
            return v
        except ValueError:
            raise
        except KeyError:
            raise ValueError(
                "Unable to interpret keys provided in distribution. {l} and {u} should be used."
            )
        except Exception:
            raise ValueError(f"Unable to interpret distribution '{v}'")

    @staticmethod
    def _to_distribution(distribution: Optional[str], min_val: Any, max_val: Any) -> list[Any]:
        """Convert distribution type and min/max values to a list of values.

        Args:
            distribution (Optional[str]): The name of the distribution to use
            min_val (Any): Minimum allowable value
            max_val (Any): Maximum allowable value

        Returns:
            list[Any]: Distribution of parameters

        Raises:
            ValueError: Unable to interpret the distribution as an iterable list of values.
        """
        if distribution == "range" or distribution is None:
            min_val_int = round(min_val)
            max_val_int = round(max_val)
            return list(range(min_val_int, max_val_int + 1))
        else:
            exec_content = distribution.format(l=min_val, u=max_val)
            evaluated_distribution = eval(exec_content)
            if isinstance(evaluated_distribution, Iterable):
                values = list(evaluated_distribution)

                return values
            else:
                raise ValueError(
                    "Unable to convert distribution" + f" '{exec_content}' to a list of values"
                )

    def to_model(self) -> tuple[str, list[Any]]:
        """Returns a model representation of the parameter schema.

        Once a range of distributions have been implemented, this will be updated to sample them
        appropriately where converting to a model.

        Returns:
            tuple[str, list[Any]]: Parameter name and parameter options
        """
        values = self._to_distribution(self.distribution, self.min_val, self.max_val)

        warn_lower = any(v < self.min_val for v in values)

        if warn_lower:
            warn(
                "At least one value created by the distribution is "
                + f"lower than the min value ({self.min_val=})"
            )

        warn_upper = any(v > self.max_val for v in values)

        if warn_upper:
            warn(
                "At least one value created by the distribution is "
                + f"greater than the max value ({self.max_val=})"
            )

        return self.name, self._to_distribution(self.distribution, self.min_val, self.max_val)


class CategoricParameterSchema(BaseModel):
    """Model schema for categoric parameters."""

    name: str
    options: list[str]

    def to_model(self) -> tuple[str, list[Any]]:
        """Returns a model representation of the parameter schema.

        Returns:
            tuple[str, list[str]]: Parameter name and parameter options
        """
        return self.name, self.options


class ExperimentalControlsSchema(BaseModel):
    """Model for experimental controls."""

    numeric_parameters: dict[str, NumericParameterSchema]
    categoric_parameters: dict[str, CategoricParameterSchema]


class ScenarioDependentParametersSchema(BaseModel):
    """Model for scenario dependent parameters."""

    numeric_parameters: dict[str, Any]
    categoric_parameters: dict[str, str]

    def to_model(self) -> Tuple[dict[str, Any], dict[str, str]]:
        """Returns a model representation of the scenario dependent parameters.

        Returns:
            dict[str, Any]: Scenario dependent parameters and values.
        """
        return self.numeric_parameters, self.categoric_parameters


class RetModelSchema(BaseModel):
    """Model for V2 Model Definition."""

    time: TimeSchema
    space: SpaceSchema
    model_name: str
    model_module_name: str
    iterations: int
    max_steps: int
    experimental_controls: ExperimentalControlsSchema
    scenario_dependent_parameters: ScenarioDependentParametersSchema
    playback_writer: str = "None"
    n_experiments: int = 1
    agent_reporters: Optional[dict[str, str]] = None
    random_state: int = 0
    collect_datacollector: bool = True

    @validator("playback_writer")
    @classmethod
    def playback_writer_type_valid(cls, v: str) -> str:
        """Validates playback writer type.

        Args:
            v (str): Playback writer type

        Raises:
            ValueError: Unknown playback writer type

        Returns:
            str: Validated playback writer
        """
        writers = get_playback_writer_register().get_register_items()
        if v not in writers and v != "None":
            raise ValueError("Unknown playback writer type.")
        return v

    def to_model(self) -> LatinHypercubeBatchRunner:
        """Convert model into a runnable batch-runner.

        Returns:
            LatinHypercubeBatchRunner: Configured batchrunner.
        """
        # This determines the model class to be run by the batch-runner
        model = get_model(self.model_module_name, self.model_name)

        variable_parameters: Optional[dict[str, list[Any]]] = None
        fixed_parameters: Optional[dict[str, Any]] = None
        output_path: str = "./ret/testsret/output"

        # individual model classes have a defined method for converting a schema
        # into the table of input parameters required to pass into the batchrunner
        if hasattr(model, "parameter_getter"):
            fixed_parameters, variable_parameters = model.parameter_getter(self)  # type: ignore

        model_reporters: Optional[dict[str, Callable]] = None

        if hasattr(model, "get_model_reporters"):
            model_reporters = model.get_model_reporters(model)  # type: ignore

        if variable_parameters == {}:
            variable_parameters = None

        return LatinHypercubeBatchRunner(
            model_cls=model,
            output_path=output_path,
            variable_parameters=variable_parameters,
            fixed_parameters=fixed_parameters,
            iterations=self.iterations,
            max_steps=self.max_steps,
            model_reporters=model_reporters,
            agent_reporters=self.agent_reporters,
            n_experiments=self.n_experiments,
            random_state=self.random_state,
            collect_datacollector=self.collect_datacollector,
            ignore_multiprocessing=True,
        )
