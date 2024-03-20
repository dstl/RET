"""Test cases for v2 model definition."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING
from warnings import catch_warnings

from ret.agents.affiliation import Affiliation
from ret.agents.airagent import AirAgent
from ret.io import v2
from ret.io.parser import Parser
from ret.model import RetModel
from ret.parameters import (
    ExperimentalControls,
    ModelParameterSpecification,
    NumericParameterSpecification,
    ScenarioDependentData,
)
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from pytest import mark, raises
from scipy.stats import triang

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from pathlib import Path
    from typing import Any, Callable, Tuple


class ParamModel(RetModel):
    """Example model that can be parametrised."""

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta,
        space: ContinuousSpaceWithTerrainAndCulture2d,
        n_agents: int,
        **kwargs,
    ):
        """Create a new parametrised model.

        Args:
            start_time (datetime): Start time for the simulation
            end_time (datetime): End time for the simulation
            time_step (timedelta): Model time step
            space (ContinuousSpaceWithTerrainAndCulture2d): Space that the model evaluates in
            n_agents (int): Number of agents in the simulation

        """
        self.n_agents = n_agents
        RetModel.__init__(
            self,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step,
            space=space,
            **kwargs,
        )

        for i in range(0, n_agents):
            _ = AirAgent(
                model=self,
                pos=(0, 0),
                name=f"Agent {i}",
                affiliation=Affiliation.NEUTRAL,
            )

    @staticmethod
    def get_parameters() -> ModelParameterSpecification:
        """Returns a list of parameters for this model."""
        experimental_controls = ExperimentalControls(
            numeric_parameters={
                "n_agents": NumericParameterSpecification(
                    name="n_agents",
                    description="Number of agents",
                    min_allowable=1,
                    max_allowable=10,
                )
            },
            categoric_parameters={},
        )

        scenario_dependent_data = ScenarioDependentData(
            numeric_parameters={}, categoric_parameters={}
        )

        return ModelParameterSpecification(
            experimental_controls=experimental_controls,
            scenario_dependent_data=scenario_dependent_data,
        )

    @staticmethod
    def parameter_getter(schema: v2.RetModelSchema) -> Tuple[dict[str, Any], dict[str, list[Any]]]:
        """Returns list of model-invariant parameters.

        Args:
            schema (v2.RetModelSchema): Model schema to convert into parameters_list

        Returns:
            Tuple[dict[str, Any], dict[str, list[Any]]]: Fixed and variable parameters
        """
        fixed_parameters = {
            "start_time": schema.time.start_time,
            "end_time": schema.time.end_time,
            "time_step": schema.time.time_step,
            "space": schema.space.to_model(),
        }

        parameter_list = dict()
        for key, categoric_parameter in schema.experimental_controls.categoric_parameters.items():
            _, p = categoric_parameter.to_model()
            parameter_list[key] = p

        for key, numeric_parameter in schema.experimental_controls.numeric_parameters.items():
            _, p = numeric_parameter.to_model()
            parameter_list[key] = p

        return fixed_parameters, parameter_list

    def get_model_reporters(self) -> dict[str, Callable[[ParamModel], datetime]]:
        """Return a list of model reporters.

        Returns:
            dict[str, Callable[[ParamModel], datetime]]: Dictionary of model reporters
        """

        def get_time(mdl: ParamModel) -> datetime:
            return mdl.get_time()

        return {"timestep": lambda mdl: get_time(mdl)}

    def get_agents(self) -> int:
        """Return the number of agents.

        Returns:
            int: Number of agents
        """
        return self.n_agents


V2_MODEL = """
    {
        "version": 2,
        "data": {
            "iterations": 5,
            "max_steps": 2000,
            "model_module_name": "testsret.test_v2_io",
            "model_name": "ParamModel",
            "output_path": "./ret/testsret/output",
            "playback_writer": "JsonWriter",
            "space": {
                "dimensions": 2,
                "x_max": 100,
                "y_max": 90,
                "x_min": -5,
                "y_min": -10,
                "terrain_image": "",
                "height_black": 1,
                "height_white": 0,
                "culture_image_path": "",
                "cultures": [],
                "clutter_background_level": 0,
                "ground_clutter_height": 0,
                "ground_clutter_value": 0
            },
            "time": {
                "start_time": "2021-01-01T00:00",
                "end_time": "2021-01-02T00:00",
                "time_step": "00:05:00"
            },
            "experimental_controls": {
                "numeric_parameters": {
                    "n_agents": {
                        "name": "n_agents",
                        "min_val": 1,
                        "max_val": 10
                    }
                },
                "categoric_parameters": {}
            },
            "scenario_dependent_parameters": {
                "numeric_parameters": {},
                "categoric_parameters": {}
            },
            "agent_reporters": {
                "id": "unique_id"
            }
        }
    }
"""


def test_run_model(tmp_path: Path):
    """Test that v2 model can via a batchrunner.

    Args:
        tmp_path (Path): Temporary path fixture
    """
    p = tmp_path / "model.json"
    p.write_text(V2_MODEL)

    parser = Parser[v2.RetModelSchema](v2.RetModelSchema)
    model: v2.RetModelSchema = parser.parse(path=str(tmp_path / "model.json"))

    batchrunner = model.to_model()
    batchrunner.output_path
    batchrunner.run_all()

    pass


def test_color_schema_conversion():
    """Test Color schema's methods for converting from schema to model."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    schema = v2.ColorSchema(r=r, g=g, b=b)

    assert schema.to_model() == (r, g, b)


@mark.parametrize("value", [-1, 256])
def test_color_schema_red_validation(value: int):
    """Test validation for creating a colour schema."""
    with raises(ValueError) as ve:
        v2.ColorSchema(r=value, g=0, b=0)
    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Color RGB values must be between 0 and 255."


@mark.parametrize("value", [-1, 256])
def test_color_schema_green_validation(value: int):
    """Test validation for creating a colour schema."""
    with raises(ValueError) as ve:
        v2.ColorSchema(r=0, g=value, b=0)
    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Color RGB values must be between 0 and 255."


@mark.parametrize("value", [-1, 256])
def test_color_schema_blue_validation(value: int):
    """Test validation for creating a colour schema."""
    with raises(ValueError) as ve:
        v2.ColorSchema(r=0, g=0, b=value)
    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Color RGB values must be between 0 and 255."


def test_culture_wavelength_attenuation_to_model():
    """Test CultureWavelengthAttenuationSchema conversion to a model."""
    schema = v2.CultureWavelengthAttenuationSchema(
        wavelength_min=0.0001, wavelength_max=10, attenuation=1.5
    )

    model = schema.to_model()
    assert model == ((0.0001, 10), 1.5)


@mark.parametrize("value", [-1, 0])
def test_culture_wavelength_attenuation_min_validation(value):
    """Test validation of minimum wavelength value."""
    with raises(ValueError) as ve:
        v2.CultureWavelengthAttenuationSchema(
            wavelength_min=value, wavelength_max=10, attenuation=1
        )

    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Minimum wavelength value must be greater than 0."


@mark.parametrize("value", [0.5, 1])
def test_culture_wavelength_attenuation_max_validation(value):
    """Test validation of maximum wavelength value."""
    with raises(ValueError) as ve:
        v2.CultureWavelengthAttenuationSchema(wavelength_min=1, wavelength_max=value, attenuation=1)
    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Maximum wavelength must be greater than minimum wavelength."


@mark.parametrize("value", [0.9999, 0])
def test_culture_wavelength_attenuation_above_one_validation(value):
    """Test validation of attenuation."""
    with raises(ValueError) as ve:
        v2.CultureWavelengthAttenuationSchema(wavelength_min=1, wavelength_max=2, attenuation=value)
    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Attenuation must be greater than or equal to 1."


def test_culture_schema_to_model():
    """Test conversion of culture schema to a model."""
    attenuations = [
        v2.CultureWavelengthAttenuationSchema(wavelength_min=0.1, wavelength_max=10, attenuation=1),
        v2.CultureWavelengthAttenuationSchema(wavelength_min=10, wavelength_max=20, attenuation=2),
    ]
    schema = v2.CultureSchema(name="Culture", height=0.5, wavelength_attenuations=attenuations)

    model = schema.to_model()
    assert model.name == "Culture"
    assert model.height == 0.5
    assert len(model.wavelength_attenuation_factors) == 2


@mark.parametrize("value", [-1, -0.00001])
def test_culture_schema_height_validation(value):
    """Test height validation for culture schema."""
    with raises(ValueError) as ve:
        v2.CultureSchema(name="Culture", height=value, wavelength_attenuations=[])
    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Culture height must be greater than or equal to 0."


def test_colour_culture_to_model():
    """Test conversion of a colour culture schema to a model."""
    color_schema = v2.ColorSchema(r=0, g=0, b=0)
    culture_schema = v2.CultureSchema(name="black", height=0, wavelength_attenuations=[])
    color, culture = v2.CultureColourSchema(color=color_schema, culture=culture_schema).to_model()

    assert color is not None
    assert culture is not None

    assert color == color_schema.to_model()
    assert culture.name == culture_schema.to_model().name


def test_space_schema_3d_to_model():
    """Test creation of 3d model."""
    schema = v2.SpaceSchema(
        dimensions=3,
        x_max=10,
        y_max=10,
        x_min=0,
        y_min=0,
        terrain_image_path=None,
        height_black=1,
        height_white=0,
        culture_image_path=None,
        cultures=[],
        clutter_background_level=0,
        ground_clutter_value=0,
        ground_clutter_height=0,
    )
    assert isinstance(schema.to_model(), ContinuousSpaceWithTerrainAndCulture3d)


def test_space_schema_2d_to_model():
    """Test creation of 2d model."""
    schema = v2.SpaceSchema(
        dimensions=2,
        x_max=10,
        y_max=10,
        x_min=0,
        y_min=0,
        terrain_image_path=None,
        height_black=1,
        height_white=0,
        culture_image_path=None,
        cultures=[],
        clutter_background_level=0,
        ground_clutter_value=0,
        ground_clutter_height=0,
    )
    assert isinstance(schema.to_model(), ContinuousSpaceWithTerrainAndCulture2d)


@mark.parametrize("dimensions", [1, 4])
def test_invalid_space_schema_to_model(dimensions):
    """Test creation of space schema with invalid number of dimensions."""
    with raises(ValueError) as ve:
        v2.SpaceSchema(
            dimensions=dimensions,
            x_max=10,
            y_max=10,
            x_min=0,
            y_min=0,
            terrain_image_path=None,
            height_black=1,
            height_white=0,
            culture_image_path=None,
            cultures=[],
            clutter_background_level=0,
            ground_clutter_value=0,
            ground_clutter_height=0,
        )

    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == f"Invalid number of dimensions: '{dimensions}' - 2 or 3 allowed"


def test_space_schema_with_culture_to_model():
    """Test creation of model where cultures are defined."""
    schema = v2.SpaceSchema(
        dimensions=3,
        x_max=10,
        y_max=10,
        x_min=0,
        y_min=0,
        terrain_image_path=None,
        height_black=1,
        height_white=0,
        culture_image_path=None,
        cultures=[
            v2.CultureColourSchema(
                color=v2.ColorSchema(r=0, g=0, b=0),
                culture=v2.CultureSchema(name="black", height=1, wavelength_attenuations=[]),
            ),
            v2.CultureColourSchema(
                color=v2.ColorSchema(r=255, g=255, b=255),
                culture=v2.CultureSchema(name="white", height=0, wavelength_attenuations=[]),
            ),
        ],
        clutter_background_level=0,
        ground_clutter_value=0,
        ground_clutter_height=0,
    )

    model = schema.to_model()
    assert len(model.get_cultures()) == 2
    assert next((c for c in model.get_cultures() if c.name == "white"), None) is not None
    assert next((c for c in model.get_cultures() if c.name == "black"), None) is not None


def test_numeric_parameter_schema_with_range_to_model():
    """Test parameter schema using a range distribution."""
    schema = v2.NumericParameterSchema(name="test", min_val=1, max_val=10, distribution="range")
    name, options = schema.to_model()

    assert name == "test"
    assert options == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_numeric_parmeter_with_invalid_format_string():
    """Test parameter created using a malformed format string."""
    distribution = "asf][!qfwe"
    with raises(ValueError) as ve:
        v2.NumericParameterSchema(name="test", min_val=1, max_val=10, distribution=distribution)

    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == f"Unable to interpret distribution '{distribution}'"


def test_numeric_parameter_with_non_iterable_format_string():
    """Test parameter creation where the distribution is a valid method, but is not iterable."""
    distribution = "math.ceil({l})"
    with raises(ValueError) as ve:
        v2.NumericParameterSchema(name="test", min_val=1, max_val=10, distribution=distribution)

    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == ("Unable to convert distribution 'math.ceil(1)' to a list of values")


def test_numeric_parameter_with_incorrect_parameter_references():
    """Test parameter creation where the distribution is a valid method, but is not iterable."""
    distribution = "math.ceil({lower})"
    with raises(ValueError) as ve:
        v2.NumericParameterSchema(name="test", min_val=1, max_val=10, distribution=distribution)

    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Unable to interpret keys provided in distribution. {l} and {u} should be used."
    )


def test_unset_distribution_to_model():
    """Test default distribution where none is provided."""
    schema = v2.NumericParameterSchema(name="test", min_val=1, max_val=10, distribution=None)
    name, options = schema.to_model()

    assert name == "test"
    assert options == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_distribution_that_causes_infeasible_lower_values():
    """Test warning messages where a distribution with values outside range are created."""
    schema = v2.NumericParameterSchema(
        name="test", min_val=1, max_val=5, distribution="range(0, 5)"
    )
    with catch_warnings(record=True) as w:
        _ = schema.to_model()

    assert len(w) == 1
    assert (
        str(w[0].message)
        == "At least one value created by the distribution is"
        + " lower than the min value (self.min_val=1)"
    )


def test_distribution_that_causes_infeasible_higher_values():
    """Test warning messages where a distribution with values outside range are created."""
    schema = v2.NumericParameterSchema(
        name="test", min_val=1, max_val=3, distribution="range(1, 5)"
    )
    with catch_warnings(record=True) as w:
        _ = schema.to_model()

    assert len(w) == 1
    assert (
        str(w[0].message)
        == "At least one value created by the distribution is"
        + " greater than the max value (self.max_val=3)"
    )


def test_min_max_validation():
    """Test validation where creating numeric parameter schemas."""
    with raises(ValueError) as ve:
        v2.NumericParameterSchema(name="test", max_val=0, min_val=1, distribution="range")

    errors = ve.value.errors()
    assert len(errors) == 1
    assert errors[0]["msg"] == "Maximum value must be greater or equal to Minimum value"


def test_numpy_function():
    """Test that numpy functions can be used in distributions."""
    schema = v2.NumericParameterSchema(
        name="test", min_val=0, max_val=3, distribution="numpy.arange({l}, {u}, 0.5)"
    )

    _, dist = schema.to_model()
    assert dist == [0, 0.5, 1, 1.5, 2, 2.5]


def test_scipy_function():
    """Test that a complex scipy function can be converted into a distribution."""
    dist = "scipy.stats.triang(c=0.25, loc={l}, scale={u}-{l}).ppf([0, 0.2, 0.4, 0.6, 0.8, 1])"
    schema = v2.NumericParameterSchema(
        name="test",
        min_val=0,
        max_val=3,
        distribution=dist,
    )

    _, vals = schema.to_model()
    assert vals == list(triang(c=0.25, loc=0, scale=3).ppf([0, 0.2, 0.4, 0.6, 0.8, 1]))


def test_categoric_parameter_to_model():
    """Test conversion of a categoric parameter to a model."""
    schema = v2.CategoricParameterSchema(name="param", options=["a", "b"])
    name, values = schema.to_model()

    assert name == "param"
    assert values == ["a", "b"]


def test_scenario_dependent_parameters_to_model():
    """Test conversion of a scenario dependent parameter schema into a model."""
    schema = v2.ScenarioDependentParametersSchema(
        numeric_parameters={"a": 1}, categoric_parameters={"b": "val"}
    )
    numeric, categoric = schema.to_model()
    assert numeric["a"] == 1
    assert categoric["b"] == "val"


def test_playback_writer_validation():
    """Test validation on playback writer settings."""
    with raises(ValueError) as ve:
        v2.RetModelSchema(playback_writer="JosnWriter")

    # Check the last exception - All previous validation exceptions are for missing values in the
    # v2.RetModelSchema constructor
    assert ve.value.errors()[-1]["msg"] == "Unknown playback writer type."
