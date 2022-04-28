"""RetGen UI component generation.

This module contains a series of classes which form a hierarchical wrapper around a RetModelSchema.
The `RetGenController` acts as the top level of this hierarchy, and contains separate controllers
for each of the major elements of the schema.

This hierarchy is responsible for publishing a series of components, which can be injected into a
Dash application in order to form a user interface that depicts the underlying schema, and a
corresponding series of callback functions, each of which accepts a single argument of a Dash app.
This list of methods can be used to programmatically apply callback functions to the application,
which in turn bind changes to the UI to changes to the schema, as well as other UI related
interactivity, such as alerts, and saving the JSON schema to file.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from json import dump, loads
from math import inf
from typing import TYPE_CHECKING
from warnings import warn

from dash.dependencies import Input, Output
from mesa_ret.io.v2 import (
    CategoricParameterSchema,
    ExperimentalControlsSchema,
    NumericParameterSchema,
    RetModelSchema,
    ScenarioDependentParametersSchema,
    SpaceSchema,
    TimeSchema,
)
from mesa_ret.parameters import get_model_metadata, get_model_parameters
from mesa_ret.visualisation.colourlibrary import FNC_GREEN, FNC_MUSTARD, FNC_TURQUOISE

from retgen.retgen_input_types import (
    CategoricParameterComponent,
    DateTimeInputComponent,
    FixedCategoricInputComponent,
    FixedNumericInputComponent,
    FixedNumericRadioInputComponent,
    NumericInputComponent,
    NumericParameterComponent,
    RetGenCollapseSection,
    RetGenModelCard,
    SideBySideComponents,
    TextInputComponent,
)

if TYPE_CHECKING:
    from typing import Callable, Optional

    from dash import Dash
    from dash.development.base_component import Component
    from mesa_ret.parameters import (
        CategoricParameterSpecification,
        ExperimentalControls,
        ModelParameterSpecification,
        NumericParameterSpecification,
        ScenarioDependentData,
    )

    from retgen.retgen_input_types import DashComponent


class ScenarioDependentComponentController:
    """Controller for scenario dependent data."""

    def __init__(
        self,
        parameter_specification: ScenarioDependentData,
        schema: RetModelSchema,
    ):
        """Create a controller for Scenario Dependent data.

        Args:
            parameter_specification (ScenarioDependentData): Specification of parameters that are in
                the scenario independent data
            schema (RetModelSchema): Schema to bind to
        """
        self._components: list[DashComponent] = []

        self.fixed_numeric_parameters: list[FixedNumericInputComponent] = []
        self.fixed_categoric_parameters: list[FixedCategoricInputComponent] = []

        self._generate(parameter_specification, schema)

    @property
    def callbacks(self) -> list[Callable[[Dash], None]]:
        """Return callbacks associated with the model.

        Returns:
            list(Callable[[Dash], None]): List of functions that can be applied to the dash model
        """
        list_of_list_of_callbacks = [c.callbacks for c in self.fixed_numeric_parameters] + [
            c.callbacks for c in self.fixed_categoric_parameters
        ]

        cbs: list[Callable[[Dash], None]] = []
        for sublist in list_of_list_of_callbacks:
            cbs.extend(sublist)

        return cbs

    @property
    def components(self) -> list[Component]:
        """Return list of Dash components that can be used to display the model.

        Returns:
            list(Component): List of dash components
        """
        list_of_list_of_components = [c.components for c in self.fixed_numeric_parameters] + [
            c.components for c in self.fixed_categoric_parameters
        ]

        comps: list[Component] = []
        for sublist in list_of_list_of_components:
            comps.extend(sublist)

        return comps

    def _generate(
        self,
        parameter_specification: ScenarioDependentData,
        schema: RetModelSchema,
    ):
        """Generates components and callbacks for scenario dependent data.

        Args:
            parameter_specification (ScenarioDependentData): Scenario dependent data parameters
            schema (RetModelSchema): Schema that backs the model
        """
        for numeric_key, numeric_parameter in parameter_specification.numeric_parameters.items():
            if numeric_key not in schema.scenario_dependent_parameters.numeric_parameters:
                warn(f"{numeric_key} parameter not in schema. Adding...")
                schema.scenario_dependent_parameters.numeric_parameters[
                    numeric_key
                ] = numeric_parameter.min_allowable
            numeric_control = self._generate_numeric_control(
                numeric_key, numeric_parameter, schema.scenario_dependent_parameters
            )

            self.fixed_numeric_parameters.append(numeric_control)

        for (
            categoric_key,
            categoric_parameter,
        ) in parameter_specification.categoric_parameters.items():
            if categoric_key not in schema.scenario_dependent_parameters.categoric_parameters:
                warn(f"{categoric_key} parameter not in schema. Adding...")
                schema.scenario_dependent_parameters.categoric_parameters[
                    categoric_key
                ] = categoric_parameter.allowable_options[0]
            categoric_control = self._generate_categoric_control(
                categoric_key, categoric_parameter, schema.scenario_dependent_parameters
            )
            self.fixed_categoric_parameters.append(categoric_control)

    def _generate_numeric_control(
        self,
        key: str,
        parameter: NumericParameterSpecification,
        schema: ScenarioDependentParametersSchema,
    ) -> FixedNumericInputComponent:
        """Generates a single numeric control parameter.

        Args:
            key: Name of the parameter
            parameter (NumericParameterSpecification): Specification of the parameter.
            schema (ScenarioDependentParametersSchema): Parameter schema to update where the
                parameter is changed.

        Returns:
            FixedNumericInputComponent: New parameter model
        """

        def new_sd_numeric_updater(key: str, value) -> None:
            schema.numeric_parameters[key] = value

        return FixedNumericInputComponent(
            dash_title=key,
            dash_description=parameter.description,
            value=schema.numeric_parameters[key],
            min_val=parameter.min_allowable,
            max_val=parameter.max_allowable,
            affected_fields=parameter.affected_fields,
            model_updater=lambda value: new_sd_numeric_updater(key=key, value=value),
        )

    def _generate_categoric_control(
        self,
        key: str,
        parameter: CategoricParameterSpecification,
        schema: ScenarioDependentParametersSchema,
    ) -> FixedCategoricInputComponent:
        """Generates a single categoric control parameter.

        Args:
            key: Name of the parameter
            parameter (CategoricParameterSpecification): Specification of the parameter.
            schema (ScenarioDependentParametersSchema): Parameter schema to update where the
                parameter is changed.

        Returns:
            FixedCategoricInputComponent: New categoric parameter
        """

        def new_sd_categoric_updater(key: str, value) -> None:
            schema.categoric_parameters[key] = value

        return FixedCategoricInputComponent(
            dash_title=key,
            dash_description=parameter.description,
            value=schema.categoric_parameters[key],
            allowable=parameter.allowable_options,
            affected_fields=parameter.affected_fields,
            model_updater=lambda value: new_sd_categoric_updater(key=key, value=value),
        )


class ExperimentalControlsController:
    """Controller for experimental controls."""

    def __init__(
        self,
        parameter_specification: ExperimentalControls,
        schema: RetModelSchema,
    ):
        """Return a new ExperimentalControlsController.

        Args:
            parameter_specification (ExperimentalControls): Specification of the parameters to be
                included in the experimental controls
            schema (RetModelSchema): Existing definition, and reference to update where values
                change
        """
        self.numeric_parameters: list[NumericParameterComponent] = []
        self.categoric_parameters: list[CategoricParameterComponent] = []
        self._generate(parameter_specification, schema)

    @property
    def callbacks(self) -> list[Callable[[Dash], None]]:
        """Return callbacks associated with the model.

        Returns:
            list(Callable[[Dash], None]): List of functions that can be applied to the dash model
        """
        list_of_list_of_callbacks = [c.callbacks for c in self.numeric_parameters] + [
            c.callbacks for c in self.categoric_parameters
        ]

        cbs: list[Callable[[Dash], None]] = []
        for sublist in list_of_list_of_callbacks:
            cbs.extend(sublist)

        return cbs

    @property
    def components(self) -> list[Component]:
        """Return list of Dash components that can be used to display the model.

        Returns:
            list(Component): List of dash components
        """
        list_of_list_of_components = [c.components for c in self.numeric_parameters] + [
            c.components for c in self.categoric_parameters
        ]

        comps: list[Component] = []
        for sublist in list_of_list_of_components:
            comps.extend(sublist)

        return comps

    def _generate(
        self,
        parameters: ExperimentalControls,
        schema: RetModelSchema,
    ):
        """Generates component and callbacks for experimental controls.

        Args:
            parameters (ExperimentalControls): Experimental controls for the model
            schema (RetModelSchema): The schema that the model writes to
        """
        self._generate_numeric(parameters, schema)
        self._generate_categoric(parameters, schema)

    def _generate_numeric(
        self,
        parameters: ExperimentalControls,
        schema: RetModelSchema,
    ):
        """Generates component and callbacks for experimental controls.

        Args:
            parameters (ExperimentalControls): Experimental controls for the model
            schema (RetModelSchema): The new schema that the model writes to
        """
        for k_num, p_num in parameters.numeric_parameters.items():
            if k_num not in schema.experimental_controls.numeric_parameters:
                schema.experimental_controls.numeric_parameters[
                    k_num
                ] = NumericParameterSchema.create(p_num)
                warn(f"{k_num} parameter not in schema. Adding...")

            existing_numeric = schema.experimental_controls.numeric_parameters[k_num]
            component = self._generate_numeric_control(k_num, p_num, existing_numeric)
            self.numeric_parameters.append(component)

    def _generate_numeric_control(
        self, key: str, parameter: NumericParameterSpecification, schema: NumericParameterSchema
    ) -> NumericParameterComponent:
        """Generate a numeric control.

        Args:
            key (str): Name of the parameter
            parameter (NumericParameterSpecification): Specification of the parameter
            schema (NumericParameterSchema): Extract of the schema to update on change

        Returns:
            NumericParameterComponent: New parameter model
        """

        def min_updater(m) -> None:
            schema.min_val = m

        def max_updater(m) -> None:
            schema.max_val = m

        def distribution_updater(d) -> None:
            schema.distribution = d

        return NumericParameterComponent(
            dash_title=key,
            dash_description=parameter.description,
            minimum_value=schema.min_val,
            maximum_value=schema.max_val,
            min_allowable=parameter.min_allowable,
            max_allowable=parameter.max_allowable,
            selected_distribution=schema.distribution,
            affected_fields=parameter.affected_fields,
            min_model_updater=min_updater,
            max_model_updater=max_updater,
            distribution_updater=distribution_updater,
        )

    def _generate_categoric(
        self,
        parameters: ExperimentalControls,
        schema: RetModelSchema,
    ):
        """Generates component and callbacks for experimental controls.

        Args:
            parameters (ExperimentalControls): Experimental controls for the model
            schema (RetModelSchema): The schema that the model writes to
        """
        for k_cat, p_cat in parameters.categoric_parameters.items():
            if k_cat not in schema.experimental_controls.categoric_parameters:
                schema.experimental_controls.categoric_parameters[k_cat] = CategoricParameterSchema(
                    name=k_cat, options=[]
                )
                warn(f"{k_cat} parameter not in schema. Adding...")

            existing_categoric = schema.experimental_controls.categoric_parameters[k_cat]
            component = self._generate_categoric_control(k_cat, p_cat, existing_categoric)
            self.categoric_parameters.append(component)

    def _generate_categoric_control(
        self, key: str, parameter: CategoricParameterSpecification, schema: CategoricParameterSchema
    ) -> CategoricParameterComponent:
        """Generate a categoric control.

        Args:
            key (str): Name of the parameter
            parameter (CategoricParameterSpecification): Specification of the parameter
            schema (CategoricParameterSchema): Extract of the schema to update on change

        Returns:
            CategoricParameterComponent: New parameter model
        """

        def choice_updater(lst) -> None:
            schema.options = lst

        return CategoricParameterComponent(
            dash_title=key,
            options=parameter.allowable_options,
            chosen_options=schema.options,
            dash_description=parameter.description,
            affected_fields=parameter.affected_fields,
            model_updater=choice_updater,
        )


class CoreComponentController:
    """Controller for core model parameters."""

    def __init__(self, model_name: str, model_module: str, schema: RetModelSchema):
        """Create a new Core Component Controller.

        Args:
            model_name (str): Name of the model class to control.
            model_module (str): Name of the module containing the model
            schema (RetModelSchema): Underlying schema backing the controller.
        """
        self._components: list[DashComponent] = []

        self._space = SpaceController(schema.space)
        self._time = TimeController(schema.time)

        self._new_schema = schema
        self._generate(model_name, model_module, schema)

    @property
    def components(self) -> list[Component]:
        """Return list of Dash components that can be used to display the model.

        Returns:
            list (Component): List of dash components
        """
        comps: list[Component] = self._space.components + self._time.components
        for component in self._components:
            comps.extend(component.components)

        return comps

    @property
    def callbacks(self) -> list[Callable[[Dash], None]]:
        """Return callbacks associated with the model.

        Returns:
            list(Callable[[Dash], None]): List of functions that can be applied to the dash model
        """
        cbs: list[Callable[[Dash], None]] = self._space.callbacks + self._time.callbacks
        for component in self._components:
            cbs.extend(component.callbacks)

        return cbs

    def _generate(
        self,
        model_name: str,
        model_module: str,
        schema: RetModelSchema,
    ):
        """Generate components and callbacks for core model parameters.

        Args:
            model_name (str): The name of the model to render
            model_module (str): The name of the module to render
            schema (RetModelSchema): The  schema that the model writes to
        """
        core_parameters: list[DashComponent] = []

        def model_updater(s) -> None:
            raise NotImplementedError("This should never be called.")

        model_name_component = TextInputComponent(
            dash_title="Model Name",
            dash_description="Name of the model",
            text_value=model_name,
            model_updater=model_updater,
            readonly=True,
        )

        core_parameters.append(model_name_component)

        model_module_component = TextInputComponent(
            dash_title="Model Module",
            dash_description="Name of the model's module",
            text_value=model_module,
            model_updater=model_updater,
            readonly=True,
        )

        core_parameters.append(model_module_component)

        iterations = schema.iterations

        def iterations_updater(i) -> None:
            schema.iterations = i

        iterations_component = NumericInputComponent[int](
            dash_title="Iterations",
            dash_description="Number of times the model is ran by the batch runner",
            numerical_value=iterations,
            minimum_value=1,
            maximum_value=10000000,
            model_updater=iterations_updater,
        )

        core_parameters.append(iterations_component)

        max_steps = schema.max_steps

        def max_steps_updater(ms) -> None:
            schema.max_steps = ms

        max_steps_component = NumericInputComponent[int](
            dash_title="Max Steps",
            dash_description="Maximum number of steps a model can run for",
            numerical_value=max_steps,
            minimum_value=1,
            maximum_value=10000000,
            model_updater=max_steps_updater,
        )

        core_parameters.append(max_steps_component)

        self._components = core_parameters


class TimeController:
    """Controller for time parameters."""

    def __init__(self, schema: TimeSchema):
        """Return a new TimeController.

        Args:
            schema (TimeSchema): The schema to bind to for updating.
        """
        self._generate(schema)

    @property
    def components(self) -> list[Component]:
        """Return list of Dash components that can be used to display the model.

        Returns:
            list (Component): List of dash components
        """
        comps: list[Component] = []
        for component in self._components:
            comps.extend(component.components)

        return comps

    @property
    def callbacks(self) -> list[Callable[[Dash], None]]:
        """Return callbacks associated with the model.

        Returns:
            list(Callable[[Dash], None]): List of functions that can be applied to the dash model
        """
        cbs: list[Callable[[Dash], None]] = self._callbacks
        for component in self._components:
            cbs.extend(component.callbacks)

        return cbs

    def _generate(self, schema: TimeSchema):
        """Generate components and controllers for time-based parameters.

        Args:
            schema (TimeSchema): The new schema that the model writes to

        """

        def update_start_day(d: str) -> None:
            tmp = schema.start_time
            new_date = datetime.strptime(d, "%Y-%m-%d")

            schema.start_time = datetime(
                year=new_date.year,
                month=new_date.month,
                day=new_date.day,
                hour=tmp.hour,
                minute=tmp.minute,
            )

        def update_start_hours(h: int) -> None:
            tmp = schema.start_time

            schema.start_time = datetime(
                year=tmp.year, month=tmp.month, day=tmp.day, hour=h, minute=tmp.minute
            )

        def update_start_minute(m: int) -> None:
            tmp = schema.start_time

            schema.start_time = datetime(
                year=tmp.year, month=tmp.month, day=tmp.day, hour=tmp.hour, minute=m
            )

        start_date_time = DateTimeInputComponent(
            dash_title="Model Start",
            dash_description="The date and time at which the model starts",
            date_time=schema.start_time,
            datetime_updater=update_start_day,
            hours_updater=update_start_hours,
            minutes_updater=update_start_minute,
            hide_divider=True,
        )

        def update_end_day(d: str) -> None:
            tmp = schema.end_time
            new_date = datetime.strptime(d, "%Y-%m-%d")

            schema.end_time = datetime(
                year=new_date.year,
                month=new_date.month,
                day=new_date.day,
                hour=tmp.hour,
                minute=tmp.minute,
            )

        def update_end_hours(h: int) -> None:
            tmp = schema.end_time

            schema.end_time = datetime(
                year=tmp.year, month=tmp.month, day=tmp.day, hour=h, minute=tmp.minute
            )

        def update_end_minute(m: int) -> None:
            tmp = schema.end_time

            schema.end_time = datetime(
                year=tmp.year, month=tmp.month, day=tmp.day, hour=tmp.hour, minute=m
            )

        end_date_time = DateTimeInputComponent(
            dash_title="Model End",
            dash_description="The date and time at which the model ends",
            date_time=schema.end_time,
            datetime_updater=update_end_day,
            hours_updater=update_end_hours,
            minutes_updater=update_end_minute,
            hide_divider=True,
        )

        def update_timestep(t) -> None:
            schema.time_step = timedelta(seconds=t)

        time_step_component = NumericInputComponent[int](
            dash_title="Model Timestep",
            dash_description="Time that passes with each model step (in seconds)",
            numerical_value=schema.time_step.seconds,
            minimum_value=0,
            maximum_value=100000,
            model_updater=update_timestep,
        )

        self._components = [
            SideBySideComponents(
                [start_date_time.components, end_date_time.components],
            ),
            time_step_component,
        ]

        self._callbacks: list[Callable[[Dash], None]] = (
            start_date_time.callbacks + end_date_time.callbacks
        )


class SpaceController:
    """Contoller for space parameters."""

    def __init__(self, schema: SpaceSchema):
        """Returns a new Space Controller.

        Args:
            schema (SpaceSchema): The schema to bind to for updating.
        """
        self._generate(schema)

    @property
    def components(self) -> list[Component]:
        """Return list of Dash components that can be used to display the model.

        Returns:
            list (Component): List of dash components
        """
        comps: list[Component] = []
        for component in self._components:
            comps.extend(component.components)

        return comps

    @property
    def callbacks(self) -> list[Callable[[Dash], None]]:
        """Return callbacks associated with the model.

        Returns:
            list(Callable[[Dash], None]): List of functions that can be applied to the dash model
        """
        cbs: list[Callable[[Dash], None]] = self._callbacks
        for component in self._components:
            cbs.extend(component.callbacks)

        return cbs

    def _generate(self, schema: SpaceSchema):
        """Generate components and callbacks for spatial parameter.

        Args:
            schema (SpaceSchema): The schema to bind to
        """
        return_list: list[DashComponent] = []

        dimensions = schema.dimensions

        def update_dimensions(d) -> None:
            schema.dimensions = d

        return_list.append(
            FixedNumericRadioInputComponent(
                dash_title="Space Dimensions",
                dash_description="Number of dimensions that the model space has",
                value=dimensions,
                allowable_values=[2, 3],
                affected_fields=None,
                model_updater=update_dimensions,
            )
        )

        x_max = schema.x_max

        def update_x_max(x) -> None:
            schema.x_max = x

        x_max_components = NumericInputComponent[float](
            dash_title="x max",
            dash_description="x distance at which the model ends",
            numerical_value=x_max,
            minimum_value=-inf,
            maximum_value=inf,
            model_updater=update_x_max,
            hide_divider=True,
            units="m,",
        )

        y_max = schema.y_max

        def update_y_max(y) -> None:
            schema.y_max = y

        y_max_components = NumericInputComponent[float](
            dash_title="y max",
            dash_description="y distance at which the model ends",
            numerical_value=y_max,
            minimum_value=-inf,
            maximum_value=inf,
            model_updater=update_y_max,
            hide_divider=True,
            units="m,",
        )

        x_min = schema.x_min

        def update_x_min(x) -> None:
            schema.x_min = x

        x_min_components = NumericInputComponent[float](
            dash_title="x min",
            dash_description="x distance at which the model starts",
            numerical_value=x_min,
            minimum_value=-inf,
            maximum_value=inf,
            model_updater=update_x_min,
            hide_divider=True,
            units="m,",
        )

        y_min = schema.y_min

        def update_y_min(y) -> None:
            schema.y_min = y

        y_min_components = NumericInputComponent(
            dash_title="y min",
            dash_description="y distance at which the model starts",
            numerical_value=y_min,
            minimum_value=-inf,
            maximum_value=inf,
            model_updater=update_y_min,
            hide_divider=True,
            units="m,",
        )

        return_list.append(
            SideBySideComponents(
                [
                    x_min_components.components,
                    x_max_components.components,
                    y_min_components.components,
                    y_max_components.components,
                ],
                "Space Boundaries:",
            )
        )

        self._components = return_list
        self._callbacks: list[Callable[[Dash], None]] = (
            x_max_components.callbacks
            + y_max_components.callbacks
            + x_min_components.callbacks
            + y_min_components.callbacks
        )


class RetGenController:
    """Controller for Dash components for a model."""

    def __init__(
        self,
        model_module: str,
        model_class: str,
        existing_schema: Optional[RetModelSchema] = None,
        output_file_location="output.json",
    ):
        """Create a new RetGenComponentGenerator.

        Args:
            model_class (str): The name of the Model class to convert into Dash components.
            model_module (str): The Python module containing the model class to convert into Dash
                components.
            existing_schema:Optional[RetModelSchema]: Existing model schema
            output_file_location (str): Location to save output file to. Defaults to "output.json"
        """
        self._model_module = model_module
        self._model_class = model_class
        self._parameters = get_model_parameters(self._model_module, self._model_class)
        self._model_metadata = get_model_metadata(self._model_module, self._model_class)
        self.output_file_location = output_file_location

        self.new_schema: RetModelSchema = existing_schema  # type: ignore

        if self.new_schema is None:
            self.new_schema = self.initialise_schema(model_module, model_class, self._parameters)

        self._top_header = RetGenModelCard(
            self._model_metadata.header, self._model_metadata.subtext
        )

        self._core_components = CoreComponentController(model_class, model_module, self.new_schema)
        self._core_ui = RetGenCollapseSection(
            "Core Parameters",
            FNC_GREEN,
            self._core_components.components,
            "core_components_expander",
        )

        self._scenario_dependent_parameters = ScenarioDependentComponentController(
            self._parameters.scenario_dependent_data, self.new_schema
        )
        self._scenario_ui = RetGenCollapseSection(
            "Scenario Dependent Data",
            FNC_MUSTARD,
            self._scenario_dependent_parameters.components,
            "scenario_data_expander",
        )

        self._experimental_controls = ExperimentalControlsController(
            self._parameters.experimental_controls, self.new_schema
        )
        self._experimental_ui = RetGenCollapseSection(
            "Experimental Controls",
            FNC_TURQUOISE,
            self._experimental_controls.components,
            "experimental_controls_expander",
        )

    @property
    def components(self) -> list[Component]:
        """Generate HTML components to represent a UI.

        Returns:
            list[Components]: Components in the model
        """
        return (
            self._top_header.components
            + self._core_ui.components
            + self._scenario_ui.components
            + self._experimental_ui.components
        )

    @property
    def callbacks(self) -> list[Callable[[Dash], None]]:
        """Generate callbacks for the model.

        Each callback takes the dash app as an argument, and registers itself to the model.

        Returns:
            list[Callable[[Dash], None]: Callbacks to register
        """

        def register_save(app: Dash) -> None:

            app.callback(
                Output(component_id="save", component_property="n_clicks"),
                Input(component_id="save", component_property="n_clicks"),
                prevent_initial_call=True,
            )(self.save)

        cb: list[Callable[[Dash], None]] = [register_save]

        return (
            cb
            + self._core_components.callbacks
            + self._core_ui.callbacks
            + self._scenario_dependent_parameters.callbacks
            + self._scenario_ui.callbacks
            + self._experimental_controls.callbacks
            + self._experimental_ui.callbacks
        )

    def save(self, n_clicks: int) -> int:
        """Save to file.

        Args:
            n_clicks (int): Counter for number of times the save button has been clicked (unused)

        Returns:
            int: Number of clicks
        """
        print(f"Saving output file to {self.output_file_location}...")

        j = loads(self.new_schema.json())
        json_version = {"version": 2, "data": j}

        with open(self.output_file_location, "w") as f:
            dump(json_version, f, indent=4, sort_keys=True)

        return n_clicks

    @staticmethod
    def initialise_schema(
        model_module: str, model_class: str, parameters: ModelParameterSpecification
    ) -> RetModelSchema:
        """Initialise a new RET Schema, based on model model, class, and parameters.

        Args:
            model_module (str): The module containing the model class
            model_class (str): The model class to represent in the schema.
            parameters (ModelParameterSpecification): Parameter specification for the model.

        Returns:
            RetModelSchema: Newly initialised RetModelSchema with default properties allocated based
                on the content of their respective parameters.
        """
        ecs = parameters.experimental_controls
        sdd = parameters.scenario_dependent_data
        return RetModelSchema(
            time=TimeSchema(
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(days=1),
                time_step=timedelta(minutes=15),
            ),
            space=SpaceSchema(
                dimensions=2,
                x_max=1000,
                y_max=1000,
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
            ),
            model_name=model_class,
            model_module_name=model_module,
            iterations=10,
            max_steps=1000,
            scenario_dependent_parameters=ScenarioDependentParametersSchema(
                numeric_parameters=dict(
                    (v, c.min_allowable) for v, c in sdd.numeric_parameters.items()
                ),
                categoric_parameters=dict(
                    (v, c.allowable_options[0]) for v, c in sdd.categoric_parameters.items()
                ),
            ),
            experimental_controls=ExperimentalControlsSchema(
                numeric_parameters=dict(
                    (
                        v,
                        NumericParameterSchema.create(c),
                    )
                    for v, c in ecs.numeric_parameters.items()
                ),
                categoric_parameters=dict(
                    (v, CategoricParameterSchema(name=v, options=c.allowable_options))
                    for v, c in ecs.categoric_parameters.items()
                ),
            ),
        )
