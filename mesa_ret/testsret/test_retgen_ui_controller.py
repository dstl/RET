"""Tests for RetGen UI Controller."""
from __future__ import annotations

import random
from pathlib import Path
from warnings import catch_warnings

from dash import Dash
from mesa_ret.testing.mocks import MockParametrisedModel
from retgen.retgen_ui_controller import (
    CoreComponentController,
    ExperimentalControlsController,
    RetGenController,
    ScenarioDependentComponentController,
    SpaceController,
    TimeController,
)


def iterator(id_list: list, component):
    """Iterates through a tree of components to get component IDs.

    Args:
        id_list (list): a list of IDs to be added to.
        component: a Dash component who's ID and that of it's children will be added to the list.
    """
    if hasattr(component, "id"):
        id_list.append(component.id)
    if hasattr(component, "children") and component.children is not None:
        if type(component.children) == list:
            for child in component.children:
                iterator(id_list, child)
        else:
            iterator(id_list, component.children)


class FakeDashApp(Dash):
    """Fake Dash app, that can be bound to each callback produced by a controller."""

    def __init__(self):
        """Create new fake app."""
        self.methods = {}

    def callback(self, *args, **kwargs):
        """This extracts the component ID's of each component that is an input."""

        def func(method):
            for input in args[1:]:
                self.methods[input.component_id] = method
            self.methods[args] = method

        return func


def test_create_scenario_dependent_controller():
    """Test the creation of scenario dependent controller.

    Confirms that methods are bound, and test the execution of each method.
    """
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )

    controller = ScenarioDependentComponentController(
        parameter_specification=parameters.scenario_dependent_data, schema=schema
    )

    app = FakeDashApp()
    for cb in controller.callbacks:
        cb(app)

    # The following checks confirm that there are UI components bound to each of the input
    # fields.

    input_field_ids = []

    for c in controller.components:
        iterator(input_field_ids, c)

    for key in schema.scenario_dependent_parameters.numeric_parameters.keys():
        assert key in app.methods
        assert key in input_field_ids
        assert key + "-alert" in input_field_ids

        # This checks that the bound method modifies the appropriate schema element
        r_val = random.randint(1, 100)
        app.methods[key](r_val)
        assert schema.scenario_dependent_parameters.numeric_parameters[key] == r_val

    for key in schema.scenario_dependent_parameters.categoric_parameters.keys():
        assert key in app.methods
        assert key in input_field_ids
        assert key + "-alert" in input_field_ids

        # This checks that the bound method modifies the appropriate schema element
        r_val = str(random.randint(1, 100))
        app.methods[key](r_val)
        assert schema.scenario_dependent_parameters.categoric_parameters[key] == r_val


def test_create_default_numeric_sd_parameter():
    """Test handling of cases where existing schema is missing numeric scenario parameter."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )
    schema.scenario_dependent_parameters.numeric_parameters.clear()

    with catch_warnings(record=True) as w:
        ScenarioDependentComponentController(
            parameter_specification=parameters.scenario_dependent_data, schema=schema
        )

    assert len(w) == 1
    assert str(w[0].message) == "x1 parameter not in schema. Adding..."


def test_create_default_categoric_sd_parameter():
    """Test handling of cases where existing schema is missing categoric scenario parameter."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )
    schema.scenario_dependent_parameters.categoric_parameters.clear()

    with catch_warnings(record=True) as w:
        ScenarioDependentComponentController(
            parameter_specification=parameters.scenario_dependent_data, schema=schema
        )

    assert len(w) == 1
    assert str(w[0].message) == "y1 parameter not in schema. Adding..."


def test_creation_of_experimental_controls():
    """Test the creation of the experimental controls controller.

    Confirms that methods are bound, and tests the execution of each method.
    """
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )

    controller = ExperimentalControlsController(
        parameter_specification=parameters.experimental_controls, schema=schema
    )

    app = FakeDashApp()
    for cb in controller.callbacks:
        cb(app)

    # The following checks confirm that there are UI components bound to each of the input
    # fields.

    input_field_ids = []

    for c in controller.components:
        iterator(input_field_ids, c)

    for key in schema.experimental_controls.numeric_parameters.keys():
        assert key + "_min" in app.methods
        assert key + "_max" in app.methods
        assert key + "_dist" in app.methods

        assert key + "_min" in input_field_ids
        assert key + "_max" in input_field_ids
        assert key + "_dist" in input_field_ids

        assert key + "-alert" in input_field_ids

        r_val = random.randint(1, 100)
        app.methods[key + "_min"](r_val)
        assert schema.experimental_controls.numeric_parameters[key].min_val == r_val

        r_val = random.randint(1, 100)
        app.methods[key + "_max"](r_val)
        assert schema.experimental_controls.numeric_parameters[key].max_val == r_val

        s_val = str(random.randint(1, 100))
        app.methods[key + "_dist"](s_val)
        assert schema.experimental_controls.numeric_parameters[key].distribution == s_val

    for key in schema.experimental_controls.categoric_parameters.keys():
        assert key in app.methods
        assert key in input_field_ids
        assert key + "-alert" in input_field_ids

        s_vals = [str(random.randint(1, 100)), str(random.randint(1, 100))]
        app.methods[key](s_vals)
        assert schema.experimental_controls.categoric_parameters[key].options == s_vals


def test_create_default_numeric_experimental_parameter():
    """Test handling of cases where existing schema is missing numeric experiment parameter."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )
    schema.experimental_controls.numeric_parameters.clear()

    with catch_warnings(record=True) as w:
        ExperimentalControlsController(
            parameter_specification=parameters.experimental_controls, schema=schema
        )

    assert len(w) == 2
    assert str(w[0].message) == "n1 parameter not in schema. Adding..."
    assert str(w[1].message) == "n2 parameter not in schema. Adding..."


def test_create_default_categoric_experimental_parameter():
    """Test handling of cases where existing schema is missing categoric experiment parameter."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )
    schema.experimental_controls.categoric_parameters.clear()

    with catch_warnings(record=True) as w:
        ExperimentalControlsController(
            parameter_specification=parameters.experimental_controls, schema=schema
        )

    assert len(w) == 1
    assert str(w[0].message) == "c1 parameter not in schema. Adding..."


def test_core_component_controller():
    """Test creation and callbacks for a core component controller."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )

    controller = CoreComponentController(
        model_module="mesa_ret.testing.mocks", model_name="MockParametrisedModel", schema=schema
    )

    app = FakeDashApp()
    for cb in controller.callbacks:
        cb(app)

    # The following checks confirm that there are UI components bound to each of the input
    # fields.

    input_field_ids = []

    for c in controller.components:
        iterator(input_field_ids, c)

    for key in ["Max Steps", "Iterations"]:
        print(app.methods)
        assert key in app.methods
        assert key in input_field_ids

    # There are no updater methods for these fields, as they cannot be modified
    for key in ["Model Name", "Model Module"]:
        assert key in input_field_ids
        assert key not in app.methods

    r_val = random.randint(0, 1000)
    app.methods["Max Steps"](r_val)
    assert schema.max_steps == r_val

    r_val = random.randint(0, 1000)
    app.methods["Iterations"](r_val)
    assert schema.iterations == r_val


def test_time_controller():
    """Test creation and callbacks for a time controller."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )

    controller = TimeController(schema.time)

    app = FakeDashApp()
    for cb in controller.callbacks:
        cb(app)

    # The following checks confirm that there are UI components bound to each of the input
    # fields.

    input_field_ids = []

    for c in controller.components:
        iterator(input_field_ids, c)

    assert "Model Start_date" in input_field_ids
    assert "Model Start_hours" in input_field_ids
    assert "Model Start_minutes" in input_field_ids
    assert "Model End_date" in input_field_ids
    assert "Model End_hours" in input_field_ids
    assert "Model End_minutes" in input_field_ids
    assert "Model Timestep" in input_field_ids

    assert "Model Start_date" in app.methods
    assert "Model Start_hours" in app.methods
    assert "Model Start_minutes" in app.methods
    assert "Model End_date" in app.methods
    assert "Model End_hours" in app.methods
    assert "Model End_minutes" in app.methods
    assert "Model Timestep" in app.methods

    app.methods["Model Start_date"]("2021-01-20")
    assert schema.time.start_time.year == 2021
    assert schema.time.start_time.month == 1
    assert schema.time.start_time.day == 20

    app.methods["Model Start_hours"](5)
    assert schema.time.start_time.year == 2021
    assert schema.time.start_time.month == 1
    assert schema.time.start_time.day == 20
    assert schema.time.start_time.hour == 5

    app.methods["Model Start_minutes"](15)
    assert schema.time.start_time.year == 2021
    assert schema.time.start_time.month == 1
    assert schema.time.start_time.day == 20
    assert schema.time.start_time.hour == 5
    assert schema.time.start_time.minute == 15

    app.methods["Model End_date"]("2021-02-15")
    assert schema.time.end_time.year == 2021
    assert schema.time.end_time.month == 2
    assert schema.time.end_time.day == 15

    app.methods["Model End_hours"](10)
    assert schema.time.end_time.year == 2021
    assert schema.time.end_time.month == 2
    assert schema.time.end_time.day == 15
    assert schema.time.end_time.hour == 10

    app.methods["Model End_minutes"](25)
    assert schema.time.end_time.year == 2021
    assert schema.time.end_time.month == 2
    assert schema.time.end_time.day == 15
    assert schema.time.end_time.hour == 10
    assert schema.time.end_time.minute == 25

    app.methods["Model Timestep"](15)
    assert schema.time.time_step.seconds == 15


def test_space_controller():
    """Test creation and callbacks for a space controller."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )

    controller = SpaceController(schema.space)

    app = FakeDashApp()
    for cb in controller.callbacks:
        cb(app)

    # The following checks confirm that there are UI components bound to each of the input
    # fields.

    input_field_ids = []

    for c in controller.components:
        iterator(input_field_ids, c)

    assert "Space Dimensions" in input_field_ids
    assert "x max" in input_field_ids
    assert "y max" in input_field_ids
    assert "x min" in input_field_ids
    assert "y min" in input_field_ids

    assert "Space Dimensions" in app.methods
    assert "x max" in app.methods
    assert "y max" in app.methods
    assert "x min" in app.methods
    assert "y min" in app.methods

    app.methods["Space Dimensions"](2)
    assert schema.space.dimensions == 2

    app.methods["x max"](10000)
    assert schema.space.x_max == 10000

    app.methods["y max"](200)
    assert schema.space.y_max == 200

    app.methods["x min"](3)
    assert schema.space.x_min == 3

    app.methods["y min"](30)
    assert schema.space.y_min == 30


def test_ret_controller_from_new():
    """Test creating a RetGenController without a pre-existing schema."""
    controller = RetGenController(
        model_module="mesa_ret.testing.mocks", model_class="MockParametrisedModel"
    )
    assert controller.new_schema is not None


def test_ret_controller_from_existing():
    """Test creating a RetGenController from a pre-existing schema."""
    parameters = MockParametrisedModel.get_parameters()
    schema = RetGenController.initialise_schema(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        parameters=parameters,
    )
    controller = RetGenController(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        existing_schema=schema,
    )

    assert controller.new_schema == schema


def test_ret_contoller_save(tmp_path: Path):
    """Test saving a schema.

    Args:
        tmp_path (Path): Temporary path fixture
    """
    path = tmp_path / "output.json"
    controller = RetGenController(
        model_module="mesa_ret.testing.mocks",
        model_class="MockParametrisedModel",
        output_file_location=str(path),
    )

    assert not Path.exists(path)
    controller.save(0)
    assert Path.exists(path)
