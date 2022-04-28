"""Tests for RetGen input types."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from retgen.retgen_input_types import (
    CategoricParameterComponent,
    DateTimeInputComponent,
    FixedCategoricInputComponent,
    FixedNumericInputComponent,
    NumericInputComponent,
    NumericParameterComponent,
    TextInputComponent,
)

from testsret.test_retgen_ui_controller import FakeDashApp, iterator

if TYPE_CHECKING:
    from retgen.retgen_input_types import NumberType


def test_numeric_input():
    """Test that a numeric input runs its updater method when modified."""
    state = []

    def update(t: int) -> None:
        state.append(t)

    numeric_input = NumericInputComponent[int](
        dash_title="Title",
        dash_description="Desc",
        numerical_value=3,
        minimum_value=1,
        maximum_value=4,
        model_updater=update,
    )

    app = FakeDashApp()

    for cb in numeric_input.callbacks:
        cb(app)

    assert "Title" in app.methods
    app.methods["Title"](4)

    assert 4 in state


def test_datetime_input():
    """Test that a datetime input runs its updater method when modifier."""
    state = {}

    def update_date(d: str) -> None:
        state["date"] = d

    def update_hours(h: int) -> None:
        state["hours"] = h

    def update_minutes(m: int) -> None:
        state["minutes"] = m

    datetime_input = DateTimeInputComponent(
        dash_title="Title",
        dash_description="Desc",
        date_time=datetime(year=2021, month=1, day=1, hour=3, minute=5),
        datetime_updater=update_date,
        hours_updater=update_hours,
        minutes_updater=update_minutes,
    )

    app = FakeDashApp()

    for cb in datetime_input.callbacks:
        cb(app)

    assert "Title_date" in app.methods
    assert "Title_hours" in app.methods
    assert "Title_minutes" in app.methods

    app.methods["Title_date"]("2020-10-30")
    assert state["date"] == "2020-10-30"

    app.methods["Title_hours"](3)
    assert state["hours"] == 3

    app.methods["Title_minutes"](5)
    assert state["minutes"] == 5


def test_text_input_editable():
    """Test creation of an editable input."""
    state = []

    def update(t: str) -> None:
        state.append(t)

    text_input = TextInputComponent(
        dash_title="Title", dash_description="Desc", text_value="some value", model_updater=update
    )

    app = FakeDashApp()

    for cb in text_input.callbacks:
        cb(app)

    assert "Title" in app.methods

    app.methods["Title"](1)
    assert 1 in state


def test_text_input_readonly():
    """Test creation of an input in readonly mode."""
    state = []

    def update(t: str) -> None:
        state.append(t)

    text_input = TextInputComponent(
        dash_title="Title",
        dash_description="Desc",
        text_value="some value",
        model_updater=update,
        readonly=True,
    )

    app = FakeDashApp()

    for cb in text_input.callbacks:
        cb(app)

    assert "Title" not in app.methods


def test_categoric_parameter():
    """Test creation of a categoric parameter."""
    state = []

    def updater(ss: list[str]) -> None:
        state.append(ss)

    param = CategoricParameterComponent(
        dash_title="Title",
        dash_description="Desc",
        options=["a", "b", "c"],
        chosen_options=["a", "c"],
        affected_fields=None,
        model_updater=updater,
    )

    app = FakeDashApp()

    for cb in param.callbacks:
        cb(app)

    assert "Title" in app.methods

    input_field_ids = []

    for c in param.components:
        iterator(input_field_ids, c)

    assert "Title" in input_field_ids
    assert "Title-alert" in input_field_ids

    app.methods["Title"](["b", "c"])

    assert ["b", "c"] in state


def test_numeric_parameter():
    """Test creation of a numeric parameter."""
    state = {}

    def min_updater(m: NumberType) -> None:
        state["min"] = m

    def max_updater(m: NumberType) -> None:
        state["max"] = m

    def dist_updater(s: str) -> None:
        state["dist"] = s

    numeric_parameter = NumericParameterComponent(
        dash_title="title",
        dash_description="Desc",
        minimum_value=0,
        maximum_value=10,
        min_allowable=0,
        max_allowable=20,
        selected_distribution="range",
        affected_fields=None,
        min_model_updater=min_updater,
        max_model_updater=max_updater,
        distribution_updater=dist_updater,
    )

    app = FakeDashApp()

    for cb in numeric_parameter.callbacks:
        cb(app)

    input_field_ids = []

    for c in numeric_parameter.components:
        iterator(input_field_ids, c)

    assert "title_min" in app.methods
    assert "title_max" in app.methods
    assert "title_dist" in app.methods

    assert "title_min" in input_field_ids
    assert "title_max" in input_field_ids
    assert "title_dist" in input_field_ids

    assert "title-alert" in input_field_ids

    app.methods["title_min"](1)
    assert state["min"] == 1

    app.methods["title_max"](3)
    assert state["max"] == 3

    app.methods["title_dist"]("range(0, 10)")
    assert state["dist"] == "range(0, 10)"


def test_fixed_categoric_parameter():
    """Test creation of a categoric parameter with a single value."""
    state = []

    def updater(v: str) -> None:
        state.append(v)

    param = FixedCategoricInputComponent(
        dash_title="Title",
        dash_description="Desc",
        allowable=["a", "b", "c"],
        affected_fields=None,
        value="a",
        model_updater=updater,
    )

    app = FakeDashApp()

    for cb in param.callbacks:
        cb(app)

    input_field_ids = []

    for c in param.components:
        iterator(input_field_ids, c)

    assert "Title" in app.methods

    assert "Title" in input_field_ids
    assert "Title-alert" in input_field_ids

    app.methods["Title"]("c")

    assert "c" in state


def test_fixed_numeric_parameter():
    """Test creation of a numeric parameter with a single value."""
    state = []

    def updater(v: NumberType) -> None:
        state.append(v)

    param = FixedNumericInputComponent(
        dash_title="Title",
        dash_description="Desc",
        min_val=0,
        max_val=100,
        affected_fields=None,
        value=10,
        model_updater=updater,
    )

    app = FakeDashApp()

    for cb in param.callbacks:
        cb(app)

    assert "Title" in app.methods

    input_field_ids = []

    for c in param.components:
        iterator(input_field_ids, c)

    assert "Title" in input_field_ids
    assert "Title-alert" in input_field_ids

    app.methods["Title"](30)

    assert 30 in state
