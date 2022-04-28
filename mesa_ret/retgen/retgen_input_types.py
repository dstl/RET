"""Low Level UI components for RET Gen, representing single values, parameters, etc."""

from __future__ import annotations

from abc import ABC
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Generic, TypeVar, Union

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from mesa_ret.visualisation.colourlibrary import FNC_LIGHT_PURPLE, FNC_PURPLE, FNC_RED

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    from dash import Dash
    from dash.development.base_component import Component

NumberType = Union[float, int, datetime, timedelta]


class DashComponent(ABC):
    """Abstract class defining an input control that can be included in the RetGen UI."""

    def __init__(self):
        """Create a new DashComponent."""
        self.components: list[Component] = []
        self.callbacks: list[Callable[[Dash], None]] = []

    def _build_affected_fields(self, ids: list[str], triggers: list[Input]):
        """Build alerts for affected field notifications.

        Args:
            ids (list[str]): List of IDs to trigger alerts on
            triggers (list[Input]): List of triggers that cause the alert
        """
        for alert_id in ids:
            self._build_affected_field(alert_id, triggers)

    def _build_affected_field(self, key, triggers: list[Input]):
        """Build the alert for a single field that must be notified of a change.

        Args:
            key (str): Component name to trigger an alert on
            triggers (list[Input]): List of inputs that cause the alert to trigger
        """

        def callback(app: Dash, k: str, t: list[Input]) -> None:
            def toggle(*args) -> bool:
                return True

            app.callback(
                Output(component_id=k + "-alert", component_property="is_open"),
                t,
                State(component_id=k + "-alert", component_property="is_open"),
                prevent_initial_call=True,
            )(toggle)

        self.callbacks.append(lambda a: callback(a, key, triggers))


T = TypeVar("T", int, float, datetime, timedelta)


class NumericInputComponent(DashComponent, Generic[T]):
    """RetGen numeric input.

    This parameter is used for basic numeric input, where the type of the numeric data can be
    known in advance.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        numerical_value: T,
        minimum_value: T,
        maximum_value: T,
        model_updater: Callable[[T], None],
        units: Optional[str] = None,
        hide_divider: bool = False,
    ) -> None:
        """Instantiate NumericInputComponent object.

        Args:
            dash_title (str): title of dash Component.
            dash_description (str): description of dash component.
            numerical_value (T): value of parameter.
            minimum_value (T): minimum value of parameter.
            maximum_value (T): maximum value of parameter.
            model_updater (Callable[[T], None]): Function that is applied where the user control is
                changed.
            units (Optional[str]): The units to display for the numeric input. Defaults to None.
            hide_divider (bool): Set True to disable the horizontal line divider for this component.
                Defaults to False.
        """
        super().__init__()

        self.numerical_value: T = numerical_value
        self.minimum_value: T = minimum_value
        self.maximum_value: T = maximum_value
        self._model_updater: Callable[[T], None] = model_updater
        self.description = dash_description
        self.title = dash_title
        self.id = dash_title
        self.units = units

        if not hide_divider:
            self.components.extend([dcc.Markdown("___")])

        self.components.extend(
            [
                dbc.Row(html.B(f"{self.title}:")),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Input(
                                id=self.id,
                                type="number",
                                value=self.minimum_value,
                                min=self.minimum_value,
                                max=self.maximum_value,
                            ),
                            width=10,
                        ),
                        dbc.Col(html.A(self.units), align="end"),
                    ]
                ),
                dbc.Row(dbc.FormText(self.description)),
            ]
        )

        def updater(app: Dash) -> None:
            print(f"Registering updater for {self.title}")

            def update(new_val: T) -> T:
                print(f"Updating {self.title} in schema to {new_val}")
                self._model_updater(new_val)

                return new_val

            app.callback(
                Output(self.id, "value"), Input(self.id, "value"), prevent_initial_call=True
            )(update)

        self.callbacks.append(updater)


class DateTimeInputComponent(DashComponent):
    """RetGen datetime input.

    This UI component is used for datetime selection.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        date_time: datetime,
        datetime_updater: Callable[[str], None],
        hours_updater: Callable[[int], None],
        minutes_updater: Callable[[int], None],
        hide_divider: bool = False,
    ) -> None:
        """Instantiate DateTimeInputComponent object.

        Args:
            dash_title (str): title of Dash Component.
            dash_description (str): description of dash component.
            date_time (datetime): datetime value of input.
            datetime_updater (Callable[str]): Updater for datetime values
            hours_updater (Callable[int]): Updater for hours values
            minutes_updater (Callable[int]): Updater for minutes values
            hide_divider (bool): Set True to disable the horizontal line divider for this component.
                Defaults to False.
        """
        super().__init__()
        self.date_time = date_time
        self.title = dash_title
        self.description = dash_description
        self.date_id = self.title + "_date"
        self.hour_id = self.title + "_hours"
        self.minute_id = self.title + "_minutes"
        self.datetime_updater = datetime_updater
        self.hours_updater = hours_updater
        self.minutes_updater = minutes_updater

        if not hide_divider:
            self.components.extend([dcc.Markdown("___")])

        self.components.extend(
            [
                dbc.Row(html.B(f"{self.title}:")),
                dbc.Row(html.A("Date:")),
                dbc.Row(
                    dcc.DatePickerSingle(
                        id=self.date_id,
                        initial_visible_month=self.date_time,
                        date=self.date_time,
                        className="DateInput_input",
                        display_format="DD/MM/YYYY",
                    )
                ),
                dbc.Row(html.A("Time:")),
                dbc.Row(
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.Input(
                                    id=self.hour_id,
                                    type="number",
                                    value=self.date_time.hour,
                                    min=0,
                                    max=23,
                                ),
                                dbc.InputGroupText(":", style={"background-color": "white"}),
                                dbc.Input(
                                    id=self.minute_id,
                                    type="number",
                                    value=self.date_time.minute,
                                    min=0,
                                    max=59,
                                ),
                            ],
                        ),
                        width=4,
                    )
                ),
            ]
        )

        def dt_updater(app: Dash) -> None:
            print(f"Registering datetime updater for {self.title}")

            def update(new_val: str) -> str:
                print(f"Updating {self.title} datetime in schema to {new_val}")
                self.datetime_updater(new_val)

                return new_val

            app.callback(
                Output(self.date_id, "date"), Input(self.date_id, "date"), prevent_initial_call=True
            )(update)

        def hrs_updater(app: Dash) -> None:
            print(f"Registering updater for hours for {self.title}")

            def update(hours: int) -> int:
                print(f"Updating {self.title} minutes in schema to {hours=}")
                self.hours_updater(hours)

                return hours

            app.callback(
                Output(self.hour_id, "value"),
                Input(self.hour_id, "value"),
                prevent_initial_call=True,
            )(update)

        def mins_updater(app: Dash) -> None:
            print(f"Registering updater for minutes for {self.title}")

            def update(minutes: int) -> int:
                print(f"Updating {self.title} minutes in schema to {minutes=}")
                self.minutes_updater(minutes)

                return minutes

            app.callback(
                Output(self.minute_id, "value"),
                Input(self.minute_id, "value"),
                prevent_initial_call=True,
            )(update)

        self.callbacks.append(hrs_updater)
        self.callbacks.append(dt_updater)
        self.callbacks.append(mins_updater)


class TextInputComponent(DashComponent):
    """RetGen text input.

    This UI component is used for plain entry boxes, with a title and description.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        text_value: str,
        model_updater: Callable[[str], None],
        readonly: bool = False,
    ) -> None:
        """Instantiate TextInputComponent object.

        Args:
            dash_title (str): title of Dash Component.
            dash_description (str): description of dash component.
            text_value (str): value of text input.
            model_updater (Callable[[str], None]): Function that is applied where the user control
                is changed.
            readonly (bool): Whether or not the entry can be modified
            hide_divider (bool): Set True to disable the horizontal line divider for this component.
                Defaults to False.
        """
        super().__init__()
        self.text_value = text_value
        self.title = dash_title
        self.description = dash_description
        self._model_updater = model_updater
        self.id = dash_title

        self.components.extend(
            [
                dcc.Markdown("___"),
                dbc.Row(html.B(f"{self.title}:")),
                dbc.Row(
                    [
                        dbc.InputGroup(
                            dbc.Input(
                                id=self.title,
                                type="text",
                                value=self.text_value,
                                disabled=readonly,
                            )
                        ),
                    ]
                ),
                dbc.Row(dbc.FormText(self.description)),
            ]
        )

        def updater(app: Dash) -> None:
            print(f"Registering updater for {self.title}")

            def update(new_val: str) -> str:
                print(f"Updating {self.title} in schema to {new_val}")
                self._model_updater(new_val)

                return new_val

            app.callback(
                Output(self.id, "value"), Input(self.id, "value"), prevent_initial_call=True
            )(update)

        if not readonly:
            self.callbacks.append(updater)


class CategoricParameterComponent(DashComponent):
    """RetGen categoric input.

    This UI component is used where a user can pick several options from a categoric list.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        options: list[str],
        chosen_options: list[str],
        affected_fields: Optional[list[str]],
        model_updater: Callable[[list[str]], None],
    ) -> None:
        """Instantiate CategoricParameterComponent.

        Intended to display numerical parameters that will be varied when passed to the
        batchrunner.

        Args:
            dash_title (str): title of Dash Component.
            dash_description (str): description of dash component.
            options (list[str]): list of category titles.
            chosen_options (list[str]): options that have been selected.
            affected_fields (Optional[list[str]]): List of fields that are affected by changing
                this parameter.
            model_updater (Callable[list[str]]): Function that is applied where the user control is
                changed.
        """
        super().__init__()

        self.options = options
        self.chosen_options = chosen_options
        self.affected_fields = affected_fields
        self.title = dash_title
        self.id = dash_title
        self.description = dash_description
        self._model_updater = model_updater

        checklist = dbc.Checklist(
            id=self.id,
            options=[{"label": o, "value": o} for o in self.options],
            value=self.chosen_options,
            inline=True,
        )
        self.components.extend(
            [
                dcc.Markdown("___"),
                dbc.Row(html.B(f"{self.title}:")),
                checklist,
                dbc.Row(dbc.FormText(self.description)),
                dbc.Alert(
                    "This parameter needs reviewing",
                    id=self.id + "-alert",
                    color="warning",
                    is_open=False,
                    dismissable=True,
                ),
            ]
        )

        def updater(app: Dash) -> None:
            print(f"Registering updater for {self.title}")

            def update(new_val: list[str]) -> list[str]:
                print(f"Updating {self.title} in schema to {new_val}")
                self._model_updater(new_val)

                return new_val

            app.callback(
                Output(self.id, "value"), Input(self.id, "value"), prevent_initial_call=True
            )(update)

        self.callbacks.append(updater)

        if self.affected_fields is not None and len(self.affected_fields) > 0:
            triggers: list[Input] = [
                Input(component_id=self.id, component_property="value"),
            ]
            self._build_affected_fields(self.affected_fields, triggers)


class NumericParameterComponent(DashComponent):
    """RetGen numeric parameter.

    The UI component is used for numeric data, where the user can define an upper and lower bound
    to a range, and define a mathematical function to dictate how values are selected from this
    range.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        minimum_value: NumberType,
        maximum_value: NumberType,
        min_allowable: Optional[NumberType],
        max_allowable: Optional[NumberType],
        selected_distribution: Optional[str],
        affected_fields: Optional[list[str]],
        min_model_updater: Callable[[NumberType], None],
        max_model_updater: Callable[[NumberType], None],
        distribution_updater: Callable[[str], None],
    ) -> None:
        """Instantiate NumericParameterComponent.

        Intended to display numerical parameters that will be varied when passed to the
            batchrunner.

        Args:
            dash_title (str): title of Dash Component.
            dash_description (str): description of dash component.
            minimum_value (NumberType): minimum value of parameter.
            maximum_value (NumberType): maximum value of parameter.
            min_allowable (Optional[NumberType]): The lowest allowable setting for min val
            max_allowable (Optional[NumberType]): The highest allowable setting for max_val
            selected_distribution (Optional[str]): The selected distribution for this parameter
            affected_fields (Optional[list[str]]): List of fields that are affected by changing
                this parameter
            min_model_updater (Callable[[NumberType], None]): Function that is applied where the
                minimum value user control is changed.
            max_model_updater (Callable[[NumberType], None]): Function that is applied where the
                maximum value user control is changed.
            distribution_updater (Callable[[str], None]): Function that is applied where the
                distribution user control is changed.
        """
        super().__init__()

        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.min_allowable = min_allowable
        self.max_allowable = max_allowable
        self.selected_distribution = selected_distribution
        self.affected_fields = affected_fields
        self._min_updater = min_model_updater
        self._max_updater = max_model_updater
        self._distribution_updater = distribution_updater
        self._min_id = dash_title + "_min"
        self._max_id = dash_title + "_max"
        self._dist_id = dash_title + "_dist"
        self.title = dash_title
        self.description = dash_description

        self.components.extend(
            [
                dcc.Markdown("___"),
                dbc.Row(html.B(f"{self.title}:")),
                dbc.Row(html.A(self.description)),
                dbc.Row(html.B("Minimum Value:")),
                dbc.Row(
                    dbc.Input(
                        id=self._min_id,
                        type="number",
                        value=self.minimum_value,
                        min=self.min_allowable,
                        max=self.max_allowable,
                    )
                ),
                dbc.Row(dbc.FormText(f"Min allowable = {self.min_allowable}")),
                dbc.Row(html.B("Maximum Value:")),
                dbc.Row(
                    dbc.Input(
                        id=self._max_id,
                        type="number",
                        value=self.maximum_value,
                        min=self.min_allowable,
                        max=self.max_allowable,
                    )
                ),
                dbc.Row(dbc.FormText(f"Max allowable = {self.max_allowable}")),
                dbc.Row(html.B("Distribution:")),
                dbc.Row(
                    dbc.Select(
                        id=self._dist_id,
                        value=self.selected_distribution,
                        options=[
                            {"label": "Range", "value": "range"},
                            # More options to be added eventually, such as:
                            # {"label": "Normal", "value": "normal"},
                            # {"label": "Random", "value": "random"},
                            # {"label": "Uniform", "value": "uniform"},
                        ],
                    )
                ),
                dbc.Alert(
                    "This parameter needs reviewing",
                    id=self.title + "-alert",
                    color="warning",
                    is_open=False,
                    dismissable=True,
                ),
            ]
        )

        def min_updater(app: Dash) -> None:
            print(f"Registering updater for {self.title} min value")

            def update(new_val: NumberType) -> NumberType:
                print(f"Updating {self.title} (min) in schema to {new_val}")
                self._min_updater(new_val)

                return new_val

            app.callback(
                Output(self._min_id, "value"),
                Input(self._min_id, "value"),
                prevent_initial_call=True,
            )(update)

        self.callbacks.append(min_updater)

        def max_updater(app: Dash) -> None:
            print(f"Registering updater for {self.title} max value")

            def update(new_val: NumberType) -> NumberType:
                print(f"Updating {self.title} (max) in schema to {new_val}")
                self._max_updater(new_val)

                return new_val

            app.callback(
                Output(self._max_id, "value"),
                Input(self._max_id, "value"),
                prevent_initial_call=True,
            )(update)

        self.callbacks.append(max_updater)

        def dist_updater(app: Dash) -> None:
            print(f"Registering updater for {self.title} distribution value")

            def update(new_val: str) -> str:
                print(f"Updating {self.title} (distribution) in schema to {new_val}")
                self._distribution_updater(new_val)

                return new_val

            app.callback(
                Output(self._dist_id, "value"),
                Input(self._dist_id, "value"),
                prevent_initial_call=True,
            )(update)

        self.callbacks.append(dist_updater)

        if self.affected_fields is not None and len(self.affected_fields) > 0:
            triggers: list[Input] = [
                Input(component_id=self._min_id, component_property="value"),
                Input(component_id=self._max_id, component_property="value"),
                Input(component_id=self._dist_id, component_property="value"),
            ]
            self._build_affected_fields(self.affected_fields, triggers)


class FixedNumericInputComponent(DashComponent):
    """RetGen fixed parameter.

    This UI component is used for numeric data, where the user can only select a single value
    within a range.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        value: NumberType,
        min_val: Optional[NumberType],
        max_val: Optional[NumberType],
        affected_fields: Optional[list[str]],
        model_updater: Callable[[NumberType], None],
    ):
        """Create a new Ret Gen fixed parameter.

        Args:
            dash_title (str): Title for the component
            dash_description (str): Description for the component
            value (NumberType): Initial value for the component
            min_val (Optional[NumberType]): Minimum allowable value of the component
            max_val (Optional[NumberType]): Maximum allowable value for the component
            affected_fields (Optional[list[str]]): List of fields that should have warning updates
                if this component changes
            model_updater (Callable[[NumberType], None]): Function that is applied where the user
                control is changed.
        """
        super().__init__()

        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.affected_fields = affected_fields
        self.title = dash_title
        self.description = dash_description
        self.id = dash_title
        self._model_updater = model_updater

        self.components.extend(
            [
                dcc.Markdown("___"),
                dbc.Row(html.B(f"{self.title}:")),
                dbc.Row(
                    dcc.Slider(
                        id=self.id,
                        value=self.value,
                        min=self.min_val,
                        max=self.max_val,
                        tooltip={"placement": "bottom", "always_visible": True},
                    )
                ),
                dbc.Row(dbc.FormText(self.description)),
                dbc.Alert(
                    "This parameter needs reviewing",
                    id=self.id + "-alert",
                    color="warning",
                    is_open=False,
                    dismissable=True,
                ),
            ]
        )

        def updater(app: Dash) -> None:
            print(f"Registering updater for {self.title}")

            def update(new_val: NumberType) -> NumberType:
                print(f"Updating {self.title} in schema to {new_val}")
                self._model_updater(new_val)

                return new_val

            app.callback(
                Output(self.id, "value"), Input(self.id, "value"), prevent_initial_call=True
            )(update)

        self.callbacks.append(updater)

        if self.affected_fields is not None and len(self.affected_fields) > 0:
            triggers: list[Input] = [Input(component_id=self.id, component_property="value")]
            self._build_affected_fields(self.affected_fields, triggers)


class FixedNumericRadioInputComponent(DashComponent):
    """RetGen fixed parameter.

    This UI component is used for numeric data, where the user can only select a single value from
    a set list.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        value: NumberType,
        allowable_values: list[NumberType],
        affected_fields: Optional[list[str]],
        model_updater: Callable[[NumberType], None],
    ):
        """Create a new Ret Gen fixed parameter.

        Args:
            dash_title (str): Title for the component
            dash_description (str): Description for the component
            value (NumberType): Initial value for the component
            allowable_values (list[NumberType]): A list of values that the user can select from.
            affected_fields (Optional[list[str]]): List of fields that should have warning updates
                if this component changes
            model_updater (Callable[[NumberType], None]): Function that is applied where the user
                control is changed.
        """
        super().__init__()

        self.value = value
        self.affected_fields = affected_fields
        self.title = dash_title
        self.description = dash_description
        self.id = dash_title
        self._model_updater = model_updater

        radio_options = []
        for value in allowable_values:
            radio_options.append({"label": str(value), "value": value})

        self.components.extend(
            [
                dcc.Markdown("___"),
                dbc.Row(html.B(f"{self.title}:")),
                dbc.Row(
                    dbc.RadioItems(
                        id=self.id,
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=radio_options,
                        value=value,
                    )
                ),
                dbc.Row(dbc.FormText(self.description)),
                dbc.Row(
                    dbc.Alert(
                        "This parameter needs reviewing",
                        id=self.id + "-alert",
                        color="warning",
                        is_open=False,
                        dismissable=True,
                    )
                ),
            ]
        )

        def updater(app: Dash) -> None:
            print(f"Registering updater for {self.title}")

            def update(new_val: NumberType) -> NumberType:
                print(f"Updating {self.title} in schema to {new_val}")
                self._model_updater(new_val)

                return new_val

            app.callback(
                Output(self.id, "value"), Input(self.id, "value"), prevent_initial_call=True
            )(update)

        self.callbacks.append(updater)

        if self.affected_fields is not None and len(self.affected_fields) > 0:
            triggers: list[Input] = [Input(component_id=self.id, component_property="value")]
            self._build_affected_fields(self.affected_fields, triggers)


class FixedCategoricInputComponent(DashComponent):
    """RetGen fixed categoric parameter.

    This UI component is used for categoric data where the user can only select a single value.
    """

    def __init__(
        self,
        dash_title: str,
        dash_description: str,
        value: Optional[str],
        allowable: list[str],
        affected_fields: Optional[list[str]],
        model_updater: Callable[[str], None],
    ):
        """Create a new Ret Gen fixed parameter.

        Args:
            dash_title (str): Title for the component
            dash_description (str): Description for the component
            value (Optional[str]): Starting value for the component
            allowable (list[str]): List of allowable options for the component
            affected_fields (Optional[list[str]]): List of fields to receive warnings if the
                component is updated
            model_updater (Callable[[list[str]], None]): Function that is applied where the user
                control is changed.
        """
        super().__init__()
        self.value = value
        self.allowable = allowable
        self.affected_fields = affected_fields
        self.id = dash_title
        self.title = dash_title
        self.description = dash_description
        self._model_updater = model_updater

        self.components.extend(
            [
                dcc.Markdown("___"),
                dbc.Row(html.B(f"{self.title}:")),
                dbc.RadioItems(
                    id=self.id,
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    value=self.value,
                    options=[{"label": o, "value": o} for o in self.allowable],
                ),
                dbc.Row(dbc.FormText(self.description)),
                dbc.Alert(
                    "This parameter needs reviewing",
                    id=self.id + "-alert",
                    color="warning",
                    is_open=False,
                    dismissable=True,
                ),
            ]
        )

        def updater(app: Dash) -> None:
            print(f"Registering updater for {self.title}")

            def update(new_val: str) -> str:
                print(f"Updating {self.title} in schema to {new_val}")
                self._model_updater(new_val)

                return new_val

            app.callback(
                Output(self.id, "value"), Input(self.id, "value"), prevent_initial_call=True
            )(update)

        self.callbacks.append(updater)

        if self.affected_fields is not None and len(self.affected_fields) > 0:
            triggers: list[Input] = [Input(component_id=self.id, component_property="value")]
            self._build_affected_fields(self.affected_fields, triggers)


class RetGenHeader(DashComponent):
    """Ret Gen header block."""

    def __init__(self, header: str, subtext: list[str]):
        """Create a new Ret Gen header block.

        Args:
            header (str): The header to display
            subtext (list[str]): List of Markdown elements to follow the header
        """
        super().__init__()
        self.header = header
        self.subtext = subtext

        self.components.append(
            dcc.Markdown(f"# {self.header}"),
        )

        if len(self.subtext) > 0:
            self.components.append(dcc.Markdown("___"))

        self.components.extend([dcc.Markdown(sp) for sp in self.subtext])


class RetGenTitle(dbc.NavbarSimple):
    """A class for the title bar for RetGen."""

    def __init__(self):
        """Create a RetGen Title bar."""
        fnc_logo = html.Img(src="assets/images/fnclogowhite.png", style={"height": "75px"})
        spacer = html.Div(style={"width": "10px"})
        retgen_logo = html.Img(src="assets/images/retgenlogowhite.png", style={"height": "75px"})
        brand = "RetGen"
        brand_style = {"fontSize": 40}
        color = FNC_RED

        super().__init__(
            children=[retgen_logo, spacer, fnc_logo],
            brand=brand,
            brand_style=brand_style,
            color=color,
            dark=True,
            id="Header",
        )


class RetGenModelCard(DashComponent):
    """A card which displays information on the selected model."""

    def __init__(self, title: str, subtext: list[str]):
        """Create model card.

        Args:
            title (str): The title of the model.
            subtext (list[str]): Text to be displayed below the title. Each list entry will begin
                on a new line.
        """
        super().__init__()
        card_header = dbc.CardHeader("Model Information")
        card_title = [html.H5(title)]
        body_text = [html.Div(a) for a in subtext]
        card_body = dbc.CardBody([*card_title, *body_text])
        save = dbc.Button(
            "Save",
            id="save",
            style={"background-color": FNC_LIGHT_PURPLE},
            className="d-grid gap-2 col-2 mx-auto",
        )
        self.components.append(
            dbc.Card(
                [card_header, card_body, save],
                color=FNC_PURPLE,
                inverse=True,
                className="w-100 mb-3",
            )
        )


class RetGenCollapseSection(DashComponent):
    """A collapsable section of the RetGen interface."""

    def __init__(self, title: str, title_color: str, components: list[Any], component_id: str):
        """Create collapse component.

        Args:
            title (str): The title of the collapsable section. Appears on the button to show/hide
                the section.
            title_color (str): The colour of the title button. Can be any valid CSS color (e.g. a
                hex code, a decimal code or a CSS color name).
            components (list[Any]): List of dash components in the hideable section.
            component_id (str): ID field to be used for Dash callbacks.
        """
        super().__init__()
        self.id = component_id
        self.content_id = self.id + "_content"
        self.components.append(
            dbc.Button(
                title,
                style={"background-color": title_color},
                id=component_id,
                className="w-100 mb-3",
            )
        )
        self.components.append(
            dbc.Collapse(
                html.Div(components),
                id=self.content_id,
                is_open=False,
            )
        )

        def register_toggle(app: Dash):
            def toggle_hidden(n, is_open):
                if n:
                    return not is_open
                return is_open

            app.callback(
                Output(self.content_id, "is_open"),
                [Input(self.id, "n_clicks")],
                [State(self.content_id, "is_open")],
            )(toggle_hidden)

        self.callbacks.append(register_toggle)


class SideBySideComponents(DashComponent):
    """Places multiple components side-by-side."""

    def __init__(
        self,
        components: list[Any],
        title: Optional[str] = None,
    ):
        """Place components side-by-side.

        Args:
            components (list[Any]): List of dash components to be placed alongside
                one another.
            title (Otional[str]): Title to be shown above the components. Defaults to None.
        """
        super().__init__()
        columns = []
        for component in components:
            columns.append(dbc.Col(component))

        self.components.extend(
            [dcc.Markdown("___"), html.B(title), dbc.Row(columns, justify="between")]
        )
