"""ret specific model representation.

The RetModel is the core component for a simulation using ret. In order to
set up a ret simulation, there are a number of steps that need to be completed
using a RetModel, which need to be completed in a specific order. The intent of
following these instructions is that they will result in a variant of the RetModel
which can be initialised with no arguments and therefore easily run in a predictable
way from a Python REPL or Jupyter notebook.

The steps for creating the model are as follows:

1.  Extend the RetModel class, and in the new class define an __init__ method with no
    arguments. Specify all arguments required by the RetModel and then call the
    super().__init__(args...) method within the constructor.
2.  Create all agents that sit within the simulation. Each agent should either be a
    RetAgent, or a class that extends RetAgent.
3.  Create communications networks between all agents that sit within the simulation.

An example of creating a model as described here can be seen in the IADS example
scenario distributed alongside the ret source code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from mesa.model import Model
from mesa.time import BaseScheduler

from mesa_ret.agents.groupagent import GroupAgent
from mesa_ret.logger.logger import RetLogger
from mesa_ret.parameters import (
    ExperimentalControls,
    ModelParameterSpecification,
    ScenarioDependentData,
    get_model_parameters,
)
from mesa_ret.scenario_independent_data import ModelMetadata
from mesa_ret.sensing.sensor import ActivelySensingAgentsManager
from mesa_ret.visualisation import get_playback_writer_register

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from typing import Any, Callable, Optional, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.io import v2
    from mesa_ret.logger.logger import LogConfig
    from mesa_ret.parameters import NumericParameterSpecification
    from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture2d
    from mesa_ret.visualisation.playback_writer import PlaybackWriter


class RetModel(Model):
    """Model containing a schedule and a space and keeps track of simulation time.

    This model provides static methods that form the basis of generating parametrised input data,
    in the form of the following methods:

        - get_parameters, which returns a list of fixed and variable parameters to be supplied
            to the model.
        - get_scenario_independent_metadata, which returns a metadata summary describing the data
            contained within the model.
        - parameter_getter, which translates a v2.RetModelSchema into a list of fixed and variable
            parameters specific to the model.
    """

    @staticmethod
    def get_parameters() -> ModelParameterSpecification:
        """Return a list of the parameters that can be provided to the model."""
        return ModelParameterSpecification(
            experimental_controls=ExperimentalControls(
                numeric_parameters={}, categoric_parameters={}
            ),
            scenario_dependent_data=ScenarioDependentData(
                numeric_parameters={}, categoric_parameters={}
            ),
        )

    @staticmethod
    def get_scenario_independent_metadata() -> ModelMetadata:
        """Return the model metadata for the model."""
        subtext = [
            "This description can be customised by the custodian of the a Ret Model, by extending "
            + "the `get_scenario_independent_metadata()` static method. ",
            "It should be used to include a description, in the form of markdown components, "
            + "of the scenario independent data that is stored in the model.",
        ]

        return ModelMetadata(header="Default Ret Model", subtext=subtext)

    @staticmethod
    def parameter_getter(
        schema: v2.RetModelSchema,
    ) -> tuple[dict[str, Any], Optional[dict[str, list[Any]]]]:
        """Convert model schema into a suitable set of fixed and variable parameters.

        Args:
            schema (v2.RetModelSchema): latest model schema

        Returns:
            tuple[dict[str, Any], Optional[dict[str, list[Any]]]]: Fixed and variable
                model parameters
        """
        fixed_parameters = {
            "start_time": schema.time.start_time,
            "end_time": schema.time.end_time,
            "time_step": schema.time.time_step,
            "space": schema.space.to_model(),
        }

        fixed_parameters["model_reporters"] = None
        fixed_parameters["agent_reporters"] = schema.agent_reporters
        fixed_parameters["tables"] = None
        fixed_parameters["log_config"] = "all"
        if schema.playback_writer != "None":
            fixed_parameters["playback_writer"] = get_playback_writer_register().get_register_item(
                schema.playback_writer
            )
        else:
            fixed_parameters["playback_writer"] = None

        required_parameters = get_model_parameters(schema.model_module_name, schema.model_name)

        for k1, p1 in required_parameters.scenario_dependent_data.numeric_parameters.items():
            if k1 in schema.scenario_dependent_parameters.numeric_parameters:
                fixed_parameters[k1] = schema.scenario_dependent_parameters.numeric_parameters[k1]
            else:
                warn(
                    f"'{k1}' not found in model file."
                    + f" Defaulting to min allowable '{p1.min_allowable}'"
                )
                fixed_parameters[k1] = p1.min_allowable

        for k2, p2 in required_parameters.scenario_dependent_data.categoric_parameters.items():
            if k2 in schema.scenario_dependent_parameters.categoric_parameters:
                fixed_parameters[k2] = schema.scenario_dependent_parameters.categoric_parameters[k2]
            else:
                warn(
                    f"'{k2}' not found in model file. "
                    + f"Defaulting to '{p2.allowable_options[0]}'"
                )
                fixed_parameters[k2] = p2.allowable_options[0]

        parameters_list = {}

        for k3, p3 in required_parameters.experimental_controls.numeric_parameters.items():
            if k3 in schema.experimental_controls.numeric_parameters:
                _, m = schema.experimental_controls.numeric_parameters[k3].to_model()
                parameters_list[k3] = m
            else:
                parameters_list[k3] = RetModel._to_default_distribution(p3)
                warn(
                    f"'{k3}' not found in model file. "
                    + f"Defaulting to only use '{parameters_list[k3]}'"
                )

        for k4, p4 in required_parameters.experimental_controls.categoric_parameters.items():
            if k4 in schema.experimental_controls.categoric_parameters:
                _, m = schema.experimental_controls.categoric_parameters[k4].to_model()
                parameters_list[k4] = m
            else:
                warn(
                    f"'{k4}' not found in model file. "
                    + f"Defaulting to use '{p4.allowable_options}'"
                )
                parameters_list[k4] = p4.allowable_options

        return fixed_parameters, parameters_list

    @staticmethod
    def _to_default_distribution(spec: NumericParameterSpecification) -> list[Any]:
        return [spec.to_default_min(), spec.to_default_max()]

    def __init__(
        self,
        start_time: datetime,
        time_step: timedelta,
        end_time: datetime,
        space: ContinuousSpaceWithTerrainAndCulture2d,
        schedule: Optional[BaseScheduler] = None,
        model_reporters: Optional[dict[str, Union[str, Callable]]] = None,
        agent_reporters: Optional[dict[str, Union[str, Callable]]] = None,
        tables: Optional[dict[str, list[str]]] = None,
        log_config: Union[LogConfig, str] = "all",
        playback_writer: Optional[PlaybackWriter] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Create a RetModel.

        Args:
            start_time (datetime): start time of the simulation
            time_step (timedelta): time step size
            end_time (datetime): end time of the simulation
            space (ContinuousSpaceWithTerrainAndCulture2d): The space for this model
            schedule (Optional[BaseScheduler]): The scheduler for this model, if None
                then a BaseScheduler will be used. Defaults to None.
            model_reporters (Optional[dict[str, Union[str, Callable]]]): Model reporters
                to be passed to the datacollector. Defaults to None.
            agent_reporters (Optional[dict[str, Union[str, Callable]]]): Agent reporters
                to be passed to the datacollector. Defaults to None.
            tables (Optional[dict[str, list[str]]]): Tables to be passed to the
                datacollector. Defaults to None.
            log_config (Optional[Union[dict[str, bool], str]]): Definition of what
                should be logged, either in the form of a dictionary of keyword
                arguments to create a LogConfig from, or descriptive string.
            playback_writer (Optional[PlaybackWriter]): A playback writer to write
                playback files for the model. Defaults to None.
            *args (Any): Any optional arguments you wish to pass into the Mesa Model.
            **kwargs (Any): Any optiona keyword arguments you wish to pass into the Mesa
                Model.
        """
        super().__init__(self, args, kwargs)

        self.start_time = start_time
        self.time_step = time_step
        self.end_time = end_time

        self.space = space

        if schedule is None:
            schedule = BaseScheduler(self)
        self.schedule = schedule

        self.logger = RetLogger.build(log_config, self)

        self.datacollector = self.logger.create_data_collector(
            model_reporters, agent_reporters, tables
        )

        self.playback_writer = playback_writer

        if self.playback_writer is not None:
            self.playback_writer.model_start(self)

        self._shot_id = 0

        self.actively_sensing_agents_manager = ActivelySensingAgentsManager()

    def get_next_shot_id(self) -> int:
        """Return the next shot ID.

        Returns:
            int: Shot ID
        """
        self._shot_id += 1
        return self._shot_id

    def get_time(self) -> datetime:
        """Return the current simulation time.

        Returns:
            datetime: the current simulation time
        """
        dt: datetime = self.start_time + (self.schedule.time * self.time_step)
        return dt

    def step(self) -> None:
        """Clear expired clutter modifiers and advance the model by one step."""
        self.space.clutter_field.remove_expired_modifiers(self.get_time())

        self.schedule.step()
        if self.playback_writer is not None:
            self.playback_writer.model_step(self)

        self.actively_sensing_agents_manager.do_actions_and_clear()

        if self.get_time() > self.end_time:
            self.running = False
            if self.playback_writer is not None:
                self.playback_writer.model_finish()

        if self.datacollector is not None:
            self.datacollector.collect(self)

    def get_all_agents(self) -> list[RetAgent]:
        """Get all agents from the schedule, including group and subordinate agents.

        Returns:
            list[RetAgent]: All agents
        """
        agents: list[RetAgent] = self.schedule.agents  # type: ignore

        for agent in agents:
            if isinstance(agent, GroupAgent):
                for subordinate_agent in agent.agents:
                    agents.append(subordinate_agent)

        return agents
