"""Configuration for logging methods."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mesa.datacollection import DataCollector
from mesa_ret.sensing.perceivedworld import PerceivedAgent

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Sequence, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.agents.sensorfusionagent import SensorFusionAgent
    from mesa_ret.behaviours import Behaviour
    from mesa_ret.model import RetModel
    from mesa_ret.orders.order import TaskLogStatus, TriggerLogStatus
    from mesa_ret.types import Coordinate2dOr3d


@dataclass
class LogConfig:
    """RET Logger configuration."""

    log_agents: bool = False
    log_triggers: bool = False
    log_tasks: bool = False
    log_behaviours: bool = False
    log_deaths: bool = False
    log_shots_fired: bool = False
    log_observations: bool = False
    log_perception: bool = False
    log_behaviour_selection: bool = False


class RetLogger:
    """RET Logger."""

    agents_table_definition: dict[str, list[str]] = {
        "agents": [
            "agent name",
            "agent ID",
            "agent initial location",
            "agent type",
            "agent affiliation",
            "message",
        ]
    }
    trigger_log_table_definition: dict[str, list[str]] = {
        "trigger_log": [
            "time",
            "agent name",
            "agent ID",
            "trigger",
            "status",
            "message",
        ]
    }
    task_log_table_definition: dict[str, list[str]] = {
        "task_log": ["time", "agent name", "agent ID", "task", "status", "message"]
    }
    behaviour_log_table_definition: dict[str, list[str]] = {
        "behaviour_log": ["time", "agent name", "agent ID", "behaviour", "message"]
    }
    deaths_table_definition: dict[str, list[str]] = {
        "deaths": [
            "time",
            "target name",
            "target ID",
            "target location",
            "target type",
            "killer name",
            "killer ID",
            "killer location",
            "killer type",
            "shot ID",
            "message",
        ]
    }
    shots_fired_table_definition: dict[str, list[str]] = {
        "shots_fired": [
            "time",
            "agent name",
            "agent ID",
            "targets",
            "aim location",
            "shot ID",
            "message",
        ]
    }
    observation_record_definition: dict[str, list[str]] = {
        "observation_record": [
            "model time",
            "sense time",
            "senser name",
            "senser ID",
            "senser location",
            "observation result - affiliation",
            "observation result - agent_type",
            "observation result - casualty_state",
            "observation result - confidence",
            "observation result - location",
            "observation result - unique_id",
            "message",
        ]
    }
    perception_log_definition: dict[str, list[str]] = {
        "perception_record": [
            "time",
            "receiver name",
            "receiver ID",
            "receiver location",
            "receiver perceived agents",
            "message",
        ]
    }
    behaviour_selection_log_definition: dict[str, list[str]] = {
        "behaviour_selection": [
            "time",
            "agent ID",
            "handler",
            "behaviour type",
            "behaviour options",
            "behaviour choice",
            "message",
        ]
    }

    @staticmethod
    def build(config: Union[LogConfig, str], model: RetModel) -> RetLogger:
        """Create an instance of a LogConfig.

        Args:
            config (Union[LogConfig, str]): Log config specification
            model (RetModel): The model to attach logging to

        Returns:
            RetLogger: Log configuration
        """
        if isinstance(config, LogConfig):
            return RetLogger(
                model,
                log_agents=config.log_agents,
                log_triggers=config.log_triggers,
                log_tasks=config.log_tasks,
                log_behaviours=config.log_behaviours,
                log_deaths=config.log_deaths,
                log_shots_fired=config.log_shots_fired,
                log_observations=config.log_observations,
                log_perception=config.log_perception,
                log_behaviour_selection=config.log_behaviour_selection,
            )
        elif str(config).lower() == "all":
            return RetLogger(
                model,
                log_agents=True,
                log_triggers=True,
                log_tasks=True,
                log_behaviours=True,
                log_deaths=True,
                log_shots_fired=True,
                log_observations=True,
                log_perception=True,
                log_behaviour_selection=True,
            )
        elif str(config).lower() == "none":
            return RetLogger(model)
        else:
            warnings.warn(f"Unknown log config '{config}' - Assuming 'none'", Warning)
            return RetLogger(model)

    def __init__(
        self,
        model: RetModel,
        log_agents: bool = False,
        log_triggers: bool = False,
        log_tasks: bool = False,
        log_behaviours: bool = False,
        log_deaths: bool = False,
        log_shots_fired: bool = False,
        log_observations: bool = False,
        log_perception: bool = False,
        log_behaviour_selection: bool = False,
    ) -> None:
        """Initialise the logging configuration object.

        Args:
            model (RetModel): The model to attach the log config to.
            log_agents (bool): Flag to log agents. Defaults to False.
            log_triggers (bool): Flag to log triggers. Defaults to False.
            log_tasks (bool): Flag to log tasks. Defaults to False.
            log_behaviours (bool): Flag to log behaviours. Defaults to False.
            log_deaths (bool): Flag to log deaths. Defaults to False.
            log_shots_fired (bool): Flag to log shots fired. Defaults to False.
            log_observations (bool): Flag to log observations. Defaults to False.
            log_perception (bool): Flag to log perception log of sensor fusion agents.
                Defaults to False.
            log_behaviour_selection (bool): Flag to log selection of behaviour from
                the behaviour pool. Defaults to False.
        """
        self.model = model
        self.log_agents = log_agents
        self.log_triggers = log_triggers
        self.log_tasks = log_tasks
        self.log_behaviours = log_behaviours
        self.log_deaths = log_deaths
        self.log_shots_fired = log_shots_fired
        self.log_observations = log_observations
        self.log_perception = log_perception
        self.log_behaviour_selection = log_behaviour_selection

    def create_data_collector(
        self,
        model_reporters: Optional[dict[str, Union[str, Callable]]] = None,
        agent_reporters: Optional[dict[str, Union[str, Callable]]] = None,
        tables: Optional[dict[str, list[str]]] = None,
    ) -> DataCollector:
        """Create data-collectors for a model.

        Args:
            model_reporters (Optional[dict[str, Union[str, Callable]]]): All of the
                defined model reporters for the datacollector. Defaults to None.
            agent_reporters (Optional[dict[str, Union[str, Callable]]]): All of the
                defined agent reporters for the datacollector. Defaults to None.
            tables (Optional[dict[str, list[str]]]): All of the user defined tables to
                be included in the datacollector. Defaults to None.

        Returns:
            DataCollector: Mesa data collector
        """
        if tables is None:
            tables = {}
        logs_to_definitions = [
            (self.log_agents, self.agents_table_definition),
            (self.log_triggers, self.trigger_log_table_definition),
            (self.log_tasks, self.task_log_table_definition),
            (self.log_behaviours, self.behaviour_log_table_definition),
            (self.log_deaths, self.deaths_table_definition),
            (self.log_shots_fired, self.shots_fired_table_definition),
            (self.log_observations, self.observation_record_definition),
            (self.log_perception, self.perception_log_definition),
            (self.log_behaviour_selection, self.behaviour_selection_log_definition),
        ]

        tables_to_define = [
            table_def for log_flag, table_def in logs_to_definitions if log_flag is True
        ]
        for table_def in tables_to_define:
            tables = {**tables, **table_def}

        return DataCollector(model_reporters, agent_reporters, tables)

    def log_agent(self, agent: RetAgent, message: str = "") -> None:
        """Log agents.

        Args:
            agent (RetAgent): The agent to be recorded.
            message (str): An optional message to be logged. Defaults to empty string.
        """
        if self.log_agents and self.model.datacollector:
            row = {
                "agent name": self.get_log_representation(agent, "name"),
                "agent ID": self.get_log_representation(agent, "unique_id"),
                "agent initial location": self.get_log_representation(agent, "pos"),
                "agent type": self.get_log_representation(agent, "agent_type"),
                "agent affiliation": self.get_log_representation(agent, "affiliation"),
                "message": message,
            }

            agent.model.datacollector.add_table_row("agents", row)

    def log_trigger(
        self,
        agent: RetAgent,
        trigger_str: str,
        status: TriggerLogStatus,
        message: str = "",
    ) -> None:
        """Log triggers.

        Args:
            agent (RetAgent): The agent checking the trigger
            trigger_str (str): The trigger name
            status (TriggerLogStatus): Status of the trigger
            message (str): Extra information. Defaults to ""
        """
        if self.log_triggers and self.model.datacollector:
            row = {
                "time": self.model.get_time(),
                "agent name": agent.name,
                "agent ID": agent.unique_id,
                "trigger": trigger_str,
                "status": status.value,
                "message": message,
            }

            agent.model.datacollector.add_table_row("trigger_log", row)

    def log_task(
        self, agent: RetAgent, task_str: str, status: TaskLogStatus, message: str = ""
    ) -> None:
        """Log tasks.

        Args:
            agent (RetAgent): The agent performing the task
            task_str (str): Name of the task
            status (TaskLogStatus): Status of the task
            message (str): Extra information. Defaults to ""
        """
        if self.log_tasks and self.model.datacollector:
            row = {
                "time": self.model.get_time(),
                "agent name": agent.name,
                "agent ID": agent.unique_id,
                "task": task_str,
                "status": status.value,
                "message": message,
            }

            self.model.datacollector.add_table_row("task_log", row)

    def log_behaviour(self, agent: RetAgent, behaviour_str: str, message: str = "") -> None:
        """Log behaviours.

        Args:
            agent (RetAgent): The agent performing the behaviour
            behaviour_str (str): Name of the behaviour
            message (str): Extra information. Defaults to ""
        """
        if self.log_behaviours and self.model.datacollector:
            row = {
                "time": self.model.get_time(),
                "agent name": agent.name,
                "agent ID": agent.unique_id,
                "behaviour": behaviour_str,
                "message": message,
            }

            self.model.datacollector.add_table_row("behaviour_log", row)

    def log_death(
        self,
        target: RetAgent,
        killer: Optional[RetAgent],
        shot_id: Optional[int],
        message: str = "",
    ) -> None:
        """Log killer victim records.

        Args:
            target (RetAgent): The killed agent
            shot_id (Optional[int]): The shot ID
            killer (Optional[RetAgent]): The killer
            message (str): Extra information. Defaults to ""
        """
        if self.log_deaths and self.model.datacollector:

            row = {
                "time": self.model.get_time(),
                "target name": self.get_log_representation(target, "name"),
                "target ID": self.get_log_representation(target, "unique_id"),
                "target location": self.get_log_representation(target, "pos"),
                "target type": self.get_log_representation(target, "agent_type"),
                "killer name": self.get_log_representation(killer, "name"),
                "killer ID": self.get_log_representation(killer, "unique_id"),
                "killer location": self.get_log_representation(killer, "pos"),
                "killer type": self.get_log_representation(killer, "agent_type"),
                "shot ID": shot_id,
                "message": message,
            }
            self.model.datacollector.add_table_row("deaths", row)

    def log_shot_fired(
        self,
        agent: RetAgent,
        targets: list[RetAgent],
        shot_id: int,
        location: Optional[Coordinate2dOr3d] = None,
        message: str = "",
    ) -> None:
        """Log shots fired records.

        Args:
            agent (RetAgent): The agent firing the shot
            targets (list[RetAgent]): The targets of the shot
            shot_id (int): ID of the shot fired
            location (Optional[Coordinate2dOr3d]): The location being fired at. Defaults
                to None.
            message (str): Additional information. Defaults to "".
        """
        if self.log_shots_fired and self.model.datacollector:
            row = {
                "time": self.model.get_time(),
                "agent name": agent.name,
                "agent ID": agent.unique_id,
                "targets": [t.unique_id for t in targets],
                "aim location": location,
                "shot ID": shot_id,
                "message": message,
            }
            self.model.datacollector.add_table_row("shots_fired", row)

    def log_observation_record(
        self,
        agent: RetAgent,
        observation_result: list[PerceivedAgent],
        message: str = "",
    ) -> None:
        """Log observation records.

        Args:
            agent (RetAgent): The agent doing the observation
            observation_result (list[PerceivedAgent]): The result of the observation
            message (str): Additional information. Defaults to "".
        """
        attributes_list = [
            "affiliation",
            "agent_type",
            "casualty_state",
            "confidence",
            "location",
            "unique_id",
        ]
        if self.log_observations and self.model.datacollector:
            for observed_agent in observation_result:
                observation_data = {}
                for attribute in attributes_list:
                    observation_data[
                        f"observation result - {attribute}"
                    ] = self.get_log_representation(observed_agent, attribute)

                row = {
                    "model time": self.model.get_time(),
                    "sense time": observed_agent.sense_time,
                    "senser name": agent.name,
                    "senser ID": agent.unique_id,
                    "senser location": agent.pos,
                    "message": message,
                } | observation_data

                self.model.datacollector.add_table_row("observation_record", row)

    def log_perception_record(self, receiver: SensorFusionAgent, message: str = "") -> None:
        """Log a record of the perceptions.

        This method is called every time the perception of a Sensor Fusion Agent is
        updated.

        Args:
            receiver (SensorFusionAgent): The agent with an updated perception record.
            message (str): Additional information. Defaults to "".
        """
        if self.log_perception and self.model.datacollector:
            row = {
                "time": self.model.get_time(),
                "receiver name": receiver.name,
                "receiver ID": receiver.unique_id,
                "receiver location": receiver.pos,
                "receiver perceived agents": [
                    agent.perception_log_representation()
                    for agent in receiver.perceived_world.get_perceived_agents()
                ],
                "message": message,
            }
            self.model.datacollector.add_table_row("perception_record", row)

    def log_behaviour_selection_record(
        self,
        agent: RetAgent,
        handler: str,
        behaviour_type: type,
        candidates: Sequence[Behaviour],
        selected: Behaviour,
        message: str = "",
    ):
        """Log an instance of selecting a behaviour.

        Args:
            agent (RetAgent): The agent for who the behaviour has been selected.
            handler (str): The handler used to select the behaviour.
            behaviour_type(type): The behaviour type selected.
            candidates (Sequence[Behaviour]): List of candidate behaviours for selection
            selected (Behaviour): The selected behaviour
            message (str): Any additional contextual message. Defaults to "".
        """
        if self.log_behaviour_selection and self.model.datacollector:
            row = {
                "time": self.model.get_time(),
                "agent ID": agent.unique_id,
                "handler": handler,
                "behaviour type": behaviour_type.__name__,
                "behaviour options": [c.name for c in candidates],
                "behaviour choice": selected.name,
                "message": message,
            }

            self.model.datacollector.add_table_row("behaviour_selection", row)

    def log_no_available_behaviour(self, agent: RetAgent, handler: str, behaviour_type: type):
        """Log an instance of selecting a behaviour where no valid options exist.

        Args:
            agent (RetAgent): The agent for who the behaviour has been selected.
            handler (str): The handler used to select the behaviour.
            behaviour_type(type): The behaviour type selected.
        """
        if self.log_behaviour_selection and self.model.datacollector:
            row = {
                "time": self.model.get_time(),
                "agent ID": agent.unique_id,
                "handler": handler,
                "behaviour type": behaviour_type.__name__,
                "behaviour options": None,
                "behaviour choice": None,
                "message": "No available behaviours to select from",
            }
            self.model.datacollector.add_table_row("behaviour_selection", row)

    def get_log_representation(self, object: Any, attribute: str) -> str:
        """Attempt to retrieve loggable representation of an object's attribute.

        If the attribute has a name property, this will be returned instead of the
        attribute itself (e.g., in the instance of enums, where the enum name will be
        returned).

        Args:
            object (Any): Object which may contain an attribute
            attribute (str): Attribute to log

        Returns:
            str: The loggable representation of the attribute
        """
        try:
            if attribute == "unique_id" and isinstance(object, PerceivedAgent):
                return object.get_unique_id_log_representation()

            property = getattr(object, attribute)

            if hasattr(property, "name"):
                return str(property.name)
            else:
                return str(property)

        except AttributeError:
            return f"{attribute} unavailable"
