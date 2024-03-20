"""Test for logging."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from warnings import catch_warnings

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.agenttype import AgentType
from ret.agents.sensorfusionagent import SensorFusionAgent
from ret.behaviours.behaviourpool import NoEqualityAdder
from ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.disablecommunication import DisableAllHostileCommsInRangeBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.move import AircraftMoveBehaviour, GroundBasedMoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.communication.sensorfusionreceiver import FusionWorldviewHandler
from ret.logger.logger import LogConfig, RetLogger
from ret.model import RetModel
from ret.orders.order import Order, TaskLogStatus, TriggerLogStatus
from ret.orders.tasks.communicate import CommunicateOrderTask, CommunicateWorldviewTask
from ret.orders.tasks.deploycountermeasure import DeployCountermeasureTask
from ret.orders.tasks.disablecommunication import DisableCommunicationTask
from ret.orders.tasks.move import MoveInBandTask, MoveTask
from ret.orders.tasks.sense import SenseTask
from ret.orders.tasks.wait import WaitTask
from ret.orders.triggers.immediate import ImmediateSensorFusionTrigger, ImmediateTrigger
from ret.orders.triggers.mission import MissionMessageTrigger
from ret.orders.triggers.position import PositionTrigger
from ret.orders.triggers.time import TimeTrigger
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import Confidence, PerceivedAgent
from ret.space.heightband import AbsoluteHeightBand
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from ret.testing.mocks import (
    MockAgentWithName,
    MockCommunicateMissionMessageBehaviour,
    MockCommunicateOrdersBehaviour,
    MockCommunicateWorldviewBehaviour,
    MockDeployCountermeasureBehaviour,
    MockDisableCommunicationBehaviour,
    MockFireBehaviour,
    MockHideBehaviour,
    MockMoveBehaviour,
    MockOrder,
    MockSenseBehaviour,
    MockSensor,
    MockTask,
    MockTrigger,
    MockWaitBehaviour,
)
from ret.types import Coordinate2d, Coordinate3d
from ret.weapons.weapon import BasicWeapon
from numpy import random
from parameterized.parameterized import parameterized

if TYPE_CHECKING:
    from typing import Optional, Union

    from ret.behaviours import Behaviour
    from ret.behaviours.loggablebehaviour import LoggableBehaviour
    from ret.orders.order import Task, Trigger
    from ret.template import Template
    from ret.types import Coordinate2dOr3d


class MockLoggableModel(RetModel):
    """Loggable model for testing."""

    def __init__(
        self,
        tables: Optional[dict[str, list[str]]] = None,
        log_config: Union[LogConfig, str] = "none",
    ):
        """Create a new MockLoggableModel.

        Args:
            tables (Optional[dict[str, list[str]]]): Log tables. Defaults to None
            log_config (Union[LogConfig, str]): Logger configuration. Defaults to "none"

        """
        space = ContinuousSpaceWithTerrainAndCulture2d(10000, 10000)
        super().__init__(
            start_time=datetime(2020, 1, 1, 0, 0),
            time_step=timedelta(hours=1),
            end_time=datetime(2020, 1, 2, 0, 0),
            space=space,
            tables=tables,
            log_config=log_config,
            record_position=2,
            behaviours=[
                MockCommunicateOrdersBehaviour(),
                MockCommunicateWorldviewBehaviour(),
                MockCommunicateMissionMessageBehaviour(),
                MockDeployCountermeasureBehaviour(),
                MockMoveBehaviour(),
                MockSenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                ),
                MockFireBehaviour(),
                MockWaitBehaviour(),
                MockDisableCommunicationBehaviour(),
                MockHideBehaviour(),
                MockWaitBehaviour(),
            ],
        )


class TestLogConfigSetup(unittest.TestCase):
    """Logging setup tests."""

    def test_data_collector_created_correctly_table(self):
        """Test the datacollector is created correctly."""
        model = MockLoggableModel(tables={"table": ["c1", "c2"]}, log_config="none")

        datacollector = model.datacollector

        assert datacollector is not None
        assert len(datacollector.tables) == 1

        assert "table" in datacollector.tables
        assert len(datacollector.tables["table"]) == 2
        assert "c1" in datacollector.tables["table"]
        assert "c2" in datacollector.tables["table"]

    def test_create_all_logger(self):
        """Test creating logger with the 'all' config."""
        model = MockLoggableModel()
        log_config = RetLogger.build("AlL", model)

        assert log_config.log_agents
        assert log_config.log_behaviours
        assert log_config.log_deaths
        assert log_config.log_observations
        assert log_config.log_perception
        assert log_config.log_shots_fired
        assert log_config.log_tasks
        assert log_config.log_triggers
        assert log_config.log_position_and_culture

    def test_create_none_logger(self):
        """Test creating a logger with the 'none' config."""
        model = MockLoggableModel()
        log_config = RetLogger.build("noNe", model)

        assert not log_config.log_agents
        assert not log_config.log_behaviours
        assert not log_config.log_deaths
        assert not log_config.log_observations
        assert not log_config.log_perception
        assert not log_config.log_shots_fired
        assert not log_config.log_tasks
        assert not log_config.log_triggers
        assert not log_config.log_position_and_culture

    def test_create_unrecognised_logger(self):
        """Test creating a logger with unrecognised config."""
        model = MockLoggableModel()
        with self.assertWarns(Warning) as w:
            log_config = RetLogger.build("noN", model)

        assert "Unknown log config 'noN' - Assuming 'none'" in str(w.warnings[0])

        assert not log_config.log_agents
        assert not log_config.log_behaviours
        assert not log_config.log_deaths
        assert not log_config.log_observations
        assert not log_config.log_perception
        assert not log_config.log_shots_fired
        assert not log_config.log_tasks
        assert not log_config.log_triggers
        assert not log_config.log_position_and_culture

    def test_create_log_config_from_dictionary(self):
        """Test creating a logger from a dictionary."""
        model = MockLoggableModel()

        log_agents = random.choice([True, False])
        log_behaviours = random.choice([True, False])
        log_deaths = random.choice([True, False])
        log_observations = random.choice([True, False])
        log_perception = random.choice([True, False])
        log_shots_fired = random.choice([True, False])
        log_tasks = random.choice([True, False])
        log_triggers = random.choice([True, False])
        log_behaviour_selection = random.choice([True, False])
        log_position_and_culture = random.choice([True, False])

        log_config = RetLogger.build(
            LogConfig(
                log_agents=log_agents,
                log_behaviours=log_behaviours,
                log_deaths=log_deaths,
                log_observations=log_observations,
                log_perception=log_perception,
                log_shots_fired=log_shots_fired,
                log_tasks=log_tasks,
                log_triggers=log_triggers,
                log_behaviour_selection=log_behaviour_selection,
                log_position_and_culture=log_position_and_culture,
            ),
            model,
        )

        assert log_agents == log_config.log_agents
        assert log_behaviours == log_config.log_behaviours
        assert log_deaths == log_config.log_deaths
        assert log_observations == log_config.log_observations
        assert log_perception == log_config.log_perception
        assert log_shots_fired == log_config.log_shots_fired
        assert log_tasks == log_config.log_tasks
        assert log_triggers == log_config.log_triggers
        assert log_behaviour_selection == log_config.log_behaviour_selection
        assert log_position_and_culture == log_config.log_position_and_culture


class TestLogging(unittest.TestCase):
    """Test cases for logging."""

    class MockAgent(RetAgent):
        """Mock Agent with name."""

        def __init__(self, model: RetModel, pos: Coordinate2dOr3d) -> None:
            """Create MockAgentWithName.

            Args:
                model (RetModel): Model the agent acts in
                pos (Coordinate2dOr3d): Position of the agent
            """
            behaviours: list[Behaviour] = [
                MockMoveBehaviour(),
                MockWaitBehaviour(),
                MockSenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                ),
                MockCommunicateOrdersBehaviour(),
                MockCommunicateWorldviewBehaviour(),
                MockCommunicateMissionMessageBehaviour(),
                MockDeployCountermeasureBehaviour(),
                MockDisableCommunicationBehaviour(),
            ]
            super().__init__(
                model=model,
                pos=pos,
                name="Mock Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                behaviours=behaviours,
            )

    class MockSensorAgent(RetAgent):
        """Mock Sensor Agent with name."""

        def __init__(self, model: RetModel, pos: Coordinate2dOr3d) -> None:
            """Create MockSensorAgentWithName.

            Args:
                model (RetModel): Model the agent acts in
                pos (Coordinate2dOr3d): Position of the agent
            """
            behaviours: list[Behaviour] = [
                MockMoveBehaviour(),
                MockWaitBehaviour(),
                MockSenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                ),
                MockCommunicateOrdersBehaviour(),
                MockCommunicateWorldviewBehaviour(),
                MockCommunicateMissionMessageBehaviour(),
                MockDeployCountermeasureBehaviour(),
                MockDisableCommunicationBehaviour(),
            ]
            super().__init__(
                model=model,
                pos=pos,
                name="Mock Sensor Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                behaviours=behaviours,
                sensors=[MockSensor()],
            )

    class MockSensorFusionAgent(SensorFusionAgent):
        """Mock Sensor Fusion Agent with name."""

        def __init__(self, model: RetModel, pos: Coordinate2dOr3d) -> None:
            """Create MockSensorFusionAgentWithName.

            Args:
                model (RetModel): Model the agent acts in
                pos (Coordinate2dOr3d): Position of the agent
            """
            behaviours: list[Behaviour] = [
                MockMoveBehaviour(),
                MockWaitBehaviour(),
                MockCommunicateWorldviewBehaviour(),
            ]
            super().__init__(
                model=model,
                pos=pos,
                name="Mock Sensor Fusion Agent",
                affiliation=Affiliation.FRIENDLY,
                behaviours=behaviours,
            )

    class MockPerceivedAgent(PerceivedAgent):
        """Mock Perceived Agent."""

        def __init__(self, model: RetModel, pos: Coordinate2dOr3d) -> None:
            """Create MockPerceivedAgent.

            Args:
                model (RetModel): Model the agent acts in
                pos (Coordinate2dOr3d): Position of the perception
            """
            super().__init__(
                sense_time=model.get_time() + model.time_step,
                confidence=Confidence.RECOGNISE,
                location=pos,
                unique_id=6,
                casualty_state=AgentCasualtyState.ALIVE,
            )

    def setUp(self):
        """Test case setup."""
        self.model = MockLoggableModel(log_config="all")
        self.agent = self.MockAgent(self.model, (10, 10))
        self.agent2 = self.MockAgent(self.model, (11, 11))
        self.agent3 = self.MockAgent(self.model, (12, 12))

        self.perceived_agent = self.MockPerceivedAgent(self.model, Coordinate3d([0, 10, 20]))

    def test_datacollector_created_correctly_all_logging(self):
        """Test the datacollector is created correctly."""
        datacollector = self.model.datacollector

        assert datacollector is not None
        assert len(datacollector.tables) == 11

    def test_datacollector_agents_headers(self):
        """Test the behaviour selection table is created correctly."""
        datacollector = self.model.datacollector
        assert "agents" in datacollector.tables
        agents = datacollector.tables["agents"]
        assert len(agents) == 6
        assert "agent name" in agents
        assert "agent ID" in agents
        assert "agent initial location" in agents
        assert "agent type" in agents
        assert "agent affiliation" in agents
        assert "message" in agents

    def test_datacollector_trigger_table_headers(self):
        """Test the triggers table is created correctly."""
        datacollector = self.model.datacollector
        assert "trigger_log" in datacollector.tables
        trigger_log = datacollector.tables["trigger_log"]
        assert len(trigger_log) == 7
        assert "time" in trigger_log
        assert "agent name" in trigger_log
        assert "agent ID" in trigger_log
        assert "trigger" in trigger_log
        assert "status" in trigger_log
        assert "message" in trigger_log
        assert "model step" in trigger_log

    def test_datacollector_task_table_headers(self):
        """Test the task table is created correctly."""
        datacollector = self.model.datacollector
        assert "task_log" in datacollector.tables
        task_log = datacollector.tables["task_log"]
        assert len(task_log) == 7
        assert "time" in task_log
        assert "agent name" in task_log
        assert "agent ID" in task_log
        assert "task" in task_log
        assert "status" in task_log
        assert "message" in task_log
        assert "model step" in task_log

    def test_datacollector_behaviour_log_headers(self):
        """Test the behaviour table is created correctly."""
        datacollector = self.model.datacollector
        assert "behaviour_log" in datacollector.tables
        behaviour_log = datacollector.tables["behaviour_log"]
        assert len(behaviour_log) == 6
        assert "time" in behaviour_log
        assert "agent name" in behaviour_log
        assert "agent ID" in behaviour_log
        assert "behaviour" in behaviour_log
        assert "message" in behaviour_log
        assert "model step" in behaviour_log

    def test_datacollector_deaths_headers(self):
        """Test the deaths table is created correctly."""
        datacollector = self.model.datacollector
        assert "deaths" in datacollector.tables
        deaths_log = datacollector.tables["deaths"]
        assert len(deaths_log) == 13
        assert "time" in deaths_log
        assert "target name" in deaths_log
        assert "target ID" in deaths_log
        assert "target location" in deaths_log
        assert "target type" in deaths_log
        assert "killer name" in deaths_log
        assert "killer ID" in deaths_log
        assert "killer location" in deaths_log
        assert "killer type" in deaths_log
        assert "message" in deaths_log
        assert "shot ID" in deaths_log
        assert "model step" in deaths_log

    def test_datacollector_shots_fired_headers(self):
        """Test the shots fired table is created correctly."""
        datacollector = self.model.datacollector
        assert "shots_fired" in datacollector.tables
        shots_fired = datacollector.tables["shots_fired"]
        assert len(shots_fired) == 11
        assert "time" in shots_fired
        assert "agent name" in shots_fired
        assert "agent ID" in shots_fired
        assert "targets" in shots_fired
        assert "aim location" in shots_fired
        assert "message" in shots_fired
        assert "shot ID" in shots_fired
        assert "model step" in shots_fired
        assert "remaining ammo" in shots_fired

    def test_datacollector_observations_headers(self):
        """Test the observations table is created correctly."""
        datacollector = self.model.datacollector
        assert "observation_record" in datacollector.tables
        observation_record = datacollector.tables["observation_record"]
        assert len(observation_record) == 13
        assert "model time" in observation_record
        assert "sense time" in observation_record
        assert "senser name" in observation_record
        assert "senser ID" in observation_record
        assert "senser location" in observation_record
        assert "observation result - affiliation" in observation_record
        assert "observation result - agent_type" in observation_record
        assert "observation result - casualty_state" in observation_record
        assert "observation result - confidence" in observation_record
        assert "observation result - location" in observation_record
        assert "observation result - unique_id" in observation_record
        assert "message" in observation_record
        assert "model step" in observation_record

    def test_datacollector_perceptions_headers(self):
        """Test the perceptions table is created correctly."""
        datacollector = self.model.datacollector
        assert "perception_record" in datacollector.tables
        perception_record = datacollector.tables["perception_record"]
        assert len(perception_record) == 7
        assert "time" in perception_record
        assert "receiver name" in perception_record
        assert "receiver ID" in perception_record
        assert "receiver location" in perception_record
        assert "receiver perceived agents" in perception_record
        assert "message" in perception_record
        assert "model step" in perception_record

    def test_datacollector_behaviour_selection_headers(self):
        """Test the behaviour selection table is created correctly."""
        datacollector = self.model.datacollector
        assert "behaviour_selection" in datacollector.tables
        behaviour_selection = datacollector.tables["behaviour_selection"]
        assert len(behaviour_selection) == 8
        assert "time" in behaviour_selection
        assert "agent ID" in behaviour_selection
        assert "handler" in behaviour_selection
        assert "behaviour type" in behaviour_selection
        assert "behaviour options" in behaviour_selection
        assert "behaviour choice" in behaviour_selection
        assert "message" in behaviour_selection
        assert "model step" in behaviour_selection

    def test_datacollector_position_and_culture_headers(self):
        """Test the position and culture table is created correctly."""
        datacollector = self.model.datacollector
        assert "position_and_culture" in datacollector.tables
        position_and_culture = datacollector.tables["position_and_culture"]
        assert len(position_and_culture) == 11
        assert "time" in position_and_culture
        assert "agent ID" in position_and_culture
        assert "agent name" in position_and_culture
        assert "location" in position_and_culture
        assert "agent killed" in position_and_culture
        assert "current culture" in position_and_culture
        assert "message" in position_and_culture
        assert "model step" in position_and_culture

    def test_datacollector_model_seed_headers(self):
        """Test the position and culture table is created correctly."""
        datacollector = self.model.datacollector
        assert "model_seed" in datacollector.tables
        model_seed = datacollector.tables["model_seed"]
        assert len(model_seed) == 2
        assert "seed" in model_seed
        assert "message" in model_seed

    def test_log_agent_on_model(self):
        """Test the log_agent method on the model behaves correctly."""
        self.model.logger.log_agent(self.agent, "Test message")

        datacollector = self.model.datacollector
        assert datacollector is not None

        df = datacollector.tables["agents"]

        assert df["agent name"][-1] == self.agent.name
        assert df["agent ID"][-1] == str(self.agent.unique_id)
        assert df["agent initial location"][-1] == str(self.agent.pos)
        assert df["agent type"][-1] == self.agent.agent_type.name
        assert df["agent affiliation"][-1] == self.agent.affiliation.name
        assert df["message"][-1] == "Test message"

    def test_log_trigger_on_model(self):
        """Test the log_trigger method on the model behaves correctly."""
        self.model.logger.log_trigger(
            self.agent, "Test Trigger", TriggerLogStatus.ACTIVATED, "Test message"
        )

        datacollector = self.model.datacollector
        assert datacollector is not None
        assert datacollector.tables["trigger_log"]["time"][0] == self.model.get_time()
        assert datacollector.tables["trigger_log"]["agent name"][0] == self.agent.name
        assert datacollector.tables["trigger_log"]["agent ID"][0] == self.agent.unique_id
        assert datacollector.tables["trigger_log"]["trigger"][0] == "Test Trigger"
        assert datacollector.tables["trigger_log"]["status"][0] == TriggerLogStatus.ACTIVATED.value
        assert datacollector.tables["trigger_log"]["message"][0] == "Test message"
        assert datacollector.tables["trigger_log"]["model step"][0] == self.model.schedule.steps

    def test_log_task_on_model(self):
        """Test the log_task method on the model behaves correctly."""
        self.model.logger.log_task(self.agent, "Test Task", TaskLogStatus.STARTED, "Test message")

        datacollector = self.model.datacollector
        assert datacollector is not None
        assert datacollector.tables["task_log"]["time"][0] == self.model.get_time()
        assert datacollector.tables["task_log"]["agent name"][0] == self.agent.name
        assert datacollector.tables["task_log"]["agent ID"][0] == self.agent.unique_id
        assert datacollector.tables["task_log"]["task"][0] == "Test Task"
        assert datacollector.tables["task_log"]["status"][0] == TaskLogStatus.STARTED.value
        assert datacollector.tables["task_log"]["message"][0] == "Test message"
        assert datacollector.tables["task_log"]["model step"][0] == self.model.schedule.steps

    def test_log_behaviour_on_model(self):
        """Test the log_behaviour method on the model behaves correctly."""
        self.model.logger.log_behaviour(self.agent, "Test Behaviour", "Test message")

        datacollector = self.model.datacollector
        assert datacollector is not None
        assert datacollector.tables["behaviour_log"]["time"][0] == self.model.get_time()
        assert datacollector.tables["behaviour_log"]["agent name"][0] == self.agent.name
        assert datacollector.tables["behaviour_log"]["agent ID"][0] == self.agent.unique_id
        assert datacollector.tables["behaviour_log"]["behaviour"][0] == "Test Behaviour"
        assert datacollector.tables["behaviour_log"]["message"][0] == "Test message"
        assert datacollector.tables["behaviour_log"]["model step"][0] == self.model.schedule.steps

    def test_log_position_on_model(self):
        """Test the log_position_and_culture method on the model behaves correctly."""
        self.model.logger.log_position_and_culture_record(self.agent, "Test message")

        datacollector = self.model.datacollector
        assert datacollector is not None
        assert datacollector.tables["position_and_culture"]["time"][0] == (
            self.model.get_time() - self.model.time_step
        )
        assert datacollector.tables["position_and_culture"]["agent name"][0] == self.agent.name
        assert datacollector.tables["position_and_culture"]["agent ID"][0] == self.agent.unique_id
        assert datacollector.tables["position_and_culture"]["location"][0] == self.agent.pos
        assert datacollector.tables["position_and_culture"]["agent killed"][0] == self.agent.killed
        assert (
            datacollector.tables["position_and_culture"]["current culture"][0]
            == self.model.space.get_culture(pos=self.agent.pos).name
        )
        assert datacollector.tables["position_and_culture"]["message"][0] == "Test message"
        assert (
            datacollector.tables["position_and_culture"]["model step"][0]
            == self.model.schedule.steps - 1
        )

    def test_log_seed_on_model(self):
        """Test the log_position_and_culture method on the model behaves correctly."""
        self.model.logger.log_model_seed("Test message")

        datacollector = self.model.datacollector
        assert datacollector is not None
        assert datacollector.tables["model_seed"]["seed"][0] == self.model._seed
        assert datacollector.tables["model_seed"]["message"][0] == "Test message"

    def test_log_deaths_on_model(self):
        """Test the log_death method on the model behaves correctly."""
        shot_id = random.randint(1, 100)
        self.model.logger.log_death(
            target=self.agent,
            killer=self.agent2,
            shot_id=shot_id,
            weapon_name="Test weapon name",
            message="Test message",
        )

        datacollector = self.model.datacollector
        assert datacollector is not None

        df = datacollector.tables["deaths"]

        assert df["time"][0] == self.model.get_time()
        assert df["target name"][0] == self.agent.name
        assert df["target ID"][0] == str(self.agent.unique_id)
        assert df["target type"][0] == self.agent.agent_type.name
        assert df["target location"][0] == str(self.agent.pos)
        assert df["killer name"][0] == self.agent2.name
        assert df["killer ID"][0] == str(self.agent2.unique_id)
        assert df["killer location"][0] == str(self.agent2.pos)
        assert df["killer type"][0] == self.agent2.agent_type.name
        assert df["shot ID"][0] == shot_id
        assert df["message"][0] == "Test message"
        assert df["model step"][0] == self.model.schedule.steps

    def test_log_observation_record_on_model(self):
        """Test the log_observation_record on the model behaves correctly."""
        self.model.logger.log_observation_record(self.agent, [self.perceived_agent], "Test message")

        datacollector = self.model.datacollector
        assert datacollector is not None
        df = datacollector.tables["observation_record"]
        assert df["model time"][0] == self.model.get_time()
        assert df["sense time"][0] == self.perceived_agent.sense_time
        assert df["senser name"][0] == self.agent.name
        assert df["senser ID"][0] == self.agent.unique_id
        assert df["senser location"][0] == self.agent.pos
        assert df["observation result - affiliation"][0] == self.perceived_agent.affiliation.name
        assert df["observation result - agent_type"][0] == self.perceived_agent.agent_type.name
        assert (
            df["observation result - casualty_state"][0] == self.perceived_agent.casualty_state.name
        )
        assert df["observation result - confidence"][0] == self.perceived_agent.confidence.name
        assert df["observation result - location"][0] == str(self.perceived_agent.location)
        assert (
            df["observation result - unique_id"][0]
            == self.perceived_agent.get_unique_id_log_representation()
        )
        assert df["message"][0] == "Test message"
        assert df["model step"][0] == self.model.schedule.steps

    def test_log_perception_record_on_model(self):
        """Test the log_behaviour method on the model behaves correctly."""
        self.sensor_fusion_agent = self.MockSensorFusionAgent(self.model, (50, 50))
        self.model.logger.log_perception_record(self.sensor_fusion_agent, "Test message")

        datacollector = self.model.datacollector
        assert datacollector is not None

        perception_record = datacollector.tables["perception_record"]

        assert perception_record["time"][0] == self.model.get_time()
        assert perception_record["receiver name"][0] == self.sensor_fusion_agent.name
        assert perception_record["receiver ID"][0] == self.sensor_fusion_agent.unique_id
        assert perception_record["receiver location"][0] == self.sensor_fusion_agent.pos

        # The only perception the Sensor Fusion Agent has is about itself
        assert len(perception_record["receiver perceived agents"][0]) == 1
        assert perception_record["receiver perceived agents"][0][0] == (
            str(self.sensor_fusion_agent.unique_id),
            str(self.sensor_fusion_agent.pos),
            Confidence.KNOWN.name,
        )

        assert perception_record["message"][0] == "Test message"

    def test_log_agent_on_agent_creation(self):
        """Test agent logging using the agent constructor behaves correctly."""
        df = self.model.datacollector.tables["agents"]

        # There were 3 MockAgents created in the setup for this test.
        assert len(df["agent name"]) == 3
        self.agent4 = self.MockAgent(self.model, (0, 0))
        assert len(df["agent name"]) == 4

        assert df["agent name"][-1] == self.agent4.name
        assert df["agent ID"][-1] == str(self.agent4.unique_id)
        assert df["agent initial location"][-1] == str(self.agent4.pos)
        assert df["agent type"][-1] == self.agent4.agent_type.name
        assert df["agent affiliation"][-1] == self.agent4.affiliation.name
        assert df["message"][-1] == ""

    def test_log_trigger_on_trigger_private_methods(self):
        """Test trigger logging using the private methods behaves correctly."""
        trigger = MockTrigger(log=True)

        assert trigger._get_log_message() == ""

        trigger._log_trigger(self.agent, TriggerLogStatus.DEACTIVATED)

        log = self.model.datacollector.tables["trigger_log"]

        assert log["time"][0] == self.model.get_time()
        assert log["agent name"][0] == self.agent.name
        assert log["agent ID"][0] == self.agent.unique_id
        assert log["trigger"][0] == "Mock Trigger"
        assert log["status"][0] == TriggerLogStatus.DEACTIVATED.value
        assert log["message"][0] == ""
        assert log["model step"][0] == self.model.schedule.steps

    @parameterized.expand(
        [
            [MockTrigger()],
            [ImmediateTrigger()],
            [ImmediateSensorFusionTrigger()],
            [MissionMessageTrigger("test")],
            [PositionTrigger(MockAgent(MockLoggableModel(), (10, 10)), (10, 10), 10)],
            [TimeTrigger(datetime(2021, 3, 4))],
        ]
    )
    def test_log_trigger_on_trigger(self, trigger: Trigger):
        """Test trigger logging from trigger works correctly.

        Args:
            trigger (Trigger): The trigger to test
        """

        def return_true(checker: RetAgent):
            return True

        def return_false(checker: RetAgent):
            return False

        log = self.model.datacollector.tables["trigger_log"]

        # type ignored as MyPy doesn't allow monkey patching
        trigger._check_condition = return_false  # type: ignore
        trigger.is_active(self.agent)
        assert len(log["time"]) == 0

        trigger._check_condition = return_true  # type: ignore
        trigger.is_active(self.agent)
        assert len(log["time"]) == 1
        assert log["status"][0] == TriggerLogStatus.ACTIVATED.value

        trigger.is_active(self.agent)
        assert len(log["time"]) == 1

        trigger._check_condition = return_false  # type: ignore
        trigger.is_active(self.agent)
        assert len(log["time"]) == 2
        assert log["status"][0] == TriggerLogStatus.ACTIVATED.value
        assert log["status"][1] == TriggerLogStatus.DEACTIVATED.value

    @parameterized.expand(
        [
            [ImmediateTrigger(), ""],
            [ImmediateSensorFusionTrigger(), ""],
            [MissionMessageTrigger("test"), "Message: test"],
            [
                PositionTrigger(MockAgent(MockLoggableModel(), (10, 10)), (10, 10), 10),
                ("Agent: Mock Agent (1), Position: (10, 10), Tolerance: 10"),
            ],
            [TimeTrigger(datetime(2021, 3, 4)), "Time: 2021-03-04 00:00:00"],
        ]
    )
    def test_trigger_log_message(self, trigger: Trigger, expected: str):
        """Test the trigger log messages are formatted correctly.

        Args:
            trigger (Trigger): The trigger to check log message
            expected (str): Expected log message
        """
        assert trigger._get_log_message() == expected

    def test_log_task_on_task_private_methods(self):
        """Test task logging using the private methods behaves correctly."""
        task = MockTask(log=True)

        assert task._get_log_message() == ""

        task._log_task(self.agent, TaskLogStatus.FINISHED)

        log = self.model.datacollector.tables["task_log"]

        assert log["time"][0] == self.model.get_time()
        assert log["agent name"][0] == self.agent.name
        assert log["agent ID"][0] == self.agent.unique_id
        assert log["task"][0] == "Mock Task"
        assert log["status"][0] == TaskLogStatus.FINISHED.value
        assert log["message"][0] == ""

    @parameterized.expand(
        [
            [MockTask()],
            [CommunicateOrderTask([])],
            [CommunicateWorldviewTask()],
            [DeployCountermeasureTask()],
            [DisableCommunicationTask()],
            [SenseTask(duration=timedelta(days=1))],
            [WaitTask(duration=timedelta(days=1))],
        ]
    )
    def test_log_task_on_task(self, task: Task):
        """Test task logging from task works correctly.

        Args:
            task (Task): the task to test
        """

        def return_true(doer: RetAgent):
            return True

        def return_false(doer: RetAgent):
            return False

        log = self.model.datacollector.tables["task_log"]

        task._is_task_complete = return_false  # type: ignore
        task.do_task_step(self.agent)
        task.is_task_complete(self.agent)
        assert len(log["time"]) == 1
        assert log["status"][0] == TaskLogStatus.STARTED.value

        task.do_task_step(self.agent)
        task.is_task_complete(self.agent)
        assert len(log["time"]) == 1

        task._is_task_complete = return_true  # type: ignore
        task.do_task_step(self.agent)
        task.is_task_complete(self.agent)
        assert len(log["time"]) == 2
        assert log["status"][0] == TaskLogStatus.STARTED.value
        assert log["status"][1] == TaskLogStatus.FINISHED.value

        task.do_task_step(self.agent)
        task.is_task_complete(self.agent)
        assert len(log["time"]) == 2

    @parameterized.expand(
        [
            [CommunicateOrderTask([]), "Orders: ()"],
            [CommunicateWorldviewTask(), ""],
            [DeployCountermeasureTask(), ""],
            [DisableCommunicationTask(), ""],
            [MoveTask(destination=(10, 10), tolerance=10), "Destination: (10, 10)"],
            [
                MoveInBandTask(
                    destination=(10, 10, "test"),
                    bands=[AbsoluteHeightBand("test", 10)],
                    space=ContinuousSpaceWithTerrainAndCulture3d(10000, 10000),
                    tolerance=10,
                ),
                "Destination: (10, 10, 'test')",
            ],
            [SenseTask(timedelta(days=1)), ""],
            [WaitTask(timedelta(days=1)), "Duration: 1 day, 0:00:00"],
        ]
    )
    def test_task_log_message(self, task: Task, expected: str):
        """Test the task log messages are formatted correctly.

        Args:
            task (Task): The task to check log message
            expected (str): Expected log message
        """
        assert task._get_log_message() == expected

    @parameterized.expand(
        [
            [CommunicateOrdersBehaviour()],
            [CommunicateWorldviewBehaviour()],
            [CommunicateMissionMessageBehaviour()],
            [DeployCountermeasureBehaviour()],
            [DisableAllHostileCommsInRangeBehaviour(range=10)],
            [FireBehaviour()],
            [
                GroundBasedMoveBehaviour(
                    base_speed=10,
                    gradient_speed_modifiers=[],
                    culture_speed_modifiers={},
                )
            ],
            [AircraftMoveBehaviour(base_speed=10, height_bands=[])],
            [
                SenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                )
            ],
            [WaitBehaviour()],
        ]
    )
    def test_log_behaviour_on_behaviour(self, behaviour: LoggableBehaviour):
        """Test behaviour logging behaves correctly.

        Args:
            behaviour: The loggable behaviour to test
        """
        log = self.model.datacollector.tables["behaviour_log"]

        behaviour.log(self.agent)
        assert len(log["time"]) == 1
        assert log["time"][0] == self.model.get_time()
        assert log["agent name"][0] == self.agent.name
        assert log["agent ID"][0] == self.agent.unique_id

    @parameterized.expand(
        [
            [CommunicateOrdersBehaviour(), "NO RECIPIENT GIVEN; orders:[NONE FOUND]"],
            [CommunicateWorldviewBehaviour(), ""],
            [CommunicateMissionMessageBehaviour(), ""],
            [DeployCountermeasureBehaviour(), ""],
            [DisableAllHostileCommsInRangeBehaviour(range=10), "Range: 10"],
            [
                FireBehaviour(),
                "No weapon",
            ],
            [
                GroundBasedMoveBehaviour(
                    base_speed=10,
                    gradient_speed_modifiers=[],
                    culture_speed_modifiers={},
                ),
                "Start: NOT PROVIDED; End: NOT PROVIDED; Destination: NOT PROVIDED; "
                + "Base speed: 10",
            ],
            [
                AircraftMoveBehaviour(base_speed=10, height_bands=[]),
                "Start: NOT PROVIDED; End: NOT PROVIDED; Destination: NOT PROVIDED; "
                + "Base speed: 10",
            ],
            [
                SenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                ),
                "",
            ],
            [WaitBehaviour(), ""],
        ]
    )
    def test_behaviour_log_message(self, behaviour: LoggableBehaviour, expected: str):
        """Test the behaviour log messages are formatted correctly.

        Args:
            behaviour (LoggableBehaviour): The loggable behaviour
            expected (str): Expected message from the behaviour

        """
        assert behaviour._get_log_message() == expected

    def test_ground_based_move_behaviour_log_message(self):
        """Test the ground based move behaviour log messages are formatted correctly."""
        assert (
            GroundBasedMoveBehaviour(
                base_speed=10,
                gradient_speed_modifiers=[],
                culture_speed_modifiers={},
            )._get_log_message(
                move_start_location=(0, 0, 0),
                move_end_location=(1, 1, 1),
                move_destination=(2, 2, 2),
            )
            == "Start: (0, 0, 0); End: (1, 1, 1); Destination: (2, 2, 2); Base speed: 10"
        )

    def test_aircraft_move_behaviour_log_message(self):
        """Test the aircraft move behaviour log messages are formatted correctly."""
        assert (
            AircraftMoveBehaviour(
                base_speed=10,
                height_bands=[],
            )._get_log_message(
                move_start_location=(0, 0, 0),
                move_end_location=(1, 1, 1),
                move_destination=(2, 2, 2),
            )
            == "Start: (0, 0, 0); End: (1, 1, 1); Destination: (2, 2, 2); Base speed: 10"
        )

    def test_aircraft_move_behaviour_with_height_bands_log_message(self):
        """Test the aircraft move behaviour log messages are formatted correctly."""
        assert (
            AircraftMoveBehaviour(
                base_speed=10,
                height_bands=[],
            )._get_log_message(
                move_start_location=(
                    0,
                    0,
                    "500m",
                ),
                move_end_location=(1, 1, 1),
                move_destination=(2, 2, 2),
            )
            == "Start: (0, 0, '500m'); End: (1, 1, 1); Destination: (2, 2, 2); Base speed: 10"
        )

    def test_communicate_single_order_behaviour_log_message(self):
        """Test the behaviour log messages are formatted correctly."""
        assert (
            CommunicateOrdersBehaviour()._get_log_message(
                recipient=MockAgentWithName(0, name="MockAgent"),
                communicated_orders=MockOrder(),
            )
            == "recipient: MockAgent (ID: 0); orders:[Mock Task: Mock Trigger]"
        )

    def test_communicate_multiple_orders_behaviour_log_message(self):
        """Test the behaviour log messages are formatted correctly."""
        assert (
            CommunicateOrdersBehaviour()._get_log_message(
                recipient=MockAgentWithName(0, name="MockAgent"),
                communicated_orders=[MockOrder(), MockOrder()],
            )
            == "recipient: MockAgent (ID: 0); orders:[Mock Task: Mock Trigger;"
            + " Mock Task: Mock Trigger]"
        )

    def test_task_interrupt(self):
        """Test that the task interrupt logging works."""
        t_1 = self.model.start_time
        t_2 = t_1 + self.model.time_step
        order_1 = Order(TimeTrigger(t_1), MockTask(), priority=0)
        order_2 = Order(
            TimeTrigger(t_2),
            MockTask(),
            priority=1,
        )
        orders: list[Template[Order]] = [order_1, order_2]

        self.agent.add_orders(orders)

        log = self.model.datacollector.tables["task_log"]
        assert len(log["time"]) == 0

        self.model.step()
        assert len(log["time"]) == 1
        assert log["status"][0] == "Started"

        self.model.step()
        assert len(log["time"]) == 3
        assert log["status"][1] == "Interrupted"
        assert log["status"][2] == "Started"

    def test_log_death_on_kill(self):
        """Test killer logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["deaths"]

        weapon = BasicWeapon(
            name="Weapon",
            radius=0.25,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=60),
            kill_probability_per_round=1,
        )

        weapon.fire_at_target(self.agent, self.agent2.pos)  # type: ignore

        assert len(log["time"]) == 1
        assert log["time"][0] == self.model.get_time()
        assert log["killer name"][0] == self.agent.name
        assert log["killer ID"][0] == str(self.agent.unique_id)
        assert log["killer location"][0] == str(self.agent.pos)
        assert log["killer type"][0] == self.agent.agent_type.name
        assert log["target ID"][0] == str(self.agent2.unique_id)
        assert log["target name"][0] == self.agent2.name
        assert log["target location"][0] == str(self.agent2.pos)
        assert log["target type"][0] == self.agent2.agent_type.name
        assert log["shot ID"][0] == self.agent.model.get_next_shot_id() - 1

    def test_log_no_kill_on_miss(self):
        """Test killer logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["deaths"]

        weapon = BasicWeapon(
            name="Weapon",
            radius=10,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=60),
            kill_probability_per_round=0,
        )

        weapon.fire_at_target(self.agent, (0, 0))
        assert len(log["time"]) == 0

    def test_log_shot_fired_for_fire_at_no_target(self):
        """Test shot-fired logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["shots_fired"]
        weapon = BasicWeapon(
            name="Weapon",
            radius=10,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=60),
            kill_probability_per_round=0.5,
            ammo_capacity=60,
        )
        firing = FireBehaviour()
        firing.fire_at_target(self.agent, weapon, Coordinate2d([-50, -50]))
        assert [] == log["targets"][0]
        assert self.agent.unique_id == log["agent ID"][0]
        assert self.model.get_time() == log["time"][0]
        assert weapon.ammo_capacity == log["remaining ammo"][0]

    def test_log_shot_fired_for_fire_at_single_target(self):
        """Test shot-fired logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["shots_fired"]
        weapon = BasicWeapon(
            name="Weapon",
            radius=0.1,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=60),
            kill_probability_per_round=0.5,
            ammo_capacity=60,
        )
        firing = FireBehaviour()
        firing.fire_at_target(self.agent, weapon, Coordinate2d([11, 11]))
        assert [2] == log["targets"][0]
        assert self.agent.unique_id == log["agent ID"][0]
        assert self.model.get_time() == log["time"][0]
        assert weapon.ammo_capacity == log["remaining ammo"][0]

    def test_log_shot_fired_for_fire_at_multiples_targets(self):
        """Test shot-fired logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["shots_fired"]
        weapon = BasicWeapon(
            name="Weapon",
            radius=10,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=60),
            kill_probability_per_round=0.5,
            ammo_capacity=60,
        )
        firing = FireBehaviour()
        firing.fire_at_target(self.agent, weapon, Coordinate2d([10, 10]))
        assert [2, 3] == log["targets"][0]
        assert self.agent.unique_id == log["agent ID"][0]
        assert self.model.get_time() == log["time"][0]
        assert weapon.ammo_capacity == log["remaining ammo"][0]

    def test_log_shot_fired_for_multiple_fires(self):
        """Test shot-fired logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["shots_fired"]
        weapon = BasicWeapon(
            name="Weapon",
            radius=10,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=60),
            kill_probability_per_round=0.5,
            ammo_capacity=60,
        )
        firing = FireBehaviour()
        firing.fire_at_target(self.agent, weapon, Coordinate2d([-50, -50]))
        assert [] == log["targets"][0]
        assert self.agent.unique_id == log["agent ID"][0]
        assert self.model.get_time() == log["time"][0]
        assert 59 == log["remaining ammo"][0]
        firing.fire_at_target(self.agent, weapon, Coordinate2d([-50, -50]))
        assert 58 == log["remaining ammo"][1]
        firing.fire_at_target(self.agent, weapon, Coordinate2d([-50, -50]))
        assert 57 == log["remaining ammo"][2]
        firing.fire_at_target(self.agent, weapon, Coordinate2d([-50, -50]))

    def test_log_observation_record_on_sense_get_results_none(self):
        """Test observation logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["observation_record"]

        self.mock_sensor_agent = self.MockSensorAgent(self.model, Coordinate2d([0, 0]))

        for s in self.mock_sensor_agent._sensors:
            perceived_agents = s.get_results(self.mock_sensor_agent)
            assert len(perceived_agents) == 0

        assert len(log["model time"]) == 0

    def test_log_observation_record_on_sense_get_results_found(self):
        """Test observation logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["observation_record"]

        self.mock_sensor_agent = self.MockSensorAgent(self.model, Coordinate2d([0, 0]))

        for s in self.mock_sensor_agent._sensors:
            s._cached_agents.append(self.perceived_agent)
            perceived_agents = s.get_results(self.mock_sensor_agent)
            assert len(perceived_agents) == 0

        self.model.schedule.time += 1

        for s in self.mock_sensor_agent._sensors:
            perceived_agents = s.get_results(self.mock_sensor_agent)
            assert len(perceived_agents) == 1

        assert len(log["model time"]) == 1
        assert log["model time"][0] == self.model.get_time()
        assert log["senser name"][0] == self.mock_sensor_agent.name
        assert log["message"][0] == ""
        assert log["observation result - affiliation"][0] == self.perceived_agent.affiliation.name
        assert log["observation result - unique_id"][0] == "6"

    def test_log_perception_record_on_sensor_fusion_receive(self):
        """Test perception logging behaves correctly."""
        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["perception_record"]

        self.mock_fusion_worldviewhandler = FusionWorldviewHandler()
        self.sensor_fusion_agent = self.MockSensorFusionAgent(self.model, (50, 50))

        assert len(self.sensor_fusion_agent.perceived_world.get_perceived_agents()) == 1

        self.mock_fusion_worldviewhandler.receive(self.sensor_fusion_agent, [self.perceived_agent])

        assert len(log["time"]) == 1
        assert log["time"][0] == self.model.get_time()
        assert log["receiver name"][0] == self.sensor_fusion_agent.name
        assert log["receiver perceived agents"][0][0] == (
            str(self.sensor_fusion_agent.unique_id),
            str(self.sensor_fusion_agent.pos),
            Confidence.KNOWN.name,
        )

        assert log["receiver perceived agents"][0][1] == (
            str(6),  # The perceived agent doesn't expose ID property directly
            str(self.perceived_agent.location),
            self.perceived_agent.confidence.name,
        )

    def test_logging_behaviour_selection(self):
        """Test logging for a behaviour selection."""
        agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent With Behaviour",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=1.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        wait_behaviour = WaitBehaviour()

        agent.behaviour_pool.add_behaviour(wait_behaviour)
        agent.wait_step()

        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["behaviour_selection"]
        assert log["time"][0] == self.model.get_time()
        assert log["agent ID"][0] == agent.unique_id
        assert log["handler"][0] == "step"
        assert log["behaviour type"][0] == "WaitBehaviour"
        assert log["behaviour options"][0] == ["WaitBehaviour"]
        assert log["behaviour choice"][0] == "WaitBehaviour"
        assert log["message"][0] == ""

    def test_logging_behaviour_selection_where_no_valid_behaviour_exists(self):
        """Test logging for invalid behaviour selection."""
        agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent With Behaviour",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=1.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        with catch_warnings(record=True) as w:
            agent.wait_step()
        assert len(w) == 1

        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["behaviour_selection"]
        assert log["time"][0] == self.model.get_time()
        assert log["agent ID"][0] == agent.unique_id
        assert log["handler"][0] == "step"
        assert log["behaviour type"][0] == "WaitBehaviour"
        assert log["behaviour options"][0] is None
        assert log["behaviour choice"][0] is None
        assert log["message"][0] == "No available behaviours to select from"

    def test_logging_multiple_behaviour_selection(self):
        """Test logging for a behaviour selection."""
        agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent With Behaviour",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=0.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
            behaviour_adder=NoEqualityAdder,
        )

        wait_behaviours = [WaitBehaviour(), WaitBehaviour(), WaitBehaviour()]

        agent.behaviour_pool.add_behaviour(wait_behaviours[0])
        agent.behaviour_pool.add_behaviour(wait_behaviours[1])
        agent.behaviour_pool.add_behaviour(wait_behaviours[2])

        agent.wait_step()

        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["behaviour_selection"]
        assert log["time"][0] == self.model.get_time()
        assert log["agent ID"][0] == agent.unique_id
        assert log["handler"][0] == "step"
        assert log["behaviour type"][0] == "WaitBehaviour"
        assert log["behaviour options"][0] == [
            "WaitBehaviour",
            "WaitBehaviour",
            "WaitBehaviour",
        ]
        assert log["behaviour choice"][0] == "WaitBehaviour"
        assert log["message"][0] == ""

    def test_logging_custom_behaviour_name(self):
        """Test logging for a behaviour selection."""

        class NamedWaitBehaviour(WaitBehaviour):
            @property
            def name(self):
                return "Custom Name"

        agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent With Behaviour",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=0.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        wait_behaviour = NamedWaitBehaviour()
        agent.behaviour_pool.add_behaviour(wait_behaviour)

        agent.wait_step()

        assert self.model.datacollector is not None
        log = self.model.datacollector.tables["behaviour_selection"]
        assert log["behaviour choice"][0] == "Custom Name"

    def test_get_log_representation(self):
        """Test get_log_representation behaves correctly."""
        id = self.model.logger.get_log_representation(self.agent, "unique_id")
        err = self.model.logger.get_log_representation(self.agent, "not_an_attribute")
        assert id == "1"
        assert err == "not_an_attribute unavailable"

    def test_not_log_agent_on_agent_creation(self):
        """Test agent logging using the agent constructor behaves correctly."""
        model = MockLoggableModel(log_config=LogConfig(log_agents=False))
        model.step()
        assert "agents" not in model.datacollector.tables.keys()
        MockAgent(model, (0, 0))

    def test_log_agent_not_in_datacollector(self):
        """Test log agent function if not in datacollector."""
        model = MockLoggableModel(log_config="none")
        row = model.logger.log_agent(self.agent)
        assert row is None
        assert "agents" not in model.datacollector.tables.keys()

    def test_log_position_not_in_datacollector(self):
        """Test log position function if not in datacollector."""
        model = MockLoggableModel(log_config="none")
        row = model.logger.log_position_and_culture_record(self.agent)
        assert row is None
        assert "position_and_culture" not in model.datacollector.tables.keys()


class MockAgent(RetAgent):
    """Mock Agent with name."""

    def __init__(self, model: RetModel, pos: Coordinate2dOr3d) -> None:
        """Create MockAgentWithName.

        Args:
            model (RetModel): Model the agent acts in
            pos (Coordinate2dOr3d): Position of the agent
        """
        super().__init__(
            model=model,
            pos=pos,
            name="Mock Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
