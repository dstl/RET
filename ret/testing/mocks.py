"""Mock objects to support testing."""

from __future__ import annotations

from abc import ABC
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from retplay.retplaymaps import RetPlayMapFromPNG

from ret.agents.agent import RetAgent
from ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.disablecommunication import DisableCommunicationBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.loggablebehaviour import LoggableBehaviour
from ret.behaviours.move import MoveBehaviour, MoveInBandBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.communication.communicationnetwork import CommunicationNetwork
from ret.logger.logger import RetLogger
from ret.model import RetModel
from ret.agents.agenttype import AgentType
from ret.orders.background_order import BackgroundOrder
from ret.orders.order import GroupAgentOrder, Order, Task, Trigger
from ret.parameters import (
    CategoricParameterSpecification,
    ExperimentalControls,
    ModelParameterSpecification,
    NumericParameterSpecification,
    ScenarioDependentData,
)
from ret.scenario_independent_data import ModelMetadata
from ret.sensing.perceivedworld import Confidence, PerceivedAgentBasedOnAgent
from ret.sensing.sensor import Sensor
from ret.space.clutter.clutter import ClutterModifier
from ret.space.clutter.countermeasure import Countermeasure
from ret.space.feature import BoxFeature
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from ret.visualisation.json_models import (
    RetPlayAgent,
    RetPlayInitialData,
    RetPlaySpace,
    RetPlayStepData,
)
from ret.weapons.fire_schedule import FireSchedule
from ret.weapons.weapon import BasicLongRangedWeapon, BasicShortRangedWeapon, BasicWeapon
from ret.weapons.mutil_kill_probability import ProbabilityByType

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Optional, Sequence, Union

    from ret.agents.affiliation import Affiliation
    from ret.agents.agentfilter import AgentFilter
    from ret.agents.airagent import AirAgent
    from ret.agents.sensorfusionagent import SensorFusionAgent
    from ret.behaviours import Behaviour
    from ret.behaviours.fire import HostileTargetResolver, TargetSelector
    from ret.orders.order import TaskLogStatus, TriggerLogStatus
    from ret.sensing.perceivedworld import PerceivedAgent, PerceivedAgentFilter
    from ret.space.culture import Culture
    from ret.space.heightband import HeightBand
    from ret.template import Template
    from ret.types import Color, Coordinate, Coordinate2d, Coordinate2dOr3d, Coordinate3d
    from ret.weapons.weapon import Weapon


class MockModel(RetModel):
    """Mock representation of a RetModel."""

    def __init__(
        self,
        space: ContinuousSpaceWithTerrainAndCulture2d,
        start_time: Optional[datetime] = None,
        time_step: Optional[timedelta] = None,
        end_time: Optional[datetime] = None,
        record_position: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new MockModel.

        Args:
            space: (ContinuousSpaceWithTerrainAndCulture2d): Space that the model
                occupies.
            start_time (Optional[datetime]): Start time. If None or unset, defaults to
                datetime(2020, 1, 1, 0, 0).
            time_step (Optional[timedelta]): Time step. If None or unset, defaults to
                timedelta(hours=1).
            end_time (Optional[datetime]): End time. If None or unset, defaults to
                datetime(2020, 1, 2, 0, 0).
            record_position (int): Definition of how often, in model timesteps,
                to record agent positions to Datacollector to prevent
                data from becoming too large. If None then every timestep will be recorded.
                Defaults to None.
        """
        if end_time is None:
            end_time = datetime(2020, 1, 2, 0, 0)

        if start_time is None:
            start_time = datetime(2020, 1, 1, 0, 0)

        if time_step is None:
            time_step = timedelta(hours=1)

        if record_position is None:
            record_position = 1

        super().__init__(
            start_time=start_time,
            time_step=time_step,
            end_time=end_time,
            space=space,
            record_position=record_position,
            log_config="None",
            **kwargs,
        )


class MockModel3d(MockModel):
    """Mock representation of a RetModel in 3D space."""

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        time_step: Optional[timedelta] = None,
        end_time: Optional[datetime] = None,
        width=10000,
        height=10000,
        **kwargs: Any,
    ):
        """Create a new MockModel3d.

        Args:
            start_time (Optional[datetime]): Start time. If None or unset, defaults to
                datetime(2020, 1, 1, 0, 0).
            time_step (Optional[timedelta]): Time step. If None or unset, defaults to
                timedelta(hours=1).
            end_time (Optional[datetime]): End time. If None or unset, defaults to
                datetime(2020, 1, 2, 0, 0).
            width (int): Width of model space. Defaults to 10000.
            height (int): Height of model space. Defaults to 10000. This is the length of the model
                (y in x-y co-ordinates), not the 3D height upwards (z co-ordinate in x-y-z
                co-ordinates).
        """
        if end_time is None:
            end_time = datetime(2020, 1, 2, 0, 0)

        if start_time is None:
            start_time = datetime(2020, 1, 1, 0, 0)

        if time_step is None:
            time_step = timedelta(hours=1)

        space = ContinuousSpaceWithTerrainAndCulture3d(width, height)
        super().__init__(space, start_time, time_step, end_time, **kwargs)


class MockModel3dWithCulture(MockModel):
    """Mock representation of a RetModel in 3D space."""

    def __init__(
        self,
        culture_dictionary: dict[Color, Culture],
        width=10000,
        height=10000,
    ):
        """Create a new MockModel3d.

        Args:
            culture_dictionary (dict[Color, Culture]): Dictionary of cultures in the
                space.
            width (int): Width of model space. Defaults to 10000.
            height (int): Height of model space. Defaults to 10000. This is the length of the model
                (y in x-y co-ordinates), not the 3D height upwards (z co-ordinate in x-y-z
                co-ordinates).
        """
        space = ContinuousSpaceWithTerrainAndCulture3d(
            width, height, culture_dictionary=culture_dictionary
        )
        super().__init__(space)


class MockModel3dWithBoxFeature(MockModel):
    """Mock representation of a RetModel in 3D space with Box feature."""

    def __init__(
        self,
        box_min_coord: Coordinate3d,
        box_max_coord: Coordinate3d,
        box_name: str,
        start_time: Optional[datetime] = None,
        time_step: Optional[timedelta] = None,
        end_time: Optional[datetime] = None,
        width=10000,
        height=10000,
    ):
        """Create a new MockModel3dWithBoxFeature.

        Args:
            box_min_coord (Coordinate3d): min coord of box feature
            box_max_coord (Coordinate3d): max coord of box feature
            box_name (str): name of box feature
            start_time (Optional[datetime]): Start time. If None or unset, defaults to
                datetime(2020, 1, 1, 0, 0).
            time_step (Optional[timedelta]): Time step. If None or unset, defaults to
                timedelta(hours=1).
            end_time (Optional[datetime]): End time. If None or unset, defaults to
                datetime(2020, 1, 2, 0, 0).
            width (int): Width of model space. Defaults to 10000.
            height (int): Height of model space. Defaults to 10000. This is the length of the model
                (y in x-y co-ordinates), not the 3D height upwards (z co-ordinate in x-y-z
                co-ordinates).
        """
        if end_time is None:
            end_time = datetime(2020, 1, 2, 0, 0)

        if start_time is None:
            start_time = datetime(2020, 1, 1, 0, 0)

        if time_step is None:
            time_step = timedelta(hours=1)

        area = BoxFeature(min_coord=box_min_coord, max_coord=box_max_coord, name=box_name)
        space = ContinuousSpaceWithTerrainAndCulture3d(width, height, features=[area])
        super().__init__(space, start_time, time_step, end_time)


class MockModel2d(MockModel):
    """Mock representation of a RetModel in 2D space."""

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        time_step: Optional[timedelta] = None,
        end_time: Optional[datetime] = None,
        record_position: Optional[int] = None,
        width=10000,
        height=10000,
        **kwargs: Any,
    ):
        """Create a new MockModel2d.

        Args:
            start_time (Optional[datetime]): Start time. If None or unset, defaults to
                datetime(2020, 1, 1, 0, 0).
            time_step (Optional[timedelta]): Time step. If None or unset, defaults to
                timedelta(hours=1).
            end_time (Optional[datetime]): End time. If None or unset, defaults to
                datetime(2020, 1, 2, 0, 0).
            record_position (int): Definition of how often, in model timesteps,
                to record agent positions to Datacollector to prevent
                data from becoming too large. If None then every timestep will be recorded.
                Defaults to None.
            width (int): Width of model space. Defaults to 10000.
            height (int): Height of model space. Defaults to 10000.
        """
        if end_time is None:
            end_time = datetime(2020, 1, 2, 0, 0)

        if start_time is None:
            start_time = datetime(2020, 1, 1, 0, 0)

        if time_step is None:
            time_step = timedelta(hours=1)

        if record_position is None:
            record_position = 1

        space = ContinuousSpaceWithTerrainAndCulture2d(width, height)
        super().__init__(
            space=space,
            start_time=start_time,
            time_step=time_step,
            end_time=end_time,
            record_position=record_position,
            **kwargs,
        )


class MockModel2dWithBoxFeature(MockModel):
    """Mock representation of a RetModel in 2D space with Box feature."""

    def __init__(
        self,
        box_min_coord: Coordinate2d,
        box_max_coord: Coordinate2d,
        box_name: str,
        start_time: Optional[datetime] = None,
        time_step: Optional[timedelta] = None,
        end_time: Optional[datetime] = None,
        width=10000,
        height=10000,
    ):
        """Create a new MockModel2dWithBoxFeature.

        Args:
            box_min_coord (Coordinate2d): min coord of box feature
            box_max_coord (Coordinate2d): max coord of box feature
            box_name (str): name of box feature
            start_time (Optional[datetime]): Start time. If None or unset, defaults to
                datetime(2020, 1, 1, 0, 0).
            time_step (Optional[timedelta]): Time step. If None or unset, defaults to
                timedelta(hours=1).
            end_time (Optional[datetime]): End time. If None or unset, defaults to
                datetime(2020, 1, 2, 0, 0).
            width (int): Width of model space. Defaults to 10000.
            height (int): Height of model space. Defaults to 10000.
        """
        if end_time is None:
            end_time = datetime(2020, 1, 2, 0, 0)

        if start_time is None:
            start_time = datetime(2020, 1, 1, 0, 0)

        if time_step is None:
            time_step = timedelta(hours=1)

        area = BoxFeature(min_coord=box_min_coord, max_coord=box_max_coord, name=box_name)
        space = ContinuousSpaceWithTerrainAndCulture2d(width, height, features=[area])
        super().__init__(space, start_time, time_step, end_time)


class MockParametrisedModel(MockModel2d):
    """Example parametrised model for testing purposes."""

    @staticmethod
    def get_scenario_independent_metadata() -> ModelMetadata:
        """Return model metadata.

        Returns:
            ModelMetadata: Model metadata for Mock Parametrised Model.
        """
        sensor_curve = """
        | Distance (m)     | Value     |
        | :------------- | :----------: |
        |  10  | 0.5    |
        | 20   | 0.25 |
        | 30 | 0.1 |
        | 40 | 0.024 |
        | 50 | 0.001 |
        """

        return ModelMetadata(
            header="Parametrised Ret Model",
            subtext=[
                "This is a simple example of a RET model which has a variety of different "
                + "parameters, both categoric and numeric, and part of the experimental controls "
                + "and scenario dependent data. It can be used to demonstrate how many of the UI "
                + "elements work, including this text entry, illustrating the ability of for each "
                + "model to be self-describing.",
                "Users can override this section further to include a description for any "
                + "Scenario Independent Data, e.g.:",
                "**Sensor Curve**:",
                sensor_curve,
            ],
        )

    @staticmethod
    def get_parameters() -> ModelParameterSpecification:
        """Return parameters object.

        Returns:
            ModelParameterSpecification: Parameter specification for the model.
        """
        return ModelParameterSpecification(
            experimental_controls=ExperimentalControls(
                numeric_parameters={
                    "n1": NumericParameterSpecification[int](
                        name="N1",
                        description="Numeric Parameter 1 (integer)",
                        min_allowable=0,
                        max_allowable=100,
                    ),
                    "n2": NumericParameterSpecification[float](
                        name="N2",
                        description="Numerical Parameter 2 (floating point) with no max",
                        min_allowable=0.1,
                    ),
                },
                categoric_parameters={
                    "c1": CategoricParameterSpecification(
                        name="C2",
                        description="Categoric Parameter 1",
                        allowable_options=["Choice 1", "Choice 2", "Choice 3"],
                    )
                },
            ),
            scenario_dependent_data=ScenarioDependentData(
                numeric_parameters={
                    "x1": NumericParameterSpecification(
                        name="X1",
                        description="Scenario Dependent Parameter",
                        min_allowable=10,
                        max_allowable=20,
                    )
                },
                categoric_parameters={
                    "y1": CategoricParameterSpecification(
                        name="y1",
                        description="Scenario Dependent Parameter",
                        allowable_options=["a", "b", "c"],
                    )
                },
            ),
        )

    def __init__(self, **kwargs: str) -> None:
        """Create a new MockParametrisedModel.

        This model simply calculates the product of all numeric values and stores the
        result in <model>.result

        Args:
            kwargs (str): {"var_1": "3", "var_2": "2.03", "var_3": "-2", ...}, with arg_i up to N
                (maximum 10 supported); value must be a string representation of a number.
        """
        # Pass it on
        super().__init__(time_step=timedelta(days=1.0))

        # Calculate the result
        self.result = np.prod([float(v) for _, v in kwargs.items() if isinstance(v, str) is False])


class MockClutterModifier(ClutterModifier):
    """Mock clutter modifier, which affects all spatial locations."""

    def _is_affected(self, pos: Coordinate2dOr3d) -> bool:
        """Return whether or not the pos coordinate is affected by the clutter modifier.

        Args:
            pos (Coordinate2dOr3d): Position

        Returns:
            bool: Return true in all instances.
        """
        return True


class MockCountermeasure(Countermeasure):
    """Mock countermeasure."""

    def __init__(
        self,
        value: float = 10,
        persist_beyond_deployer: bool = False,
        agent_filter: Optional[AgentFilter] = None,
    ):
        """Create a new MockCountermeasure.

        Args:
            value (float): Countermeasure value - The amount that the
                countermeasure impacts the perceived world. Defaults to 10.
            persist_beyond_deployer (bool): Flag indicating if the countermeasure
                survives after the agent that deployed it is killed. Defaults to False
            agent_filter (Optional[AgentFilter]): An optional filter which determines
                which agents the countermeasure applies to. Defaults to None.
        """
        super().__init__(persist_beyond_deployer=persist_beyond_deployer, agent_filter=agent_filter)
        self._value = value

    def _create_clutter_modifier(self, deployer: AirAgent) -> ClutterModifier:
        """Get the clutter modifier for the countermeasure.

        Args:
            deployer (AirAgent): The agent creating the clutter modifier

        Returns:
            ClutterModifier: A new MockClutterModifier with value equal to self._value.
        """
        return MockClutterModifier(self._value, agent_filter=self.agent_filter)

    def get_new_instance(self) -> MockCountermeasure:
        """Return a new instance of a functionally identical countermeasure.

        Returns:
            MockCountermeasure: New instance of the countermeasure
        """
        return MockCountermeasure(
            value=self._value,
            persist_beyond_deployer=self.persist_beyond_deployer,
            agent_filter=self.agent_filter,
        )


class MockTrigger(Trigger):
    """Mock trigger that never triggers."""

    def __init__(self, log: bool = True):
        """Create a new MockTrigger.

        Args:
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log=log, sticky=False)

    def __str__(self) -> str:
        """Return string representation of trigger.

        Returns:
            str: String representation of the trigger
        """
        return "Mock Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:  # pragma: no cover
        """Return false at all times.

        Args:
            checker (RetAgent): Agent for the trigger

        Returns:
            bool: Always False
        """
        return False

    def get_new_instance(self) -> MockTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            MockTrigger: New instance of the trigger
        """
        return MockTrigger(log=self._log)


class MockTask(Task):
    """Mock Task that does nothing."""

    def __init__(self, log: bool = True):
        """Create a new MockTask.

        Args:
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)

    def __str__(self) -> str:
        """Return string representation of task.

        Returns:
            str: String representation of the task.
        """
        return "Mock Task"

    def _do_task_step(self, doer: RetAgent) -> None:  # pragma: no cover
        """Do nothing.

        Args:
            doer (RetAgent): the agent that will do the task
        """
        pass

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Task never completes.

        Args:
            doer (RetAgent): The agent doing the task

        Returns:
            bool: true when the task is completed, false otherwise
        """
        return False

    def get_new_instance(self) -> MockTask:
        """Return a new instance of a functionally identical task.

        Returns:
            MockTask: New instance of the task
        """
        return MockTask(log=self._log)


class MockOrder(Order):
    """Mock order with no behaviour."""

    def __init__(self):
        """Create a new MockOrder.

        Overrides parent order initialisation which requires a trigger and a task to be
        defined.
        """
        super().__init__(MockTrigger(), MockTask())


class MockBackgroundOrder(BackgroundOrder):
    """Mock background order with no behaviour."""

    def __init__(self):
        """Create a new MockBackgroundOrder.

        Overrides parent background order initialisation which requires a time period, trigger and
        a task to be defined.
        """
        super().__init__(time_period=timedelta(minutes=20), trigger=MockTrigger(), task=MockTask())


class MockGroupAgentOrder(GroupAgentOrder):
    """Mock group order with no behaviour."""

    def __init__(self):
        """Create a new MockGroupAgentOrders.

        Overrides parent order initialisation which requires a trigger and a task to be
        defined.
        """
        super().__init__(MockTrigger(), MockTask())


class MockOrderWithId(MockOrder):
    """Mock order, which can be uniquely identified, but imparts no behaviour."""

    def __init__(self, id: int) -> None:
        """Create a new Order.

        Args:
            id (int): Order identity
        """
        super().__init__()
        self.id = id

    def get_new_instance(self) -> Order:
        """Return a new instance of a functionally identical order.

        Returns:
            Order: New instance of the trigger
        """
        return MockOrderWithId(id=self.id)


class MockBackgroundOrderWithId(MockBackgroundOrder):
    """Mock order, which can be uniquely identified, but imparts no behaviour."""

    def __init__(self, id: int) -> None:
        """Create a new Order.

        Args:
            id (int): Order identity
        """
        super().__init__()
        self.id = id

    def get_new_instance(self) -> MockBackgroundOrderWithId:
        """Return a new instance of a functionally identical order.

        Returns:
            Order: New instance of the trigger
        """
        return MockBackgroundOrderWithId(id=self.id)


class MockGroupAgentOrderWithId(MockGroupAgentOrder):
    """Mock group order, which can be uniquely identified, but imparts no behaviour."""

    def __init__(self, id: int) -> None:
        """Create a new group order.

        Args:
            id (int): Order identity
        """
        super().__init__()
        self.id = id

    def get_new_instance(self) -> GroupAgentOrder:
        """Return a new instance of a functionally identical group order.

        Returns:
            Order: New instance of the trigger
        """
        return MockGroupAgentOrderWithId(id=self.id)


class MockAgent(RetAgent):
    """Mock RetAgent."""

    def __init__(self, unique_id: int, pos: Coordinate2dOr3d) -> None:
        """Create a new MockAgent.

        Args:
            unique_id (int): Agent's unique ID
            pos (Coordinate2dOr3d): Agent's position
        """
        self.unique_id = unique_id
        self.pos = pos
        self.weapons: list[Weapon] = []
        self.agent_type = AgentType.GENERIC


class MockAgentWithAffiliation(MockAgent):
    """Mock Agent with Affiliation property."""

    def __init__(self, unique_id: int, affiliation: Affiliation):
        """Create a new MockAgentWithAffiliation.

        Args:
            unique_id (int): Unique ID
            affiliation (Affiliation): Affiliation
        """
        super().__init__(unique_id=unique_id, pos=(0, 0))
        self.affiliation = affiliation


class MockAgentWithName(MockAgent):
    """Mock Agent with Name property."""

    def __init__(self, unique_id: int, name: str):
        """Create a new MockAgentWithName.

        Args:
            unique_id (int): Unique ID
            name (str): Name
        """
        super().__init__(unique_id=unique_id, pos=(0, 0))
        self.name = name


class MockBehaviour(LoggableBehaviour, ABC):
    """Mock behaviour to record it has been stepped."""

    stepped = False

    def record_step(self) -> None:
        """Record the step has been called."""
        self.stepped = True


class MockWaitBehaviour(WaitBehaviour, MockBehaviour):
    """Mock wait behaviour."""

    def _step(self, waiter: RetAgent) -> None:
        """Perform a time step's worth of wait behaviour.

        Args:
            waiter (RetAgent): Agent doing the waiting
        """
        self.record_step()


class MockMoveBehaviour(MoveBehaviour, MockBehaviour):
    """Mock move behaviour."""

    def _step(self, mover: RetAgent, destination: Coordinate) -> None:
        """Perform a time step's worth of move behaviour.

        Args:
            mover (RetAgent): Agent doing the moving
            destination (Coordinate): Target the agent is moving towards.
        """
        self.record_step()
        mover.model.space.move_agent(mover, destination)  # type: ignore


class MockBasicMoveBehaviour(MoveBehaviour, MockBehaviour):
    """Basic mock move behaviour.

    Agent moves in straight line towards destination at constant speed.
    """

    def __init__(self, speed: float, log: bool = True) -> None:
        """Create a new MockBasicMoveBehaviour.

        Args:
            speed (float): movement speed of move behaviour.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log=log)
        self._speed = speed

    def _step(self, mover: RetAgent, destination: Coordinate) -> None:
        """Perform a time step's worth of move behaviour.

        Args:
            mover (RetAgent): Agent doing the moving
            destination (Coordinate): Target the agent is moving towards.
        """
        self.record_step()

        move_vector = np.subtract(tuple(destination), tuple(mover.pos))
        move_unit_vector = move_vector / np.linalg.norm(move_vector)

        intermediate_destination = mover.pos + (move_unit_vector * self._speed)

        move_distance = mover.model.space.get_distance(mover.pos, intermediate_destination)
        max_move_distance = mover.model.space.get_distance(mover.pos, destination)

        if move_distance > max_move_distance:
            intermediate_destination = destination

        mover.model.space.move_agent(mover, intermediate_destination)  # type: ignore


class MockMoveInBandBehaviour(MoveInBandBehaviour, MockBehaviour):
    """Mock move in band behaviour."""

    def __init__(self, height_bands: Optional[list[HeightBand]] = None):
        """Create a new MockMoveInBandBehaviour.

        Args:
            height_bands (list[HeightBand]): Height bands. Defaults to no height bands.
        """
        if height_bands is None:
            height_bands = []
        super().__init__(height_bands)

    def _step(self, mover: RetAgent, destination: Coordinate) -> None:
        """Perform a time step's worth of move in band behaviour.

        Args:
            mover (RetAgent): Agent doing the moving
            destination (Coordinate): Target the agent is moving towards
        """
        self.record_step()
        destination_pos = mover.model.space.get_coordinate_in_correct_dimension(
            destination, self.height_bands
        )
        mover.model.space.move_agent(mover, destination_pos)  # type: ignore


class MockCommunicationNetwork(CommunicationNetwork):
    """Mock CommunicationNetwork with defined behaviour.

    Performs no other action other than tracking whether the agent has communicated
    worldview and orders.
    """

    def __init__(self):
        """Create a new CommunicationNetwork."""
        super().__init__()
        self.worldview_stepped = False
        self.orders_stepped = False

    def communicate_worldview_step(
        self,
        communicator: RetAgent,
        worldview_filter: Optional[PerceivedAgentFilter] = None,
        recipient_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Do one time step's worth of communicating worldview.

        Args:
            communicator (RetAgent): Agent doing the communication.
            worldview_filter (Optional[PerceivedAgentFilter]): Worldview filter to apply
            recipient_filter (Optional[AgentFilter]): Filter to apply to agent doing the
                communicating's communication network
        """
        self.worldview_stepped = True
        super().communicate_worldview_step(communicator, worldview_filter, recipient_filter)

    def communicate_orders_step(
        self,
        communicator: RetAgent,
        orders: list[Template[Order]],
        recipient_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Do one time step's worth of communicating orders.

        Args:
            communicator (RetAgent): Agent doing the communicating
            orders (list[Template[Order]]): Orders to communicate
            recipient_filter (Optional[AgentFilter]): Filter to apply to agent doing the
                communicating's communication network. Defaults to no filter
        """
        self.orders_stepped = True
        super().communicate_orders_step(communicator, orders, recipient_filter)


class MockCommunicateOrdersBehaviour(CommunicateOrdersBehaviour, MockBehaviour):
    """Mock communicate orders behaviour."""

    def _step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        orders: Union[Template[Order], list[Template[Order]]],
    ):
        """Do one time step's worth of communicating orders.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            orders (Union[Template[Order], list[Template[Order]]]): Order(s) sent from
                communicator to recipient
        """
        self.record_step()
        super()._step(communicator, recipient, orders)


class MockCommunicateWorldviewBehaviour(CommunicateWorldviewBehaviour, MockBehaviour):
    """Mock communicate worldview behaviour."""

    def _step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        worldview_filter: Optional[PerceivedAgentFilter] = None,
    ):
        """Do one time step's worth of world-view communication.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            worldview_filter (Optional[PerceivedAgentFilter]): Filter to apply to the
                communicator's world-view. Defaults to None.
        """
        self.record_step()
        super()._step(communicator, recipient, worldview_filter=worldview_filter)


class MockCommunicateMissionMessageBehaviour(CommunicateMissionMessageBehaviour, MockBehaviour):
    """Mock communicate mission message behaviour."""

    def _step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        message: str,
    ):
        """Do one time step's worth of mission message communication and log.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            message (str): Mission message to send
        """
        self.record_step()
        super()._step(communicator, recipient, message)


class MockDisableCommunicationBehaviour(DisableCommunicationBehaviour, MockBehaviour):
    """Mock disable communication behaviour.

    Does not disable any agents.
    """

    def _step(self, disabler: RetAgent) -> None:
        """Disable all applicable agents.

        Step through all agents identified by `identify_agents_to_disable`,
        which should be overridden in implementations of this abstract class,
        and disables comms from these agents, either disabling all comms, or
        comms to specific recipients, based on the payload returned from
        `identify_agents_to_disable`.

        Args:
            disabler (RetAgent): Agent doing the disabling.
        """
        self.record_step()
        super()._step(disabler)

    def identify_agents_to_disable(
        self, disabler: RetAgent
    ) -> list[tuple[RetAgent, Union[None, list[RetAgent]]]]:
        """Identify the agents to disable.

        Does not identify any agents.

        Args:
            disabler (RetAgent): Agent doing the disabling.

        Returns:
            list[tuple[RetAgent, Union[None, list[RetAgent]]]]: Agents to disable.
        """
        return []


class MockDeployCountermeasureBehaviour(DeployCountermeasureBehaviour, MockBehaviour):
    """Mock deploy countermeasure behaviour."""

    def _step(self, deployer: RetAgent, countermeasure: Countermeasure) -> None:
        """Do one time steps worth of countermeasure deployment.

        Args:
            deployer (RetAgent): The agent doing the deployment
            countermeasure (Countermeasure): The countermeasure being deployed
        """
        self.record_step()
        super()._step(deployer, countermeasure)


class RoundPerStepFireSchedule(FireSchedule):
    """Fire schedule which instructs a single round to be fired per time step."""

    def __init__(self, rounds: int):
        """Create a new RoundPerStepFireSchedule.

        Args:
            rounds (int): Number of rounds/steps to fire for
        """
        super().__init__()
        self.validate_rounds(rounds)
        self._steps = [1] * rounds


class MockFireBehaviour(FireBehaviour, MockBehaviour):
    """Mock fire behaviour."""

    def __init__(
        self,
        hostile_agent_resolver: Optional[HostileTargetResolver] = None,
        target_selector: Optional[TargetSelector] = None,
    ):
        """Create a fire behaviour, override as necessary in subclasses.

        Args:
            hostile_agent_resolver (Optional[HostileTargetResolver]): Approach for
                determining duplicate hostile agents. Defaults to no approach
            target_selector (Optional[TargetSelector]): Approach for selecting targets.
                Defaults to no approach
        """
        super().__init__(
            hostile_agent_resolver=hostile_agent_resolver,
            target_selector=target_selector,
        )

    def _step(
        self,
        firer: RetAgent,
        rounds: int,
        weapon: Weapon,
        location: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d]]] = None,
    ) -> None:
        """Do one time steps worth of fire behaviour, override in subclasses.

        Args:
            firer (RetAgent): The agent doing the firing
            rounds (int): Number of rounds to fire
            weapon (Weapon): The weapon doing the firing
            location (Optional[Coordinate2dOr3d]): The target of the fire. Defaults to
                no location
        """
        self.record_step()
        super()._step(firer, rounds, weapon, location)

    def get_fire_schedule(self, firer: RetAgent, number_of_rounds: int) -> FireSchedule:
        """Get a firing schedule for the firer.

        Args:
            firer (RetAgent): Firer.
            number_of_rounds (int): Number of rounds the firer will fire.

        Returns:
            FireSchedule: The firing schedule for the firer
        """
        return RoundPerStepFireSchedule(number_of_rounds)


class MockHideBehaviour(HideBehaviour, MockBehaviour):
    """Mock hide behaviour."""

    def _step(self, hider: RetAgent):
        """Do one time steps worth of hiding.

        Args:
            hider (RetAgent): Agent that is hiding
        """
        self.record_step()
        super()._step(hider)


class MockSenseBehaviour(SenseBehaviour, MockBehaviour):
    """Mock sense behaviour."""

    def _step(
        self, senser: RetAgent, direction: Optional[Union[float, Coordinate2dOr3d, str]] = None
    ) -> None:
        """Do one time steps worth of sensing.

        Args:
            senser (RetAgent): Agent doing the sensing.
            direction (Optional[float | Coordinate2dor3d | str]): Optional direction to sense in,
                can be input as float: degrees clockwise from y-axis, Coordinate2dor3d: a position
                used to calculate heading from sensor agent, str: the name of an Area in the space,
                a random point within the area will be chosen to look towards.
        """
        self.record_step()
        super()._step(senser, direction)


class MockSensor(Sensor):
    """Mock sensor."""

    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:
        """Run mock detection.

        Args:
            sensor_agent (RetAgent): agent using mock sensor.
            all_target_agents (list[RetAgent]): A list of potentially perceivable agents

        """
        return []

    def get_new_instance(self) -> MockSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            MockSensor: New instance of the sensor
        """
        return MockSensor(
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
        )


class MockFieldOfViewSensor(Sensor):
    """Mock sensor."""

    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:
        """Run mock detection.

        Args:
            sensor_agent (RetAgent): agent using mock sensor.
            all_target_agents (list[RetAgent]): A list of potentially perceivable agents

        """
        return [
            PerceivedAgentBasedOnAgent(
                sense_time=sensor_agent.model.get_time(),
                confidence=Confidence.IDENTIFY,
                agent=agent,
            )
            for agent in all_target_agents
        ]

    def get_new_instance(self) -> MockFieldOfViewSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            MockFieldOfViewSensor: New instance of the sensor
        """
        return MockFieldOfViewSensor(
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
        )


class MockFriendlyRetPlayAgent(RetPlayAgent):
    """Mock RetPlay Agent."""

    def __init__(self) -> None:
        """Create the Mock RetPlay Agent."""
        super().__init__(
            name="mock agent",
            affiliation="FRIENDLY",
            agent_type="",
            id=0,
            pos=[0, 0],
            icon="path",
            active_order="No active order",
            killed=False,
            in_group=False,
            perceived_agents=[],
            mission_messages=[],
        )


class MockRetPlaySpace(RetPlaySpace):
    """Mock RetPlay Space."""

    def __init__(self) -> None:
        """Create the Mock RetPlay Space."""
        super().__init__(x_min=0, x_max=1, y_min=0, y_max=1)


class MockRetPlayInitialData(RetPlayInitialData):
    """Mock RetPlay Initial Data."""

    def __init__(self) -> None:
        """Create the Mock RetPlay Initial Data."""
        super().__init__(map_size=MockRetPlaySpace())


class MockRetPlayStepData(RetPlayStepData):
    """Mock RetPlay Step Data."""

    def __init__(self, agent: Optional[RetPlayAgent] = None) -> None:
        """Create the Mock RetPlay Step Data.

        Args:
            agent (Optional[RetPlayAgent]): The RetPlayAgent for the time step. Where None, no agent
                is used. Defaults to None.
        """
        if agent is None:
            super().__init__(
                step_number=0, model_time=datetime(year=2023, month=12, day=20), agents=[]
            )
        else:
            super().__init__(
                step_number=0, model_time=datetime(year=2023, month=12, day=20), agents=[agent]
            )


class MockRetPlayMapFromPNG(RetPlayMapFromPNG):
    """Mock RetPlay Map from PNG."""

    def __init__(self, base_path: Path) -> None:
        """Create the Mock RetPlay Map from PNG.

        Args:
            base_path (Path): The base path for the PNG file.
        """
        super().__init__(MockRetPlaySpace(), [MockRetPlayStepData()], base_path)


class MockWeapon(BasicWeapon):
    """Mock Weapon."""

    def __init__(
        self,
        radius: float = 10,
        time_between_rounds: Optional[timedelta] = None,
        time_before_first_shot: Optional[timedelta] = None,
        kill_probability_per_round: float = 0.5,
        name: str = "Mock Weapon",
        min_range: Optional[float] = None,
        max_range: Optional[float] = None,
    ):
        """Create a new weapon.

        Args:
            radius (float): The radius of a single firing.
            time_between_rounds (Optional[timedelta]): The time between rounds fired. If None or
                unspecified, defaults to 30s.
            time_before_first_shot (Optional[timedelta]): The time before the first shot. If None or
                unspecified, defaults to 0s.
            kill_probability_per_round (float): The kill probability per shot fired. Default to 0.5
            name (str): Weapon name, defaults to 'Mock Weapon'.
            min_range (Optional[float]): Minimum range that the weapon can fire at. If None, there
                is no minimum range. Defaults to None
            max_range (Optional[float]): Maximum range that the weapon can fire at. If None, there
                is no maximum range. Defaults to None
        """
        if time_before_first_shot is None:
            time_before_first_shot = timedelta(seconds=0)

        if time_between_rounds is None:
            time_between_rounds = timedelta(seconds=30)

        super().__init__(
            name=name,
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=kill_probability_per_round,
            min_range=min_range,
            max_range=max_range,
        )


class MockWeaponWithVariableKillProb(BasicWeapon):
    """Mock Weapon with differing kill probability by agent type."""

    variable_kill = ProbabilityByType(
        base_kill_probability_per_round=0.5,
        kill_probability_by_agent_type={
            AgentType.ARMOUR: 0.0,
            AgentType.AIR: 1.0,
            AgentType.GENERIC: 0.2,
            AgentType.GROUP: 0.4,
            AgentType.OTHER: 0.6,
        },
    )

    def __init__(
        self,
        radius: float = 10,
        time_between_rounds: Optional[timedelta] = None,
        time_before_first_shot: Optional[timedelta] = None,
        kill_probability_per_round: Union[float, ProbabilityByType] = variable_kill,
        name: str = "Mock Weapon with variable kill prob",
        min_range: Optional[float] = None,
        max_range: Optional[float] = None,
    ):
        """Create a new weapon.

        Args:
            radius (float): The radius of a single firing.
            time_between_rounds (Optional[timedelta]): The time between rounds fired. If None or
                unspecified, defaults to 30s.
            time_before_first_shot (Optional[timedelta]): The time before the first shot. If None or
                unspecified, defaults to 0s.
            kill_probability_per_round (float or ProbabilityByType): Probability of killing a target
            name (str): Weapon name, defaults to 'Mock Weapon with variable kill prob'.
            min_range (Optional[float]): Minimum range that the weapon can fire at. If None, there
                is no minimum range. Defaults to None
            max_range (Optional[float]): Maximum range that the weapon can fire at. If None, there
                is no maximum range. Defaults to None
        """
        if time_before_first_shot is None:
            time_before_first_shot = timedelta(seconds=0)

        if time_between_rounds is None:
            time_between_rounds = timedelta(seconds=30)

        super().__init__(
            name=name,
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=kill_probability_per_round,
            min_range=min_range,
            max_range=max_range,
        )


class MockWeaponWithAmmo(BasicWeapon):
    """Mock Weapon with 60 rounds of ammo."""

    def __init__(
        self,
        radius: float = 10,
        time_between_rounds: Optional[timedelta] = None,
        time_before_first_shot: Optional[timedelta] = None,
        kill_probability_per_round: float = 0.5,
        name: str = "Mock Weapon with ammo",
        min_range: Optional[float] = None,
        max_range: Optional[float] = None,
        ammo_capacity: Optional[int] = 60,
    ):
        """Create a new weapon.

        Args:
            radius (float): The radius of a single firing.
            time_between_rounds (Optional[timedelta]): The time between rounds fired. If None or
                unspecified, defaults to 30s.
            time_before_first_shot (Optional[timedelta]): The time before the first shot. If None or
                unspecified, defaults to 0s.
            kill_probability_per_round (float): The kill probability per shot fired. Default to 0.5
            name (str): Weapon name, defaults to 'Mock Weapon with ammo'.
            min_range (Optional[float]): Minimum range that the weapon can fire at. If None, there
                is no minimum range. Defaults to None
            max_range (Optional[float]): Maximum range that the weapon can fire at. If None, there
                is no maximum range. Defaults to None
            ammo_capacity (Optional[int]): Amount of rounds gun has to fire before reload.
                If None can fire without reload. Defaults to None.
        """
        if time_before_first_shot is None:
            time_before_first_shot = timedelta(seconds=0)

        if time_between_rounds is None:
            time_between_rounds = timedelta(seconds=30)

        super().__init__(
            name=name,
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=kill_probability_per_round,
            min_range=min_range,
            max_range=max_range,
            ammo_capacity=ammo_capacity,
        )


class MockShortRangedWeapon(BasicShortRangedWeapon):
    """Mock Short Range Weapon."""

    def __init__(
        self,
        radius: float = 10,
        time_between_rounds: Optional[timedelta] = None,
        time_before_first_shot: Optional[timedelta] = None,
        kill_probability_per_round: float = 0.5,
        max_angle_inaccuracy: float = 0,
    ):
        """Create a new weapon.

        Args:
            radius (float): The radius of a single firing.
            time_between_rounds (Optional[timedelta]): The time between rounds fired. If None or
                unspecified, defaults to 30s.
            time_before_first_shot (Optional[timedelta]): The time before the first shot. If None or
                unspecified, defaults to 0s.
            kill_probability_per_round (float): The kill probability per shot fired. Default to 0.5
            max_angle_inaccuracy (float): The maximum angle of displacement to the shot representing
                inaccuracy. Defaults to 0.
        """
        if time_before_first_shot is None:
            time_before_first_shot = timedelta(seconds=0)

        if time_between_rounds is None:
            time_between_rounds = timedelta(seconds=30)

        super().__init__(
            name="Mock Short Range Weapon",
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=kill_probability_per_round,
            max_angle_inaccuracy=max_angle_inaccuracy,
        )


class MockLongRangedWeapon(BasicLongRangedWeapon):
    """Mock Long Range Weapon."""

    def __init__(
        self,
        radius: float = 10,
        time_between_rounds: Optional[timedelta] = None,
        time_before_first_shot: Optional[timedelta] = None,
        kill_probability_per_round: float = 0.5,
        max_percentage_inaccuracy: float = 0,
    ):
        """Create a new weapon.

        Args:
            radius (float): The radius of a single firing.
            time_between_rounds (Optional[timedelta]): The time between rounds fired. If None or
                unspecified, defaults to 30s.
            time_before_first_shot (Optional[timedelta]): The time before the first shot. If None or
                unspecified, defaults to 0s.
            kill_probability_per_round (float): The kill probability per shot fired. Default to 0.5
            max_percentage_inaccuracy (float): max_percentage_inaccuracy (float): Maximum percentage
                inaccuracy (as a number from 0 to 1). Defaults to 0.
        """
        if time_before_first_shot is None:
            time_before_first_shot = timedelta(seconds=0)

        if time_between_rounds is None:
            time_between_rounds = timedelta(seconds=30)

        super().__init__(
            name="Mock Long Range Weapon.",
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=kill_probability_per_round,
            max_percentage_inaccuracy=max_percentage_inaccuracy,
        )


class MockLogger(RetLogger):
    """A blank logger which does nothing.

    Override methods in subclasses to expose desired logs/information.
    """

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
        log_position_and_culture: bool = False,
    ):
        """Create a MockLogger.

        Args:
            model (RetModel): The model to log.
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
            log_position_and_culture (bool): Flag to log position and cultures of agents.
                Defaults to False.
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
        self.log_position_and_culture = log_position_and_culture

    def log_agent(self, agent: RetAgent, message: str = ""):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent to be recorded.
            message (str): An optional message to be logged. Defaults to empty string.
        """
        pass

    def log_behaviour(
        self, agent: RetAgent, behaviour_str: str, message: str = ""
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent performing the behaviour
            behaviour_str (str): Name of the behaviour
            message (str): Extra information. Defaults to ""
        """
        pass

    def log_behaviour_selection_record(
        self,
        agent: RetAgent,
        handler: str,
        behaviour_type: type,
        candidates: Sequence[Behaviour],
        selected: Behaviour,
        message: str = "",
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent for who the behaviour has been selected.
            handler (str): The handler used to select the behaviour.
            behaviour_type(type): The behaviour type selected.
            candidates (Sequence[Behaviour]): List of candidate behaviours for selection
            selected (Behaviour): The selected behaviour
            message (str): Any additional contextual message. Defaults to "".
        """
        pass

    def log_death(
        self,
        target: RetAgent,
        killer: Optional[RetAgent],
        shot_id: Optional[int],
        weapon_name: Optional[str],
        message: str = "",
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            target (RetAgent): The killed agent
            shot_id (Optional[int]): The shot ID
            weapon_name (Optional[str]): The killer weapon's name
            killer (Optional[RetAgent]): The killer
            message (str): Extra information. Defaults to ""
        """
        pass

    def log_no_available_behaviour(
        self, agent: RetAgent, handler: str, behaviour_type: type
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent for who the behaviour has been selected.
            handler (str): The handler used to select the behaviour.
            behaviour_type(type): The behaviour type selected.
        """
        pass

    def log_observation_record(
        self,
        agent: RetAgent,
        observation_result: list[PerceivedAgent],
        message: str = "",
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent doing the observation
            observation_result (list[PerceivedAgent]): The result of the observation
            message (str): Additional information. Defaults to "".
        """
        pass

    def log_perception_record(
        self, receiver: SensorFusionAgent, message: str = ""
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            receiver (SensorFusionAgent): The agent with an updated perception record.
            message (str): Additional information. Defaults to "".
        """
        pass

    def log_shot_fired(
        self,
        agent: RetAgent,
        targets: list[RetAgent],
        shot_id: int,
        weapon_name: str,
        weapon_radius: float,
        location: Optional[Coordinate2dOr3d] = None,
        message: str = "",
        remaining_ammo: Optional[int] = None,
    ) -> None:  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent firing the shot
            targets (list[RetAgent]): The targets of the shot
            shot_id (int): ID of the shot fired
            weapon_name (str): The name of the weapon firing
            weapon_radius (float): The blast radius of the weapon the fired the shot.
            location (Optional[Coordinate2dOr3d]): The location being fired at. Defaults
                to None.
            message (str): Additional information. Defaults to "".
            remaining_ammo (Optional[int]): The ammo remaining on weapon, if None
                remaining ammo is infinite, Defaults to None
        """
        pass

    def log_task(
        self, agent: RetAgent, task_str: str, status: TaskLogStatus, message: str = ""
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent performing the task
            task_str (str): Name of the task
            status (TaskLogStatus): Status of the task
            message (str): Extra information. Defaults to ""
        """
        pass

    def log_trigger(
        self,
        agent: RetAgent,
        trigger_str: str,
        status: TriggerLogStatus,
        message: str = "",
    ):  # pragma: no cover
        """Placeholder method preventing RetLogger from doing anything. Override as needed.

        Args:
            agent (RetAgent): The agent checking the trigger
            trigger_str (str): The trigger name
            status (TriggerLogStatus): Status of the trigger
            message (str): Extra information. Defaults to ""
        """
        pass


class MockShotFiredLogger(MockLogger):
    """A Mock logger that records shots fired."""

    def log_shot_fired(
        self,
        agent: RetAgent,
        targets: list[RetAgent],
        shot_id: int,
        weapon_name: str,
        weapon_radius: float,
        location: Optional[Coordinate2dOr3d] = None,
        message: str = "",
        remaining_ammo: Optional[int] = None,
    ) -> None:
        """Log shots fired records.

        Args:
            agent (RetAgent): The agent firing the shot
            targets (list[RetAgent]): The targets of the shot
            shot_id (int): ID of the shot fired
            weapon_name (str): The name of the weapon firing
            weapon_radius (float): The blast radius of the weapon the fired the shot.
            location (Optional[Coordinate2dOr3d]): The location being fired at. Defaults
                to None.
            message (str): Additional information. Defaults to "".
            remaining_ammo (Optional[int]): The ammo remaining on weapon, if None
                remaining ammo is infinite, Defaults to None
        """
        self.recorded_firing = {
            "time": self.model.get_time(),
            "agent name": agent.name,
            "agent ID": agent.unique_id,
            "targets": [t.unique_id for t in targets],
            "aim location": location,
            "shot ID": shot_id,
            "shot radius": weapon_radius,
            "weapon name": weapon_name,
            "message": message,
            "remaining ammo": remaining_ammo,
        }


class MockMultipleShotsFiredLogger(MockLogger):
    """A Mock logger that records shots fired."""

    logged_shots_list: list = []

    def log_shot_fired(
        self,
        agent: RetAgent,
        targets: list[RetAgent],
        shot_id: int,
        weapon_name: str,
        weapon_radius: float,
        location: Optional[Coordinate2dOr3d] = None,
        message: str = "",
        remaining_ammo: Optional[int] = None,
    ) -> None:
        """Log shots fired records.

        Args:
            agent (RetAgent): The agent firing the shot
            targets (list[RetAgent]): The targets of the shot
            shot_id (int): ID of the shot fired
            weapon_name (str): The name of the weapon firing
            weapon_radius (float): The blast radius of the weapon the fired the shot.
            location (Optional[Coordinate2dOr3d]): The location being fired at. Defaults
                to None.
            message (str): Additional information. Defaults to "".
            remaining_ammo (Optional[int]): The ammo remaining on weapon, if None
                remaining ammo is infinite, Defaults to None
        """
        self.logged_shots_list.append(
            {
                "time": self.model.get_time(),
                "agent name": agent.name,
                "agent ID": agent.unique_id,
                "targets": [t.unique_id for t in targets],
                "aim location": location,
                "shot radius": weapon_radius,
                "weapon name": weapon_name,
                "shot ID": shot_id,
                "message": message,
                "remaining ammo": remaining_ammo,
            }
        )
