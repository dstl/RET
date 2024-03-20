"""Tests for formation and formation move tasks."""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.groupagent import GroupAgent
from ret.communication.communicationreceiver import GroupAgentCommunicationReceiver
from ret.formations import (
    SquareFormationFullColumns,
    SquareFormationFullRows,
    SquareFormationRounded,
)
from ret.orders.order import GroupAgentOrder, Order
from ret.orders.tasks.move import GroupFormationMoveTask, MoveTask
from ret.orders.triggers.immediate import ImmediateTrigger
from ret.testing.mocks import MockModel2d, MockMoveBehaviour
from parameterized import parameterized
from pytest import raises

if TYPE_CHECKING:
    from typing import Tuple, Type

    from ret.behaviours import Behaviour
    from ret.formations import Formation
    from ret.types import Vector2d


def test_basic_move_task():
    """Test basic formation movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    subordinate_agent_2 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 2",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1, subordinate_agent_2],
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(task, SquareFormationRounded(separation=200))

    formation_move_task.do_task_step(group_agent)
    assert formation_move_task.is_task_complete(group_agent)

    assert subordinate_agent_1._orders is not None
    assert len(subordinate_agent_1._orders) == 1
    assert isinstance(subordinate_agent_1._orders[0], Order)
    assert isinstance(subordinate_agent_1._orders[0]._task, MoveTask)
    assert subordinate_agent_1._orders[0]._task._destination == (4900, 4900)  # type: ignore
    assert isinstance(subordinate_agent_1._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_1._orders[0].priority == 0

    assert subordinate_agent_2._orders is not None
    assert len(subordinate_agent_2._orders) == 1
    assert isinstance(subordinate_agent_2._orders[0], Order)
    assert isinstance(subordinate_agent_2._orders[0]._task, MoveTask)
    assert subordinate_agent_2._orders[0]._task._destination == (4900, 5100)  # type: ignore
    assert isinstance(subordinate_agent_2._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_2._orders[0].priority == 0


def test_single_basic_move_task_with_move():
    """Test basic formation movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1],
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(task, SquareFormationRounded(separation=200))

    group_agent.add_orders(Order(ImmediateTrigger(), formation_move_task))

    model.step()
    assert formation_move_task.is_task_complete(group_agent)

    assert subordinate_agent_1._orders is not None
    assert len(subordinate_agent_1._orders) == 1
    assert isinstance(subordinate_agent_1._orders[0], Order)
    assert isinstance(subordinate_agent_1._orders[0]._task, MoveTask)
    assert subordinate_agent_1._orders[0]._task._destination == (5000, 5000)  # type: ignore
    assert isinstance(subordinate_agent_1._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_1._orders[0].priority == 0

    assert subordinate_agent_1.pos == (5000, 5000)

    model.step()


def test_basic_formation_layout_move_task():
    """Test basic formation layout movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    subordinate_agent_2 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 2",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    subordinate_agent_3 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 3",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    subordinate_agent_4 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 4",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1, subordinate_agent_2, subordinate_agent_3, subordinate_agent_4],
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(task, SquareFormationRounded(separation=200))

    formation_move_task.do_task_step(group_agent)
    assert formation_move_task.is_task_complete(group_agent)

    assert subordinate_agent_1._orders[0]._task._destination == (4900, 4900)  # type: ignore
    assert subordinate_agent_2._orders[0]._task._destination == (4900, 5100)  # type: ignore
    assert subordinate_agent_3._orders[0]._task._destination == (5100, 4900)  # type: ignore
    assert subordinate_agent_4._orders[0]._task._destination == (5100, 5100)  # type: ignore


def test_warning_formation_no_agents_move_task():
    """Test a warning is given in formation move task with no agents."""
    model = MockModel2d()

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(task, SquareFormationRounded(separation=200))
    with warnings.catch_warnings(record=True) as w:
        formation_move_task.do_task_step(group_agent)
    assert "GroupAgent has no agents to move." in str(w[0].message)


def test_error_formation_non_group_agents_move_task():
    """Test a ValueError is given in formation move task with a non group agents."""
    model = MockModel2d()

    test_agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(task, SquareFormationRounded(separation=200))
    with raises(
        ValueError,
        match=(
            "Formation move task not given to a GroupAgent, "
            + f"receiving agent was of type: '{test_agent.agent_type.name}'"
        ),
    ):
        formation_move_task.do_task_step(test_agent)


@parameterized.expand(
    [
        [
            SquareFormationRounded,
            [(-50, -50), (-50, 50), (50, -50), (50, 50), (-150, -50), (-150, 50), (-50, -150)],
        ],
        [
            SquareFormationFullRows,
            [(-50, -50), (-50, 50), (50, -50), (50, 50), (-150, -50), (-150, 50), (150, -50)],
        ],
        [
            SquareFormationFullColumns,
            [(-50, -50), (-50, 50), (50, -50), (50, 50), (-50, -150), (-50, 150), (50, -150)],
        ],
    ]
)
def test_formation_layout_move_task(
    formation_type: Type[Formation], agent_positions: list[Tuple[float, float]]
):
    """Test basic formation layout movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agents: list = []
    for i in range(1, 8):
        subordinate_agents.append(
            RetAgent(
                model=model,
                pos=(0, 0),
                name=f"Agent {i}",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                behaviours=behaviours,
            )
        )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=subordinate_agents,
    )

    task = MoveTask((0, 0), 1)

    formation_move_task = GroupFormationMoveTask(
        task, formation_type(separation=100, grid_ratio=(3, 2))
    )

    formation_move_task.do_task_step(group_agent)
    assert formation_move_task.is_task_complete(group_agent)

    for i in range(0, 7):
        agent = subordinate_agents[i]
        assert agent._orders[0]._task._destination == agent_positions[i]  # type: ignore


@parameterized.expand(
    [
        [
            (0, 0),
            [(-50, -50), (-50, 50), (50, -50), (50, 50), (-150, -50), (-150, 50), (-50, -150)],
        ],
        [
            (1, 0),
            [(-50, 50), (50, 50), (-50, -50), (50, -50), (-50, 150), (50, 150), (-150, 50)],
        ],
        [
            (5, 0),
            [(-50, 50), (50, 50), (-50, -50), (50, -50), (-50, 150), (50, 150), (-150, 50)],
        ],
        [
            (0.1, 0),
            [(-50, 50), (50, 50), (-50, -50), (50, -50), (-50, 150), (50, 150), (-150, 50)],
        ],
        [
            (0, 1),
            [(-50, -50), (-50, 50), (50, -50), (50, 50), (-150, -50), (-150, 50), (-50, -150)],
        ],
        [
            (1, 1),
            [
                (-70.7106781187, 0),
                (0, 70.7106781187),
                (0, -70.7106781187),
                (70.7106781187, 0),
                (-141.4213562373095, 70.7106781187),
                (-70.7106781187, 141.4213562373095),
                (-141.4213562373095, -70.7106781187),
            ],
        ],
        [
            (-1, 1),
            [
                (0, -70.7106781187),
                (-70.7106781187, 0),
                (70.7106781187, 0),
                (0, 70.7106781187),
                (-70.7106781187, -141.4213562373095),
                (-141.4213562373095, -70.7106781187),
                (70.7106781187, -141.4213562373095),
            ],
        ],
        [
            (1, -1),
            [
                (0, 70.7106781187),
                (70.7106781187, 0),
                (-70.7106781187, 0),
                (0, -70.7106781187),
                (70.7106781187, 141.4213562373095),
                (141.4213562373095, 70.7106781187),
                (-70.7106781187, 141.4213562373095),
            ],
        ],
        [
            (-1, -1),
            [
                (70.7106781187, 0),
                (0, -70.7106781187),
                (0, 70.7106781187),
                (-70.7106781187, 0),
                (141.4213562373095, -70.7106781187),
                (70.7106781187, -141.4213562373095),
                (141.4213562373095, 70.7106781187),
            ],
        ],
        [
            (0.5, 1),
            [
                (-67.0820393249937, -22.360679774997898),
                (-22.360679774997898, 67.0820393249937),
                (22.360679774997898, -67.0820393249937),
                (67.0820393249937, 22.360679774997898),
                (-156.52475842498527, 22.360679774997898),
                (-111.80339887498948, 111.80339887498948),
                (-111.80339887498948, -111.80339887498948),
            ],
        ],
    ]
)
def test_formation_heading_move_task(heading: Vector2d, agent_positions: list[Tuple[float, float]]):
    """Test basic formation layout movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agents: list = []
    for i in range(1, 8):
        subordinate_agents.append(
            RetAgent(
                model=model,
                pos=(0, 0),
                name=f"Agent {i}",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                behaviours=behaviours,
            )
        )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=subordinate_agents,
    )

    task = MoveTask((0, 0), 1)

    formation_move_task = GroupFormationMoveTask(
        task, SquareFormationRounded(separation=100, grid_ratio=(3, 2), heading=heading)
    )

    formation_move_task.do_task_step(group_agent)
    assert formation_move_task.is_task_complete(group_agent)

    for i in range(0, 7):
        destination = subordinate_agents[i]._orders[0]._task._destination
        assert math.isclose(destination[0], agent_positions[i][0], abs_tol=1e-9)
        assert math.isclose(destination[1], agent_positions[i][1], abs_tol=1e-9)


def test_basic_move_include_killed_task():
    """Test basic formation movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )
    subordinate_agent_1._killed = True

    subordinate_agent_2 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 2",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    subordinate_agent_3 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 3",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1, subordinate_agent_2, subordinate_agent_3],
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(
        task, SquareFormationRounded(separation=200, include_killed=True)
    )

    formation_move_task.do_task_step(group_agent)
    assert formation_move_task.is_task_complete(group_agent)

    assert subordinate_agent_1._orders is not None
    assert len(subordinate_agent_1._orders) == 1
    assert isinstance(subordinate_agent_1._orders[0], Order)
    assert isinstance(subordinate_agent_1._orders[0]._task, MoveTask)
    assert subordinate_agent_1._orders[0]._task._destination == (4900, 4900)  # type: ignore
    assert isinstance(subordinate_agent_1._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_1._orders[0].priority == 0

    assert subordinate_agent_2._orders is not None
    assert len(subordinate_agent_2._orders) == 1
    assert isinstance(subordinate_agent_2._orders[0], Order)
    assert isinstance(subordinate_agent_2._orders[0]._task, MoveTask)
    assert subordinate_agent_2._orders[0]._task._destination == (4900, 5100)  # type: ignore
    assert isinstance(subordinate_agent_2._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_2._orders[0].priority == 0

    assert subordinate_agent_3._orders is not None
    assert len(subordinate_agent_3._orders) == 1
    assert isinstance(subordinate_agent_3._orders[0], Order)
    assert isinstance(subordinate_agent_3._orders[0]._task, MoveTask)
    assert subordinate_agent_3._orders[0]._task._destination == (5100, 4900)  # type: ignore
    assert isinstance(subordinate_agent_3._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_3._orders[0].priority == 0


def test_basic_move_exclude_killed_task():
    """Test basic formation movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )
    subordinate_agent_1._killed = True

    subordinate_agent_2 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 2",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    subordinate_agent_3 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 3",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1, subordinate_agent_2, subordinate_agent_3],
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(
        task, SquareFormationRounded(separation=200, include_killed=False)
    )

    formation_move_task.do_task_step(group_agent)
    assert formation_move_task.is_task_complete(group_agent)

    assert subordinate_agent_1._orders is not None
    assert len(subordinate_agent_1._orders) == 0

    assert subordinate_agent_2._orders is not None
    assert len(subordinate_agent_2._orders) == 1
    assert isinstance(subordinate_agent_2._orders[0], Order)
    assert isinstance(subordinate_agent_2._orders[0]._task, MoveTask)
    assert subordinate_agent_2._orders[0]._task._destination == (4900, 4900)  # type: ignore
    assert isinstance(subordinate_agent_2._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_2._orders[0].priority == 0

    assert subordinate_agent_3._orders is not None
    assert len(subordinate_agent_3._orders) == 1
    assert isinstance(subordinate_agent_3._orders[0], Order)
    assert isinstance(subordinate_agent_3._orders[0]._task, MoveTask)
    assert subordinate_agent_3._orders[0]._task._destination == (4900, 5100)  # type: ignore
    assert isinstance(subordinate_agent_3._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_3._orders[0].priority == 0


def test_basic_move_task_low_priority():
    """Test basic formation movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )
    agent_order = Order(ImmediateTrigger(), MoveTask((1000, 1000), 1), priority=10)

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(
        task,
        SquareFormationRounded(separation=200, include_killed=False),
        subordinate_move_priority=1,
    )
    group_order = Order(ImmediateTrigger(), formation_move_task, priority=0)

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1],
        agent_orders=[agent_order],
        orders=[group_order],
    )

    assert group_agent._orders is not None
    assert len(group_agent._orders) == 1
    assert group_agent.pos == (0, 0)
    assert subordinate_agent_1._orders is not None
    assert len(subordinate_agent_1._orders) == 1
    assert subordinate_agent_1.pos == (0, 0)

    model.step()
    assert len(group_agent._orders) == 1
    assert len(subordinate_agent_1._orders) == 2
    assert group_agent.pos == (1000, 1000)
    assert subordinate_agent_1.pos == (1000, 1000)

    model.step()
    assert len(group_agent._orders) == 0
    assert len(subordinate_agent_1._orders) == 1
    assert group_agent.pos == (5000, 5000)
    assert subordinate_agent_1.pos == (5000, 5000)

    model.step()
    assert len(group_agent._orders) == 0
    assert len(subordinate_agent_1._orders) == 0
    assert group_agent.pos == (5000, 5000)
    assert subordinate_agent_1.pos == (5000, 5000)


def test_basic_move_task_high_priority():
    """Test basic formation movement task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )
    agent_order = Order(ImmediateTrigger(), MoveTask((1000, 1000), 1), priority=1)

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(
        task,
        SquareFormationRounded(separation=200, include_killed=False),
        subordinate_move_priority=10,
    )
    group_order = Order(ImmediateTrigger(), formation_move_task, priority=0)

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1],
        agent_orders=[agent_order],
        orders=[group_order],
    )

    assert group_agent._orders is not None
    assert len(group_agent._orders) == 1
    assert group_agent.pos == (0, 0)
    assert subordinate_agent_1._orders is not None
    assert len(subordinate_agent_1._orders) == 1
    assert subordinate_agent_1.pos == (0, 0)

    model.step()
    assert len(group_agent._orders) == 1
    assert len(subordinate_agent_1._orders) == 2
    assert group_agent.pos == (5000, 5000)
    assert subordinate_agent_1.pos == (5000, 5000)

    model.step()
    assert len(group_agent._orders) == 0
    assert len(subordinate_agent_1._orders) == 1
    assert group_agent.pos == (1000, 1000)
    assert subordinate_agent_1.pos == (1000, 1000)

    model.step()
    assert len(group_agent._orders) == 0
    assert len(subordinate_agent_1._orders) == 0
    assert group_agent.pos == (1000, 1000)
    assert subordinate_agent_1.pos == (1000, 1000)


def test_basic_move_task_alternative_trigger():
    """Test basic formation movement task with alternative trigger."""
    model = MockModel2d()

    class MockImmediateTrigger(ImmediateTrigger):
        """A Mock ImmediateTrigger which triggers immediately."""

        def __init__(self, log: bool = True):
            """Create a new MockImmediateTrigger.

            Args:
                log (bool): whether to log or not. Defaults to True.
            """
            super().__init__(log=log)

        def get_new_instance(self) -> MockImmediateTrigger:
            """Return a new instance of a functionally identical trigger.

            Returns:
                ImmediateTrigger: New instance of the trigger
            """
            return MockImmediateTrigger(log=self._log)

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(
        task,
        SquareFormationRounded(separation=200, include_killed=False),
        subordinate_move_trigger=MockImmediateTrigger(),
    )
    group_order = Order(ImmediateTrigger(), formation_move_task, priority=0)

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1],
        orders=[group_order],
    )

    assert group_agent._orders is not None
    assert len(group_agent._orders) == 1
    assert group_agent.pos == (0, 0)
    assert subordinate_agent_1._orders is not None
    assert len(subordinate_agent_1._orders) == 0
    assert subordinate_agent_1.pos == (0, 0)
    assert isinstance(group_agent._orders[0]._trigger, ImmediateTrigger)

    model.step()
    assert len(group_agent._orders) == 1
    assert len(subordinate_agent_1._orders) == 1
    assert group_agent.pos == (5000, 5000)
    assert subordinate_agent_1.pos == (5000, 5000)
    assert isinstance(subordinate_agent_1._orders[0]._trigger, MockImmediateTrigger)

    model.step()
    assert len(group_agent._orders) == 0
    assert len(subordinate_agent_1._orders) == 0
    assert group_agent.pos == (5000, 5000)
    assert subordinate_agent_1.pos == (5000, 5000)


def test_non_integer_grid_ratio_first_element():
    """Tests the right exception is raised if the first ratio element is a float."""
    formation = SquareFormationRounded(grid_ratio=(0.3, 5))
    with raises(ValueError) as e:
        formation.get_formation(number_of_agents=1)
    assert e.value.args[0] == "Ratio values must be integers (0.3, 5)."


def test_non_integer_grid_ratio_last_element():
    """Tests the right exception is raised if the second ratio element is a float."""
    formation = SquareFormationRounded(grid_ratio=(5, 0.2))
    with raises(ValueError) as e:
        formation.get_formation(number_of_agents=1)
    assert e.value.args[0] == "Ratio values must be integers (5, 0.2)."


def test_column_formation():
    """Tests the a single line column formation is returned if a ratio of (0, X) is given."""
    formation = SquareFormationRounded(grid_ratio=(0, 6))
    agent_positions = formation.get_formation(number_of_agents=7)
    expected_positions = [(0, 0), (0, -1), (0, 1), (0, -2), (0, 2), (0, -3), (0, 3)]

    assert agent_positions == expected_positions


def test_row_formation():
    """Tests the a single line row formation is returned if a ratio of (0, X) is given."""
    formation = SquareFormationRounded(grid_ratio=(5, 0))
    agent_positions = formation.get_formation(number_of_agents=7)
    expected_positions = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (-3, 0), (3, 0)]

    assert agent_positions == expected_positions


def test_communicate_formation_move_task():
    """Tests that FormationMove tasks sent over the communication network are executed."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour()]

    subordinate_agent_1 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 1",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    subordinate_agent_2 = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent 2",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
    )

    group_agent = GroupAgent(
        model=model,
        name="Group Agent",
        affiliation=Affiliation.FRIENDLY,
        agents=[subordinate_agent_1, subordinate_agent_2],
    )

    task = MoveTask((5000, 5000), 1)

    formation_move_task = GroupFormationMoveTask(
        task, SquareFormationRounded(separation=200), subordinate_move_priority=1
    )

    communication_receiver = GroupAgentCommunicationReceiver()

    order = GroupAgentOrder(
        trigger=ImmediateTrigger(),
        task=formation_move_task,
    )

    payload = {"orders": order}
    communication_receiver.receive(group_agent, payload)

    assert len(group_agent._orders) == 1
    assert len(subordinate_agent_1._orders) == 0
    assert len(subordinate_agent_2._orders) == 0

    assert isinstance(group_agent._orders[0], GroupAgentOrder)
    assert isinstance(group_agent._orders[0]._task, GroupFormationMoveTask)
    assert isinstance(group_agent._orders[0]._trigger, ImmediateTrigger)
    assert group_agent._orders[0].priority == 0

    assert subordinate_agent_1.pos == (0, 0)
    assert subordinate_agent_2.pos == (0, 0)

    model.step()

    # Subordinate agents are always stepped after the group agent using the
    # group agent's scheduler so subordinate agents will do their disseminated
    # MoveTasks in the same step as the GroupFormationMoveTask.

    assert len(group_agent._orders) == 1
    assert len(subordinate_agent_1._orders) == 1
    assert len(subordinate_agent_2._orders) == 1

    assert subordinate_agent_1.pos == (4900, 4900)
    assert subordinate_agent_2.pos == (4900, 5100)

    assert isinstance(subordinate_agent_1._orders[0], Order)
    assert isinstance(subordinate_agent_1._orders[0]._task, MoveTask)
    assert isinstance(subordinate_agent_1._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_1._orders[0].priority == 1
    assert subordinate_agent_1._orders[0]._task._destination == (4900, 4900)  # type: ignore

    assert isinstance(subordinate_agent_2._orders[0], Order)
    assert isinstance(subordinate_agent_2._orders[0]._task, MoveTask)
    assert isinstance(subordinate_agent_2._orders[0]._trigger, ImmediateTrigger)
    assert subordinate_agent_2._orders[0].priority == 1
    assert subordinate_agent_2._orders[0]._task._destination == (4900, 5100)  # type: ignore

    model.step()
    # Orders get cleared next timestep
    assert len(group_agent._orders) == 0
    assert len(subordinate_agent_1._orders) == 0
    assert len(subordinate_agent_2._orders) == 0
