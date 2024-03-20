"""Test Move and Fire task."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.groupagent import GroupAgent
from ret.logger.logger import RetLogger
from ret.orders.order import Order
from ret.orders.tasks.act_and_move import ActAndMoveTask
from ret.orders.tasks.fire import FireAtTargetTask
from ret.orders.tasks.move import FixedRandomMoveTask
from ret.orders.triggers.immediate import ImmediateTrigger
from ret.testing.mocks import MockFireBehaviour, MockModel2d, MockMoveBehaviour
from ret.weapons.weapon import BasicWeapon

if TYPE_CHECKING:
    from ret.behaviours import Behaviour
    from ret.model import RetModel
    from ret.orders.order import TaskLogStatus


def test_basic_fire_and_move():
    """Test basic 2D ground movement and fire task."""
    model = MockModel2d()

    behaviours: list[Behaviour] = [MockMoveBehaviour(), MockFireBehaviour()]

    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
        weapons=[
            BasicWeapon(
                name="Weapon",
                radius=5,
                time_between_rounds=timedelta(seconds=15),
                time_before_first_shot=timedelta(seconds=0),
                kill_probability_per_round=1,
            )
        ],
    )

    target = RetAgent(
        model=model,
        pos=(28, 28),
        name="Target",
        affiliation=Affiliation.HOSTILE,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20,
    )

    min_displacement = 10
    max_displacement = 20

    fire_task = FireAtTargetTask(target=(30, 30))
    move_task = FixedRandomMoveTask(
        min_displacement=min_displacement,
        max_displacement=max_displacement,
        x_min=model.space.x_min,
        x_max=model.space.x_max,
        y_min=model.space.y_min,
        y_max=model.space.y_max,
        tolerance=0.5,
    )

    fire_and_move_task = ActAndMoveTask(action_task=fire_task, move_task=move_task)

    fire_and_move_task.do_task_step(agent)
    assert not fire_and_move_task.is_task_complete(agent)
    assert agent.pos == (0, 0)
    assert target.killed

    fire_and_move_task.do_task_step(agent)
    assert fire_and_move_task.is_task_complete(agent)
    assert agent.pos is not None

    displacement = np.linalg.norm(agent.pos)
    assert displacement >= min_displacement
    assert displacement <= max_displacement


def test_group_agent_move_and_fire():
    """Test that a group of agents can be commanded with an action and move task."""
    model = MockModel2d()

    def gen_agent(name: str) -> RetAgent:
        behaviours: list[Behaviour] = [MockMoveBehaviour(), MockFireBehaviour()]

        return RetAgent(
            model=model,
            pos=(0, 0),
            name=name,
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=behaviours,
            weapons=[
                BasicWeapon(
                    name="Weapon",
                    radius=5,
                    time_between_rounds=timedelta(seconds=15),
                    time_before_first_shot=timedelta(seconds=0),
                    kill_probability_per_round=1,
                )
            ],
        )

    a1 = gen_agent("Agent 1")
    a2 = gen_agent("Agent 2")
    a3 = gen_agent("Agent 3")
    a4 = gen_agent("Agent 4")

    min_displacement = 10
    max_displacement = 20

    fire_task = FireAtTargetTask(target=(30, 30))
    move_task = FixedRandomMoveTask(
        min_displacement=min_displacement,
        max_displacement=max_displacement,
        x_min=model.space.x_min,
        x_max=model.space.x_max,
        y_min=model.space.y_min,
        y_max=model.space.y_max,
        tolerance=0.5,
    )

    fire_and_move_task = ActAndMoveTask(fire_task, move_task=move_task)

    order = Order(trigger=ImmediateTrigger(), task=fire_and_move_task, persistent=False)

    GroupAgent(
        model=model,
        name="Unit",
        affiliation=Affiliation.FRIENDLY,
        agents=[a1, a2, a3, a4],
        agent_orders=[order],
    )

    model.step()

    for a in [a1, a2, a3, a4]:
        assert a.pos == (0, 0)

    model.step()

    for a in [a1, a2, a3, a4]:
        displacement = np.linalg.norm(a.pos)
        assert displacement >= min_displacement
        assert displacement <= max_displacement


class MockTaskLogger(RetLogger):
    """Mock logger that collected task messages."""

    def __init__(self, model: RetModel):
        """Create a new MockTaskLogger.

        Args:
            model (RetModel): Model that is being logged.
        """
        self.task_logs: list[str] = []
        super().__init__(model=model)

    def log_task(
        self, agent: RetAgent, task_str: str, status: TaskLogStatus, message: str = ""
    ) -> None:
        """Log task.

        Args:
            Args:
            agent (RetAgent): The agent performing the task
            task_str (str): Name of the task
            status (TaskLogStatus): Status of the task
            message (str): Extra information. Defaults to ""
        """
        self.task_logs.append(task_str)


def test_act_and_move_logging():
    """Test log message from act and move task."""
    fire_task = FireAtTargetTask(target=(30, 30))
    move_task = FixedRandomMoveTask(
        min_displacement=0,
        max_displacement=20,
        x_min=0,
        x_max=50,
        y_min=0,
        y_max=50,
        tolerance=0.5,
    )

    task = ActAndMoveTask(fire_task, move_task=move_task)

    model = MockModel2d()
    model.logger = MockTaskLogger(model)

    behaviours: list[Behaviour] = [MockMoveBehaviour(), MockFireBehaviour()]

    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
        behaviours=behaviours,
        weapons=[
            BasicWeapon(
                name="Weapon",
                radius=5,
                time_between_rounds=timedelta(seconds=15),
                time_before_first_shot=timedelta(seconds=0),
                kill_probability_per_round=1,
            )
        ],
    )

    task.do_task_step(agent)

    assert "Act [Fire at Target Task] and Move" in model.logger.task_logs
    assert "Fire at Target Task" in model.logger.task_logs


def test_compound_task_incomplete_on_initialisation():
    """Test that act and fire task is incomplete prior to move being resolved."""
    fire_task = FireAtTargetTask(target=(30, 30))
    move_task = FixedRandomMoveTask(
        min_displacement=0,
        max_displacement=20,
        x_min=0,
        x_max=50,
        y_min=0,
        y_max=50,
        tolerance=0.5,
    )

    task = ActAndMoveTask(fire_task, move_task=move_task)
    model = MockModel2d()

    agent = RetAgent(
        model=model,
        pos=(0, 0),
        name="Agent",
        affiliation=Affiliation.FRIENDLY,
        critical_dimension=2.0,
        reflectivity=0.1,
        temperature=20.0,
    )
    assert task.is_task_complete(agent) is False
