"""Compound task which comprises of an action followed by a random move order."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.orders.order import CompoundTask

if TYPE_CHECKING:
    from mesa_ret.orders.order import Task
    from mesa_ret.orders.tasks.move import FixedRandomMoveTask


class ActAndMoveTask(CompoundTask):
    """Wrapper around action and move composite task."""

    def __init__(self, action_task: Task, move_task: FixedRandomMoveTask, log: bool = True):
        """Create a new act and move task.

        Args:
            action_task (Task): The task to do before moving
            move_task (RandomMoveTask): The movement task to complete
            log (bool): Whether or not to log the task. Defaults to true
        """
        self._action_task = action_task
        self._move_task = move_task
        super().__init__(tasks=[action_task, move_task], log=log)

    def __str__(self):
        """Output a human readable list of tasks in the compound task.

        Returns:
            string: brief description of the compound task
        """
        return f"Act [{str(self._action_task)}] and Move"
