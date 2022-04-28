"""Operate task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.orders.order import Task

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent


class OperateTask(Task):
    """Task for an agent to operate."""

    def __init__(self, default_task: Task, log: bool = True) -> None:
        """Create task.

        Args:
            default_task (Task): Default task to perform while operating.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)
        self._default_task = default_task

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Operate Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Operate for one time step.

        Args:
            doer (RetAgent): the agent operating
        """
        self._default_task._do_task_step(doer)

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Check whether the operate task has been completed in the last time step.

        Args:
            doer (RetAgent): the agent operating

        Returns:
            bool: true when the agent has completed its operation, false otherwise
        """
        return self._default_task._is_task_complete(doer)

    def get_new_instance(self) -> OperateTask:
        """Return a new instance of a functionally identical task.

        Returns:
            OperateTask: New instance of the task
        """
        return OperateTask(default_task=self._default_task, log=self._log)
