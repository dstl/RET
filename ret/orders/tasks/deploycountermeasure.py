"""Deploy countermeasure task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ret.orders.order import Task

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent


class DeployCountermeasureTask(Task):
    """Task for deploying a countermeasure."""

    def __init__(self, log: bool = True):
        """Create a new Deploy Countermeasure Task.

        Args:
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Deploy Countermeasure Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one time steps worth of deploying a countermeasure.

        Args:
            doer (RetAgent): the agent doing the task
        """
        doer.deploy_countermeasure_step()

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Return True if the task is complete, False otherwise.

        Args:
            doer (RetAgent): Agent doing the disabling

        Returns:
            bool: True - Deploying a countermeasure only takes one time step
        """
        return True

    def get_new_instance(self) -> DeployCountermeasureTask:
        """Return a new instance of a functionally identical task.

        Returns:
            DeployCountermeasureTask: New instance of the task
        """
        return DeployCountermeasureTask(log=self._log)
