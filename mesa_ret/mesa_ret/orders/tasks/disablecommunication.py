"""Disable communication task."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent

from mesa_ret.orders.order import Task


class DisableCommunicationTask(Task):
    """Task for disabling comms from an asset."""

    def __init__(self, log: bool = True):
        """Create a new Disable Communication Task.

        Args:
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Disable Communication Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one time step's worth of disabling communications.

        Args:
            doer (RetAgent): Agent doing the disabling
        """
        doer.disable_communication_step()

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Return True if the task is complete, False otherwise.

        Args:
            doer (RetAgent): Agent doing the disabling

        Returns:
            bool: True - Disabling comms only takes one time step
        """
        return True

    def get_new_instance(self) -> DisableCommunicationTask:
        """Return a new instance of a functionally identical task.

        Returns:
            DisableCommunicationTask: New instance of the task
        """
        return DisableCommunicationTask(log=self._log)
