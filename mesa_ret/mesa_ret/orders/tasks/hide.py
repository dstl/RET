"""Hide task."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from mesa_ret.orders.order import Task

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from typing import Optional

    from mesa_ret.agents.agent import RetAgent


class HideTask(Task):
    """Task for an agent to hide for a specified duration."""

    _duration: timedelta

    _hide_end_time: Optional[datetime] = None

    def __init__(self, duration: timedelta, log: bool = True) -> None:
        """Create task.

        Args:
            duration (timedelta): duration for the agent to wait
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log)
        self._duration = duration

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Hide Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Hide for one time step.

        Args:
            doer (RetAgent): the agent hiding
        """
        if self._hide_end_time is None:
            self._hide_end_time = doer.model.get_time() + self._duration
        doer.hide_step()

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Check whether the hide task has been completed in the last time step.

        Args:
            doer (RetAgent): the agent hiding

        Returns:
            bool: true when the agent has hidden, false otherwise
        """
        current_time = doer.model.get_time()
        time_step = doer.model.time_step

        if self._hide_end_time is None:
            warnings.warn("Hide end time was not set")
            return False

        if (self._hide_end_time <= (current_time)) and (self._duration > time_step):
            warnings.warn("Task was completed before this time step")

        return self._hide_end_time <= (current_time + time_step)

    def get_new_instance(self) -> HideTask:
        """Return a new instance of a functionally identical task.

        Returns:
            HideTask: New instance of the task
        """
        return HideTask(duration=self._duration, log=self._log)
