"""Sense task."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from mesa_ret.orders.order import Task

if TYPE_CHECKING:
    from datetime import timedelta
    from typing import Optional, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.behaviours.sense import SenseSchedule
    from mesa_ret.types import Coordinate2dOr3d


class SenseTask(Task):
    """Task for sensing."""

    def __init__(
        self,
        duration: timedelta,
        direction: Optional[Union[float, Coordinate2dOr3d, str]] = None,
        log: bool = True,
    ):
        """Create a new Sense Task.

        Args:
            duration (timedelta): The duration of the task, over which the agent
                periodically checks its sensors
            direction (Optional[float | Coordinate2dor3d | str]): Optional direction to sense
                in, can be input as float: degrees clockwise from y-axis, Coordinate2dor3d: a
                position used to calculate heading from sensor agent, str: the name of an Area in
                the space, a random point within the area will be chosen to look towards.
            log (bool): whether to log or not. Defaults to True.
        """
        self._direction = direction
        super().__init__(log)
        self.duration = duration
        self.sense_schedule: Optional[SenseSchedule] = None

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Sense Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one time steps worth of sensing.

        Args:
            doer (RetAgent): the agent doing the task
        """
        if self.sense_schedule is None:
            self.sense_schedule = doer.get_sense_schedule(self.duration)

        if self.sense_schedule.is_complete():
            warnings.warn(f"{str(self)} is already complete.")
            doer.clear_sense_behaviour()
        elif self.sense_schedule.get_step():
            doer.sense_step(sense_direction=self._direction)

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Return True if the task is complete, False otherwise.

        Args:
            doer (RetAgent): Agent doing the disabling

        Returns:
            bool: True - Sensing only takes one time step
        """
        if self.sense_schedule is None:
            return False
        return self.sense_schedule.is_complete()

    def get_new_instance(self) -> SenseTask:
        """Return a new instance of a functionally identical task.

        Returns:
            SenseTask: New instance of the task
        """
        return SenseTask(duration=self.duration, direction=self._direction, log=self._log)
