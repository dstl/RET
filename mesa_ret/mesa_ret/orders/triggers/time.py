"""Time based triggers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.orders.order import Trigger

if TYPE_CHECKING:
    from datetime import datetime

    from mesa_ret.agents.agent import RetAgent


class TimeTrigger(Trigger):
    """Trigger that activates when a certain time has been reached in the simulation."""

    _time: datetime

    def __init__(
        self, time: datetime, sticky: bool = False, log: bool = True, invert: bool = False
    ) -> None:
        """Create trigger.

            Note: if the trigger time doesn't fall on an integer time step,
                this trigger will be active from the preceding integer time step.

        Args:
            time (datetime): the time that needs to be reached for this trigger to
                activate
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log=log, sticky=sticky, invert=invert)
        self._time = time

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Time Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if the given time occurred during the last time step.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if the given time occurred during the last time step
        """
        current_time = checker.model.get_time()
        time_step = checker.model.time_step

        return current_time <= self._time < (current_time + time_step)

    def get_new_instance(self) -> TimeTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            TimeTrigger: New instance of the trigger
        """
        return TimeTrigger(time=self._time, sticky=self._sticky, log=self._log, invert=self._invert)

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return f"Time: {self._time}"
