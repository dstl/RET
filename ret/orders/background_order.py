"""Agent background order."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from ret.orders.order import Order
from ret.orders.triggers.immediate import ImmediateTrigger

if TYPE_CHECKING:
    from datetime import timedelta

    from ret.agents.agent import RetAgent
    from ret.orders.order import Task, Trigger
    from ret.template import Template


class BackgroundOrder(Order):
    """Order to be completed in the background.

    Carried out simultaneously to active orders.
    If no trigger is specified, the Trigger does not
    log and is always true. The order is persistent.
    """

    def __init__(
        self, time_period: timedelta, task: Template[Task], trigger: Template[Trigger] = None
    ) -> None:
        """Create a background order.

        Args:
            time_period (timedelta): time interval between steps of background order
            task (Template[Task]): template for the task to be completed
            trigger (Optional[Template[Trigger]]): template for the trigger for the condition(s)
                that must met to activate this order. If None or unspecified, defaults to
                ImmediateTrigger
        """
        if trigger is None:
            trigger = ImmediateTrigger(log=False)

        super().__init__(
            trigger=trigger,
            task=task.get_new_instance(),
            persistent=True,
        )

        self._time_period = time_period
        self._last_execution: datetime = datetime.min

    def execute_step(self, doer: RetAgent) -> None:
        """Execute a time step of the task.

        Record time at which last step was executed.

        Args:
            doer (RetAgent): agent doing the task
        """
        if self._check_execution_timing_validity(doer):
            super().execute_step(doer=doer)
            self._last_execution = doer.model.get_time()

    def get_new_instance(self) -> BackgroundOrder:
        """Return a new instance of a functionally identical background order.

        Returns:
            BackgroundOrder: New instance of the background order
        """
        return BackgroundOrder(
            time_period=self._time_period, task=self._task, trigger=self._trigger
        )

    def _check_execution_timing_validity(self, doer: RetAgent) -> bool:
        """Check whether it is time for the task to be executed.

        Args:
            doer (RetAgent): agent doing the task

        Returns:
            bool: true if it is time for the task to be executed
        """
        time_since_last_execution = doer.model.get_time() - self._last_execution

        return time_since_last_execution >= self._time_period
