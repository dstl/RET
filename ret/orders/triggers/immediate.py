"""Immediate triggers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent
    from ret.agents.sensorfusionagent import SensorFusionAgent

from ret.orders.order import Trigger


class ImmediateTrigger(Trigger):
    """Immediate trigger."""

    def __init__(self, log: bool = True):
        """Create a new Immediate Trigger.

        Args:
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log=log, sticky=False)

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Immediate Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Return true at all times.

        Args:
            checker (RetAgent): Agent for the trigger

        Returns:
            bool: Always True
        """
        return True

    def get_new_instance(self) -> ImmediateTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            ImmediateTrigger: New instance of the trigger
        """
        return ImmediateTrigger(log=self._log)


class ImmediateSensorFusionTrigger(Trigger):
    """Immediate trigger for sensor fusion agent."""

    def __init__(self, log: bool = True):
        """Create a new ImmediateSensorFusionTrigger.

        Args:
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log=log, sticky=False)

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Immediate Sensor Fusion Trigger"

    def _check_condition(self, checker: SensorFusionAgent) -> bool:
        """Check whether the agent is triggered.

        The agent is triggered as soon as it's new_info property is set to true

        Args:
            checker (SensorFusionAgent): Agent for the trigger

        Returns:
            bool: Whether or not the agent has triggered
        """
        return checker.new_info

    def get_new_instance(self) -> ImmediateSensorFusionTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            ImmediateSensorFusionTrigger: New instance of the trigger
        """
        return ImmediateSensorFusionTrigger(log=self._log)
