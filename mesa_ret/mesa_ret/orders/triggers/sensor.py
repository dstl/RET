"""Sensor based triggers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.agents.agent import AgentSensedStatus
from mesa_ret.orders.order import Trigger

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent


class IlluminatedBySensorTrigger(Trigger):
    """Illuminated by sensor trigger."""

    _agent: RetAgent

    def __init__(self, agent: RetAgent, sticky: bool = False, log: bool = True):
        """Create a new illuminated by sensor Trigger.

        Args:
            agent (RetAgent): The agent that needs to be illuminated.
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log=log, sticky=sticky)
        self._agent = agent

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Illuminated by Sensor Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if a sensed event occurred during the last time step.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if a sensed event occurred during the last time step
        """
        recent_sensed_events = [
            event for event in self._agent.get_statuses() if isinstance(event, AgentSensedStatus)
        ]

        return len(recent_sensed_events) > 0

    def get_new_instance(self) -> IlluminatedBySensorTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            IlluminatedBySensorTrigger: New instance of the trigger
        """
        return IlluminatedBySensorTrigger(agent=self._agent, sticky=self._sticky, log=self._log)
