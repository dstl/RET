"""Agent is killed trigger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.orders.order import Trigger
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.sensing.perceivedworld import AgentById, AgentsAt

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.types import Coordinate2dOr3d


class KilledAgentsAtPositionTrigger(Trigger):
    """Trigger for any killed agents in the checker's perceived world at given position."""

    def __init__(
        self,
        position: Coordinate2dOr3d,
        log: bool = True,
        sticky: bool = True,
        invert: bool = False,
    ):
        """Initialise a killed agents at position trigger.

        Args:
            position (Coordinate2dOr3d): position at which you are checking for killed
                agents.
            log (bool): whether to log or not. Defaults to True.
            sticky (bool): if true once activated this trigger will remain
                activated. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log, sticky, invert)
        self._position = position

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Killed Agents at Position Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Return true if perceived agent with killed casualty state found at position.

        Args:
            checker (RetAgent): Agent doing the checking.

        Returns:
            bool: Whether or not killed perceived agent has been found.
        """
        agent = [
            a
            for a in checker.perceived_world.get_perceived_agents(
                AgentsAt(
                    distance_calculator=checker.perceived_world.get_distance,
                    location=self._position,
                )
            )
            if a.casualty_state == AgentCasualtyState.KILLED
        ]

        return len(agent) > 0

    def get_new_instance(self) -> KilledAgentsAtPositionTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            KilledAgentsAtPositionTrigger: New instance of the trigger
        """
        return KilledAgentsAtPositionTrigger(
            position=self._position, log=self._log, sticky=self._sticky, invert=self._invert
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return f"Killed agents at position trigger target: {self._position}"


class AgentKilledTrigger(Trigger):
    """Trigger for when given agent is killed."""

    _target_agent: RetAgent

    def __init__(
        self,
        agent: RetAgent,
        log: bool = True,
        sticky: bool = True,
        invert: bool = False,
    ):
        """Initialise an agent killed trigger.

        Args:
            agent (RetAgent): The agent to check if killed.
            log (bool): whether to log or not. Defaults to True.
            sticky (bool): if true once activated this trigger will remain
                activated. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log, sticky, invert)
        self._target_agent = agent

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Agent Killed Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Return true if perceived target agent has killed casualty state.

        Args:
            checker (RetAgent): Agent doing the checking.

        Returns:
            bool: Whether or not target agent has been perceived as killed.
        """
        agents = checker.perceived_world.get_perceived_agents(
            AgentById(self._target_agent.unique_id)
        )
        if len(agents) == 1:
            if agents[0].casualty_state == AgentCasualtyState.KILLED:
                return True
        elif len(agents) > 0:
            raise ValueError(
                f"Multiple perceived agents found with id: {self._target_agent.unique_id}"
            )

        return False

    def get_new_instance(self) -> AgentKilledTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            AgentKilledTrigger: New instance of the trigger
        """
        return AgentKilledTrigger(
            agent=self._target_agent, log=self._log, sticky=self._sticky, invert=self._invert
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return (
            f"Agent killed trigger. Agent: {self._target_agent.agent_type} "
            f"(Agent ID: {self._target_agent.unique_id})."
        )
