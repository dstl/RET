"""IADS specific triggers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.orders.order import Trigger
from mesa_ret.sensing.perceivedworld import (
    AgentsAt,
    And,
    FriendlyAgents,
    HostileAgents,
    IdentifiedAgents,
    UnknownAgents,
)

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent


class IdentifiedHostileAgentsAtPositionTrigger(Trigger):
    """Identified hostile agents at position trigger."""

    def __init__(self, position, log=True, sticky=False):
        """Initialise trigger."""
        super().__init__(log, sticky)
        self.position = position

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Identified Hostile Agents at Position Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for identified hostile agents at position."""
        perceived_agents = checker.perceived_world.get_perceived_agents(
            And(
                [
                    IdentifiedAgents(),
                    AgentsAt(
                        distance_calculator=checker.perceived_world.get_distance,
                        location=self.position,
                    ),
                    HostileAgents(),
                ]
            )
        )
        return len(perceived_agents) > 0

    def get_new_instance(self) -> IdentifiedHostileAgentsAtPositionTrigger:
        """Get new instance of Trigger."""
        return IdentifiedHostileAgentsAtPositionTrigger(
            position=self.position, log=self._log, sticky=self._sticky
        )


class HostileAgentsAtPositionTrigger(Trigger):
    """Hostile agents at position trigger."""

    def __init__(self, position, log=True, sticky=False):
        """Initialise trigger."""
        super().__init__(log, sticky)
        self.position = position

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Hostile Agents at Position Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for hostile agents at position."""
        perceived_agents = checker.perceived_world.get_perceived_agents(
            And(
                [
                    AgentsAt(
                        distance_calculator=checker.perceived_world.get_distance,
                        location=self.position,
                    ),
                    HostileAgents(),
                ]
            )
        )
        return len(perceived_agents) > 0

    def get_new_instance(self) -> HostileAgentsAtPositionTrigger:
        """Get new instance of Trigger."""
        return HostileAgentsAtPositionTrigger(
            position=self.position, log=self._log, sticky=self._sticky
        )


class UnknownAgentsTrigger(Trigger):
    """Unknown agents trigger."""

    def __init__(self, log=True, sticky=False):
        """Initialise trigger."""
        super().__init__(log, sticky)

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Unknown Agents Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for unknown agents."""
        unknown_agents = checker.perceived_world.get_perceived_agents(UnknownAgents())
        return len(unknown_agents) > 0

    def get_new_instance(self) -> UnknownAgentsTrigger:
        """Get new instance of Trigger."""
        return UnknownAgentsTrigger(log=self._log, sticky=self._sticky)


class FriendlyIdentifiedAgentsTrigger(Trigger):
    """Friendly identified agents trigger."""

    def __init__(self, log=True, sticky=False):
        """Initialise trigger."""
        super().__init__(log, sticky)

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Friendly Identified Agents Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for identified friendly agents."""
        friendly_agents = checker.perceived_world.get_perceived_agents(
            And([FriendlyAgents(), IdentifiedAgents()])
        )
        return len(friendly_agents) > 0

    def get_new_instance(self) -> FriendlyIdentifiedAgentsTrigger:
        """Get new instance of Trigger."""
        return FriendlyIdentifiedAgentsTrigger(log=self._log, sticky=self._sticky)


class HostileIdentifiedAgentsTrigger(Trigger):
    """Hostile identified agents trigger."""

    def __init__(self, log=True, sticky=False):
        """Initialise trigger."""
        super().__init__(log, sticky)

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Hostile Identified Agents Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for identified hostile agents."""
        hostile_agents = checker.perceived_world.get_perceived_agents(
            And([HostileAgents(), IdentifiedAgents()])
        )
        return len(hostile_agents) > 0

    def get_new_instance(self) -> HostileIdentifiedAgentsTrigger:
        """Get new instance of Trigger."""
        return HostileIdentifiedAgentsTrigger(log=self._log, sticky=self._sticky)
