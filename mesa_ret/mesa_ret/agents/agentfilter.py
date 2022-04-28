"""Agent filters."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation

if TYPE_CHECKING:
    from typing import Union

    from mesa_ret.agents.agent import RetAgent


class AgentFilter(ABC):
    """Abstract class representing an Agent Filter."""

    def run(self, agents: list[RetAgent]) -> list[RetAgent]:  # pragma: no cover
        """Filter a list of agents.

        Args:
            agents (list[RetAgent]): Initial list of agents

        Returns:
            list[RetAgent]: Reduced list of agents
        """
        return NotImplemented


class FilterByAffiliation(AgentFilter):
    """AgentFilter for agent affiliation."""

    def __init__(self, affiliation: Affiliation) -> None:
        """Create a new FilterByAffiliation AgentFilter.

        Args:
            affiliation (Affiliation): Affiliation to filter by
        """
        self._affiliation = affiliation

    def run(self, agents: list[RetAgent]) -> list[RetAgent]:
        """Filter list of agents by affiliation.

        Args:
            agents (list[RetAgent]): Initial list of agents

        Returns:
            list[RetAgent]: Reduced list of agents
        """
        return [a for a in agents if a.affiliation == self._affiliation]


class FilterFriendly(FilterByAffiliation):
    """Affiliation-based agent-filter for friendly agents."""

    def __init__(self):
        """Create a new FilterFriendly AgentFilter."""
        super().__init__(Affiliation.FRIENDLY)


class FilterHostile(FilterByAffiliation):
    """Affiliation-based agent-filter for hostile agents."""

    def __init__(self):
        """Create a new FilterHostile AgentFilter."""
        super().__init__(Affiliation.HOSTILE)


class FilterUnknown(FilterByAffiliation):
    """Affiliation-based agent-filter for unknown agents."""

    def __init__(self):
        """Create a new FilterUnknown AgentFilter."""
        super().__init__(Affiliation.UNKNOWN)


class FilterNeutral(FilterByAffiliation):
    """Affiliation-based agent-filter for neutral agents."""

    def __init__(self):
        """Create a new FilterNeutral AgentFilter."""
        super().__init__(Affiliation.NEUTRAL)


class FilterByID(AgentFilter):
    """Agent filter for Agent ID."""

    def __init__(self, id: Union[int, list[int]]) -> None:
        """Create a new FilterByID AgentFilter.

        Args:
            id (int, list[int]): Id(s) to include in the filter.
        """
        if isinstance(id, Iterable):
            self._ids = [i for i in id]
        else:
            self._ids = [id]

    def run(self, agents: list[RetAgent]) -> list[RetAgent]:
        """Filter agents by ID.

        Args:
            agents (list[RetAgent]): Unfiltered agents

        Returns:
            list[RetAgent]: Filtered agents
        """
        return [a for a in agents if a.unique_id in self._ids]


class FilterNot(AgentFilter):
    """Agent filter which inverts given filter."""

    def __init__(self, agent_filter_to_invert: AgentFilter) -> None:
        """Create a new FilterNot AgentFilter.

        Args:
            agent_filter_to_invert (AgentFilter): The filter to invert.
        """
        self._agent_filter_to_invert = agent_filter_to_invert

    def run(self, agents: list[RetAgent]) -> list[RetAgent]:
        """Filter a list of agents.

        Args:
            agents (list[RetAgent]): Initial list of agents

        Returns:
            list[RetAgent]: Reduced list of agents
        """
        filtered_agents = self._agent_filter_to_invert.run(agents)
        return [agent for agent in agents if agent not in filtered_agents]


class FilterNotFriendly(FilterNot):
    """Affiliation-based agent-filter for non-friendly agents."""

    def __init__(self) -> None:
        """Create a new FilterNotFriendly AgentFilter."""
        super().__init__(FilterFriendly())


class FilterNotHostile(FilterNot):
    """Affiliation-based agent-filter for non-hostile agents."""

    def __init__(self) -> None:
        """Create a new FilterNotHostile AgentFilter."""
        super().__init__(FilterHostile())


class FilterNotUnknown(FilterNot):
    """Affiliation-based agent-filter for non-unknown agents."""

    def __init__(self) -> None:
        """Create a new FilterNotUnknown AgentFilter."""
        super().__init__(FilterUnknown())


class FilterNotNeutral(FilterNot):
    """Affiliation-based agent-filter for non-neutral agents."""

    def __init__(self) -> None:
        """Create a new FilterNotNeutral AgentFilter."""
        super().__init__(FilterNeutral())


class FilterNotID(FilterNot):
    """Agent filter for Agents which don't match ID."""

    def __init__(self, id: Union[int, list[int]]) -> None:
        """Create a new FilterNotID AgentFilter.

        Args:
            id (int, list[int]): Id(s) to include in the filter.
        """
        super().__init__(FilterByID(id))
