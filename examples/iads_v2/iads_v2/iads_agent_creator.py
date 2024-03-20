"""Utilities to generate hostile agents for IADS scenario."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent
    from ret.model import RetModel


class IADSAgentCreator(ABC):
    """Utility class for making a force of agents with a functioning comms network."""

    def __init__(self, iads_model: RetModel) -> None:
        """Initialise IADS agent creator."""
        self.model = iads_model

    def create_agents(self) -> list[RetAgent]:
        """Create list of agents with comms networks set up."""
        agents = self._make_agents()

        self._set_up_comms(agents)

        return agents

    @abstractmethod
    def _make_agents(self) -> list[RetAgent]:
        """Make all agents required."""
        return []

    @abstractmethod
    def _set_up_comms(self, agents: list[RetAgent]) -> None:
        """Set up comms network for supplied agents."""
        pass
