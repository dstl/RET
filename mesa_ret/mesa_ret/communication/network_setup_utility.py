"""Utility for setting up communication networks."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent


def network_setup_complete_graph(agents: list[RetAgent]) -> None:
    """Add all agents to each other's comminication network.

    Args:
        agents (list[RetAgent]): The agents to be added to each other's communication network.
    """
    for sending_agent in agents:
        recipients = agents.copy()
        recipients.remove(sending_agent)
        sending_agent.communication_network.add_recipient(recipients)
