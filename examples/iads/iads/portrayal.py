"""IADS portrayal."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.sensing.perceivedworld import (
    And,
    DetectedAgents,
    HostileAgents,
    IdentifiedAgents,
    RecognisedAgents,
)
from mesa_ret.visualisation.retportrayal import ret_portrayal

if TYPE_CHECKING:
    from typing import Any, Optional, Union

    from mesa_ret.agents.agent import RetAgent


def unit_portrayal(agent: RetAgent) -> Optional[dict[str, Any]]:
    """IADS unit portrayal.

    Args:
        agent (RetAgent): The agent to produce the portrayal for

    Returns:
        Optional[dict[str, Any]]: The portrayal
    """
    portrayal: Optional[dict[str, Any]] = ret_portrayal(agent)

    if portrayal is not None:

        if agent._orders is None:
            orders: Union[str, list[str]] = "None"
        else:
            orders = [str(o) for o in agent._orders]
        portrayal["Orders"] = orders

        portrayal["Killed"] = agent.killed

        portrayal["Detected"] = len(
            agent.perceived_world.get_perceived_agents(And([DetectedAgents(), HostileAgents()]))
        )
        portrayal["Recognised"] = len(
            agent.perceived_world.get_perceived_agents(And([RecognisedAgents(), HostileAgents()]))
        )
        portrayal["Identified Agents"] = len(
            agent.perceived_world.get_perceived_agents(And([IdentifiedAgents(), HostileAgents()]))
        )
        portrayal["Active Order"] = str(agent.active_order)

    return portrayal
