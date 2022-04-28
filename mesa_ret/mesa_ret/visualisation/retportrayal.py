"""Default portrayal for ret agents."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional

    from mesa_ret.agents.agent import RetAgent


def ret_portrayal(agent: RetAgent) -> Optional[dict[str, Any]]:
    """Create an agent portrayal for visualisation.

    Agents in a group are not displayed.

    Args:
        agent (RetAgent): The agent to be portrayed.

    Returns:
        portrayal: The portrayal of the agent.
    """
    if agent is None:
        return None

    if agent.in_group is True:
        return None

    portrayal: dict[str, Any] = {
        "Shape": "svg",
        "svgSource": agent.icon,
        "Name": agent.name,
    }

    portrayal["Coord"] = str([round(coord, 2) for coord in agent.pos])

    portrayal["Layer"] = 0

    return portrayal
