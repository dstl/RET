"""Area-based clutter field modifiers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ret.space.clutter.clutter import ClutterModifier

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Optional

    from ret.agents.agentfilter import AgentFilter
    from ret.space.feature import Area
    from ret.types import Coordinate2dOr3d


class AreaClutterModifier(ClutterModifier):
    """A class representing an area modification to a clutter field."""

    area: Area

    def __init__(
        self,
        value: float,
        area: Area,
        expiry_time: Optional[datetime] = None,
        agent_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Create an area clutter modifier.

        Args:
            value (float): The value of the modifier, will be summed in the clutter
                field
            area (Area): the area the modifier affects
            expiry_time (Optional[datetime]): The time that this modifier will stop being
                 active, if None then it does not expire. Defaults to None.
            agent_filter (Optional[AgentFilter]): An optional filter which determines
                which agents the clutter modifier applies to. Defaults to None.
        """
        super().__init__(value, expiry_time=expiry_time, agent_filter=agent_filter)
        self.area = area

    def _is_affected(self, pos: Coordinate2dOr3d) -> bool:
        """Check if a given location is inside the area.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if inside the area, False otherwise
        """
        return self.area.contains(pos)
