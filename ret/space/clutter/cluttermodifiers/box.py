"""Box-based clutter field modifiers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ret.space.clutter.cluttermodifiers.area import AreaClutterModifier
from ret.space.feature import BoxFeature

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Optional

    from ret.agents.agentfilter import AgentFilter
    from ret.types import Coordinate2dOr3d


class BoxClutterModifier(AreaClutterModifier):
    """A class representing a box modification to a clutter field."""

    def __init__(
        self,
        value: float,
        min_coord: Coordinate2dOr3d,
        max_coord: Coordinate2dOr3d,
        expiry_time: Optional[datetime] = None,
        agent_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Create a box clutter modifier.

        Args:
            value (float): The value of the modifier, will be summed in the clutter
                field
            min_coord (Coordinate2dOr3d): Coordinate of the corner with lowest x, y, and
                optional z values
            max_coord (Coordinate2dOr3d): Coordinate of the corner with highest x, y and
                optional z values
            expiry_time (Optional[datetime]): The time that this modifier will stop being
                 active, if None then it does not expire. Defaults to None.
            agent_filter (Optional[AgentFilter]): An optional filter which determines
                which agents the clutter modifier applies to. Defaults to None.
        """
        super().__init__(
            value,
            area=BoxFeature(min_coord, max_coord, "Box Clutter Modifier"),
            expiry_time=expiry_time,
            agent_filter=agent_filter,
        )
