"""Ground plane clutter modifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.space.clutter.clutter import ClutterModifier

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Optional

    from mesa_ret.agents.agentfilter import AgentFilter
    from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture2d
    from mesa_ret.types import Coordinate2dOr3d


class GroundPlaneClutterModifier(ClutterModifier):
    """Modification to a clutter field that affects up to a height from the terrain.

    If in 2D space then this modifer is assumed to affect all space
    """

    space: ContinuousSpaceWithTerrainAndCulture2d
    height: float

    def __init__(
        self,
        value: float,
        space: ContinuousSpaceWithTerrainAndCulture2d,
        height: float,
        expiry_time: Optional[datetime] = None,
        agent_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Create a ground plane modifier.

        Args:
            value (float): The value of the modifier, will be summed in the clutter
                field
            space (ContinuousSpaceWithTerrainAndCulture2d): The space containing the
                terrain
            height (float): The height above the terrain that this modifier affects
            expiry_time (datetime, optional): The time that this modifer will stop being
                active, if None then it does not expire. Defaults to None.
            agent_filter (AgentFilter): An optional filter which determines which agents
                the clutter modifier applies to. Defaults to None.

        Raises:
            ValueError: Height not positive
        """
        if height < 0:
            raise ValueError("Height must be positive")

        super().__init__(value, expiry_time=expiry_time, agent_filter=agent_filter)
        self.space = space
        self.height = height

    def _is_affected(self, pos: Coordinate2dOr3d) -> bool:
        """Check if a given location is lower than height above the terrain.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if lower than height above the terrain, False otherwise
        """
        if len(pos) == 2:
            return True
        else:
            terrain_height = self.space.get_terrain_height(self.space.get_coordinate_2d(pos))
            # Mypy is not sopisticated enough to realise that a position here must
            # have three points, and therefore raises a warning that it is out of range
            return pos[2] <= terrain_height + self.height  # type: ignore
