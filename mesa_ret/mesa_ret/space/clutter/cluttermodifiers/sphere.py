"""Spherical clutter modifiers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.space.clutter.cluttermodifiers.area import AreaClutterModifier
from mesa_ret.space.feature import SphereFeature

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Optional

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.agents.agentfilter import AgentFilter
    from mesa_ret.types import Coordinate2dOr3d


class SphereClutterModifier(AreaClutterModifier):
    """A class representing a spherical modification to a clutter field."""

    def __init__(
        self,
        value: float,
        center: Coordinate2dOr3d,
        radius: float,
        expiry_time: Optional[datetime] = None,
        agent_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Create a spherical clutter modifier.

        Args:
            value (float): The value of the modifier, will be summed in the clutter
                field
            center (Coordinate2dOr3d): The center of the sphere
            radius (float): The radius of the sphere
            expiry_time (Optional[datetime]): The time that this modifer will stop being
                active, if None then it does not expire. Defaults to None.
            agent_filter (Optional[AgentFilter]): An optional filter which determines
                which agents the clutter modifier applies to. Defaults to None.
        """
        super().__init__(
            value,
            area=SphereFeature(center, radius, "Sphere Clutter Modifier"),
            expiry_time=expiry_time,
            agent_filter=agent_filter,
        )


class SphereFollowerClutterModifier(SphereClutterModifier):
    """A spherical modification to a clutter field that follows an agent."""

    agent: RetAgent

    def __init__(
        self,
        value: float,
        agent: RetAgent,
        radius: float,
        expiry_time: Optional[datetime] = None,
        agent_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Create a spherical follower clutter modifier.

        Args:
            value (float): The value of the modifier, will be summed in the clutter
                field
            agent (RetAgent): The agent at the center of the sphere
            radius (float): The radius of the sphere
            expiry_time (Optional[datetime]): The time that this modifer will stop being
                active, if None then it does not expire. Defaults to None.
            agent_filter (Optional[AgentFilter]): Filter to determine which agents are
                affected. Defaults to no filter.
        """
        super().__init__(
            value, agent.pos, radius, expiry_time=expiry_time, agent_filter=agent_filter
        )
        self.agent = agent

    def _is_affected(self, pos: Coordinate2dOr3d) -> bool:
        """Check if a given location is inside the sphere.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if inside the sphere, False otherwise
        """
        self.area: SphereFeature
        self.area.center = self.agent.pos
        return super()._is_affected(pos)
