"""Clutter field management and modifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Optional

    from ret.agents.agent import RetAgent
    from ret.agents.agentfilter import AgentFilter
    from ret.types import Coordinate2dOr3d


class ClutterModifier(ABC):
    """An abstract class representing a modification to a clutter field."""

    _value: float
    expiry_time: Optional[datetime]
    agent_filter: Optional[AgentFilter]

    def __init__(
        self,
        value: float,
        expiry_time: Optional[datetime] = None,
        agent_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Create a clutter modifier.

        Args:
            value (float): The value of the modifier, will be summed in the clutter
                field
            expiry_time (Optional[datetime]): The time that this modifier will stop being
                active, if None then it does not expire. Defaults to None.
            agent_filter (Optional[AgentFilter]): An optional filter which determines
                which agents the clutter modifier applies to. Defaults to None.
        """
        self._value = value
        self.expiry_time = expiry_time
        self.agent_filter = agent_filter

    def get_value(self, pos: Coordinate2dOr3d, agent: RetAgent) -> float:
        """Return the value of this modifier at the given location.

        Args:
            pos (Coordinate2dOr3d): The location
            agent (RetAgent): The agent that the clutter value is requested for

        Returns:
            float: The value at the given location
        """
        if self._is_agent_affected_by_clutter(agent) and self._is_affected(pos):
            return self._value
        else:
            return 0

    def _is_agent_affected_by_clutter(self, agent: RetAgent) -> bool:
        """Check whether the clutter applies to the given agent.

        Args:
            agent (RetAgent): The agent to check for

        Returns:
            bool: Whether the clutter applies to the given agent
        """
        return self.agent_filter is None or agent in self.agent_filter.run([agent])

    @abstractmethod
    def _is_affected(self, pos: Coordinate2dOr3d) -> bool:  # pragma: no cover
        """Check if a given location is affected by this modifier.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if in affected area, False otherwise
        """
        pass


class ClutterField:
    """Class representing a clutter field in a space."""

    _background_level: float
    _modifiers: list[ClutterModifier]

    def __init__(
        self,
        background_level: float = 0,
        modifiers: Optional[list[ClutterModifier]] = None,
    ) -> None:
        """Create a clutter field.

        Args:
            background_level (float): The background clutter level, this
                exists everywhere. Defaults to 0.
            modifiers (Optional[list[ClutterModifier]]): A list of clutter modifiers,
                representing any modifications to the clutter field. Defaults to None.
        """
        self._background_level = background_level
        if modifiers is None:
            self._modifiers = []
        else:
            self._modifiers = modifiers

    def get_value(self, pos: Coordinate2dOr3d, agent: RetAgent) -> float:
        """Return the clutter value at a given location.

        Args:
            pos (Coordinate2dOr3d): The location
            agent (RetAgent): The agent that the clutter field is requested for

        Returns:
            float: The clutter value at the given location
        """
        return self._background_level + sum([m.get_value(pos, agent) for m in self._modifiers])

    def modify(self, modifier: ClutterModifier) -> None:
        """Add a modifier to the clutter field.

        Args:
            modifier (ClutterModifier): The clutter modifier to add
        """
        self._modifiers.append(modifier)

    def remove_expired_modifiers(self, current_time: datetime) -> None:
        """Remove all expired modifiers from the clutter field.

        Args:
            current_time (datetime): The current time
        """
        self._modifiers = [
            m for m in self._modifiers if (m.expiry_time is None) or (m.expiry_time > current_time)
        ]
