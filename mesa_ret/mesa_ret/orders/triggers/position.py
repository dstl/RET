"""Position based order triggers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from mesa_ret.orders.order import Trigger
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.sensing.perceivedworld import AgentById, AgentNotInPerceivedWorldWarning, Not

if TYPE_CHECKING:
    from typing import Optional

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.space.feature import Area, Boundary
    from mesa_ret.types import Coordinate2dOr3d


class PositionTrigger(Trigger):
    """Trigger that activates when a given agent has reached a certain location.

    Position is defined with a tolerance. The position of an agent is taken from the checker's
    perceived world. If the target agent isn't in the checker's perceived world then the trigger
    will return False, even if inverted.
    """

    _agent: RetAgent
    _position: Coordinate2dOr3d
    _tolerance: float

    def __init__(
        self,
        agent: RetAgent,
        position: Coordinate2dOr3d,
        tolerance: float,
        sticky: bool = False,
        log: bool = True,
        invert: bool = False,
    ) -> None:
        """Create trigger.

        Args:
            agent (RetAgent): the agent that needs to be in the given location
            position (Coordinate2dOr3d): the position the agent needs to be
            tolerance (float): the tolerance on the agent position
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log=log, sticky=sticky, invert=invert)
        self._agent = agent
        self._position = position
        self._tolerance = tolerance

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Position Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if the given agent is in the given location.

        The position of the agent will be based on the observing agent's perception. If
        The observing agent is unaware of it's own position, or has an incorrect
        perception of it's position, this will be accounted for.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if the agent is within the tolerance of the given position
        """
        with warnings.catch_warnings():  # suppress warning if agent is not in perceived world
            warnings.simplefilter("ignore", category=AgentNotInPerceivedWorldWarning)
            agent_pos = checker.perceived_world.get_agent_pos(self._agent.unique_id)
        if agent_pos is None:
            return self._invert
        distance = checker.perceived_world.get_distance(agent_pos, self._position)
        return distance <= self._tolerance

    def get_new_instance(self) -> PositionTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            PositionTrigger: New instance of the trigger
        """
        return PositionTrigger(
            agent=self._agent,
            position=self._position,
            tolerance=self._tolerance,
            sticky=self._sticky,
            log=self._log,
            invert=self._invert,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return (
            f"Agent: {self._agent.name} ({self._agent.unique_id}), "
            f"Position: {self._position}, "
            f"Tolerance: {self._tolerance}"
        )


class InAreaTrigger(Trigger):
    """Trigger that activates when a given agent is inside a given area."""

    _agent: RetAgent
    _area: Area

    def __init__(
        self,
        agent: RetAgent,
        area: Area,
        sticky: bool = False,
        log: bool = True,
    ) -> None:
        """Create trigger.

        Args:
            agent (RetAgent): the agent that needs to be in the given location
            area (Area): the area the agent needs to be
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log=log, sticky=sticky)
        self._agent = agent
        self._area = area

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "In Area Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if the given agent is in the given location.

        The position of the agent will be based on the observing agent's perception. If
        The observing agent is unaware of it's own position, or has an incorrect
        perception of it's position, this will be accounted for.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if the agent is inside the area
        """
        with warnings.catch_warnings():  # suppress warning if agent is not in perceived world
            warnings.simplefilter("ignore", category=AgentNotInPerceivedWorldWarning)
            agent_pos = checker.perceived_world.get_agent_pos(self._agent.unique_id)
        if agent_pos is None:
            return False
        return self._area.contains(agent_pos)

    def get_new_instance(self) -> InAreaTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            InAreaTrigger: New instance of the trigger
        """
        return InAreaTrigger(
            agent=self._agent,
            area=self._area,
            sticky=self._sticky,
            log=self._log,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return f"Agent: {self._agent.name} ({self._agent.unique_id}), Area: {self._area.name}"


class NotInAreaTrigger(InAreaTrigger):
    """Trigger that activates when a given agent is outside a given area."""

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Not In Area Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if the given agent is not in the given location.

        The position of the agent will be based on the observing agent's perception. If
        The observing agent is unaware of it's own position, or has an incorrect
        perception of it's position, this will be accounted for.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if the agent is not inside the area
        """
        with warnings.catch_warnings():  # suppress warning if agent is not in perceived world
            warnings.simplefilter("ignore", category=AgentNotInPerceivedWorldWarning)
            agent_pos = checker.perceived_world.get_agent_pos(self._agent.unique_id)
        if agent_pos is None:
            return False
        return not self._area.contains(agent_pos)

    def get_new_instance(self) -> NotInAreaTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            NotInAreaTrigger: New instance of the trigger
        """
        return NotInAreaTrigger(
            agent=self._agent,
            area=self._area,
            sticky=self._sticky,
            log=self._log,
        )


class CrossedBoundaryTrigger(Trigger):
    """Trigger that activates when a given agent has crossed a boundary."""

    _agent: RetAgent
    _boundary: Boundary
    _previous_position: Optional[Coordinate2dOr3d] = None

    def __init__(
        self,
        agent: RetAgent,
        boundary: Boundary,
        sticky: bool = False,
        log: bool = True,
        invert: bool = False,
    ) -> None:
        """Create trigger.

        Args:
            agent (RetAgent): the agent that needs to cross the boundary
            boundary (Boundary): the boundary the agent needs to cross
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log=log, sticky=sticky, invert=invert)
        self._agent = agent
        self._boundary = boundary

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Crossed Boundary Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if the given agent has crossed the boundary.

        The position of the agent will be based on the observing agent's perception. If
        The observing agent is unaware of it's own position, or has an incorrect
        perception of it's position, this will be accounted for.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if the agent crossed the boundary during the last time step
        """
        with warnings.catch_warnings():  # suppress warning if agent is not in perceived world
            warnings.simplefilter("ignore", category=AgentNotInPerceivedWorldWarning)
            agent_pos = checker.perceived_world.get_agent_pos(self._agent.unique_id)
        if agent_pos is None:
            return self._invert

        has_crossed: bool
        if self._previous_position is None:
            has_crossed = False
        else:
            has_crossed = self._boundary.has_crossed(self._previous_position, agent_pos)

        self._previous_position = agent_pos
        return has_crossed

    def get_new_instance(self) -> CrossedBoundaryTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            CrossedBoundaryTrigger: New instance of the trigger
        """
        return CrossedBoundaryTrigger(
            agent=self._agent,
            boundary=self._boundary,
            sticky=self._sticky,
            log=self._log,
            invert=self._invert,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return (
            f"Agent: {self._agent.name} ({self._agent.unique_id}), Boundary: {self._boundary.name}"
        )


class MovedOutOfAreaTrigger(Trigger):
    """Trigger that activates when a given agent moves outside a given area."""

    _agent: RetAgent
    _area: Area
    _previous_position: Optional[Coordinate2dOr3d] = None

    def __init__(
        self,
        agent: RetAgent,
        area: Area,
        sticky: bool = False,
        log: bool = True,
        invert: bool = False,
    ) -> None:
        """Create trigger.

        Args:
            agent (RetAgent): the agent that needs to move outside the given location
            area (Area): the area the agent needs to move out of
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log=log, sticky=sticky, invert=invert)
        self._agent = agent
        self._area = area

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Moved Out Of Area Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if the given agent has moved out of the given location.

        The position of the agent will be based on the observing agent's perception. If
        The observing agent is unaware of it's own position, or has an incorrect
        perception of it's position, this will be accounted for.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if the agent has moved outside the area
        """
        with warnings.catch_warnings():  # suppress warning if agent is not in perceived world
            warnings.simplefilter("ignore", category=AgentNotInPerceivedWorldWarning)
            agent_pos = checker.perceived_world.get_agent_pos(self._agent.unique_id)
        if agent_pos is None:
            return self._invert
        has_crossed: bool
        if self._previous_position is None:
            has_crossed = False
        else:
            was_inside = self._area.contains(self._previous_position)
            now_inside = self._area.contains(agent_pos)
            has_crossed = was_inside and not now_inside

        self._previous_position = agent_pos
        return has_crossed

    def get_new_instance(self) -> MovedOutOfAreaTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            MovedOutOfAreaTrigger: New instance of the trigger
        """
        return MovedOutOfAreaTrigger(
            agent=self._agent,
            area=self._area,
            sticky=self._sticky,
            log=self._log,
            invert=self._invert,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return f"Agent: {self._agent.name} ({self._agent.unique_id}), Area: {self._area.name}"


class AliveAgentsAtPositionTrigger(Trigger):
    """Trigger that activates when any alive agents are perceived at the given location.

    Excludes the checker agent.

    Position is defined with a tolerance. The agents at the location are taken from the checker's
    perceived world. If agents at the location are not in the checker's perceived world then the
    trigger will return False.
    """

    _position: Coordinate2dOr3d
    _tolerance: float

    def __init__(
        self,
        position: Coordinate2dOr3d,
        tolerance: float,
        sticky: bool = False,
        log: bool = True,
        invert: bool = False,
    ) -> None:
        """Create trigger.

        Args:
            position (Coordinate2dOr3d): the position to search for alive agents at
            tolerance (float): the tolerance on the position, inclusive
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log=log, sticky=sticky, invert=invert)
        self._position = position
        self._tolerance = tolerance

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Alive Agents at Position Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if there are any alive agents at the given location, excluding the checker.

        The position of the agent will be based on the observing agent's perception. If
        The observing agent is unaware of it's own position, or has an incorrect
        perception of it's position, this will be accounted for.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if there are any agents (excluding the checker agent) within the tolerance
            of the given position, inclusive
        """
        perceived_agents = checker.perceived_world.get_perceived_agents(
            Not(AgentById(checker.unique_id))
        )
        for agent in perceived_agents:
            if agent.casualty_state == AgentCasualtyState.ALIVE:
                distance = checker.perceived_world.get_distance(agent.location, self._position)
                if distance <= self._tolerance:
                    return True
        return False

    def get_new_instance(self) -> AliveAgentsAtPositionTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            AliveAgentsAtPositionTrigger: New instance of the trigger
        """
        return AliveAgentsAtPositionTrigger(
            position=self._position,
            tolerance=self._tolerance,
            sticky=self._sticky,
            log=self._log,
            invert=self._invert,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return f"Position: {self._position}, Tolerance: {self._tolerance}"
