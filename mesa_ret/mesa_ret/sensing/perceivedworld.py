"""Simulation of perceived world - The view that an agent has of the world."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mesa_ret.agents.agenttype import AgentType
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState

if TYPE_CHECKING:
    from typing import Callable, Optional, Union
    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.model import RetModel
    from mesa_ret.types import Coordinate2dOr3d, Coordinate
    from datetime import datetime
    from random import Random

from collections.abc import Iterable
from enum import Enum

from mesa_ret.agents.affiliation import Affiliation


class Confidence(Enum):
    """Target acquisition states."""

    DETECT = 1
    RECOGNISE = 2
    IDENTIFY = 3
    KNOWN = 4


class AgentNotInPerceivedWorldWarning(Warning):
    """Custom warning type for agent not in perceived world."""

    pass


class PerceivedAgent:
    """Representation of an agent within a PerceivedWorld.

    This may represent a real or spurious agent, and may or may not be represented
    at the location it is actually at.
    """

    _confidence: Confidence
    _location: Coordinate2dOr3d
    _unique_id: Optional[int]
    _affiliation: Affiliation
    _agent_type: AgentType
    _sense_time: datetime
    _casualty_state: AgentCasualtyState

    def __init__(
        self,
        sense_time: datetime,
        confidence: Confidence,
        location: Coordinate2dOr3d,
        unique_id: Optional[int],
        casualty_state: AgentCasualtyState,
        affiliation: Affiliation = Affiliation.UNKNOWN,
        agent_type: AgentType = AgentType.UNKNOWN,
        casualty_state_confidence: Confidence = Confidence.RECOGNISE,
    ):
        """Return a new perceived agent.

        Args:
            sense_time (datetime): The time at which the sensing was completed.
            confidence (Confidence): Confidence of the target acquisition.
            location (Coordinate2dOr3d): Detection location.
            unique_id (Optional[int]): Unique identifier of agent.
            casualty_state (AgentCasualtyState): Casualty state of perceived agent.
            affiliation (Affiliation): Affiliation of agent. Defaults to UNKNOWN.
            agent_type (AgentType): Type of agent. Defaults to UNKNOWN.
            casualty_state_confidence (Confidence): Minimum confidence level at which
                casualty state is known to an external observer. Defaults to RECOGNISE.
        """
        self._sense_time = sense_time
        self._confidence = confidence
        self._location = location
        self._unique_id = unique_id
        self._affiliation = affiliation
        self._agent_type = agent_type
        self._casualty_state = casualty_state
        self._casualty_state_confidence = casualty_state_confidence

    @property
    def affiliation(self) -> Affiliation:
        """Get the perceived agent affiliation.

        Returns:
            Affiliation: If the agent's confidence is equal to or above identify, return
                the agent's affiliation. Otherwise, return Affiliation.UNKNOWN.
        """
        if self._confidence.value >= Confidence.IDENTIFY.value:
            return self._affiliation
        return Affiliation.UNKNOWN

    @property
    def confidence(self) -> Confidence:
        """Get the perceived agent confidence.

        Returns:
            Confidence: confidence
        """
        return self._confidence

    @property
    def location(self) -> Coordinate2dOr3d:
        """Get the perceived agent location.

        Returns:
            Coordinate2dOr3d: location
        """
        return self._location

    @property
    def unique_id(self) -> Optional[int]:
        """Get the perceived agent unique ID.

        Returns:
            Optional[int]: If the perceived agent's confidence is Known or identified, returns the
                Unique ID, otherwise, returns None
        """
        if self._confidence == Confidence.KNOWN or self._confidence == Confidence.IDENTIFY:
            return self._unique_id
        return None

    @property
    def agent_type(self) -> AgentType:
        """Get the perceived agent type.

        Returns:
            AgentType: If the perceived agent's confidence is Recognise or greater, the
                agent's type, otherwise Unknown
        """
        if self._confidence.value >= Confidence.RECOGNISE.value:
            return self._agent_type
        return AgentType.UNKNOWN

    @property
    def sense_time(self) -> datetime:
        """Get the time that the perceived agent is sensed.

        Returns:
            datetime: sense time
        """
        return self._sense_time

    @property
    def casualty_state(self) -> AgentCasualtyState:
        """Get casualty state of perceived agent.

        If confidence level is greater than or equal to casualty_state_confidence,
        correct casualty state is returned, otherwise UNKNOWN is returned.

        Returns:
            AgentCasualtyState: casualty state.
        """
        if self._confidence.value >= self._casualty_state_confidence.value:
            return self._casualty_state
        return AgentCasualtyState.UNKNOWN

    def perception_log_representation(self) -> tuple[str, str, str]:
        """Get a representation of the Perceived Agent to include in logging.

        Returns:
            tuple[str, str, str]: Agent ID, Agent Position, Confidence
        """
        return (str(self._unique_id), str(self.location), str(self.confidence.name))

    def get_unique_id_log_representation(self) -> str:
        """Get a representation of the Perceived Agent ID to include in logging.

        Returns:
            str: A string representation of the unique agent ID for the agent the
                perception was based off.
        """
        return str(self._unique_id)

    @classmethod
    def to_perceived_agent(cls, agent: RetAgent) -> PerceivedAgent:
        """Convert an agent into a known perceived agent.

        Args:
            agent (RetAgent): Agent to convert

        Returns:
            PerceivedAgent: Perceived Agent representing agent
        """
        sense_time = agent.model.get_time()

        perceived_agent = PerceivedAgent(
            sense_time=sense_time,
            confidence=Confidence.KNOWN,
            location=agent.pos,
            unique_id=agent.unique_id,
            casualty_state=agent.casualty_state,
            affiliation=agent.affiliation,
            agent_type=agent.agent_type,
        )

        return perceived_agent


class PerceivedAgentBasedOnAgent(PerceivedAgent):
    """Perceived agent generated using a RetAgent."""

    def __init__(
        self,
        sense_time: datetime,
        confidence: Confidence,
        agent: RetAgent,
        casualty_state_confidence: Confidence = Confidence.RECOGNISE,
        location: Optional[Coordinate2dOr3d] = None,
    ):
        """Initialise a perceived agent based on an agent.

        Args:
            sense_time (datetime): The time at which the sensing was completed.
            confidence (Confidence): Confidence of the target acquisition.
            agent (RetAgent): RetAgent that perceived agent is based upon.
            casualty_state_confidence (Confidence): Minimum confidence level at which
                casualty state is known to an external observer. Defaults to RECOGNISE.
            location (Optional[Coordinate2dOr3D]): optional sensed agent location, if
                none is provided the true agent location is used
        """
        if location is not None:
            _location = location
        else:
            _location = agent.pos
        _unique_id = agent.unique_id
        _affiliation = agent.affiliation
        _agent_type = agent.agent_type
        _casualty_state = agent.casualty_state
        _casualty_state_confidence = casualty_state_confidence

        super().__init__(
            sense_time=sense_time,
            confidence=confidence,
            location=_location,
            unique_id=_unique_id,
            affiliation=_affiliation,
            agent_type=_agent_type,
            casualty_state=_casualty_state,
            casualty_state_confidence=_casualty_state_confidence,
        )


class PerceivedAgentFilter(ABC):
    """Filter for perceived agents.

    To be extended for specific filtering behaviour.
    """

    @abstractmethod
    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:  # pragma: no cover
        """Filter the agents by the filter's conditions.

        To be overridden in child classes with explicit filter behaviour.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        pass


class AffiliatedAgents(PerceivedAgentFilter):
    """Agent filter which filters bad on affiliation."""

    affiliation: Affiliation

    def __init__(self, affiliation: Affiliation):
        """Generate a new PerceivedAgentFilter filter.

        Args:
            affiliation (Affiliation): Affiliation to filter by.
        """
        self.affiliation = affiliation

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove perceived agents by affiliation.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        return [a for a in agents if a.affiliation == self.affiliation]


class NeutralAgents(AffiliatedAgents):
    """Agent filter for neutral agents filter."""

    def __init__(self):
        """Generate a new NeutralAgent filter."""
        super().__init__(Affiliation.NEUTRAL)


class UnknownAgents(AffiliatedAgents):
    """Agent filter for unknown agents filter."""

    def __init__(self):
        """Generate a new UnknownAgent filter."""
        super().__init__(Affiliation.UNKNOWN)


class FriendlyAgents(AffiliatedAgents):
    """Agent filter for friendly agents filter."""

    def __init__(self):
        """Generate a new FriendlyAgent filter."""
        super().__init__(Affiliation.FRIENDLY)


class HostileAgents(AffiliatedAgents):
    """Agent filter for hostile agents filter."""

    def __init__(self):
        """Generate a new HostileAgent filter."""
        super().__init__(Affiliation.HOSTILE)


class AgentsWithinConfidence(PerceivedAgentFilter):
    """Agent filter for agents acquired at or above a confidence."""

    confidence: Confidence

    def __init__(self, confidence: Confidence):
        """Generate a new PerceivedAgentFilter filter.

        Args:
            confidence (Confidence): Minimum confidence level.
        """
        self.confidence = confidence

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove all perceived agents below confidence.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        return [a for a in agents if a.confidence.value >= self.confidence.value]


class DetectedAgents(AgentsWithinConfidence):
    """Agent filter for agents who have been detected."""

    def __init__(self):
        """Generate a new DetectedAgent filter."""
        super().__init__(Confidence.DETECT)


class RecognisedAgents(AgentsWithinConfidence):
    """Agent filter for agents who have been recognised."""

    def __init__(self):
        """Generate a new RecognisedAgents filter."""
        super().__init__(Confidence.RECOGNISE)


class IdentifiedAgents(AgentsWithinConfidence):
    """Agent filter for agents who have been recognised."""

    def __init__(self):
        """Generate a new IdentifiedAgents filter."""
        super().__init__(Confidence.IDENTIFY)


class RandomSelection(PerceivedAgentFilter):
    """Agent filter for selecting up to a number of agent from those available.

    If a set of perceptions has less than the number to select, the entire set of
    perceptions will be returned.

    Where randomly selecting, the RandomSelection will not repeatedly select the same
    view.
    """

    _random_generator: Random
    _number_to_select: int

    def __init__(self, random_generator: Random, number_to_select: int):
        """Create a new RandomSelection PerceivedAgentFilter.

        Args:
            random_generator (Random): Random generator
            number_to_select (int): Number of perceptions to select.

        Raises:
            ValueError: Raised where the number of agents to select is less than 1
        """
        if number_to_select < 1:
            raise ValueError(f"`number_to_select` is {number_to_select}, it must be greater than 0")

        self._random_generator = random_generator
        self._number_to_select = number_to_select

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Return a random selection of agents from within the candidate agents.

        Args:
            agents (list[PerceivedAgent]): Agents to filter

        Returns:
            list[PerceivedAgent]: Random selection of agents
        """
        results: list[PerceivedAgent] = []

        if len(agents) <= self._number_to_select:
            return agents

        while len(results) < self._number_to_select:
            index = self._random_generator.randint(0, len(agents) - 1)
            results.append(agents.pop(index))

        return results


class AgentsWithinRange(PerceivedAgentFilter):
    """Agent filter for agents within a distance of a location."""

    location: Coordinate2dOr3d
    range: float

    def __init__(self, distance_calculator: Callable, location: Coordinate2dOr3d, range: float):
        """Create a new PerceivedAgentFilter.

        Args:
            distance_calculator (Callable): Function reference for calculating distance
                between coordinate points.
            location (Coordinate2dOr3d): Coordinate to locate at.
            range (float): Range to locate agent within.
        """
        self.location = location
        self.range = range
        self.distance_calculator = distance_calculator

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove all agents that are not within range of a point.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        return [
            a for a in agents if self.distance_calculator(a.location, self.location) <= self.range
        ]


class AgentsAt(AgentsWithinRange):
    """Agent filter for agents at a specific point.

    Single point is considered to be within 1e-6, to account for any floating point
    incaccuracies in range calculations.
    """

    def __init__(self, distance_calculator: Callable, location: Coordinate2dOr3d):
        """Create a new AgentsAt filter.

        Args:
            distance_calculator (Callable): Function handle used for calculating the
                distance between two coordinates
            location (Coordinate2dOr3d): Coordinate to filter at
        """
        super().__init__(distance_calculator, location, 1e-6)


class NearbyAgents(PerceivedAgentFilter):
    """Agent filter for agents within a distance of an agent."""

    location: Coordinate2dOr3d
    range: float

    def __init__(self, distance_calculator: Callable, agent: RetAgent, range: float):
        """Create a new PerceivedAgentFilter.

        Args:
            distance_calculator (Callable): Function reference for calculating distance
                between coordinate points.
            agent (RetAgent): Agent to detect nearby.
            range (float): Range to locate agent within.
        """
        self.agent = agent
        self.range = range
        self.distance_calculator = distance_calculator

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove all agents that are not within range of a point.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        return [
            a for a in agents if self.distance_calculator(a.location, self.agent.pos) <= self.range
        ]


class AgentById(PerceivedAgentFilter):
    """Agent filter for an agent by id."""

    _agent_id: int

    def __init__(self, id: int):
        """Create a new AgentById filter.

        Args:
            id (int): Agent ID to filter by.
        """
        self._agent_id = id

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove all agents who's id is not agent_id.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        return [a for a in agents if a.unique_id == self._agent_id]


class AgentByType(PerceivedAgentFilter):
    """Filter for agents by type."""

    _agent_type: AgentType

    def __init__(self, agent_type: AgentType):
        """Create a new agent type filter.

        Args:
            agent_type (AgentType): AgentType to filter by.
        """
        self._agent_type = agent_type

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove all agents which are not self._agent_type.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        return [a for a in agents if a.agent_type == self._agent_type]


class AirAgents(AgentByType):
    """Filter for air agents."""

    def __init__(self):
        """Create a new AirAgents filter."""
        super().__init__(agent_type=AgentType.AIR)


class AirDefenceAgents(AgentByType):
    """Filter for air defence agents."""

    def __init__(self):
        """Create a new AirDefenceAgents filter."""
        super().__init__(agent_type=AgentType.AIR_DEFENCE)


class ArmourAgents(AgentByType):
    """Filter for armour agents."""

    def __init__(self):
        """Create a new ArmourAgents filter."""
        super().__init__(agent_type=AgentType.ARMOUR)


class InfantryAgents(AgentByType):
    """Filter for infantry agents."""

    def __init__(self):
        """Create a new InfantryAgents filter."""
        super().__init__(agent_type=AgentType.INFANTRY)


class GenericAgents(AgentByType):
    """Filter for generic agents."""

    def __init__(self):
        """Create a new GenericAgents filter."""
        super().__init__(agent_type=AgentType.GENERIC)


class OtherAgents(AgentByType):
    """Filter for other agents."""

    def __init__(self):
        """Create a new OtherAgents filter."""
        super().__init__(agent_type=AgentType.OTHER)


class UnknownTypeAgents(AgentByType):
    """Filter for unknown agents."""

    def __init__(self):
        """Create a new UnknownTypeAgents filter."""
        super().__init__(agent_type=AgentType.UNKNOWN)


class AgentsByCasualtyState(PerceivedAgentFilter):
    """Filter for agents by perceived killed status."""

    _casualty_state: AgentCasualtyState

    def __init__(self, casualty_state: AgentCasualtyState):
        """Create a new casualty state filter.

        Args:
            casualty_state (AgentCasualtyState): Casualty state to filter by.
        """
        self._casualty_state = casualty_state

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove all agents which are not self._casualty_state.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        return [a for a in agents if a._casualty_state == self._casualty_state]


class AliveAgents(AgentsByCasualtyState):
    """Filter for alive agents."""

    def __init__(self):
        """Create a new alive agents filter."""
        super().__init__(casualty_state=AgentCasualtyState.ALIVE)


class KilledAgents(AgentsByCasualtyState):
    """Filter for killed agents."""

    def __init__(self):
        """Create a new killed agents filter."""
        super().__init__(casualty_state=AgentCasualtyState.KILLED)


class UnknownCasualtyStateAgents(AgentsByCasualtyState):
    """Filter for agents whose casualty state is unknown."""

    def __init__(self):
        """Create a new unknown killed agents filter."""
        super().__init__(casualty_state=AgentCasualtyState.UNKNOWN)


class And(PerceivedAgentFilter):
    """PerceivedAgentFilter for combining zero or more agent filters.

    Agents must be present in all filtered lists.

    Where no filters are provided, all agents are returned.
    """

    filters: list[PerceivedAgentFilter]

    def __init__(self, filters: list[PerceivedAgentFilter]):
        """Create a new And filter which combines zero or more other filters.

        Args:
            filters (list[PerceivedAgentFilter]): List of filters.
        """
        self.filters = filters

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents by each filter, keeping any agents present in all filters.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        for filter in self.filters:
            agents = filter.run(agents)
        return agents


class Or(PerceivedAgentFilter):
    """PerceivedAgentFilter for combining zer or more agent filters.

    Agents must be present in any of the filtered lists.

    Where no filters are provided, all agents are returned.
    """

    filters: list[PerceivedAgentFilter]

    def __init__(self, filters: list[PerceivedAgentFilter]):
        """Create a new Or filter, combining zero or more existing filters.

        Args:
            filters (list[PerceivedAgentFilter]): List of filters.
        """
        self.filters = filters

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents by each filter, keeping any agents present in any filter.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        if self.filters == []:
            return agents

        matching = []
        for filter in self.filters:
            matching.extend(filter.run(agents))

        return list(set(matching))


class Not(PerceivedAgentFilter):
    """PerceivedAgentFilter which negates an agent filter."""

    _filter: PerceivedAgentFilter

    def __init__(self, filter: PerceivedAgentFilter):
        """Create a new Not filter which negates the given filter.

        Args:
            filter (PerceivedAgentFilter): The filter to negate.
        """
        self._filter = filter

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents and return all which do not meet the filter requirements.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        filtered_agents = self._filter.run(agents)
        return [agent for agent in agents if agent not in filtered_agents]


class RemoveDuplicates(PerceivedAgentFilter):
    """PerceivedAgentFilter for removing duplicate agents.

    Where duplicate agents are found, the one with the latest sense time is selected.
    Where multiple agents with the same sense time are found, the one with the highest
    confidence is selected.
    """

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents to remove duplicates.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        unique_ids = set([a._unique_id for a in agents if a._unique_id is not None])

        non_duplicate_agents = [a for a in agents if a._unique_id is None]

        for id in unique_ids:
            agents_with_this_id = [a for a in agents if a._unique_id == id]
            newest = sorted(
                agents_with_this_id,
                key=lambda a: (a.sense_time, a.confidence.value),
                reverse=True,
            )[0]
            non_duplicate_agents.append(newest)

        return non_duplicate_agents


class RemoveDuplicatesAtLocation(PerceivedAgentFilter):
    """Perceived agent filter for removing duplicate agents at a location.

    Duplicate agents considered for this filter to be agents at the same location. Where
    duplicates occur, the perception with the highest confidence will be kept.
    """

    def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter the agents by to remove any duplicate agents at the same location.

        Args:
            agents (list[PerceivedAgent]): Existing list of agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents.
        """
        all_locations = set(a.location for a in agents)

        subset_of_agents = []

        for location in all_locations:
            agents_at_location = [p for p in agents if p.location == location]

            agents_by_confidence = sorted(
                agents_at_location, key=lambda a: a.confidence.value, reverse=True
            )

            subset_of_agents.append(agents_by_confidence[0])

        return subset_of_agents


class PerceivedWorld:
    """Representation of an agent's world-view.

    A PerceivedWorld is used to translate from the real model to what an agent can
    perceive about the world.
    """

    _model: RetModel
    _perceived_agents: list[PerceivedAgent]
    _agents: list[RetAgent]
    _refresh_technique: PerceivedAgentFilter

    def __init__(
        self,
        model: RetModel,
        refresh_technique: Optional[PerceivedAgentFilter] = None,
    ) -> None:
        """Create a new PerceivedWorld.

        Args:
            model (RetModel): the model that the agent will try to perceive
            refresh_technique (PerceivedAgentFilter, optional): Methodology for
                refreshing the perceived world
        """
        self._model = model
        self._perceived_agents = []
        self._agents = []
        if refresh_technique is None:
            refresh_technique = RemoveDuplicates()
        self._refresh_technique = refresh_technique

    def add_acquisitions(self, acquisition: Union[PerceivedAgent, list[PerceivedAgent]]) -> None:
        """Add perceived agent(s) to world view.

        Args:
            acquisition (Union[list[PerceivedAgent], PerceivedAgent]): New agent(s)
        """
        if isinstance(acquisition, Iterable):
            self._perceived_agents.extend(acquisition)
        else:
            self._perceived_agents.append(acquisition)

    def add_agent(self, agent: RetAgent):
        """Add an agent whose position is always known by the perceived world.

        Args:
            agent (RetAgent): Agent
        """
        self._agents.append(agent)

    def reset_worldview(self):
        """Reset the perceived agent's world-view."""
        self._perceived_agents.clear()

    def get_distance(self, pos_1: Coordinate, pos_2: Coordinate) -> float:
        """Return the perceived straight line distance between two points.

        Args:
            pos_1 (Coordinate): point 1
            pos_2 (Coordinate): point 2

        Returns:
            float: perceived straight line distance between the two points
        """
        distance: float = self._model.space.get_distance(pos_1, pos_2)
        return distance

    def get_perceived_agents(
        self,
        filter: Optional[PerceivedAgentFilter] = None,
    ) -> list[PerceivedAgent]:
        """Get a filtered list of perceived agents.

        Args:
            filter (Optional[PerceivedAgentFilter]): Filter. Defaults to None.

        Returns:
            list[PerceivedAgent]: List of agents
        """
        self._perceived_agents = self._refresh_technique.run(self._perceived_agents)

        # A new list is created here to avoid any of the filter operations or downstream
        # actions on the returned list having inadvertent side-effects on
        # self._perceived_agents
        agents = [a for a in self._perceived_agents]

        agents += [PerceivedAgent.to_perceived_agent(agent) for agent in self._agents]
        # Run a refresh on this combined list to remove any duplicates due to both
        # knowing about an agent and having sensed an agent (or received information
        # about an agent) that is known
        agents = self._refresh_technique.run(agents)

        if filter:
            return filter.run(agents)
        return agents

    def get_agent_pos(self, unique_id: int) -> Optional[Coordinate2dOr3d]:
        """Get the position of an agent by ID.

        Args:
            unique_id (int): agent id

        Raises:
            Warning: Uniquely identified agent not found in perceived world.
            ValueError: Uniquely identifiable agent identified in multiple locations.

        Returns:
            Optional[Coordinate2dOr3d]: Optional agent location if agent is in the perceived world.
        """
        agents = self.get_perceived_agents(AgentById(unique_id))

        if len(agents) == 1:
            return agents[0].location
        elif len(agents) == 0:
            warnings.warn(
                "Uniquely identified agent not found in perceived world.",
                category=AgentNotInPerceivedWorldWarning,
            )
            return None
        else:
            raise ValueError("Uniquely identified agent located in more than one place.")
