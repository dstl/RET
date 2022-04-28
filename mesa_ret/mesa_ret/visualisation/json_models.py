"""Display agent type definition."""
from __future__ import annotations

# Optional, Union, TimeT, RetAgent, ContinuousSpaceWithTerrainAndCulture2d, and Coordinate2dOr3d
# have to be imported directly in order to suppert their use as types within pydantic models.
import copy
from typing import TYPE_CHECKING, Optional, Union  # noqa: TC002

from mesa.time import TimeT  # noqa: TC002
from mesa_ret.agents.agent import RetAgent  # noqa: TC002
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture2d  # noqa: TC002
from mesa_ret.types import Coordinate2dOr3d  # noqa: TC002
from mesa_ret.visualisation.json_icon_handler import IconCopier
from pydantic import BaseModel  # noqa: TC002

if TYPE_CHECKING:
    from mesa_ret.sensing.perceivedworld import PerceivedAgent


class RetPlayPerceivedAgent(BaseModel):
    """A class to hold the data relevant for displaying an agent's perceived world.

    The PerceivedAgent creates a copy of all the relevant fields for display and makes them
    available. Known agents are treated differently from perceived agents as they will
    have different fields available. This manifests as some fields being null.
    """

    name: Optional[str] = None
    affiliation: str
    agent_type: str
    location: tuple[float, ...]
    icon: Optional[str] = None
    unique_id: Optional[int]

    sense_time: Optional[str] = None
    confidence: Optional[str] = None
    casualty_state: str
    casualty_state_confidence: Optional[str] = None

    @staticmethod
    def get_retplay_perceived_agent_from_perceived_agent(
        perceived_agent: PerceivedAgent,
    ) -> RetPlayPerceivedAgent:
        """Generate a RetPlayPerceivedAgent from a perceived agent.

        Args:
            perceived_agent (PerceivedAgent): the perceived agent in the perceived world to be
                converted into a RetPlayPerceivedAgent.

        Returns:
            RetPlayPerceivedAgent: The RetPlay representation of the perceived agent.
        """
        affiliation = str(perceived_agent.affiliation)
        agent_type = str(perceived_agent.agent_type)
        pos: tuple[float, ...] = perceived_agent.location
        id: Optional[int] = perceived_agent.unique_id
        sense_time = str(perceived_agent.sense_time)
        confidence = str(perceived_agent.confidence)
        casualty_state = str(perceived_agent.casualty_state)
        casualty_state_confidence = str(perceived_agent._casualty_state_confidence)

        return RetPlayPerceivedAgent(
            affiliation=affiliation,
            agent_type=agent_type,
            location=pos,
            unique_id=id,
            sense_time=sense_time,
            confidence=confidence,
            casualty_state=casualty_state,
            casualty_state_confidence=casualty_state_confidence,
        )

    @staticmethod
    def get_retplay_perceived_agent_from_known_agent(
        known_agent: RetAgent, icon_handler: Optional[IconCopier] = None
    ) -> RetPlayPerceivedAgent:
        """Generate a RetPlayPerceivedAgent from a known RetAgent.

        Args:
            known_agent (RetAgent): The agent which is always known in the perceived world to be
                converted into a RetPlayPerceivedAgent.
            icon_handler (Optional[IconCopier]): An icon handler. If None or unspecified defaults to
                IconCopier.

        Returns:
            RetPlayPerceivedAgent: The RetPlay representation of the known RetAgent.
        """
        if icon_handler is None:
            icon_handler = IconCopier()

        name = known_agent.name
        affiliation = str(known_agent.affiliation)
        agent_type = str(known_agent.agent_type)
        location: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d]]] = known_agent.pos
        icon = icon_handler.copy_icon(known_agent.icon_path)
        unique_id = known_agent.unique_id
        casualty_state = str(
            AgentCasualtyState.KILLED if known_agent._killed else AgentCasualtyState.ALIVE
        )

        return RetPlayPerceivedAgent(
            name=name,
            affiliation=affiliation,
            agent_type=agent_type,
            location=location,
            icon=icon,
            unique_id=unique_id,
            casualty_state=casualty_state,
        )

    @staticmethod
    def from_model(
        agent: RetAgent, icon_handler: Optional[IconCopier] = None
    ) -> list[RetPlayPerceivedAgent]:
        """Generate a list of perceived agents.

        Args:
            agent (RetAgent): The agent whose perceptions will be recorded.
            icon_handler (Optional[IconCopier]): A conversion utility to copy the icons into an
                assets folder. If None or unspecified, defaults to IconCopier.

        Returns:
            list[RetPlayPerceivedAgent]: All of the perceived agents in the agents perceived world.
                This excludes the agent themselves but includes other agents that will always be
                known.
        """
        if icon_handler is None:
            icon_handler = IconCopier()

        perceived_agents = []

        for perceived_agent in agent.perceived_world._perceived_agents:

            perceived_agents.append(
                RetPlayPerceivedAgent.get_retplay_perceived_agent_from_perceived_agent(
                    perceived_agent
                )
            )

        for known_agent in agent.perceived_world._agents:

            if known_agent.unique_id is not agent.unique_id:

                perceived_agents.append(
                    RetPlayPerceivedAgent.get_retplay_perceived_agent_from_known_agent(
                        known_agent, icon_handler
                    )
                )

        return perceived_agents


class RetPlayAgent(BaseModel):
    """A class to hold the agent data relevant for displaying.

    The DisplayAgent creates a copy of all the fields in the supplied RetAgent which are
    relevant to the display and makes them available.
    """

    name: str
    affiliation: str
    agent_type: str
    id: int
    pos: tuple[float, ...]
    icon: str

    active_order: str

    killed: bool
    in_group: bool

    perceived_agents: list[RetPlayPerceivedAgent]
    mission_messages: list[str]

    @staticmethod
    def from_model(agent: RetAgent, icon_handler: Optional[IconCopier] = None) -> RetPlayAgent:
        """Create a DisplayAgent.

        Args:
            agent (RetAgent): The agent which the DisplayAgent will represent.
            icon_handler (Optional[IconCopier]): A conversion utility to copy the icons into an
                assets folder. If None or unspecified, defaults to an in built class for handling
                icon copying.

        Returns:
            RetPlayAgent: RetPlayAgent instance
        """
        if icon_handler is None:
            icon_handler = IconCopier()

        name = agent.name
        affiliation = str(agent.affiliation.name)
        agent_type = str(agent.agent_type.name)
        pos: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d]]] = agent.pos
        icon = icon_handler.copy_icon(agent.icon_path)
        id = agent.unique_id

        if agent.active_order:
            active_order = agent.active_order.__str__()
        else:
            active_order = "No active order"

        killed = agent._killed
        in_group = agent.in_group

        perceived_agents: list[RetPlayPerceivedAgent] = RetPlayPerceivedAgent.from_model(agent)
        mission_messages: list[str] = copy.deepcopy(agent.mission_messages)
        return RetPlayAgent(
            name=name,
            affiliation=affiliation,
            agent_type=agent_type,
            pos=pos,
            icon=icon,
            id=id,
            active_order=active_order,
            killed=killed,
            in_group=in_group,
            perceived_agents=perceived_agents,
            mission_messages=mission_messages,
        )


class RetPlaySpace(BaseModel):
    """Class to hold the map size."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @staticmethod
    def from_model(space: ContinuousSpaceWithTerrainAndCulture2d) -> RetPlaySpace:
        """Create a RetPlay space representation.

        Args:
            space (ContinuousSpaceWithTerrainAndCulture2d): The space from which the
                dimensions are taken.

        Returns:
            RetPlaySpace: RetPlaySpace instance
        """
        return RetPlaySpace(
            x_min=space.x_min,
            x_max=space.x_max,
            y_min=space.y_min,
            y_max=space.y_max,
        )


class RetPlayInitialData(BaseModel):
    """Class to hold initial information for model playback."""

    map_size: RetPlaySpace

    @staticmethod
    def from_model(space: ContinuousSpaceWithTerrainAndCulture2d) -> RetPlayInitialData:
        """Create the initial RetPlay information from the model setup.

        Args:
            space (ContinuousSpaceWithTerrainAndCulture2d): The space from which to take
                the map size.

        Returns:
            RetPlayInitialData: RetPlayInitialData instance
        """
        return RetPlayInitialData(map_size=RetPlaySpace.from_model(space))


class RetPlayStepData(BaseModel):
    """Class to hold step by step information for model playback."""

    step_number: TimeT
    agents: list[RetPlayAgent]

    @staticmethod
    def from_model(step_number: TimeT, agents: list[RetPlayAgent]) -> RetPlayStepData:
        """Create a step data object to store information from one time step.

        Args:
            step_number (TimeT): The number of the step from which the data comes.
            agents (list[RetPlayAgent]): A list of representations of the agents and
                their state in the recorded timestep.

        Returns:
            RetPlayStepData: RetPlayStepData instance
        """
        return RetPlayStepData(step_number=step_number, agents=agents)


class JsonOutObject(BaseModel):
    """Class to hold JSON information for model playback."""

    initial_data: RetPlayInitialData
    step_data: list[RetPlayStepData]

    @staticmethod
    def from_model(space: ContinuousSpaceWithTerrainAndCulture2d) -> JsonOutObject:
        """Create a JSON output object.

        Args:
            space (ContinuousSpaceWithTerrainAndCulture2d): The space required to
                generate the initial information for the JSON representation.

        Returns:
            JsonOutObject: JsonOutObject instance
        """
        return JsonOutObject(step_data=[], initial_data=RetPlayInitialData.from_model(space))

    def add_timestep_data(self, step_number: TimeT, agents: list[RetPlayAgent]):
        """Add one timestep's worth of data to the JsonOutObject.

        Args:
            step_number (TimeT): The number of the step from which the data comes.
            agents (list[RetPlayAgent]): A list of representations of the agents and
                their state in the recorded timestep.

        Raises:
            ValueError: If there is an invalid step number given, this error will throw.
        """
        if step_number is not None:
            self.step_data.append(RetPlayStepData.from_model(step_number, agents))
        else:
            raise ValueError(
                "The time step information failed as there was no step number defined."
            )
