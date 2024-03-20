"""Sensor based triggers."""

from __future__ import annotations
from ret.agents.agent import AgentSensedStatus
from ret.orders.order import Trigger
from typing import TYPE_CHECKING, Optional
from ret.sensing.perceivedworld import (
    And,
    Or,
    Not,
    AgentByType,
    AgentType,
    AffiliatedAgents,
    AgentsWithinConfidence,
    AgentsByCasualtyState,
    AgentById,
)

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent
    from ret.agents.affiliation import Affiliation
    from ret.agents.agent import AgentCasualtyState
    from ret.sensing.sensor import Confidence


class IlluminatedBySensorTrigger(Trigger):
    """Illuminated by sensor trigger."""

    _agent: RetAgent

    def __init__(self, agent: RetAgent, sticky: bool = False, log: bool = True):
        """Create a new illuminated by sensor Trigger.

        Args:
            agent (RetAgent): The agent that needs to be illuminated.
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(log=log, sticky=sticky)
        self._agent = agent

    def __str__(self) -> str:
        """Output a human readable name for the trigger.

        Returns:
            str: brief description of the trigger
        """
        return "Illuminated by Sensor Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check if a sensed event occurred during the last time step.

        Args:
            checker (RetAgent): the agent doing the checking

        Returns:
            bool: true if a sensed event occurred during the last time step
        """
        recent_sensed_events = [
            event for event in self._agent.get_statuses() if isinstance(event, AgentSensedStatus)
        ]

        return len(recent_sensed_events) > 0

    def get_new_instance(self) -> IlluminatedBySensorTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            IlluminatedBySensorTrigger: New instance of the trigger
        """
        return IlluminatedBySensorTrigger(agent=self._agent, sticky=self._sticky, log=self._log)


class AgentsSensedTrigger(Trigger):
    """Agents sensed trigger."""

    def __init__(
        self,
        agent_affiliation: Affiliation,
        confidence_level: Confidence,
        agent_types: Optional[list[AgentType]] = None,
        casualty_state: Optional[AgentCasualtyState] = None,
        log=True,
        sticky=False,
    ):
        """Initialise trigger.

        Args:
            agent_affiliation (Affiliation): The affiliation of the agents that
                will activate the trigger
            confidence_level (Confidence): The confidence level of detection that
                will activate the trigger.
            agent_types (list[AgentType]): The agent types that will activate the trigger.
                If None all agent types will activate the trigger. Defaults to None.
            casualty_state (AgentCasualtyState): The casualty state of agents sensed that
                will activate the trigger. If None any casualty state will activate the
                trigger. Defaults to None.
            log (bool): whether to log or not.
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
        """
        super().__init__(log, sticky)
        self._affiliation = agent_affiliation
        self._agent_types = agent_types
        self._confidence = confidence_level
        self._casualty_state = casualty_state

    def __str__(self) -> str:
        """Output a human readable name for the trigger.

        Returns:
            str: brief description of the trigger
        """
        return "Agents Sensed Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for agents.

        Args:
            checker (RetAgent): The agent checking the condition.

        Returns:
            bool: true if checker categories are satisfied.
        """
        checker_categories = []
        agent_type_categories = []

        # Add Affiliation
        checker_categories.append(AffiliatedAgents(affiliation=self._affiliation))

        # Add Confidence Levels
        checker_categories.append(
            AgentsWithinConfidence(confidence=self._confidence)  # type: ignore
        )

        # Add Agent Types
        if self._agent_types:
            for agent_type in self._agent_types:
                agent_type_to_add = AgentByType(agent_type=agent_type)
                agent_type_categories.append(agent_type_to_add)

        # Add Casualty State
        if self._casualty_state:
            checker_categories.append(
                AgentsByCasualtyState(casualty_state=self._casualty_state)  # type: ignore
            )

        # Perform check
        self.sensed_agents = checker.perceived_world.get_perceived_agents(
            And(
                [
                    And(checker_categories),  # type: ignore
                    Or(agent_type_categories),  # type: ignore
                    Not(AgentById(checker.unique_id)),
                ]
            )
        )

        return len(self.sensed_agents) > 0

    def get_new_instance(self) -> AgentsSensedTrigger:
        """Get new instance of Trigger.

        Returns:
            AgentsSensedTrigger: New instance of the trigger.
        """
        return AgentsSensedTrigger(
            agent_affiliation=self._affiliation,
            agent_types=self._agent_types,
            confidence_level=self._confidence,
            casualty_state=self._casualty_state,
            log=self._log,
            sticky=self._sticky,
        )


class AgentsSensedInRangeTrigger(AgentsSensedTrigger):
    """Agents sensed in range trigger."""

    def __init__(
        self,
        agent_affiliation: Affiliation,
        confidence_level: Confidence,
        sense_range: int,
        agent_types: Optional[list[AgentType]] = None,
        casualty_state: Optional[AgentCasualtyState] = None,
        log=True,
        sticky=False,
    ):
        """Initialise trigger.

        Args:
            agent_affiliation (Affiliation): The affiliation of the agents that
                will activate the trigger
            confidence_level (Confidence): The confidence level of detection that
                will activate the trigger.
            agent_types (list[AgentType]): The agent types that will activate the trigger.
                If None all agent types will activate the trigger. Defaults to None.
            sense_range (int): The range that sensed agents
                 must be inside of to activate the trigger.
            casualty_state (AgentCasualtyState): The casualty state of agents sensed that
                will activate the trigger. If None any casualty state will activate the
                trigger. Defaults to None.
            log (bool): whether to log or not.
            sticky (bool): if true once activated this trigger will remain activated.
                Defaults to False.
        """
        super().__init__(
            agent_affiliation, confidence_level, agent_types, casualty_state, log, sticky
        )
        self._sense_range = sense_range

    def __str__(self) -> str:
        """Output a human readable name for the trigger.

        Returns:
            str: brief description of the trigger
        """
        return "Agents Sensed In Range Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Check for agents.

        Args:
            checker (RetAgent): The agent checking the condition.

        Returns:
            bool: true if checker categories are satisfied.
        """
        super()._check_condition(checker)
        self.sensed_agents = [
            a
            for a in self.sensed_agents
            if checker.model.space.get_distance(checker.pos, a.location) < self._sense_range
        ]
        return len(self.sensed_agents) > 0

    def get_new_instance(self) -> AgentsSensedInRangeTrigger:
        """Get new instance of Trigger.

        Returns:
            AgentsSensedInRangeTrigger: New instance of the trigger.
        """
        return AgentsSensedInRangeTrigger(
            agent_affiliation=self._affiliation,
            agent_types=self._agent_types,
            sense_range=self._sense_range,
            confidence_level=self._confidence,
            casualty_state=self._casualty_state,
            log=self._log,
            sticky=self._sticky,
        )
