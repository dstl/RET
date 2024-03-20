"""Disable communication behaviour."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ret.agents.affiliation import Affiliation
from ret.behaviours.loggablebehaviour import LoggableBehaviour
from ret.communication.communicationnetwork import CommunicationNetworkModifier

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Optional, Union

    from ret.agents.agent import RetAgent


class DisableCommunicationBehaviour(LoggableBehaviour, ABC):
    """Abstract class for disabling communication."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(self, disabler: RetAgent) -> None:
        """Disable all applicable agents and log.

        Step through all agents identified by `identify_agents_to_disable`,
        which should be overridden in implementations of this abstract class,
        and disables comms from these agents, either disabling all comms, or
        comms to specific recipients, based on the payload returned from
        `identify_agents_to_disable`.

        Args:
            disabler (RetAgent): Agent doing the disabling.
        """
        self.log(disabler)
        self._step(disabler)

    def _step(self, disabler: RetAgent) -> None:
        """Disable all applicable agents.

        Step through all agents identified by `identify_agents_to_disable`,
        which should be overridden in implementations of this abstract class,
        and disables comms from these agents, either disabling all comms, or
        comms to specific recipients, based on the payload returned from
        `identify_agents_to_disable`.

        Args:
            disabler (RetAgent): Agent doing the disabling.
        """
        expiry_time = self.get_expiry_time()

        agents_to_disable = self.identify_agents_to_disable(disabler)

        for comms_source, comms_targets in agents_to_disable:
            if comms_targets is None:
                comms_source.communication_network.add_modifier(
                    CommunicationNetworkModifier(expiry_time=expiry_time)
                )
            else:
                for comms_target in comms_targets:
                    comms_source.communication_network.add_modifier(
                        CommunicationNetworkModifier(expiry_time=expiry_time, target=comms_target)
                    )

    @abstractmethod
    def identify_agents_to_disable(
        self, disabler: RetAgent
    ) -> list[tuple[RetAgent, Union[None, list[RetAgent]]]]:  # pragma: no cover
        """Return a list of agents to be disabled. Overridden in extending classes.

        Return items in the list are in the form of tuples. The first value in the tuple
        is the agent to disable, and the second item is the agents to disable comms to.
        If this is none, all outbound comms are disabled.

        Args:
            disabler (RetAgent): Agent doing the disabling

        Returns:
            list[tuple[RetAgent, Union[None, list[RetAgent]]]: Agents to disable
        """
        pass

    def get_expiry_time(self) -> Union[datetime, None]:
        """Return expiry time for the disable task. Overridden in extending classes.

        Returns:
            Union[datetime, None]: Expiry time, or None, where None indicates permanent
                disabling
        """
        return None


class DisableAllHostileCommsInRangeBehaviour(DisableCommunicationBehaviour):
    """Disable all hostile communication within a range of the agent."""

    _range: float

    def __init__(self, range: float, log: bool = True):
        """Create a new behaviour.

        Args:
            range (float): Range to disable
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)
        self._range = range

    def identify_agents_to_disable(
        self, disabler: RetAgent
    ) -> list[tuple[RetAgent, Union[None, list[RetAgent]]]]:
        """Identify all agents which are HOSTILE and within a range.

        Args:
            disabler (RetAgent): Disabling agent

        Returns:
            list[tuple[RetAgent, Union[None, list[RetAgent]]]]: Agents to disable
        """
        agents_in_range: list[RetAgent] = disabler.model.space.get_neighbors(
            disabler.pos, self._range, True  # type: ignore
        )

        agents: list[tuple[RetAgent, Optional[list[RetAgent]]]] = [
            (agent, None) for agent in agents_in_range if agent.affiliation == Affiliation.HOSTILE
        ]

        return agents

    def _get_log_message(self, **kwargs: Any) -> str:
        """Get the log message.

        Args:
            **kwargs(Any): Key word arguments.
                None used by this override.

        Returns:
            str: The log message
        """
        return f"Range: {self._range}"
