"""Agent is killed trigger."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ret.agents.agent import WeaponFiredStatus
from ret.orders.order import Trigger

if TYPE_CHECKING:
    from typing import Optional

    from ret.agents.agent import RetAgent
    from ret.types import Coordinate2dOr3d


class AgentFiredWeaponTrigger(Trigger):
    """Trigger for when a given agent fires a weapon.

    This trigger will check the firing agent's status regardless of whether the agent is in the
    checker's perceived world.
    """

    _firer: RetAgent
    _weapon_name: Optional[str]

    def __init__(
        self,
        firer: RetAgent,
        weapon_name: Optional[str] = None,
        log: bool = True,
        sticky: bool = False,
    ):
        """Initialise a weapon fired trigger.

        Args:
            weapon_name (str): The name of the weapon.
            firer (RetAgent): The agent firing the weapon.
            log (bool): whether to log or not. Defaults to True.
            sticky (bool): if true once activated this trigger will remain
                activated. Defaults to True.
        """
        super().__init__(log=log, sticky=sticky, invert=False)
        self._weapon_name = weapon_name
        self._firer = firer

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Agent Fired Weapon Trigger"

    def _check_condition(self, checker: RetAgent) -> bool:
        """Return true if the firer agent fired the weapon in the previous time step.

        This trigger will check the firer's status regardless of whether the firer is in the
        checker's perceived world.

        Args:
            checker (RetAgent): Agent doing the checking.

        Returns:
            bool: Whether or the firer agent fired the weapon.
        """
        agent_fired_statuses = [
            status for status in self._firer.get_statuses() if isinstance(status, WeaponFiredStatus)
        ]
        if self._weapon_name is not None:
            agent_fired_statuses = [
                status for status in agent_fired_statuses if status.weapon_name == self._weapon_name
            ]
        return len(agent_fired_statuses) > 0

    def get_new_instance(self) -> AgentFiredWeaponTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            WeaponFiredTrigger: New instance of the trigger
        """
        return AgentFiredWeaponTrigger(
            weapon_name=self._weapon_name,
            firer=self._firer,
            log=self._log,
            sticky=self._sticky,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return (
            f"Weapon {self._weapon_name} fired by agent: {self._firer.agent_type} "
            "(Agent ID: {self._firer.unique_id}). trigger."
        )


class WeaponFiredNearTrigger(Trigger):
    """Abstract trigger for when a weapon is fired within a tolerance of a target.

    This trigger will check for weapons fired regardless of whether the firing agent is in the
    checker's perceived world.

    Override abstract _get_location method to return location of the target.
    """

    _tolerance: float
    _weapon_name: Optional[str]

    def __init__(
        self,
        tolerance: float,
        weapon_name: Optional[str] = None,
        log: bool = True,
        sticky: bool = False,
    ):
        """Initialise a weapon fired trigger.

        Args:
            tolerance (float): The tolerance to check for weapons fired strictly within.
            weapon_name (Optional[str]): The name of the weapon, if None the trigger will check for
                all weapons. Defaults to None.
            log (bool): whether to log or not. Defaults to True.
            sticky (bool): if true once activated this trigger will remain
                activated. Defaults to True.
        """
        super().__init__(log=log, sticky=sticky, invert=False)
        self._tolerance = tolerance
        self._weapon_name = weapon_name

    def _check_condition(self, checker: RetAgent) -> bool:
        """Return true if any agent fired a weapon within tolerance of agent in the previous step.

        Args:
            checker (RetAgent): Agent doing the checking.

        Returns:
            bool: Whether any weapons were fired within tolerance of the agent.
        """
        agents: list[RetAgent] = checker.model.get_all_agents()
        target_location = self._get_location(checker)
        if target_location is not None:
            for agent in agents:
                if self._has_fired_within_tolerance(checker, target_location, agent):
                    return True

        return False

    @abstractmethod
    def _get_location(self, checker: RetAgent) -> Optional[Coordinate2dOr3d]:  # pragma: no cover
        """Get location to check within tolerance of.

        Args:
            checker (RetAgent): The agent doing the checking.

        Returns:
            Optional[Coordinate2dOr3d]: The location to check within tolerance of.
        """
        pass

    def _has_fired_within_tolerance(
        self, checker: RetAgent, location: Coordinate2dOr3d, agent: RetAgent
    ) -> bool:
        """Check whether agent has fired within tolerance range of the agent.

        Args:
            checker (RetAgent): The agent doing the check.
            location (Coordinate2dOr3d): The location to check within tolerance of.
            agent (RetAgent): The agent to check firing statuses of.

        Returns:
            bool: Whether agent has fired within tolerance of the location
        """
        weapon_fired_statuses = [
            status for status in agent.get_statuses() if isinstance(status, WeaponFiredStatus)
        ]
        for status in weapon_fired_statuses:
            if self._weapon_name is None or status.weapon_name == self._weapon_name:
                if status.target_in_range:
                    distance = checker.model.space.get_distance(location, status.target_location)
                    if distance < self._tolerance:
                        return True

        return False


class WeaponFiredNearAgentTrigger(WeaponFiredNearTrigger):
    """Trigger for when a weapon is fired within a given tolerance of an agent.

    The position of the target agent will be determined from the checker agent's perceived world.
    However, this trigger will check for weapons fired regardless of whether the firing agent is in
    the checker's perceived world.
    """

    _agent: RetAgent

    def __init__(
        self,
        agent: RetAgent,
        tolerance: float,
        weapon_name: Optional[str] = None,
        log: bool = True,
        sticky: bool = False,
    ):
        """Initialise a weapon fired trigger.

        Args:
            agent (RetAgent): The target agent.
            tolerance (float): The tolerance to check for weapons fired strictly within.
            weapon_name (Optional[str]): The name of the weapon, if None the trigger will check for
                all weapons. Defaults to None.
            log (bool): whether to log or not. Defaults to True.
            sticky (bool): if true once activated this trigger will remain
                activated. Defaults to True.
        """
        super().__init__(weapon_name=weapon_name, tolerance=tolerance, log=log, sticky=sticky)
        self._agent = agent

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Weapon Fired Near Agent Trigger"

    def _get_location(self, checker: RetAgent) -> Optional[Coordinate2dOr3d]:
        """Get perceived location of the target agent from the checker's perceived world.

        Args:
            checker (RetAgent): The agent doing the checking.

        Returns:
            Optional[Coordinate2dOr3d]: The perceived location of the target agent.
        """
        return checker.perceived_world.get_agent_pos(self._agent.unique_id)

    def get_new_instance(self) -> WeaponFiredNearAgentTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            WeaponFiredNearAgentTrigger: New instance of the trigger
        """
        return WeaponFiredNearAgentTrigger(
            tolerance=self._tolerance,
            agent=self._agent,
            weapon_name=self._weapon_name,
            log=self._log,
            sticky=self._sticky,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return (
            f"Weapon fired within tolerance {self._tolerance} of agent: {self._agent.agent_type} "
            "(Agent ID: {self._agent.unique_id})."
        )


class WeaponFiredNearLocationTrigger(WeaponFiredNearTrigger):
    """Trigger for when a weapon is fired within tolerance of a location.

    This trigger will check for weapons fired regardless of whether the firing agent is in the
    checker's perceived world.
    """

    _location: Coordinate2dOr3d

    def __init__(
        self,
        location: Coordinate2dOr3d,
        tolerance: float,
        weapon_name: Optional[str] = None,
        log: bool = True,
        sticky: bool = False,
    ):
        """Initialise a weapon fired trigger.

        Args:
            location (Coordinate2dOr3s): The target location.
            tolerance (float): The tolerance to check for weapons fired strictly within.
            weapon_name (Optional[str]): The name of the weapon, if None the trigger will check for
                all weapons. Defaults to None.
            log (bool): whether to log or not. Defaults to True.
            sticky (bool): if true once activated this trigger will remain
                activated. Defaults to True.
        """
        super().__init__(weapon_name=weapon_name, tolerance=tolerance, log=log, sticky=sticky)
        self._location = location

    def __str__(self):
        """Output a human readable name for the trigger.

        Returns:
            string: brief description of the trigger
        """
        return "Weapon Fired Near Location Trigger"

    def _get_location(self, checker: RetAgent) -> Optional[Coordinate2dOr3d]:
        """Get target location.

        Args:
            checker (RetAgent): The agent doing the checking.

        Returns:
            Optional[Coordinate2dOr3d]: The target location.
        """
        return self._location

    def get_new_instance(self) -> WeaponFiredNearLocationTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            WeaponFiredNearLocationTrigger: New instance of the trigger
        """
        return WeaponFiredNearLocationTrigger(
            tolerance=self._tolerance,
            location=self._location,
            weapon_name=self._weapon_name,
            log=self._log,
            sticky=self._sticky,
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return f"Weapon fired within tolerance {self._tolerance} of location {self._location}."
