"""Countermeasure."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mesa_ret.space.clutter.cluttermodifiers.sphere import SphereClutterModifier
from mesa_ret.template import Template

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from typing import Optional

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.agents.agentfilter import AgentFilter
    from mesa_ret.space.clutter.clutter import ClutterModifier


class Countermeasure(ABC, Template):
    """An abstract class representing a countermeasure."""

    clutter_modifier: Optional[ClutterModifier] = None
    agent_filter: Optional[AgentFilter]

    def __init__(self, persist_beyond_deployer: bool = False, agent_filter: AgentFilter = None):
        """Create a countermeasure.

        Args:
            persist_beyond_deployer (bool): Flag indicating if the countermearure
                survives after the agent that deployed it is killed. Defaults to False
            agent_filter (AgentFilter): An optional filter which determines which agents
                the countermeasure applies to. Defaults to None.
        """
        self.deployed = False
        self._killed = False
        self.persist_beyond_deployer = persist_beyond_deployer
        self.agent_filter = agent_filter

    @property
    def killed(self) -> bool:
        """Get whether the countermeasure has been killed or not.

        Returns:
            bool: Whether or not the countermeasure has been killed
        """
        return self._killed

    def deploy(self, deployer: RetAgent) -> None:
        """Deploy the countermeasure.

        Args:
            deployer (RetAgent): The agent doing the deploying
        """
        if not self.deployed:
            self.deployed = True
            deployer.model.space.clutter_field.modify(self._get_clutter_modifier(deployer))

    def kill(self, time_killed: datetime) -> None:
        """Kill the countermeasure.

        Args:
            time_killed (datetime): Time the countermeasure is killed
        """
        if not self.killed:
            self._killed = True
            if self.clutter_modifier is not None:
                self.clutter_modifier.expiry_time = time_killed

    def _get_clutter_modifier(self, deployer: RetAgent) -> ClutterModifier:
        """Return the clutter modifier for this countermeasure.

        Args:
            deployer (RetAgent): The agent deploying the countermeasure

        Returns:
            ClutterModifier: The clutter modifier for this countermeasure
        """
        if self.clutter_modifier is None:
            self.clutter_modifier = self._create_clutter_modifier(deployer)
        return self.clutter_modifier

    @abstractmethod
    def _create_clutter_modifier(self, deployer: RetAgent) -> ClutterModifier:  # pragma: no cover
        """Create the clutter modifier for this countermeasure.

        Args:
            deployer (RetAgent): The agent deploying the countermeasure

        Returns:
            ClutterModifier: The clutter modifier for this countermeasure
        """
        pass

    @abstractmethod
    def get_new_instance(self) -> Countermeasure:  # pragma: no cover
        """Return a new instance of a functionally identical countermeasure.

        This abstract method is only defined here, rather than just inherited from
        the Template class, to provide more specific type hints.

        Returns:
            Countermeasure: New instance of the countermeasure
        """
        pass


class SphereCountermeasure(Countermeasure):
    """A countermeasure with a spherical clutter modifier."""

    def __init__(
        self,
        clutter_value: float,
        clutter_radius: float,
        life_time: timedelta,
        persist_beyond_deployer: bool = False,
        agent_filter: AgentFilter = None,
    ):
        """Create a spherical countermeasure.

        Args:
            clutter_value (float): The value of the clutter
            clutter_radius (float): The radius of the sphere
            life_time (timedelta): The life time of the countermeasure
            persist_beyond_deployer (bool): Flag indicating if the countermearure
                survives after the agent that deployed it is killed. Defaults to False
            agent_filter (AgentFilter): An optional filter which determines which agents
                the countermeasure applies to. Defaults to None.
        """
        super().__init__(persist_beyond_deployer, agent_filter=agent_filter)
        self._clutter_value = clutter_value
        self._clutter_radius = clutter_radius
        self._life_time = life_time

    def _create_clutter_modifier(self, deployer: RetAgent) -> ClutterModifier:
        """Return the clutter modifier for this countermeasure.

        Args:
            deployer (RetAgent): The agent deploying the countermeasure

        Returns:
            ClutterModifier: The clutter modifier for this countermeasure
        """
        expiry_time = deployer.model.get_time() + self._life_time
        return SphereClutterModifier(
            self._clutter_value,
            deployer.pos,
            self._clutter_radius,
            expiry_time=expiry_time,
            agent_filter=self.agent_filter,
        )

    def get_new_instance(self) -> SphereCountermeasure:
        """Return a new instance of a functionally identical countermeasure.

        Returns:
            SphereCountermeasure: New instance of the countermeasure
        """
        return SphereCountermeasure(
            clutter_value=self._clutter_value,
            clutter_radius=self._clutter_radius,
            life_time=self._life_time,
            persist_beyond_deployer=self.persist_beyond_deployer,
            agent_filter=self.agent_filter,
        )
