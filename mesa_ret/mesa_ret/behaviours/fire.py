"""Fire behaviour."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import AffiliationVisitor
from mesa_ret.behaviours.loggablebehaviour import LoggableBehaviour
from mesa_ret.sensing.perceivedworld import (
    FriendlyAgents,
    HostileAgents,
    PerceivedAgentFilter,
    RandomSelection,
)

if TYPE_CHECKING:
    from typing import Any, Optional, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.sensing.perceivedworld import PerceivedAgent
    from mesa_ret.types import Coordinate2dOr3d
    from mesa_ret.weapons.weapon import Weapon


class HostileTargetResolver(ABC):
    """Abstract base class for a hostile target resolver.

    This class is used to determine a subset of agents from a perceived world that are
    valid targets to be fired upon.
    """

    @abstractmethod
    def run(
        self, detector: RetAgent, views: list[PerceivedAgent]
    ) -> list[PerceivedAgent]:  # pragma: no cover
        """Filter views to remove any perceived agents that are not valid targets.

        Args:
            detector (RetAgent): Agent doing the detection
            views (list[PerceivedAgent]): Unfiltered list of views

        Returns:
            list[PerceivedAgent]: Filtered list of views
        """
        pass


class HostileAgentAffiliationVisitor(AffiliationVisitor):
    """AffiliationVisitor which creates a PerceivedAgentFilter based on affiliation.

    The HostileAgentAffiliationVisitor uses a visitor pattern to determine the correct
    filter to apply to a set of world-views to ensure that an agent will only fire at
    enemy combatants.

    A Unknown or Neutral unknown affiliation will result in no views being returned.

    A Hostile Affiliation will filter views to only consider perceived views with
    Friendly Affiliations as targets.

    A Friendly Affiliation will filter views to only consider perceived views with
    Hostile Affiliations as targets.

    """

    class NoAgents(PerceivedAgentFilter):
        """PerceivedAgentFilter which always returns nothing."""

        def run(self, agents: list[PerceivedAgent]) -> list[PerceivedAgent]:
            """Filter perceived agents, and return nothing.

            Args:
                agents (list[PerceivedAgent]): List of perceived agents

            Returns:
                list[PerceivedAgent]: An empty list
            """
            return []

    enemy: PerceivedAgentFilter

    def visit_unknown(self):
        """Set the filter to always return no agents."""
        self.enemy = self.NoAgents()

    def visit_hostile(self):
        """Set the filter to always return Friendly agents."""
        self.enemy = FriendlyAgents()

    def visit_neutral(self):
        """Set the filter to always return no agents."""
        self.enemy = self.NoAgents()

    def visit_friendly(self):
        """Set the filter to always return Hostile agents."""
        self.enemy = HostileAgents()


class DefaultHostileTargetResolver(HostileTargetResolver):
    """Default Hostile Target Resolver."""

    def run(self, detector: RetAgent, views: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter perceived world.

        Args:
            detector (RetAgent): Agent doing the filtering
            views (list[PerceivedAgent]): Unfiltered views

        Returns:
            list[PerceivedAgent]: Filtered views
        """
        visitor = HostileAgentAffiliationVisitor()
        detector.affiliation.accept(visitor)
        return visitor.enemy.run(views)


class TargetSelector(ABC):
    """Abstract base class for target selection mechanism."""

    @abstractmethod
    def run(
        self, detector: RetAgent, views: list[PerceivedAgent]
    ) -> list[Coordinate2dOr3d]:  # pragma: no cover
        """Filter the perceived world.

        To be overridden by classes which extend the Target Selector

        Args:
            detector (RetAgent): Agent doing the detection
            views (list[PerceivedAgent]): Unfiltered list of Perceived Agents

        Returns:
            list[PerceivedAgent]: Filtered list of agents
        """
        pass


class RandomTargetSelector(TargetSelector):
    """TargetSelector which randomly selects a single target."""

    def run(self, detector: RetAgent, views: list[PerceivedAgent]) -> list[Coordinate2dOr3d]:
        """Filter perceived world to randomly select a single target.

        Args:
            detector (RetAgent): Agent doing the filtering
            views (list[PerceivedAgent]): Unfiltered perceptions

        Returns:
            list[Coordinate2dOr3d]: Up to a single target location
        """
        filter = RandomSelection(random_generator=detector.model.random, number_to_select=1)

        views = filter.run(views)

        return [v.location for v in views]


class FireBehaviour(LoggableBehaviour):
    """A class representing fire behaviour."""

    hostile_agent_resolver: HostileTargetResolver
    target_selector: TargetSelector

    def __init__(
        self,
        hostile_agent_resolver: Optional[HostileTargetResolver] = None,
        target_selector: Optional[TargetSelector] = None,
        average_angle_inaccuracy: float = 0,
        log: bool = True,
    ):
        """Create a fire behaviour, override as necessary in subclasses.

        Args:
            hostile_agent_resolver (Optional[HostileTargetResolver]): Supply this class to override
                the default behaviour for identifying which components of the worldview are
                considered to be candidates to fire upon. Defaults to None, in which case the
                DefaultHostileTargetResolver is used.
            target_selector (Optional[TargetSelector]): Supply this class to override the default
                behaviour for picking targets to fire upon from the range of valid targets. Defaults
                to None, in which case RandomTargetSelector is used.
            log (bool): If True logging occurs for this behaviour, if False it does not. Defaults to
                True.
            average_angle_inaccuracy (float): Maximum angular inaccuracy from firer (given in
                degrees). Must be greater than or equal to zero and less than 180. It is the offset
                in one direction so the total range will be double this value. Defaults to 0.
        """
        super().__init__(log)

        if target_selector is None:
            target_selector = RandomTargetSelector()
        self.target_selector = target_selector

        if hostile_agent_resolver is None:
            hostile_agent_resolver = DefaultHostileTargetResolver()
        self.hostile_agent_resolver = hostile_agent_resolver

        if 0 <= average_angle_inaccuracy < 180:
            self.average_angle_inaccuracy = average_angle_inaccuracy
        else:
            self.average_angle_inaccuracy = 0
            warnings.warn(
                f"maximum angle of inaccuracy for firing behaviour must be within range 0-180."
                f" An angle of {average_angle_inaccuracy} degrees was specified."
                f" An angle of 0 degrees will be used instead."
            )

    def step(
        self,
        firer: RetAgent,
        rounds: int,
        weapon: Weapon,
        location: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d]]] = None,
    ) -> None:
        """Do one time steps worth of fire behaviour and log.

        If no location is provided, the behaviour will determine the targets to fire
        upon, based upon the information contained in the agent's perceived world.

        Args:
            firer (RetAgent): The agent doing the firing
            rounds (int): The number of rounds to fire
            weapon (Weapon): The weapon to fire with.
            location (Optional[Coordinate2dOr3d]): The target of the fire. Defaults to
                None
        """
        self.log(firer, weapon=weapon)
        self._step(firer, rounds, weapon, location)

    def _step(
        self,
        firer: RetAgent,
        rounds: int,
        weapon: Weapon,
        location: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d]]] = None,
    ) -> None:
        """Do one time steps worth of fire behaviour, override in subclasses.

        If no location is provided, the behaviour will determine the targets to fire
        upon, based upon the information contained in the agent's perceived world.

        If firing more than one round, the firer may determine a different target to
        receive each shot, should the method for determining fire recipient be delegated
        to the behaviour.

        Args:
            firer (RetAgent): The agent doing the firing
            rounds (int): The number of rounds to fire
            weapon (Weapon): The weapon to fire with
            location (Optional[Coordinate2dOr3d]): The target of the fire. Defaults to
                None
        """
        if isinstance(location, list):
            if len(location) != rounds:
                warnings.warn(
                    "The number of locations given is not equal to the number of rounds"
                    + "to fire. Will fire one round at each location instead."
                )
            for loc in location:
                loc = self.random_aiming_location_offset(firer, loc)
                self.fire_at_target(firer, weapon, loc)
        else:
            for _ in range(0, rounds):
                if location:
                    location = self.random_aiming_location_offset(firer, location)  # type: ignore
                    self.fire_at_target(firer, weapon, location)
                else:
                    worldview = firer.perceived_world.get_perceived_agents()
                    hostile_views = self.hostile_agent_resolver.run(firer, worldview)
                    locations = self.target_selector.run(firer, hostile_views)
                    for location in locations:
                        location = self.random_aiming_location_offset(firer, location)
                        self.fire_at_target(firer, weapon, location)

    def fire_at_target(self, firer: RetAgent, weapon: Weapon, location: Coordinate2dOr3d) -> None:
        """Fire at a specific target.

        Args:
            firer (RetAgent): The agent doing the firing
            weapon (Weapon): The weapon doing the firing.
            location (Coordinate2dOr3d): The target of the fire.
        """
        in_range = weapon.fire_at_target(firer, location)
        firer.add_fired_status(weapon, location, in_range)

    def random_aiming_location_offset(
        self, firer: RetAgent, location: Coordinate2dOr3d
    ) -> Coordinate2dOr3d:
        """Modify the location with a random offset to simulate operator inaccuracy.

        This method may be overridden by subclasses to represent differing inaccuracies in different
        firing behaviours. It differs from the weapon inaccuracy which may be represented in the
        weapon. The offset is given by the normal distribution with standard deviation of the given
        angle. The result is then clamped to prevent outlier values. The new target location will be
        the same distance away but rotated away from the original location. This is a 'rotational'
        random offset which preserves distance to target. This is opposed to allowing the target
        location to be in front or behind the original target location.

        Args:
            firer (RetAgent): The agent doing the firing.
            location (Coordinate2dOr3d): The original location to be fired upon.

        Returns:
            Coordinate2dOr3d: The new location to be fired upon after the offset.
        """
        if not self.average_angle_inaccuracy:
            return location

        distance = firer.model.space.get_distance_in_xy_plane(firer.pos, location)
        angle = firer.model.space.get_clockwise_heading_in_degrees_in_xy_plane(firer.pos, location)

        offset_angle: float = firer.model.random.normalvariate(0, self.average_angle_inaccuracy)

        offset_angle = self._clamp_value(
            original_value=offset_angle,
            minimum_value=-3 * self.average_angle_inaccuracy,
            maximum_value=3 * self.average_angle_inaccuracy,
        )

        location_x = firer.pos[0] + distance * math.sin(math.radians(angle + offset_angle))
        location_y = firer.pos[1] + distance * math.cos(math.radians(angle + offset_angle))

        if len(location) > 2:
            return (location_x, location_y, location[2])  # type: ignore
        else:
            return (location_x, location_y)

    def _clamp_value(
        self, original_value: float, minimum_value: float, maximum_value: float
    ) -> float:
        """Clamp an input value to be between the minimum and maximum.

        Args:
            original_value (float): Supplied value to be clamped.
            minimum_value (float): The minimum value to be clamped to.
            maximum_value (float): The maximum value to be clamped to.

        Returns:
            float: The new clamped value.
        """
        return max(min(original_value, maximum_value), minimum_value)

    def _get_log_message(self, **kwargs: dict[str, Any]) -> str:
        """Get the log message.

        Args:
            **kwargs(dict[str, Any]): Key word arguments

        Returns:
            str: The log message
        """
        if "weapon" in kwargs:
            weapon: Weapon = kwargs["weapon"]  # type: ignore

            log_message: str = weapon.log_fire_message()
            return log_message
        else:
            return "No weapon"
