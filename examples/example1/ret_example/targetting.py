"""Targetting for Example model."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ret.behaviours.fire import (
    DefaultHostileTargetResolver,
    FireBehaviour,
    RandomTargetSelector,
)
from ret.sensing.perceivedworld import (
    AirAgents,
    ArmourAgents,
    InfantryAgents,
    Or,
    RandomSelection,
)

if TYPE_CHECKING:
    from typing import Optional

    from ret.agents.agent import RetAgent
    from ret.behaviours.fire import HostileTargetResolver, TargetSelector
    from ret.sensing.perceivedworld import PerceivedAgent
    from ret.types import Coordinate2dOr3d
    from ret.weapons.weapon import Weapon


class RandomTargetSelectorWithAim(RandomTargetSelector):
    """Randomly select an agent and if you can shoot at where it really is."""

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
        locations = []
        for agent in views:
            if agent.unique_id is not None:
                location = next(
                    a for a in detector.model.get_all_agents() if a.unique_id == agent.unique_id
                ).pos
            else:
                location = agent.location
            locations.append(location)
        return locations


class AirTargetResolver(DefaultHostileTargetResolver):
    """Allows only air units to be valid targets."""

    def run(self, detector: RetAgent, views: list[PerceivedAgent]) -> list[PerceivedAgent]:
        """Filter perceived world.

        Args:
            detector (RetAgent): Agent doing the filtering
            views (list[PerceivedAgent]): Unfiltered perceptions

        Returns:
            list[PerceivedAgent]: Up to a single target location
        """
        enemies = super().run(detector=detector, views=views)
        filter = AirAgents()
        views = filter.run(enemies)
        return views


class GroundTargetResolver(DefaultHostileTargetResolver):
    """Allows only air units to be valid targets."""

    def run(self, detector: RetAgent, views: list[PerceivedAgent]) -> list[Coordinate2dOr3d]:
        """Filter perceived world.

        Args:
            detector (RetAgent): Agent doing the filtering
            views (list[PerceivedAgent]): Unfiltered perceptions

        Returns:
            list[Coordinate2dOr3d]: Up to a single target location
        """
        enemies = super().run(detector=detector, views=views)
        filter = Or([ArmourAgents(), InfantryAgents()])
        views = filter.run(enemies)
        return [v.location for v in views]


class DirectFireBehaviour(FireBehaviour):
    """Specific direct fire behaviour which fires at the most up to date position available."""

    def __init__(
        self,
        aim_error: float,
        hostile_agent_resolver: Optional[HostileTargetResolver] = None,
        target_selector: Optional[TargetSelector] = None,
        log: bool = True,
    ):
        """Initialise direct fire behaviour.

        Args:
            aim_error (float): The error in meters applied to aiming with this behaviour.
                If the firing agent has perfect aim (and uses only the p_kill to
                determine how effective the shot is) then set this to zero. Otherwise
                a random error will be applied to any aim point chosen based on a normal
                distribution with a mean of 0 and this as the standard deviation)
            hostile_agent_resolver (Optional[HostileTargetResolver]): Supply this class
                to override the default behaviour for identifying which components of
                the worldview are considered to be candidates to fire upon. Defaults to
                None, in which case the DefaultHostileTargetResolver is used.
            target_selector (Optional[TargetSelector]): Supply this class to override
                the default behaviour for picking targets to fire upon from the range of
                valid targets. Defaults to None, in which case RandomTargetSelector is
                used.
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True
        """
        super().__init__(
            hostile_agent_resolver=hostile_agent_resolver,
            target_selector=target_selector,
            log=log,
            average_angle_inaccuracy=aim_error,
        )

    def _step(
        self,
        firer: RetAgent,
        rounds: int,
        weapon: Weapon,
        location: Optional[Coordinate2dOr3d] = None,
    ) -> None:
        """Do one time steps worth of fire behaviour and log.

        Just like the parent fire behaviour except if no location is provided the locations
        are chosen based on the agent aiming first to get an up to date target location and
        have an aiming error is one is provided.

        Args:
            firer (RetAgent): The agent doing the firing
            rounds (int): The number of rounds to fire
            weapon (Weapon): The weapon to fire with.
            location (Optional[Coordinate2dOr3d]): The target of the fire. Defaults to
                None
        """
        for _ in range(0, rounds):
            if location:
                self.fire_at_target(firer, weapon, location)
            else:
                worldview = firer.perceived_world.get_perceived_agents()
                hostile_views = self.hostile_agent_resolver.run(firer, worldview)
                locations = self.target_selector.run(firer, hostile_views)
                for location in locations:
                    location = super().random_aiming_location_offset(firer, location)
                    self.fire_at_target(firer, weapon, location)
