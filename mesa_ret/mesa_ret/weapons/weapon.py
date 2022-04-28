"""Code constructs for handling different weapon types.

Weapon is the basic abstract class that all weapons should be extended from.

BasicWeapon contains a simple example implementation of a weapon with minimum and maximum range.
"""
from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TYPE_CHECKING
from warnings import warn

from mesa_ret.weapons.fire_schedule import TimeBasedFireSchedule

if TYPE_CHECKING:
    from typing import Optional, Tuple

    from mesa.space import GridContent
    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.types import Coordinate2dOr3d
    from mesa_ret.weapons.fire_schedule import FireSchedule


class Weapon(ABC):
    """Weapon interface."""

    def __init__(self, name: str):
        """Create a new Weapon.

        Args:
            name (str): Weapon name
        """
        self.name = name

    @abstractmethod
    def try_kill(
        self, firer: RetAgent, target: GridContent, shot_id: Optional[int]
    ):  # pragma: no cover
        """Try and kill the target.

        It is expected that this implementation will be overridden as appropriate to account for
        differing lethality versus particular target types, local cover and separation.

        Args:
            firer (RetAgent): The agent firing the weapon
            target (GridContent): The agent being fired upon
            shot_id (Optional[int]): The ID of the shot being fired, if known
        """
        raise NotImplementedError()

    @abstractmethod
    def create_fire_schedule(
        self, rounds_to_fire: int, time_step: timedelta
    ) -> FireSchedule:  # pragma: no cover
        """Create a fire schedule for the weapon.

        Args:
            rounds_to_fire (int): Number of rounds to fire.
            time_step (timedelta): The length of model time step

        Returns:
            FireSchedule: Weapon's fire schedule
        """
        raise NotImplementedError()

    @abstractmethod
    def get_new_instance(self) -> Weapon:  # pragma: no cover
        """Create a new instance of the weapon.

        Returns:
            Weapon: New instance of the weapon
        """
        raise NotImplementedError()

    @abstractmethod
    def fire_at_target(
        self, firer: RetAgent, location: Coordinate2dOr3d
    ) -> bool:  # pragma: no cover
        """Fire at a target.

        Args:
            firer (RetAgent): Agent doing the firing
            location (Coordinate2dOr3d): Location to fire at

        Returns:
            bool: Whether location is in range
        """
        raise NotImplementedError()

    @abstractmethod
    def log_fire_message(self) -> str:  # pragma: no cover
        """Return string message to include in firing log.

        Returns:
            str: log message
        """
        raise NotImplementedError()


class BasicWeapon(Weapon):
    """Simple representation of a weapon, that can be expanded where required."""

    def __init__(
        self,
        name: str,
        radius: float,
        time_between_rounds: timedelta,
        time_before_first_shot: timedelta,
        kill_probability_per_round: float,
        min_range: Optional[float] = None,
        max_range: Optional[float] = None,
    ):
        """Create a new Weapon.

        Args:
            name (str): Name of the weapon
            radius (float): Radius of targets that are hit where firing at a specific location.
            time_between_rounds (timedelta): time between shots are fired
            time_before_first_shot (timedelta): Time before first shot is fired
            kill_probability_per_round (float): Probability of killing a target
            min_range (Optional[float]): Minimum range that the weapon can fire at. If None, there
                is no minimum range. Defaults to None
            max_range (Optional[float]): Maximum range that the weapon can fire at. If None, there
                is no maximum range. Defaults to None

        Raises:
            ValueError: Invalid specification of input data:
                - Negative value for time between rounds
                - Negative time before first shot
                - Invalid kill probability
                - Invalid min range
                - Invalid max range
                - Incompatible min and max range
        """
        super().__init__(name=name)

        if radius <= 0.0:
            raise ValueError("Weapon kill radius must be a positive value")

        if time_before_first_shot < timedelta(seconds=0):
            raise ValueError("Time before first shot must be a non-negative timedelta")

        if time_between_rounds <= timedelta(seconds=0):
            raise ValueError("Time between rounds must be a positive value")

        if kill_probability_per_round < 0.0 or kill_probability_per_round > 1.0:
            raise ValueError("Kill probability per round must be in the range [0, 1]")

        if min_range is not None and max_range is not None and min_range > max_range:
            raise ValueError("Minimum range cannot be greater than maximum range")

        if min_range is not None and min_range < 0:
            raise ValueError("Minimum range must be a non-negative value")

        if max_range is not None and max_range <= 0:
            raise ValueError("Maximum range must be a positive value")

        self._radius = radius
        self._time_between_rounds = time_between_rounds
        self._time_before_first_shot = time_before_first_shot
        self._kill_probability_per_round = kill_probability_per_round
        self.rounds_fired = 0
        self._min_range = min_range
        self._max_range = max_range

    def fire_at_target(self, firer: RetAgent, location: Coordinate2dOr3d) -> bool:
        """Fire at a target.

        Args:
            firer (RetAgent): Agent doing the firing
            location (Coordinate2dOr3d): Location to fire at

        Returns:
            bool: Whether location is in range
        """
        shot_id = firer.model.get_next_shot_id()
        self.rounds_fired += 1

        if not self.is_shot_in_range(firer, location):
            return False

        location = self.random_shot_location_offset(firer, location)

        targets = self.get_content_under_fire(firer=firer, location=location)

        if targets:
            for target in targets:
                self.try_kill(firer=firer, target=target, shot_id=shot_id)

        firer.model.logger.log_shot_fired(firer, targets, location=location, shot_id=shot_id)

        return True

    def is_shot_in_range(self, firer: RetAgent, location: Coordinate2dOr3d) -> bool:
        """Determine whether a shot is in or our of range.

        Args:
            firer (RetAgent): The agent doing the firing.
            location (Coordinate2dOr3d): The location being fired upon.

        Returns:
            (bool): True if the location is in range, false otherwise
        """
        if self._min_range is None and self._max_range is None:
            return True

        rng = firer.model.space.get_distance(firer.pos, location)  # type: ignore

        if self._min_range:
            if rng < self._min_range:
                warn(
                    f"'{firer.name}' attempting to fire '{self.name}' weapon "
                    + "at target closer than minimum range."
                    + f" Firer position = {firer.pos}, "
                    + f"Target Location = {location}, "
                    + f"Min range = {self._min_range}"
                )
                return False

        if self._max_range:
            if rng > self._max_range:
                warn(
                    f"'{firer.name}' attempting to fire '{self.name}' weapon "
                    + "at target beyond maximum range."
                    + f" Firer position = {firer.pos}, "
                    + f"Target Location = {location}, "
                    + f"Max range = {self._max_range}"
                )
                return False
        return True

    def try_kill(self, firer: RetAgent, target: GridContent, shot_id: Optional[int]):
        """Try and kill the target.

        Args:
            firer (RetAgent): The agent firing the weapon
            target (GridContent): The agent being fired upon
            shot_id (Optional[int]): The ID of the shot being fired, if known

        Raises:
            AttributeError: Target is not a RetAgent, or doesn't have a kill method, and therefore
                it is not possible to determine how to treat the target being hit.
        """
        if firer.model.random.random() < self._kill_probability_per_round:
            try:
                target.kill(killer=firer, shot_id=shot_id)  # type: ignore
            except AttributeError:
                raise AttributeError("Target is not a RetAgent, or doesn't have a 'kill' method.")

    def create_fire_schedule(self, rounds_to_fire: int, time_step: timedelta) -> FireSchedule:
        """Create a fire schedule for the weapon.

        Args:
            rounds_to_fire (int): Number of rounds to fire.
            time_step (timedelta): The length of model time step

        Returns:
            FireSchedule: Weapon's fire schedule
        """
        return TimeBasedFireSchedule(
            rounds=rounds_to_fire,
            time_between_rounds=self._time_between_rounds,
            time_before_first_round=self._time_before_first_shot,
            time_step=time_step,
        )

    def get_new_instance(self) -> Weapon:
        """Create a new instance of the weapon.

        Returns:
            Weapon: New instance of the weapon
        """
        return BasicWeapon(
            name=self.name,
            radius=self._radius,
            time_between_rounds=self._time_between_rounds,
            time_before_first_shot=self._time_before_first_shot,
            kill_probability_per_round=self._kill_probability_per_round,
            min_range=self._min_range,
            max_range=self._max_range,
        )

    def get_content_under_fire(self, firer: RetAgent, location: Coordinate2dOr3d) -> list[RetAgent]:
        """Determine the targets that occupy a location.

        Note that this method uses the the targets that are present at the
        location, rather than those that the agent perceives to be at the
        location.

        It is not possible to fire at one's self.

        Args:
            firer (RetAgent): The agent doing the firing
            location (Coordinate2dOr3d): The target of the firing.

        Returns:
            list[RetAgent]: List of zero or more agents that occupy the target location
        """
        grid_content: list[GridContent] = firer.model.space.get_neighbors(
            location, radius=self._radius, include_center=True
        )

        if firer in grid_content:
            grid_content.remove(firer)

        neighbours: list[RetAgent] = grid_content  # type: ignore

        return neighbours

    def random_shot_location_offset(
        self, firer: RetAgent, location: Coordinate2dOr3d
    ) -> Coordinate2dOr3d:
        """Modify the target location with some form of offset to simulate weapon inaccuracy.

        This does not provide an offset - override in subclass for shot location offset.

        Args:
            firer (RetAgent): The agent doing the firing.
            location (Coordinate2dOr3d): The original location to be fired upon.

        Returns:
            Coordinate2dOr3d: The new location to be fired upon after the offset.
        """
        return location

    def log_fire_message(self) -> str:
        """Return string message to include in firing log.

        Returns:
            str: log message
        """
        return "Weapon: {self.name}, Radius: {self._radius}"


class BasicShortRangedWeapon(BasicWeapon):
    """Simple representation of a short range weapon, that can be expanded where required.

    This differs from a BasicWeapon in that it has inaccuracy built in to it. The inaccuracy is an
    angular offset in the aiming direction (in the x-y plane). This is to represent not being able
    to shoot enemies directly behind your target because of inaccuracy.
    """

    def __init__(
        self,
        name: str,
        radius: float,
        time_between_rounds: timedelta,
        time_before_first_shot: timedelta,
        kill_probability_per_round: float,
        min_range: Optional[float] = None,
        max_range: Optional[float] = None,
        max_angle_inaccuracy: float = 0,
    ):
        """Create a new short range Weapon.

        Args:
            name (str): Name of the weapon.
            radius (float): Radius of targets that are hit where firing at a specific location.
            time_between_rounds (timedelta): time between shots are fired.
            time_before_first_shot (timedelta): Time before first shot is fired.
            kill_probability_per_round (float): Probability of killing a target.
            min_range (Optional[float]): Minimum range that the weapon can fire at. If None, there
                is no minimum range. Defaults to None.
            max_range (Optional[float]): Maximum range that the weapon can fire at. If None, there
                is no maximum range. Defaults to None.
            max_angle_inaccuracy (Optional[float]): Maximum angular inaccuracy resulting from the
                weapon (given in degrees). Defaults to 0.

        Raises:
            ValueError: Invalid specification of input data:
                - Negative value for time between rounds
                - Negative time before first shot
                - Invalid kill probability
                - Invalid min range
                - Invalid max range
                - Incompatible min and max range
        """
        super().__init__(
            name=name,
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=kill_probability_per_round,
            min_range=min_range,
            max_range=max_range,
        )
        if 0 <= max_angle_inaccuracy < 180:
            self._max_angle_inaccuracy = max_angle_inaccuracy
        else:
            self._max_angle_inaccuracy = 0
            warnings.warn(
                f"maximum angle of inaccuracy for weapon '{name}' must be within range 0-180."
                f" An angle of {max_angle_inaccuracy} degrees was specified."
                f" An angle of 0 degrees will be used instead."
            )

    def random_shot_location_offset(
        self, firer: RetAgent, location: Coordinate2dOr3d
    ) -> Coordinate2dOr3d:
        """Modify the target location angle with a random offset to simulate weapon inaccuracy.

        This method may be overridden by subclasses to represent differing inaccuracies in different
        weapon types. It differs from the operator inaccuracy which may be represented in the
        firing behaviour. The offset is up to the specified maximum angle inaccuracy in degrees
        (plus or minus) of the angle of the heading from the firing agent to the target agent. The
        new target location will be the same distance away but rotated away from the original
        location. This is a 'rotational' random offset which preserves distance to target. This is
        opposed to allowing the target location to be in front or behind the original target
        location.

        Args:
            firer (RetAgent): The agent doing the firing.
            location (Coordinate2dOr3d): The original location to be fired upon.

        Returns:
            Coordinate2dOr3d: The new location to be fired upon after the offset.
        """
        if not self._max_angle_inaccuracy:
            return location

        distance = firer.model.space.get_distance_in_xy_plane(firer.pos, location)
        angle = firer.model.space.get_clockwise_heading_in_degrees_in_xy_plane(firer.pos, location)
        offset_angle: float = firer.model.random.uniform(
            -self._max_angle_inaccuracy, self._max_angle_inaccuracy
        )

        location_x = firer.pos[0] + distance * math.sin(math.radians(angle + offset_angle))
        location_y = firer.pos[1] + distance * math.cos(math.radians(angle + offset_angle))

        if len(location) > 2:
            return (location_x, location_y, location[2])  # type: ignore
        else:
            return (location_x, location_y)

    def get_new_instance(self) -> BasicShortRangedWeapon:
        """Create a new instance of the weapon.

        Returns:
            Weapon: New instance of the weapon
        """
        return BasicShortRangedWeapon(
            name=self.name,
            radius=self._radius,
            time_between_rounds=self._time_between_rounds,
            time_before_first_shot=self._time_before_first_shot,
            kill_probability_per_round=self._kill_probability_per_round,
            min_range=self._min_range,
            max_range=self._max_range,
            max_angle_inaccuracy=self._max_angle_inaccuracy,
        )


class BasicLongRangedWeapon(BasicWeapon):
    """Simple representation of a long range weapon, that can be expanded where required.

    This differs from a BasicWeapon in that it has inaccuracy built in to it. The inaccuracy is a
    distance-based offset applied to the location (in the x-y plane). This is to represent weapons
    which have their shots land from above.
    """

    def __init__(
        self,
        name: str,
        radius: float,
        time_between_rounds: timedelta,
        time_before_first_shot: timedelta,
        kill_probability_per_round: float,
        min_range: Optional[float] = None,
        max_range: Optional[float] = None,
        max_percentage_inaccuracy: float = 0,
    ):
        """Create a new long range Weapon.

        Args:
            name (str): Name of the weapon
            radius (float): Radius of targets that are hit where firing at a specific location.
            time_between_rounds (timedelta): time between shots are fired
            time_before_first_shot (timedelta): Time before first shot is fired
            kill_probability_per_round (float): Probability of killing a target
            min_range (Optional[float]): Minimum range that the weapon can fire at. If None, there
                is no minimum range. Defaults to None
            max_range (Optional[float]): Maximum range that the weapon can fire at. If None, there
                is no maximum range. Defaults to None
            max_percentage_inaccuracy (float): Maximum percentage inaccuracy (as a number
                from 0 to 1). Defaults to 0.

        Raises:
            ValueError: Invalid specification of input data:
                - Negative value for time between rounds
                - Negative time before first shot
                - Invalid kill probability
                - Invalid min range
                - Invalid max range
                - Incompatible min and max range
        """
        super().__init__(
            name=name,
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=kill_probability_per_round,
            min_range=min_range,
            max_range=max_range,
        )
        if 0 <= max_percentage_inaccuracy:
            self._max_percentage_inaccuracy = max_percentage_inaccuracy
        else:
            self._max_percentage_inaccuracy = 0
            warnings.warn(
                f"maximum percentage of inaccuracy (given as a decimal) for weapon '{name}'"
                f" must be 0 or greater."
                f" A percentage of {max_percentage_inaccuracy} was specified."
                f" A value of 0% will be used instead."
            )

    def random_shot_location_offset(
        self, firer: RetAgent, location: Coordinate2dOr3d
    ) -> Coordinate2dOr3d:
        """Modify the location with a random offset to simulate operator inaccuracy.

        This method may be overridden by subclasses to represent differing inaccuracies in different
        firing behaviours. It differs from the operator inaccuracy which may be represented in the
        firing behaviour. The offset is up to a given maximum percent (given as a number between 0
        and 1), plus or minus, of the distance to the target in each the x and y direction. This is
        a 'top-down' random offset in two directions (as opposed to modifying the heading between
        the first and second agent).

        Args:
            firer (RetAgent): The agent doing the firing.
            location (Coordinate2dOr3d): The original location to be fired upon.

        Returns:
            Coordinate2dOr3d: The new location to be fired upon after the offset.
        """
        if not self._max_percentage_inaccuracy:
            return location

        distance = firer.model.space.get_distance_in_xy_plane(location, firer.pos)
        offset_x: float = firer.model.random.uniform(
            -self._max_percentage_inaccuracy * distance, self._max_percentage_inaccuracy * distance
        )
        offset_y: float = firer.model.random.uniform(
            -self._max_percentage_inaccuracy * distance, self._max_percentage_inaccuracy * distance
        )

        location_offsets: Tuple[float, ...]
        if len(location) > 2:
            location_offsets = (offset_x, offset_y, 0.0)
        else:
            location_offsets = (offset_x, offset_y)

        return [sum(x) for x in zip(location, location_offsets)]  # type: ignore

    def get_new_instance(self) -> BasicLongRangedWeapon:
        """Create a new instance of the weapon.

        Returns:
            Weapon: New instance of the weapon
        """
        return BasicLongRangedWeapon(
            name=self.name,
            radius=self._radius,
            time_between_rounds=self._time_between_rounds,
            time_before_first_shot=self._time_before_first_shot,
            kill_probability_per_round=self._kill_probability_per_round,
            min_range=self._min_range,
            max_range=self._max_range,
            max_percentage_inaccuracy=self._max_percentage_inaccuracy,
        )
