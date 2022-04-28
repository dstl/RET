"""Movement behaviour."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np
from mesa.space import FloatCoordinate
from mesa_ret.behaviours.loggablebehaviour import LoggableBehaviour
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture2d
from mesa_ret.types import Coordinate3d, Coordinate3dBand

if TYPE_CHECKING:
    from typing import Any, Optional

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.agents.airagent import AirAgent
    from mesa_ret.space.culture import Culture
    from mesa_ret.space.heightband import HeightBand
    from mesa_ret.types import Coordinate, Coordinate2dOr3d, Vector


class MoveBehaviour(LoggableBehaviour, ABC):
    """An abstract class representing move behaviour."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(self, mover: RetAgent, destination: Coordinate) -> None:
        """Do one time steps worth of the move behaviour and log.

        Args:
            mover (RetAgent): the agent doing the moving
            destination (Coordinate): the target destination
        """
        mover_start_position = mover.pos
        self._step(mover, destination)
        self.log(
            mover,
            move_start_location=mover_start_position,
            move_end_location=mover.pos,
            move_destination=destination,
        )

    def _get_log_message(self, **kwargs: Any) -> str:
        """Get the log message.

        This log message includes the start and end location of the movement along
            with the destination. In the case where the destination was reached in that
            time step, the end location and destination will be the same.

        Args:
            **kwargs(Any): Key word arguments.
                move_start_location(Coordinate2dOr3d): The location of the agent before
                    the move.
                move_destination(Coordinate): The intended destination for the agent's
                    move.
                move_end_location(Coordinate2dOr3d): The location of the agent after the
                    move.

        Returns:
            str: The log message
        """
        start_location = kwargs.get("move_start_location", "NOT PROVIDED")
        destination = kwargs.get("move_destination", "NOT PROVIDED")
        end_location = kwargs.get("move_end_location", "NOT PROVIDED")

        return f"Start: {start_location}; End: {end_location}; Destination: {destination}"

    @abstractmethod
    def _step(self, mover: RetAgent, destination: Coordinate) -> None:  # pragma: no cover
        """Do one time steps worth of the move behaviour, override in subclasses.

        Args:
            mover (RetAgent): the agent doing the moving
            destination (Coordinate): the target destination
        """
        pass

    @staticmethod
    def calculate_unit_direction(direction: Vector) -> np.ndarray:
        """Calculate a vector with magnitude 1 from a given vector.

        Args:
            direction (Vector): Vector

        Returns:
            np.array: Magnitude 1 vector

        Raises:
            ValueError: Cannot determine direction of zero-length vector
        """
        if all([v == 0 for v in direction]):
            raise ValueError("Cannot determine orientation of zero-length vector")
        unit_direction: np.ndarray = np.array(direction) / np.linalg.norm(direction)
        return unit_direction


class MoveInBandBehaviour(MoveBehaviour, ABC):
    """Abstract move in band behaviour."""

    def __init__(self, height_bands: list[HeightBand], log: bool = True):
        """Create a move in band behaviour, override as necessary in subclasses.

        Args:
            height_bands (list[HeightBand]): The height bands this behaviour can operate
                in
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True
        """
        super().__init__(log)
        self.height_bands = height_bands


class GroundBasedMoveBehaviour(MoveBehaviour):
    """Move behaviour for ground based agents."""

    def __init__(
        self,
        base_speed: float,
        gradient_speed_modifiers: list[tuple[tuple[float, float], float]],
        culture_speed_modifiers: Optional[dict[Culture, float]] = None,
        area_speed_modifiers: Optional[dict[str, float]] = None,
        impassable_boundaries: Optional[list[str]] = None,
        log: bool = True,
    ) -> None:
        """Create ground based move behaviour.

        Args:
            base_speed (float): base speed of agents using this behaviour, units moved per second
            gradient_speed_modifiers (list[tuple[tuple[float, float], float]]): gradient modifiers
                for speed, this should take into account speed change and extra distance due to
                slope
            culture_speed_modifiers (Optional[dict[Culture, float]]): culture modifiers for speed.
                Defaults to None.
            area_speed_modifiers (Optional[dict[str, float]]): Area modifiers for speed, when inside
                the named area the modifier will be applied. Defaults to None.
            impassable_boundaries (Optional[list[str]]): list of boundaries that will not be
                crossed. Defaults to None.
            log (bool): If True logging occurs for this behaviour, if False it does not. Defaults
                to True.
        """
        super().__init__(log)
        self.base_speed = base_speed
        self.gradient_speed_modifiers = gradient_speed_modifiers
        self.culture_speed_modifiers = culture_speed_modifiers
        self.area_speed_modifiers = area_speed_modifiers
        self.impassable_boundaries = impassable_boundaries

    def _step(self, mover: RetAgent, destination: Coordinate) -> None:
        """Move one time steps worth of distance in a straight line to the destination.

        This assumes that the gradient is constant at the value where the mover starts
        their movement from. Hence the time step should be small enough so that this is
        appropriate (of the terrain flat enough that this is not an issue).

        Args:
            mover (RetAgent): agent doing the moving
            destination (Coordinate): target destination

        Raises:
            TypeError: This step method cannot be called with band-based coordinates.

        """
        if len(destination) == 3 and isinstance(destination[2], str):  # type: ignore
            msg = f"{type(self)} cannot be called with band-based coordinates."
            raise TypeError(msg)

        perceived_pos = mover.perceived_world.get_agent_pos(mover.unique_id)
        if perceived_pos is None:
            warnings.warn(
                f"Agent {mover.unique_id} cannot find it's own position in it's perceived world so "
                "no move occurred."
            )
            return

        distance_to_destination_2d = (
            ContinuousSpaceWithTerrainAndCulture2d.get_distance_in_xy_plane(
                perceived_pos, destination
            )
        )

        if distance_to_destination_2d > 0 and mover.pos is not None:
            direction_to_destination_2d = (
                destination[0] - perceived_pos[0],
                destination[1] - perceived_pos[1],
            )
            unit_direction_2d = self.calculate_unit_direction(direction_to_destination_2d)
            mover_pos_2d = mover.model.space.get_coordinate_2d(mover.pos)

            gradient = mover.model.space.get_terrain_gradient_along_vec(
                mover_pos_2d, direction_to_destination_2d
            )
            culture = mover.model.space.get_culture(mover_pos_2d)

            speed = (
                self.base_speed
                * self.get_gradient_modifer(gradient)
                * self.get_culture_modifier(culture)
                * self.get_area_modifier(mover)
            )

            distance_moved_2d = speed * mover.model.time_step.total_seconds()
            if distance_moved_2d >= distance_to_destination_2d:
                distance_moved_2d = distance_to_destination_2d

            movement = distance_moved_2d * unit_direction_2d
            new_x = mover.pos[0] + movement[0]
            new_y = mover.pos[1] + movement[1]

            new_pos = mover.model.space.get_coordinate_in_correct_dimension((new_x, new_y))

            if not self.has_crossed_boundary(mover, new_pos):
                mover.model.space.move_agent(mover, cast(FloatCoordinate, new_pos))

    def get_gradient_modifer(self, gradient: float) -> float:
        """Return the gradient speed modifier.

        Args:
            gradient (float): the gradient

        Returns:
            float: the modifer
        """
        modifier = next(
            g[1] for g in self.gradient_speed_modifiers if (g[0][0] < gradient <= g[0][1])
        )
        return modifier

    def get_culture_modifier(self, culture: Culture) -> float:
        """Return the culture speed modifier.

        Args:
            culture (Culture): The culture

        Returns:
            float: the modifer

        Raises:
            KeyError: No modifier for specified culture
        """
        if self.culture_speed_modifiers:
            if culture in self.culture_speed_modifiers.keys():
                return self.culture_speed_modifiers[culture]
            else:
                raise KeyError(
                    "culture movement modifier not specified for culture: " + f"""{culture.name}"""
                )
        else:
            return 1.0

    def get_area_modifier(self, mover: RetAgent) -> float:
        """Return the area speed modifier.

        Args:
            mover (RetAgent): The agent moving

        Returns:
            float: the modifer

        Raises:
            KeyError: No area exists for a specified modifier
        """
        modifer: float = 1
        if self.area_speed_modifiers:
            for area, area_modifer in self.area_speed_modifiers.items():
                if area in mover.model.space.areas.keys():
                    if mover.model.space.areas[area].contains(mover.pos):
                        modifer *= area_modifer
                else:
                    raise KeyError(
                        "area movement modifier specified for an area that does not exist: "
                        + f"""{area}"""
                    )

        return modifer

    def has_crossed_boundary(self, mover: RetAgent, new_pos: Coordinate2dOr3d) -> bool:
        """Check if the agent will cross an impassable boundary.

        Args:
            mover (RetAgent): The agent moving
            new_pos (Coordinate2dOr3d): The position the agent is trying to move to

        Returns:
            bool: true if the agent would cross an impassable boundary

        Raises:
            KeyError: No boundary exists for a specified impassable boundary
        """
        if self.impassable_boundaries is not None:
            for boundary in self.impassable_boundaries:
                if boundary in mover.model.space.boundaries.keys():
                    if mover.model.space.boundaries[boundary].has_crossed(mover.pos, new_pos):
                        return True
                else:
                    raise KeyError(
                        "impassable boundary specified for an boundary that does not exist: "
                        + f"""{boundary}"""
                    )
        return False

    def _get_log_message(self, **kwargs) -> str:
        """Get the log message.

        Args:
            kwargs: Keyword arguments to the function

        Returns:
            str: The log message
        """
        base_message = super()._get_log_message(**kwargs)
        return base_message + "; Base speed: " + str(self.base_speed)


class AircraftMoveBehaviour(MoveInBandBehaviour):
    """Movement behaviour for an aircraft."""

    def __init__(self, base_speed: float, height_bands: list[HeightBand], log: bool = True):
        """Create aircraft move behaviour.

        Args:
            base_speed (float): base speed of agents using this behaviour
            height_bands (list[HeightBand]): the hight bands this behaviour can operate
                in
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(height_bands, log)
        self.base_speed = base_speed

    def _step(self, mover: AirAgent, destination: Coordinate) -> None:
        """Move one time steps worth of distance in a straight line to the destination.

        The mover immediately shifts to the band and then moves within this
        (still considers the straight line distance and speed to give the correct travel
        time)

        Args:
            mover (AirAgent): agent doing the moving
            destination (Coordinate): target destination

        Raises:
            TypeError:  Destination, or interim destination at end of time step
            ValueError: Destination below ground level. this can only occur in 3D space
        """
        if len(destination) != 3 or isinstance(destination[2], float):  # type: ignore
            raise TypeError("Move called with invalid destination")

        dest_3d_band: Coordinate3dBand = cast(Coordinate3dBand, destination)
        band = dest_3d_band[2]

        real_destination = mover.model.space.get_coordinate_in_correct_dimension(
            dest_3d_band, self.height_bands
        )

        distance_to_destination = mover.perceived_world.get_distance(
            mover._real_pos, real_destination
        )

        if distance_to_destination > 0:
            direction_to_destination: Coordinate3d = cast(
                Coordinate3d,
                tuple((d - p) for d, p in zip(real_destination, mover._real_pos)),
            )

            distance_moved = self.base_speed * mover.model.time_step.total_seconds()
            if distance_moved >= distance_to_destination:
                distance_moved = distance_to_destination

            unit_direction = self.calculate_unit_direction(direction_to_destination)
            movement = distance_moved * unit_direction

            new_real_position = cast(Coordinate3d, mover._real_pos + movement)
            new_band_position = (
                new_real_position[0],
                new_real_position[1],
                band,
            )
            new_position_in_band = cast(
                Coordinate3dBand,
                mover.model.space.get_coordinate_in_correct_dimension(
                    new_band_position, self.height_bands
                ),
            )

            if self.is_below_ground(mover, new_position_in_band):
                raise ValueError("Invalid destination: below ground")

            mover._real_pos = new_real_position
            mover.model.space.move_agent(mover, new_position_in_band)

    def is_below_ground(self, mover: RetAgent, destination: Coordinate) -> bool:
        """Check if a destination is below ground level.

        Where model is in 2D space, always assume it is above ground.

        Args:
            mover (RetAgent): agent that is moving to the destination
            destination (Coordinate): Destination

        Returns:
            bool: Destination is/is not below ground
        """
        ground_level = mover.model.space.get_terrain_height(
            mover.model.space.get_coordinate_2d(destination)
        )
        if len(destination) == 2:
            return False

        return destination[2] < ground_level  # type: ignore

    def _get_log_message(self, **kwargs) -> str:
        """Get the log message.

        Args:
            kwargs: Keyword arguments supplied to the function

        Returns:
            str: The log message
        """
        base_message = super()._get_log_message(**kwargs)
        return base_message + "; Base speed: " + str(self.base_speed)
