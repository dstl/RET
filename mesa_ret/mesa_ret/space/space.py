"""Spatial definitions."""

from __future__ import annotations

import math
import warnings
from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np
from mesa.space import ContinuousSpace
from mesa_ret.space.clutter.clutter import ClutterField
from mesa_ret.space.clutter.cluttermodifiers.groundplane import GroundPlaneClutterModifier
from mesa_ret.space.culturemap import CultureMap
from mesa_ret.space.feature import Area
from mesa_ret.space.terrain import Terrain
from mesa_ret.types import Coordinate2d, Coordinate3d
from mesa_ret.utilities import geometric_utilities

if TYPE_CHECKING:
    from typing import Generator, Optional

    from mesa.space import FloatCoordinate
    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.space.culture import Culture
    from mesa_ret.space.feature import Boundary
    from mesa_ret.space.heightband import HeightBand
    from mesa_ret.types import Color, Coordinate, Coordinate2dOr3d, Vector2d


class Precipitation(Enum):
    """Precipitation type."""

    CLEAR = "Clear"
    DRIZZLE_LIGHT = "Light Drizzle"
    DRIZZLE_HEAVY = "Heavy Drizzle"
    RAIN_MODERATE = "Moderate Rain"
    RAIN_HEAVY = "Heavy Rain"
    THUNDERSTORM = "Thunderstorm"
    SNOW = "Snow"


def get_height_band_from_name(name: str, bands: list[HeightBand]) -> HeightBand:
    """Get a band from the list based on the name.

    Args:
        name (str): name of the band
        bands (list[HeightBand]): list of bands to search

    Raises:
        ValueError: if there is no matching band or more than one

    Returns:
        HeightBand: The band with the matching name
    """
    selected_bands = [b for b in bands if b.name == name]

    if len(selected_bands) == 0:
        raise ValueError(f'Destination band with name "{name}" does not exist')
    elif len(selected_bands) > 1:
        raise ValueError(f'More than one band with name "{name}" exists')
    else:
        return selected_bands[0]


class ContinuousSpaceWithTerrainAndCulture2d(ContinuousSpace):
    """Continuous space that can represent terrain and cultures in 2d.

    Assumes that all agents are point objects, and have a pos property storing
    their position as an (x, y) tuple. This class uses a numpy array internally
    to store agent objects, to speed up neighborhood lookups.
    """

    def __init__(
        self,
        x_max: float,
        y_max: float,
        x_min: float = 0,
        y_min: float = 0,
        terrain_image_path: Optional[str] = None,
        height_black: float = 0,
        height_white: float = 100,
        culture_image_path: Optional[str] = None,
        culture_dictionary: Optional[dict[Color, Culture]] = None,
        clutter_background_level: float = 0,
        ground_clutter_value: Optional[float] = None,
        ground_clutter_height: Optional[float] = None,
        features: Optional[list[Boundary]] = None,
        relative_humidity: float = 0.75,
        precipitation: Precipitation = Precipitation.CLEAR,
        dust: float = 0.0,
        meteorological_visibility_range: float = 320.0,
        ambient_temperature: float = 0.0,
        sky_background_reflectivity: float = 0.122,
    ) -> None:
        """Create a new continuous space.

        Args:
            x_max (float): Maximum x coordinate for the space.
            y_max (float): Maximum y coordinate for the space.
            x_min (float): Minimum x coordinate for the space. Defaults to 0.
            y_min (float): Minimum y coordinate for the space. Defaults to 0.
            terrain_image_path (Optional[str]): The path to a terrain image, assumed to
                be an 8-bit greyscale png. If not supplied, entire space is assumed to
                be at 0m. Defaults to None.
            height_black (float): The height (in m) of the color black in the terrain
                image. Defaults to 0.
            height_white (float): The height (in m) of the color white in the terrain
                image. Defaults to 100.
            culture_image_path (Optional[str]): The path to a culture image, assumed to
                be an 8-bit color png. If not supplied, culture is undefined everywhere.
                Defaults to None.
            culture_dictionary (Optional[dict[Color, Culture]]): A dictionary of culture
                keyed by RGB Color, used as a lookup for the culture image, all colors
                in the image must be in the dictionary or an exception will be thrown.
                Defaults to None.
            clutter_background_level (float): The background clutter level for the
                clutter field. Defaults to 0.
            ground_clutter_value (Optional[float]): The clutter value for the ground, if
                not supplied no additional clutter is added for the ground.
                Defaults to None.
            ground_clutter_height (Optional[float]): The height above the terrain at
                which the ground clutter applies, must be supplied if
                ground_clutter_value is otherwise throws an exception. Defaults to None.
            features (Optional[list[Boundary]]): List of areas and boundaries present in the space.
                Area extends Boundary so areas can also be treated as boundaries. Therefore, all
                features will be stored on the space as boundaries but all area features will also
                be stored separately as areas. Defaults to None.
            relative_humidity (float): The relative humidity of the environment, between 0 and 1,
                exclusive. Default = 0.7.
            precipitation (Precipitation): The precipitation enum category for the environment.
                Default = Precipitation.CLEAR
            dust (float): The atmospheric dust level (g/m3), must be greater than or equal to 0.
                Default = 0.0.
            meteorological_visibility_range (float): The meteorological visibility (km).
                Default = 16.0.
            ambient_temperature (float): The ambient temperature (degrees C). Default = 17.0
            sky_background_reflectivity (float): The background reflectivity of the sky, greater
                than 0 and less than or equal to 1. Default 0.122.

        Raises:
            TypeError: If ground_clutter_value is supplied but ground_clutter_height is not.
            ValueError: Parameters defined outside allowable ranges
        """
        super().__init__(x_max, y_max, False, x_min, y_min)

        self._terrain = Terrain(
            x_max=x_max,
            y_max=y_max,
            x_min=x_min,
            y_min=y_min,
            image_path=terrain_image_path,
            height_black=height_black,
            height_white=height_white,
        )

        self._culture_map = CultureMap(
            x_max=x_max,
            y_max=y_max,
            x_min=x_min,
            y_min=y_min,
            image_path=culture_image_path,
            culture_dictionary=culture_dictionary,
        )
        self.clutter_field = ClutterField(clutter_background_level)
        if ground_clutter_value is not None:
            if ground_clutter_height is None:
                raise TypeError(
                    "Must provide ground clutter height if providing ground clutter value"
                )
            self.clutter_field.modify(
                GroundPlaneClutterModifier(ground_clutter_value, self, ground_clutter_height)
            )

        self.areas: dict[str, Area] = {}
        if features is not None:
            areas = [area for area in features if isinstance(area, Area)]
            names = [area.name for area in areas]
            if len(names) != len(set(names)):
                raise ValueError("All provided areas must have unique names")
            self.areas = {area.name: area for area in areas}

        # All areas can also be treated as boundaries so will be added to the boundaries list
        self.boundaries: dict[str, Boundary] = {}
        if features is not None:
            names = [boundary.name for boundary in features]
            if len(names) != len(set(names)):
                raise ValueError("All provided boundaries must have unique names")
            self.boundaries = {boundary.name: boundary for boundary in features}

        self._set_environmental_parameters(
            relative_humidity=relative_humidity,
            precipitation=precipitation,
            dust=dust,
            meteorological_visibility_range=meteorological_visibility_range,
            ambient_temperature=ambient_temperature,
            sky_background_reflectivity=sky_background_reflectivity,
        )

    def _set_environmental_parameters(
        self,
        relative_humidity: float,
        precipitation: Precipitation,
        dust: float,
        meteorological_visibility_range: float,
        ambient_temperature: float,
        sky_background_reflectivity: float,
    ):
        """Set environmental parameters.

        Args:
            relative_humidity (float): The relative humidity of the environment, between 0 and 1,
                exclusive.
            precipitation (Precipitation): The precipitation enum category for the environment.
            dust (float): The atmospheric dust level (g/m3), must be greater than or equal to 0.
            meteorological_visibility_range (float): The meteorological visibility (km).
            ambient_temperature (float): The ambient temperature (degrees C).
            sky_background_reflectivity (float): The background reflectivity of the sky.

        Raises:
            ValueError: If relative humidity is outside of the range (0,1).
            ValueError: If atmospheric dust is < 0.
            ValueError: If meteorological visibility range is < 0.
            ValueError: If sky background reflectivity is less than or equal to 0 or greater than 1.
        """
        if relative_humidity <= 0 or relative_humidity >= 1:
            raise ValueError(
                f"Relative humidity ({relative_humidity}) " + "must be between 0 and 1, exclusive."
            )

        if dust < 0:
            raise ValueError(f"Atmospheric dust ({dust} g/m3) cannot be negative.")

        if meteorological_visibility_range < 0:
            raise ValueError(
                f"Meteorological visibility range ({meteorological_visibility_range} m) "
                + "cannot be negative."
            )

        self._relative_humidity = relative_humidity
        self._precipitation = precipitation
        self._dust = dust
        self._meteorological_visibility_range = meteorological_visibility_range
        self._ambient_temperature = ambient_temperature

        if sky_background_reflectivity <= 0 or sky_background_reflectivity > 1:
            raise ValueError(
                "Reflectivity of sky must be greater than 0 and less than or equal to 1."
            )

        self.sky_background_reflectivity = sky_background_reflectivity

    def torus_adj(self, pos: Coordinate2d) -> FloatCoordinate:
        """Adjust coordinates to handle torus looping.

        Overridden here to provide type checking.

        Args:
            pos (Coordinate2d): Coordinate tuple to convert.

        Returns:
            FloatCoordinate: Adjusted coordinate.
        """
        return super().torus_adj(pos)

    def get_terrain_height(self, pos: Coordinate2d) -> float:
        """Get the height of the terrain at a given coordinate.

        Args:
            pos (Coordinate2d): Position to query the hight of.

        Returns:
            float: The height of the terrain at the given position.
                Returns 0 if no height map is defined.
        """
        pos = self.torus_adj(pos)
        return self._terrain.get_height(pos)

    def get_terrain_gradient(self, pos: Coordinate2d) -> Vector2d:
        """Get the gradient of the terrain at a given coordinate.

        Args:
            pos (Coordinate2d): Position to query the gradient of.

        Returns:
            Vector2d: The gradient of the terrain at the given position.
                Returns (0, 0) if no height map is defined.
        """
        pos = self.torus_adj(pos)
        return self._terrain.get_gradient(pos)

    def get_terrain_gradient_along_vec(self, pos: Coordinate2d, vec: Vector2d) -> float:
        """Get the gradient of the terrain at a given coordinate along a given vector.

        Args:
            pos (Coordinate2d): Position to query the gradient of.
            vec (Vector2d): Vector to query the gradient along.

        Returns:
            float: The gradient of the terrain at the given position along the given
                vector. Returns (0, 0) if no height map is defined.
        """
        pos = self.torus_adj(pos)
        return self._terrain.get_gradient_along_vec(pos, vec)

    def get_culture(self, pos: Coordinate2d) -> Culture:
        """Return culture at a given coordinate.

        Args:
            pos (Coordinate2d): tuple of x and y position.

        Returns:
            Culture: Culture at coordinate, returns default_culture() if culture map is
                not provided.
        """
        pos = self.torus_adj(pos)
        return self._culture_map.get_culture(pos)

    def get_culture_temperature(self, pos: Coordinate2d) -> float:
        """Return temperature of culture at a given coordinate.

        Args:
            pos (Coordinate2d): tuple of x and y position.

        Returns:
            float: Temperature of Culture at coordinate, returns the ambient temperature
                of the space if Culture temperature is not provided.
        """
        culture = self.get_culture(pos)
        if culture.temperature is not None:
            return culture.temperature
        return self._ambient_temperature

    @staticmethod
    def get_distance_in_xy_plane(coordinate_1: Coordinate, coordinate_2: Coordinate) -> float:
        """Get distance between two coordinates in the x-y plane.

        Args:
            coordinate_1 (Coordinate): First coordinate.
            coordinate_2 (Coordinate): Second coordinate.

        Returns:
            float: Distance between coordinate_1 and coordinate_2 in x-y plane.
        """
        return math.sqrt(
            (coordinate_1[0] - coordinate_2[0]) ** 2 + (coordinate_1[1] - coordinate_2[1]) ** 2
        )

    @staticmethod
    def get_clockwise_heading_in_degrees_in_xy_plane(
        coordinate_1: Coordinate, coordinate_2: Coordinate
    ) -> float:
        """Get the clockwise heading in degrees from one coordinate to another, in the x-y plane.

        Args:
            coordinate_1 (Coordinate): First coordinate.
            coordinate_2 (Coordinate): Second coordinate.

        Returns:
            float: Heading from coordinate_1 to coordinate_2 in x-y plane, clockwise in degrees.
        """
        heading: Vector2d = (
            (coordinate_2[0] - coordinate_1[0]),
            (coordinate_2[1] - coordinate_1[1]),
        )
        return geometric_utilities._heading_to_clockise_angle_degrees(heading)

    def get_coordinate_2d(self, in_coordinate: Coordinate) -> Coordinate2d:
        """Get the XY coordinate from a (potentially) 3D coordinate.

        Args:
            in_coordinate (Coordinate): The coordinate to project

        Raises:
            TypeError: If the input coordinate is not 2D or 3D

        Returns:
            Coordinate2d: The XY projection of the input coordinate
        """
        if hasattr(in_coordinate, "__len__") and (
            len(in_coordinate) == 2 or len(in_coordinate) == 3
        ):
            return (in_coordinate[0], in_coordinate[1])
        else:
            raise TypeError("Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand")

    def get_coordinate_in_correct_dimension(
        self, in_coordinate: Coordinate, bands: list[HeightBand] = None
    ) -> Coordinate2dOr3d:
        """Return input coordinate in the number of dimensions required for this space.

        This should be overridden in subclasses as appropriate.

        Args:
            in_coordinate (Coordinate): The coordinate to convert
            bands (list[HeightBand], optional): Bands if needing to convert a
                Coordinate3dBand. Defaults to None.

        Returns:
            Coordinate2dOr3d: Coordinate in correct number of dimensions
        """
        return self.get_coordinate_2d(in_coordinate)

    def get_cultures(self) -> set[Culture]:
        """Return a set of cultures defined in the space.

        Returns:
            Set[Culture]: A set of cultures or an empty set if none.
        """
        return self._culture_map.cultures

    def get_relative_humidity(self) -> float:
        """Return the environmental relative humidity.

        Returns:
            float: The relative humidity, between 0 and 1, exclusive
        """
        return self._relative_humidity

    def get_precipitation(self) -> Precipitation:
        """Return the precipitation enum category for the environment.

        Returns:
            Precipitation: The precpitation enum.
        """
        return self._precipitation

    def get_atmospheric_dust(self) -> float:
        """Return the atmospheric dust level of the environment.

        Returns:
            float: The atmospheric dust level (g/m3).
        """
        return self._dust

    def get_meteorological_visibility_range(self) -> float:
        """Return the meteorological visibility range of the environment.

        Returns:
            float: The meteorological visibility range in km.
        """
        return self._meteorological_visibility_range

    def get_ambient_temperature(self) -> float:
        """Return the ambient temperature of the environment.

        Returns:
            float: The ambient temperature in degrees C.
        """
        return self._ambient_temperature


class ContinuousSpaceWithTerrainAndCulture3d(ContinuousSpaceWithTerrainAndCulture2d):
    """Continuous space that can represent terrain and cultures in 3d.

    Assumes that all agents are point objects, and have a pos property storing
    their position as an (x, y, z) tuple. This class uses a numpy array internally
    to store agent objects, to speed up neighborhood lookups.

    """

    def __init__(
        self,
        x_max: float,
        y_max: float,
        x_min: float = 0,
        y_min: float = 0,
        terrain_image_path: Optional[str] = None,
        height_black: float = 0,
        height_white: float = 100,
        culture_image_path: Optional[str] = None,
        culture_dictionary: dict[Color, Culture] = None,
        clutter_background_level: float = 0,
        ground_clutter_value: Optional[float] = None,
        ground_clutter_height: Optional[float] = None,
        features: Optional[list[Boundary]] = None,
        relative_humidity: float = 0.75,
        precipitation: Precipitation = Precipitation.CLEAR,
        dust: float = 0.0,
        meteorological_visibility_range: float = 320.0,
        ambient_temperature: float = 0.0,
    ) -> None:
        """Create a new continuous space.

        Args:
            x_max (float): Maximum x coordinate for the space.
            y_max (float): Maximum y coordinate for the space.
            x_min (float): Minimum x coordinate for the space. Defaults to 0.
            y_min (float): Minimum y coordinate for the space. Defaults to 0.
            terrain_image_path (Optional[str]): The path to a terrain image, assumed to
                be an 8-bit greyscale png. If not supplied, entire space is assumed to
                be at 0m. Defaults to None.
            height_black (float): The height (in m) of the color black in the terrain
                image. Defaults to 0.
            height_white (float): The height (in m) of the color white in the terrain
                image. Defaults to 100.
            culture_image_path (Optional[str]): The path to a culture image, assumed to
                be an 8-bit color png. If not supplied, culture is undefined everywhere.
                Defaults to None.
            culture_dictionary (Optional[dict[Color, Culture]]): A dictionary of culture
                keyed by RGB Color, used as a lookup for the culture image, all colors
                in the image must be in the dictionary or an exception will be thrown.
                Defaults to None.
            clutter_background_level (float): The background clutter level for the
                clutter field. Defaults to 0.
            ground_clutter_value (Optional[float]): The clutter value for the ground, if
                not supplied no additional clutter is added for the ground. Defaults to
                None.
            ground_clutter_height (Optional[optional]): The height above the terrain at
                which the ground clutter applies, must be supplied if
                ground_clutter_value is otherwise throws an exception. Defaults to None.
            features (Optional[list[Boundary]]): List of areas and boundaries present in the space.
                Defaults to None.
            relative_humidity (float): The relative humidity of the environment, between 0 and 1,
                exclusive. Default = 0.7.
            precipitation (Precipitation): The precipitation enum category for the environment.
                Default = Precipitation.CLEAR
            dust (float): The atmospheric dust level (g/m3), must be greater than or equal to 0.
                Default = 0.0.
            meteorological_visibility_range (float): The meteorological visibility (km).
                Default = 16.0.
            ambient_temperature (float): The ambient temperature (degrees C). Default = 17.0
        """
        super().__init__(
            x_max=x_max,
            y_max=y_max,
            x_min=x_min,
            y_min=y_min,
            terrain_image_path=terrain_image_path,
            height_black=height_black,
            height_white=height_white,
            culture_image_path=culture_image_path,
            culture_dictionary=culture_dictionary,
            clutter_background_level=clutter_background_level,
            ground_clutter_value=ground_clutter_value,
            ground_clutter_height=ground_clutter_height,
            features=features,
            relative_humidity=relative_humidity,
            precipitation=precipitation,
            dust=dust,
            meteorological_visibility_range=meteorological_visibility_range,
            ambient_temperature=ambient_temperature,
        )

    def place_agent(self, agent: RetAgent, pos: Coordinate2dOr3d) -> None:
        """Place a new agent in the space.

        Args:
            agent (RetAgent): Agent object to place.
            pos (Coordinate2dOr3d): Coordinate tuple for where to place the agent.
        """
        super().place_agent(agent, self.get_coordinate_3d(pos))

    def move_agent(self, agent: RetAgent, pos: Coordinate2dOr3d) -> None:
        """Move an agent from its current position to a new position.

        Args:
            agent (RetAgent): The agent object to move.
            pos (Coordinate2dOr3d): Coordinate tuple to move the agent to.
        """
        pos_3d = self.get_coordinate_3d(pos)
        super().move_agent(agent, pos_3d)
        self._agent_points[self._agent_to_index[agent], 2] = pos_3d[2]

    def get_neighbors(
        self,
        pos: Coordinate2dOr3d,
        radius: float,
        include_center: bool = True,
    ) -> list[RetAgent]:
        """Get all objects within a certain radius.

        Args:
            pos (Coordinate2dOr3d): coordinate tuple to center the search at.
            radius (float): Get all the objects within this distance of the center.
            include_center (bool): If True, include an object at the *exact* provided
                coordinates. i.e. if you are searching for the neighbors of a given
                agent, True will include that agent in the results. Defaults to True.

        Returns:
            list[RetAgent]: The agents within the given radius of the given position.
        """
        deltas = self._agent_points - np.array(self.get_coordinate_3d(pos))
        dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2 + deltas[:, 2] ** 2

        (idxs,) = np.where(dists <= radius ** 2)
        neighbors = [self._index_to_agent[x] for x in idxs if include_center or dists[x] > 0]
        return neighbors

    def get_heading(
        self,
        pos_1: Coordinate2dOr3d,
        pos_2: Coordinate2dOr3d,
    ) -> Coordinate3d:
        """Get the heading angle between two points.

        Args:
            pos_1 (Coordinate2dOr3d): Coordinate tuple for point 1.
            pos_2 (Coordinate2dOr3d): Coordinate tuple for point 2.

        Returns:
            Coordinate3d: Heading between the two points.
        """
        one = np.array(self.get_coordinate_3d(pos_1))
        two = np.array(self.get_coordinate_3d(pos_2))
        tmp: Coordinate3d = tuple(two - one)  # type: ignore
        return tmp

    def get_distance(
        self,
        pos_1: Coordinate2dOr3d,
        pos_2: Coordinate2dOr3d,
    ) -> float:
        """Get the distance between two points.

        Args:
            pos_1 (Coordinate2dOr3d): Coordinate tuple for point 1.
            pos_2 (Coordinate2dOr3d): Coordinate tuple for point 2.

        Returns:
            float: Distance between the two points.
        """
        x1, y1, z1 = self.get_coordinate_3d(pos_1)
        x2, y2, z2 = self.get_coordinate_3d(pos_2)

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def torus_adj(self, pos: Coordinate2dOr3d) -> Coordinate2dOr3d:
        """Adjust coordinates to handle torus looping.

        If the coordinate is out-of-bounds and the space is toroidal, return
        the corresponding point within the space. If the space is not toroidal,
        raise an exception.

        Args:
            pos (Coordinate2dOr3d): Coordinate to convert.

        Returns:
            Coordinate2dOr3d: Adjusted coordinate.

        Raises:
            ValueError: Coordinates must be 2D or 3D
        """
        pos_2d: Coordinate2d = super().torus_adj(self.get_coordinate_2d(pos))

        if len(pos) == 2:
            return pos_2d
        elif len(pos) == 3:
            return (pos_2d[0], pos_2d[1], pos[2])  # type: ignore
        else:
            raise ValueError("Coordinate must be 2d or 3d.")

    def check_line_of_sight(
        self,
        pos_a: Coordinate3d,
        pos_b: Coordinate3d,
        sample_distance: float = None,
    ) -> bool:
        """Check whether terrain obstructs Line Of Sight between two positions.

        Args:
            pos_a (Coordinate3d): Coordinate tuple for position A
            pos_b (Coordinate3d): Coordinate tuple for position B
            sample_distance (float): Interval distance at which terrain is sampled for
                line of sight obstruction, defaults to None, which will give a sample
                size equal to half a pixel in the original terrain image.

        Returns:
            bool: true if two positions have unobstructed line of sight, false if
                terrain obstructs line of sight
        """
        return self._terrain.check_line_of_sight(
            pos_a, pos_b, self.get_points_between_coordinates, sample_distance
        )

    def check_culture_penetration(
        self,
        pos_a: Coordinate3d,
        pos_b: Coordinate3d,
        sample_distance: float = None,
    ) -> dict[Culture, float]:
        """Check culture penetrated by a vector between two Coordinate3d positions.

        Args:
            pos_a (Coordinate3d): Coordinate tuple for position A
            pos_b (Coordinate3d): Coordinate tuple for position B
            sample_distance (float): Interval distance at which culture is sampled for
                culture penetration, defaults to None, which will give a sample size
                equal to half a pixel in the original culture image.

        Returns:
            dict[Culture, float]: Dictionary of cultures and distance travelled through
                those cultures
        """
        return self._culture_map.check_culture_penetration(
            pos_a,
            pos_b,
            self.get_points_between_coordinates,
            self._terrain,
            sample_distance,
        )

    def get_coordinate_3d(
        self, in_coordinate: Coordinate, bands: list[HeightBand] = None
    ) -> Coordinate3d:
        """Get the 3D coordinate from a 2D, 3D, or 3D band coordinate.

        Args:
            in_coordinate (Coordinate): The coordinate to convert
            bands (list[HeightBand], optional): Bands if needing to convert a
                Coordinate3dBand. Defaults to None.

        Raises:
            TypeError: If trying to convert a Coordinate3dBand but a list of bands is
                not provided.
            TypeError: If trying to convert anything other than Coordinate2d,
                Coordinate3d and Coordinate3dBand

        Returns:
            Coordinate3d: Resulting 3D coordinate
        """
        in_as_2d = cast(Coordinate2d, in_coordinate)
        in_as_3d = cast(Coordinate3d, in_coordinate)

        hes_len = hasattr(in_coordinate, "__len__")

        if hes_len and len(in_coordinate) == 2:
            # Coordinate2d so assume on terrain
            warnings.warn(
                "2D coordinate was entered in 3D terrain, " + "placing coordinate at terrain level"
            )
            z = self.get_terrain_height(in_as_2d)
        elif hes_len and len(in_coordinate) == 3 and isinstance(in_as_3d[2], (int, float)):
            # Coordinate3d so just return as is
            z = in_as_3d[2]
        elif hes_len and len(in_coordinate) == 3 and isinstance(in_as_3d[2], str):
            # Coordinate3dBand therefore lookup band height
            if bands is None:
                raise TypeError(
                    "When converting a Coordinate3dBand to Coordinate3d you must "
                    + "provide the list of bands"
                )
            band = get_height_band_from_name(in_as_3d[2], bands)
            z = band.get_height(self.get_coordinate_2d(in_coordinate))
        else:
            raise TypeError("Space only supports Coordinate2d, Coordinate3d and Coordinate3dBand")

        return (in_coordinate[0], in_coordinate[1], z)

    def get_coordinate_in_correct_dimension(
        self, in_coordinate: Coordinate, bands: list[HeightBand] = None
    ) -> Coordinate2dOr3d:
        """Return input coordinate in the number of dimensions required for this space.

        This should be overridden in subclasses as necessary.

        Args:
            in_coordinate (Coordinate): The coordinate to convert
            bands (list[HeightBand], optional): Bands if needing to convert a
                Coordinate3dBand. Defaults to None.

        Returns:
            Coordinate2dOr3d: Coordinate in correct number of dimensions
        """
        return self.get_coordinate_3d(in_coordinate, bands)

    def get_points_between_coordinates(
        self,
        point_a: Coordinate3d,
        point_b: Coordinate3d,
        sample_distance: float,
    ) -> Optional[Generator]:
        """Return generator of points between two positions.

        Points are generated with separation distance of sample_distance

        Args:
            point_a (Coordinate3d): Coordinate tuple for position A
            point_b (Coordinate3d): Coordinate tuple for position B
            sample_distance (float): Interval distance at which points are sampled
                between positions A and B.

        Returns:
            Optional[Generator]: Generator of points between positions A and B with
                distance between them equal to sample_distance. Returns None if distance
                between positions A and B is less than or equal to the sample distance.
        """
        a_3d: Coordinate3d = self.get_coordinate_3d(point_a)
        b_3d: Coordinate3d = self.get_coordinate_3d(point_b)

        a_3d_adjusted: Coordinate3d = cast(Coordinate3d, self.torus_adj(a_3d))
        b_3d_adjusted: Coordinate3d = cast(Coordinate3d, self.torus_adj(b_3d))

        pos_a_vec = np.asfarray(a_3d_adjusted)
        pos_b_vec = np.asfarray(b_3d_adjusted)

        vector_ab = pos_b_vec - pos_a_vec
        magnitude_ab = np.linalg.norm(vector_ab)

        if magnitude_ab <= sample_distance:
            return None

        normalised_vector_ab = vector_ab / magnitude_ab
        total_number_of_samples = math.floor(magnitude_ab / sample_distance)

        sample_positions = (
            pos_a_vec + (x * sample_distance * normalised_vector_ab)
            for x in range(total_number_of_samples)
        )

        return sample_positions

    def get_culture_attenuated_distance_between(
        self,
        pos_a: Coordinate3d,
        pos_b: Coordinate3d,
        wavelength: float,
        sampling_distance: Optional[float] = None,
        distance: Optional[float] = None,
    ) -> float:
        """Get the culture attenuated distance between two positions.

        Args:
            pos_a (Coordinate3d): Coordinate tuple for position A
            pos_b (Coordinate3d): Coordinate tuple for position B
            wavelength (float): Signal wavelength
            sampling_distance (Optional[float]): Interval distance at which culture is
                sampled for culture penetration, defaults to None, which will give a sample
                size equal to half a pixel in the original culture image.
            distance (Optional[float]): Optional pre-calculated distance between positions.
                If None, the distance will be calculated in this method. Defaults to None.

        Returns:
            float: The distance between the positions attenuated by culture for the given wavelength
        """
        if distance is None:
            distance = self.get_distance(pos_a, pos_b)

        culture_distance_dict: dict[Culture, float] = self.check_culture_penetration(
            pos_a, pos_b, sampling_distance
        )

        total_distance_through_culture = sum(culture_distance_dict.values())

        total_distance_through_no_culture = distance - total_distance_through_culture

        attenuated_distance = total_distance_through_no_culture

        for culture, culture_distance in culture_distance_dict.items():
            attenuated_distance += culture_distance * culture.get_attenuation_factor(wavelength)
        return attenuated_distance

    def is_agent_above_terrain(self, agent: RetAgent) -> bool:
        """Check whether agent is above the terrain height.

        Args:
            agent (RetAgent): The agent to check.

        Returns:
            bool: Whether the agent is above terrain height.
        """
        terrain_height = self.get_terrain_height(agent.pos)
        pos_3d = self.get_coordinate_3d(agent.pos)
        z = pos_3d[2]
        return z > terrain_height
