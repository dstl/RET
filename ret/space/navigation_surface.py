"""Navigation surface for use in astar pathfinding functionality."""
from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from ret.space.space import ContinuousSpaceWithTerrainAndCulture2d
    from ret.types import Coordinate2d, Coordinate2dOr3d, Coordinate3d


class GridNavigationSurface:
    """Grid-based navigation surface."""

    def __init__(self, space: ContinuousSpaceWithTerrainAndCulture2d, resolution: float) -> None:
        """Instantiate GridNavigationSurface.

        Args:
            space (ContinuousSpaceWithTerrainAndCulture2d): space to model navigation surface upon.
            resolution (float): resolution of grid used to model space.
        """
        self.space = space
        self.resolution = resolution
        self._navigation_vertices: list[Coordinate3d] = self._generate_navigation_vertices()
        self._kd_tree: KDTree = KDTree([(i[0], i[1]) for i in self._navigation_vertices])
        self._coordinate_neighbour_dict: dict[
            Coordinate3d, tuple[Coordinate3d, ...]
        ] = self._generate_coordinate_neighbour_dict()

    def get_neighbour_coordinates(self, coordinate: Coordinate3d) -> tuple[Coordinate3d, ...]:
        """Get the neighbour coordinates of a given coordinate.

        Args:
            coordinate (Coordinate3d): coordinate to find neighbours of.

        Returns:
            tuple[Coordinate3d, ...]: tuple of neighbour coordinates.
        """
        return self._coordinate_neighbour_dict[coordinate]

    def _generate_coordinate_neighbour_dict(self) -> dict[Coordinate3d, tuple[Coordinate3d, ...]]:
        """Generate dictionary of coordinates mapped to a tuple of their neighbour coordinates.

        Returns:
            dict[Coordinate3d, tuple[Coordinate3d, ...]]: dictionary of coordinates mapped to
                their neighbour coordinates.
        """
        return_dict: dict[Coordinate3d, tuple[Coordinate3d, ...]] = {}

        for vertex in self._navigation_vertices:
            return_dict[vertex] = self._find_neighbour_coordinates(vertex)

        return return_dict

    def _generate_navigation_vertices(self) -> list[Coordinate3d]:
        """Generate grid of navigation vertices for navigation surface.

        Returns:
            list[Coordinate3d]: list of navigation vertex coordinates.
        """
        xs: list[float] = np.arange(
            self.space.x_min + self.resolution, self.space.x_max, self.resolution, dtype=float
        ).tolist()

        ys: list[float] = np.arange(
            self.space.y_min + self.resolution, self.space.y_max, self.resolution, dtype=float
        ).tolist()

        if xs[-1] == self.space.x_max:
            xs.pop(-1)
        if ys[-1] == self.space.y_max:
            ys.pop(-1)

        x, y = np.meshgrid(xs, ys)

        coords_2d: list[Coordinate2d] = list(zip(x.ravel(), y.ravel()))
        coords_3d: list[Coordinate3d] = [
            (c[0], c[1], self.space.get_terrain_height(c)) for c in coords_2d
        ]

        return coords_3d

    def _find_neighbour_coordinates(self, coordinate: Coordinate3d) -> tuple[Coordinate3d, ...]:
        """Get neighbour coordinates of a coordinate.

        Args:
            coordinate (Coordinate3d): coordinate to get neighbours of.

        Returns:
            tuple[Coordinate3d, ...]: list of neighbour coordinates.
        """
        # p refers to Minkowski p-norm
        # https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions
        closest_neighbours_list_indexes: list[int] = self._kd_tree.query_ball_point(
            x=(coordinate[0], coordinate[1]), r=1.1 * self.resolution, p=1, return_sorted=True
        )

        return_coordinates_list = [
            self._navigation_vertices[i]
            for i in closest_neighbours_list_indexes
            if self._navigation_vertices[i] != coordinate
        ]

        return tuple(return_coordinates_list)

    def get_approx_closest_node_coordinate(self, coordinate: Coordinate2dOr3d) -> Coordinate3d:
        """Get navigation vertex coordinate closest to arbitrary coordinate.

        Args:
            coordinate (Coordinate2dOr3d): coordinate to find closest coordinate to.

        Returns:
            Coordinate3d: coordinate of closest navigation vertex.
        """
        # Slightly larger than (sqrt(2) * resolution) to catch directly diagonal points
        search_radius = 1.42 * self.resolution

        # p refers to Minkowski p-norm
        # https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions
        # p = 2 dictates a circular search area of radius r
        closest_coordinate_list_indexes = self._kd_tree.query_ball_point(
            x=(coordinate[0], coordinate[1]), r=search_radius, p=2, return_sorted=True
        )

        if len(closest_coordinate_list_indexes) == 0:
            raise ValueError("Pathfinder unable to find nearby coordinate on navigation surface.")

        return_coordinate = ((0.0, 0.0, 0.0), math.inf)

        candidate_coordinates = [
            self._navigation_vertices[i] for i in closest_coordinate_list_indexes
        ]

        # return coordinate with smallest distance in xy plane
        for candidate_coordinate in candidate_coordinates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                distance_to_input_coordinate = self.space.get_distance(
                    (candidate_coordinate[0], candidate_coordinate[1]),
                    (coordinate[0], coordinate[1]),
                )
                if distance_to_input_coordinate < return_coordinate[1]:
                    return_coordinate = (candidate_coordinate, distance_to_input_coordinate)

        return return_coordinate[0]
