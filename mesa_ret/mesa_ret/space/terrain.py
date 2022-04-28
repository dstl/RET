"""Terrain."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator, Optional

import imageio
import numpy as np
from mesa_ret.space.imageutils import check_image_aspect_ratio, get_image_pixel_size
from mesa_ret.types import Coordinate3d
from scipy.interpolate.interpolate import RegularGridInterpolator

if TYPE_CHECKING:

    from mesa_ret.types import Coordinate2d, Vector2d


PointGenerator = Callable[[Coordinate3d, Coordinate3d, float], Optional[Generator]]


class Terrain:
    """Terrain that stores information relating to a space's ground profile."""

    def __init__(
        self,
        x_max: float,
        y_max: float,
        x_min: float = 0,
        y_min: float = 0,
        image_path: Optional[str] = None,
        height_black: float = 0,
        height_white: float = 100,
    ) -> None:
        """Create a new terrain.

        Args:
            x_max (float): Maximum x coordinate for the space.
            y_max (float): Maximum y coordinate for the space.
            x_min (float): Minimum x coordinate for the space. Defaults to 0.
            y_min (float): Minimum y coordinate for the space. Defaults to 0.
            image_path (Optional[str]): The path to a terrain image, assumed to be an
                8-bit greyscale png. If not supplied, entire space is assumed to be at
                0m. Defaults to None.
            height_black (float): The height (in m) of the color black in the terrain
                image. Defaults to 0.
            height_white (float): The height (in m) of the color white in the terrain
                image. Defaults to 100.
        """
        self._height_map = None
        self._x_gradient_map = None
        self._y_gradient_map = None

        x_distance = x_max - x_min
        y_distance = y_max - y_min

        if image_path is not None:
            image = imageio.imread(image_path, pilmode="L")
            image = np.rot90(image, k=-1)
            check_image_aspect_ratio(image.shape, "Terrain", (x_distance, y_distance))

            x = np.linspace(x_min, x_max, num=image.shape[0])
            y = np.linspace(y_min, y_max, num=image.shape[1])
            z = height_black + ((image / ((2 ** 8) - 1)) * (height_white - height_black))

            (
                self.smallest_pixel_size,
                x_factor,
                y_factor,
            ) = get_image_pixel_size(image, x_distance, y_distance)

            gradient = np.gradient(z)
            x_gradient = gradient[0] * x_factor
            y_gradient = gradient[1] * y_factor

            self._height_map = RegularGridInterpolator((x, y), z)
            self._x_gradient_map = RegularGridInterpolator((x, y), x_gradient)
            self._y_gradient_map = RegularGridInterpolator((x, y), y_gradient)

    def get_height(self, pos: Coordinate2d) -> float:
        """Get the height of the terrain at a given coordinate.

        Args:
            pos (Coordinate2d): Position to query the hight of.

        Returns:
            float: The height of the terrain at the given position.
                Returns 0 if no height map is defined.
        """
        if self._height_map is not None:
            height: float = self._height_map([pos[0], pos[1]])[0]
            return height
        else:
            return 0

    def get_gradient(self, pos: Coordinate2d) -> Vector2d:
        """Get the gradient of the terrain at a given coordinate.

        Args:
            pos (Coordinate2d): Position to query the gradient of.

        Returns:
            Vector2d: The gradient of the terrain at the given position.
                Returns (0, 0) if no height map is defined.
        """
        if (self._x_gradient_map is not None) and (self._y_gradient_map is not None):
            x_gradient = self._x_gradient_map(pos)
            y_gradient = self._y_gradient_map(pos)
            return (x_gradient, y_gradient)
        else:
            return (0, 0)

    def get_gradient_along_vec(self, pos: Coordinate2d, vec: Vector2d) -> float:
        """Get the gradient of the terrain at a given coordinate along a given vector.

        Args:
            pos (Coordinate2d): Position to query the gradient of.
            vec (Vector2d): Vector to query the gradient along.

        Returns:
            float: The gradient of the terrain at the given position along the given
                vector. Returns (0, 0) if no height map is defined.
        """
        normal_vec = np.array(vec) / np.linalg.norm(vec)
        gradient: float = self.get_gradient(pos) @ normal_vec
        return gradient

    def check_line_of_sight(
        self,
        pos_a: Coordinate3d,
        pos_b: Coordinate3d,
        point_generator: PointGenerator,
        sample_distance: float = None,
    ) -> bool:
        """Check whether terrain obstructs Line Of Sight between two positions.

        Args:
            pos_a (Coordinate3d): Coordinate tuple for position A
            pos_b (Coordinate3d): Coordinate tuple for position B
            point_generator (PointGenerator): Function
                that takes in two positions and a sample distance and returns a generator of the
                points between the two positions seperated by the sample distance
            sample_distance (float): Interval distance at which terrain is sampled for
                line of sight obstruction, defaults to None, which will give a sample
                size equal to half a pixel in the original terrain image

        Returns:
            bool: true if two positions have unobstructed line of sight, false if
                terrain obstructs line of sight.
        """  # noqa: E501
        if self._height_map is None:
            return True
        else:
            if sample_distance is None:
                sample_distance = self.smallest_pixel_size / 2

            sample_positions = point_generator(pos_a, pos_b, sample_distance)

            if sample_positions is None:
                return True
            else:
                positions_above_terrain = (
                    position[2] >= self.get_height(position) for position in sample_positions
                )

            return all(positions_above_terrain)
