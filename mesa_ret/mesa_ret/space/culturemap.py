"""Culture map."""
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, cast

import imageio
import numpy as np
from mesa_ret.space.culture import default_culture
from mesa_ret.space.imageutils import check_image_aspect_ratio, get_image_pixel_size
from mesa_ret.types import Color

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Optional

    from mesa_ret.space.culture import Culture
    from mesa_ret.space.terrain import PointGenerator, Terrain
    from mesa_ret.types import Coordinate2d, Coordinate3d


class LineOfSightException(ValueError):  # pragma: no cover
    """A custom error to raise when there is not line of sight."""

    pass


class CultureMap:
    """Culture map that stores information relating to a space's culture."""

    def __init__(
        self,
        x_max: float,
        y_max: float,
        x_min: float = 0,
        y_min: float = 0,
        image_path: Optional[str] = None,
        culture_dictionary: dict[Color, Culture] = None,
    ) -> None:
        """Create a new culture map.

        Args:
            x_max (float): Maximum x coordinate for the space.
            y_max (float): Maximum y coordinate for the space.
            x_min (float): Minimum x coordinate for the space. Defaults to 0.
            y_min (float): Minimum y coordinate for the space. Defaults to 0.
            image_path (Optional[str]): The path to a culture image, assumed to be an
                8-bit color png. If None, culture is undefined everywhere. Defaults to
                None.
            culture_dictionary (Optional[dict[Color, Culture]]): A dictionary of culture
                keyed by RGB Color, used as a lookup for the culture image, all colors
                in the image must be in the dictionary or an exception will be thrown.
                Defaults to None.

        Raises:
            TypeError: Must provide culture dictionary if providing culture image
        """
        self._culture_map = None

        self._x_min = x_min
        self._y_min = y_min
        self._width = x_max - x_min
        self._height = y_max - y_min

        if culture_dictionary:
            self.cultures = set(culture_dictionary.values())
        else:
            self.cultures = set()

        if image_path is not None:
            if culture_dictionary is None:
                raise TypeError("Must provide culture dictionary if providing culture image")

            initial_image = imageio.imread(image_path, pilmode="RGB")
            image = np.rot90(initial_image, k=-1)
            check_image_aspect_ratio(image.shape, "Culture", (self._width, self._height))

            self.smallest_pixel_size, _, _ = get_image_pixel_size(image, self._width, self._height)

            # This has to be represented as the general type of Sequence[Any] to be recognised as a
            # valid first argument to np.reshape, it is actually Sequence[Culture]
            culture_map: Sequence[Any] = [
                self._get_culture_from_color(culture_dictionary, cast(Color, tuple(color)))
                for column in image
                for color in column
            ]
            self._culture_map = np.reshape(culture_map, image.shape[0:2])

    def _get_culture_from_color(
        self, culture_dictionary: dict[Color, Culture], color: Color
    ) -> Culture:
        """Get culture from the dictionary by color.

        Args:
            culture_dictionary (dict[Color, Culture]): A dictionary of culture keyed by
                RGB Color.
            color (Color): Tuple representing the RGB color.

        Raises:
            KeyError: If color cannot be found in culture_dictionary

        Returns:
            Culture: Culture found by color
        """
        if color in culture_dictionary:
            return culture_dictionary[color]
        else:
            raise KeyError(
                f"Culture image has color ({color}) that is not defined in the culture "
                + "dictionary."
            )

    def get_culture(self, pos: Coordinate2d) -> Culture:
        """Return culture at a given coordinate.

        Args:
            pos (Coordinate2d): Tuple of x and y position.

        Returns:
            Culture: Culture at coordinate, or default culture is no map is provided.
        """
        if self._culture_map is not None:
            culture_map_shape = self._culture_map.shape
            x_index = math.floor(((pos[0] - self._x_min) / self._width) * culture_map_shape[0])
            y_index = math.floor(((pos[1] - self._y_min) / self._height) * culture_map_shape[1])
            culture: Culture = self._culture_map[x_index, y_index]
            return culture
        else:
            return default_culture()

    def check_culture_penetration(
        self,
        pos_a: Coordinate3d,
        pos_b: Coordinate3d,
        point_generator: PointGenerator,
        terrain: Terrain,
        sample_distance: float = None,
    ) -> dict[Culture, float]:
        """Check culture penetrated by a vector between two Coordinate3d positions.

        Args:
            pos_a (Coordinate3d): Coordinate tuple for position A
            pos_b (Coordinate3d): Coordinate tuple for position B
            point_generator (PointGenerator): Function that takes in two positions and a sample
                distance and returns a generator of the points between the two positions
                seperated by the sample distance
            terrain (Terrain): The terrain that the culture is on top of
            sample_distance (float): Interval distance at which culture is sampled for
                culture penetration, defaults to None, which will give a sample size
                equal to half a pixel in the original culture image

        Raises:
            ValueError: Line of sight cannot be found between position A and position B

        Returns:
            dict[Culture, float]: Dictionary of cultures and distance travelled through
                those cultures
        """  # noqa: E501
        if self._culture_map is None:
            return {}

        if sample_distance is None:
            sample_distance = self.smallest_pixel_size / 2

        if not terrain.check_line_of_sight(pos_a, pos_b, point_generator, sample_distance):
            raise LineOfSightException("Input positions do not have line of sight.")
        else:

            sample_positions = point_generator(pos_a, pos_b, sample_distance)

            if sample_positions is None:
                return {}

            culture_at_positions = [
                self.get_culture(position)
                for position in sample_positions
                if position[2] < (terrain.get_height(position) + self.get_culture(position).height)
            ]

            output_dict: dict[Culture, float] = dict(Counter(culture_at_positions))

            output_dict.update((x, y * sample_distance) for x, y in output_dict.items())

            return output_dict
