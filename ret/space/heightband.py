"""Height bands for agents moving in 3d space."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
    from ret.types import Coordinate2d


class HeightBand(ABC):
    """An abstract class representing a height band that agents can move in."""

    name: str
    height: float

    def __init__(self, name: str, height: float) -> None:
        """Initialise a height band.

        Args:
            name (str): The name of the band (used for identification)
            height (float): The height for the band
        """
        self.name = name
        self.height = height

    @abstractmethod
    def get_height(self, pos: Coordinate2d) -> float:  # pragma: no cover
        """Return the height of the band at a given x,y coordinate.

        To be overridden in child classes as necessary.

        Args:
            pos (Coordinate2d): The x,y position to query the height of the band at

        Returns:
            float: the height of the band at the given x,y coordinate
        """
        pass


class AbsoluteHeightBand(HeightBand):
    """A height band at an absolute height."""

    def __init__(self, name: str, height: float) -> None:
        """Initialise a height band.

        Args:
            name (str): The name of the band (used for identification)
            height (float): The height for the band
        """
        super().__init__(name, height)

    def get_height(self, pos: Coordinate2d) -> float:
        """Return the hight of the band.

        Args:
            pos (Coordinate2d): The x,y position to query the height of the band at

        Returns:
            float: the height of the band at the given x,y coordinate
        """
        return self.height


class RelativeHeightBand(HeightBand):
    """A height band that is a fixed height above the underlying terrain."""

    terrain: ContinuousSpaceWithTerrainAndCulture3d

    def __init__(
        self, name: str, height: float, terrain: ContinuousSpaceWithTerrainAndCulture3d
    ) -> None:
        """Initialise a height band.

        Args:
            name (str): The name of the band (used for identification)
            height (float): The height for the band
            terrain (ContinuousSpaceWithTerrainAndCulture3d): The terrain that the band
                is based on
        """
        super().__init__(name, height)
        self.terrain = terrain

    def get_height(self, pos: Coordinate2d) -> float:
        """Return the hight of the band.

        Args:
            pos (Coordinate2d): The x,y position to query the height of the band at

        Returns:
            float: the height of the band at the given x,y coordinate
        """
        return self.terrain.get_terrain_height(pos) + self.height
