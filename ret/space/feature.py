"""Features."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from random import Random

    from ret.types import Coordinate2d, Coordinate2dOr3d


class Boundary(ABC):
    """An abstract class representing a feature that can be crossed (or entered)."""

    _name: str

    @abstractmethod
    def has_crossed(
        self, pos_1: Coordinate2dOr3d, pos_2: Coordinate2dOr3d
    ) -> bool:  # pragma: no cover
        """Check if moving between two points has crossed (or entered) the feature.

        Args:
            pos_1 (Coordinate2dOr3d): first position
            pos_2 (Coordinate2dOr3d): second position

        Returns:
            bool: True if moving has crossed (or entered) the feature
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Get name.

        Returns:
            str: feature name
        """
        return self._name


class Area(Boundary):
    """An abstract class representing a feature that can contain points."""

    @abstractmethod
    def contains(self, pos: Coordinate2dOr3d) -> bool:  # pragma: no cover
        """Check if a given location is inside the feature.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if inside, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def get_coord_inside(self, random: Random) -> Coordinate2dOr3d:  # pragma: no cover
        """Return a random point inside the area.

        Args:
            random (Random): random number generator

        Returns:
            Coordinate2dOr3d: a random coordinate from inside the area
        """
        raise NotImplementedError

    def has_crossed(self, pos_1: Coordinate2dOr3d, pos_2: Coordinate2dOr3d) -> bool:
        """Check if moving between two points has entered the area.

        Args:
            pos_1 (Coordinate2dOr3d): first position
            pos_2 (Coordinate2dOr3d): second position

        Returns:
            bool: True if moving has entered the feature
        """
        return not self.contains(pos_1) and self.contains(pos_2)


class LineFeature(Boundary):
    """A class representing a line feature."""

    coord_1: Coordinate2d
    coord_2: Coordinate2d

    def __init__(self, coord_1: Coordinate2d, coord_2: Coordinate2d, name: str) -> None:
        """Create a line feature.

        Args:
            coord_1 (Coordinate2d): one end of line
            coord_2 (Coordinate2d): other end of line
            name (str): name of the line
        """
        self.coord_1 = coord_1
        self.coord_2 = coord_2
        self._name = name

    def has_crossed(self, pos_1: Coordinate2dOr3d, pos_2: Coordinate2dOr3d) -> bool:
        """Check if moving between two points has crossed the line.

        See https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

        Args:
            pos_1 (Coordinate2dOr3d): first position
            pos_2 (Coordinate2dOr3d): second position

        Returns:
            bool: True if moving has crossed the line
        """
        p1 = self.coord_1
        q1 = self.coord_2
        p2 = (pos_1[0], pos_1[1])
        q2 = (pos_2[0], pos_2[1])

        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        # Special case, the line touch but don't cross, returning True as we assume
        # the agent cannot land exactly on a boundary
        if (
            ((o1 == 0) and self._colinear_on_segment(p1, p2, q1))
            or ((o2 == 0) and self._colinear_on_segment(p1, q2, q1))
            or ((o3 == 0) and self._colinear_on_segment(p2, p1, q2))
            or ((o4 == 0) and self._colinear_on_segment(p2, q1, q2))
        ):
            return True

        # General case, the line fully cross
        if (o1 != o2) and (o3 != o4):
            return True

        # No intersection
        return False

    def _colinear_on_segment(self, p: Coordinate2d, q: Coordinate2d, r: Coordinate2d) -> bool:
        """Given three colinear points, checks if q lies on pr.

        See https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

        Args:
            p (Coordinate2d): point 1
            q (Coordinate2d): point 2
            r (Coordinate2d): point 3

        Returns:
            bool: True if q lies on pr
        """
        return (
            (q[0] <= max(p[0], r[0]))
            and (q[0] >= min(p[0], r[0]))
            and (q[1] <= max(p[1], r[1]))
            and (q[1] >= min(p[1], r[1]))
        )

    def _orientation(self, p: Coordinate2d, q: Coordinate2d, r: Coordinate2d) -> int:
        """Find the orientation of an ordered triplet.

        See https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

        Args:
            p (Coordinate2d): point 1
            q (Coordinate2d): point 2
            r (Coordinate2d): point 3

        Returns:
            int: -1 if anti-clockwise, 0 if colinear, 1 if clockwise
        """
        val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))

        if val > 0:  # clockwise
            return 1
        elif val < 0:  # anti-clockwise
            return -1
        else:  # colinear
            return 0


class MultiLineFeature(Boundary):
    """A class representing a multi-line boundary."""

    lines: list[LineFeature]

    def __init__(self, coords: list[Coordinate2d], name: str) -> None:
        """Create a multi-line feature.

        Args:
            coords (list[Coordinate2d]): ordered list of points defining the line segments
            name (str): name of the multi-line
        """
        self.lines = [
            LineFeature(coord_1, coord_2, name) for coord_1, coord_2 in zip(coords, coords[1:])
        ]

    def has_crossed(self, pos_1: Coordinate2dOr3d, pos_2: Coordinate2dOr3d) -> bool:
        """Check if moving between two points has crossed the feature.

        Args:
            pos_1 (Coordinate2dOr3d): first position
            pos_2 (Coordinate2dOr3d): second position

        Returns:
            bool: True if moving has crossed the feature
        """
        n_crossings = sum(line.has_crossed(pos_1, pos_2) for line in self.lines)

        # if odd number of crossings then must be on the other side
        return self._is_odd(n_crossings)

    @staticmethod
    def _is_odd(n: int) -> bool:
        """Utility method to check if integer is odd.

        Args:
            n (int): integer to check

        Returns:
            bool: True if odd
        """
        return (n % 2) != 0


class BoxFeature(Area):
    """A class representing a rectangle (2D) or cuboid (3D) feature."""

    min_coord: Coordinate2dOr3d
    max_coord: Coordinate2dOr3d

    def __init__(self, min_coord: Coordinate2dOr3d, max_coord: Coordinate2dOr3d, name: str) -> None:
        """Create a box feature.

        If 2D coordinates are provided the boundary will extend to positive and
        negative infinity in the z dimension (i.e. represent an infinite height cuboid)

        Args:
            min_coord (Coordinate2dOr3d): Coordinate of the corner with lowest x, y, and
                optional z values
            max_coord (Coordinate2dOr3d): Coordinate of the corner with highest x, y and
                optional z values
            name: name of the box feature

        Raises:
            ValueError: If max_coord is not greater than min_coord in all directions
        """
        if any([((h - l) < 0) for h, l in zip(max_coord, min_coord)]):
            raise ValueError(
                f"max_coord {max_coord} must be greater than "
                f"min_coord {min_coord} in all directions"
            )

        self.min_coord = min_coord
        self.max_coord = max_coord
        self._name = name

    def contains(self, pos: Coordinate2dOr3d) -> bool:
        """Check if a given location is inside the box.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if inside the box, False otherwise
        """
        a = all([(p >= c) for p, c in zip(pos, self.min_coord)])
        b = all([(p <= c) for p, c in zip(pos, self.max_coord)])
        return a and b

    def get_coord_inside(self, random: Random) -> Coordinate2dOr3d:
        """Return a uniformly distributed random point inside the box.

        Args:
            random (Random): random number generator

        Returns:
            Coordinate2dOr3d: a uniformly distributed random coordinate from inside the
            box
        """
        coord: Coordinate2dOr3d = tuple(
            random.uniform(low, high) for low, high in zip(self.min_coord, self.max_coord)
        )  # type: ignore

        return coord


class SphereFeature(Area):
    """A class representing a circular (2D) or spherical (3D) feature."""

    center: Coordinate2dOr3d
    radius_squared: float

    def __init__(self, center: Coordinate2dOr3d, radius: float, name: str) -> None:
        """Create a circular or spherical feature.

        If a 2D coordinate is provided the boundary will extend to positive and
        negative infinity in the z dimension (i.e. represent an infinite height
        cylinder)

        Args:
            center (Coordinate2dOr3d): The centre of the circle/sphere
            radius (float): The radius of the circle/sphere
            name (str): name of the sphere feature

        Raises:
            ValueError: radius not positive
        """
        if radius < 0:
            raise ValueError("Radius must be positive")

        self.center = center
        self.radius_squared = radius**2
        self._name = name

    def contains(self, pos: Coordinate2dOr3d) -> bool:
        """Check if a given location is inside the circle/sphere.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if inside the circle/sphere, False otherwise
        """
        distance_squared = sum([(p - c) ** 2 for p, c in zip(pos, self.center)])
        return distance_squared <= self.radius_squared

    def get_coord_inside(self, random: Random) -> Coordinate2dOr3d:
        """Return a random point inside the circle/sphere.

        Args:
            random (Random): random number generator

        Returns:
            Coordinate2dOr3d: a random coordinate from inside the sphere, with higher
            density nearer the centre
        """
        random_vector = tuple(random.uniform(-1, 1) for _ in self.center)

        unit_vector: np.ndarray = np.array(random_vector) / np.linalg.norm(random_vector)

        random_radius = random.uniform(0, math.sqrt(self.radius_squared))

        vector = random_radius * unit_vector

        coord: Coordinate2dOr3d
        coord = tuple(c + v for c, v in zip(self.center, vector))  # type: ignore

        return coord


class LineFeatureWithWidth(LineFeature, Area):
    """A line feature with width."""

    width: float

    def __init__(
        self, coord_1: Coordinate2d, coord_2: Coordinate2d, width: float, name: str
    ) -> None:
        """Create a line feature with width.

        Args:
            coord_1 (Coordinate2d): one end of line
            coord_2 (Coordinate2d): other end of line
            width (float): width of line feature
            name (str): name of the line
        """
        super().__init__(coord_1, coord_2, name)
        self.width = width

    def contains(self, pos: Coordinate2dOr3d) -> bool:
        """Check location is within half the width of the line feature.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if within half the width of the line feature, False otherwise
        """
        pos_2d = (pos[0], pos[1])
        return self._get_distance_from_line(pos_2d) <= self.width / 2

    def _get_distance_from_line(self, pos: Coordinate2d) -> float:
        """Calculate the distance of a point from the line.

        Args:
            pos (Coordinate2d): the point

        Returns:
            float: the distance from the line
        """
        a = np.array(self.coord_1)
        b = np.array(self.coord_2)
        c = np.array(pos)

        ab = b - a
        ac = c - a

        line_length = np.linalg.norm(ab)

        dot_prod = np.dot(ab, ac)
        proportion_along: float = dot_prod / (line_length**2)

        if proportion_along < 0:
            proportion_along = 0.0
        elif proportion_along > 1:
            proportion_along = 1.0

        closest_point = proportion_along * ab  # type: ignore

        distance: float = np.linalg.norm(closest_point - ac)

        return distance

    def get_coord_inside(self, random: Random) -> Coordinate2d:
        """Return a uniformly distributed random point along the length of the line.

        Args:
            random (Random): random number generator

        Returns:
            Coordinate2d: a uniformly distributed random coordinate from along line
        """
        proportion_along = random.random()
        x = self.coord_1[0] + (proportion_along * (self.coord_2[0] - self.coord_1[0]))
        y = self.coord_1[1] + (proportion_along * (self.coord_2[1] - self.coord_1[1]))
        return (x, y)


class MultiLineFeatureWithWidth(LineFeature, Area):
    """A multi-line feature with width."""

    lines: list[LineFeatureWithWidth]

    def __init__(self, coords: list[Coordinate2d], width: float, name: str) -> None:
        """Create a line feature with width.

        Args:
            coords (list[Coordinate2d]): ordered list of points defining the line segments
            width (float): width of line feature
            name (str): name of the line
        """
        self.lines = [
            LineFeatureWithWidth(coord_1, coord_2, width, name)
            for coord_1, coord_2 in zip(coords, coords[1:])
        ]

    def has_crossed(self, pos_1: Coordinate2dOr3d, pos_2: Coordinate2dOr3d) -> bool:
        """Check if moving between two points has crossed the feature.

        Args:
            pos_1 (Coordinate2dOr3d): first position
            pos_2 (Coordinate2dOr3d): second position

        Returns:
            bool: True if moving has crossed the feature
        """
        n_crossings = sum(line.has_crossed(pos_1, pos_2) for line in self.lines)

        # if odd number of crossings then must be on the other side
        return self._is_odd(n_crossings)

    @staticmethod
    def _is_odd(n: int) -> bool:
        """Utility method to check if integer is odd.

        Args:
            n (int): integer to check

        Returns:
            bool: True if odd
        """
        return (n % 2) != 0

    def contains(self, pos: Coordinate2dOr3d) -> bool:
        """Check location is within half the width any of the line segments.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if within half the width of any of the line segments, False otherwise
        """
        return any([line.contains(pos) for line in self.lines])

    def get_coord_inside(self, random: Random) -> Coordinate2d:
        """Return a uniformly distributed random point along the length of a random line segment.

        Args:
            random (Random): random number generator

        Returns:
            Coordinate2d: a uniformly distributed random coordinate from along a randomly chosen
                segment
        """
        return random.choice(self.lines).get_coord_inside(random)


class CompoundAreaFeature(Area):
    """A class representing a compound area feature."""

    areas: list[Area]

    def __init__(self, areas: list[Area], name: str) -> None:
        """Create a compand area feature.

        Args:
            areas (list[Area]): list of the areas that make up the compound area
            name: name of the compound area feature
        """
        self.areas = areas
        self._name = name

    def contains(self, pos: Coordinate2dOr3d) -> bool:
        """Check if a given location is inside the compound area.

        Args:
            pos (Coordinate2dOr3d): The location

        Returns:
            bool: True if inside the area, False otherwise
        """
        return any(area.contains(pos) for area in self.areas)

    def get_coord_inside(self, random: Random) -> Coordinate2dOr3d:
        """Return a random point inside a random area.

        Args:
            random (Random): random number generator

        Returns:
            Coordinate2dOr3d: a random coordinate from inside a random area
        """
        return random.choice(self.areas).get_coord_inside(random)
