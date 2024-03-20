"""Geometric utilities to be used in RET."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

    from ret.types import Vector2d


def _heading_to_clockwise_angle(vector: Vector2d) -> float:
    """Convert a 2D heading vector to an angle.

    Args:
        vector (Vector2d): the heading to be converted into an angle.

    Returns:
        angle (float): angle counter-clockwise from y-axis in radians
    """
    if vector[0] == vector[1] == 0:
        return 0.0

    # args ordered as x, y for clockwise from positive-y convention
    return math.atan2(vector[0], vector[1])


def _heading_to_counter_clockwise_angle(vector: Vector2d) -> float:
    """Convert a 2D heading vector to an angle.

    Args:
        vector (Vector2d): the heading to be converted into an angle.

    Returns:
        angle (float): angle counter clockwise from y-axis in radians
    """
    return -_heading_to_clockwise_angle(vector)


def _heading_to_clockwise_angle_degrees(vector: Vector2d) -> float:
    """Convert a 2D heading vector to an angle.

    Args:
        vector (Vector2d): 2-Dimensional vector to translate

    Returns:
        angle (float): angle clockwise from y-axis in degrees
    """
    return math.degrees(_heading_to_clockwise_angle(vector))


def rotate(point: Tuple[float, float], angle: float) -> Tuple[float, float]:
    """Rotate a point counter-clockwise by a given angle.

        The angle should be given in radians. The rotation is about the origin.

    Args:
        point (Tuple[float, float]): The point to be rotated
        angle (float): The angle in radians through which to rotate.

    Returns:
        Tuple[float, float]: The new position following rotation.
    """
    px, py = point

    qx = math.cos(angle) * px - math.sin(angle) * py
    qy = math.sin(angle) * px + math.cos(angle) * py
    return (qx, qy)
