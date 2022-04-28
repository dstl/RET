"""Tests for the geometric utilities."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mesa_ret.utilities.geometric_utilities as gu
from parameterized import parameterized

if TYPE_CHECKING:
    from typing import Tuple

    from mesa_ret.types import Vector2d


@parameterized.expand(
    [
        [1, 1, 45],
        [-1, 1, -45],
        [-1, -1, -135],
        [1, -1, 135],
        [-math.sqrt(3), 1.0, -60],
        [0, 0, 0],
    ]
)
def test_heading_to_clockise_angle_degrees(vector_x: float, vector_y: float, expected_angle: float):
    """Parametrised unit test of the geometric_utilities._heading_to_clockise_angle_degrees method.

    Args:
        vector_x (float): x component of the input vector.
        vector_y (float): y component of the input vector.
        expected_angle (float): expected returned angle in degrees.
    """
    vector: Vector2d = (vector_x, vector_y)

    assert math.isclose(gu._heading_to_clockise_angle_degrees(vector), expected_angle)


@parameterized.expand(
    [
        [1, 1, (math.pi / 4)],
        [-1, 1, -(math.pi / 4)],
        [-1, -1, -(3 * math.pi / 4)],
        [1, -1, (3 * math.pi / 4)],
        [-math.sqrt(3), 1.0, -(math.pi / 3)],
        [0, 0, 0],
    ]
)
def test_heading_to_angle_clockwise(vector_x: float, vector_y: float, expected_angle: float):
    """Parametrised unit test of the geometric_utilities._heading_to_clockwise_angle method.

    Args:
        vector_x (float): x component of the input vector.
        vector_y (float): y component of the input vector.
        expected_angle (float): expected returned angle in radians.
    """
    vector: Vector2d = (vector_x, vector_y)

    assert math.isclose(gu._heading_to_clockwise_angle(vector), expected_angle)


@parameterized.expand(
    [
        [1, 1, -(math.pi / 4)],
        [-1, 1, (math.pi / 4)],
        [-1, -1, (3 * math.pi / 4)],
        [1, -1, -(3 * math.pi / 4)],
        [-math.sqrt(3), 1.0, (math.pi / 3)],
        [0, 0, 0],
    ]
)
def test_heading_to_angle_counter_clockwise(
    vector_x: float, vector_y: float, expected_angle: float
):
    """Parametrised unit test of the geometric_utilities._heading_to_counter_clockwise_angle method.

    Args:
        vector_x (float): x component of the input vector.
        vector_y (float): y component of the input vector.
        expected_angle (float): expected returned angle in radians.
    """
    vector: Vector2d = (vector_x, vector_y)

    assert math.isclose(gu._heading_to_counter_clockwise_angle(vector), expected_angle)


@parameterized.expand(
    [
        [(1, 1), -(math.pi / 4), (math.sqrt(2), 0)],
        [(-1, 1), (math.pi / 4), (-math.sqrt(2), 0)],
        [(-1, -1), (3 * math.pi / 4), (math.sqrt(2), 0)],
        [(1, -1), -(3 * math.pi / 4), (-math.sqrt(2), 0)],
        [(-math.sqrt(3), 1.0), (2 * math.pi / 3), (0, -2)],
        [(0, 0), 0, (0, 0)],
    ]
)
def test_rotate(
    point: Tuple[float, float], rotation_angle: float, expected_poisition: Tuple[float, float]
):
    """Parametrised unit test of the geometric_utilities.rotate method.

    Args:
        point (Tuple[float, float]): [description]
        rotation_angle (float): [description]
        expected_poisition (Tuple[float, float]): [description]
    """
    assert math.isclose(gu.rotate(point, rotation_angle)[0], expected_poisition[0], abs_tol=0.00001)
    assert math.isclose(gu.rotate(point, rotation_angle)[1], expected_poisition[1], abs_tol=0.00001)
