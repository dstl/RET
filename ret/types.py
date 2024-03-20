"""ret specific type definitions."""

from typing import Union

Coordinate2d = tuple[float, float]
Coordinate3d = tuple[float, float, float]
Coordinate2dOr3d = Union[Coordinate2d, Coordinate3d]
Coordinate3dBand = tuple[float, float, str]
Coordinate = Union[Coordinate2dOr3d, Coordinate3dBand]
Vector2d = tuple[float, float]
Vector3d = tuple[float, float, float]
Vector = Union[Vector2d, Vector3d]
Color = tuple[int, int, int]
