"""Coordinate system tools."""


from geopy.distance import GeodesicDistance, distance
from geopy.point import Point


class LatLongConverter:
    """Utility for converting latitude longitude points to xy and vice versa.

    The Lat Long generator must be initialised with a Latitude Longitude reference
    point, which represents X-Y in the grid.

    This utility provides an approximate method for representing latitude-longitude
    coordinates on an XY grid. It's validity is where the reference point, or any
    coordinates on the grid, are close to either the north or south poles.
    """

    _ref: Point

    def __init__(self, reference: tuple[float, float]) -> None:
        """Create a new LatLongGenerator.

        Args:
            reference (tuple[float, float]): Reference point, which corresponds
                to X=0, Y=0
        """
        self._ref = Point(latitude=reference[0], longitude=reference[1])

    def to_lat_long(self, point: tuple[float, float]) -> Point:
        """Convert point to Lat Long.

        Args:
            point (tuple[float, float]): X-Y point

        Returns:
            Point: Lat/Long point
        """
        x_mover = GeodesicDistance(0.001 * point[0])
        x_trans: Point = x_mover.destination(self._ref, bearing=90)  # type: ignore

        longitude: float = x_trans.longitude  # type: ignore

        y_mover = GeodesicDistance(0.001 * point[1])
        y_trans: Point = y_mover.destination(self._ref, bearing=0)  # type: ignore

        latitude: float = y_trans.latitude  # type: ignore

        return Point(latitude=latitude, longitude=longitude)

    def to_xy(self, point: Point) -> tuple[float, float]:
        """Convert point to X-Y.

        Args:
            point (Point): Lat-long point

        Returns:
            tuple[float, float]: X-Y point

        """
        lat_trans = Point(point.latitude, self._ref.longitude)  # type: ignore
        long_trans = Point(self._ref.latitude, point.longitude)  # type: ignore

        lat_m: float = distance(self._ref, lat_trans).meters  # type: ignore
        long_m: float = distance(self._ref, long_trans).meters  # type: ignore

        if point.longitude < self._ref.longitude:
            long_m = -long_m

        if point.latitude < self._ref.latitude:
            lat_m = -lat_m

        return (long_m, lat_m)
