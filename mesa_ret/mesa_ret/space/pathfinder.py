"""Classes for astar pathfinding functionality."""
from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from queue import PriorityQueue
from typing import TYPE_CHECKING, cast

import numpy as np
from mesa_ret.types import Coordinate3d

if TYPE_CHECKING:
    from mesa_ret.space.culture import Culture
    from mesa_ret.space.navigation_surface import GridNavigationSurface
    from mesa_ret.types import Coordinate2dOr3d

AStarPath = list[Coordinate3d]


class AStarNode(ABC):
    """Node class that stores cost information for paths that have reached the node."""

    def __init__(
        self,
        coordinate: Coordinate3d,
        target_coordinate: Coordinate3d,
        parent_node: AStarNode = None,
    ) -> None:
        """Instantiate AStarNode.

        Args:
            coordinate (Coordinate3d): Coordinate of node.
            target_coordinate (Coordinate3d): target coordinate of path that has reached this node.
            parent_node (AStarNode, optional): Node that path has come from to reach this node.
                Defaults to None.
        """
        self.coordinate = coordinate
        self.target_coordinate = target_coordinate
        self.parent_node = parent_node
        self._g_cost: float = 0.0
        self.h_cost: float = self._calculate_h_cost()
        self.f_cost: float = 0.0
        self.stale = False

    @property
    def g_cost(self) -> float:
        """Getter for g_cost of node.

        Returns:
            float: g_cost of node.
        """
        return self._g_cost

    @g_cost.setter
    def g_cost(self, value: float):
        """Setter for g_cost of this node.

        Calculates f_cost of the node automatically when g_cost is set.

        Args:
            value (float): value of g_cost.
        """
        self._g_cost = value
        self.f_cost = self._g_cost + self.h_cost

    def __eq__(self, other) -> bool:
        """Equality operator definition.

        Args:
            other (AStarNode): comparison AStarNode.

        Returns:
            bool: True if coordinates of other AStarnode are the same.
        """
        if not isinstance(other, AStarNode):
            return NotImplemented
        return self.coordinate == other.coordinate

    def __lt__(self, other):
        """Less than operator definition.

        Args:
            other (AStarNode): comparison AStarNode.

        Returns:
            bool: True if f_cost of self is less than f_cost of other AStarnode.
        """
        if not isinstance(other, AStarNode):
            return NotImplemented
        return self.f_cost < other.f_cost

    def __gt__(self, other):
        """Greater than operator definition.

        Args:
            other (AStarNode): comparison AStarNode.

        Returns:
            bool: True if f_cost of self is greater than f_cost of other AStarnode.
        """
        if not isinstance(other, AStarNode):
            return NotImplemented
        return self.f_cost > other.f_cost

    @abstractmethod
    def _calculate_h_cost(self) -> float:  # pragma: no cover
        """Calculate Heuristic cost (h_cost) of node.

        Returns:
            float:  h_cost of node.
        """
        pass

    def get_path(self) -> AStarPath:
        """Get path from starting AStarNode to this node.

        Returns:
            AStarPath: list of Coordinate3d values to reach this node.
        """
        path = []
        current = self
        while current:
            path.append(current.coordinate)
            current: AStarNode = current.parent_node  # type: ignore
        path.reverse()
        return path


class RetAStarNode(AStarNode):
    """AStarNode implementing Manhattan distance h_cost."""

    def __init__(
        self,
        coordinate: Coordinate3d,
        target_coordinate: Coordinate3d,
        parent_node: AStarNode = None,
    ):
        """Instantiate AStarNode.

        Args:
            coordinate (Coordinate3d): Coordinate of node.
            target_coordinate (Coordinate3d): target coordinate of path that has reached this node.
            parent_node (AStarNode, optional): Node that path has come from to reach this node.
                Defaults to None.
        """
        super().__init__(coordinate, target_coordinate, parent_node)

    def _calculate_h_cost(self) -> float:
        """Calculate Heuristic cost (h_cost) of node.

        h_cost is calculated as Manhattan distance between self and target coordinate.

        Returns:
            float:  Manhattan distance between self and target coordinate.
        """
        x_distance = abs(self.coordinate[0] - self.target_coordinate[0])
        y_distance = abs(self.coordinate[1] - self.target_coordinate[1])
        return x_distance + y_distance


class AStarPriorityQueue:
    """Wrapper class for python PriorityQueue for astar pathfinding."""

    def __init__(self):
        """Instantiate AStarPriorityQueue."""
        self.priority_queue = PriorityQueue()

    def add_node(self, node: AStarNode):
        """Add node to queue, mark nodes currently in the queue with same coordinates as stale.

        Args:
            node (AStarNode): node to add to queue.
        """
        self.priority_queue.put((node.f_cost, node))
        self._mark_nodes_as_stale(node)

    def get_node(self) -> AStarNode:
        """Get node with lowest f_cost.

        Returns:
            AStarNode: node with lowest f_cost in the priority queue.
        """
        return_tuple: tuple[float, AStarNode] = self.priority_queue.get()
        return return_tuple[1]

    def _mark_nodes_as_stale(self, node: AStarNode):
        """Mark nodes in queue as stale if necessary.

        Check if any nodes with same coordinates as supplied node is in priority queue, if so,
            marks them as stale.

        Args:
            node (AStarNode): node to check against.
        """
        nodes_to_check: list[AStarNode] = [n[1] for n in self.priority_queue.queue if n[1] == node]

        smallest_g_cost = min([n.g_cost for n in nodes_to_check])

        # Gather all nodes with g_cost larger than minimum
        nodes_to_stale = [n for n in nodes_to_check if n.g_cost > smallest_g_cost]

        # Gather all nodes with g_cost equal to minimum
        smallest_g_nodes = [n for n in nodes_to_check if n.g_cost == smallest_g_cost]

        if len(smallest_g_nodes) > 1:
            # Remove one of these nodes
            del smallest_g_nodes[0]
            # Stale the rest
            nodes_to_stale.extend(smallest_g_nodes)

        for node in nodes_to_stale:
            node.stale = True


class GridAStarPathfinder(ABC):
    """Abstract class for a grid-based AStar pathfinder."""

    def __init__(self, navigation_surface: GridNavigationSurface) -> None:
        """Instantiate AStarGridPathfinder.

        Args:
            navigation_surface (GridNavigationSurface): navigation surface to pathfind on.
        """
        self.navigation_surface = navigation_surface

    def find_path(
        self,
        start_location: Coordinate2dOr3d,
        end_location: Coordinate2dOr3d,
    ) -> AStarPath:
        """Find a low-cost path between two coordinates.

        Args:
            start_location (Coordinate2dOr3d): start of path
            end_location (Coordinate2dOr3d): target of path

        Raises:
            Exception: raised if a path cannot be found.

        Returns:
            AStarPath: list of Coordinate2d values to reach the end location.
        """
        start_navigation_vertex_coordinate_3d = (
            self.navigation_surface.get_approx_closest_node_coordinate((start_location))
        )

        end_navigation_vertex_coordinate_3d = (
            self.navigation_surface.get_approx_closest_node_coordinate((end_location))
        )

        start_node = RetAStarNode(
            coordinate=start_navigation_vertex_coordinate_3d,
            target_coordinate=end_navigation_vertex_coordinate_3d,
            parent_node=None,
        )
        start_node.g_cost = 0.0

        open_queue = AStarPriorityQueue()
        closed_list: list[AStarNode] = []

        # Put starting node in open list to begin pathfinding
        open_queue.add_node(start_node)

        while open_queue.priority_queue.not_empty:

            # pop highest priority node from open list (lowest f_cost)
            current_working_node: AStarNode = open_queue.get_node()
            closed_list.append(current_working_node)

            if current_working_node.stale is True:
                continue

            # if the latest node is the end node, return the path
            if current_working_node.coordinate == end_navigation_vertex_coordinate_3d:
                return current_working_node.get_path()

            neighbour_coordinates = self.navigation_surface.get_neighbour_coordinates(
                current_working_node.coordinate
            )

            # for each coordinate in the neighbour coordinates, check if it exists, if not
            # create it and add it to the open_queue
            for coord in neighbour_coordinates:

                neighbour_node = RetAStarNode(
                    coordinate=coord,
                    target_coordinate=end_navigation_vertex_coordinate_3d,
                    parent_node=current_working_node,
                )

                if neighbour_node in closed_list:
                    continue

                neighbour_node.g_cost = (
                    current_working_node.g_cost
                    + self._calculate_g_cost_between_nodes(neighbour_node, current_working_node)
                )

                open_queue.add_node(neighbour_node)

        raise Exception("Path couldn't be found.")

    @abstractmethod
    def _calculate_g_cost_between_nodes(
        self, current_node: AStarNode, neighbour_node: AStarNode
    ) -> float:  # pragma: no cover
        """Calculate g_cost between two nodes, a current working node and neighbour node.

        Args:
            current_node (AStarNode): current node.
            neighbour_node (AStarNode): neighbour node.

        Returns:
            float: g_cost between nodes.
        """
        pass


class RetAStarPathfinder(GridAStarPathfinder):
    """Pathfinder that finds paths on a navigation surface.

    Uses euclidean distance between nodes to calculate g_cost.
    """

    def __init__(
        self,
        navigation_surface: GridNavigationSurface,
        gradient_multipliers: list[tuple[tuple[float, float], float]] = None,
        culture_multipliers: dict[Culture, float] = None,
    ) -> None:
        """Instantiate AStarGridPathfinder.

        Args:
            navigation_surface (GridNavigationSurface): navigation surface to pathfind on.
            gradient_multipliers (list[tuple[tuple[float, float], float]]): gradient multiplier
                for terrain gradient. The g_cost from one node to another is multiplied by this
                factor based on the gradient between them. Defaults to None.
            culture_multipliers (Optional[dict[Culture, float]]): culture multiplier, the g_cost
                from one node to another us multiplied by this factor based on the culture of the
                first node. Defaults to None.
        """
        super().__init__(navigation_surface)
        self._gradient_multipliers = gradient_multipliers
        self._culture_multipliers = culture_multipliers

    def _calculate_g_cost_between_nodes(
        self, current_node: AStarNode, neighbour_node: AStarNode
    ) -> float:
        """Calculate g_cost between two nodes, a current working node and neighbour node.

        Args:
            current_node (AStarNode): current node.
            neighbour_node (AStarNode): neighbour node.

        Returns:
            float: g_cost between nodes.
        """
        gradient_multiplier = 1.0
        culture_multiplier = 1.0

        # Determine culture, get culture modifier
        culture = self.navigation_surface.space.get_culture(
            pos=(current_node.coordinate[0], current_node.coordinate[1])
        )

        if self._culture_multipliers:
            if culture in self._culture_multipliers.keys():
                culture_multiplier = self._culture_multipliers[culture]
            else:
                warnings.warn(f"Pathfinder culture modifier not defined for culture: {culture}")
                culture_multiplier = 1.0

        # Determine gradient, get gradient modifier
        vector_to_neighbour = cast(
            tuple[float, float],
            tuple(np.subtract(neighbour_node.coordinate, current_node.coordinate)),
        )

        gradient = self.navigation_surface.space.get_terrain_gradient_along_vec(
            (neighbour_node.coordinate[0], neighbour_node.coordinate[1]), vector_to_neighbour[0:2]
        )

        if self._gradient_multipliers:
            gradient_multiplier = next(
                g[1] for g in self._gradient_multipliers if (g[0][0] < gradient <= g[0][1])
            )

        # Determine euclidian distance
        euclidian_distance = math.sqrt(
            (current_node.coordinate[0] - neighbour_node.coordinate[0]) ** 2
            + (current_node.coordinate[1] - neighbour_node.coordinate[1]) ** 2
            + (current_node.coordinate[2] - neighbour_node.coordinate[2]) ** 2
        )

        g_cost = euclidian_distance * gradient_multiplier * culture_multiplier

        return g_cost
