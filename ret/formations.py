"""A manager to select a formation and apply separation and rotation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import ret.utilities.geometric_utilities as gu

if TYPE_CHECKING:
    from typing import Optional, Tuple

    from ret.types import Vector2d


class Formation(ABC):
    """A formation class to hold formation configuration and handle generating formations."""

    def __init__(
        self,
        separation: float = 1,
        grid_ratio: Optional[Tuple[int, int]] = None,
        heading: Vector2d = (0, 0),
        include_killed: bool = False,
    ):
        """Create a new formation class.

        Args:
            separation (float): The separation between units. This functions
                as a scaling factor for the formation. Defaults to 1.
            grid_ratio (Optional[Tuple[float, float]]): The ratio of width to height for the
                formation. If the ratio is (0, X) then the laydown will be a single column. If the
                ratio is (X, 0) for non-zero X then the laydown will be a single row. (0, 0) will
                be interpreted as a single row. Defaults to None.
            heading (Vector2d): The direction the formation is facing. This functions as a
                rotation factor. Defaults to (0, 0).
            include_killed (bool): Whether to include killed agents in the formation.
                Defaults to False.
        """
        self.separation = separation
        self.grid_ratio = grid_ratio
        self.heading = heading
        self.include_killed = include_killed

    def scale_positions(
        self, relative_position: Tuple[float, float], separation: float
    ) -> Tuple[float, float]:
        """Scale a relative position by a desired separation.

        Args:
            relative_position (Tuple[float, float]): The relative location to be scaled.
            separation (float): The scaling for the separation between agents.

        Returns:
            Tuple[float, float]: The new relative position after scaling.
        """
        p1, p2 = relative_position
        return p1 * float(separation), p2 * float(separation)

    def find_grid_size(self, agent_count: int, ratio: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
        """Given a number of agents and a ratio for the formation, generate a minimum grid size.

        If the ratio is (0,X) then the laydown will be a single row. If the ratio is (X,0) for
        non-zero X then the laydown will be a single column. (0,0) will be interpreted as a single
        row.

        Args:
            agent_count (int): The number of agents to place in formation.
            ratio (Optional[Tuple[int, int]]): The ratio of width to depth of the formation.
                Defaults to a square (1, 1).

        Returns:
            Tuple[int, int]: The smallest grid that adhere to the given ratio and can fit all of the
                agents.

        Raises:
            ValueError: Ratio values must be integer values and are not
        """
        if not isinstance(ratio[0], int) or not isinstance(ratio[1], int):
            raise ValueError(f"Ratio values must be integers ({ratio[0]}, {ratio[1]}).")
        if ratio[0] == 0:
            return (1, agent_count)
        if ratio[1] == 0:
            return (agent_count, 1)

        greatest_common_divisor = math.gcd(ratio[0], ratio[1])
        int_ratio_x = ratio[0] // greatest_common_divisor
        int_ratio_y = ratio[1] // greatest_common_divisor

        scale_factor = math.ceil(math.sqrt(agent_count / (int_ratio_x * int_ratio_y)))

        return (int_ratio_x * scale_factor, int_ratio_y * scale_factor)

    def create_centred_square_formation(self, size: Tuple[int, int]) -> list[Tuple[float, float]]:
        """Return a formation layout centred on the origin.

        This method takes the size of the grid and returns a list of the positions within that grid
        aligned in rows and columns. The separation between positions is 1.

        Args:
            size (Tuple[int, int]): The size of grid (given in agents across by agents deep) needed
                to accommodate all agents.

        Returns:
            list[Tuple[float, float]]: Ordered list of the positions, centred on the origin.
        """
        positions: list[Tuple[float, float]] = []
        for x in range(1, size[0] + 1):
            for y in range(1, size[1] + 1):
                positions.append((x - 0.5 - size[0] / 2, y - 0.5 - size[1] / 2))
        return positions

    def get_formation(self, number_of_agents: int) -> list[Tuple[float, float]]:
        """Get a formation based on the given configuration.

        Args:
            number_of_agents (int): The number of agents to create a formation for.

        Returns:
            list[Tuple[float, float]]: A list of position modifiers to give to the agents creating a
                formation around the central point.
        """
        if self.grid_ratio is not None:
            grid_size = self.find_grid_size(number_of_agents, self.grid_ratio)
        else:
            grid_size = self.find_grid_size(number_of_agents)

        unit_length_grid_formation = self.create_formation(grid_size, number_of_agents)

        scaled_grid_formation: list[Tuple[float, float]] = [
            self.scale_positions(relative_position, self.separation)
            for relative_position in unit_length_grid_formation
        ]

        counter_clockwise_angle = gu._heading_to_counter_clockwise_angle(self.heading)
        directed_formation = [
            (gu.rotate(position, counter_clockwise_angle)) for position in scaled_grid_formation
        ]

        return directed_formation

    @abstractmethod
    def create_formation(
        self, size: Tuple[int, int], agent_count: int
    ) -> list[Tuple[float, float]]:
        """Create a formation.

        Args:
            size (Tuple[int, int]): The size of grid (given in agents across by agents deep) needed
                to accommodate all agents.
            agent_count (int): The number of agents to be put into formation.

        Returns:
            list[Tuple[float, float]]: Ordered list of the positions, moving radially outwards from
                the centre of the formation.
        """
        return []


class SquareFormationRounded(Formation):
    """A formation class to hold formation configuration and handle generating formations.

    The formations generated by this class will fill the smallest satisfying grid radially from the
    centre. This may lead to the corners being 'rounded off'.
    """

    def create_formation(
        self, size: Tuple[int, int], agent_count: int
    ) -> list[Tuple[float, float]]:
        """Return a formation layout based around a central point in rows and columns.

        The formation will follow the given ratio to find the smallest rectangular layout which fits
        all of the agents. Then the agents will fill this formation radially. If agents are
        equidistant from the origin then they will appear starting from the bottom left to the top
        right (upper left before lower right) as this follows the order from the rectangle filling
        before being centred on the origin.
        For example, putting 7 agents in a formation with ratio 3x2 will require a grid of size 6x4.
        These 7 positions will occupy the central 7 spaces::
            +------+
            |XXXXXX|
            |X624XX|
            |X513XX|
            |XX7XXX|
            +------+

        Args:
            size (Tuple[int, int]): The size of grid (given in agents across by agents deep) needed
                to accommodate all agents.
            agent_count (int): The number of agents to be put into formation.

        Returns:
            list[Tuple[float, float]]: Ordered list of the positions, moving radially outwards from
                the centre of the formation.
        """
        positions = self.create_centred_square_formation(size)
        positions.sort(key=lambda a: a[0] ** 2 + a[1] ** 2)
        return positions[0:agent_count]


class SquareFormationFullRows(Formation):
    """A formation class to hold formation configuration and handle generating formations.

    The formations generated by this class will fill rows from the centre.
    """

    def create_formation(
        self, size: Tuple[int, int], agent_count: int
    ) -> list[Tuple[float, float]]:
        """Return a formation layout preferentially filling rows before columns.

        The formation will follow the given ratio to find the smallest rectangular layout which fits
        all of the agents. Then the agents will fill this formation from the central row(s),
        depending on an odd or even number of rows.
        For example, putting 9 agents in a formation with ratio 3x2 will require a grid of size 6x4.
        These 9 positions will occupy the the middle two rows first then the rows above and below::
            +------+
            |XXXXXX|
            |X6248X|
            |95137X|
            |XXXXXX|
            +------+

        Args:
            size (Tuple[int, int]): The size of grid (given in agents across by agents deep) needed
                to accommodate all agents.
            agent_count (int): The number of agents to be put into formation.

        Returns:
            list[Tuple[float, float]]: Ordered list of the positions, filling rows incrementally
                from the central row(s) of the formation.
        """
        positions = self.create_centred_square_formation(size)
        positions.sort(key=lambda a: (abs(a[1]), abs(a[0])))
        return positions[0:agent_count]


class SquareFormationFullColumns(Formation):
    """A formation class to hold formation configuration and handle generating formations.

    The formations generated by this class will fill columns from the centre.
    """

    def create_formation(
        self, size: Tuple[int, int], agent_count: int
    ) -> list[Tuple[float, float]]:
        """Return a formation layout preferentially filling columns before rows.

        The formation will follow the given ratio to find the smallest rectangular layout which fits
        all of the agents. Then the agents will fill this formation from the central column(s),
        depending on an odd or even number of columns.
        For example, putting 9 agents in a formation with ratio 3x2 will require a grid of size 6x4.
        These 9 positions will occupy the first row and some of the second::
            +------+
            |XX68XX|
            |XX24XX|
            |XX13XX|
            |X957XX|
            +------+

        Args:
            size (Tuple[int, int]): The size of grid (given in agents across by agents deep) needed
                to accommodate all agents.
            agent_count (int): The number of agents to be put into formation.

        Returns:
            list[Tuple[float, float]]: Ordered list of the positions, filling columns incrementally
                from the central column(s) of the formation.
        """
        positions = self.create_centred_square_formation(size)
        positions.sort(key=lambda a: (abs(a[0]), abs(a[1])))
        return positions[0:agent_count]
