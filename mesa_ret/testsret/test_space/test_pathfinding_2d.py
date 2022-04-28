"""Pathfinding tests in 2d continuous space."""

import math
from datetime import datetime, timedelta
from math import inf, isclose
from pathlib import Path

import pytest
from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.behaviours.move import GroundBasedMoveBehaviour
from mesa_ret.model import RetModel
from mesa_ret.orders.order import Order
from mesa_ret.orders.tasks.move import MoveAlongPathTask
from mesa_ret.orders.triggers.time import TimeTrigger
from mesa_ret.space.navigation_surface import GridNavigationSurface
from mesa_ret.space.pathfinder import RetAStarNode, RetAStarPathfinder
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture2d


class MockPathfindingModel2d(RetModel):
    """Test model for testing pathfinding."""

    def __init__(self):
        """Instantiate TestPathfindingModel."""
        super().__init__(
            start_time=datetime(2020, 1, 1, 0, 0),
            end_time=datetime(2020, 2, 1, 0, 0),
            time_step=timedelta(minutes=1),
            space=MockPathfindingSpace2d(),
        )


class MockPathfindingSpace2d(ContinuousSpaceWithTerrainAndCulture2d):
    """Basic space for pathfinding tests."""

    def __init__(self):
        """Instantiate TestPathfindingSpace."""
        super().__init__(
            x_max=1000,
            y_max=1000,
            terrain_image_path=str(Path(__file__).parent.joinpath("pathfinding_terrain.png")),
        )


class MockGridNavigationSurface2d(GridNavigationSurface):
    """Test grid navigation surface using test space."""

    def __init__(self):
        """Instantiate TestGridNavigationSurface."""
        super().__init__(space=MockPathfindingSpace2d(), resolution=25.0)


class MockGridPathfinder2d(RetAStarPathfinder):
    """Test pathfinder using test navigation surface."""

    def __init__(self):
        """Instantiate TestGridPathfinder."""
        super().__init__(
            navigation_surface=MockGridNavigationSurface2d(),
            gradient_multipliers=[
                ((0.2, inf), 2),
                ((0.1, 0.2), 1.5),
                ((-0.1, 0.1), 1),
                ((-0.2, -0.1), 0.5),
                ((-inf, -0.2), 0.1),
            ],
        )


class MockPathfindingAgent2d(RetAgent):
    """Test agent for testing pathfinding."""

    def __init__(self):
        """Instantiate TestPathfindingAgent."""
        super().__init__(
            model=MockPathfindingModel2d(),
            pos=(25, 25),
            name="Pathfinding Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            behaviours=[
                GroundBasedMoveBehaviour(
                    base_speed=1,
                    gradient_speed_modifiers=[
                        ((-inf, inf), 1),
                    ],
                )
            ],
            orders=[MockPathfindingOrder()],
            pathfinder=MockGridPathfinder2d(),
        )


class MockPathfindingOrder(Order):
    """Test order to move using MoveAlongPathTask."""

    def __init__(self):
        """Instantiate MockPathfindingOrder to (975, 975)."""
        super().__init__(
            trigger=TimeTrigger(time=datetime(2020, 1, 1, 0, 0)),
            task=MoveAlongPathTask(destination=(975, 975), tolerance=1.0),
        )


@pytest.fixture
def pathfinding_agent() -> MockPathfindingAgent2d:
    """Fixture for test pathfinding agent."""
    return MockPathfindingAgent2d()


@pytest.fixture
def nav_surface() -> MockGridNavigationSurface2d:
    """Fixture for test navigation surface."""
    return MockGridNavigationSurface2d()


@pytest.fixture
def pathfinder() -> MockGridPathfinder2d:
    """Fixture for test pathfinder."""
    return MockGridPathfinder2d()


def test_a_star_nodes():
    """Test instantiation and basic properties of AStar nodes."""
    node_1 = RetAStarNode(coordinate=(0, 0, 0), target_coordinate=(1000, 1000, 0))
    node_2 = RetAStarNode(coordinate=(500, 500, 0), target_coordinate=(1000, 1000, 0))
    node_3 = RetAStarNode(coordinate=(0, 0, 0), target_coordinate=(1000, 1000, 0))

    # Test h_cost calculations
    assert node_1.h_cost == 2000
    assert node_2.h_cost == 1000

    # Test equality method
    assert node_1 != node_2
    assert node_1 == node_3

    node_1.g_cost = 0
    node_2.g_cost = 500

    # Test comparison methods
    assert node_1 > node_2
    assert node_2 < node_1

    # Test invalid comparisons
    with pytest.raises(TypeError):
        node_1 > 5  # noqa: B015
    with pytest.raises(TypeError):
        node_1 < "a"  # noqa: B015


def test_vertex_generation(nav_surface: MockGridNavigationSurface2d):
    """Test generation of surface vertex generation."""
    assert len(nav_surface._navigation_vertices) == 1521


def test_approx_closest_neighbours(nav_surface: MockGridNavigationSurface2d):
    """Test get_approx_closest_node_coordinate method."""
    assert nav_surface.get_approx_closest_node_coordinate((49, 49)) == (50, 50, 81.58823529411765)
    assert nav_surface.get_approx_closest_node_coordinate((45, 50)) == (
        50,
        50,
        81.58823529411765,
    )


def test_neighbour_dict(nav_surface: MockGridNavigationSurface2d):
    """Test generation of coordinate neighbour dictionary."""
    for key, val in nav_surface._coordinate_neighbour_dict.items():
        if key[0] == 25 and key[1] == 25:
            assert len(val) == 2
        if key[0] == 50 and key[1] == 50:
            assert len(val) == 4


def test_g_cost_calculation(
    pathfinder: MockGridPathfinder2d, nav_surface: MockGridNavigationSurface2d
):
    """Test correct g_cost is calculated between two nodes."""
    node_1_terrain_height = nav_surface.space.get_terrain_height((25, 25))
    node_2_terrain_height = nav_surface.space.get_terrain_height((250, 25))
    node_3_terrain_height = nav_surface.space.get_terrain_height((25, 250))

    node_1 = RetAStarNode(
        coordinate=(25, 25, node_1_terrain_height), target_coordinate=(500, 25, 0)
    )
    node_2 = RetAStarNode(
        coordinate=(250, 25, node_2_terrain_height), target_coordinate=(500, 25, 0)
    )
    node_3 = RetAStarNode(
        coordinate=(25, 250, node_3_terrain_height), target_coordinate=(500, 25, 0)
    )

    assert (
        pathfinder._calculate_g_cost_between_nodes(node_1, node_2)
        == math.sqrt(225 ** 2 + 49.402205882352945 ** 2) * 0.5 * 1
    )
    assert (
        pathfinder._calculate_g_cost_between_nodes(node_1, node_3)
        == math.sqrt(225 ** 2 + 65.45857843137256 ** 2) * 1.5 * 1
    )


def test_end_to_end_find_path(pathfinder: MockGridPathfinder2d):
    """Test end to end path generation."""
    # Check path 1
    path_1 = pathfinder.find_path(start_location=(25, 25), end_location=(975, 975))
    path_2 = pathfinder.find_path(start_location=(25, 975), end_location=(975, 25))

    # Check that the path begins at the start location and ends at the end location
    assert path_1[0][0:2] == (25, 25)
    assert path_1[-1][0:2] == (975, 975)
    assert path_2[0][0:2] == (25, 975)
    assert path_2[-1][0:2] == (975, 25)

    # Check that length of route is reasonable, and not actively pathing away from the target
    assert len(path_1) <= 100
    assert len(path_2) <= 100

    # Check path that starts and finishes in the same place
    path_3 = pathfinder.find_path(start_location=(25, 25), end_location=(25, 25))
    assert path_3[0][0:2] == (25, 25)
    assert path_3[-1][0:2] == (25, 25)
    assert (len(path_3) - 1) * 25 == 0

    with pytest.raises(ValueError) as e:
        _ = pathfinder.find_path(start_location=(-1000, -1000), end_location=(975, 975))
        assert str(e.value) == "Pathfinder unable to find nearby coordinate on navigation surface."


def test_agent_move_along_path_task(pathfinding_agent: MockPathfindingAgent2d):
    """Test ground based movement along generated paths."""
    model = pathfinding_agent.model
    model.space.place_agent(pathfinding_agent, (pathfinding_agent.pos[0], pathfinding_agent.pos[1]))

    # Check agent is in correct starting position of path
    assert pathfinding_agent.perceived_world.get_agent_pos(pathfinding_agent.unique_id) == (
        25,
        25,
    )

    # Step model 100 times
    for _ in range(0, 100):
        model.step()

    # Check agent has moved to end of path
    agent_pos = pathfinding_agent.perceived_world.get_agent_pos(pathfinding_agent.unique_id)
    assert agent_pos is not None
    assert isclose(agent_pos[0], 975, abs_tol=1,) and isclose(
        agent_pos[1],
        975,
        abs_tol=1,
    )
