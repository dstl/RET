"""Tests for non-toroidal space with terrain and culture."""

from mesa_ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from mesa_ret.testing.mocks import MockAgent
from tests.test_space import TEST_AGENTS, TestSpaceNonToroidal
from testsret.test_space.test_space import OUTSIDE_POSITIONS_3D, TEST_AGENTS_3D


class TestSpaceNonToroidalWithTerrainAndCulture(TestSpaceNonToroidal):
    """Repeating all tests in base class with ContinuousSpaceWithTerrainAndCulture2d.

    This performs a set of regression tests to ensure no bugs have been introduced.
    """

    def setUp(self):
        """Create a test space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture2d(70, 20, -30, -30)
        self.agents = []
        for i, pos in enumerate(TEST_AGENTS):
            a = MockAgent(i, None)
            self.agents.append(a)
            self.space.place_agent(a, pos)

    def test_move_agent(self):
        """Test moving the agent updates position, where move is valid."""
        agent = self.agents[0]
        self.space.move_agent(agent, (10, 10))
        assert agent.pos == (10, 10)

        with self.assertRaises(Exception) as e:
            self.space.move_agent(agent, (-40, -40))
        self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))


class TestSpaceNonToroidalWithTerrainAndCulture3d(TestSpaceNonToroidal):
    """Repeating all tests in base class with ContinuousSpaceWithTerrainAndCulture3d.

    This performs a set of regression tests to ensure no bugs have been introduced.
    """

    def setUp(self):
        """Create a test space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(70, 20, -30, -30)
        self.agents = []
        for i, pos in enumerate(TEST_AGENTS_3D):
            a = MockAgent(i, None)
            self.agents.append(a)
            self.space.place_agent(a, pos)

    def test_agent_positions(self):
        """Ensure that the agents are all placed properly."""
        for i, pos in enumerate(TEST_AGENTS_3D):
            a = self.agents[i]
            assert a.pos == pos

    def test_distance_calculations(self):
        """Test toroidal distance calculations."""
        pos_2 = (70, 20, 0)
        pos_3 = (-30, -20, 0)
        assert self.space.get_distance(pos_2, pos_3) == 107.70329614269008

        pos_4 = (70, 20, 20)
        pos_5 = (-30, -20, -20)
        assert self.space.get_distance(pos_4, pos_5) == 114.89125293076057

    def test_heading(self):
        """Test heading calculations."""
        pos_1 = (-30, -30, -30)
        pos_2 = (70, 20, 20)
        self.assertEqual((100, 50, 50), self.space.get_heading(pos_1, pos_2))

        pos_1 = (65, -25, -25)
        pos_2 = (-25, -25, -25)
        self.assertEqual((-90, 0, 0), self.space.get_heading(pos_1, pos_2))

    def test_neighborhood_retrieval(self):
        """Test neighborhood retrieval."""
        neighbors_1 = self.space.get_neighbors((-20, -20, -20), 1)
        assert len(neighbors_1) == 2

        neighbors_2 = self.space.get_neighbors((40, -10, 0), 10)
        assert len(neighbors_2) == 0

        neighbors_3 = self.space.get_neighbors((-30, -30, 0), 10)
        assert len(neighbors_3) == 0

    def test_bounds(self):
        """Test positions outside of boundary."""
        for i, pos in enumerate(OUTSIDE_POSITIONS_3D):
            a = MockAgent(len(self.agents) + i, None)
            with self.assertRaises(Exception) as e:
                self.space.place_agent(a, pos)
            self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))

        a = self.agents[0]
        for pos in OUTSIDE_POSITIONS_3D:
            assert self.space.out_of_bounds((pos[0], pos[1]))
            with self.assertRaises(Exception) as e:
                self.space.move_agent(a, pos)
            self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))

    def test_move_agent(self):
        """Test move agent method."""
        agent = self.agents[0]
        self.space.move_agent(agent, (10, 10, 10))
        assert agent.pos == (10, 10, 10)

        with self.assertRaises(Exception) as e:
            self.space.move_agent(agent, (-40, -40, 0))
        self.assertEqual("Point out of bounds, and space non-toroidal.", str(e.exception))
