"""Tests for space with agent mapping, terrain and culture."""
from mesa_ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from mesa_ret.testing.mocks import MockAgent
from tests.test_space import REMOVAL_TEST_AGENTS, TestSpaceAgentMapping
from testsret.test_space.test_space import REMOVAL_TEST_AGENTS_3D


class TestSpaceAgentMappingWithTerrainAndCulture(TestSpaceAgentMapping):
    """Repeating all tests in base class with ContinuousSpaceWithTerrainAndCulture2d.

    This performs a set of regression tests to ensure no bugs have been introduced.
    """

    def setUp(self):
        """Create a test space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture2d(70, 50, -30, -30)
        self.agents = []
        for i, pos in enumerate(REMOVAL_TEST_AGENTS):
            a = MockAgent(i, None)
            self.agents.append(a)
            self.space.place_agent(a, pos)


class TestSpaceAgentMappingWithTerrainAndCulture3d(TestSpaceAgentMapping):
    """Repeating all tests in base class with ContinuousSpaceWithTerrainAndCulture3d.

    This performs a set of regression tests to ensure no bugs have been introduced.
    """

    def setUp(self):
        """Create a test space and populate with Mock Agents."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(70, 50, -30, -30)
        self.agents = []
        for i, pos in enumerate(REMOVAL_TEST_AGENTS_3D):
            a = MockAgent(i, None)
            self.agents.append(a)
            self.space.place_agent(a, pos)
