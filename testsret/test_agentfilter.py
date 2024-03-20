"""Tests for Agent Filters."""

from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.agentfilter import (
    FilterByAffiliation,
    FilterByID,
    FilterFriendly,
    FilterHostile,
    FilterNeutral,
    FilterNot,
    FilterNotFriendly,
    FilterNotHostile,
    FilterNotID,
    FilterNotNeutral,
    FilterNotUnknown,
    FilterUnknown,
)
from ret.testing.mocks import MockAgentWithAffiliation
from parameterized import parameterized


class TestAgentFilters(TestCase):
    """Tests for AgentFilters."""

    def setUp(self):
        """Set up test cases."""
        self.agents = [
            MockAgentWithAffiliation(1, Affiliation.NEUTRAL),
            MockAgentWithAffiliation(2, Affiliation.NEUTRAL),
            MockAgentWithAffiliation(3, Affiliation.UNKNOWN),
            MockAgentWithAffiliation(4, Affiliation.UNKNOWN),
            MockAgentWithAffiliation(5, Affiliation.HOSTILE),
            MockAgentWithAffiliation(6, Affiliation.HOSTILE),
            MockAgentWithAffiliation(7, Affiliation.FRIENDLY),
            MockAgentWithAffiliation(8, Affiliation.FRIENDLY),
        ]

    def test_filter_friendly(self):
        """Test performance of FilterFriendly."""
        agents = FilterFriendly().run(self.agents)  # type: ignore
        assert len(agents) == 2
        assert agents == FilterByAffiliation(Affiliation.FRIENDLY).run(self.agents)
        assert set([a.unique_id for a in agents]) == set([7, 8])

    def test_filter_hostile(self):
        """Test performance of FilterHostile."""
        agents = FilterHostile().run(self.agents)  # type: ignore

        assert len(agents) == 2
        assert agents == FilterByAffiliation(Affiliation.HOSTILE).run(self.agents)
        assert set([a.unique_id for a in agents]) == set([5, 6])

    def test_filter_unknown(self):
        """Test performance of FilterUnknown."""
        agents = FilterUnknown().run(self.agents)  # type: ignore

        assert len(agents) == 2
        assert agents == FilterByAffiliation(Affiliation.UNKNOWN).run(self.agents)
        assert set([a.unique_id for a in agents]) == set([3, 4])

    def test_filter_neutral(self):
        """Test performance of FilterNeutral."""
        agents = FilterNeutral().run(self.agents)  # type: ignore

        assert len(agents) == 2
        assert agents == FilterByAffiliation(Affiliation.NEUTRAL).run(self.agents)
        assert set([a.unique_id for a in agents]) == set([1, 2])

    @parameterized.expand([[i] for i in range(1, 9)])  # type: ignore
    def test_filter_by_id(self, id: int):
        """Test performance of FilterByID for single ID.

        Args:
            id (int): Agent ID
        """
        assert len(FilterByID(id).run(self.agents)) == 1  # type: ignore

    @parameterized.expand([[i, i + 1]] for i in range(1, 8))  # type: ignore
    def test_filter_by_ids(self, ids: list[int]):
        """Test performance of FilterByID for list of IDs.

        Args:
            ids (list[int]): Agent IDs
        """
        assert len(FilterByID(ids).run(self.agents)) == 2  # type: ignore

    def test_filter_not(self):
        """Test performance of FilterNot using FilterByID for excluding a single ID."""
        first_agent_id = self.agents[0].unique_id
        agents = FilterNot(FilterByID(first_agent_id)).run(self.agents)  # type: ignore
        assert len(agents) == 7
        assert agents == self.agents[1:]

    def test_filter_not_id(self):
        """Test performance of FilterNotID for excluding a single ID."""
        first_agent_id = self.agents[0].unique_id
        agents = FilterNotID(first_agent_id).run(self.agents)  # type: ignore
        assert len(agents) == 7
        assert agents == self.agents[1:]

    def test_not_friendly_filter(self):
        """Test performance of FilterNotFriendly."""
        agents = FilterNotFriendly().run(self.agents)  # type: ignore
        assert len(agents) == 6
        assert agents == FilterNot(FilterByAffiliation(Affiliation.FRIENDLY)).run(
            self.agents  # type: ignore
        )

    def test_not_hostile_filter(self):
        """Test performance of FilterNotHostile."""
        agents = FilterNotHostile().run(self.agents)  # type: ignore
        assert agents == FilterNot(FilterByAffiliation(Affiliation.HOSTILE)).run(
            self.agents  # type: ignore
        )
        assert len(agents) == 6

    def test_not_unknown_filter(self):
        """Test performance of FilterNotUnknown."""
        agents = FilterNotUnknown().run(self.agents)  # type: ignore
        assert len(agents) == 6
        assert agents == FilterNot(FilterByAffiliation(Affiliation.UNKNOWN)).run(
            self.agents  # type: ignore
        )

    def test_not_neutral_filter(self):
        """Test performance of FilterNotNeutral."""
        agents = FilterNotNeutral().run(self.agents)  # type: ignore
        assert len(agents) == 6
        assert agents == FilterNot(FilterByAffiliation(Affiliation.NEUTRAL)).run(
            self.agents  # type: ignore
        )

    def test_applying_not_filter_twice(self):
        """Test FilterNot applied twice for net-zero effect on FilterByID."""
        agent_id = 1
        assert len(FilterNot(FilterNot(FilterByID(agent_id))).run(self.agents)) == 1  # type: ignore

    def test_not_filter_with_duplicate_agents(self):
        """Test FilterNot with duplicate agents in the list."""
        friendly_agent = MockAgentWithAffiliation(20, Affiliation.FRIENDLY)
        hostile_agent = MockAgentWithAffiliation(21, Affiliation.HOSTILE)
        duplicate_agents = [friendly_agent, friendly_agent, hostile_agent]
        friendly_agents = FilterFriendly().run(duplicate_agents)  # type: ignore
        not_friendly_agents = FilterNotFriendly().run(duplicate_agents)  # type: ignore

        assert len(friendly_agents) == 2
        assert len(not_friendly_agents) == 1

    def test_not_filter_with_empty_list(self):
        """Test FilterNot with empty agents list."""
        agents = FilterNot(FilterFriendly()).run([])
        assert len(agents) == 0
