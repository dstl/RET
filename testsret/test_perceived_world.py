"""Tests for perceived world."""

from __future__ import generator_stop  # noqa: TC002

import unittest
import warnings
from datetime import datetime
from random import Random

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.agenttype import AgentType
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import (
    AgentByType,
    AgentsAt,
    AgentsByCasualtyState,
    AgentsWithinRange,
    AirAgents,
    AirDefenceAgents,
    AliveAgents,
    And,
    ArmourAgents,
    Confidence,
    DetectedAgents,
    FriendlyAgents,
    GenericAgents,
    HostileAgents,
    IdentifiedAgents,
    InfantryAgents,
    KilledAgents,
    NearbyAgents,
    NeutralAgents,
    Or,
    OtherAgents,
    PerceivedAgent,
    PerceivedAgentBasedOnAgent,
    PerceivedWorld,
    RandomSelection,
    RecognisedAgents,
    UnknownAgents,
    UnknownCasualtyStateAgents,
    UnknownTypeAgents,
)
from ret.testing.mocks import MockModel2d, MockModel3d, MockMoveBehaviour


class TestPerceivedWorld(unittest.TestCase):
    """Test cases for PerceivedWorld."""

    def setUp(self):
        """Test case setup."""
        self.model = MockModel2d()
        self.agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

    def test_get_self_pos(self):
        """Test that an agent can access it's own position from the world (by ID)."""
        assert self.agent.perceived_world.get_agent_pos(1) == (0, 0)

    def test_get_self_pos_missing_agent(self):
        """Test that an exception is thrown where a non-existent agent is accessed."""
        with warnings.catch_warnings(record=True) as w:
            assert self.agent.perceived_world.get_agent_pos(2) is None
        assert "Uniquely identified agent not found in perceived world." == str(w[-1].message)

    def test_get_self_pos_with_self_sensed(self):
        """Test that an agent that has sensed itself does not throw an exception."""
        self.agent.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))
        assert self.agent.perceived_world.get_agent_pos(1) == (0, 0)

    def test_get_pos_multiple_locations(self):
        """Test that an exception is thrown where an agent is in multiple places."""
        self.agent.perceived_world.add_agent(self.agent)
        self.agent.perceived_world._refresh_technique = IdentifiedAgents()
        with self.assertRaises(ValueError) as e:
            self.agent.perceived_world.get_agent_pos(1)
        self.assertEqual(
            "Uniquely identified agent located in more than one place.",
            str(e.exception),
        )

    def test_get_distance(self):
        """Test that perceived world can be used to get distance between points."""
        assert self.agent.perceived_world.get_distance((0, 0), (1, 0)) == 1


class TestPerceivedAgentFilters(unittest.TestCase):
    """Tests for perceived world agent filters."""

    def setUp(self):
        """Test case setup."""
        self.perceived_world = PerceivedWorld(MockModel2d())

        agents = [
            PerceivedAgent(
                unique_id=8,
                sense_time=datetime(2020, 1, 1),
                affiliation=Affiliation.FRIENDLY,
                confidence=Confidence.IDENTIFY,
                location=(0, 0),
                agent_type=AgentType.AIR,
                casualty_state=AgentCasualtyState.ALIVE,
            ),
            PerceivedAgent(
                unique_id=9,
                sense_time=datetime(2020, 1, 1),
                affiliation=Affiliation.FRIENDLY,
                confidence=Confidence.IDENTIFY,
                location=(10, 10),
                agent_type=AgentType.AIR,
                casualty_state=AgentCasualtyState.KILLED,
            ),
            PerceivedAgent(
                unique_id=3,
                sense_time=datetime(2020, 1, 1),
                affiliation=Affiliation.HOSTILE,
                confidence=Confidence.IDENTIFY,
                location=(20, 20),
                agent_type=AgentType.AIR_DEFENCE,
                casualty_state=AgentCasualtyState.KILLED,
            ),
            PerceivedAgent(
                unique_id=4,
                sense_time=datetime(2020, 1, 1),
                affiliation=Affiliation.HOSTILE,
                confidence=Confidence.IDENTIFY,
                location=(30, 30),
                agent_type=AgentType.ARMOUR,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
            PerceivedAgent(
                unique_id=5,
                sense_time=datetime(2020, 1, 1),
                affiliation=Affiliation.NEUTRAL,
                confidence=Confidence.IDENTIFY,
                location=(0, 0),
                agent_type=AgentType.INFANTRY,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
            PerceivedAgent(
                unique_id=6,
                sense_time=datetime(2020, 1, 1),
                affiliation=Affiliation.NEUTRAL,
                confidence=Confidence.IDENTIFY,
                location=(10, 10),
                agent_type=AgentType.GENERIC,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
            PerceivedAgent(
                unique_id=7,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.DETECT,
                location=(20, 20),
                agent_type=AgentType.AIR,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
            PerceivedAgent(
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.RECOGNISE,
                location=(30, 30),
                agent_type=AgentType.OTHER,
                unique_id=10,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
            PerceivedAgent(
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.IDENTIFY,
                location=(30, 30),
                agent_type=AgentType.OTHER,
                unique_id=10,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
            PerceivedAgent(
                sense_time=datetime(2020, 1, 1, second=5),
                confidence=Confidence.RECOGNISE,
                location=(30, 30),
                agent_type=AgentType.OTHER,
                unique_id=10,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
            PerceivedAgent(
                sense_time=datetime(2020, 1, 1, second=5),
                confidence=Confidence.DETECT,
                location=(30, 30),
                agent_type=AgentType.OTHER,
                unique_id=10,
                casualty_state=AgentCasualtyState.UNKNOWN,
            ),
        ]

        self.perceived_world.add_acquisitions(agents)

    def test_get_perceived_world_always_removes_duplicates(self):
        """Test that perceived world always removed duplicate IDs.

        The latest, and best confidence perceptions for a set of perceptions with the
        same ID is always retained.
        """
        agents = self.perceived_world.get_perceived_agents()

        assert len(agents) == 8

        unique_id_10 = [a for a in agents if a._unique_id == 10]

        assert len(unique_id_10) == 1

        # picks latest
        assert unique_id_10[0].sense_time == datetime(2020, 1, 1, second=5)

        # Picks highest within latest bunch - Does not pick IDENTIFY from older senses
        assert unique_id_10[0].confidence == Confidence.RECOGNISE

    def test_neutral_agent_filter(self):
        """Test neutral agent filter."""
        assert len(self.perceived_world.get_perceived_agents(NeutralAgents())) == 2

    def test_friendly_agent_filter(self):
        """Test friendly agent filter."""
        assert len(self.perceived_world.get_perceived_agents(FriendlyAgents())) == 2

    def test_hostile_agent_filter(self):
        """Test hostile agent filter."""
        assert len(self.perceived_world.get_perceived_agents(HostileAgents())) == 2

    def test_unknown_agent_filter(self):
        """Test unknown agent filter."""
        assert len(self.perceived_world.get_perceived_agents(UnknownAgents())) == 2

    def test_detect_filter(self):
        """Test detect state agent filter."""
        assert len(self.perceived_world.get_perceived_agents(DetectedAgents())) == 8

    def test_recognise_filter(self):
        """Test recognise state agent filter."""
        assert len(self.perceived_world.get_perceived_agents(RecognisedAgents())) == 7

    def test_identify_filter(self):  #
        """Test identify state agent filter."""
        assert len(self.perceived_world.get_perceived_agents(IdentifiedAgents())) == 6

    def test_location_filter(self):
        """Test location proximity agent filter."""
        assert (
            len(
                self.perceived_world.get_perceived_agents(
                    AgentsWithinRange(self.perceived_world.get_distance, (1, 1), 2)
                )
            )
            == 2
        )

    def test_at_filter(self):
        """Test location-based agent filter."""
        assert (
            len(
                self.perceived_world.get_perceived_agents(
                    AgentsAt(self.perceived_world.get_distance, (0, 0))
                )
            )
        ) == 2

    def test_agent_type_filter(self):
        """Test agent type filter."""
        assert (len(self.perceived_world.get_perceived_agents(AgentByType(AgentType.AIR)))) == 2

    def test_air_type_filter(self):
        """Test Air Agent filter."""
        assert (len(self.perceived_world.get_perceived_agents(AirAgents()))) == 2

    def test_air_defence_type_filter(self):
        """Test Air Defence Agent Filter."""
        assert len(self.perceived_world.get_perceived_agents(AirDefenceAgents())) == 1

    def test_armour_type_filter(self):
        """Test Armour Type Filter."""
        assert len(self.perceived_world.get_perceived_agents(ArmourAgents())) == 1

    def test_infantry_type_filter(self):
        """Test Infantry Type Filter."""
        assert len(self.perceived_world.get_perceived_agents(InfantryAgents())) == 1

    def test_generic_type_filter(self):
        """Test Generic Agent Type Filter."""
        assert len(self.perceived_world.get_perceived_agents(GenericAgents())) == 1

    def test_other_type_filter(self):
        """Test Other type agent filter."""
        assert len(self.perceived_world.get_perceived_agents(OtherAgents())) == 1

    def test_unknown_type_filter(self):
        """Test unknown type agent filter."""
        assert len(self.perceived_world.get_perceived_agents(UnknownTypeAgents())) == 1

    def test_agents_by_casualty_state_filter(self):
        """Test agents by casualty state filter."""
        assert (
            len(
                self.perceived_world.get_perceived_agents(
                    AgentsByCasualtyState(AgentCasualtyState.ALIVE)
                )
            )
            == 1
        )

    def test_alive_filter(self):
        """Test alive state filter."""
        assert len(self.perceived_world.get_perceived_agents(AliveAgents())) == 1

    def test_killed_filter(self):
        """Test killed state filter."""
        assert len(self.perceived_world.get_perceived_agents(KilledAgents())) == 2

    def test_unknown_killed_filter(self):
        """Test unknown casualty state filter."""
        assert len(self.perceived_world.get_perceived_agents(UnknownCasualtyStateAgents())) == 5

    def test_and_filter(self):
        """Test combined agent filter using 'And'."""
        assert (
            len(
                self.perceived_world.get_perceived_agents(
                    And([RecognisedAgents(), HostileAgents()])
                )
            )
            == 2
        )

    def test_or_filter(self):
        """Test combined agent filter using 'Or'."""
        assert (
            len(
                self.perceived_world.get_perceived_agents(
                    Or(
                        [
                            AgentsWithinRange(self.perceived_world.get_distance, (1, 1), 2),
                            IdentifiedAgents(),
                        ]
                    )
                )
            )
            == 6
        )

    def test_nearby_filter(self):
        """Test the nearby agents filter."""
        model = MockModel2d()
        agent = RetAgent(
            model=model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            behaviours=[MockMoveBehaviour()],
            temperature=20.0,
        )

        assert (
            len(
                self.perceived_world.get_perceived_agents(
                    NearbyAgents(self.perceived_world.get_distance, agent, 2)
                )
            )
            == 2
        )

        agent.move_step((40, 40))

        assert agent.pos == (40, 40)

        assert (
            len(
                self.perceived_world.get_perceived_agents(
                    NearbyAgents(self.perceived_world.get_distance, agent, 2)
                )
            )
            == 0
        )

    def test_reset_world_view(self):
        """Test that resetting world-view clears perceived agents."""
        assert len(self.perceived_world.get_perceived_agents()) == 8
        self.perceived_world.reset_worldview()
        assert len(self.perceived_world.get_perceived_agents()) == 0

    def test_add_to_worldview(self):
        """Test adding agents to perceived worldview."""
        assert len(self.perceived_world.get_perceived_agents()) == 8
        self.perceived_world.add_acquisitions(
            PerceivedAgent(
                unique_id=101,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.DETECT,
                location=(0, 0),
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )
        assert len(self.perceived_world.get_perceived_agents()) == 9

    def test_add_list_to_worldview(self):
        """Test adding list of agents to perceived worldview."""
        assert len(self.perceived_world.get_perceived_agents()) == 8
        self.perceived_world.add_acquisitions(
            [
                PerceivedAgent(
                    unique_id=101,
                    sense_time=datetime(2020, 1, 1),
                    confidence=Confidence.DETECT,
                    location=(0, 0),
                    casualty_state=AgentCasualtyState.ALIVE,
                ),
                PerceivedAgent(
                    unique_id=102,
                    sense_time=datetime(2020, 1, 1),
                    confidence=Confidence.IDENTIFY,
                    location=(1, 1),
                    casualty_state=AgentCasualtyState.ALIVE,
                ),
            ]
        )
        assert len(self.perceived_world.get_perceived_agents()) == 10

    def test_and_filter_no_args(self):
        """Test behaviour of And filter with no conditions."""
        assert (
            self.perceived_world.get_perceived_agents()
            == self.perceived_world.get_perceived_agents(And([]))
        )

    def test_or_filter_no_args(self):
        """Test behaviour of Or filter with no conditions."""
        assert (
            self.perceived_world.get_perceived_agents()
            == self.perceived_world.get_perceived_agents(Or([]))
        )

    def test_known_agents(self):
        """Test known agents filter."""
        model = MockModel2d()
        agent1 = RetAgent(
            model,
            (0, 0),
            "Agent 1",
            Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        agent2 = RetAgent(
            model,
            (1, 1),
            "Agent 2",
            Affiliation.UNKNOWN,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.perceived_world.add_agent(agent1)
        self.perceived_world.add_agent(agent2)

        agent1.pos = (100, 100)

        agent_pos = self.perceived_world.get_agent_pos(1)
        assert agent_pos == (100, 100)

        perceived_agents = self.perceived_world.get_perceived_agents(
            AgentsAt(self.perceived_world.get_distance, (100, 100))
        )
        assert len(perceived_agents) == 1
        assert perceived_agents[0].unique_id == 1

    def test_random_selection_size(self):
        """Test than random selection returns a n random agents."""
        random = Random()

        for _ in range(0, 10):
            n = random.randint(1, 5)
            random_filter = RandomSelection(random_generator=random, number_to_select=n)
            view = self.perceived_world.get_perceived_agents(random_filter)
            assert len(view) == n

    def test_random_selection_size_invalid(self):
        """Test than random selection returns a n random agents."""
        random = Random()

        with self.assertRaises(ValueError) as e:
            RandomSelection(random_generator=random, number_to_select=0)
        self.assertEqual("`number_to_select` is 0, it must be greater than 0", str(e.exception))

    def test_random_repeatability(self):
        """Test that random selections repeat with a fixed random seed.

        Also confirms that the results do not repeat if the random seed changes.
        """
        random_1 = Random(1)
        random_2 = Random(1)
        random_3 = Random(2)

        random_filter_1 = RandomSelection(random_generator=random_1, number_to_select=5)
        views_1 = self.perceived_world.get_perceived_agents(random_filter_1)

        random_filter_2 = RandomSelection(random_generator=random_2, number_to_select=5)
        views_2 = self.perceived_world.get_perceived_agents(random_filter_2)

        assert views_1 == views_2

        random_filter_3 = RandomSelection(random_generator=random_3, number_to_select=5)
        views_3 = self.perceived_world.get_perceived_agents(random_filter_3)

        assert views_1 != views_3

    def test_random_selection_invalid_creation(self):
        """Test that RandomSelection raises exception where created invalidly.

        Number to select must be greater than or equal to 1.
        """
        with self.assertRaises(ValueError) as e:
            RandomSelection(Random(1), 0)
        self.assertEqual("`number_to_select` is 0, it must be greater than 0", str(e.exception))


class TestPerceivedAgent(unittest.TestCase):
    """Test cases for perceived agents."""

    def test_id_requires_known_or_identified(self):
        """Test that accessing unique_id requires agent to be Known."""
        perceived_unknown = PerceivedAgent(
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.RECOGNISE,
            location=(0, 0),
            unique_id=1,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert perceived_unknown.unique_id is None

        perceived_unknown = PerceivedAgent(
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.IDENTIFY,
            location=(0, 0),
            unique_id=1,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert perceived_unknown.unique_id == 1

        perceived_known = PerceivedAgent(
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.KNOWN,
            location=(0, 0),
            unique_id=1,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert perceived_known.unique_id == 1

    def test_affiliation_required_identified(self):
        """Test accessing affiliation requires agent to be identified."""
        perceived_detected = PerceivedAgent(
            unique_id=1,
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.RECOGNISE,
            location=(0, 0),
            affiliation=Affiliation.FRIENDLY,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert perceived_detected.affiliation == Affiliation.UNKNOWN

        perceived_recognised = PerceivedAgent(
            unique_id=1,
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.IDENTIFY,
            location=(0, 0),
            affiliation=Affiliation.FRIENDLY,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert perceived_recognised.affiliation == Affiliation.FRIENDLY

    def test_type_requires_recognise(self):
        """Test accessing type requires agent to be recognised."""
        detected_agent = PerceivedAgent(
            unique_id=1,
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.DETECT,
            location=(0, 0),
            agent_type=AgentType.AIR,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert detected_agent.agent_type == AgentType.UNKNOWN
        recognised_agent = PerceivedAgent(
            unique_id=1,
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.RECOGNISE,
            location=(0, 0),
            agent_type=AgentType.AIR,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert recognised_agent.agent_type == AgentType.AIR

    def test_casualty_state_requires_recognise(self):
        """Test accessing casualty state requires agent to be recognised."""
        detected_agent = PerceivedAgent(
            unique_id=1,
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.DETECT,
            location=(0, 0),
            agent_type=AgentType.AIR,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert detected_agent.casualty_state == AgentCasualtyState.UNKNOWN
        recognised_agent = PerceivedAgent(
            unique_id=1,
            sense_time=datetime(2020, 1, 1),
            confidence=Confidence.RECOGNISE,
            location=(0, 0),
            agent_type=AgentType.AIR,
            casualty_state=AgentCasualtyState.ALIVE,
        )
        assert recognised_agent.casualty_state == AgentCasualtyState.ALIVE

    def test_perceived_agent_based_on_agent(self):
        """Test creating agent from known agent works correctly."""
        ret_agent = RetAgent(
            model=MockModel3d(),
            pos=(0, 0, 0),
            name="test",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        perceived_agent = PerceivedAgentBasedOnAgent(
            sense_time=datetime(2020, 1, 1, 0, 0),
            confidence=Confidence.KNOWN,
            agent=ret_agent,
        )

        assert ret_agent.unique_id == perceived_agent._unique_id
        assert ret_agent.pos == perceived_agent._location
        assert ret_agent.affiliation == perceived_agent._affiliation
        assert ret_agent.agent_type == perceived_agent._agent_type
        assert ret_agent.casualty_state == perceived_agent._casualty_state
