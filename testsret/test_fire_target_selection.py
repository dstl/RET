"""Tests relating to target selection for fire behaviour where no target is defined.

These tests cover the utility methods that underpin the fire behaviour, rather than the
fire behaviour itself.
"""

from datetime import datetime
from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.behaviours.fire import DefaultHostileTargetResolver
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import Confidence, PerceivedAgent
from ret.behaviours.fire import AgentTypeTargetResolver
from ret.agents.agenttype import AgentType
from parameterized import parameterized


class TestHostileAgentSelector(TestCase):
    """Tests relating to target selection for fire behaviour."""

    class AffiliatedAgent(RetAgent):
        """Mock agent that just has an affiliation."""

        def __init__(self, affiliation: Affiliation):
            """Create a new AffiliatedAgent.

            Args:
                affiliation (Affiliation): Affiliation of the agent.
            """
            self.affiliation = affiliation

    @parameterized.expand(
        [
            [Affiliation.HOSTILE, Affiliation.FRIENDLY, 2],
            [Affiliation.FRIENDLY, Affiliation.HOSTILE, 2],
            [Affiliation.UNKNOWN, None, 0],
            [Affiliation.NEUTRAL, None, 0],
        ]
    )
    def test_hostile_selector(self, agent_affiliation, target_affiliation, count):
        """Test that the default hostile selector correctly detects hostile forces.

        Hostile agents consider all friendly agents to be targets.
        Friendly agents consider all hostile agents to be targets.
        Neutral and Unknown agents to not consider any agents to be targets.

        The perceived world is initialised here to contain two instances of each agent
        type, where all perceived agents are at 'IDENTIFY' and therefore their
        affiliation can be actioned upon.

        Args:
            agent_affiliation (Affiliation): The affiliation of the agent under test
            target_affiliation (Affiliation): The affiliation that targets must match
            count (int): The number of targets
        """
        perceived_world = self.create_perceived_world()

        agent = self.AffiliatedAgent(agent_affiliation)
        resolver = DefaultHostileTargetResolver()

        targets = resolver.run(agent, perceived_world)
        assert len(targets) == count

        for t in targets:
            assert t.affiliation == target_affiliation

    def test_custom_selector(self):
        """Test that a custom resolver correctly detects the correct forces.

        The perceived world is initialised here to contain two instances of each agent
        type, where all perceived agents are at 'IDENTIFY' and therefore their
        type can be actioned upon.
        """
        perceived_world = self.create_perceived_world_with_types()

        for type in AgentType:
            resolver = AgentTypeTargetResolver(agent_types=[type])

            targets = resolver.run(self.AffiliatedAgent(Affiliation.FRIENDLY), perceived_world)
            assert len(targets) == 2
            for t in targets:
                assert t.agent_type == type

    def create_perceived_world(self) -> list[PerceivedAgent]:
        """Create a perceived world to be tested against.

        Returns:
            list[PerceivedAgent]: List of eight perceived agents, two of each agent type
        """
        perceived_world = []
        unique_id = 0

        # perceived world contains 2x of each agent type
        for agent_affiliation in Affiliation:
            perceived_world.append(
                PerceivedAgent(
                    location=(0, 0),
                    sense_time=datetime.now(),
                    confidence=Confidence.IDENTIFY,
                    unique_id=unique_id,
                    affiliation=agent_affiliation,
                    casualty_state=AgentCasualtyState.ALIVE,
                )
            )
            unique_id = unique_id + 1
            perceived_world.append(
                PerceivedAgent(
                    location=(0, 0),
                    sense_time=datetime.now(),
                    confidence=Confidence.IDENTIFY,
                    unique_id=unique_id,
                    affiliation=agent_affiliation,
                    casualty_state=AgentCasualtyState.ALIVE,
                )
            )

        return perceived_world

    def create_perceived_world_with_types(self) -> list[PerceivedAgent]:
        """Create a perceived world to be tested against.

        Returns:
            list[PerceivedAgent]: List of perceived agents, two of each agent type
        """
        perceived_world = []
        unique_id = 0

        # perceived world contains 2x of each agent type
        for type in AgentType:
            perceived_world.append(
                PerceivedAgent(
                    location=(0, 0),
                    sense_time=datetime.now(),
                    confidence=Confidence.IDENTIFY,
                    unique_id=unique_id,
                    affiliation=Affiliation.HOSTILE,
                    casualty_state=AgentCasualtyState.ALIVE,
                    agent_type=type,
                )
            )
            unique_id = unique_id + 1
            perceived_world.append(
                PerceivedAgent(
                    location=(0, 0),
                    sense_time=datetime.now(),
                    confidence=Confidence.IDENTIFY,
                    unique_id=unique_id,
                    affiliation=Affiliation.HOSTILE,
                    casualty_state=AgentCasualtyState.ALIVE,
                    agent_type=type,
                )
            )

        return perceived_world
