"""Tests for sensor fusion agent."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from unittest import TestCase

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.sensorfusionagent import SensorFusionAgent
from ret.behaviours.behaviourpool import AlwaysAdder
from ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from ret.behaviours.disablecommunication import DisableCommunicationBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.behaviours.hide import HideBehaviour
from ret.behaviours.move import GroundBasedMoveBehaviour, MoveBehaviour
from ret.behaviours.sense import SenseBehaviour
from ret.behaviours.wait import WaitBehaviour
from ret.orders.order import Order
from ret.orders.tasks.communicate import CommunicateWorldviewTask
from ret.orders.triggers.time import TimeTrigger
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import Confidence, PerceivedAgent
from ret.testing.mocks import MockCommunicateWorldviewBehaviour, MockModel2d
from parameterized import parameterized

if TYPE_CHECKING:
    from typing import Optional

    from ret.behaviours import Behaviour
    from ret.model import RetModel
    from ret.sensing.perceivedworld import PerceivedAgentFilter
    from ret.types import Coordinate2dOr3d


class AirDefenceAgentInitialisationTest(TestCase):
    """Tests for initialisation of Air Defence Agent."""

    def setUp(self):
        """Set up test model."""
        self.model = MockModel2d()

    @parameterized.expand(
        [
            [WaitBehaviour, 1],
            [MoveBehaviour, 1],
            [GroundBasedMoveBehaviour, 1],
            [FireBehaviour, 0],
            [DeployCountermeasureBehaviour, 0],
            [CommunicateWorldviewBehaviour, 0],
            [CommunicateMissionMessageBehaviour, 0],
            [CommunicateOrdersBehaviour, 0],
            [DisableCommunicationBehaviour, 0],
            [SenseBehaviour, 0],
            [HideBehaviour, 1],
        ]
    )
    def test_init_no_behaviour(self, behaviour_type: type, expected: int):
        """Test initialising Sensor Fusion Agent with no user-defined behaviour.

        Args:
            behaviour_type (type): Behaviour type
            expected (int): Expected number of behaviours
        """
        agent = SensorFusionAgent(
            self.model,
            (0.0, 0.0),
            "Sensor Fusion Agent Under Test",
            Affiliation.FRIENDLY,
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert len(behaviours) == expected

    @parameterized.expand(
        [
            [WaitBehaviour, WaitBehaviour()],
            [HideBehaviour, HideBehaviour()],
            [GroundBasedMoveBehaviour, GroundBasedMoveBehaviour(0, [])],
        ]
    )
    def test_init_with_custom_behaviour(self, behaviour_type: type, new_behaviour: Behaviour):
        """Test SensorFusionAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours

        Args:
            behaviour_type (type): Behaviour type
            new_behaviour (Behaviour): New behaviour to add
        """
        agent = SensorFusionAgent(
            self.model,
            (0, 0),
            "Sensor Fusion Agent Under Test",
            Affiliation.FRIENDLY,
            behaviours=[new_behaviour],
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour]

    @parameterized.expand(
        [
            [WaitBehaviour, WaitBehaviour()],
            [HideBehaviour, HideBehaviour()],
            [GroundBasedMoveBehaviour, GroundBasedMoveBehaviour(0, [])],
        ]
    )
    def test_adding_behaviours_to_existing_agent(
        self, behaviour_type: type, new_behaviour: Behaviour
    ):
        """Test SensorFusionAgent initialisation with custom behaviours.

        Checks that the agent uses custom behaviour instead of default behaviours

        Args:
            behaviour_type (type): Behaviour type
            new_behaviour (Behaviour): New behaviour to add
        """
        agent = SensorFusionAgent(
            self.model,
            (0, 0),
            "Sensor Fusion Agent Under Test",
            Affiliation.FRIENDLY,
            behaviours=[new_behaviour],
            behaviour_adder=AlwaysAdder,
        )

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour]

        agent.behaviour_pool.add_behaviour(new_behaviour)

        behaviours = agent.behaviour_pool.expose_behaviour("step", behaviour_type)
        assert behaviours == [new_behaviour, new_behaviour]


class CommunicateOneNewAgentBehaviour(CommunicateWorldviewBehaviour):
    """CommunicateWorldviewBehaviour, where communication contains predefined view."""

    _location: Coordinate2dOr3d
    _id: int

    def __init__(self, location: Coordinate2dOr3d, id: int) -> None:
        """Create a new CommunicateOneNewAgentBehaviour.

        Args:
            location (Coordinate2dOr3d): Location of perceived agent.
            id (int): ID of perceived agent.
        """
        super().__init__()
        self._location = location
        self._id = id

    def get_new_views(
        self, communicator: RetAgent, worldview_filter: Optional[PerceivedAgentFilter]
    ) -> list[PerceivedAgent]:
        """Return new views.

        Args:
            communicator (RetAgent): Agent doing the communication
            worldview_filter (Optional[PerceivedAgentFilter]): Worldview filter

        Returns:
            list[PerceivedAgent]: Single perceived agent based on location and ID
        """
        return [
            PerceivedAgent(
                unique_id=self._id,
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.DETECT,
                location=self._location,
                casualty_state=AgentCasualtyState.ALIVE,
            )
        ]


class TestSensorFusionAgent(TestCase):
    """Tests for sensor fusion agent."""

    def setUp(self):
        """Set up test cases."""
        model = MockModel2d()
        self.model = model
        self.agent = SensorFusionAgent(
            model=model,
            pos=(0, 0),
            name="Fusion",
            affiliation=Affiliation.NEUTRAL,
            behaviours=[MockCommunicateWorldviewBehaviour()],
        )

        self.sensor_agent_1 = RetAgent(
            model=model,
            name="Sensor 1",
            pos=(0, 0),
            affiliation=Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[CommunicateOneNewAgentBehaviour((1, 2), 101)],
        )

        self.sensor_agent_2 = RetAgent(
            model=model,
            name="Sensor 1",
            pos=(25, 25),
            affiliation=Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[CommunicateOneNewAgentBehaviour((2, 1), 102)],
        )

        self.fusion_recipient = RetAgent(
            model=model,
            name="Fusion Recipient",
            pos=(10, 10),
            affiliation=Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.sensor_agent_1.communication_network.add_recipient(self.agent)
        self.sensor_agent_2.communication_network.add_recipient(self.agent)

        self.agent.communication_network.add_recipient(self.fusion_recipient)

    def test_initial_fusion_agent_state(self):
        """Test sensor fusion agents initial state."""
        assert self.agent.active_order is None
        self.assertFalse(self.agent.new_info)

    def test_fusion_agent_state_after_receiving_signal(self):
        """Test state of sensor fusion agent after receiving a signal."""
        self.sensor_agent_1.communicate_worldview_step()
        self.assertTrue(self.agent.new_info)
        self.agent.step()

        agents = self.fusion_recipient.perceived_world.get_perceived_agents()
        assert len(agents) == 3
        agent_at_2_1 = [a for a in agents if a.location == (2, 1)]
        agent_at_1_2 = [a for a in agents if a.location == (1, 2)]
        assert len(agent_at_2_1) == 0
        assert len(agent_at_1_2) == 1

        self.fusion_recipient.perceived_world.reset_worldview()
        for _ in range(0, 10):
            self.agent.step()
            agents = self.fusion_recipient.perceived_world.get_perceived_agents()
            assert len(agents) == 1

        self.sensor_agent_2.communicate_worldview_step()
        self.agent.step()
        agents = self.fusion_recipient.perceived_world.get_perceived_agents()
        agent_at_2_1 = [a for a in agents if a.location == (2, 1)]
        agent_at_1_2 = [a for a in agents if a.location == (1, 2)]
        assert len(agent_at_2_1) == 1
        assert len(agent_at_1_2) == 1


class TestFusionHierarchy(unittest.TestCase):
    """Tests for a hierarchy of sensor fusion agents."""

    class CommunicatePredefinedSense(CommunicateWorldviewBehaviour):
        """CommunicateWorldViewBehaviour which communicates a predefined agent."""

        def __init__(self, perceived_agent: PerceivedAgent):
            """Create a new CommunicatePredefinedSense behaviour.

            Args:
                perceived_agent (PerceivedAgent): Agent to perceive.
            """
            super().__init__()
            self.perceived_agent = perceived_agent

        def get_new_views(
            self,
            communicator: RetAgent,
            worldview_filter: Optional[PerceivedAgentFilter],
        ) -> list[PerceivedAgent]:
            """Get perceived worldview - Always return single predefined agent.

            Args:
                communicator (RetAgent): Agent doing the communication
                worldview_filter (Optional[PerceivedAgentFilter]): Worldview filter

            Returns:
                list[PerceivedAgent]: List containing single, predefined agent
            """
            return [self.perceived_agent]

    def create_agent(self, model: RetModel, view: PerceivedAgent, fusion: RetAgent):
        """Create a single agent, which communicates to a fusion agent.

        Args:
            model (RetModel): Model
            view (PerceivedAgent): View the agent will automatically return
            fusion (RetAgent): Agent to communicate view to
        """
        agent = RetAgent(
            model=model,
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            name="Agent 1",
            pos=(0, 0),
            behaviours=[self.CommunicatePredefinedSense(view)],
            orders=[
                Order(
                    TimeTrigger(model.get_time(), sticky=True),
                    CommunicateWorldviewTask(),
                )
            ],
        )

        agent.communication_network.add_recipient(fusion)

    def create_group(self, model: RetModel, views: list[PerceivedAgent], parent: RetAgent):
        """Create a fusion agent and agents which communicate to it.

        Args:
            model (RetModel): model
            views (list[PerceivedAgent]): List of predefined perceived agents to
                observe. Each child agent will observe a single one of these
            parent (RetAgent): Parent agent for the fusion agent to communicate to

        Returns:
            SensorFusionAgent: Sensor fusion agent.
        """
        agent = SensorFusionAgent(
            model=model,
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Fusion",
            behaviours=[MockCommunicateWorldviewBehaviour()],
        )

        agent.communication_network.add_recipient(parent)

        for view in views:
            self.create_agent(model, view, agent)

        return agent

    class NamedPerceivedAgent(PerceivedAgent):
        """Extension to PerceivedAgent which supports unique naming.

        This extensions has been included because it makes debugging this test more
        straightforward, and serves no purpose that influences the logic of these tests.
        """

        name: str

        def __init__(
            self,
            sense_time: datetime,
            confidence: Confidence,
            location: Coordinate2dOr3d,
            unique_id: int,
            affiliation: Affiliation,
            name: str,
        ):
            """Create a new NamedPerceivedAgent.

            Args:
                sense_time (datetime): The time at which the agent is perceived.
                confidence (Confidence): The confidence of the perceived.
                location (Coordinate2dOr3d): The location of the perceived.
                unique_id (int): The unique ID of the perceived agent.
                affiliation (Affiliation): The affiliation of the perceived agent.
                name (str): The name of the perceived agent.

            """
            super().__init__(
                sense_time=sense_time,
                confidence=confidence,
                location=location,
                unique_id=unique_id,
                affiliation=affiliation,
                casualty_state=AgentCasualtyState.ALIVE,
            )
            self.name = name

    def test_hierarchy(self):
        """Test the communication network hierarchy.

        After two time steps, it can be guaranteed that all information will have been
        conveyed from the lowest level of the network to the sensor fusion agents.

        The sensor fusion agents will apply logical rules to remove duplicate agents.

        After a further two time steps, all information will be communicated to the top
        level sensor fusion agent.

        The sensor fusion agent will apply logical rules to remove duplicate agents.

        After a further two time steps, all information will be send to a recipient.
        """
        model = MockModel2d()

        fusion_agent = SensorFusionAgent(
            model=model,
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Fusion",
            behaviours=[MockCommunicateWorldviewBehaviour()],
        )

        recipient = RetAgent(
            model=model,
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Recipient",
        )

        fusion_agent.communication_network.add_recipient(recipient)

        duplicate_perception_newer = self.NamedPerceivedAgent(
            unique_id=100,
            affiliation=Affiliation.UNKNOWN,
            confidence=Confidence.DETECT,
            sense_time=model.get_time(),
            location=(100, 100),
            name="Perception Newer",
        )
        duplicate_perception_older = self.NamedPerceivedAgent(
            unique_id=100,
            affiliation=Affiliation.UNKNOWN,
            confidence=Confidence.KNOWN,
            sense_time=model.get_time() - timedelta(seconds=10),
            location=(110, 110),
            name="Perception Older",
        )
        unique_1 = self.NamedPerceivedAgent(
            unique_id=101,
            affiliation=Affiliation.HOSTILE,
            confidence=Confidence.IDENTIFY,
            sense_time=model.get_time(),
            location=(130, 130),
            name="Unique 1",
        )
        unique_2 = self.NamedPerceivedAgent(
            unique_id=102,
            affiliation=Affiliation.HOSTILE,
            confidence=Confidence.DETECT,
            sense_time=model.get_time(),
            location=(140, 140),
            name="Unique 2",
        )
        duplicate_worse = self.NamedPerceivedAgent(
            unique_id=103,
            affiliation=Affiliation.HOSTILE,
            confidence=Confidence.DETECT,
            sense_time=model.get_time(),
            location=(150, 150),
            name="Duplicate Worse",
        )
        duplicate_better = self.NamedPerceivedAgent(
            unique_id=103,
            affiliation=Affiliation.HOSTILE,
            confidence=Confidence.RECOGNISE,
            sense_time=model.get_time(),
            location=(160, 160),
            name="Duplicate Better",
        )
        duplicate_even_better = self.NamedPerceivedAgent(
            unique_id=103,
            affiliation=Affiliation.HOSTILE,
            confidence=Confidence.IDENTIFY,
            sense_time=model.get_time(),
            location=(170, 170),
            name="Duplicate Best",
        )

        g1 = self.create_group(
            model,
            [duplicate_better, unique_1, duplicate_perception_older],
            fusion_agent,
        )
        g2 = self.create_group(
            model,
            [
                duplicate_even_better,
                duplicate_worse,
                unique_2,
                duplicate_perception_newer,
            ],
            fusion_agent,
        )

        model.step()
        model.step()

        g1_perceived_agents = g1.perceived_world.get_perceived_agents()

        assert duplicate_better in g1_perceived_agents
        assert unique_1 in g1_perceived_agents
        assert duplicate_perception_older in g1_perceived_agents
        assert duplicate_even_better not in g1_perceived_agents
        assert duplicate_worse not in g1_perceived_agents
        assert unique_2 not in g1_perceived_agents
        assert duplicate_perception_newer not in g1_perceived_agents

        g2_perceived_agents = g2.perceived_world.get_perceived_agents()

        assert duplicate_even_better in g2_perceived_agents
        assert unique_2 in g2_perceived_agents
        assert duplicate_perception_newer in g2_perceived_agents
        assert duplicate_worse not in g2_perceived_agents
        assert unique_1 not in g2_perceived_agents
        assert duplicate_better not in g2_perceived_agents
        assert duplicate_perception_older not in g2_perceived_agents

        model.step()
        model.step()

        fusion_perceived_agents = fusion_agent.perceived_world.get_perceived_agents()

        assert duplicate_even_better in fusion_perceived_agents
        assert duplicate_better not in fusion_perceived_agents
        assert duplicate_worse not in fusion_perceived_agents
        assert unique_1 in fusion_perceived_agents
        assert unique_2 in fusion_perceived_agents
        assert duplicate_perception_newer in fusion_perceived_agents
        assert duplicate_perception_older not in fusion_perceived_agents

        model.step()
        model.step()

        recipient_perceived_agents = recipient.perceived_world.get_perceived_agents()

        assert duplicate_even_better in recipient_perceived_agents
        assert duplicate_better not in recipient_perceived_agents
        assert duplicate_worse not in recipient_perceived_agents
        assert unique_1 in recipient_perceived_agents
        assert unique_2 in recipient_perceived_agents
        assert duplicate_perception_newer in recipient_perceived_agents
        assert duplicate_perception_older not in recipient_perceived_agents
