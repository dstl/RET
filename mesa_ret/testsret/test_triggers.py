"""Tests for triggers."""

import random
import string
from datetime import datetime
from unittest import TestCase

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.agents.sensorfusionagent import SensorFusionAgent
from mesa_ret.orders.order import CompoundAndTrigger, CompoundOrTrigger
from mesa_ret.orders.triggers.immediate import ImmediateSensorFusionTrigger, ImmediateTrigger
from mesa_ret.orders.triggers.killed import AgentKilledTrigger, KilledAgentsAtPositionTrigger
from mesa_ret.orders.triggers.mission import MissionMessageTrigger
from mesa_ret.orders.triggers.position import (
    AliveAgentsAtPositionTrigger,
    CrossedBoundaryTrigger,
    InAreaTrigger,
    MovedOutOfAreaTrigger,
    NotInAreaTrigger,
    PositionTrigger,
)
from mesa_ret.orders.triggers.sensor import IlluminatedBySensorTrigger
from mesa_ret.orders.triggers.time import TimeTrigger
from mesa_ret.orders.triggers.weapon import (
    AgentFiredWeaponTrigger,
    WeaponFiredNearAgentTrigger,
    WeaponFiredNearLocationTrigger,
)
from mesa_ret.sensing.perceivedworld import PerceivedAgent
from mesa_ret.sensing.sensor import (
    DistanceAttenuatedSensor,
    NotifySensedBySideEffect,
    SensorDistanceThresholds,
)
from mesa_ret.space.feature import LineFeature, SphereFeature
from mesa_ret.testing.mocks import (
    MockCommunicateWorldviewBehaviour,
    MockFireBehaviour,
    MockModel2d,
    MockMoveBehaviour,
    MockWeapon,
)


class TestTriggers(TestCase):
    """Test cases for triggers."""

    move_behaviour = MockMoveBehaviour()

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()
        self.agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[self.move_behaviour],
        )

    def test_immediate(self):
        """Test immediate trigger."""
        trigger = ImmediateTrigger()

        assert trigger.is_active(self.agent)

    def test_position(self):
        """Test position-based trigger."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1)
        trigger3 = PositionTrigger(self.agent, (10, 0), 1)

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_position_perceived_agent(self):
        """Test position trigger check condition when agent not in perceived world."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)

        checker = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert trigger1.is_active(checker)

    def test_position_inverted(self):
        """Test position-based trigger with inverted logic."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1, invert=True)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1, invert=True)
        trigger3 = PositionTrigger(self.agent, (10, 0), 1, invert=True)

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

    def test_position_inverted_perceived_agent(self):
        """Test inverted position trigger check condition when agent not in perceived world."""
        trigger1 = PositionTrigger(self.agent, (2, 2), 1, invert=True)

        checker = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert trigger1.is_active(checker)

    def test_in_area(self):
        """Test in area trigger."""
        trigger1 = InAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"))
        trigger2 = InAreaTrigger(self.agent, SphereFeature((1, 0), 1, "feature"))
        trigger3 = InAreaTrigger(self.agent, SphereFeature((10, 0), 1, "feature"))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_in_area_perceived_agent(self):
        """Test in area trigger check condition when agent not in perceived world."""
        trigger1 = InAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"))

        checker = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert trigger1.is_active(checker)

    def test_not_in_area(self):
        """Test not in area trigger."""
        trigger1 = NotInAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"))
        trigger2 = NotInAreaTrigger(self.agent, SphereFeature((1, 0), 1, "feature"))
        trigger3 = NotInAreaTrigger(self.agent, SphereFeature((10, 0), 1, "feature"))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

    def test_not_in_area_perceived_agent(self):
        """Test not in area trigger check condition when agent not in perceived world."""
        trigger1 = NotInAreaTrigger(self.agent, SphereFeature((2, 2), 1, "feature"))

        checker = RetAgent(
            model=self.model,
            pos=(10, 10),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert trigger1.is_active(checker)

    def test_crossed_boundary(self):
        """Test crossed boundary trigger."""
        trigger1 = CrossedBoundaryTrigger(self.agent, LineFeature((5, -1), (5, 1), "feature"))
        trigger2 = CrossedBoundaryTrigger(self.agent, LineFeature((0, 1), (10, 1), "feature"))
        trigger3 = CrossedBoundaryTrigger(self.agent, LineFeature((11, 0), (20, 0), "feature"))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

    def test_crossed_boundary_perceived_agent(self):
        """Test crossed boundary trigger check condition when agent not in perceived world."""
        trigger1 = CrossedBoundaryTrigger(self.agent, LineFeature((5, -1), (5, 1), "feature"))

        checker = RetAgent(
            model=self.model,
            pos=(10, 10),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)
        self.agent.move_step((10, 0))
        assert not trigger1.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert not trigger1.is_active(checker)
        self.agent.move_step((0, 0))
        checker.model.step()
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))
        assert trigger1.is_active(checker)

    def test_crossed_boundary_inverted(self):
        """Test crossed boundary trigger with inverted logic."""
        trigger1 = CrossedBoundaryTrigger(
            self.agent, LineFeature((5, -1), (5, 1), "feature"), invert=True
        )
        trigger2 = CrossedBoundaryTrigger(
            self.agent, LineFeature((0, 1), (10, 1), "feature"), invert=True
        )
        trigger3 = CrossedBoundaryTrigger(
            self.agent, LineFeature((11, 0), (20, 0), "feature"), invert=True
        )

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert not trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_crossed_boundary_inverted_perceived_agent(self):
        """Test inverted crossed boundary trigger when agent not in perceived world."""
        trigger1 = CrossedBoundaryTrigger(
            self.agent, LineFeature((5, -1), (5, 1), "feature"), invert=True
        )

        checker = RetAgent(
            model=self.model,
            pos=(10, 10),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)
        self.agent.move_step((10, 0))
        assert not trigger1.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert trigger1.is_active(checker)
        self.agent.move_step((0, 0))
        checker.model.step()
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))
        assert not trigger1.is_active(checker)

    def test_moved_out_of_area(self):
        """Test moved out of area trigger."""
        trigger1 = MovedOutOfAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"))
        trigger2 = MovedOutOfAreaTrigger(self.agent, SphereFeature((1, 0), 1, "feature"))
        trigger3 = MovedOutOfAreaTrigger(self.agent, SphereFeature((10, 0), 1, "feature"))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

    def test_moved_out_of_area_perceived_agent(self):
        """Test moved out of area trigger when agent not in perceived world."""
        trigger1 = MovedOutOfAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"))

        checker = RetAgent(
            model=self.model,
            pos=(10, 10),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)
        self.agent.move_step((10, 0))
        assert not trigger1.is_active(checker)

        self.agent.move_step((0, 0))
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert not trigger1.is_active(checker)
        self.agent.move_step((10, 0))
        checker.model.step()
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))
        assert trigger1.is_active(checker)

    def test_moved_out_of_area_inverted(self):
        """Test moved out of area trigger with inverted logic."""
        trigger1 = MovedOutOfAreaTrigger(
            self.agent, SphereFeature((0, 0), 1, "feature"), invert=True
        )
        trigger2 = MovedOutOfAreaTrigger(
            self.agent, SphereFeature((1, 0), 1, "feature"), invert=True
        )
        trigger3 = MovedOutOfAreaTrigger(
            self.agent, SphereFeature((10, 0), 1, "feature"), invert=True
        )

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_moved_out_of_area_inverted_perceived_agent(self):
        """Test inverted moved out of area trigger when agent not in perceived world."""
        trigger1 = MovedOutOfAreaTrigger(
            self.agent, SphereFeature((0, 0), 1, "feature"), invert=True
        )

        checker = RetAgent(
            model=self.model,
            pos=(10, 10),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not trigger1.is_active(checker)
        self.agent.move_step((10, 0))
        assert not trigger1.is_active(checker)

        self.agent.move_step((0, 0))
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert trigger1.is_active(checker)
        self.agent.move_step((10, 0))
        checker.model.step()
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))
        assert not trigger1.is_active(checker)

    def test_alive_agents_at_position(self):
        """Test alive agents at position trigger."""
        trigger1 = AliveAgentsAtPositionTrigger((0, 0), 1)
        trigger2 = AliveAgentsAtPositionTrigger((1, 0), 1)
        trigger3 = AliveAgentsAtPositionTrigger((10, 0), 1)

        target_agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[self.move_behaviour],
        )
        self.agent.perceived_world.add_agent(target_agent)

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        target_agent.move_step((10, 0))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        target_agent.kill()

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

    def test_alive_agents_at_perceived_agent(self):
        """Test alive agents at position trigger when agent not in perceived world."""
        trigger1 = AliveAgentsAtPositionTrigger((0, 0), 1)
        trigger2 = AliveAgentsAtPositionTrigger((10, 10), 1)

        checker = RetAgent(
            model=self.model,
            pos=(20, 20),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        killed_agent = RetAgent(
            model=self.model,
            pos=(10, 10),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        killed_agent.kill()

        assert not trigger1.is_active(checker)
        assert not trigger2.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(killed_agent))

        assert trigger1.is_active(checker)
        assert not trigger2.is_active(checker)

    def test_alive_agents_at_position_inverted(self):
        """Test alive agents at position trigger with inverted logic."""
        trigger1 = AliveAgentsAtPositionTrigger((0, 0), 1, invert=True)
        trigger2 = AliveAgentsAtPositionTrigger((1, 0), 1, invert=True)
        trigger3 = AliveAgentsAtPositionTrigger((10, 0), 1, invert=True)
        target_agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[self.move_behaviour],
        )
        self.agent.perceived_world.add_agent(target_agent)

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        target_agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

    def test_alive_agents_at_position_inverted_perceived_agent(self):
        """Test inverted alive agents at position trigger when agent not in perceived world."""
        trigger1 = AliveAgentsAtPositionTrigger((2, 2), 1, invert=True)

        checker = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        # Trigger is active as no alive agents are perceived at the location
        assert trigger1.is_active(checker)

        self.agent.move_step((2, 2))
        assert trigger1.is_active(checker)

        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        # Agent is now perceived at the location
        assert not trigger1.is_active(checker)

        self.agent.move_step((0, 0))
        checker.model.step()  # progress time step and update perceived location
        checker.perceived_world.add_acquisitions(PerceivedAgent.to_perceived_agent(self.agent))

        assert trigger1.is_active(checker)

    def test_time(self):
        """Test time-based triggers."""
        trigger1 = TimeTrigger(datetime(2019, 12, 31, 23, 0))
        trigger2 = TimeTrigger(datetime(2020, 1, 1, 0, 0))
        trigger3 = TimeTrigger(datetime(2020, 1, 1, 0, 30))
        trigger4 = TimeTrigger(datetime(2020, 1, 1, 1, 0))
        trigger5 = TimeTrigger(datetime(2020, 1, 1, 1, 30))
        trigger6 = TimeTrigger(datetime(2020, 1, 1, 2, 0))

        assert not trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)
        assert not trigger4.is_active(self.agent)
        assert not trigger5.is_active(self.agent)
        assert not trigger6.is_active(self.agent)

        self.model.step()

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)
        assert trigger4.is_active(self.agent)
        assert trigger5.is_active(self.agent)
        assert not trigger6.is_active(self.agent)

    def test_time_inverted(self):
        """Test time-based triggers with inverted logic."""
        trigger1 = TimeTrigger(datetime(2019, 12, 31, 23, 0), invert=True)
        trigger2 = TimeTrigger(datetime(2020, 1, 1, 0, 0), invert=True)
        trigger3 = TimeTrigger(datetime(2020, 1, 1, 0, 30), invert=True)
        trigger4 = TimeTrigger(datetime(2020, 1, 1, 1, 0), invert=True)
        trigger5 = TimeTrigger(datetime(2020, 1, 1, 1, 30), invert=True)
        trigger6 = TimeTrigger(datetime(2020, 1, 1, 2, 0), invert=True)

        assert trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)
        assert trigger4.is_active(self.agent)
        assert trigger5.is_active(self.agent)
        assert trigger6.is_active(self.agent)

        self.model.step()

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)
        assert not trigger4.is_active(self.agent)
        assert not trigger5.is_active(self.agent)
        assert trigger6.is_active(self.agent)

    def test_position_sticky(self):
        """Test sticky position based triggers."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1, True)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1, True)
        trigger3 = PositionTrigger(self.agent, (10, 0), 1, True)

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_position_sticky_inverted(self):
        """Test sticky position based triggers with inverted logic."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1, True, invert=True)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1, True, invert=True)
        trigger3 = PositionTrigger(self.agent, (10, 0), 1, True, invert=True)

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_alive_agents_at_position_sticky(self):
        """Test sticky alive agents at position triggers."""
        trigger1 = AliveAgentsAtPositionTrigger((0, 0), 1, True)
        trigger2 = AliveAgentsAtPositionTrigger((1, 0), 1, True)
        trigger3 = AliveAgentsAtPositionTrigger((10, 0), 1, True)
        target_agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[self.move_behaviour],
        )
        self.agent.perceived_world.add_agent(target_agent)

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        target_agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        target_agent.kill()

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_alive_agents_at_position_sticky_inverted(self):
        """Test sticky alive agents at position triggers with inverted logic."""
        trigger1 = AliveAgentsAtPositionTrigger((0, 0), 1, True, invert=True)
        trigger2 = AliveAgentsAtPositionTrigger((1, 0), 1, True, invert=True)
        trigger3 = AliveAgentsAtPositionTrigger((10, 0), 1, True, invert=True)
        target_agent = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            behaviours=[self.move_behaviour],
        )
        self.agent.perceived_world.add_agent(target_agent)

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        target_agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        target_agent.kill()

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_in_area_sticky(self):
        """Test sticky in area triggers."""
        trigger1 = InAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"), True)
        trigger2 = InAreaTrigger(self.agent, SphereFeature((1, 0), 1, "feature"), True)
        trigger3 = InAreaTrigger(self.agent, SphereFeature((10, 0), 1, "feature"), True)

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_not_in_area_sticky(self):
        """Test sticky not in area triggers."""
        trigger1 = NotInAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"), True)
        trigger2 = NotInAreaTrigger(self.agent, SphereFeature((1, 0), 1, "feature"), True)
        trigger3 = NotInAreaTrigger(self.agent, SphereFeature((10, 0), 1, "feature"), True)

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_crossed_boundary_sticky(self):
        """Test crossed boundary trigger."""
        trigger1 = CrossedBoundaryTrigger(self.agent, LineFeature((5, -1), (5, 1), "feature"), True)
        trigger2 = CrossedBoundaryTrigger(self.agent, LineFeature((0, 1), (10, 1), "feature"), True)
        trigger3 = CrossedBoundaryTrigger(
            self.agent, LineFeature((11, 0), (20, 0), "feature"), True
        )

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((20, 0))

        assert trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_moved_out_of_area_sticky(self):
        """Test sticky moved out of area triggers."""
        trigger1 = MovedOutOfAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"), True)
        trigger2 = MovedOutOfAreaTrigger(self.agent, SphereFeature((1, 0), 1, "feature"), True)
        trigger3 = MovedOutOfAreaTrigger(self.agent, SphereFeature((10, 0), 1, "feature"), True)

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert not trigger3.is_active(self.agent)

        self.agent.move_step((20, 0))

        assert trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)

    def test_time_sticky(self):
        """Test sticky time-based triggers."""
        trigger1 = TimeTrigger(datetime(2019, 12, 31, 23, 0), True)
        trigger2 = TimeTrigger(datetime(2020, 1, 1, 0, 0), True)
        trigger3 = TimeTrigger(datetime(2020, 1, 1, 0, 30), True)
        trigger4 = TimeTrigger(datetime(2020, 1, 1, 1, 0), True)
        trigger5 = TimeTrigger(datetime(2020, 1, 1, 1, 30), True)
        trigger6 = TimeTrigger(datetime(2020, 1, 1, 2, 0), True)

        assert not trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)
        assert not trigger4.is_active(self.agent)
        assert not trigger5.is_active(self.agent)
        assert not trigger6.is_active(self.agent)

        self.model.step()

        assert not trigger1.is_active(self.agent)
        assert trigger2.is_active(self.agent)
        assert trigger3.is_active(self.agent)
        assert trigger4.is_active(self.agent)
        assert trigger5.is_active(self.agent)
        assert not trigger6.is_active(self.agent)

    def test_compound_and_trigger(self):
        """Test compound 'and' triggers."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1)

        compound_trigger = CompoundAndTrigger([trigger1, trigger2])

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((1.5, 0))

        assert not compound_trigger.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert not compound_trigger.is_active(self.agent)

    def test_compound_and_trigger_inverted(self):
        """Test compound 'and' triggers with inverted logic."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1)

        compound_trigger = CompoundAndTrigger([trigger1, trigger2], invert=True)

        assert not compound_trigger.is_active(self.agent)

        self.agent.move_step((1.5, 0))

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert compound_trigger.is_active(self.agent)

    def test_compound_or_trigger(self):
        """Test compound 'or' triggers."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (5, 0), 1)

        compound_trigger = CompoundOrTrigger([trigger1, trigger2])

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((5, 0))

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert not compound_trigger.is_active(self.agent)

    def test_compound_or_trigger_inverted(self):
        """Test compound 'or' triggers with inverted logic."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (5, 0), 1)

        compound_trigger = CompoundOrTrigger([trigger1, trigger2], invert=True)

        assert not compound_trigger.is_active(self.agent)

        self.agent.move_step((5, 0))

        assert not compound_trigger.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert compound_trigger.is_active(self.agent)

    def test_compound_and_trigger_sticky(self):
        """Test sticky compound 'and' triggers."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1)

        compound_trigger = CompoundAndTrigger([trigger1, trigger2], sticky=True)

        self.agent.move_step((10, 0))

        assert not compound_trigger.is_active(self.agent)

        self.agent.move_step((1.5, 0))

        assert not compound_trigger.is_active(self.agent)

        self.agent.move_step((0, 0))

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert compound_trigger.is_active(self.agent)

    def test_compound_or_trigger_sticky(self):
        """Test sticky compound 'or' triggers."""
        trigger1 = PositionTrigger(self.agent, (5, 0), 1)
        trigger2 = PositionTrigger(self.agent, (10, 0), 1)

        compound_trigger = CompoundOrTrigger([trigger1, trigger2], sticky=True)

        assert not compound_trigger.is_active(self.agent)

        self.agent.move_step((5, 0))

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((0, 0))

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((10, 0))

        assert compound_trigger.is_active(self.agent)

        self.agent.move_step((0, 0))

        assert compound_trigger.is_active(self.agent)

    def test_immediate_sensor_fusion_trigger(self):
        """Test immediate sensor fusion trigger."""
        fusion_agent = SensorFusionAgent(
            model=self.model,
            pos=(0, 0),
            name="Fusion",
            affiliation=Affiliation.NEUTRAL,
            behaviours=[MockCommunicateWorldviewBehaviour()],
        )
        immediate_trigger = ImmediateSensorFusionTrigger()

        self.assertFalse(immediate_trigger.is_active(fusion_agent))

        fusion_agent.new_info = True
        self.assertTrue(immediate_trigger.is_active(fusion_agent))

        self.model.step()

        self.assertFalse(immediate_trigger.is_active(fusion_agent))

    def test_illuminated_by_sensor_trigger(self):
        """Test the illuminated by sensor trigger."""
        sensor_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            is_active_sensor=True,
            sensor_side_effects=[NotifySensedBySideEffect()],
        )

        out_of_range_agent = RetAgent(
            model=self.model,
            pos=(800.0, 0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        trigger = IlluminatedBySensorTrigger(self.agent)
        trigger2 = IlluminatedBySensorTrigger(out_of_range_agent)

        assert not trigger.is_active(self.agent)
        assert not trigger.is_active(out_of_range_agent)

        distance_attenuated_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=0)
        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=sensor_agent)

        assert len(perceived_agents) == 1
        assert perceived_agents[0]._unique_id == self.agent.unique_id

        assert not trigger.is_active(self.agent)
        assert not trigger2.is_active(out_of_range_agent)

        self.model.step()

        assert trigger.is_active(self.agent)
        assert not trigger2.is_active(out_of_range_agent)

        self.model.step()

        assert not trigger.is_active(self.agent)
        assert not trigger2.is_active(out_of_range_agent)

    def test_illuminated_by_sensor_trigger_sticky(self):
        """Test the illuminated by sensor trigger when sticky."""
        sensor_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            is_active_sensor=True,
            sensor_side_effects=[NotifySensedBySideEffect()],
        )

        out_of_range_agent = RetAgent(
            model=self.model,
            pos=(800.0, 0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        trigger = IlluminatedBySensorTrigger(self.agent, sticky=True)
        trigger2 = IlluminatedBySensorTrigger(out_of_range_agent, sticky=True)

        assert not trigger.is_active(self.agent)
        assert not trigger2.is_active(out_of_range_agent)

        distance_attenuated_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=0)
        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=sensor_agent)

        assert len(perceived_agents) == 1
        assert perceived_agents[0]._unique_id == self.agent.unique_id

        assert not trigger.is_active(self.agent)
        assert not trigger2.is_active(out_of_range_agent)

        self.model.step()

        assert trigger.is_active(self.agent)
        assert not trigger2.is_active(out_of_range_agent)

        self.model.step()

        assert trigger.is_active(self.agent)
        assert not trigger2.is_active(out_of_range_agent)

    def test_agent_fired_weapon_trigger(self):
        """Test the agent fired weapon trigger."""
        weapon1 = MockWeapon(name="Weapon1")
        weapon2 = MockWeapon(name="Weapon2")
        firer_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1, weapon2],
        )
        firer_agent2 = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent 2",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1, weapon2],
        )

        trigger1 = AgentFiredWeaponTrigger(firer=firer_agent, weapon_name=weapon1.name)
        trigger2 = AgentFiredWeaponTrigger(firer=firer_agent, weapon_name=weapon2.name)
        trigger_any = AgentFiredWeaponTrigger(firer=firer_agent)
        trigger_agent2 = AgentFiredWeaponTrigger(firer=firer_agent2)

        fire_behaviour = MockFireBehaviour()
        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon1, location=(0, 0))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

        self.model.step()

        assert trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

        self.model.step()

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

    def test_agent_fired_weapon_trigger_sticky(self):
        """Test the agent fired weapon trigger when sticky."""
        weapon1 = MockWeapon(name="Weapon1")
        weapon2 = MockWeapon(name="Weapon2")
        firer_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1, weapon2],
        )
        firer_agent2 = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent 2",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1, weapon2],
        )

        trigger1 = AgentFiredWeaponTrigger(firer=firer_agent, weapon_name=weapon1.name, sticky=True)
        trigger2 = AgentFiredWeaponTrigger(firer=firer_agent, weapon_name=weapon2.name, sticky=True)
        trigger_any = AgentFiredWeaponTrigger(firer=firer_agent, sticky=True)
        trigger_agent2 = AgentFiredWeaponTrigger(firer=firer_agent2, sticky=True)

        fire_behaviour = MockFireBehaviour()
        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon1, location=(0, 0))

        assert not trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert not trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

        self.model.step()

        assert trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

        self.model.step()

        assert trigger1.is_active(self.agent)
        assert not trigger2.is_active(self.agent)
        assert trigger_any.is_active(self.agent)
        assert not trigger_agent2.is_active(self.agent)

    def test_weapon_fired_near_agent_trigger(self):
        """Test the weapon fired near agent trigger."""
        weapon1 = MockWeapon(name="Weapon1")
        weapon2 = MockWeapon(name="Weapon2")
        weapon3 = MockWeapon(name="Weapon3")
        firer_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1],
        )

        trigger_weapon1 = WeaponFiredNearAgentTrigger(
            tolerance=10, agent=self.agent, weapon_name=weapon1.name
        )
        trigger_weapon2 = WeaponFiredNearAgentTrigger(
            tolerance=10, agent=self.agent, weapon_name=weapon2.name
        )
        trigger_weapon3 = WeaponFiredNearAgentTrigger(
            tolerance=10, agent=self.agent, weapon_name=weapon3.name
        )
        trigger_any_weapons = WeaponFiredNearAgentTrigger(tolerance=10, agent=self.agent)

        fire_behaviour = MockFireBehaviour()
        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        # weapon 1 within tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon1, location=(5, 0))
        # weapon 2 just outside tolerance (i.e. exactly 10m away)
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon2, location=(10, 0))
        # weapon 3 outside tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon3, location=(15, 0))

        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

    def test_weapon_fired_near_agent_trigger_sticky(self):
        """Test the weapon fired near agent trigger when sticky."""
        weapon1 = MockWeapon(name="Weapon1")
        weapon2 = MockWeapon(name="Weapon2")
        weapon3 = MockWeapon(name="Weapon3")
        firer_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1],
        )

        trigger_weapon1 = WeaponFiredNearAgentTrigger(
            tolerance=10, agent=self.agent, weapon_name=weapon1.name, sticky=True
        )
        trigger_weapon2 = WeaponFiredNearAgentTrigger(
            tolerance=10, agent=self.agent, weapon_name=weapon2.name, sticky=True
        )
        trigger_weapon3 = WeaponFiredNearAgentTrigger(
            tolerance=10, agent=self.agent, weapon_name=weapon3.name, sticky=True
        )
        trigger_any_weapons = WeaponFiredNearAgentTrigger(
            tolerance=10, agent=self.agent, sticky=True
        )

        fire_behaviour = MockFireBehaviour()
        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        # weapon 1 within tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon1, location=(5, 0))
        # weapon 2 just outside tolerance (i.e. exactly 10m away)
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon2, location=(10, 0))
        # weapon 3 outside tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon3, location=(15, 0))

        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert trigger_any_weapons.is_active(self.agent)

    def test_weapon_fired_near_location_trigger(self):
        """Test the weapon fired near location trigger."""
        weapon1 = MockWeapon(name="Weapon1")
        weapon2 = MockWeapon(name="Weapon2")
        weapon3 = MockWeapon(name="Weapon3")
        firer_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1],
        )

        trigger_weapon1 = WeaponFiredNearLocationTrigger(
            tolerance=10, location=(20, 20), weapon_name=weapon1.name
        )
        trigger_weapon2 = WeaponFiredNearLocationTrigger(
            tolerance=10, location=(20, 20), weapon_name=weapon2.name
        )
        trigger_weapon3 = WeaponFiredNearLocationTrigger(
            tolerance=10, location=(20, 20), weapon_name=weapon3.name
        )
        trigger_any_weapons = WeaponFiredNearLocationTrigger(tolerance=10, location=(20, 20))

        fire_behaviour = MockFireBehaviour()
        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        # weapon 1 within tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon1, location=(15, 20))
        # weapon 2 just outside tolerance (i.e. exactly 10m away)
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon2, location=(10, 20))
        # weapon 3 outside tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon2, location=(5, 20))

        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

    def test_weapon_fired_near_location_trigger_sticky(self):
        """Test the weapon fired near location trigger when sticky."""
        weapon1 = MockWeapon(name="Weapon1")
        weapon2 = MockWeapon(name="Weapon2")
        weapon3 = MockWeapon(name="Weapon3")
        firer_agent = RetAgent(
            model=self.model,
            pos=(300.0, 0),
            name="Firer Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            weapons=[weapon1],
        )

        trigger_weapon1 = WeaponFiredNearLocationTrigger(
            tolerance=10, location=(20, 20), weapon_name=weapon1.name, sticky=True
        )
        trigger_weapon2 = WeaponFiredNearLocationTrigger(
            tolerance=10, location=(20, 20), weapon_name=weapon2.name, sticky=True
        )
        trigger_weapon3 = WeaponFiredNearLocationTrigger(
            tolerance=10, location=(20, 20), weapon_name=weapon3.name, sticky=True
        )
        trigger_any_weapons = WeaponFiredNearLocationTrigger(
            tolerance=10, location=(20, 20), sticky=True
        )

        fire_behaviour = MockFireBehaviour()
        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        # weapon 1 within tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon1, location=(15, 20))
        # weapon 2 just outside tolerance (i.e. exactly 10m away)
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon2, location=(10, 20))
        # weapon 3 outside tolerance
        fire_behaviour.step(firer=firer_agent, rounds=1, weapon=weapon2, location=(5, 20))

        assert not trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert not trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert trigger_any_weapons.is_active(self.agent)

        self.model.step()

        assert trigger_weapon1.is_active(self.agent)
        assert not trigger_weapon2.is_active(self.agent)
        assert not trigger_weapon3.is_active(self.agent)
        assert trigger_any_weapons.is_active(self.agent)

    def test_immediate_trigger_str_method(self):
        """Test the immediate trigger string method."""
        trigger = ImmediateTrigger()
        assert str(trigger) == "Immediate Trigger"

    def test_immediate_sensor_fusion_trigger_str_method(self):
        """Test the immediate sensor fusion trigger string method."""
        trigger = ImmediateSensorFusionTrigger()
        assert str(trigger) == "Immediate Sensor Fusion Trigger"

    def test_position_trigger_str_method(self):
        """Test the position trigger string method."""
        trigger = PositionTrigger(self.agent, (0, 0), 1, True)
        assert str(trigger) == "Position Trigger"

    def test_in_area_trigger_str_method(self):
        """Test the in area trigger string method."""
        trigger = InAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"), True)
        assert str(trigger) == "In Area Trigger"

    def test_not_in_area_trigger_str_method(self):
        """Test the not in area trigger string method."""
        trigger = NotInAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"), True)
        assert str(trigger) == "Not In Area Trigger"

    def test_crossed_boundary_trigger_str_method(self):
        """Test the crossed boundary trigger string method."""
        trigger = CrossedBoundaryTrigger(self.agent, LineFeature((0, 0), (10, 10), "feature"), True)
        assert str(trigger) == "Crossed Boundary Trigger"

    def test_moved_out_of_area_trigger_str_method(self):
        """Test the moved out of area trigger string method."""
        trigger = MovedOutOfAreaTrigger(self.agent, SphereFeature((0, 0), 1, "feature"), True)
        assert str(trigger) == "Moved Out Of Area Trigger"

    def test_alive_agents_at_position_trigger_str_method(self):
        """Test the alive agents at position trigger string method."""
        trigger = AliveAgentsAtPositionTrigger((0, 0), 1, True)
        assert str(trigger) == "Alive Agents at Position Trigger"

    def test_time_trigger_str_method(self):
        """Test the time trigger string method."""
        trigger = TimeTrigger(datetime(2019, 12, 31, 23, 0), True)
        assert str(trigger) == "Time Trigger"

    def test_compound_and_trigger_str_method(self):
        """Test the compound 'and' trigger string method."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1)
        trigger = CompoundAndTrigger([trigger1, trigger2])
        assert (
            str(trigger) == "Compound 'And' Trigger: (" + str(trigger1) + ", " + str(trigger1) + ")"
        )

    def test_compound_or_trigger_str_method(self):
        """Test the compound 'or' trigger string method."""
        trigger1 = PositionTrigger(self.agent, (0, 0), 1)
        trigger2 = PositionTrigger(self.agent, (1, 0), 1)
        trigger = CompoundOrTrigger([trigger1, trigger2])
        assert (
            str(trigger) == "Compound 'Or' Trigger: (" + str(trigger1) + ", " + str(trigger1) + ")"
        )

    def test_illuminated_by_sensor_trigger_str_method(self):
        """Test the illuminated by sensor trigger string method."""
        trigger = IlluminatedBySensorTrigger(agent=self.agent)
        assert str(trigger) == "Illuminated by Sensor Trigger"

    def test_agent_fired_weapon_trigger_str_method(self):
        """Test the agent fired weapon trigger string method."""
        trigger = AgentFiredWeaponTrigger(firer=self.agent, weapon_name="Test")
        assert str(trigger) == "Agent Fired Weapon Trigger"

    def test_weapon_fired_near_agent_trigger_str_method(self):
        """Test the weapon fired near agent trigger string method."""
        trigger = WeaponFiredNearAgentTrigger(agent=self.agent, tolerance=1, weapon_name="Test")
        assert str(trigger) == "Weapon Fired Near Agent Trigger"

    def test_weapon_fired_near_location_trigger_str_method(self):
        """Test the weapon fired near location trigger string method."""
        trigger = WeaponFiredNearLocationTrigger(location=(0, 0), tolerance=1, weapon_name="Test")
        assert str(trigger) == "Weapon Fired Near Location Trigger"

    def test_mission_trigger_tripped(self):
        """Test that a trigger is tripped where agent contains trigger message."""
        mission = self.generate_message()
        self.agent.mission_messages = [mission]
        trigger = MissionMessageTrigger(mission)

        assert trigger.is_active(self.agent)

    def test_mission_trigger_not_tripped(self):
        """Test that a trigger is not tripped where checker contains no messages."""
        mission = self.generate_message()
        self.agent.mission_messages = []
        trigger = MissionMessageTrigger(mission)

        assert not trigger.is_active(self.agent)

    def test_mission_trigger_not_tripped_for_alternative_message(self):
        """Test that a trigger is not tripped where checker doesn't contain mission."""
        mission_1 = self.generate_message()
        mission_2 = self.generate_message()
        self.agent.mission_messages = [mission_2]
        trigger = MissionMessageTrigger(mission_1)

        assert not trigger.is_active(self.agent)

    def test_mission_trigger_string_representation(self):
        """Test that mission trigger __str__ magic method returns expected content."""
        mission = self.generate_message()
        trigger = MissionMessageTrigger(mission)

        assert str(trigger) == "Mission Message Trigger"

    def generate_message(self) -> str:
        """Return a random mission message.

        Returns:
            str: Random message
        """
        letters = string.ascii_letters
        return "".join(random.choice(letters) for i in range(10))

    def test_immediate_trigger_get_new_instance(self):
        """Test the immediate trigger get_new_instance method."""
        trigger = ImmediateTrigger(log=False)
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, ImmediateTrigger)
        assert trigger._log == clone._log

    def test_immediate_sensor_fusion_trigger_get_new_instance(self):
        """Test the immediate sensor fusion trigger get_new_instance method."""
        trigger = ImmediateSensorFusionTrigger(log=False)
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, ImmediateSensorFusionTrigger)
        assert trigger._log == clone._log

    def test_position_trigger_get_new_instance(self):
        """Test the position trigger get_new_instance method."""
        trigger = PositionTrigger(
            agent=self.agent, position=(0, 0), tolerance=1, sticky=True, log=False, invert=False
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, PositionTrigger)
        assert trigger._agent == clone._agent
        assert trigger._position == clone._position
        assert trigger._tolerance == clone._tolerance
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_position_trigger_inverted_get_new_instance(self):
        """Test the inverted position trigger get_new_instance method."""
        trigger = PositionTrigger(
            agent=self.agent, position=(0, 0), tolerance=1, sticky=True, log=False, invert=True
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, PositionTrigger)
        assert trigger._agent == clone._agent
        assert trigger._position == clone._position
        assert trigger._tolerance == clone._tolerance
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_in_area_trigger_get_new_instance(self):
        """Test the in area trigger get_new_instance method."""
        trigger = InAreaTrigger(
            agent=self.agent, area=SphereFeature((0, 0), 1, "feature"), sticky=True, log=False
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, InAreaTrigger)
        assert trigger._agent == clone._agent
        assert trigger._area == clone._area
        assert trigger._agent == clone._agent
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log

    def test_not_in_area_trigger_get_new_instance(self):
        """Test the not in area trigger get_new_instance method."""
        trigger = NotInAreaTrigger(
            agent=self.agent, area=SphereFeature((0, 0), 1, "feature"), sticky=True, log=False
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, NotInAreaTrigger)
        assert trigger._agent == clone._agent
        assert trigger._area == clone._area
        assert trigger._agent == clone._agent
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log

    def test_crossed_boundary_trigger_get_new_instance(self):
        """Test the crossed boundary trigger get_new_instance method."""
        trigger = CrossedBoundaryTrigger(
            agent=self.agent,
            boundary=LineFeature((0, 0), (10, 10), "feature"),
            sticky=True,
            log=False,
            invert=True,
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, CrossedBoundaryTrigger)
        assert trigger._agent == clone._agent
        assert trigger._boundary == clone._boundary
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_moved_out_of_area_trigger_get_new_instance(self):
        """Test the moved out of area trigger get_new_instance method."""
        trigger = MovedOutOfAreaTrigger(
            agent=self.agent,
            area=SphereFeature((0, 0), 1, "feature"),
            sticky=True,
            log=False,
            invert=True,
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, MovedOutOfAreaTrigger)
        assert trigger._agent == clone._agent
        assert trigger._area == clone._area
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_alive_agents_at_position_trigger_get_new_instance(self):
        """Test the alive agents at position trigger get_new_instance method."""
        trigger = AliveAgentsAtPositionTrigger(
            position=(0, 0), tolerance=1, sticky=True, log=False, invert=False
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, AliveAgentsAtPositionTrigger)
        assert trigger._position == clone._position
        assert trigger._tolerance == clone._tolerance
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_alive_agents_at_position_trigger_inverted_get_new_instance(self):
        """Test the inverted alive agents at position trigger get_new_instance method."""
        trigger = AliveAgentsAtPositionTrigger(
            position=(0, 0), tolerance=1, sticky=True, log=False, invert=True
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, AliveAgentsAtPositionTrigger)
        assert trigger._position == clone._position
        assert trigger._tolerance == clone._tolerance
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_time_trigger_get_new_instance(self):
        """Test the time trigger get_new_instance method."""
        trigger = TimeTrigger(
            time=datetime(2019, 12, 31, 23, 0), sticky=True, log=False, invert=True
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, TimeTrigger)
        assert trigger._time == clone._time
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_mission_message_trigger_get_new_instance(self):
        """Test the mission message trigger get_new_instance method."""
        mission = self.generate_message()
        trigger = MissionMessageTrigger(message=mission, sticky=True, log=False, invert=True)
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, MissionMessageTrigger)
        assert trigger._message == clone._message
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_compound_and_trigger_get_new_instance(self):
        """Test the compound 'and' trigger get_new_instance method."""
        trigger1 = PositionTrigger(
            agent=self.agent, position=(0, 0), tolerance=1, sticky=True, log=False, invert=False
        )
        trigger2 = PositionTrigger(
            agent=self.agent, position=(1, 0), tolerance=1, sticky=False, log=True, invert=True
        )
        trigger = CompoundAndTrigger([trigger1, trigger2], sticky=True, log=False, invert=True)
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, CompoundAndTrigger)
        assert len(trigger._triggers) == len(clone._triggers)
        for i in range(len(trigger._triggers)):
            assert trigger._triggers[i]._agent == clone._triggers[i]._agent
            assert trigger._triggers[i]._position == clone._triggers[i]._position
            assert trigger._triggers[i]._tolerance == clone._triggers[i]._tolerance
            assert trigger._triggers[i]._sticky == clone._triggers[i]._sticky
            assert trigger._triggers[i]._log == clone._triggers[i]._log
            assert trigger._triggers[i]._invert == clone._triggers[i]._invert

        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_compound_or_trigger_get_new_instance(self):
        """Test the compound 'or' trigger get_new_instance method."""
        trigger1 = PositionTrigger(
            agent=self.agent, position=(0, 0), tolerance=1, sticky=True, log=False, invert=False
        )
        trigger2 = PositionTrigger(
            agent=self.agent, position=(5, 0), tolerance=1, sticky=False, log=True, invert=True
        )
        trigger = CompoundOrTrigger([trigger1, trigger2], sticky=True, log=False, invert=True)
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, CompoundOrTrigger)
        assert len(trigger._triggers) == len(clone._triggers)
        for i in range(len(trigger._triggers)):
            assert trigger._triggers[i]._agent == clone._triggers[i]._agent
            assert trigger._triggers[i]._position == clone._triggers[i]._position
            assert trigger._triggers[i]._tolerance == clone._triggers[i]._tolerance
            assert trigger._triggers[i]._sticky == clone._triggers[i]._sticky
            assert trigger._triggers[i]._log == clone._triggers[i]._log
            assert trigger._triggers[i]._invert == clone._triggers[i]._invert

        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_illuminated_by_sensor_get_new_instance(self):
        """Test the illuminated by sensor trigger get_new_instance method."""
        trigger = IlluminatedBySensorTrigger(agent=self.agent, sticky=True, log=False)
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, IlluminatedBySensorTrigger)
        assert trigger._agent == clone._agent
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log

    def test_agent_fired_weapon_get_new_instance(self):
        """Test the agent fired weapon trigger get_new_instance method."""
        trigger = AgentFiredWeaponTrigger(
            firer=self.agent, weapon_name="Test", sticky=True, log=False
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, AgentFiredWeaponTrigger)
        assert trigger._firer == clone._firer
        assert trigger._weapon_name == clone._weapon_name
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log

    def test_weapon_fired_near_agent_get_new_instance(self):
        """Test the weapon fired near agent trigger get_new_instance method."""
        trigger = WeaponFiredNearAgentTrigger(
            agent=self.agent, tolerance=1, weapon_name="Test", sticky=True, log=False
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, WeaponFiredNearAgentTrigger)
        assert trigger._agent == clone._agent
        assert trigger._tolerance == clone._tolerance
        assert trigger._weapon_name == clone._weapon_name
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log

    def test_weapon_fired_near_location_get_new_instance(self):
        """Test the weapon fired near location trigger get_new_instance method."""
        trigger = WeaponFiredNearLocationTrigger(
            location=(0, 0), tolerance=1, weapon_name="Test", sticky=True, log=False
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, WeaponFiredNearLocationTrigger)
        assert trigger._location == clone._location
        assert trigger._tolerance == clone._tolerance
        assert trigger._weapon_name == clone._weapon_name
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log

    def test_trigger_get_new_instance_sticky_status_maintained_not_triggered(self):
        """Test the sticky status maintainted method without triggering."""
        trigger = PositionTrigger(self.agent, (10, 10), 1)
        sticky_trigger = PositionTrigger(self.agent, (10, 10), 1, sticky=True)

        self.assertFalse(trigger._sticky)
        self.assertTrue(sticky_trigger._sticky)
        self.assertFalse(trigger.is_active(self.agent))
        self.assertFalse(sticky_trigger.is_active(self.agent))

        trigger_2 = trigger.get_new_instance_sticky_status_maintained()
        sticky_trigger_2 = sticky_trigger.get_new_instance_sticky_status_maintained()

        self.assertFalse(trigger._sticky)
        self.assertTrue(sticky_trigger._sticky)
        self.assertFalse(trigger_2.is_active(self.agent))
        self.assertFalse(sticky_trigger_2.is_active(self.agent))

    def test_trigger_get_new_instance_sticky_status_maintained_triggered(self):
        """Test the sticky status maintainted method with triggering."""
        trigger = PositionTrigger(self.agent, (10, 10), 1)
        sticky_trigger = PositionTrigger(self.agent, (10, 10), 1, sticky=True)

        self.assertFalse(trigger._sticky)
        self.assertTrue(sticky_trigger._sticky)
        self.assertFalse(trigger.is_active(self.agent))
        self.assertFalse(sticky_trigger.is_active(self.agent))

        self.agent.move_step((10, 10))

        self.assertTrue(trigger.is_active(self.agent))
        self.assertTrue(sticky_trigger.is_active(self.agent))

        self.agent.move_step((0, 0))

        self.assertFalse(trigger.is_active(self.agent))
        self.assertTrue(sticky_trigger.is_active(self.agent))

        trigger_2 = trigger.get_new_instance_sticky_status_maintained()
        sticky_trigger_2 = sticky_trigger.get_new_instance_sticky_status_maintained()

        self.assertFalse(trigger_2._sticky)
        self.assertTrue(sticky_trigger_2._sticky)
        self.assertFalse(trigger_2.is_active(self.agent))
        self.assertTrue(sticky_trigger_2.is_active(self.agent))

    def test_killed_trigger_sticky(self):
        """Test sticky variant of killed agent at position trigger."""
        killed_trigger_sticky = KilledAgentsAtPositionTrigger(self.agent.pos)
        checking_agent = RetAgent(
            model=self.model,
            pos=(1, 1),
            name="Checking Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not killed_trigger_sticky.is_active(checking_agent)
        checking_agent.perceived_world.add_agent(self.agent)
        assert not killed_trigger_sticky.is_active(checking_agent)
        self.agent._killed = True
        assert killed_trigger_sticky.is_active(checking_agent)
        self.agent._killed = False
        assert killed_trigger_sticky.is_active(checking_agent)

    def test_killed_trigger_not_sticky(self):
        """Test non-sticky variant of killed agent at position trigger."""
        killed_trigger_not_sticky = KilledAgentsAtPositionTrigger(self.agent.pos, sticky=False)
        checking_agent = RetAgent(
            model=self.model,
            pos=(1, 1),
            name="Checking Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not killed_trigger_not_sticky.is_active(checking_agent)
        checking_agent.perceived_world.add_agent(self.agent)
        assert not killed_trigger_not_sticky.is_active(checking_agent)
        self.agent._killed = True
        assert killed_trigger_not_sticky.is_active(checking_agent)
        self.agent._killed = False
        assert not killed_trigger_not_sticky.is_active(checking_agent)

    def test_killed_trigger_get_new_instance(self):
        """Test the killed trigger get_new_instance method."""
        trigger = KilledAgentsAtPositionTrigger(
            position=self.agent.pos, sticky=True, log=False, invert=True
        )
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, KilledAgentsAtPositionTrigger)
        assert trigger._position == clone._position
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_killed_trigger_str_method(self):
        """Test the killed trigger string method."""
        trigger = KilledAgentsAtPositionTrigger(self.agent.pos)
        assert str(trigger) == "Killed Agents at Position Trigger"

    def test_killed_trigger_log_method(self):
        """Test the killed trigger _get_log_message."""
        trigger = KilledAgentsAtPositionTrigger(self.agent.pos)
        assert (
            trigger._get_log_message()
            == f"Killed agents at position trigger target: {self.agent.pos}"
        )

    def test_agent_killed_trigger_sticky(self):
        """Test sticky variant of agent killed trigger."""
        killed_trigger_sticky = AgentKilledTrigger(self.agent)
        checking_agent = RetAgent(
            model=self.model,
            pos=(1, 1),
            name="Checking Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not killed_trigger_sticky.is_active(checking_agent)
        checking_agent.perceived_world.add_agent(self.agent)
        assert not killed_trigger_sticky.is_active(checking_agent)
        self.agent._killed = True
        assert killed_trigger_sticky.is_active(checking_agent)
        self.agent._killed = False
        assert killed_trigger_sticky.is_active(checking_agent)

    def test_agent_killed_trigger_not_sticky(self):
        """Test non-sticky variant of agent killed trigger."""
        killed_trigger_not_sticky = AgentKilledTrigger(self.agent, sticky=False)
        checking_agent = RetAgent(
            model=self.model,
            pos=(1, 1),
            name="Checking Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not killed_trigger_not_sticky.is_active(checking_agent)
        checking_agent.perceived_world.add_agent(self.agent)
        assert not killed_trigger_not_sticky.is_active(checking_agent)
        self.agent._killed = True
        assert killed_trigger_not_sticky.is_active(checking_agent)
        self.agent._killed = False
        assert not killed_trigger_not_sticky.is_active(checking_agent)

    def test_agent_killed_trigger_sticky_inverted(self):
        """Test sticky variant of agent killed trigger when inverted."""
        not_killed_trigger_sticky = AgentKilledTrigger(self.agent, invert=True)
        checking_agent = RetAgent(
            model=self.model,
            pos=(1, 1),
            name="Checking Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not_killed_trigger_sticky.is_active(checking_agent)
        checking_agent.perceived_world.add_agent(self.agent)
        assert not_killed_trigger_sticky.is_active(checking_agent)
        self.agent._killed = True
        assert not_killed_trigger_sticky.is_active(checking_agent)
        self.agent._killed = False
        assert not_killed_trigger_sticky.is_active(checking_agent)

    def test_agent_killed_trigger_not_sticky_inverted(self):
        """Test non-sticky variant of agent killed trigger when inverted."""
        not_killed_trigger_not_sticky = AgentKilledTrigger(self.agent, sticky=False, invert=True)
        checking_agent = RetAgent(
            model=self.model,
            pos=(1, 1),
            name="Checking Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        assert not_killed_trigger_not_sticky.is_active(checking_agent)
        checking_agent.perceived_world.add_agent(self.agent)
        assert not_killed_trigger_not_sticky.is_active(checking_agent)
        self.agent._killed = True
        assert not not_killed_trigger_not_sticky.is_active(checking_agent)
        self.agent._killed = False
        assert not_killed_trigger_not_sticky.is_active(checking_agent)

    def test_agent_killed_trigger_get_new_instance(self):
        """Test the agent killed trigger get_new_instance method."""
        trigger = AgentKilledTrigger(agent=self.agent, sticky=False, log=False, invert=True)
        clone = trigger.get_new_instance()

        assert trigger is not clone
        assert isinstance(clone, AgentKilledTrigger)
        assert trigger._target_agent == clone._target_agent
        assert trigger._sticky == clone._sticky
        assert trigger._log == clone._log
        assert trigger._invert == clone._invert

    def test_agent_killed_trigger_str_method(self):
        """Test the agent killed trigger string method."""
        trigger = AgentKilledTrigger(self.agent)
        assert str(trigger) == "Agent Killed Trigger"

    def test_agent_killed_trigger_log_method(self):
        """Test the agent killed trigger _get_log_message."""
        trigger = AgentKilledTrigger(self.agent)
        assert trigger._get_log_message() == (
            f"Agent killed trigger. Agent: {self.agent.agent_type} "
            f"(Agent ID: {self.agent.unique_id})."
        )
