"""Test cases for countermeasures."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agentfilter import AgentFilter, FilterByAffiliation
from mesa_ret.agents.airagent import AirAgent
from mesa_ret.space.clutter.clutter import ClutterModifier
from mesa_ret.space.clutter.countermeasure import SphereCountermeasure
from mesa_ret.testing.mocks import MockAgentWithAffiliation, MockCountermeasure, MockModel3d

if TYPE_CHECKING:
    from mesa_ret.types import Coordinate3d


class TestCountermeasure(unittest.TestCase):
    """Tests for Countermeasures."""

    def setUp(self):
        """Test case set up."""
        self.model = MockModel3d()
        self.agent = AirAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="agent",
            affiliation=Affiliation.FRIENDLY,
        )
        self.neutral_agent = MockAgentWithAffiliation(1, Affiliation.NEUTRAL)

    def test_deploy_countermeasure(self):
        """Test deploying a countermeasure."""
        countermeasure = MockCountermeasure(value=10)

        self.assertFalse(countermeasure.deployed)
        assert countermeasure.clutter_modifier is None
        assert self.model.space.clutter_field.get_value(self.agent.pos, self.neutral_agent) == 0

        countermeasure.deploy(self.agent)

        self.assertTrue(countermeasure.deployed)
        assert countermeasure.clutter_modifier is not None
        assert self.model.space.clutter_field.get_value(self.agent.pos, self.neutral_agent) == 10

    def test_deploy_countermeasure_twice(self):
        """Test deploying a countermeasure twice."""
        countermeasure = MockCountermeasure(value=10)

        countermeasure.deploy(self.agent)
        countermeasure.deploy(self.agent)

        self.assertTrue(countermeasure.deployed)
        assert countermeasure.clutter_modifier is not None
        assert self.model.space.clutter_field.get_value(self.agent.pos, self.neutral_agent) == 10

    def test_kill_countermeasure(self):
        """Test killing a countermeasure."""
        countermeasure = MockCountermeasure(value=10)

        self.assertFalse(countermeasure.killed)
        assert self.model.space.clutter_field.get_value(self.agent.pos, self.neutral_agent) == 0

        countermeasure.deploy(self.agent)
        assert self.model.space.clutter_field.get_value(self.agent.pos, self.neutral_agent) == 10
        kill_time = datetime(2020, 1, 1)
        countermeasure.kill(kill_time)

        self.assertTrue(countermeasure.killed)
        assert countermeasure.clutter_modifier.expiry_time == kill_time
        self.model.step()
        assert self.model.space.clutter_field.get_value(self.agent.pos, self.neutral_agent) == 0

    def test_kill_countermeasure_twice(self):
        """Test killing a countermeasure twice."""
        countermeasure = MockCountermeasure(value=10)

        countermeasure.deploy(self.agent)
        kill_time_1 = datetime(2020, 1, 1)
        countermeasure.kill(kill_time_1)
        kill_time_2 = datetime(2020, 1, 2)
        countermeasure.kill(kill_time_2)

        self.assertTrue(countermeasure.killed)
        assert countermeasure.clutter_modifier.expiry_time == kill_time_1

    def test_kill_before_deploy(self):
        """Test killing a countermeasure before deploying."""
        countermeasure = MockCountermeasure(value=10)

        kill_time_1 = datetime(2020, 1, 1)
        countermeasure.kill(kill_time_1)

        self.assertFalse(countermeasure.deployed)
        self.assertTrue(countermeasure.killed)
        assert countermeasure.clutter_modifier is None

    def test_get_clutter_modifer(self):
        """Test clutter modifier is only created once."""
        countermeasure = MockCountermeasure(value=10)

        cm_1 = countermeasure._get_clutter_modifier(self.agent)
        cm_2 = countermeasure._get_clutter_modifier(self.agent)

        assert isinstance(cm_1, ClutterModifier)
        assert isinstance(cm_2, ClutterModifier)
        assert cm_1 is cm_2

    def test_sphere_countermeasure(self):
        """Test the area and duration that a spherical countermeasure influences."""
        countermeasure = SphereCountermeasure(10, 10, timedelta(minutes=5))

        countermeasure.deploy(self.agent)

        assert countermeasure.clutter_modifier.get_value(self.agent.pos, self.neutral_agent) == 10
        shifted_pos: Coordinate3d = (
            self.agent.pos[0] + 5,
            self.agent.pos[1] + 5,
            self.agent.pos[2] + 5,
        )
        assert countermeasure.clutter_modifier.get_value(shifted_pos, self.neutral_agent) == 10
        assert countermeasure.clutter_modifier.expiry_time == self.model.start_time + timedelta(
            minutes=5
        )

    def test_sphere_countermeasure_get_new_instance(self):
        """Test the sphere countermeasure get instance method."""
        countermeasure = SphereCountermeasure(
            10, 10, timedelta(minutes=5), persist_beyond_deployer=True, agent_filter=AgentFilter()
        )
        clone = countermeasure.get_new_instance()

        assert countermeasure is not clone
        assert countermeasure._clutter_value == clone._clutter_value
        assert countermeasure._clutter_radius == clone._clutter_radius
        assert countermeasure._life_time == clone._life_time
        assert countermeasure.persist_beyond_deployer == clone.persist_beyond_deployer
        assert countermeasure.agent_filter == clone.agent_filter

    def test_deploy_with_agent_filter(self):
        """Test deploying a countermeasure with an agent filter."""
        hostile_agent = MockAgentWithAffiliation(2, Affiliation.HOSTILE)

        countermeasure = MockCountermeasure(
            value=10, agent_filter=FilterByAffiliation(Affiliation.HOSTILE)
        )

        countermeasure.deploy(self.agent)

        self.assertTrue(countermeasure.deployed)
        assert countermeasure.clutter_modifier is not None
        assert self.model.space.clutter_field.get_value(self.agent.pos, self.neutral_agent) == 0
        assert self.model.space.clutter_field.get_value(self.agent.pos, hostile_agent) == 10
