"""Test cases for JSON models."""

from __future__ import annotations

import unittest

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.space.space import ContinuousSpaceWithTerrainAndCulture2d
from ret.testing.mocks import MockModel2d
from ret.visualisation.json_models import (
    JsonOutObject,
    RetPlayAgent,
    RetPlayInitialData,
    RetPlaySpace,
    RetPlayStepData,
)
from datetime import datetime


class TestRetPlayAgent(unittest.TestCase):
    """Tests for the RetPlayAgent."""

    def test_init(self):
        """Test the initialisation of RetPlayAgent."""
        model = MockModel2d()
        ret_agent = RetAgent(
            model,
            (0, 0),
            "Test Agent",
            Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        retplay_agent = RetPlayAgent.from_model(ret_agent)

        assert retplay_agent.name == ret_agent.name
        assert retplay_agent.affiliation == str(ret_agent.affiliation.name)
        assert retplay_agent.agent_type == str(ret_agent.agent_type.name)
        assert retplay_agent.pos == ret_agent.pos
        assert retplay_agent.id == ret_agent.unique_id
        assert retplay_agent.killed == ret_agent.killed
        assert retplay_agent.in_group == ret_agent.in_group
        assert retplay_agent.mission_messages == ret_agent.mission_messages


class TestRetPlaySpace(unittest.TestCase):
    """Tests for the ReplaySpace."""

    def test_init(self):
        """Tests the initialisation of the RetPlaySpace."""
        space = ContinuousSpaceWithTerrainAndCulture2d(1000, 1000)
        retplay_space = RetPlaySpace.from_model(space)
        assert retplay_space.x_min == space.x_min
        assert retplay_space.x_max == space.x_max
        assert retplay_space.y_min == space.y_min
        assert retplay_space.y_max == space.y_max


class TestRetPlayInitialData(unittest.TestCase):
    """Tests for the RetPlayInitialData."""

    def test_init(self):
        """Test the initialisation of RetPlayInitialData."""
        space = ContinuousSpaceWithTerrainAndCulture2d(1000, 1000)
        initial_data = RetPlayInitialData.from_model(space)

        assert isinstance(initial_data.map_size, RetPlaySpace)
        assert initial_data.map_size.x_min == space.x_min
        assert initial_data.map_size.x_max == space.x_max
        assert initial_data.map_size.y_min == space.y_min
        assert initial_data.map_size.y_max == space.y_max


class TestRetPlayStepData(unittest.TestCase):
    """Tests for the RetPlayStepData."""

    def test_init(self):
        """Test the initialisation of RetPlayStepData."""
        model = MockModel2d()
        ret_agent = RetAgent(
            model,
            (0, 0),
            "Test Agent",
            Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        retplay_agent_list = [RetPlayAgent.from_model(ret_agent)]
        retplay_step_data = RetPlayStepData.from_model(
            0, datetime(2023, 12, 20), retplay_agent_list
        )

        assert retplay_step_data.step_number == 0
        assert retplay_step_data.agents == retplay_agent_list


class TestJsonOutObject(unittest.TestCase):
    """Tests for JsonOutObject."""

    def setUp(self):
        """Create the setup for the JsonOutObject tests."""
        model = MockModel2d()
        self.space = ContinuousSpaceWithTerrainAndCulture2d(1000, 1000)
        self.ret_agent = RetAgent(
            model,
            (0, 0),
            "Test Agent",
            Affiliation.NEUTRAL,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        self.retplay_agent_list = [RetPlayAgent.from_model(self.ret_agent)]

    def test_init(self):
        """Test the initialisation of the JsonOutObject."""
        json_out_object = JsonOutObject.from_model(self.space)

        assert json_out_object.initial_data.map_size.x_min == self.space.x_min
        assert json_out_object.initial_data.map_size.x_max == self.space.x_max
        assert json_out_object.initial_data.map_size.y_min == self.space.y_min
        assert json_out_object.initial_data.map_size.y_max == self.space.y_max
        assert json_out_object.step_data == []

    def test_add_timestep_data(self):
        """Test adding one timestep worth of data to the JsonOutObject."""
        json_out_object = JsonOutObject.from_model(self.space)

        assert len(json_out_object.step_data) == 0

        json_out_object.add_timestep_data(0, datetime(2023, 12, 20), self.retplay_agent_list)

        assert len(json_out_object.step_data) == 1
        assert json_out_object.step_data[0].step_number == 0
        assert len(json_out_object.step_data[0].agents) == 1

        json_agent = json_out_object.step_data[0].agents[0]

        assert json_agent.name == self.ret_agent.name
        assert json_agent.affiliation == str(self.ret_agent.affiliation.name)
        assert json_agent.agent_type == str(self.ret_agent.agent_type.name)
        assert json_agent.pos == self.ret_agent.pos
        assert json_agent.id == self.ret_agent.unique_id
        assert json_agent.killed == self.ret_agent.killed
        assert json_agent.in_group == self.ret_agent.in_group
        assert json_agent.mission_messages == self.ret_agent.mission_messages

    def test_add_invalid_timestep_data(self):
        """Test adding an invalid timestep data to the JsonOutObject."""
        json_out_object = JsonOutObject.from_model(self.space)

        with self.assertRaises(ValueError) as e:
            json_out_object.add_timestep_data(None, datetime(2023, 12, 20), self.retplay_agent_list)
        self.assertEqual(
            "The time step information failed as there was no step number defined.",
            str(e.exception),
        )
