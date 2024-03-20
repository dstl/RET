"""Tests for CEP sensors."""

from __future__ import annotations

import math
import unittest
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.agents.agenttype import AgentType
from ret.sensing.perceivedworld import AgentById, Confidence, Not
from ret.sensing.sensor import (
    ActiveFilter,
    CEPSensor,
    RangeToCepData,
    SensorClutterAttenuator,
    SensorDetectionTimings,
    SensorDistanceThresholds,
    SensorSamplingDistance,
    SensorWavelength,
    SigIntSensor,
)
from ret.testing.mocks import MockModel3d
from parameterized import parameterized

from testsret.test_sensor import SensorTestModel

if TYPE_CHECKING:
    from ret.sensing.perceivedworld import PerceivedAgent
    from ret.space.space import (
        ContinuousSpaceWithTerrainAndCulture2d,
        ContinuousSpaceWithTerrainAndCulture3d,
    )
    from ret.types import Coordinate2dOr3d


class TestCEPSensor(unittest.TestCase):
    """Test sensor class calculations."""

    @staticmethod
    def is_less_than_or_close_to(value: float, compare_value: float) -> bool:
        """Check whether value is less than or close to compare_value.

        Args:
            value (float): the value to check
            compare_value (float): the value to compare against

        Returns:
            bool: whether value is less than or close to compare_value
        """
        return value <= compare_value or math.isclose(value, compare_value)

    @staticmethod
    def assert_count_confidence_levels(
        agents: list[PerceivedAgent], confidence: Confidence, n: int
    ):
        """Assert a given number of perceived agent have the expected confidence level.

        Args:
            agents (list[PerceivedAgent]): A list of perceived agents to check
            confidence (Confidence): The confidence level expected
            n (int): The expected number of perceived agents at the given
                confidence level
        """
        assert len([agent for agent in agents if agent.confidence == confidence]) == n

    @staticmethod
    def assert_single_close_agent(
        space: ContinuousSpaceWithTerrainAndCulture2d,
        agents: list[PerceivedAgent],
        location: Coordinate2dOr3d,
        distance: float,
    ):
        """Assert that a single agent is close to a given distance from a location.

        Args:
            space (ContinuousSpaceWithTerrainAndCulture2d): the model space
            agents (list[PerceivedAgent]): a list of perceived agents
            location (Coordinate2dOr3d): the location to check against
            distance (float): the expected distance from the given location
        """
        n = len(
            [
                agent
                for agent in agents
                if math.isclose(
                    space.get_distance(agent.location, location),
                    distance,
                )
            ]
        )

        assert n == 1

    def test_cep_sensor(self):
        """Test CEP sensor in 3D gives correct results."""
        cep_model = MockModel3d()

        sensor_agent = RetAgent(
            model=cep_model,
            pos=(300.0, 100.0, 100.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 10.0),
                    (200.0, 20.0),
                    (300.0, 30.0),
                    (400.0, 40.0),
                ]
            ),
            AgentType.AIR: RangeToCepData(
                [
                    (10.0, 5.0),
                    (20.0, 10.0),
                ]
            ),
        }

        # Too close
        RetAgent(
            model=cep_model,
            pos=(300.0, 100.0, 105.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.AIR,
        )

        target_agent_air = RetAgent(
            model=cep_model,
            pos=(300.0, 100.0, 115.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.AIR,
        )

        target_agent_generic_100 = RetAgent(
            model=cep_model,
            pos=(300.0, 200.0, 100.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        target_agent_generic_240 = RetAgent(
            model=cep_model,
            pos=(300.0, 340.0, 100.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        target_agent_generic_200 = RetAgent(
            model=cep_model,
            pos=(100.0, 100.0, 100.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        # Too far
        RetAgent(
            model=cep_model,
            pos=(300.0, 510.0, 100.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        cep_sensor = CEPSensor(
            cep_range_data=cep_data,
            distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 400.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
            clutter_attenuator=SensorClutterAttenuator(1.0),
        )

        cep_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=90)

        perceived_agents = cep_sensor.get_results(sensor_agent=sensor_agent)

        assert len(perceived_agents) == 4

        self.assert_count_confidence_levels(perceived_agents, Confidence.IDENTIFY, 4)

        self.assert_single_close_agent(cep_model.space, perceived_agents, target_agent_air.pos, 7.5)

        self.assert_single_close_agent(
            cep_model.space,
            perceived_agents,
            target_agent_generic_100.pos,
            15.0,
        )

        self.assert_single_close_agent(
            cep_model.space,
            perceived_agents,
            target_agent_generic_240.pos,
            24.0,
        )

        self.assert_single_close_agent(
            cep_model.space,
            perceived_agents,
            target_agent_generic_200.pos,
            20.0,
        )

    def test_cep_sensor_at_space_boundary(self):
        """Test CEP sensor (3D) handles cases where agents are the edge of the space."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 10.0),
                    (500.0, 50.0),
                ]
            ),
        }

        location = (0.0, 0.0, 0.0)

        cep_model = MockModel3d()

        cep_sensor_agent = RetAgent(
            model=cep_model,
            pos=(300.0, 400.0, 0.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        agents: list[RetAgent] = []
        for _ in range(10):
            agents.append(
                RetAgent(
                    model=cep_model,
                    pos=location,
                    name="Target Agent",
                    affiliation=Affiliation.FRIENDLY,
                    critical_dimension=2.0,
                    reflectivity=0.1,
                    temperature=20.0,
                )
            )

        space_x_min = cep_model.space.x_min
        space_y_min = cep_model.space.y_min

        cep_sensor = CEPSensor(
            cep_range_data=cep_data,
            distance_thresholds=SensorDistanceThresholds(700.0, 600.0, 500.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
            clutter_attenuator=SensorClutterAttenuator(1.0),
        )

        cep_sensor.run_detection(sensor_agent=cep_sensor_agent, sense_direction=90)

        perceived_agents = cep_sensor.get_results(sensor_agent=cep_sensor_agent)

        assert len(perceived_agents) == 10

        for perceived_agent in perceived_agents:
            assert not cep_model.space.out_of_bounds(
                (perceived_agent.location[0], perceived_agent.location[1])
            )

        assert (
            len(
                [
                    agent
                    for agent in perceived_agents
                    if self.is_less_than_or_close_to(
                        cep_model.space.get_distance(agent.location, location),
                        50.0,
                    )
                ]
            )
            == 10
        )

        assert (
            len(
                [
                    agent
                    for agent in perceived_agents
                    if agent.location[0] >= space_x_min
                    and agent.location[1] >= space_y_min
                    and agent.location[2] >= 0
                ]
            )
            == 10
        )

    def test_cep_sensor_confidence(self):
        """Test CEP sensor confidence levels."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }

        cep_model = MockModel3d(seed=1.0)

        sensor_agent = RetAgent(
            model=cep_model,
            pos=(300.0, 400.0, 50.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        target_agent_generic_100 = RetAgent(
            model=cep_model,
            pos=(300.0, 500.0, 50.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        target_agent_generic_200 = RetAgent(
            model=cep_model,
            pos=(300.0, 600.0, 50.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )
        target_agent_generic_400 = RetAgent(
            model=cep_model,
            pos=(300.0, 800.0, 50.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        cep_sensor = CEPSensor(
            cep_range_data=cep_data,
            distance_thresholds=SensorDistanceThresholds(450.0, 250.0, 150.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
            clutter_attenuator=SensorClutterAttenuator(1.0),
        )

        cep_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=90)

        perceived_agents = cep_sensor.get_results(sensor_agent=sensor_agent)

        assert len(perceived_agents) == 3

        self.assert_count_confidence_levels(perceived_agents, Confidence.DETECT, 1)
        self.assert_count_confidence_levels(perceived_agents, Confidence.RECOGNISE, 1)
        self.assert_count_confidence_levels(perceived_agents, Confidence.IDENTIFY, 1)

        for original_agent, distance in [
            (target_agent_generic_100, 10.0),
            (target_agent_generic_200, 20.0),
            (target_agent_generic_400, 40.0),
        ]:
            self.assert_single_close_agent(
                cep_model.space, perceived_agents, original_agent.pos, distance
            )

    def test_cep_sensor_offset(self):
        """Test CEP sensor offset location."""
        cep_detect_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }

        cep_sensor = CEPSensor(
            cep_range_data=cep_detect_data,
            distance_thresholds=SensorDistanceThresholds(450.0, 250.0, 150.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
            clutter_attenuator=SensorClutterAttenuator(1.0),
        )

        cep_model = MockModel3d(seed=1.0)

        original_location = (10.0, 10.0, 10.0)
        offset_distance = 5.0
        offset_location = cep_sensor._offset_coordinate(
            original_location, offset_distance, cep_model
        )
        distance = cep_model.space.get_distance(original_location, offset_location)

        assert math.isclose(distance, offset_distance)

    def test_cep_sensor_offset_at_terrain_level(self):
        """Test CEP sensor offset location at terrain level."""
        cep_detect_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }

        cep_sensor = CEPSensor(
            cep_range_data=cep_detect_data,
            distance_thresholds=SensorDistanceThresholds(450.0, 250.0, 150.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
            clutter_attenuator=SensorClutterAttenuator(1.0),
        )

        cep_model = MockModel3d(seed=1.0)

        space: ContinuousSpaceWithTerrainAndCulture3d = cep_model.space
        with warnings.catch_warnings(record=True) as w:
            # no z dimension so location is at terrain level
            original_location = space.get_coordinate_3d((10.0, 10.0))
            offset_distance = 5.0
            offset_location = cep_sensor._offset_coordinate(
                original_location, offset_distance, cep_model
            )
            distance = space.get_distance(original_location, offset_location)
            assert math.isclose(distance, offset_distance)
            assert len(w) == 0

    def test_cep_sensor_2d_coordinates(self):
        """Test CEP sensor confidence levels when agents are all given 2d coordinates."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }

        cep_model = MockModel3d(seed=1.0)
        with warnings.catch_warnings(record=True) as w:
            sensor_agent = RetAgent(
                model=cep_model,
                pos=(300.0, 400.0),  # no 3rd dimension so agent will be placed at terrain level
                name="Sensor Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            target_agent_generic_100 = RetAgent(
                model=cep_model,
                pos=(300.0, 500.0),
                name="Target Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                agent_type=AgentType.GENERIC,
            )

            target_agent_generic_200 = RetAgent(
                model=cep_model,
                pos=(300.0, 600.0),
                name="Target Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                agent_type=AgentType.GENERIC,
            )
            target_agent_generic_400 = RetAgent(
                model=cep_model,
                pos=(300.0, 800.0),
                name="Target Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                agent_type=AgentType.GENERIC,
            )

            cep_sensor = CEPSensor(
                cep_range_data=cep_data,
                distance_thresholds=SensorDistanceThresholds(600, 500, 400),
                sampling_distance=SensorSamplingDistance(10.0),
                wavelength=SensorWavelength(500.0),
                clutter_attenuator=SensorClutterAttenuator(1.0),
            )

            cep_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=90)

            perceived_agents = cep_sensor.get_results(sensor_agent=sensor_agent)

            for original_agent, distance in [
                (target_agent_generic_100, 10.0),
                (target_agent_generic_200, 20.0),
                (target_agent_generic_400, 40.0),
            ]:
                self.assert_single_close_agent(
                    cep_model.space, perceived_agents, original_agent.pos, distance
                )
            assert len(w) == 1
            assert (
                "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
                in str(w[0].message)
            )

    def test_range_to_cep_data_negative_values(self):
        """Test RangeToCepData exception handling for negative Range/CEP values."""
        with self.assertRaises(ValueError) as e_range:
            RangeToCepData(
                [
                    (0.0, 0.0),
                    (-400.0, 40.0),
                ]
            )
        assert "Range value (-400.0) cannot be negative." == str(e_range.exception)

        with self.assertRaises(ValueError) as e_cep:
            RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, -40.0),
                ]
            )
        assert "CEP value (-40.0) cannot be negative for range 400.0" == str(e_cep.exception)

    def test_cep_sensor_get_new_instance(self):
        """Test the get new instance method."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }
        sensor = CEPSensor(
            cep_range_data=cep_data,
            wavelength=SensorWavelength(500.0),
            distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 400.0),
            detection_timings=SensorDetectionTimings(
                detection_delay=timedelta(seconds=5),
            ),
            clutter_attenuator=SensorClutterAttenuator(1.0),
            sampling_distance=SensorSamplingDistance(10.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
        )
        clone = sensor.get_new_instance()

        assert sensor is not clone
        assert isinstance(clone, CEPSensor)
        assert sensor._cep_range_data == clone._cep_range_data
        assert (
            sensor._detection_timings.get_detection_delay()
            == clone._detection_timings.get_detection_delay()
        )
        assert sensor._distance_thresholds == clone._distance_thresholds
        assert sensor._sampling_distance == clone._sampling_distance
        assert sensor._wavelength == clone._wavelength
        assert sensor._clutter_attenuator == clone._clutter_attenuator
        assert sensor.sensor_filters == clone.sensor_filters
        assert sensor.is_active_sensor == clone.is_active_sensor

    def test_cep_agent_not_detectable(self):
        """Test if _is_agent_detectable returns false for agent out of range."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }
        sensor = CEPSensor(
            cep_range_data=cep_data,
            wavelength=SensorWavelength(500.0),
            distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 400.0),
            detection_timings=SensorDetectionTimings(
                detection_delay=timedelta(seconds=5),
            ),
            clutter_attenuator=SensorClutterAttenuator(1.0),
            sampling_distance=SensorSamplingDistance(10.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
        )
        cep_model = MockModel3d(seed=1.0)
        sensor_agent = RetAgent(
            model=cep_model,
            pos=(0.0, 0.0, 0.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        target_agent_generic_100 = RetAgent(
            model=cep_model,
            pos=(1000.0, 1000.0, 1000.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        assert not sensor._is_agent_detectable(sensor_agent, target_agent_generic_100)

    def test_sig_int_sensor(self):
        """Test Signals Intelligence sensor."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 10.0),
                    (200.0, 20.0),
                    (300.0, 30.0),
                    (400.0, 40.0),
                ]
            ),
        }

        cep_model = MockModel3d(seed=1.0)
        sensor_agent = RetAgent(
            model=cep_model,
            pos=(300.0, 400.0, 50.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        target_agent_generic_100 = RetAgent(
            model=cep_model,
            pos=(300.0, 500.0, 50.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            agent_type=AgentType.GENERIC,
        )

        sig_int_sensor = SigIntSensor(
            cep_range_data=cep_data,
            distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 400.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
            clutter_attenuator=SensorClutterAttenuator(1.0),
        )

        # Check that SigInt sensor action has been added
        sig_int_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=90)
        assert len(cep_model.actively_sensing_agents_manager._actions) == 1

        # Action hasn't been run so no agents should have been perceived
        perceived_agents = sensor_agent.perceived_world.get_perceived_agents(
            Not(AgentById(sensor_agent.unique_id))
        )
        assert len(perceived_agents) == 0
        assert len(sig_int_sensor.get_results(sensor_agent=sensor_agent)) == 0

        # Do SigInt sensor action: there are still no actively sensing agents so no
        # perceived agents should be found
        cep_model.actively_sensing_agents_manager.do_actions_and_clear()
        perceived_agents = sensor_agent.perceived_world.get_perceived_agents(
            Not(AgentById(sensor_agent.unique_id))
        )
        assert len(perceived_agents) == 0
        assert len(sig_int_sensor.get_results(sensor_agent=sensor_agent)) == 0

        # Run SigInt detection again and add an agent to the actively_sensing_agents
        # store then do SigInt sensor action again. Now the actively sensing agent
        # should be found
        sig_int_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=90)
        cep_model.actively_sensing_agents_manager.add_agent(target_agent_generic_100)
        cep_model.actively_sensing_agents_manager.do_actions_and_clear()
        perceived_agents = sensor_agent.perceived_world.get_perceived_agents(
            Not(AgentById(sensor_agent.unique_id))
        )
        assert len(perceived_agents) == 1
        self.assert_count_confidence_levels(perceived_agents, Confidence.IDENTIFY, 1)
        self.assert_single_close_agent(
            cep_model.space,
            perceived_agents,
            target_agent_generic_100.pos,
            15.0,
        )

    def test_sig_int_sensor_with_clutter(self):
        """Test Signals Intelligence sensor with cluttered space."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 10.0),
                    (400.0, 40.0),
                ]
            ),
        }
        with warnings.catch_warnings(record=True) as w:
            # All culture attenuated by 1.1 in SensorTestModel and
            # clutter value of 10 between 0-10 above terrain.
            cep_model = SensorTestModel(seed=1.0)
            sensor_agent = RetAgent(
                model=cep_model,
                pos=(300.0, 400.0),  # place at ground level
                name="Sensor Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            # Target agent is distance 400 from sensor agent
            target_agent_generic_100 = RetAgent(
                model=cep_model,
                pos=(300.0, 800.0),  # place at ground level
                name="Target Agent",
                affiliation=Affiliation.FRIENDLY,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
                agent_type=AgentType.GENERIC,
            )

            sig_int_sensor_with_clutter_attenuator = SigIntSensor(
                cep_range_data=cep_data,
                distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 445.0),
                sampling_distance=SensorSamplingDistance(10.0),
                wavelength=SensorWavelength(500.0),
                clutter_attenuator=SensorClutterAttenuator(1.0),
            )
            assert len(w) == 1
            assert (
                "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
                in str(w[0].message)
            )

        def get_sig_int_perceived_agents(sig_int_sensor: SigIntSensor):
            sig_int_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=90)
            cep_model.actively_sensing_agents_manager.add_agent(target_agent_generic_100)
            cep_model.actively_sensing_agents_manager.do_actions_and_clear()
            return sensor_agent.perceived_world.get_perceived_agents(
                Not(AgentById(sensor_agent.unique_id))
            )

        perceived_agents = get_sig_int_perceived_agents(sig_int_sensor_with_clutter_attenuator)
        assert len(perceived_agents) == 1
        # Clutter attenuation takes distance from 440 (culture attenuated) to 450 so outside
        # of "Identify" distance and into "Recognise".
        self.assert_count_confidence_levels(perceived_agents, Confidence.RECOGNISE, 1)
        self.assert_single_close_agent(
            cep_model.space,
            perceived_agents,
            target_agent_generic_100.pos,
            40.0,
        )

        sig_int_sensor_without_clutter_attenuator = SigIntSensor(
            cep_range_data=cep_data,
            distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 445.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
        )

        perceived_agents = get_sig_int_perceived_agents(sig_int_sensor_without_clutter_attenuator)
        assert len(perceived_agents) == 1
        # No clutter attenuation so culture attenuated distance 440 falls within "Identify" range
        self.assert_count_confidence_levels(perceived_agents, Confidence.IDENTIFY, 1)
        self.assert_single_close_agent(
            cep_model.space,
            perceived_agents,
            target_agent_generic_100.pos,
            40.0,
        )

    def test_sig_int_sensor_get_new_instance(self):
        """Test the get new instance method."""
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }
        sensor = SigIntSensor(
            cep_range_data=cep_data,
            wavelength=SensorWavelength(500.0),
            distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 400.0),
            detection_timings=SensorDetectionTimings(
                detection_delay=timedelta(seconds=5),
            ),
            clutter_attenuator=SensorClutterAttenuator(1.0),
            sampling_distance=SensorSamplingDistance(10.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
            height_of_sensor=6,
        )
        clone = sensor.get_new_instance()

        assert sensor is not clone
        assert isinstance(clone, SigIntSensor)
        assert sensor._cep_range_data == clone._cep_range_data
        assert (
            sensor._detection_timings.get_detection_delay()
            == clone._detection_timings.get_detection_delay()
        )
        assert sensor._distance_thresholds == clone._distance_thresholds
        assert sensor._sampling_distance == clone._sampling_distance
        assert sensor._wavelength == clone._wavelength
        assert sensor._clutter_attenuator == clone._clutter_attenuator
        assert sensor.sensor_filters == clone.sensor_filters
        assert sensor.is_active_sensor == clone.is_active_sensor
        assert sensor.height_of_sensor == clone.height_of_sensor

    @parameterized.expand(
        [
            [0, (0, 0, 0), (0, 0, 0)],
            [5, (0, 0, 0), (0, 0, 5)],
            [0.01, (0, 0, 0), (0, 0, 0.01)],
            [-2.3, (0, 0, 0), (0, 0, -2.3)],
            [0, (2, 3, 4), (2, 3, 4)],
            [5, (2, 3, 4), (2, 3, 9)],
            [0.01, (2, 3, 4), (2, 3, 4.01)],
            [-2.3, (2, 3, 4), (2, 3, 1.7)],
            [0, (-6, 0.8, 0), (-6, 0.8, 0)],
            [5, (-6, 0.8, -5), (-6, 0.8, 0)],
            [0.01, (-6, 0.8, -0.01), (-6, 0.8, 0)],
            [-2.3, (-6, 0.8, 2.3), (-6, 0.8, 0)],
            [-2.3, (-6, 0.8), (-6, 0.8)],
            [0, (-6, 0.8), (-6, 0.8)],
            [150, (0, 0.8), (0, 0.8)],
        ]
    )
    def test_sensor_height_sig_int(
        self,
        height: float,
        agent_position: tuple[float, float, float],
        sensor_position: tuple[float, float, float],
    ):
        """Test the sensor height sets and returns correctly.

        Args:
            height (float): Height of sensor above agent.
            agent_position (tuple[float, float, float]): The position of the agent with the sensor.
            sensor_position (tuple[float, float, float]): The expected position of sensor.
        """
        cep_data: dict[AgentType, RangeToCepData] = {
            AgentType.GENERIC: RangeToCepData(
                [
                    (0.0, 0.0),
                    (400.0, 40.0),
                ]
            ),
        }
        sensor = SigIntSensor(
            cep_range_data=cep_data,
            wavelength=SensorWavelength(500.0),
            distance_thresholds=SensorDistanceThresholds(600.0, 500.0, 400.0),
            detection_timings=SensorDetectionTimings(
                detection_delay=timedelta(seconds=5),
            ),
            clutter_attenuator=SensorClutterAttenuator(1.0),
            sampling_distance=SensorSamplingDistance(10.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
            height_of_sensor=height,
        )
        sensor_pos = sensor.get_elevated_sensor_position(agent_position)
        assert len(sensor_pos) == len(agent_position)
        assert math.isclose(sensor_pos[0], sensor_position[0], abs_tol=0.0001)
        assert math.isclose(sensor_pos[1], sensor_position[1], abs_tol=0.0001)
        if len(agent_position) == 3:
            assert math.isclose(sensor_pos[2], sensor_position[2], abs_tol=0.0001)  # type: ignore
