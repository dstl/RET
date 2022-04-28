"""Tests for sensors."""
from __future__ import annotations

import os
import unittest
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import Mock

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import AgentSensedStatus, RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.sensing.distribution import TriangularDistribution
from mesa_ret.sensing.perceivedworld import Confidence
from mesa_ret.sensing.sensor import (
    ActiveFilter,
    ActivelySensingAgentsManager,
    CultureAttenuatedSensor,
    DistanceAttenuatedSensor,
    LineOfSightSensor,
    NotifySensedBySideEffect,
    PassiveFilter,
    SensorClutterAttenuator,
    SensorDetectionTimings,
    SensorDistanceThresholds,
    SensorSamplingDistance,
    SensorWavelength,
)
from mesa_ret.space.culture import Culture
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
from mesa_ret.testing.mocks import MockModel3d
from parameterized import parameterized

if TYPE_CHECKING:

    from typing import Any

culture_red = Culture("red culture", 11, wavelength_attenuation_factors={(450, 550): 1.1})
culture_green = Culture("green culture", 11, wavelength_attenuation_factors={(450, 550): 1.1})
culture_blue = Culture("blue culture", 11, wavelength_attenuation_factors={(450, 550): 1.1})
culture_yellow = Culture("yellow culture", 11, wavelength_attenuation_factors={(450, 550): 1.1})


class SensorTestModel(MockModel3d):
    """Initialise model for sensor tests."""

    width = 1000
    height = 1000

    culture_map = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "test_space/TestCulture.png")
    )

    terrain_image_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "test_space/TestLineOfSight.png")
    )

    culture_dictionary = {
        (255, 0, 0): culture_red,
        (0, 255, 0): culture_green,
        (0, 0, 255): culture_blue,
        (255, 255, 0): culture_yellow,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Create a new SensorTestModel."""
        super().__init__(time_step=timedelta(seconds=10), kwargs=kwargs)

        self.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=self.width,
            y_max=self.height,
            terrain_image_path=self.terrain_image_path,
            culture_image_path=self.culture_map,
            culture_dictionary=self.culture_dictionary,
            clutter_background_level=0.0,
            ground_clutter_value=10.0,
            ground_clutter_height=10.0,
        )


class TestSensor(unittest.TestCase):
    """Test sensor class calculations."""

    def setUp(self):
        """Set up test cases."""
        self.model = SensorTestModel()

        self.sensor_agent = RetAgent(
            model=self.model,
            pos=(300.0, 1.0, 1.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.target_agent_21 = RetAgent(
            model=self.model,
            pos=(300.0, 1.0, 21.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.target_agent_199 = RetAgent(
            model=self.model,
            pos=(300.0, 199.0, 1.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.target_agent_299 = RetAgent(
            model=self.model,
            pos=(300.0, 299.0, 1.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.target_agent_399 = RetAgent(
            model=self.model,
            pos=(300.0, 399.0, 1.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.target_agent_410 = RetAgent(
            model=self.model,
            pos=(300.0, 410.0, 1.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.target_agent_no_LOS = RetAgent(
            model=self.model,
            pos=(650.0, 1.0, 1.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

    def test_distance_attenuated_sensor(self):
        """Test distance attenuated sensor gives correct results."""
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
        )

        distance_attenuated_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=90)

        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        assert len(perceived_agents) == 5

        assert (
            len([agent for agent in perceived_agents if agent.confidence == Confidence.IDENTIFY])
            == 2
        )

        assert (
            len([agent for agent in perceived_agents if agent.confidence == Confidence.RECOGNISE])
            == 1
        )

        assert (
            len([agent for agent in perceived_agents if agent.confidence == Confidence.DETECT]) == 2
        )

    def test_casualty_state_confidence_default(self):
        """Test sensor returns correct casualty state based on confidence setting.

        Tested using default confidence setting, which defaults to RECOGNISE.
        """
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
        )

        distance_attenuated_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=90)

        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        unknown_casualty_state_agents = [
            p for p in perceived_agents if p.casualty_state is AgentCasualtyState.UNKNOWN
        ]

        assert len(unknown_casualty_state_agents) == 2

    def test_casualty_state_confidence_identify(self):
        """Test sensor returns correct casualty state based on confidence setting.

        Tested using IDENTIFY confidence setting.
        """
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            casualty_state_confidence=Confidence.IDENTIFY,
        )

        distance_attenuated_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=90)

        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        unknown_casualty_state_agents = [
            p for p in perceived_agents if p.casualty_state is AgentCasualtyState.UNKNOWN
        ]

        assert len(unknown_casualty_state_agents) == 3

    def test_los_sensor(self):
        """Test LOS sensor gives correct results."""
        los_sensor = LineOfSightSensor(
            distance_thresholds=SensorDistanceThresholds(1500.0, 1400.0, 1300.0),
        )

        los_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=90)

        perceived_agents = los_sensor.get_results(sensor_agent=self.sensor_agent)

        assert len(perceived_agents) == 5

    def test_culture_attenuated_sensor_through_culture(self):
        """Test culture attenuated sensor gives correct results."""
        culture_attenuated_sensor = CultureAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400.0, 300.0, 200.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
        )

        culture_attenuated_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=90)

        perceived_agents = culture_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        assert len(perceived_agents) == 3

    def test_culture_attenuated_sensor_through_part_culture(self):
        """Test LOS through culture.

        Test culture attenuated sensor gives correct results when line of sight
        partially passes through culture.

        - Closest agent is 20 units away
            - 10 units through culture with attenuation factor of 1.1
            - 10 units through no culture

        - Attenuated distance is (10 * 1.1) + 10 = 21
            - Sensor with max range of 20.5 should detect 0 agents
            - Sensor with max range of 21.5 should detect 1 agent
        """
        culture_attenuated_sensor_0_detections = CultureAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(20.5, 20.4, 20.3),
            sampling_distance=SensorSamplingDistance(0.01),
            wavelength=SensorWavelength(500.0),
        )

        culture_attenuated_sensor_1_detection = CultureAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(21.5, 21.4, 21.3),
            sampling_distance=SensorSamplingDistance(0.01),
            wavelength=SensorWavelength(500.0),
        )

        culture_attenuated_sensor_0_detections.run_detection(
            sensor_agent=self.sensor_agent, sense_direction=90
        )
        culture_attenuated_sensor_1_detection.run_detection(
            sensor_agent=self.sensor_agent, sense_direction=90
        )

        perceived_agents_sensor_0 = culture_attenuated_sensor_0_detections.get_results(
            sensor_agent=self.sensor_agent
        )
        perceived_agents_sensor_1 = culture_attenuated_sensor_1_detection.get_results(
            sensor_agent=self.sensor_agent
        )

        assert len(perceived_agents_sensor_0) == 0
        assert len(perceived_agents_sensor_1) == 1

    def test_clutter_culture_attenuated_sensor(self):
        """Test clutter attenuated sensor gives correct results."""
        clutter_culture_attenuated_sensor = CultureAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400.0, 300.0, 200.0),
            sampling_distance=SensorSamplingDistance(10.0),
            wavelength=SensorWavelength(500.0),
            clutter_attenuator=SensorClutterAttenuator(1.0),
        )

        clutter_culture_attenuated_sensor.run_detection(
            sensor_agent=self.sensor_agent, sense_direction=90
        )

        perceived_agents = clutter_culture_attenuated_sensor.get_results(
            sensor_agent=self.sensor_agent
        )

        assert len(perceived_agents) == 3

    def test_invalid_distance_thresholds_conditions(self):
        """Test exception raised given invalid distance thresholds."""
        with self.assertRaises(ValueError) as e:
            SensorDistanceThresholds(200, 300, 400)
        self.assertEqual(
            "Maximum identify distance must be less than maximum recognise "
            + "distance, which must be less than maximum detect distance.",
            str(e.exception),
        )

    def test_negative_clutter_attenuator(self):
        """Test warning given negative clutter attenuation factor."""
        with warnings.catch_warnings(record=True) as w:
            SensorClutterAttenuator(-1.0)
        assert len(w) == 1
        assert "Attenuation strength provided is negative" in str(w[0].message)

    def test_negative_wavelength(self):
        """Test SensorWavelength raises exception given negative wavelength."""
        with self.assertRaises(ValueError) as e:
            SensorWavelength(-500.0)
        self.assertEqual(
            "Wavelength cannot be negative",
            str(e.exception),
        )

    def test_negative_delay(self):
        """Test SensorDetectionTimings raises exception with negative delay."""
        with self.assertRaises(ValueError) as e:
            SensorDetectionTimings(detection_delay=timedelta(-5.0))
        self.assertEqual(
            "Detection delay cannot be negative",
            str(e.exception),
        )

    def test_sensor_timings(self):
        """Test distance attenuated sensor gives correct results after detection delay.

        Sensor checked at t = 0, no agents should be returned, checked again after a
        model time step (which is longer than the detection delay), 4 agents should be
        returned.
        """
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            detection_timings=SensorDetectionTimings(
                detection_delay=timedelta(seconds=5),
            ),
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
        )

        distance_attenuated_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=90)

        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        assert len(perceived_agents) == 0

        self.model.step()

        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        assert len(perceived_agents) == 5

    def test_negative_detection_delay(self):
        """Test SensorDetectionTimings raises exception given negative detection delay.

        Tested with negative timedelta value, and triangular distribution that returns
        negative value.
        """
        with self.assertRaises(ValueError) as e:
            SensorDetectionTimings(detection_delay=timedelta(seconds=-5))
        self.assertEqual("Detection delay cannot be negative", str(e.exception))
        with self.assertRaises(ValueError) as e:
            detection_timings = SensorDetectionTimings(
                detection_delay=TriangularDistribution(lower_limit=-20, upper_limit=-10, mode=-15)
            )
            detection_timings.get_detection_delay()
        self.assertEqual(
            "Detection delay sampled from sensor distribution is negative.",
            str(e.exception),
        )

    def test_distance_attenuated_sensor_get_new_instance(self):
        """Test the get new instance method."""
        sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            detection_timings=SensorDetectionTimings(detection_delay=timedelta(seconds=30)),
            clutter_attenuator=SensorClutterAttenuator(10.0),
            casualty_state_confidence=Confidence.IDENTIFY,
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
        )
        clone = sensor.get_new_instance()

        assert sensor is not clone
        assert isinstance(clone, DistanceAttenuatedSensor)
        assert (
            sensor._distance_thresholds.confidence_distances
            == clone._distance_thresholds.confidence_distances
        )
        assert (
            sensor._detection_timings.get_detection_delay()
            == clone._detection_timings.get_detection_delay()
        )
        assert (
            sensor._clutter_attenuator.attenuation_strength
            == clone._clutter_attenuator.attenuation_strength
        )
        assert sensor._casualty_state_confidence == clone._casualty_state_confidence
        assert sensor.is_active_sensor == clone.is_active_sensor
        assert sensor.sensor_filters == clone.sensor_filters

    def test_los_sensor_get_new_instance(self):
        """Test the get new instance method."""
        sensor = LineOfSightSensor(
            distance_thresholds=SensorDistanceThresholds(1500.0, 1400.0, 1300.0),
            detection_timings=SensorDetectionTimings(detection_delay=timedelta(seconds=30)),
            clutter_attenuator=SensorClutterAttenuator(10.0),
            sampling_distance=SensorSamplingDistance(5.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
        )
        clone = sensor.get_new_instance()

        assert sensor is not clone
        assert isinstance(clone, LineOfSightSensor)
        assert (
            sensor._distance_thresholds.confidence_distances
            == clone._distance_thresholds.confidence_distances
        )
        assert (
            sensor._detection_timings.get_detection_delay()
            == clone._detection_timings.get_detection_delay()
        )
        assert (
            sensor._clutter_attenuator.attenuation_strength
            == clone._clutter_attenuator.attenuation_strength
        )
        assert (
            sensor._sampling_distance.sampling_distance
            == clone._sampling_distance.sampling_distance
        )
        assert sensor.is_active_sensor == clone.is_active_sensor
        assert sensor.sensor_filters == clone.sensor_filters

    def test_culture_attenuated_sensor_get_new_instance(self):
        """Test the get new instance method."""
        sensor = CultureAttenuatedSensor(
            wavelength=SensorWavelength(500.0),
            distance_thresholds=SensorDistanceThresholds(400.0, 300.0, 200.0),
            detection_timings=SensorDetectionTimings(detection_delay=timedelta(seconds=30)),
            clutter_attenuator=SensorClutterAttenuator(10.0),
            sampling_distance=SensorSamplingDistance(5.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
        )
        clone = sensor.get_new_instance()

        assert sensor is not clone
        assert isinstance(clone, CultureAttenuatedSensor)
        assert sensor._wavelength.wavelength == clone._wavelength.wavelength
        assert (
            sensor._distance_thresholds.confidence_distances
            == clone._distance_thresholds.confidence_distances
        )
        assert (
            sensor._detection_timings.get_detection_delay()
            == clone._detection_timings.get_detection_delay()
        )
        assert (
            sensor._clutter_attenuator.attenuation_strength
            == clone._clutter_attenuator.attenuation_strength
        )
        assert (
            sensor._sampling_distance.sampling_distance
            == clone._sampling_distance.sampling_distance
        )
        assert sensor.is_active_sensor == clone.is_active_sensor
        assert sensor.sensor_filters == clone.sensor_filters

    def test_active_sensor(self):
        """Test that agents are recorded correctly when using active sensors."""
        distance_attenuated_sensor_active = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            is_active_sensor=True,
        )
        distance_attenuated_sensor_passive = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            is_active_sensor=False,
        )

        assert len(self.model.actively_sensing_agents_manager._actively_sensing_agents) == 0

        distance_attenuated_sensor_active.run_detection(
            sensor_agent=self.sensor_agent, sense_direction=90
        )
        distance_attenuated_sensor_passive.run_detection(
            sensor_agent=self.target_agent_21, sense_direction=90
        )

        actively_sensing_agents = (
            self.model.actively_sensing_agents_manager._actively_sensing_agents
        )
        assert len(actively_sensing_agents) == 1
        assert actively_sensing_agents[0] == self.sensor_agent

    def test_actively_sensing_agents_manager_add_and_clear(self):
        """Test actively sensing agents manager adding and clearing data."""
        manager = ActivelySensingAgentsManager()
        assert len(manager._actively_sensing_agents) == 0
        assert len(manager._actions) == 0

        manager.add_agent(self.sensor_agent)
        assert len(manager._actively_sensing_agents) == 1
        assert manager._actively_sensing_agents[0] == self.sensor_agent

        actions_test = []

        manager.add_action(lambda agents: actions_test.append(0))
        assert len(manager._actions) == 1

        manager.do_actions_and_clear()
        assert len(actions_test) == 1
        assert len(manager._actively_sensing_agents) == 0
        assert len(manager._actions) == 0

    def test_active_sensor_when_hiding(self):
        """Test active sensing is disallowed when agent is hiding."""
        self.sensor_agent.hiding = True
        sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            is_active_sensor=True,
        )
        with warnings.catch_warnings(record=True) as w:
            sensor.run_detection(self.sensor_agent, sense_direction=90)
            assert len(w) == 1
            assert "Cannot actively sense when a sensor agent is hiding." == str(w[0].message)

    def test_passive_sensor_when_hiding(self):
        """Test passive sensing is allowed when agent is hiding."""
        self.sensor_agent.hiding = True
        sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            is_active_sensor=False,
        )
        with warnings.catch_warnings(record=True) as w:
            sensor.run_detection(self.sensor_agent, sense_direction=90)
            assert len(w) == 0

    def test_perceived_agents_with_filter(self):
        """Test the perceived agents are valid when an active filter is applied."""
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
        )
        self.target_agent_21.hiding = True
        self.target_agent_199.hiding = True
        self.target_agent_299.hiding = False
        self.target_agent_399.hiding = True
        self.target_agent_410.hiding = True
        self.target_agent_no_LOS.hiding = True

        distance_attenuated_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=0)

        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        assert len(perceived_agents) == 1

        assert (
            len([agent for agent in perceived_agents if agent.confidence == Confidence.RECOGNISE])
            == 1
        )

    def test_perceived_agents_with_side_effect(self):
        """Test the sensor notify sensed by side effect is applied."""
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            sensor_side_effects=[NotifySensedBySideEffect()],
        )

        distance_attenuated_sensor.run_detection(sensor_agent=self.sensor_agent, sense_direction=0)
        perceived_agents = distance_attenuated_sensor.get_results(sensor_agent=self.sensor_agent)

        found_agents = [
            self.target_agent_21,
            self.target_agent_299,
            self.target_agent_399,
            self.target_agent_no_LOS,
        ]
        non_found_agents = [self.target_agent_410]

        for found_agent in found_agents:
            assert len(found_agent._statuses) == 1
            assert isinstance(found_agent._statuses[0], AgentSensedStatus)
            assert found_agent._statuses[0].sensing_agent == self.sensor_agent
            perceived_agent = next(
                (agent for agent in perceived_agents if agent._unique_id == found_agent.unique_id),
                None,
            )
            assert perceived_agent is not None
            assert found_agent._statuses[0].time_step == perceived_agent.sense_time

        for non_found_agent in non_found_agents:
            assert len(non_found_agent._statuses) == 0
            assert (
                len(
                    [
                        agent
                        for agent in perceived_agents
                        if agent._unique_id == non_found_agent.unique_id
                    ]
                )
                == 0
            )

    def test_perceived_agents_with_side_effect_multiple_steps(self):
        """Test the sensor notify sensed by side effect is applied correctly over multiple steps."""
        distance_attenuated_sensor = DistanceAttenuatedSensor(
            distance_thresholds=SensorDistanceThresholds(400, 300, 200),
            sensor_side_effects=[NotifySensedBySideEffect()],
        )
        model = SensorTestModel()
        sensor_agent = RetAgent(
            model=model,
            pos=(0.0, 0.0, 1.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        agent_a = RetAgent(
            model=model,
            pos=(0.0, 100.0, 1.0),
            name="Target Agent A",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        agent_b = RetAgent(
            model=model,
            pos=(0.0, 200.0, 1.0),
            name="Target Agent B",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        agent_c = RetAgent(
            model=model,
            pos=(0.0, 500.0, 1.0),
            name="Target Agent C",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        distance_attenuated_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=0)
        time_step1 = model.get_time()

        assert len(agent_a._statuses) == 1
        assert len(agent_b._statuses) == 1
        assert len(agent_c._statuses) == 0

        assert isinstance(agent_a._statuses[0], AgentSensedStatus)
        assert agent_a._statuses[0].sensing_agent == sensor_agent
        assert agent_a._statuses[0].time_step == time_step1

        assert isinstance(agent_b._statuses[0], AgentSensedStatus)
        assert agent_b._statuses[0].sensing_agent == sensor_agent
        assert agent_b._statuses[0].time_step == time_step1

        sensor_agent.model.space.move_agent(agent_b, (0.0, 600.0, 1.0))
        sensor_agent.model.space.move_agent(agent_c, (0.0, 300.0, 1.0))

        model.step()
        distance_attenuated_sensor.run_detection(sensor_agent=sensor_agent, sense_direction=0)
        time_step2 = model.get_time()

        assert len(agent_a._statuses) == 2
        assert len(agent_b._statuses) == 1
        assert len(agent_c._statuses) == 1

        assert isinstance(agent_a._statuses[0], AgentSensedStatus)
        assert agent_a._statuses[0].sensing_agent == sensor_agent
        assert agent_a._statuses[0].time_step == time_step1
        assert isinstance(agent_a._statuses[1], AgentSensedStatus)
        assert agent_a._statuses[1].sensing_agent == sensor_agent
        assert agent_a._statuses[1].time_step == time_step2

        assert isinstance(agent_b._statuses[0], AgentSensedStatus)
        assert agent_b._statuses[0].sensing_agent == sensor_agent
        assert agent_b._statuses[0].time_step == time_step1

        assert isinstance(agent_c._statuses[0], AgentSensedStatus)
        assert agent_c._statuses[0].sensing_agent == sensor_agent
        assert agent_c._statuses[0].time_step == time_step2

    @parameterized.expand(
        [
            [0, 0],
            [5, 0],
            [55, 1],
        ]
    )
    def test_sensor_look_over_culture(
        self,
        height: float,
        number_of_perceived_agents: int,
    ):
        """Test that the sensor height can make the difference between seeing over culture or not.

        Args:
            height (float): Height of the sensor above the agent.
            number_of_perceived_agents (int): The expected number of agents perceived.
        """
        culture_red = Culture("red culture", 0.0)
        culture_green = Culture("green culture", 0.0)
        culture_blue = Culture("blue culture", 0.0)
        culture_yellow = Culture(
            "yellow culture", 50.0, wavelength_attenuation_factors={(450, 550): 100.1}
        )

        sensor = CultureAttenuatedSensor(
            wavelength=SensorWavelength(500.0),
            distance_thresholds=SensorDistanceThresholds(400.0, 300.0, 200.0),
            sampling_distance=SensorSamplingDistance(5.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
            height_of_sensor=height,
        )
        model = MockModel3d()
        model.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            x_min=0,
            y_min=0,
            culture_image_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "test_space/TestCulture.png")
            ),
            culture_dictionary={
                (255, 0, 0): culture_red,
                (0, 255, 0): culture_green,
                (0, 0, 255): culture_blue,
                (255, 255, 0): culture_yellow,
            },
        )
        # random number generator always returns fixed probability
        model.random.random = Mock(return_value=0.5)

        sensor_agent = RetAgent(
            model=model,
            pos=(49.0, 75.0, 0.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        RetAgent(
            model=model,
            pos=(75.0, 75.0, 100.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        sensor.run_detection(sensor_agent, sense_direction=90)

        perceived_agents = sensor.get_results(sensor_agent)

        assert len(perceived_agents) == number_of_perceived_agents

    @parameterized.expand(
        [
            [0, 0],
            [50, 0],  # 41 + 50 = 91
            [60, 1],  # 41 + 60 = 101
        ]
    )
    def test_sensor_look_over_terrain(
        self,
        height: float,
        number_of_perceived_agents: int,
    ):
        """Test that the sensor height can make the difference between seeing over terrain or not.

        Args:
            height (float): Height of the sensor above the agent.
            number_of_perceived_agents (int): The expected number of agents perceived.
        """
        sensor = LineOfSightSensor(
            distance_thresholds=SensorDistanceThresholds(400.0, 300.0, 200.0),
            sampling_distance=SensorSamplingDistance(5.0),
            is_active_sensor=True,
            sensor_filters=[ActiveFilter()],
            height_of_sensor=height,
        )
        model = MockModel3d()
        model.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            x_min=0,
            y_min=0,
            terrain_image_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "test_space/TestTerrain.png")
            ),
        )
        # random number generator always returns fixed probability
        model.random.random = Mock(return_value=0.5)

        sensor_agent = RetAgent(
            model=model,
            pos=(49.0, 75.0),  # place at terrain height (41.0)
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        RetAgent(
            model=model,
            pos=(75.0, 75.0),  # place at terrain height (100.0)
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        sensor.run_detection(sensor_agent, sense_direction=90)

        perceived_agents = sensor.get_results(sensor_agent)

        assert len(perceived_agents) == number_of_perceived_agents


class TestSensorFilters(unittest.TestCase):
    """Test sensorfilter class."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel3d()

        self.agent_1 = RetAgent(
            model=self.model,
            pos=(1.0, 1.0, 1.0),
            name="Agent Open",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=1.0,
            temperature=20,
        )
        self.agent_1.hiding = False

        self.agent_2 = RetAgent(
            model=self.model,
            pos=(1.0, 1.0, 1.0),
            name="Agent Hiding",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=1.0,
            temperature=20,
        )
        self.agent_2.hiding = True

    def test_active_filter_include_agent(self):
        """Test the include_agent method for the ActiveFilter Class."""
        sensor_filter = ActiveFilter()

        assert sensor_filter.include_agent(self.agent_1) is True
        assert sensor_filter.include_agent(self.agent_2) is False

    def test_passive_filter_include_agent(self):
        """Test the include_agent method for the PassiveFilter Class."""
        sensor_filter = PassiveFilter()

        assert sensor_filter.include_agent(self.agent_1) is True
        assert sensor_filter.include_agent(self.agent_2) is True
