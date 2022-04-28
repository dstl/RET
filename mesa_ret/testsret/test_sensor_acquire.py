"""Tests for 1d Acquire algorithm sensors."""

from __future__ import annotations

import math
import os
import unittest
from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import Mock

from mesa_ret.agents.agent import Affiliation, RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.sensing.perceivedworld import Confidence
from mesa_ret.sensing.sensor import (
    ContrastAcquire1dSensor,
    EOContrastSensorType,
    IIContrastSensorType,
    SensorDetectionTimings,
    SensorSamplingDistance,
    SensorWavelength,
    TemperatureAcquire1dSensor,
)
from mesa_ret.space.culture import Culture, default_culture
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture3d, Precipitation
from mesa_ret.testing.mocks import MockModel3d
from parameterized import parameterized

from testsret.test_sensor import SensorTestModel

if TYPE_CHECKING:
    from typing import Any, Optional

    from mesa_ret.sensing.sensor import JohnsonCriteria, PerformanceCurve


class TestAcquireSensor(unittest.TestCase):
    """Test 1d Acquire algorithm sensor class calculations."""

    def setUp(self):
        """Set up test cases."""
        self.contrast_performance_curve: PerformanceCurve = [
            (0.00001, 0.00001),
            (0.2, 0.4),
            (0.5, 1.0),
            (1.0, 2.0),
        ]
        self.temperature_performance_curve: PerformanceCurve = [
            (0.00001, 0.00001),
            (4.0, 0.4),
            (10.0, 1.0),
            (20.0, 2.0),
        ]
        self.magnification = 1.0
        self.johnson_criteria: JohnsonCriteria = {
            Confidence.DETECT: 0.75,
            Confidence.RECOGNISE: 4.0,
            Confidence.IDENTIFY: 8.0,
        }

        self.model = MockModel3d()

        self.target_agent = RetAgent(
            model=self.model,
            pos=(300.0, 400.0, 0.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.eo_sensor = ContrastAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.contrast_performance_curve,
            johnson_criteria=self.johnson_criteria,
            sensor_type=EOContrastSensorType(),
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
        )

        self.ii_sensor = ContrastAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.contrast_performance_curve,
            johnson_criteria=self.johnson_criteria,
            sensor_type=IIContrastSensorType(),
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
        )

        self.ti_sensor = TemperatureAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.temperature_performance_curve,
            johnson_criteria=self.johnson_criteria,
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
        )

    @staticmethod
    def assert_probabilities_close(
        probabilities: tuple[float, float, float],
        expected_probabilities: tuple[float, float, float],
    ):
        """Assert two tuples of probabilities are equal.

        Args:
            probabilities (tuple[float, float, float]): The calculated probabilities
                (p_detect, p_recognise, p_identify)
            expected_probabilities (tuple[float, float, float]): The expected probabilities
                (p_detect, p_recognise, p_identify)
        """
        for probability, expected_probability in zip(probabilities, expected_probabilities):
            assert math.isclose(probability, expected_probability, abs_tol=0.000001)

    def test_acquire_1d_eo_sensor_get_observation_confidence_probabilities(self):
        """Test 1D Acquire EO sensor get observation confidence probabilities."""
        probabilities = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )
        self.assert_probabilities_close(probabilities, (0.965571, 0.065717, 0.006876))

    def test_acquire_1d_eo_sensor_get_observation_confidence_probabilities_zero_attenuated(self):
        """Test 1D Acquire EO sensor confidence probabilities with zero attenuated distance."""
        probabilities = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 0, self.target_agent, timedelta(minutes=5)
        )
        self.assertEqual(probabilities, (1, 1, 1))

    def test_acquire_1d_ii_sensor_get_observation_confidence_probabilities(self):
        """Test 1D Acquire II sensor get observation confidence probabilities."""
        probabilities = self.ii_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )
        self.assert_probabilities_close(probabilities, (0.96462, 0.064836, 0.006731))

    def test_acquire_1d_ti_sensor_get_observation_confidence_probabilities(self):
        """Test 1D Acquire TI sensor get observation confidence probabilities."""
        probabilities = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )
        self.assert_probabilities_close(probabilities, (0.999592, 0.335633, 0.063222))

    @parameterized.expand(
        [
            [1000, 0.75, Precipitation.CLEAR, 0.0, 320.0, 0.968289],
            [5000, 0.75, Precipitation.CLEAR, 0.0, 320.0, 0.851186],  # Vary distance
            [1000, 0.2, Precipitation.CLEAR, 0.0, 320.0, 0.968289],  # Vary relative humidity
            [1000, 0.75, Precipitation.RAIN_MODERATE, 0.0, 320.0, 0.167294],  # Vary precipitation
            [1000, 0.75, Precipitation.CLEAR, 0.5, 320.0, 0.825121],  # Vary dust
            [1000, 0.75, Precipitation.CLEAR, 0.0, 5.0, 0.448252],  # Vary meteorological vis.
        ]
    )
    def test_acquire_1d_eo_sensor_get_atmospheric_transmission_factor(
        self,
        distance: float,
        relative_humidity: float,
        precipitation: Precipitation,
        dust: float,
        meteorological_visibility_range: float,
        expected_atmospheric_transmission: float,
    ):
        """Test EO sensor get atmospheric transmission value.

        Args:
            distance (float): Distance to target (m)
            relative_humidity (float): The relative humidity between 0 and 1, exclusive
            precipitation (Precipitation): The type of precipitation
            dust (float): The atmospheric dust (g/m3) (>=0)
            meteorological_visibility_range (float): The meteorological visibility in km
            expected_atmospheric_transmission (float): The expected atmospheric transmission
        """
        atmospheric_transmission = self.eo_sensor._get_atmospheric_transmission(
            distance, relative_humidity, precipitation, dust, meteorological_visibility_range
        )

        assert math.isclose(
            atmospheric_transmission, expected_atmospheric_transmission, abs_tol=0.000001
        )

    @parameterized.expand(
        [
            [1000, 0.75, Precipitation.CLEAR, 0.0, 320.0, 0.963273],
            [5000, 0.75, Precipitation.CLEAR, 0.0, 320.0, 0.829366],  # Vary distance
            [1000, 0.2, Precipitation.CLEAR, 0.0, 320.0, 0.968101],  # Vary relative humidity
            [1000, 0.75, Precipitation.RAIN_MODERATE, 0.0, 320.0, 0.16563],  # Vary precipitation
            [1000, 0.75, Precipitation.CLEAR, 0.5, 320.0, 0.820847],  # Vary dust
            [1000, 0.75, Precipitation.CLEAR, 0.0, 5.0, 0.603626],  # Vary meteorological vis.
        ]
    )
    def test_acquire_1d_ii_sensor_get_atmospheric_transmission_factor(
        self,
        distance: float,
        relative_humidity: float,
        precipitation: Precipitation,
        dust: float,
        meteorological_visibility_range: float,
        expected_atmospheric_transmission: float,
    ):
        """Test II sensor get atmospheric transmission value.

        Args:
            distance (float): Distance to target (m)
            relative_humidity (float): The relative humidity between 0 and 1, exclusive
            precipitation (Precipitation): The type of precipitation
            dust (float): The atmospheric dust (g/m3) (>=0)
            meteorological_visibility_range (float): The meteorological visibility in km
            expected_atmospheric_transmission (float): The expected atmospheric transmission
        """
        atmospheric_transmission = self.ii_sensor._get_atmospheric_transmission(
            distance, relative_humidity, precipitation, dust, meteorological_visibility_range
        )

        assert math.isclose(
            atmospheric_transmission, expected_atmospheric_transmission, abs_tol=0.000001
        )

    @parameterized.expand(
        [
            [1000, 0.75, 320.0, 0.0, 0.811445],
            [5000, 0.75, 320.0, 0.0, 0.351800],  # Vary distance
            [1000, 0.2, 320.0, 0.0, 0.850249],  # Vary relative humidity
            [1000, 0.75, 5.0, 0.0, 0.703323],  # Vary meteorological vis.
            [1000, 0.75, 320.0, 15.0, 0.717814],  # Vary ambient temperature
        ]
    )
    def test_acquire_1d_ti_sensor_get_atmospheric_transmission_factor(
        self,
        distance: float,
        relative_humidity: float,
        meteorological_visibility_range: float,
        ambient_temperature: float,
        expected_atmospheric_transmission: float,
    ):
        """Test TI sensor get atmospheric transmission value.

        Args:
            distance (float): Distance to target (m)
            relative_humidity (float): The relative humidity between 0 and 1, exclusive
            meteorological_visibility_range (float): Meteorological visibility range in km
            ambient_temperature (float): Ambient temperature in degrees C
            expected_atmospheric_transmission (float): The expected atmospheric transmission
        """
        atmospheric_transmission = self.ti_sensor._get_atmospheric_transmission(
            distance, relative_humidity, meteorological_visibility_range, ambient_temperature
        )

        assert math.isclose(
            atmospheric_transmission, expected_atmospheric_transmission, abs_tol=0.000001
        )

    def test_contrast_acquire_1d_sensor_distance(self):
        """Test 1D Acquire contrast sensor at difference distances."""
        (
            p_detect_1km,
            p_recognise_1km,
            p_identify_1km,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        (
            p_detect_10km,
            p_recognise_10km,
            p_identify_10km,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            10000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_10km < p_detect_1km
        assert p_recognise_10km < p_recognise_1km
        assert p_identify_10km < p_identify_1km

    def test_temperature_acquire_1d_sensor_distance(self):
        """Test 1D Acquire temperature sensor at difference distances."""
        (
            p_detect_1km,
            p_recognise_1km,
            p_identify_1km,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        (
            p_detect_10km,
            p_recognise_10km,
            p_identify_10km,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            10000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_10km < p_detect_1km
        assert p_recognise_10km < p_recognise_1km
        assert p_identify_10km < p_identify_1km

    def test_contrast_acquire_1d_sensor_sense_time(self):
        """Test 1D Acquire contrast sensor with different sense times."""
        (
            p_detect_30secs,
            p_recognise_30secs,
            p_identify_30secs,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(seconds=30)
        )

        (
            p_detect_5mins,
            p_recognise_5mins,
            p_identify_5mins,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_30secs < p_detect_5mins
        assert p_recognise_30secs < p_recognise_5mins
        assert p_identify_30secs < p_identify_5mins

    def test_temperature_acquire_1d_sensor_sense_time(self):
        """Test 1D Acquire temperature sensor with different sense times."""
        (
            p_detect_30secs,
            p_recognise_30secs,
            p_identify_30secs,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(seconds=30)
        )

        (
            p_detect_5mins,
            p_recognise_5mins,
            p_identify_5mins,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_30secs < p_detect_5mins
        assert p_recognise_30secs < p_recognise_5mins
        assert p_identify_30secs < p_identify_5mins

    def test_contrast_acquire_1d_sensor_critical_dimension(self):
        """Test 1D Acquire contrast sensor with different critical dimensions."""
        (
            p_detect_2m,
            p_recognise_2m,
            p_identify_2m,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        self.target_agent.critical_dimension = 10.0
        (
            p_detect_10m,
            p_recognise_10m,
            p_identify_10m,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_2m < p_detect_10m
        assert p_recognise_2m < p_recognise_10m
        assert p_identify_2m < p_identify_10m

    def test_temperature_acquire_1d_sensor_critical_dimension(self):
        """Test 1D Acquire temperature sensor with different critical dimensions."""
        (
            p_detect_2m,
            p_recognise_2m,
            p_identify_2m,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        self.target_agent.critical_dimension = 10.0
        (
            p_detect_10m,
            p_recognise_10m,
            p_identify_10m,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_2m < p_detect_10m
        assert p_recognise_2m < p_recognise_10m
        assert p_identify_2m < p_identify_10m

    def test_contrast_acquire_1d_sensor_magnification(self):
        """Test 1D Acquire contrast sensor with different magnifications."""
        (
            p_detect_x1,
            p_recognise_x1,
            p_identify_x1,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        eo_sensor_clone = self.eo_sensor.get_new_instance()
        eo_sensor_clone._magnification = 10.0
        (
            p_detect_x10,
            p_recognise_x10,
            p_identify_x10,
        ) = eo_sensor_clone._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_x1 < p_detect_x10
        assert p_recognise_x1 < p_recognise_x10
        assert p_identify_x1 < p_identify_x10

    def test_temperature_acquire_1d_sensor_magnification(self):
        """Test 1D Acquire temperature sensor with different magnifications."""
        (
            p_detect_x1,
            p_recognise_x1,
            p_identify_x1,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        ti_sensor_clone = self.ti_sensor.get_new_instance()
        ti_sensor_clone._magnification = 10.0
        (
            p_detect_x10,
            p_recognise_x10,
            p_identify_x10,
        ) = ti_sensor_clone._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_x1 < p_detect_x10
        assert p_recognise_x1 < p_recognise_x10
        assert p_identify_x1 < p_identify_x10

    def test_contrast_acquire_1d_sensor_target_reflectivity(self):
        """Test 1D Acquire contrast sensor with different target reflectivity."""
        (
            p_detect_high_contrast,
            p_recognise_high_contrast,
            p_identify_high_contrast,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        self.target_agent._reflectivity = default_culture().reflectivity
        (
            p_detect_no_contrast,
            p_recognise_no_contrast,
            p_identify_no_contrast,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_high_contrast > p_detect_no_contrast
        assert p_recognise_high_contrast > p_recognise_no_contrast
        assert p_identify_high_contrast > p_identify_no_contrast

    def test_temperature_acquire_1d_target_temperature(self):
        """Test 1D Acquire temperature sensor with different target temperatures."""
        (
            p_detect_high_delta_t,
            p_recognise_high_delta_t,
            p_identify_high_delta_t,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        self.target_agent._temperature = self.target_agent.model.space._ambient_temperature
        (
            p_detect_no_delta_t,
            p_recognise_no_delta_t,
            p_identify_no_delta_t,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_high_delta_t > p_detect_no_delta_t
        assert p_recognise_high_delta_t > p_recognise_no_delta_t
        assert p_identify_high_delta_t > p_identify_no_delta_t

    def test_contrast_acquire_1d_sensor_attenuated_distance(self):
        """Test 1D Acquire contrast sensor with different attenuated distances."""
        (
            p_detect_no_attenuation,
            p_recognise_no_attenuation,
            p_identify_no_attenuation,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        (
            p_detect_attenuated,
            p_recognise_attenuated,
            p_identify_attenuated,
        ) = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 2000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_no_attenuation > p_detect_attenuated
        assert p_recognise_no_attenuation > p_recognise_attenuated
        assert p_identify_no_attenuation > p_identify_attenuated

    def test_temperature_acquire_1d_sensor_temperature_std_dev(self):
        """Test 1D Acquire temperature sensor with different temperature std devs."""
        (
            p_detect_no_temp_std_dev,
            p_recognise_no_temp_std_dev,
            p_identify_no_temp_std_dev,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        self.target_agent.temperature_std_dev = 2.0

        (
            p_detect_with_std_dev,
            p_recognise_with_std_dev,
            p_identify_with_std_dev,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_no_temp_std_dev < p_detect_with_std_dev
        assert p_recognise_no_temp_std_dev < p_recognise_with_std_dev
        assert p_identify_no_temp_std_dev < p_identify_with_std_dev

    def test_temperature_acquire_1d_sensor_attenuated_distance(self):
        """Test 1D Acquire temperature sensor with different attenuated distances."""
        (
            p_detect_no_attenuation,
            p_recognise_no_attenuation,
            p_identify_no_attenuation,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        (
            p_detect_attenuated,
            p_recognise_attenuated,
            p_identify_attenuated,
        ) = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 2000.0, self.target_agent, timedelta(minutes=5)
        )

        assert p_detect_no_attenuation > p_detect_attenuated
        assert p_recognise_no_attenuation > p_recognise_attenuated
        assert p_identify_no_attenuation > p_identify_attenuated

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
    def test_sensor_height_acquire_1d_eo(
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
        self.eo_sensor_height = ContrastAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.contrast_performance_curve,
            johnson_criteria=self.johnson_criteria,
            sensor_type=EOContrastSensorType(),
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
            height_of_sensor=height,
        )
        sensor_pos = self.eo_sensor_height.get_elevated_sensor_position(agent_position)
        assert len(sensor_pos) == len(agent_position)
        assert math.isclose(sensor_pos[0], sensor_position[0], abs_tol=0.0001)
        assert math.isclose(sensor_pos[1], sensor_position[1], abs_tol=0.0001)
        if len(agent_position) == 3:
            assert math.isclose(sensor_pos[2], sensor_position[2], abs_tol=0.0001)  # type: ignore

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
    def test_sensor_height_acquire_1d_ii(
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
        self.ii_sensor_height = ContrastAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.contrast_performance_curve,
            johnson_criteria=self.johnson_criteria,
            sensor_type=IIContrastSensorType(),
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
            height_of_sensor=height,
        )
        sensor_pos = self.ii_sensor_height.get_elevated_sensor_position(agent_position)
        assert len(sensor_pos) == len(agent_position)
        assert math.isclose(sensor_pos[0], sensor_position[0], abs_tol=0.0001)
        assert math.isclose(sensor_pos[1], sensor_position[1], abs_tol=0.0001)
        if len(agent_position) == 3:
            assert math.isclose(sensor_pos[2], sensor_position[2], abs_tol=0.0001)  # type: ignore

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
    def test_sensor_height_acquire_1d_ti(
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
        self.ti_sensor_height = TemperatureAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.temperature_performance_curve,
            johnson_criteria=self.johnson_criteria,
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
            height_of_sensor=height,
        )
        sensor_pos = self.ti_sensor_height.get_elevated_sensor_position(agent_position)
        assert len(sensor_pos) == len(agent_position)
        assert math.isclose(sensor_pos[0], sensor_position[0], abs_tol=0.0001)
        assert math.isclose(sensor_pos[1], sensor_position[1], abs_tol=0.0001)
        if len(agent_position) == 3:
            assert math.isclose(sensor_pos[2], sensor_position[2], abs_tol=0.0001)  # type: ignore

    @parameterized.expand(
        [
            [0.01, Confidence.IDENTIFY],
            [0.05, Confidence.RECOGNISE],
            [0.5, Confidence.DETECT],
            [1.0, None],
        ]
    )
    def test_contrast_acquire_1d_sensor_run_detection(
        self, probability: float, expected_confidence: Optional[Confidence]
    ):
        """Test 1d Acquire sensor run detection method.

        Fix random number outcome and test that the expected confidence level is chosen.
        Expected probabilities are:

        Detect: 0.965571
        Recognise: 0.065913
        Identify: 0.0011044
        Detect: 0.999641
        Recognise: 0.304075
        Identify: 0.055961

        Args:
            probability (float): Fixed probability value to inject into faked random number gen
            expected_confidence (Optional[Confidence]): The confidence level that will be detected
                at for given confidence
        """
        model = MockModel3d()
        # random number generator always returns fixed probability
        model.random.random = Mock(return_value=probability)

        sensor_agent = RetAgent(
            model=model,
            pos=(0.0, 0.0, 0.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        RetAgent(
            model=model,
            pos=(0.0, 1000.0, 0.0),  # 1 km away
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.eo_sensor.run_detection(sensor_agent, sense_direction=90)

        perceived_agents = self.eo_sensor.get_results(sensor_agent)

        if expected_confidence is not None:
            assert len(perceived_agents) == 1
            assert perceived_agents[0].confidence == expected_confidence
        else:
            assert len(perceived_agents) == 0

    @parameterized.expand(
        [
            [0.0],
            [-1.0],
        ]
    )
    def test_acquire_1d_sensor_magnification_error(self, magnification):
        """Test Acquire 1D sensor exception handling for non-positive magnification values.

        Args:
            magnification (float): The non-positive magnification value to test
        """
        with self.assertRaises(ValueError) as e_magnification:
            ContrastAcquire1dSensor(
                magnification=magnification,
                performance_curve=self.contrast_performance_curve,
                johnson_criteria=self.johnson_criteria,
                sensor_type=EOContrastSensorType(),
                wavelength=SensorWavelength(500.0),
                sampling_distance=SensorSamplingDistance(10.0),
                is_active_sensor=True,
            )
        assert f"Acquire 1D Sensor magnification value ({magnification}) must be postive." == str(
            e_magnification.exception
        )

    def test_acquire_1d_sensor_sampling_distance_none(self):
        """Test Acquire 1D sensor when sampling distance is none."""
        sensor = ContrastAcquire1dSensor(
            magnification=1.0,
            performance_curve=self.contrast_performance_curve,
            johnson_criteria=self.johnson_criteria,
            sensor_type=EOContrastSensorType(),
            wavelength=SensorWavelength(500.0),
            sampling_distance=None,
            is_active_sensor=True,
        )
        assert isinstance(sensor._sampling_distance, SensorSamplingDistance)

    @parameterized.expand(
        [
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    def test_contrast_acquire_1d_sensor_water_vapour_transmission_coefficient_error(
        self, relative_humidity
    ):
        """Test Acquire 1D sensor exception handling for invalid relative humidity values.

        Args:
            relative_humidity (float): The invalid relative humidity value to test
        """
        with self.assertRaises(ValueError) as e:
            self.eo_sensor._get_water_vapour_transmission_coefficient(relative_humidity)
        assert (
            f"Relative humidity ({relative_humidity}) must be between 0 and 1, exclusive."
            == str(e.exception)
        )

    @parameterized.expand(
        [
            [0.1, 0.02],
            [0.16, 0.025],
            [0.63, 0.025],
            [0.8, 0.03],
        ]
    )
    def test_contrast_acquire_1d_ii_sensor_water_vapour_transmission_coefficient(
        self, relative_humidity, expected_result
    ):
        """Test Acquire 1D II sensor relative humidity values.

        Args:
            relative_humidity (float): The relative humidity value to test.
            expected_result (float): The expected return value.
        """
        assert (
            self.ii_sensor._get_water_vapour_transmission_coefficient(relative_humidity)
            == expected_result
        )

    def test_contrast_acquire_1d_sensor_atmospheric_transmission_dust_error(self):
        """Test Acquire 1D sensor exception handling for negative dust values."""
        dust = -1.0
        with self.assertRaises(ValueError) as e:
            self.eo_sensor._get_atmospheric_transmission(1000, 0.7, Precipitation.CLEAR, dust, 16.0)
        assert f"Atmospheric dust ({dust} g/m3) cannot be negative." == str(e.exception)

    def test_get_observation_confidence_probabilities_non_default_culture(
        self,
    ):
        """Test observation confidence probabilities with non-default Culture."""
        # always return test culture
        self.model.space.get_culture = Mock(
            return_value=Culture("test culture", height=0.0, reflectivity=0.8, temperature=25.0)
        )

        eo_probabilities = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        ii_probabilities = self.ii_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        ti_probabilities = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        self.assert_probabilities_close(eo_probabilities, (0.999856, 0.366828, 0.070711))
        self.assert_probabilities_close(ii_probabilities, (0.999847, 0.362986, 0.069770))
        self.assert_probabilities_close(ti_probabilities, (0.385025, 0.006466, 0.000253))

    def test_contrast_run_detection_with_culture_attenuation(
        self,
    ):
        """Test contrast sensor run detection with Culture attenuation."""
        attenuated_model = SensorTestModel()  # all culture is attenuated by 1.1

        attenuated_model_sensor_agent = RetAgent(
            model=attenuated_model,
            pos=(300.0, 1.0, 1.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        RetAgent(
            model=attenuated_model,
            pos=(300.0, 201.0, 1.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        # random number generator always returns fixed probability
        attenuated_model.random.random = Mock(return_value=0.4)

        self.eo_sensor.run_detection(attenuated_model_sensor_agent, sense_direction=90)

        perceived_agents = self.eo_sensor.get_results(attenuated_model_sensor_agent)

        assert len(perceived_agents) == 1
        assert perceived_agents[0].confidence == Confidence.RECOGNISE

        # Artificially return no culture penetration data to simulate no culture attenuation
        attenuated_model.space.check_culture_penetration = Mock(return_value={})

        # Run again with no attenuation
        self.eo_sensor.run_detection(attenuated_model_sensor_agent, sense_direction=90)

        perceived_agents = self.eo_sensor.get_results(attenuated_model_sensor_agent)

        assert len(perceived_agents) == 1
        assert perceived_agents[0].confidence == Confidence.IDENTIFY

    def test_temperature_run_detection_with_culture_attenuation(
        self,
    ):
        """Test temperature sensor run detection with Culture attenuation."""
        attenuated_model = SensorTestModel()  # all culture is attenuated by 1.1

        attenuated_model_sensor_agent = RetAgent(
            model=attenuated_model,
            pos=(300.0, 1.0, 1.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        RetAgent(
            model=attenuated_model,
            pos=(300.0, 201.0, 1.0),
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        # random number generator always returns fixed probability
        attenuated_model.random.random = Mock(return_value=0.91)

        self.ti_sensor.run_detection(attenuated_model_sensor_agent, sense_direction=90)

        perceived_agents = self.ti_sensor.get_results(attenuated_model_sensor_agent)

        assert len(perceived_agents) == 1
        assert perceived_agents[0].confidence == Confidence.RECOGNISE

        # Artificially return no culture penetration data to simulate no culture attenuation
        attenuated_model.space.check_culture_penetration = Mock(return_value={})

        # Run again simulating no attenuation
        self.ti_sensor.run_detection(attenuated_model_sensor_agent, sense_direction=90)

        perceived_agents = self.ti_sensor.get_results(attenuated_model_sensor_agent)

        assert len(perceived_agents) == 1
        assert perceived_agents[0].confidence == Confidence.IDENTIFY

    def test_get_observation_confidence_probabilities_non_default_space(
        self,
    ):
        """Test observation confidence probabilities with non-default environmental factors."""
        space = ContinuousSpaceWithTerrainAndCulture3d(
            self.model.space.width,
            self.model.space.height,
            relative_humidity=0.8,
            meteorological_visibility_range=11.0,
            ambient_temperature=8.0,
            dust=0.1,
            precipitation=Precipitation.DRIZZLE_LIGHT,
        )
        self.model.space = space

        eo_probabilities = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        ii_probabilities = self.ii_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        ti_probabilities = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, self.target_agent, timedelta(minutes=5)
        )

        self.assert_probabilities_close(eo_probabilities, (0.481379, 0.004212, 0.000152))
        self.assert_probabilities_close(ii_probabilities, (0.472989, 0.004032, 0.000145))
        self.assert_probabilities_close(ti_probabilities, (0.966857, 0.087233, 0.010677))

    def test_contrast_acquire_1d_sensor_get_new_instance(self):
        """Test the get new instance method for ContrastAcquire1dSensor."""
        sensor = ContrastAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.contrast_performance_curve,
            johnson_criteria=self.johnson_criteria,
            sensor_type=EOContrastSensorType(),
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
            detection_timings=SensorDetectionTimings(timedelta(seconds=30)),
            is_active_sensor=True,
        )
        clone = sensor.get_new_instance()

        assert sensor is not clone
        assert isinstance(clone, ContrastAcquire1dSensor)
        assert sensor._magnification == clone._magnification
        assert sensor._johnson_criteria == clone._johnson_criteria
        assert sensor._performance_curve == clone._performance_curve
        assert sensor.sensor_type == clone.sensor_type
        assert sensor._wavelength.wavelength == clone._wavelength.wavelength
        assert (
            sensor._sampling_distance.sampling_distance
            == clone._sampling_distance.sampling_distance
        )
        assert (
            sensor._detection_timings.get_detection_delay()
            == clone._detection_timings.get_detection_delay()
        )
        assert sensor.is_active_sensor == clone.is_active_sensor

    def test_temperature_acquire_1d_sensor_get_new_instance(self):
        """Test the get new instance method for TemperatureAcquire1dSensor."""
        sensor = TemperatureAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.temperature_performance_curve,
            johnson_criteria=self.johnson_criteria,
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
            detection_timings=SensorDetectionTimings(timedelta(seconds=30)),
            is_active_sensor=True,
        )
        clone = sensor.get_new_instance()

        assert sensor is not clone
        assert isinstance(clone, TemperatureAcquire1dSensor)
        assert sensor._magnification == clone._magnification
        assert sensor._johnson_criteria == clone._johnson_criteria
        assert sensor._performance_curve == clone._performance_curve
        assert sensor._wavelength.wavelength == clone._wavelength.wavelength
        assert (
            sensor._sampling_distance.sampling_distance
            == clone._sampling_distance.sampling_distance
        )
        assert (
            sensor._detection_timings.get_detection_delay()
            == clone._detection_timings.get_detection_delay()
        )
        assert sensor.is_active_sensor == clone.is_active_sensor

    def test_contrast_get_observation_confidence_probabilities_air_agent(self):
        """Test contrast sensor observation confidence probabilities for target Air agent."""
        model = MockModel3d()

        target_agent = RetAgent(
            model=model,
            pos=(0.0, 1000.0, 10.0),  # 1 km away
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            critical_dimension=2.0,
            reflectivity=0.2,
            temperature=20.0,
        )
        probabilities = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, target_agent, timedelta(minutes=5)
        )

        self.assert_probabilities_close(probabilities, (0.997492, 0.182044, 0.029729))

        model.space.sky_background_reflectivity = 0.5

        probabilities = self.eo_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, target_agent, timedelta(minutes=5)
        )

        self.assert_probabilities_close(probabilities, (0.995956, 0.156159, 0.024372))

    def test_temperature_get_observation_confidence_probabilities_air_agent(self):
        """Test temperature sensor observation confidence probabilities for target Air agent."""
        model = MockModel3d()
        model.space._ambient_temperature = 10

        target_generic_agent = RetAgent(
            model=model,
            pos=(0.0, 1000.0, 10.0),  # 1 km away
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.2,
            temperature=20.0,
        )
        target_air_agent = RetAgent(
            model=model,
            pos=(0.0, 1000.0, 10.0),  # 1 km away
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.AIR,
            critical_dimension=2.0,
            reflectivity=0.2,
            temperature=20.0,
        )
        generic_agent_probabilities = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, target_generic_agent, timedelta(minutes=5)
        )
        air_agent_probabilities = self.ti_sensor._get_observation_confidence_probabilities(
            1000.0, 1000.0, target_air_agent, timedelta(minutes=5)
        )

        self.assert_probabilities_close(generic_agent_probabilities, (0.909074, 0.053145, 0.004898))
        self.assert_probabilities_close(air_agent_probabilities, (0.999223, 0.291519, 0.053145))

    def test_run_detection_group_agent(self):
        """Test that group agents are ignored by Acquire sensor run detection."""
        model = MockModel3d()
        # random number generator always returns fixed probability of 0.0 so agents
        # are always detected if possible
        model.random.random = Mock(return_value=0.0)

        sensor_agent = RetAgent(
            model=model,
            pos=(0.0, 0.0, 0.0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        RetAgent(
            model=model,
            pos=(0.0, 1000.0, 0.0),  # 1 km away
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GROUP,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        RetAgent(
            model=model,
            pos=(0.0, 1000.0, 0.0),  # 1 km away
            name="Target Agent",
            affiliation=Affiliation.FRIENDLY,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.eo_sensor.run_detection(sensor_agent, sense_direction=90)

        perceived_agents = self.eo_sensor.get_results(sensor_agent)

        # only the generic agent is detected
        assert len(perceived_agents) == 1
        assert perceived_agents[0].agent_type == AgentType.GENERIC

    @parameterized.expand(
        [
            [Precipitation.CLEAR, 1.0, 3.912],
            [Precipitation.CLEAR, 2.0, 3.912 / 2.0],
            [Precipitation.DRIZZLE_LIGHT, 1.0, 0.789],
            [Precipitation.DRIZZLE_HEAVY, 1.0, 1.403],
            [Precipitation.RAIN_MODERATE, 1.0, 1.768],
            [Precipitation.RAIN_HEAVY, 1.0, 2.735],
            [Precipitation.THUNDERSTORM, 1.0, 2.429],
            [Precipitation.SNOW, 1.0, 3.912],
            [Precipitation.SNOW, 2.0, 3.912 / 2.0],
        ]
    )
    def test_contrast_acquire_1d_sensor_aerosol_transmission_coefficient(
        self, precipitation: Precipitation, visibility: float, expected_result: float
    ):
        """Test the get aerosol transmission coefficient method for ContrastAcquire1dSensor."""
        sensor = ContrastAcquire1dSensor(
            magnification=self.magnification,
            performance_curve=self.contrast_performance_curve,
            johnson_criteria=self.johnson_criteria,
            sensor_type=EOContrastSensorType(),
            wavelength=SensorWavelength(500.0),
            sampling_distance=SensorSamplingDistance(10.0),
            is_active_sensor=True,
        )

        m_precipitation = sensor._get_aerosol_transmission_coefficient(
            precipitation=precipitation, meteorological_visibility_range=visibility
        )

        assert m_precipitation == expected_result

    def test_no_line_of_sight(self):
        """Test that the 1d acquire sensor works without line fo sight."""
        culture_red = Culture("red culture", 100.0)
        culture_green = Culture("green culture", 100.0)
        culture_blue = Culture("blue culture")
        culture_yellow = Culture("yellow culture", 25.0)
        position_red_z1 = (75.0, 25.0, 1.0)
        position_green_z1 = (25.0, 25.0, 1.0)

        model = MockModel3d()
        model.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            x_min=0,
            y_min=0,
            terrain_image_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "test_space/TestLineOfSight.png")
            ),
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

        # random number generator always returns fixed probability of 0.0 so agents
        # are always detected if possible
        model.random.random = Mock(return_value=0.0)

        sensor_agent = RetAgent(
            model=model,
            pos=position_red_z1,
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        detect_agent = RetAgent(
            model=model,
            pos=position_green_z1,
            name="Target Agent",
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.eo_sensor._sampling_distance = SensorSamplingDistance(1)

        perceived_agents = self.eo_sensor._run_detection(
            sensor_agent=sensor_agent, all_target_agents=[detect_agent]
        )

        assert len(perceived_agents) == 0

    def test_sensor_value_error_catching(self):
        """Test that a ValueError exception is still thrown despite catching LineOfSight Error."""

        class TesterContinuousSpaceWithTerrainAndCulture3d(ContinuousSpaceWithTerrainAndCulture3d):
            """Tester class to override the get_culture_attenuated_distance_between method."""

            def get_culture_attenuated_distance_between(
                self,
                pos_a: Any,
                pos_b: Any,
                wavelength: float,
                sampling_distance: Optional[float] = None,
                distance: Optional[float] = None,
            ):
                """Test method to raise ValueError.

                Args:
                    pos_a (Any): Placeholder argument
                    pos_b (Any): Placeholder argument
                    wavelength (float): Placeholder argument
                    sampling_distance (Optional[float], optional): Placeholder argument.
                    distance (Optional[float], optional): Placeholder argument.

                Raises:
                    ValueError: Test value error handling
                """
                raise ValueError("Test Value error")

        model = MockModel3d()
        model.space = TesterContinuousSpaceWithTerrainAndCulture3d(100, 100)

        sensor_agent = RetAgent(
            model=model,
            pos=(0, 0, 0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        detect_agent = RetAgent(
            model=model,
            pos=(0, 0, 0),
            name="Target Agent",
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        with self.assertRaises(ValueError) as e:
            self.eo_sensor._run_detection(
                sensor_agent=sensor_agent, all_target_agents=[detect_agent]
            )
        self.assertEqual("Test Value error", str(e.exception))

    def test_sensor_error_catching(self):
        """Test that an error exception is thrown if not a type of ValueError."""

        class TesterContinuousSpaceWithTerrainAndCulture3d(ContinuousSpaceWithTerrainAndCulture3d):
            """Tester class to override the get_culture_attenuated_distance_between method."""

            def get_culture_attenuated_distance_between(
                self,
                pos_a: Any,
                pos_b: Any,
                wavelength: float,
                sampling_distance: Optional[float] = None,
                distance: Optional[float] = None,
            ):
                """Test method to raise ValueError.

                Args:
                    pos_a (Any): Placeholder argument
                    pos_b (Any): Placeholder argument
                    wavelength (float): Placeholder argument
                    sampling_distance (Optional[float], optional): Placeholder argument.
                    distance (Optional[float], optional): Placeholder argument.

                Raises:
                    ValueError: Test value error handling
                """
                raise KeyError("Test key error")

        model = MockModel3d()
        model.space = TesterContinuousSpaceWithTerrainAndCulture3d(100, 100)

        sensor_agent = RetAgent(
            model=model,
            pos=(0, 0, 0),
            name="Sensor Agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        detect_agent = RetAgent(
            model=model,
            pos=(0, 0, 0),
            name="Target Agent",
            affiliation=Affiliation.HOSTILE,
            agent_type=AgentType.GENERIC,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        with self.assertRaises(KeyError) as e:
            self.eo_sensor._run_detection(
                sensor_agent=sensor_agent, all_target_agents=[detect_agent]
            )
        self.assertEqual("'Test key error'", str(e.exception))
