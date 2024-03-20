"""Sensor representation."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.stats import rv_discrete

import ret.utilities.geometric_utilities as gu
from ret.agents.agenttype import AgentType
from ret.agents.agent import AgentCasualtyState
from ret.sensing.distribution import SingleValueDistribution
from ret.sensing.perceivedworld import Confidence, PerceivedAgentBasedOnAgent
from ret.space.culturemap import LineOfSightException
from ret.space.space import Precipitation
from ret.template import Template
from ret.types import Coordinate3d  # noqa: TC001

if TYPE_CHECKING:
    from typing import Callable, Optional, Union

    from mesa.space import GridContent
    from ret.agents.agent import RetAgent
    from ret.agents.projectileagent import ProjectileAgent
    from ret.model import RetModel
    from ret.sensing.distribution import Distribution
    from ret.sensing.perceivedworld import PerceivedAgent
    from ret.space.space import ContinuousSpaceWithTerrainAndCulture3d
    from ret.types import Coordinate2dOr3d

PerformanceCurve = list[tuple[float, float]]
JohnsonCriteria = dict[Confidence, float]
ArcOfRegard = dict[tuple[float, float], float]


class SensorSideEffect(ABC):
    """An abstract class representing a side effect of a sensor.

    To create a new sensor side effect using custom logic, create a new SensorSideEffect class
    that extends this class.
    """

    @abstractmethod
    def is_applicable(
        self, senser: RetAgent, target: RetAgent, found: bool
    ) -> bool:  # pragma: no cover
        """Method to determine whether the target is affected by the side effect.

        Args:
            senser (RetAgent): The agent doing the sensing.
            target (RetAgent): The agent being sensed.
            found (bool): Whether the target agent was successfully found by the sensor.

        Returns:
            bool: Whether the side effect is applicable to the target agent.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_side_effect(
        self,
        senser: RetAgent,
        target_agent: RetAgent,
        perceived_target_agent: Optional[PerceivedAgent],
    ) -> None:  # pragma: no cover
        """Apply the side effect to the target agent.

        Args:
            senser (RetAgent): The agent doing the sensing.
            target_agent (RetAgent): The agent being sensed.
            perceived_target_agent (Optional[PerceivedAgent]): An (optional) perceived version
                of the target agent as perceived by the sensing agent.
        """
        raise NotImplementedError()


class NotifySensedBySideEffect(SensorSideEffect):
    """Add sensed status to any target agent sensed by the senser agent."""

    def is_applicable(self, senser: RetAgent, target: RetAgent, found: bool) -> bool:
        """Determine whether side effect is applicable to the target.

        Args:
            senser (RetAgent): The agent doing the sensing.
            target (RetAgent): The agent being sensed.
            found (bool): Whether the target agent was successfully found by the sensor.

        Returns:
            bool: True if the target agent was found by the sensor.
        """
        return found

    def apply_side_effect(
        self,
        senser: RetAgent,
        target_agent: RetAgent,
        perceived_target_agent: Optional[PerceivedAgent],
    ) -> None:
        """Add a sensed status to the target agent's statuses.

        Args:
            senser (RetAgent): The agent doing the sensing.
            target_agent (RetAgent): The agent being sensed.
            perceived_target_agent (Optional[PerceivedAgent]): An (optional) perceived version
                of the target agent as perceived by the sensing agent.
        """
        if perceived_target_agent is not None:
            target_agent.add_sensed_by_status(
                sense_time=perceived_target_agent._sense_time, senser=senser
            )


class SensorFilter(ABC):
    """An abstract class representing a filter for sensors.

    To create a new sensor filter using custom logic, create a new SensorFilter class that extends
    this class and then override the following methods:
    include_agent: This must be overridden and should return a bool, determining whether an agent is
        included or not.
    """

    @abstractmethod
    def include_agent(self, agent: RetAgent) -> bool:  # pragma: no cover
        """Whether or not to include an agent within the filter.

        Override in subclasses.

        Args:
            agent (RetAgent): Agent to be included or excluded from the filter.
        """
        pass


class ActiveFilter(SensorFilter):
    """Filter allowing for all agents, unless they are hiding."""

    def include_agent(self, agent: RetAgent) -> bool:
        """Whether or not to include an agent within the filter.

        Args:
            agent (RetAgent): Agent to be included or excluded from the filter.

        Returns:
            bool: True if the agent should be included.
        """
        return not agent.hiding


class PassiveFilter(SensorFilter):
    """Filter allowing for all agents."""

    def include_agent(self, agent: RetAgent) -> bool:
        """Whether or not to include an agent within the filter.

        Args:
            agent (RetAgent): Agent to be included or excluded from the filter.

        Returns:
            bool: True if the agent should be included.
        """
        return True


class Sensor(ABC, Template):
    """An abstract class representing a sensor that can perceive agents.

    This class is abstract and should be overridden by classes with specific behaviour
    as and when required. They contain both logic to perform a detection, and to
    provide results to an agent doing the detection.

    To create a new sensor using custom logic, create a new Sensor class that extends
    this class and then override the following methods:

        run_detection: This task must be overridden and should perform logic to
            complete a single time-step's worth of detection using the sensor. The
            method should add any new detections to the `_cached_agents` list.
        get_new_instance: This must be overridden and should return a new instance of
            your sensor that is functionally identical, but does not contain any
            cached detections, i.e., the `_cached_agents` property should be an empty
            list. Note that this methods is required to ensure that each agent that uses
            a sensor can create a version of it that is independent of other agent's
            sensors, and not receive information from another agent using the same type
            of sensor.

    For examples of sensors and how to override these methods see the sensors already
    implemented.
    """

    def __init__(
        self,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ) -> None:
        """Initialise a sensor.

        Args:
            detection_timings (Optional[SensorDetectionTimings], optional): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        if detection_timings is None:
            detection_timings = SensorDetectionTimings()

        if sensor_filters is None:
            sensor_filters = []

        if sensor_side_effects is None:
            sensor_side_effects = []

        self._detection_timings = detection_timings
        self.view_angle = view_angle
        self._cached_agents: list[PerceivedAgent] = []
        self.is_active_sensor = is_active_sensor
        self.sensor_filters = sensor_filters
        self.sensor_side_effects = sensor_side_effects
        self.height_of_sensor = height_of_sensor

    def run_detection(
        self,
        sensor_agent: RetAgent,
        sense_direction: Optional[Union[float, Coordinate2dOr3d, str]] = None,
    ) -> None:
        """Run a sensor detection, adding perceived agents to self._cached_agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
            sense_direction (Optional[float | Coordinate2dor3d | str]): Optional direction to sense
                in, can be input as float: degrees clockwise from y-axis, Coordinate2dor3d: a
                position used to calculate heading from sensor agent, str: the name of an Area in
                the space, a random point within the area will be chosen to look towards.
                (default = None)

        Raises:
            KeyError: Sense direction is not defined
        """
        if self.is_active_sensor & sensor_agent.hiding:
            warnings.warn("Cannot actively sense when a sensor agent is hiding.", stacklevel=2)
            return

        self.try_add_actively_sensing_agent(sensor_agent)

        if isinstance(sense_direction, tuple):
            heading = sensor_agent.model.space.get_heading(
                sensor_agent.pos,  # type: ignore
                sense_direction,  # type: ignore
            )
            sense_direction_angle = gu._heading_to_clockwise_angle_degrees(heading)  # type: ignore
        elif isinstance(sense_direction, str):
            if sense_direction in sensor_agent.model.space.areas:
                area = sensor_agent.model.space.areas[sense_direction]
                random_location = area.get_coord_inside(sensor_agent.model.random)
                heading = sensor_agent.model.space.get_heading(
                    sensor_agent.pos,  # type: ignore
                    random_location,  # type: ignore
                )
                sense_direction_angle = gu._heading_to_clockwise_angle_degrees(
                    heading  # type: ignore
                )
            else:
                raise (
                    KeyError("Sense direction area (" + f"""{sense_direction}""" + ") not defined.")
                )
        elif isinstance(sense_direction, (int, float)):
            sense_direction_angle = sense_direction
        else:
            if self.view_angle is not None:
                sense_direction_angle = 90.0  # default look directly down x-axis
                warnings.warn(
                    f"No sense direction provided for a sensor with view angle of {self.view_angle}"
                    ", assuming positive x direction.",
                    stacklevel=2,
                )
            else:
                sense_direction_angle = 90.0  # default look directly down x-axis

        all_target_agents: list[RetAgent] = sensor_agent.model.get_all_agents()
        all_target_agents.remove(sensor_agent)

        for f in self.sensor_filters:
            all_target_agents = [a for a in all_target_agents if f.include_agent(a)]

        target_agents = self._filter_target_agents_by_view_angle(
            sensor_agent, sense_direction_angle, all_target_agents
        )

        perceived_agents = self._run_detection(sensor_agent, target_agents)
        self._cached_agents.extend(perceived_agents)

        self._apply_side_effects(sensor_agent, all_target_agents, perceived_agents)

    @abstractmethod
    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:  # pragma: no cover
        """Run a sensor detection, returning a list of perceived agents.

        Override in subclasses.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
            all_target_agents (list[RetAgent]): A list of potentially perceivable agents

        Returns
            list[PerceivedAgent]: A list of agents perceived by the sensor.
        """
        pass

    def get_elevated_sensor_position(self, position: Coordinate2dOr3d) -> Coordinate2dOr3d:
        """Transform a coordinate to raise it by the sensor height.

        Args:
            position (Coordinate2dOr3d): The original position of the agent to be raised.

        Returns:
            Coordinate2dOr3d: The position of the agent's sensor.
        """
        if len(position) > 2:
            return (position[0], position[1], position[2] + self.height_of_sensor)  # type: ignore
        else:
            return position

    def _apply_side_effects(
        self,
        sensor_agent: RetAgent,
        all_target_agents: list[RetAgent],
        perceived_agents: list[PerceivedAgent],
    ):
        """Apply sensor side effects.

        Args:
            sensor_agent (RetAgent): The sensing agent.
            all_target_agents (list[RetAgent]): A list of all target agents.
            perceived_agents (list[PerceivedAgent]): A list of agents perceived by the sensor.
        """
        perceived_agent_ids = [a._unique_id for a in perceived_agents]
        for side_effect in self.sensor_side_effects:
            appliable_agents = []
            for agent in all_target_agents:
                found = agent.unique_id in perceived_agent_ids
                if side_effect.is_applicable(sensor_agent, agent, found):
                    appliable_agents.append(agent)

            for agent in appliable_agents:
                perceived_target_agent = next(
                    (agent for agent in perceived_agents if agent.unique_id == agent.unique_id),
                    None,
                )
                side_effect.apply_side_effect(sensor_agent, agent, perceived_target_agent)

    def get_results(self, sensor_agent: RetAgent) -> list[PerceivedAgent]:
        """Get valid perceived agents detected using this sensor.

        Valid agents are those with a sense time equal to or earlier than the current
        model time. Once these agents are filtered, only agents with sense times in the
        future are retained in the perceived agent cache of the sensor.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection

        Returns:
            list[PerceivedAgent]: list of perceived agents with a sense time equal to or
                earlier than the current model time
        """
        current_time = sensor_agent.model.get_time()

        valid_agents = [a for a in self._cached_agents if a._sense_time <= current_time]

        sensor_agent.model.logger.log_observation_record(sensor_agent, valid_agents)

        self._cached_agents = [a for a in self._cached_agents if a._sense_time > current_time]

        return valid_agents

    def _filter_target_agents_by_view_angle(
        self,
        sensor_agent: RetAgent,
        sense_direction: float,
        all_target_agents: list[RetAgent],
    ) -> list[RetAgent]:
        """Get target agents within the the sensor view angle using the Arc of Regard.

        This does not restrict by which agents are in line of sight, only the viewing angle.

        Args:
            sensor_agent (RetAgent): The sensor agent
            sense_direction (float): The sense direction (degrees)
            all_target_agents (list[RetAgent]): All possible target agents

        Returns:
            list[RetAgent]: The filtered target agents in the view angle
        """
        if sensor_agent.arc_of_regard is None or self.view_angle is None or sense_direction is None:
            return all_target_agents

        sector = self._choose_arc_of_regard_sector(sensor_agent.arc_of_regard, sensor_agent.model)
        offset_angle = self._get_sector_midpoint_angle(sector)
        offset_sense_direction: float = sense_direction + offset_angle
        return self._reduce_to_field_of_view(
            sensor_agent, all_target_agents, offset_sense_direction
        )

    def _choose_arc_of_regard_sector(
        self, arc_of_regard: ArcOfRegard, model: RetModel
    ) -> tuple[float, float]:
        """Choose a sector from the arc of regard based on their probabilities.

        Args:
            arc_of_regard (ArcOfRegard): The arc of regard
            model (RetModel): The model, to get the random seed for the distribution

        Returns:
            tuple[float, float]: The chosen sector
        """
        # Give each arc an integer index and create lists of indices and probabilities for
        # use in rv_discrete distribution
        sector_to_index = {}
        idx = []
        pk = []
        i = 0
        for sector, probability in arc_of_regard.items():
            idx.append(i)
            pk.append(probability)
            sector_to_index[i] = sector
            i += 1
        # Create a discrete distribution and draw one value from it
        distribution = rv_discrete(values=(idx, pk), seed=model._seed)
        # not specifying 'size' in rvs() defaults to 1 and returns scalar value
        rv_idx = distribution.rvs()
        return sector_to_index[rv_idx]

    @staticmethod
    def _get_sector_midpoint_angle(sector: tuple[float, float]):
        """Get midpoint of a sector in degrees.

        Sector is defined by two angles between 0 and 360 degrees
        and the sector is clockwise from the first to the second angle.

        Args:
            sector (tuple[float, float]): The sector

        Returns:
            float: The midpoint angle (degrees)
        """
        angle1 = sector[0]
        angle2 = sector[1]

        if angle1 > angle2:
            angle1 -= 360

        return ((angle2 + angle1) / 2) % 360

    def _reduce_to_field_of_view(
        self, senser: RetAgent, agents: list[RetAgent], sense_direction: float
    ) -> list[RetAgent]:
        """Reduce a list of agents to those within the sensor's field of view.

        Args:
            senser (RetAgent): agent this sensor belongs to.
            agents (list[RetAgent]): list of agents to be reduced.
            sense_direction (float): direction the sensor is facing in degrees from
                the positive y-axis

        Returns:
            list[RetAgent]: agents within the sensor's field of view.
        """
        sense_direction = sense_direction % 360
        if self.view_angle is not None:
            return [
                agent
                for agent in agents
                if self._get_angle_size_to_target(senser, agent, sense_direction)
                <= self.view_angle / 2
            ]
        return agents

    def _get_angle_size_to_target(
        self, sensor_agent: RetAgent, target_agent: RetAgent, sense_direction: float
    ) -> float:
        """Angle to target calculation.

        Get absolute angle between sensor agent and target agent when
        looking in the sense direction.

        Args:
            sensor_agent (RetAgent): The agent doing the sensing
            target_agent (RetAgent): The target agent
            sense_direction (float): The direction the sensor agent is sensing in (degrees)

        Returns:
            float: The absolute angle from the sensor agent to the target agent assuming the sensor
                agent is facing the sense direction
        """
        angle = (
            gu._heading_to_clockwise_angle_degrees(
                (
                    sensor_agent.model.space.get_heading(
                        sensor_agent.pos,  # type: ignore
                        target_agent.pos,  # type: ignore
                    )
                )
            )
            - sense_direction
        )
        # Make sure angle is between -180 and 180
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360

        return abs(angle)  # type: ignore

    def try_add_actively_sensing_agent(self, sensor_agent: RetAgent):
        """Add sensor agent to actively sensing agents manager if sensor is active.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
        """
        if self.is_active_sensor:
            sensor_agent.model.actively_sensing_agents_manager.add_agent(sensor_agent)

    @abstractmethod
    def get_new_instance(self) -> Sensor:  # pragma: no cover
        """Return a new instance of a functionally identical sensor.

        This abstract method is only defined here, rather than just inherited from
        the Template class, to provide more specific type hints.

        Returns:
            Sensor: New instance of the sensor
        """
        pass


class Acquire1dSensor(Sensor):
    """An abstract class representing a sensor which uses the 1D Acquire algorithm.

    The 1D Acquire algorithm is used to determine DRI probability.
    """

    def __init__(
        self,
        magnification: float,
        performance_curve: PerformanceCurve,
        johnson_criteria: JohnsonCriteria,
        wavelength: SensorWavelength,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ):
        """Initialise a 1D Acquire sensor.

        Args:
            magnification (float): The sensor magnification level (> 0)
            performance_curve (PerformanceCurve): MRC (Minimum Resolvable Contrast) or MRTD
                (Minimum Resolvable Temperature Difference) curve, a list of
                contrast/temperature difference, frequency (Cycles/milliradian) pairs
            johnson_criteria (JohnsonCriteria): A dictionary of confidence level to
                the Johnson's criteria value. i.e. the number of cycles required across the
                target for 50% of the observers to achieve the required confidence level.
            wavelength (SensorWavelength): SensorWavelength defining the operating
                wavelength of the sensor.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight and culture
                penetration calculations. Defaults to None, resulting in a sampling
                distance of 1/2 the pixel size of the terrain image for line of sight
                calculations, and 1/2 the pixel size of the culture image for culture
                penetration calculations.
            detection_timings (Optional[SensorDetectionTimings]): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.

        Raises:
            ValueError: Invalid magnification parameter
        """
        super().__init__(
            detection_timings=detection_timings,
            view_angle=view_angle,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )
        if magnification <= 0:
            raise ValueError(
                f"Acquire 1D Sensor magnification value ({magnification}) must be positive."
            )
        self._magnification = magnification
        self._performance_curve = performance_curve
        self._johnson_criteria = johnson_criteria

        if sampling_distance is None:
            sampling_distance = SensorSamplingDistance()

        self._sampling_distance = sampling_distance
        self._wavelength = wavelength

    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:
        """Run a sensor detection, returning a list of perceived agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
            all_target_agents (list[RetAgent]): A list of potentially perceivable agents

        Returns
            list[PerceivedAgent]: A list of agents perceived by the sensor.
        """
        sensor_use_time = sensor_agent.model.time_step
        target_agents = [
            agent for agent in all_target_agents if agent.agent_type != AgentType.GROUP
        ]
        perceived_agents: list[PerceivedAgent] = []

        for agent in target_agents:
            try:
                distance = sensor_agent.model.space.get_distance(
                    self.get_elevated_sensor_position(sensor_agent.pos), agent.pos
                )

                attenuated_distance = (
                    sensor_agent.model.space.get_culture_attenuated_distance_between(
                        pos_a=self.get_elevated_sensor_position(sensor_agent.pos),
                        pos_b=agent.pos,
                        sampling_distance=self._sampling_distance.sampling_distance,
                        wavelength=self._wavelength.wavelength,
                        distance=distance,
                    )
                )

                p_detect, p_recognise, p_identify = self._get_observation_confidence_probabilities(
                    distance, attenuated_distance, agent, sensor_use_time
                )
                p = sensor_agent.model.random.random()
                if p < p_identify:
                    perceived_agents.append(
                        self._create_perceived_agent(agent, Confidence.IDENTIFY)
                    )
                elif p < p_recognise:
                    perceived_agents.append(
                        self._create_perceived_agent(agent, Confidence.RECOGNISE)
                    )
                elif p < p_detect:
                    perceived_agents.append(self._create_perceived_agent(agent, Confidence.DETECT))
            except LineOfSightException:
                continue

        return perceived_agents

    def _create_perceived_agent(self, agent: RetAgent, confidence: Confidence) -> PerceivedAgent:
        return PerceivedAgentBasedOnAgent(
            confidence=confidence,
            sense_time=agent.model.get_time() + self._detection_timings.get_detection_delay(),
            agent=agent,
        )

    @abstractmethod
    def _get_observation_confidence_probabilities(
        self,
        distance: float,
        attenuated_distance: float,
        target_agent: RetAgent,
        sensor_use_time: timedelta,
    ) -> tuple[float, float, float]:  # pragma: no cover
        """Get probability of observation at each confidence level.

        Args:
            distance (float): Distance to target (m)
            attenuated_distance (float): Distance to target attenuated by culture (m)
            target_agent (RetAgent): The target agent
            sensor_use_time (timedelta): How long the sensor has been sensing for

        Returns:
            tuple[float, float, float]: The probabilities of observing the target at
                each confidence level (Detect, Recognise, Identify)
        """
        pass

    def _calculate_confidence_probability(
        self,
        confidence: Confidence,
        n_cycles: float,
        sensor_use_time_seconds: int,
        n50: Optional[float] = None,
    ) -> float:
        """Calculate the probability of observing to a confidence level.

        Args:
            confidence (Confidence): The confidence level of the observation
            n_cycles (float): The (interpolated) number of resolvable cycles across the target
            sensor_use_time_seconds (int): How long the sensor has been sensing for in seconds
            n50 (Optional[float]): Optional value for N50 (the number of cycles required across
                the target for 50% of the observers to achieve the specified confidence level),
                if None then the Johnson's Criteria value for the confidence level is used.

        Returns:
            float: The probability of observing at the given confidence level
        """
        if n50 is None:
            n50 = self._johnson_criteria[confidence]

        a = n_cycles / n50 if n50 > 0 else 0.0
        gamma = 2.7 + 0.7 * a
        try:
            p_inf = math.pow(a, gamma) / (1 + math.pow(a, gamma))
        except OverflowError:
            p_inf = 1.0
        return p_inf * (1 - math.exp((-1.0 * p_inf * sensor_use_time_seconds) / 3.4))

    def _get_contrast_values(self) -> list[float]:
        """Returns list of contrast values from contrast to frequency data.

        Returns:
            list[float]: list of contrast values
        """
        return [v[0] for v in self._performance_curve]

    def _get_frequency_values(self) -> list[float]:
        """Returns list of frequency values from contrast to frequency data.

        Returns:
            list[float]: list of frequency (Cycles/milliradian) values
        """
        return [v[1] for v in self._performance_curve]


class ContrastAcquire1dSensor(Acquire1dSensor):
    """A EO/II sensor which uses the 1D Acquire algorithm to determine DRI probability.

    EO = Electro-Optical.
    II = Image Intensifying.
    """

    def __init__(
        self,
        magnification: float,
        performance_curve: PerformanceCurve,
        johnson_criteria: JohnsonCriteria,
        sensor_type: ContrastSensorType,
        wavelength: SensorWavelength,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ):
        """Initialise a 1D Acquire Electro-Optical or Image Intensifying sensor.

        Args:
            magnification (float): The sensor magnification level (> 0)
            performance_curve (PerformanceCurve): MRC (Minimum Resolvable Contrast)
                curve, a list of contrast, frequency (Cycles/milliradian) pairs
            johnson_criteria (JohnsonCriteria): A dictionary of confidence level to
                the Johnson's criteria value. i.e. the number of cycles required across the
                target for 50% of the observers to achieve the required confidence level.
            sensor_type (ContrastSensorType): The type of contrast sensor (EO or II)
            wavelength (SensorWavelength): SensorWavelength defining the operating
                wavelength of the sensor.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight and culture
                penetration calculations. Defaults to None, resulting in a sampling
                distance of 1/2 the pixel size of the terrain image for line of sight
                calculations, and 1/2 the pixel size of the culture image for culture
                penetration calculations.
            detection_timings (Optional[SensorDetectionTimings]): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        super().__init__(
            magnification=magnification,
            performance_curve=performance_curve,
            johnson_criteria=johnson_criteria,
            wavelength=wavelength,
            sampling_distance=sampling_distance,
            detection_timings=detection_timings,
            view_angle=view_angle,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )
        self.sensor_type = sensor_type

    def _get_aerosol_transmission_coefficient(
        self, precipitation: Precipitation, meteorological_visibility_range: float
    ) -> float:
        """Get the transmission coefficient for aerosols.

        Args:
            precipitation (Precipitation): The type of precipitation
            meteorological_visibility_range (float): The meteorological visibility (km)

        Returns:
            float: The transmission coefficient for aerosols.

        Raises:
            ValueError: If the sensor type or precipitation type is not recognised.
        """
        if precipitation == Precipitation.CLEAR:
            m_precipitation = (
                self.sensor_type.get_clear_precipitation_aerosol_transmission_factor()
                / meteorological_visibility_range
            )
        elif precipitation == Precipitation.DRIZZLE_LIGHT:
            m_precipitation = 0.789
        elif precipitation == Precipitation.DRIZZLE_HEAVY:
            m_precipitation = 1.403
        elif precipitation == Precipitation.RAIN_MODERATE:
            m_precipitation = 1.768
        elif precipitation == Precipitation.RAIN_HEAVY:
            m_precipitation = 2.735
        elif precipitation == Precipitation.THUNDERSTORM:
            m_precipitation = 2.429
        elif precipitation == Precipitation.SNOW:
            m_precipitation = 3.912 / meteorological_visibility_range
        else:
            raise ValueError(f"Precipitation Type {precipitation.value} unknown.")

        return m_precipitation

    def _get_water_vapour_transmission_coefficient(self, relative_humidity: float) -> float:
        """Get the transmission coefficient for water vapour and gaseous absorption.

        Args:
            relative_humidity (float): The relative humidity, between 0 and 1, exclusive

        Returns:
            float: The transmission coefficient for water vapour and gaseous absorption.

        Raises:
            ValueError: If the relative humidity is not between 0 and 1, exclusive, or
                the sensor type is not recognised.
        """
        if relative_humidity <= 0 or relative_humidity >= 1:
            raise ValueError(
                f"Relative humidity ({relative_humidity}) must be between 0 and 1, exclusive."
            )

        return self.sensor_type.get_water_vapour_transmission_coefficient(relative_humidity)

    def _get_atmospheric_transmission(
        self,
        distance: float,
        relative_humidity: float,
        precipitation: Precipitation,
        dust: float,
        meteorological_visibility_range: float,
    ) -> float:
        """Get atmospheric transmission value.

        Args:
            distance (float): Distance to target (m)
            relative_humidity (float): The relative humidity, between 0 and 1, exclusive
            precipitation (Precipitation): The type of precipitation
            dust (float): The atmospheric dust (g/m3) (>=0)
            meteorological_visibility_range (float): The meteorological visibility (km)

        Returns:
            float: The atmospheric transmission value

        Raises:
            ValueError: If the given atmospheric dust value is negative.
        """
        if dust < 0:
            raise ValueError(f"Atmospheric dust ({dust} g/m3) cannot be negative.")

        distance_km = distance / 1000
        k_relative_humidity = self._get_water_vapour_transmission_coefficient(relative_humidity)
        m_precipitation = self._get_aerosol_transmission_coefficient(
            precipitation, meteorological_visibility_range
        )
        # Transmission factor for water vapour
        t_rh = math.exp(-1.0 * k_relative_humidity * distance_km)
        # Transmission factor for aerosols
        t_precipitation = math.exp(-1.0 * m_precipitation * distance_km)
        # Transmission factor for dust
        t_dust = math.exp(-1.0 * 0.32 * dust * distance_km)
        return t_rh * t_precipitation * t_dust

    def _get_observation_confidence_probabilities(
        self,
        distance: float,
        attenuated_distance: float,
        target_agent: RetAgent,
        sensor_use_time: timedelta,
    ) -> tuple[float, float, float]:
        """Get probability of observation at each confidence level.

        Args:
            distance (float): Distance to target (m)
            attenuated_distance (float): Distance to target attenuated by culture (m)
            target_agent (RetAgent): The target agent
            sensor_use_time (timedelta): How long the sensor has been sensing for

        Returns:
            tuple[float, float, float]: The probabilities of observing the target at
                each confidence level (Detect, Recognise, Identify)
        """
        if attenuated_distance == 0:
            return (1.0, 1.0, 1.0)
        space: ContinuousSpaceWithTerrainAndCulture3d = target_agent.model.space
        # Environmental variables
        relative_humidity = space.get_relative_humidity()
        precipitation = space.get_precipitation()
        dust = space.get_atmospheric_dust()
        meteorological_visibility_range = space.get_meteorological_visibility_range()
        # Culture variables
        if target_agent.agent_type == AgentType.AIR and space.is_agent_above_terrain(target_agent):
            background_reflectivity = space.sky_background_reflectivity
        else:
            background_reflectivity = space.get_culture(target_agent.pos).reflectivity
        # Target agent variables
        target_reflectivity = target_agent.reflectivity
        critical_dimension = target_agent.critical_dimension

        contrast_at_target = (
            abs(target_reflectivity - background_reflectivity) / background_reflectivity
        )
        contrast_at_sensor = contrast_at_target * self._get_atmospheric_transmission(
            distance, relative_humidity, precipitation, dust, meteorological_visibility_range
        )
        n_cycles = np.interp(
            contrast_at_sensor,
            self._get_contrast_values(),
            self._get_frequency_values(),
        )
        n_cycles *= critical_dimension * self._magnification / (attenuated_distance / 1000)
        sensor_use_time_seconds = sensor_use_time.seconds
        p_detect = self._calculate_confidence_probability(
            Confidence.DETECT, n_cycles, sensor_use_time_seconds
        )
        p_recognise = self._calculate_confidence_probability(
            Confidence.RECOGNISE, n_cycles, sensor_use_time_seconds
        )
        p_identify = self._calculate_confidence_probability(
            Confidence.IDENTIFY, n_cycles, sensor_use_time_seconds
        )
        return (p_detect, p_recognise, p_identify)

    def get_new_instance(self) -> ContrastAcquire1dSensor:
        """Return a new instance of an ContrastAcquire1dSensor.

        Returns:
            ContrastAcquire1dSensor: New instance of the sensor
        """
        return ContrastAcquire1dSensor(
            magnification=self._magnification,
            performance_curve=self._performance_curve,
            johnson_criteria=self._johnson_criteria,
            sensor_type=self.sensor_type,
            wavelength=self._wavelength,
            sampling_distance=self._sampling_distance,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class TemperatureAcquire1dSensor(Acquire1dSensor):
    """A TI sensor which uses the 1D Acquire algorithm to determine DRI probability.

    TI = Temperature Intensifying.

    """

    def __init__(
        self,
        magnification: float,
        performance_curve: PerformanceCurve,
        johnson_criteria: JohnsonCriteria,
        wavelength: SensorWavelength,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ):
        """Initialise a 1D Acquire Temperature Intensifying sensor.

        Args:
            magnification (float): The sensor magnification level (> 0)
            performance_curve (PerformanceCurve): MRTD (Minimum Resolvable Temperature
                Difference) curve, a list of temperature difference (degrees C), frequency
                (Cycles/milliradian) pairs
            johnson_criteria (JohnsonCriteria): A dictionary of confidence level to
                the Johnson's criteria value. i.e. the number of cycles required across the
                target for 50% of the observers to achieve the required confidence level.
            wavelength (SensorWavelength): SensorWavelength defining the operating
                wavelength of the sensor.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight and culture
                penetration calculations. Defaults to None, resulting in a sampling
                distance of 1/2 the pixel size of the terrain image for line of sight
                calculations, and 1/2 the pixel size of the culture image for culture
                penetration calculations.
            detection_timings (Optional[SensorDetectionTimings]): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        super().__init__(
            magnification=magnification,
            performance_curve=performance_curve,
            johnson_criteria=johnson_criteria,
            wavelength=wavelength,
            sampling_distance=sampling_distance,
            detection_timings=detection_timings,
            view_angle=view_angle,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )

    def _get_atmospheric_transmission(
        self,
        distance: float,
        relative_humidity: float,
        meteorological_visibility_range: float,
        ambient_temperature: float,
    ) -> float:
        """Get atmospheric transmission value.

        Args:
            distance (float): Distance to target (m)
            relative_humidity (float): The relative humidity, between 0 and 1, exclusive
            meteorological_visibility_range (float): Meteorological visibility range (km)
            ambient_temperature (float): Ambient temperature (degrees C)

        Returns:
            float: The atmospheric transmission value

        Raises:
            ValueError: If the relative humidity is not between 0 and 1, exclusive
        """
        if relative_humidity <= 0 or relative_humidity >= 1:
            raise ValueError(
                f"Relative humidity ({relative_humidity}) must be between 0 and 1, exclusive."
            )

        if meteorological_visibility_range < 2.5:
            beta = 0.8667 - 0.1667 * meteorological_visibility_range
        elif meteorological_visibility_range < 6.0:
            beta = 0.45 - 0.0628 * (meteorological_visibility_range - 2.5)
        elif meteorological_visibility_range < 10.0:
            beta = 0.23 - 0.02 * (meteorological_visibility_range - 6.0)
        else:
            beta = 0.15

        beta_atmos = beta + 0.083 * math.pow(relative_humidity, 1.19) * math.exp(
            0.075 * ambient_temperature
        )
        return math.exp(-1.0 * beta_atmos * distance / 1000)

    def _get_observation_confidence_probabilities(
        self,
        distance: float,
        attenuated_distance: float,
        target_agent: RetAgent,
        sensor_use_time: timedelta,
    ) -> tuple[float, float, float]:
        """Get probability of observation at each confidence level.

        Args:
            distance (float): Distance to target (m)
            attenuated_distance (float): Distance to target attenuated by culture (m)
            target_agent (RetAgent): The target agent
            sensor_use_time (timedelta): How long the sensor has been sensing for

        Returns:
            tuple[float, float, float]: The probabilities of observing the target at
                each confidence level (Detect, Recognise, Identify)
        """
        if attenuated_distance == 0:
            return (1.0, 1.0, 1.0)
        space: ContinuousSpaceWithTerrainAndCulture3d = target_agent.model.space
        # Environmental variables
        relative_humidity = space.get_relative_humidity()
        meteorological_visibility_range = space.get_meteorological_visibility_range()
        ambient_temperature = space.get_ambient_temperature()
        # Culture variables
        if target_agent.agent_type == AgentType.AIR and space.is_agent_above_terrain(target_agent):
            background_temperature = 0.0
        else:
            background_temperature = space.get_culture_temperature(target_agent.pos)
        # Target agent variables
        target_temperature = target_agent.temperature
        target_temperature_std_dev = target_agent.temperature_std_dev
        critical_dimension = target_agent.critical_dimension

        delta_t_at_target = math.sqrt(
            math.pow(target_temperature - background_temperature, 2)
            + math.pow(target_temperature_std_dev, 2)
        )
        delta_t_at_sensor = delta_t_at_target * self._get_atmospheric_transmission(
            distance, relative_humidity, meteorological_visibility_range, ambient_temperature
        )

        n_cycles = np.interp(
            delta_t_at_sensor,
            self._get_contrast_values(),
            self._get_frequency_values(),
        )
        n_cycles *= critical_dimension * self._magnification / (attenuated_distance / 1000)

        sensor_use_time_seconds = sensor_use_time.seconds

        # assume non-cluttered culture
        n50_detect = (0.75 / delta_t_at_sensor) + 0.75 if delta_t_at_sensor > 0 else 0.0

        p_detect = self._calculate_confidence_probability(
            Confidence.DETECT, n_cycles, sensor_use_time_seconds, n50=n50_detect
        )
        p_recognise = self._calculate_confidence_probability(
            Confidence.RECOGNISE, n_cycles, sensor_use_time_seconds
        )
        p_identify = self._calculate_confidence_probability(
            Confidence.IDENTIFY, n_cycles, sensor_use_time_seconds
        )
        return (p_detect, p_recognise, p_identify)

    def get_new_instance(self) -> TemperatureAcquire1dSensor:
        """Return a new instance of an TemperatureAcquire1dSensor.

        Returns:
            TemperatureAcquire1dSensor: New instance of the sensor
        """
        return TemperatureAcquire1dSensor(
            magnification=self._magnification,
            performance_curve=self._performance_curve,
            johnson_criteria=self._johnson_criteria,
            wavelength=self._wavelength,
            sampling_distance=self._sampling_distance,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class DistanceAttenuatedSensor(Sensor):
    """A sensor with a detection probability that is attenuated by distance.

    Also attenuated by clutter.
    """

    def __init__(
        self,
        distance_thresholds: SensorDistanceThresholds,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        clutter_attenuator: Optional[SensorClutterAttenuator] = None,
        casualty_state_confidence: Confidence = Confidence.RECOGNISE,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ) -> None:
        """Initialise a distance attenuated sensor.

        Args:
            distance_thresholds (SensorDistanceThresholds): object defining distance
                thresholds associated with perceived agent confidence levels.
            detection_timings (Optional[SensorDetectionTimings): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            clutter_attenuator (Optional[SensorClutterAttenuator]): object
                defining how clutter attenuates sensor results. Defaults to None,
                resulting in zero attenuation due to clutter.
            casualty_state_confidence (Confidence): Minimum confidence level
                at which casualty state is known to sensor. Defaults to RECOGNISE.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        super().__init__(
            detection_timings=detection_timings,
            view_angle=view_angle,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )

        if clutter_attenuator is None:
            clutter_attenuator = SensorClutterAttenuator()

        self._clutter_attenuator = clutter_attenuator
        self._distance_thresholds = distance_thresholds
        self._casualty_state_confidence = casualty_state_confidence

    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:
        """Run a sensor detection, returning a list of perceived agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
            all_target_agents (list[RetAgent]): A list of potentially perceivable agents

        Returns
            list[PerceivedAgent]: A list of agents perceived by the sensor.
        """
        all_detectable_agents = [
            agent for agent in self.detectable_agents(sensor_agent) if agent in all_target_agents
        ]

        agent_distance_dict = {
            agent: sensor_agent.model.space.get_distance(
                self.get_elevated_sensor_position(sensor_agent.pos), agent.pos
            )
            for agent in all_detectable_agents
        }

        return self.perceived_agent_generator(sensor_agent, agent_distance_dict)

    def detectable_agents(self, sensor_agent: RetAgent) -> list[RetAgent]:
        """Return agents within DETECT confidence range of sensor agent.

        Args:
            sensor_agent (RetAgent): Agent using sensor

        Returns:
            list[RetAgent]: Agents within DETECT confidence range of sensor, excluding
                the sensor agent
        """
        grid_content: list[GridContent] = sensor_agent.model.space.get_neighbors(
            self.get_elevated_sensor_position(sensor_agent.pos),
            self._distance_thresholds.confidence_distances[Confidence.DETECT],
            False,
        )

        agents: list[RetAgent] = grid_content  # type: ignore

        return agents

    def clutter_attenuation(
        self, sensor_agent: RetAgent, detected_agent_dict: dict[RetAgent, float]
    ) -> dict[RetAgent, float]:
        """Increases perceived distance of agents.

        Determined by value of clutter at agent location, and sensor clutter
        attenuator strength.

        Args:
            sensor_agent (RetAgent): Agent using sensor
            detected_agent_dict (dict[RetAgent, float]): Dictionary of agents and
                their corresponding distance from the sensor agent

        Returns:
            dict[RetAgent, float]: Dictionary of agents and their corresponding
                distance from the sensor agent, modified by clutter
        """
        clutter_attenuated_dict = {}

        for agent, distance_value in detected_agent_dict.items():
            clutter_at_target = agent.model.space.clutter_field.get_value(agent.pos, sensor_agent)

            clutter_attenuated_dict[agent] = distance_value + (
                clutter_at_target * self._clutter_attenuator.attenuation_strength
            )

        return clutter_attenuated_dict

    def get_perceived_agent(
        self, sensor_agent: RetAgent, detected_agent: RetAgent, confidence: Confidence
    ) -> PerceivedAgent:
        """Create a perceived agent from a RetAgent class with given confidence.

        Args:
            sensor_agent (RetAgent): Agent using sensor.
            detected_agent (RetAgent): The detected agent.
            confidence (Confidence): The confidence level of detection.

        Returns:
            PerceivedAgent: The corresponding perceived agent.
        """
        return PerceivedAgentBasedOnAgent(
            confidence=confidence,
            sense_time=detected_agent.model.get_time()
            + self._detection_timings.get_detection_delay(),
            agent=detected_agent,
            casualty_state_confidence=self._casualty_state_confidence,
        )

    def perceived_agent_generator(
        self, sensor_agent: RetAgent, detected_agent_dict: dict[RetAgent, float]
    ) -> list[PerceivedAgent]:
        """Generate a list of of perceived agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor
            detected_agent_dict (dict[RetAgent, float]): Dictionary of agents and their
                attenuation modified distance from the sensor

        Returns:
            list[PerceivedAgent]: List of perceived agents
        """
        detected_agent_dict = self.clutter_attenuation(sensor_agent, detected_agent_dict)

        all_perceived_agents: list[PerceivedAgent] = []

        for detected_agent, test_value in detected_agent_dict.items():
            if test_value < self._distance_thresholds.confidence_distances[Confidence.IDENTIFY]:
                all_perceived_agents.append(
                    self.get_perceived_agent(sensor_agent, detected_agent, Confidence.IDENTIFY)
                )
            elif test_value < self._distance_thresholds.confidence_distances[Confidence.RECOGNISE]:
                all_perceived_agents.append(
                    self.get_perceived_agent(sensor_agent, detected_agent, Confidence.RECOGNISE)
                )
            elif test_value < self._distance_thresholds.confidence_distances[Confidence.DETECT]:
                all_perceived_agents.append(
                    self.get_perceived_agent(sensor_agent, detected_agent, Confidence.DETECT)
                )

        return all_perceived_agents

    def get_new_instance(self) -> DistanceAttenuatedSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            DistanceAttenuatedSensor: New instance of the sensor
        """
        return DistanceAttenuatedSensor(
            distance_thresholds=self._distance_thresholds,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            clutter_attenuator=self._clutter_attenuator,
            casualty_state_confidence=self._casualty_state_confidence,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class LineOfSightSensor(DistanceAttenuatedSensor):
    """Line of Sight sensor with distance attenuation."""

    def __init__(
        self,
        distance_thresholds: SensorDistanceThresholds,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        clutter_attenuator: Optional[SensorClutterAttenuator] = None,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ) -> None:
        """Initialise a distance attenuated sensor that requires LOS.

        Args:
            distance_thresholds (SensorDistanceThresholds): object defining distance
                thresholds associated with perceived agent confidence levels.
            detection_timings (Optional[SensorDetectionTimings], optional): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            clutter_attenuator (Optional[SensorClutterAttenuator], optional): object
                defining how clutter attenuates sensor results. Defaults to None,
                resulting in zero attenuation due to clutter.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight calculations.
                Defaults to None, resulting in a sampling distance of 1/2 the pixel size
                of the terrain image for line of sight calculations.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        super().__init__(
            detection_timings=detection_timings,
            view_angle=view_angle,
            clutter_attenuator=clutter_attenuator,
            distance_thresholds=distance_thresholds,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )

        if sampling_distance is None:
            sampling_distance = SensorSamplingDistance()

        self._sampling_distance = sampling_distance

    def detectable_agents(self, sensor_agent: RetAgent) -> list[RetAgent]:
        """Return detectable agent.

        Detectable agents are those within DETECT confidence range of sensor agent, with
        line of sight.

        Args:
            sensor_agent (RetAgent): Agent using sensor.

        Returns:
            list[RetAgent]: Agents within DETECT confidence range of sensor, with line
                of sight, excluding the sensor agent.
        """
        agents_in_detect_range = super().detectable_agents(sensor_agent)

        return self.check_agents_for_los(sensor_agent, agents_in_detect_range)

    def check_agents_for_los(
        self, sensor_agent: RetAgent, all_detectable_agents: list[RetAgent]
    ) -> list[RetAgent]:
        """Check whether agents are detectable.

        Wrapper for space.check_line_of_sight that operates on a list of agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor.
            all_detectable_agents (list[RetAgent]): List of agents within detection
                range.

        Returns:
            list[RetAgent]: all agents from all_detectable_agents list where line of
                sight is true.
        """
        return [
            agent
            for agent in all_detectable_agents
            if sensor_agent.model.space.check_line_of_sight(
                self.get_elevated_sensor_position(sensor_agent.pos),
                agent.pos,
                self._sampling_distance.sampling_distance,
            )
            is True
        ]

    def get_new_instance(self) -> LineOfSightSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            LineOfSightSensor: New instance of the sensor
        """
        return LineOfSightSensor(
            distance_thresholds=self._distance_thresholds,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            clutter_attenuator=self._clutter_attenuator,
            sampling_distance=self._sampling_distance,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class CultureAttenuatedSensor(LineOfSightSensor):
    """Line of sight sensor attenuated by distance and culture."""

    def __init__(
        self,
        wavelength: SensorWavelength,
        distance_thresholds: SensorDistanceThresholds,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        clutter_attenuator: Optional[SensorClutterAttenuator] = None,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ) -> None:
        """Initialise a distance attenuated sensor.

        Requires LOS and is attenuated by culture.

        Args:
            wavelength (SensorWavelength): SensorWavelength defining the operating
                wavelength of the sensor.
            distance_thresholds (SensorDistanceThresholds): object defining distance
                thresholds associated with perceived agent confidence levels.
            detection_timings (Optional[SensorDetectionTimings], optional): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            clutter_attenuator (Optional[SensorClutterAttenuator], optional): object
                defining how clutter attenuates sensor results. Defaults to None,
                resulting in zero attenuation due to clutter.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight and culture
                penetration calculations. Defaults to None, resulting in a sampling
                distance of 1/2 the pixel size of the terrain image for line of sight
                calculations, and 1/2 the pixel size of the culture image for culture
                penetration calculations.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        super().__init__(
            detection_timings=detection_timings,
            view_angle=view_angle,
            clutter_attenuator=clutter_attenuator,
            distance_thresholds=distance_thresholds,
            sampling_distance=sampling_distance,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )

        self._wavelength = wavelength

    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:
        """Run a sensor detection, returning a list of perceived agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
            all_target_agents (list[RetAgent]): A list of potentially perceivable agents

        Returns
            list[PerceivedAgent]: A list of agents perceived by the sensor.
        """
        all_detectable_agents = [
            agent for agent in self.detectable_agents(sensor_agent) if agent in all_target_agents
        ]

        agent_attenuated_distance_dict = {}

        for agent in all_detectable_agents:
            agent_attenuated_distance_dict[
                agent
            ] = sensor_agent.model.space.get_culture_attenuated_distance_between(
                pos_a=self.get_elevated_sensor_position(sensor_agent.pos),
                pos_b=agent.pos,
                sampling_distance=self._sampling_distance.sampling_distance,
                wavelength=self._wavelength.wavelength,
            )

        return self.perceived_agent_generator(sensor_agent, agent_attenuated_distance_dict)

    def get_new_instance(self) -> CultureAttenuatedSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            CultureAttenuatedSensor: New instance of the sensor
        """
        return CultureAttenuatedSensor(
            wavelength=self._wavelength,
            distance_thresholds=self._distance_thresholds,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            clutter_attenuator=self._clutter_attenuator,
            sampling_distance=self._sampling_distance,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class WeaponPlatformLocatingSensor(LineOfSightSensor):
    """Sensor for detecting projectile weapons and the agents that fired them."""

    def __init__(
        self,
        distance_thresholds: SensorDistanceThresholds,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        clutter_attenuator: Optional[SensorClutterAttenuator] = None,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ) -> None:
        """Initialise a weapon platform locating sensor.

        Requires LOS. Can only sense unguided projectile agents.

        Args:
            distance_thresholds (SensorDistanceThresholds): object defining distance
                thresholds associated with perceived agent confidence levels.
            detection_timings (Optional[SensorDetectionTimings], optional): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            clutter_attenuator (Optional[SensorClutterAttenuator], optional): object
                defining how clutter attenuates sensor results. Defaults to None,
                resulting in zero attenuation due to clutter.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight and culture
                penetration calculations. Defaults to None, resulting in a sampling
                distance of 1/2 the pixel size of the terrain image for line of sight
                calculations, and 1/2 the pixel size of the culture image for culture
                penetration calculations.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        super().__init__(
            detection_timings=detection_timings,
            view_angle=view_angle,
            clutter_attenuator=clutter_attenuator,
            distance_thresholds=distance_thresholds,
            sampling_distance=sampling_distance,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )

    def perceived_agent_generator(
        self, sensor_agent: RetAgent, detected_agent_dict: dict[RetAgent, float]
    ) -> list[PerceivedAgent]:
        """Generate a list of of perceived agents including weapon platforms.

        Args:
            sensor_agent (RetAgent): Agent using sensor
            detected_agent_dict (dict[RetAgent, float]): Dictionary of agents and their
                attenuation modified distance from the sensor

        Returns:
            list[PerceivedAgent]: List of perceived agents
        """
        detected_agent_dict = self.clutter_attenuation(sensor_agent, detected_agent_dict)

        all_perceived_agents: list[PerceivedAgent] = []

        for detected_agent, test_value in detected_agent_dict.items():
            if (
                test_value < self._distance_thresholds.confidence_distances[Confidence.IDENTIFY]
            ) & (detected_agent.agent_type == AgentType.PROJECTILE):
                all_perceived_agents.append(
                    self.get_perceived_agent(sensor_agent, detected_agent, Confidence.IDENTIFY)
                )
            elif (
                test_value < self._distance_thresholds.confidence_distances[Confidence.RECOGNISE]
            ) & (detected_agent.agent_type == AgentType.PROJECTILE):
                all_perceived_agents.append(
                    self.get_perceived_agent(sensor_agent, detected_agent, Confidence.RECOGNISE)
                )
            elif (
                test_value < self._distance_thresholds.confidence_distances[Confidence.DETECT]
            ) & (detected_agent.agent_type == AgentType.PROJECTILE):
                all_perceived_agents.append(
                    self.get_perceived_agent(sensor_agent, detected_agent, Confidence.DETECT)
                )

        all_agents = sensor_agent.model.get_all_agents()
        for perception in all_perceived_agents:
            if (
                (perception.agent_type == AgentType.PROJECTILE)
                & (perception.casualty_state == AgentCasualtyState.ALIVE)
                & (
                    (perception.confidence == Confidence.RECOGNISE)
                    | (perception.confidence == Confidence.IDENTIFY)
                )
            ):
                projectile: ProjectileAgent = [
                    a
                    for a in all_agents
                    if str(a.unique_id) == perception.get_unique_id_log_representation()
                ][0]
                all_perceived_agents.append(  # noqa: B038
                    PerceivedAgentBasedOnAgent(
                        confidence=Confidence.IDENTIFY,
                        sense_time=detected_agent.model.get_time()
                        + self._detection_timings.get_detection_delay(),
                        agent=projectile.firer,
                        casualty_state_confidence=self._casualty_state_confidence,
                        location=projectile.pos_when_fired,
                    )
                )

        return all_perceived_agents

    def get_results(self, sensor_agent: RetAgent) -> list[PerceivedAgent]:
        """Get valid perceived agents detected using this sensor.

        Valid agents are those with a sense time equal to or earlier than the current
        model time. Once these agents are filtered, only agents with sense times in the
        future are retained in the perceived agent cache of the sensor.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection

        Returns:
            list[PerceivedAgent]: list of perceived agents with a sense time equal to or
                earlier than the current model time
        """
        current_time = sensor_agent.model.get_time()

        valid_agents = [a for a in self._cached_agents if a._sense_time <= current_time]

        sensor_agent.model.logger.log_observation_record(
            sensor_agent, valid_agents, message="Identified with weapon tracking sensor"
        )

        self._cached_agents = [a for a in self._cached_agents if a._sense_time > current_time]

        return valid_agents

    def get_new_instance(self) -> WeaponPlatformLocatingSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            WeaponPlatformLocatingSensor: New instance of the sensor
        """
        return WeaponPlatformLocatingSensor(
            distance_thresholds=self._distance_thresholds,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            clutter_attenuator=self._clutter_attenuator,
            sampling_distance=self._sampling_distance,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class CEPSensor(CultureAttenuatedSensor):
    """A sensor with a CEP (circular error probable) accuracy range.

    https://en.wikipedia.org/wiki/Circular_error_probable
    """

    def __init__(
        self,
        cep_range_data: dict[AgentType, RangeToCepData],
        wavelength: SensorWavelength,
        distance_thresholds: SensorDistanceThresholds,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        clutter_attenuator: Optional[SensorClutterAttenuator] = None,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ) -> None:
        """Initialise a CEP sensor.

        Args:
            cep_range_data (dict[AgentType, RangeToCepData]): CEP range data
                which consists of a RangeToCepData for each AgentType
            wavelength (SensorWavelength): SensorWavelength defining the operating
                wavelength of the sensor.
            distance_thresholds (SensorDistanceThresholds): object defining distance
                thresholds associated with perceived agent confidence levels.
            detection_timings (Optional[SensorDetectionTimings], optional): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            clutter_attenuator (Optional[SensorClutterAttenuator], optional): object
                defining how clutter attenuates sensor results. Defaults to None,
                resulting in zero attenuation due to clutter.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight and culture
                penetration calculations. Defaults to None, resulting in a sampling
                distance of 1/2 the pixel size of the terrain image for line of sight
                calculations, and 1/2 the pixel size of the culture image for culture
                penetration calculations.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        if sensor_filters is None:
            sensor_filters = []

        super().__init__(
            wavelength=wavelength,
            distance_thresholds=distance_thresholds,
            detection_timings=detection_timings,
            view_angle=view_angle,
            clutter_attenuator=clutter_attenuator,
            sampling_distance=sampling_distance,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )
        self._cep_range_data = cep_range_data

    def _is_agent_detectable(self, sensor_agent: RetAgent, agent: RetAgent) -> bool:
        """Check whether agent is within detectable range of sensor agent.

        Args:
            sensor_agent (RetAgent): The sensor agent.
            agent (RetAgent): The target agent.

        Returns:
            bool: Whether the target agent is within detectable range of the sensor agent.
        """
        if agent.agent_type not in self._cep_range_data:
            return False

        agent_type = agent.agent_type
        distance = sensor_agent.model.space.get_distance(
            self.get_elevated_sensor_position(sensor_agent.pos), agent.pos
        )
        return self._is_distance_within_range(distance, agent_type)

    def detectable_agents(self, sensor_agent: RetAgent) -> list[RetAgent]:
        """Return detectable agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor.

        Returns:
            list[RetAgent]: Agents within DETECT confidence range of sensor, with line
                of sight and within the max distance of the CEP range for the agent type,
                excluding the sensor agent.
        """
        agents_in_detect_range = super().detectable_agents(sensor_agent)

        return [
            agent
            for agent in agents_in_detect_range
            if self._is_agent_detectable(sensor_agent, agent)
        ]

    def get_perceived_agent(
        self, sensor_agent: RetAgent, detected_agent: RetAgent, confidence: Confidence
    ) -> PerceivedAgent:
        """Create a perceived agent from a RetAgent class with given confidence.

        Args:
            sensor_agent (RetAgent): Agent using sensor.
            detected_agent (RetAgent): The detected agent.
            confidence (Confidence): The confidence level of detection.

        Returns:
            PerceivedAgent: The corresponding perceived agent.
        """
        distance = sensor_agent.model.space.get_distance(
            self.get_elevated_sensor_position(sensor_agent.pos), detected_agent.pos
        )
        cep = self._get_cep_distance(distance, detected_agent.agent_type)
        perceived_location = self._offset_coordinate(detected_agent.pos, cep, sensor_agent.model)
        return PerceivedAgentBasedOnAgent(
            confidence=confidence,
            sense_time=detected_agent.model.get_time()
            + self._detection_timings.get_detection_delay(),
            agent=detected_agent,
            location=perceived_location,
            casualty_state_confidence=self._casualty_state_confidence,
        )

    def _is_distance_within_range(self, distance: float, agent_type: AgentType) -> bool:
        """Checks whether given distance is within range for the given agent type.

        Args:
            distance (float): the distance of the agent to check
            agent_type (AgentType): the type of agent

        Returns:
            bool: whether the distance is within the agent type range
        """
        agent_ranges = self._cep_range_data[agent_type].get_range_values()
        return min(agent_ranges) <= distance <= max(agent_ranges)

    def _get_cep_distance(self, distance: float, agent_type: AgentType) -> float:
        """Get the interpolated CEP distance (m) for a given distance and agent type.

        Args:
            distance (float): the distance to the sensed agent
            agent_type (AgentType): the type of sensed agent

        Returns:
            float: interpolated CEP distance (m)
        """
        agent_data = self._cep_range_data[agent_type]
        cep: float = np.interp(
            distance,
            agent_data.get_range_values(),
            agent_data.get_cep_values(),
        )
        return cep

    def _offset_coordinate(
        self,
        location: Coordinate2dOr3d,
        cep: float,
        model: RetModel,
    ) -> Coordinate3d:
        """Offset a 3D location by the CEP distance in a random direction.

        Args:
            location (Coordinate2dOr3d): the 3D location to offset
            cep (float): the CEP distance (m)
            model (RetModel): The RetModel

        Returns:
            Coordinate3d: the new, offset, 3D location
        """
        location = cast("Coordinate3d", location)
        space: ContinuousSpaceWithTerrainAndCulture3d = model.space
        random = model.random
        terrain_height = space.get_terrain_height((location[0], location[1]))
        if location[2] > terrain_height:
            # Offset in 3 dimensions
            # https://mathworld.wolfram.com/SpherePointPicking.html
            u = random.uniform(0, 1)
            v = random.uniform(0, 1)
            theta = 2 * math.pi * u
            phi = math.acos(2 * v - 1)
            w = math.cos(phi)
            x = cep * math.sqrt(1 - w**2) * math.cos(theta)
            y = cep * math.sqrt(1 - w**2) * math.sin(theta)
            z = cep * w

            loc_x = max(min(location[0] + x, space.x_max), space.x_min)
            loc_y = max(min(location[1] + y, space.y_max), space.y_min)
            loc_z = max(location[2] + z, terrain_height)
        else:
            # If agent is on terrain then only offset in x,y dimensions
            heading_error = random.uniform(0.0, 2 * math.pi)
            loc_x = location[0] + math.cos(heading_error) * cep
            loc_y = location[1] + math.sin(heading_error) * cep
            loc_x = max(min(loc_x, space.x_max), space.x_min)
            loc_y = max(min(loc_y, space.y_max), space.y_min)
            loc_z = location[2]

        return (loc_x, loc_y, loc_z)

    def get_new_instance(self) -> CEPSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            CEPSensor: New instance of the CEP sensor
        """
        return CEPSensor(
            cep_range_data=self._cep_range_data,
            wavelength=self._wavelength,
            distance_thresholds=self._distance_thresholds,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            clutter_attenuator=self._clutter_attenuator,
            sampling_distance=self._sampling_distance,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class SigIntSensor(CEPSensor):
    """A Signals Intelligence Sensor which uses CEP range data.

    Uses the CEPSensor detection method but can only sense agents which
    are currently sensing using an active sensor.

    When using this sensor type agents are sensed at the end of the timestep,
    so any other orders (e.g. communicate tasks) must wait until the next
    timestep to access the perceived agents.
    """

    def __init__(
        self,
        cep_range_data: dict[AgentType, RangeToCepData],
        wavelength: SensorWavelength,
        distance_thresholds: SensorDistanceThresholds,
        detection_timings: Optional[SensorDetectionTimings] = None,
        view_angle: Optional[float] = None,
        clutter_attenuator: Optional[SensorClutterAttenuator] = None,
        sampling_distance: Optional[SensorSamplingDistance] = None,
        is_active_sensor: bool = False,
        sensor_filters: Optional[list[SensorFilter]] = None,
        sensor_side_effects: Optional[list[SensorSideEffect]] = None,
        height_of_sensor: float = 0,
    ) -> None:
        """Initialise a SigInt sensor.

        Args:
            cep_range_data (dict[AgentType, RangeToCepData]): CEP range data
                which consists of a RangeToCepData for each AgentType
            wavelength (SensorWavelength): SensorWavelength defining the operating
                wavelength of the sensor.
            distance_thresholds (SensorDistanceThresholds): object defining distance
                thresholds associated with perceived agent confidence levels.
            detection_timings (Optional[SensorDetectionTimings], optional): object that
                defines sensor detection delay and how long perceived agents are cached.
                Defaults to None, which results in a default SensorDetectionTimings
                object (zero detection delay).
            view_angle (Optional[float]): angular width of the sensor's field of view.
            clutter_attenuator (Optional[SensorClutterAttenuator], optional): object
                defining how clutter attenuates sensor results. Defaults to None,
                resulting in zero attenuation due to clutter.
            sampling_distance (Optional[SensorSamplingDistance], optional): object
                defining how sampling distance used for line of sight and culture
                penetration calculations. Defaults to None, resulting in a sampling
                distance of 1/2 the pixel size of the terrain image for line of sight
                calculations, and 1/2 the pixel size of the culture image for culture
                penetration calculations.
            is_active_sensor (bool): whether the sensor is active, i.e. whether sensing
                could make an agent more visible to other sensors. Defaults to False
            sensor_filters (Optional[list[SensorFilter]]): a list of sensor filters, used to
                customise which agents a sensor can detect. Defaults to None
            sensor_side_effects (Optional[list[SensorFilter]]): a list of sensor side effects,
                used to add custom side effects when agents are sensed. Defaults to None
            height_of_sensor (float): the height of the sensor above the agent.
        """
        if sensor_filters is None:
            sensor_filters = [ActiveFilter()]

        super().__init__(
            cep_range_data=cep_range_data,
            wavelength=wavelength,
            distance_thresholds=distance_thresholds,
            detection_timings=detection_timings,
            view_angle=view_angle,
            clutter_attenuator=clutter_attenuator,
            sampling_distance=sampling_distance,
            is_active_sensor=is_active_sensor,
            sensor_filters=sensor_filters,
            sensor_side_effects=sensor_side_effects,
            height_of_sensor=height_of_sensor,
        )

    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:
        """Add action to sense agents at the end of the timestep.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
            all_target_agents (list[RetAgent]): A list of potentially perceivable agents,
                unused but supplied to meet template pattern.

        Returns
            list[PerceivedAgent]: An empty list.
        """
        self._add_process_actively_sensing_agents_action(sensor_agent)
        return []

    def _add_process_actively_sensing_agents_action(self, sensor_agent: RetAgent) -> None:
        """Add action to process actively sensing agents to be called at end of timestep.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
        """
        sensor_agent.model.actively_sensing_agents_manager.add_action(
            lambda agents: self._process_actively_sensing_agents(sensor_agent, agents)
        )

    def _process_actively_sensing_agents(
        self, sensor_agent: RetAgent, actively_sensing_agents: list[RetAgent]
    ) -> None:
        """Try to sense the provided actively sensing agents.

        Update the sensor agent's perceived world with any perceived agents.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection
            actively_sensing_agents (list[RetAgent]): A list of actively sensing agents.
        """
        perceived_agents = super()._run_detection(sensor_agent, actively_sensing_agents)
        sensor_agent.perceived_world.add_acquisitions(perceived_agents)

    def get_new_instance(self) -> SigIntSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            SigIntSensor: New instance of the SigInt sensor
        """
        return SigIntSensor(
            cep_range_data=self._cep_range_data,
            wavelength=self._wavelength,
            distance_thresholds=self._distance_thresholds,
            detection_timings=self._detection_timings,
            view_angle=self.view_angle,
            clutter_attenuator=self._clutter_attenuator,
            sampling_distance=self._sampling_distance,
            is_active_sensor=self.is_active_sensor,
            sensor_filters=self.sensor_filters,
            sensor_side_effects=self.sensor_side_effects,
            height_of_sensor=self.height_of_sensor,
        )


class ContrastSensorType(ABC):
    """Abstract class that defines an image contrast sensor type."""

    @abstractmethod
    def get_clear_precipitation_aerosol_transmission_factor(self) -> float:  # pragma: no cover
        """Get factor for aerosol transmission factor with Clear precipitation.

        Override in sub-class.

        Returns:
            float: The aerosol transmission factor for Clear precipitation.
        """
        pass

    @abstractmethod
    def get_water_vapour_transmission_coefficient(
        self, relative_humidity: float
    ) -> float:  # pragma: no cover
        """Get transmission coefficient for water vapour and gaseous absorption.

        Override in sub-class.

        Args:
            relative_humidity (float): The relative humidity between 0 and 1, exclusive

        Returns:
            float: The transmission coefficient for water vapour and gaseous absorption
        """
        pass


class EOContrastSensorType(ContrastSensorType):
    """A class that defines an Electro-Optical contrast sensor type."""

    def get_clear_precipitation_aerosol_transmission_factor(self) -> float:
        """Get factor for aerosol transmission factor with Clear precipitation.

        Returns:
            float: The aerosol transmission factor for Clear precipitation.
        """
        return 3.912

    def get_water_vapour_transmission_coefficient(self, relative_humidity: float) -> float:
        """Get transmission coefficient for water vapour and gaseous absorption.

        Args:
            relative_humidity (float): The relative humidity between 0 and 1, exclusive

        Returns:
            float: The transmission coefficient for water vapour and gaseous absorption
        """
        return 0.02


class IIContrastSensorType(ContrastSensorType):
    """A class that defines an Image Intensifying contrast sensor type."""

    def get_clear_precipitation_aerosol_transmission_factor(self) -> float:
        """Get factor for aerosol transmission factor with Clear precipitation.

        Returns:
            float: The aerosol transmission factor for Clear precipitation.
        """
        return 2.374

    def get_water_vapour_transmission_coefficient(self, relative_humidity: float) -> float:
        """Get transmission coefficient for water vapour and gaseous absorption.

        Args:
            relative_humidity (float): The relative humidity between 0 and 1, exclusive

        Returns:
            float: The transmission coefficient for water vapour and gaseous absorption
        """
        if relative_humidity < 0.16:
            return 0.02
        elif relative_humidity <= 0.63:
            return 0.025
        else:
            return 0.03


class SensorDetectionTimings:
    """Class that defines timings associated with sensor detections."""

    def __init__(
        self,
        detection_delay: Optional[Union[timedelta, Distribution]] = None,
    ) -> None:
        """Initialise a SensorDetectionTimings object.

        Args:
            detection_delay (Optional[Union[timedelta, Distribution]]): Time delay between when a
                detection is run, and when the result is returned. The sense time of a perceived
                agent detected using this sensor is sensed "detection_delay" time from when the
                detection is initiated. If provided as a distribution, units are interpreted as
                seconds. If None, defaults to zero seconds. Defaults to None

        Raises:
            ValueError: When supplied as a timedelta, detection delay cannot be
                negative.
        """
        if detection_delay is None:
            detection_delay = timedelta(seconds=0)

        if isinstance(detection_delay, timedelta):
            if detection_delay < timedelta(seconds=0):
                raise ValueError("Detection delay cannot be negative")

            seconds = detection_delay.total_seconds()
            detection_delay = SingleValueDistribution(mean=seconds)

        self._detection_delay = detection_delay

    def get_detection_delay(self) -> timedelta:
        """Sample from distribution to return detection delay for a single detection.

        Returns:
            timedelta: Detection delay for detection run.

        Raises:
            ValueError: Negative detection delay time
        """
        detection_seconds = self._detection_delay.sample()

        if detection_seconds < 0:
            raise ValueError("Detection delay sampled from sensor distribution is negative.")

        return timedelta(seconds=detection_seconds)


class SensorDistanceThresholds:
    """Class that defines distance thresholds for DRI sensing outcomes."""

    def __init__(
        self,
        max_detect_dist: float,
        max_recognise_dist: float,
        max_identify_dist: float,
    ) -> None:
        """Initialise SensorDistanceThresholds.

        Args:
            max_detect_dist (float): Maximum range at which an agent can be detected
                with DETECT confidence level.
            max_recognise_dist (float): Maximum range at which an agent can be detected
                with RECOGNISE confidence level.
            max_identify_dist (float): Maximum range at which an agent can be detected
                with IDENTIFY confidence level.

        Raises:
            ValueError: Detect distance thresholds cannot be equal and must be ordered
                DETECT > RECOGNISE > IDENTIFY.
        """
        if not max_identify_dist < max_recognise_dist < max_detect_dist:
            raise ValueError(
                "Maximum identify distance must be less than maximum recognise "
                + "distance, which must be less than maximum detect distance."
            )

        self.confidence_distances = {
            Confidence.DETECT: max_detect_dist,
            Confidence.RECOGNISE: max_recognise_dist,
            Confidence.IDENTIFY: max_identify_dist,
        }


class SensorClutterAttenuator:
    """Class that defines how clutter attenuates a sensor signal."""

    def __init__(
        self,
        attenuation_strength: float = 0.0,
    ) -> None:
        """Initialise a SensorClutterAttenuator.

        Args:
            attenuation_strength (float): Multiplication factor that clutter at a
                location increases perceived distance by. Defaults to 0.0, which results
                in zero attenuation.
        """
        if attenuation_strength < 0.0:
            warnings.warn(
                f"Attenuation strength provided is negative: {attenuation_strength}, "
                + "this will result in increased sensor performance when attenuated by "
                + "clutter.",
                stacklevel=2,
            )

        self.attenuation_strength = attenuation_strength


class SensorWavelength:
    """Class that defines the wavelength of a sensor attenuated by culture."""

    def __init__(
        self,
        wavelength: float,
    ) -> None:
        """Initialise SensorWavelength.

        Args:
            wavelength (float): Operating wavelength of sensor.

        Raises:
            ValueError: Wavelength cannot be negative.
        """
        if wavelength < 0:
            raise ValueError("Wavelength cannot be negative")

        self.wavelength = wavelength


class SensorSamplingDistance:
    """Class that defines the sampling distance."""

    def __init__(
        self,
        sampling_distance: Optional[float] = None,
    ) -> None:
        """Initialise SensorSamplingDistance.

        Args:
            sampling_distance (Optional[float], optional): Distance at which space
                is sampled when calculating line of sight and culture penetration.
                Defaults to None, resulting in a sampling distance of 1/2 the pixel size
                of the terrain image for line of sight calculations, and 1/2 the pixel
                size of the culture image for culture penetration calculations.
        """
        self.sampling_distance = sampling_distance


class ActivelySensingAgentsManager:
    """Manager to store and action agents which are actively sensing in a timestep."""

    def __init__(self) -> None:
        """Create a new actively sensing agents manager."""
        self._actively_sensing_agents: list[RetAgent] = []
        self._actions: list[Callable[[list[RetAgent]], None]] = []

    def add_agent(self, agent: RetAgent) -> None:
        """Add actively sensing agent if it hasn't been already.

        Args:
            agent (RetAgent): The actively sensing agent.
        """
        if agent not in self._actively_sensing_agents:
            self._actively_sensing_agents.append(agent)

    def add_action(self, action: Callable[[list[RetAgent]], None]):
        """Add action to list of actions to perform.

        Args:
            action (Callable[[list[RetAgent]], None]): The action to add
        """
        self._actions.append(action)

    def do_actions_and_clear(self) -> None:
        """Call all actions with the current actively sensing agents."""
        for action in self._actions:
            action(self._actively_sensing_agents)
        self._clear()

    def _clear(self) -> None:
        """Clear the list of actions and actively sensing agents."""
        self._actively_sensing_agents.clear()
        self._actions.clear()


class RangeToCepData:
    """Class to store range (km) to CEP (m) data."""

    def __init__(self, data: list[tuple[float, float]]):
        """Initialise a RangeToCepData object.

        Args:
            data (list[tuple[float, float]]): list of tuples, where each tuple
                contains a range (m) and CEP distance (m)

        Raises:
            ValueError: Range values or CEP value was negative
        """
        for range_m, cep_m in data:
            if range_m < 0:
                raise ValueError(f"Range value ({range_m}) cannot be negative.")
            if cep_m < 0:
                raise ValueError(f"CEP value ({cep_m}) cannot be negative for range {range_m}")

        self.data = data

    def get_range_values(self) -> list[float]:
        """Converts RangeToCepData to a list of range values.

        Returns:
            list[float]: list of range (m) values
        """
        return [v[0] for v in self.data]

    def get_cep_values(self) -> list[float]:
        """Converts RangeToCepData to a list of cep values.

        Returns:
            list[float]: list of cep (m) values
        """
        return [v[1] for v in self.data]
