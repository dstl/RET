"""Sense behaviour."""

from __future__ import annotations

from collections import deque
from datetime import timedelta
from math import floor
from typing import TYPE_CHECKING

from mesa_ret.behaviours.loggablebehaviour import LoggableBehaviour
from numpy import histogram

if TYPE_CHECKING:
    from typing import Optional, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.types import Coordinate2dOr3d


class SenseSchedule:
    """Wrapper class used to schedule sense events independently from the model timestep."""

    def __init__(self) -> None:
        """Create a new empty Sense Schedule."""
        self._steps: deque[bool] = deque()

    def get_step(self) -> bool:
        """Determine if the sensors are used at the next time step.

        Raises:
            ValueError: The sense schedule is complete

        Returns:
            bool: Whether the sensors are used at the next time step or not.
        """
        if len(self._steps) == 0:
            raise ValueError("This sense schedule is already complete")
        return self._steps.popleft()

    def is_complete(self) -> bool:
        """Determine whether the sense schedule is complete.

        Returns:
            bool: Whether or not the sense schedule is complete
        """
        return len(self._steps) == 0


class TimeBasedSenseSchedule(SenseSchedule):
    """Sense Schedule based on time between sensor use."""

    def __init__(
        self,
        duration: timedelta,
        time_before_first_sense: timedelta,
        time_between_senses: timedelta,
        time_step: timedelta,
    ):
        """Create a new TimeBasedSenseSchedule.

        Args:
            duration (timedelta): The duration of the task, over which the agent
                periodically checks its sensors
            time_before_first_sense (timedelta): Time before the first sensor is used
            time_between_senses (timedelta): Time between each sensor use
            time_step (timedelta): Model time_step
        """
        super().__init__()
        n_sense_times = floor((duration - time_before_first_sense) / time_between_senses) + 1
        sense_times = [
            (x * time_between_senses) + time_before_first_sense for x in range(0, n_sense_times)
        ]
        n_time_steps = floor(duration / time_step) + 1
        bins = [(x * time_step) for x in range(0, n_time_steps + 1)]
        hist = histogram(sense_times, bins=bins)[0]
        self._steps = deque((hist > 0).tolist())


class SenseBehaviour(LoggableBehaviour):
    """An abstract class representing sense behaviour."""

    def __init__(
        self,
        time_before_first_sense: timedelta,
        time_between_senses: timedelta,
        log: bool = True,
    ) -> None:
        """Create behaviour.

        Args:
            time_before_first_sense (timedelta): Time before the first sensor is used
            time_between_senses (timedelta): Time between each sensor use
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.

        Raises:
            ValueError: Error in sense properties
        """
        super().__init__(log)
        if time_between_senses < timedelta(0):
            msg = (
                "Time between senses must be non-negative. "
                + f"{time_between_senses.total_seconds()}s provided."
            )
            raise ValueError(msg)
        self.time_between_senses = time_between_senses

        if time_before_first_sense < timedelta(0):
            msg = (
                "Time before first sense must be non-negative. "
                + f"{time_before_first_sense.total_seconds()}s provided."
            )
            raise ValueError(msg)
        self.time_before_first_sense = time_before_first_sense

    def step(
        self, senser: RetAgent, direction: Optional[Union[float, Coordinate2dOr3d, str]] = None
    ) -> None:
        """Do one time steps worth of the sense behaviour and log.

        Args:
            senser (RetAgent): the agent doing the sensing
            direction (Optional[Union[float, Coordinate2dOr3d, str]]): Optional direction to sense
                in, can be input as float: degrees clockwise from y-axis, Coordinate2dor3d: a
                position used to calculate heading from sensor agent, str: the name of an Area in
                the space, a random point within the area will be chosen to look towards.
        """
        self.log(senser)
        self._step(senser, direction=direction)

    def _step(
        self, senser: RetAgent, direction: Optional[Union[float, Coordinate2dOr3d, str]] = None
    ) -> None:
        """Do one time steps worth of the sense behaviour, override in subclasses.

        Args:
            senser (RetAgent): the agent doing the sensing
            direction (Optional[Union[float, Coordinate2dOr3d, str]]): Optional direction to sense
                in, can be input as float: degrees clockwise from y-axis, Coordinate2dor3d: a
                position used to calculate heading from sensor agent, str: the name of an Area in
                the space, a random point within the area will be chosen to look towards.
        """
        for s in senser._sensors:
            s.run_detection(sensor_agent=senser, sense_direction=direction)

    def get_sense_schedule(self, senser: RetAgent, duration: timedelta) -> SenseSchedule:
        """Get a sense schedule for the senser.

        Args:
            senser (RetAgent): Sensing Agent.
            duration (timedelta): The duration of the sense task, over which the agent
                periodically checks its sensors

        Returns:
            SenseSchedule: The sense schedule for the senser.
        """
        return TimeBasedSenseSchedule(
            duration=duration,
            time_between_senses=self.time_between_senses,
            time_before_first_sense=self.time_before_first_sense,
            time_step=senser.model.time_step,
        )
