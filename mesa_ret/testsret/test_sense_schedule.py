"""Tests for sense schedules."""

from datetime import timedelta
from unittest import TestCase
from collections import deque

from mesa_ret.behaviours.sense import SenseSchedule, TimeBasedSenseSchedule
from parameterized import parameterized


class TestSenseSchedule(TestCase):
    """Tests for the SenseSchedule base class."""

    class ConcreteSenseSchedule(SenseSchedule):
        """Concrete implementation of Sense Schedule."""

        def __init__(self):
            """Create a basic Sense Schedule consisting of sensing 3 time steps.

            Within the three time steps, sensing is distributed as follows:
            * True in the first step
            * False in the second step
            * True in the final step
            """
            self._steps = deque([True, False, True])

    def test_sense_schedule_get_step(self):
        """Test basic sense schedule get_step behaviour."""
        sense_schedule = self.ConcreteSenseSchedule()

        assert sense_schedule.get_step()
        assert not sense_schedule.get_step()
        assert sense_schedule.get_step()

        with self.assertRaises(ValueError) as e:
            sense_schedule.get_step()
        self.assertEqual("This sense schedule is already complete", str(e.exception))


class TestTimeBasedSenseSchedule(TestCase):
    """Tests for TimeBasedSenseSchedule SenseSchedule."""

    @parameterized.expand(  # type: ignore
        [
            [60, [True, True, True, True, True, True]],
            [90, [True, True, False, True, True, False]],
            [120, [True, False, True, False, True, False]],
        ]
    )
    def test_low_frequency_sampling(self, sec_between_senses: int, expected_schedule: list[bool]):
        """Test generation of sense schedule with multiple time steps per sense.

        Args:
            sec_between_senses (int): Time between senses, in seconds
            expected_schedule (list[bool]): Calculated sense schedule
        """
        schedule = TimeBasedSenseSchedule(
            duration=timedelta(seconds=300),
            time_between_senses=timedelta(seconds=sec_between_senses),
            time_before_first_sense=timedelta(seconds=0),
            time_step=timedelta(seconds=60),
        )

        steps: list[bool] = []
        while not schedule.is_complete():
            steps.append(schedule.get_step())

        assert len(steps) == 6
        assert steps == expected_schedule

    @parameterized.expand([[1, [True, True, True, True, True, True]]])
    def test_high_frequency_sampling(self, sec_between_senses: int, expected_schedule: list[bool]):
        """Test generation of sense schedule with multiple senses pet time step.

        Args:
            sec_between_senses (int): Time between senses, in seconds
            expected_schedule (list[bool]): Calculated sense schedule
        """
        schedule = TimeBasedSenseSchedule(
            duration=timedelta(seconds=300),
            time_between_senses=timedelta(seconds=sec_between_senses),
            time_before_first_sense=timedelta(seconds=0),
            time_step=timedelta(seconds=60),
        )

        steps: list[bool] = []
        while not schedule.is_complete():
            steps.append(schedule.get_step())

        assert len(steps) == 6
        assert steps == expected_schedule

    @parameterized.expand(
        [
            [30, [True, True, True, True, True, True]],
            [60, [False, True, True, True, True, True]],
            [120, [False, False, True, True, True, True]],
        ]
    )  # type: ignore
    def test_first_sense_delay(self, delay: int, expected_schedule: list[int]):
        """Test specification of first sense delay.

        Args:
            delay (int): Time delay
            expected_schedule (list[bool]): Expected sense schedule
        """
        schedule = TimeBasedSenseSchedule(
            duration=timedelta(seconds=300),
            time_between_senses=timedelta(seconds=30),
            time_before_first_sense=timedelta(seconds=delay),
            time_step=timedelta(seconds=60),
        )

        steps: list[bool] = []
        while not schedule.is_complete():
            steps.append(schedule.get_step())

        assert steps == expected_schedule
