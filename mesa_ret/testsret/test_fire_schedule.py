"""Tests for firing schedules."""

from datetime import timedelta
from unittest import TestCase

from mesa_ret.testing.mocks import RoundPerStepFireSchedule
from mesa_ret.weapons.fire_schedule import FireSchedule, TimeBasedFireSchedule
from parameterized import parameterized


class TestFireSchedule(TestCase):
    """Tests for the FireSchedule base class."""

    class ConcreteFireSchedule(FireSchedule):
        """Concrete implementation of Fire Schedule."""

        def __init__(self):
            """Create a basic Fire Schedule consisting of firing 3 time steps.

            Within the three time steps, firing is distributed as follows:
            * 4 rounds in the first step
            * 2 rounds in the second step
            * 1 round in the final step
            """
            self._steps = [4, 2, 1]

    def test_fire_schedule_get_rounds(self):
        """Test basic fire schedule get_rounds behaviour."""
        fire_schedule = self.ConcreteFireSchedule()

        assert fire_schedule.get_round() == 4
        assert fire_schedule.get_round() == 2
        assert fire_schedule.get_round() == 1

        with self.assertRaises(ValueError) as e:
            fire_schedule.get_round()
        self.assertEqual("This firing schedule is already complete", str(e.exception))


class TestTimeBasedFireSchedule(TestCase):
    """Tests for TimeBasedFireSchedule FireSchedule."""

    @parameterized.expand([[0], [-1]])  # type: ignore
    def test_invalid_number_of_rounds(self, rounds: int):
        """Test initialisation of RoundPerStepFireSchedule with invalid rounds.

        Args:
            rounds (int): Number of f
        """
        msg = f"Number of rounds must be greater than zero. {rounds} provided."

        with self.assertRaises(ValueError) as e:
            TimeBasedFireSchedule(
                rounds=rounds,
                time_before_first_round=timedelta(seconds=0),
                time_between_rounds=timedelta(seconds=10),
                time_step=timedelta(seconds=20),
            )
        self.assertEqual(
            msg,
            str(e.exception),
        )

    @parameterized.expand(  # type: ignore
        [
            [60, [1, 1, 1, 1, 1]],
            [90, [1, 1, 0, 1, 1, 0, 1]],
            [120, [1, 0, 1, 0, 1, 0, 1, 0, 1]],
        ]
    )
    def test_low_frequency_sampling(self, sec_between_rounds: int, expected_schedule: list[int]):
        """Test generation of fire schedule with multiple rounds per time step.

        Args:
            sec_between_rounds (int): Time between rounds, in seconds
            expected_schedule (list[int]): Calculated firing schedule
        """
        schedule = TimeBasedFireSchedule(
            rounds=5,
            time_between_rounds=timedelta(seconds=sec_between_rounds),
            time_before_first_round=timedelta(seconds=0),
            time_step=timedelta(seconds=60),
        )

        steps: list[int] = []
        while not schedule.is_complete():
            steps.append(schedule.get_round())

        assert sum(steps) == 5
        assert steps == expected_schedule

    @parameterized.expand([[0, [20]], [1, [20]], [5, [12, 8]], [10, [6, 6, 6, 2]]])
    def test_high_frequency_sampling(self, sec_between_rounds: int, expected_schedule: list[int]):
        """Test generation of fire schedule with multiple time steps per round.

        Args:
            sec_between_rounds (int): Time between rounds, in seconds
            expected_schedule (list[int]): Calculated firing schedule
        """
        schedule = TimeBasedFireSchedule(
            rounds=20,
            time_between_rounds=timedelta(seconds=sec_between_rounds),
            time_before_first_round=timedelta(seconds=0),
            time_step=timedelta(seconds=60),
        )

        steps: list[int] = []
        while not schedule.is_complete():
            steps.append(schedule.get_round())

        assert sum(steps) == 20
        assert steps == expected_schedule

    @parameterized.expand([[30, [5]], [60, [0, 5]], [120, [0, 0, 5]]])  # type: ignore
    def test_first_round_delay(self, delay: int, expected_schedule: list[int]):
        """Test specification of first round fire delay.

        Args:
            delay (int): Time delay
            expected_schedule (list[int]): Expected calculated firing schedule
        """
        schedule = TimeBasedFireSchedule(
            rounds=5,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_round=timedelta(seconds=delay),
            time_step=timedelta(seconds=60),
        )

        steps: list[int] = []
        while not schedule.is_complete():
            steps.append(schedule.get_round())

        assert steps == expected_schedule


class TestRoundPerStepFireSchedule(TestCase):
    """Tests for the rounds per step fire schedule."""

    @parameterized.expand([[0], [-1]])  # type: ignore
    def test_invalid_number_of_rounds(self, rounds: int):
        """Test initialisation of RoundPerStepFireSchedule with invalid rounds.

        Args:
            rounds (int): Number of rounds
        """
        msg = f"Number of rounds must be greater than zero. {rounds} provided."

        with self.assertRaises(ValueError) as e:
            RoundPerStepFireSchedule(rounds)
        self.assertEqual(msg, str(e.exception))

    @parameterized.expand([[1], [2], [10]])  # type: ignore
    def test_rounds_per_step_schedule(self, rounds: int):
        """Test initialisation of RoundPerStepFireSchedule.

        Args:
            rounds (int): Rounds per step
        """
        schedule = RoundPerStepFireSchedule(rounds=rounds)

        for _ in range(0, rounds):
            assert not schedule.is_complete()
            round = schedule.get_round()
            assert round == 1

        assert schedule.is_complete()
