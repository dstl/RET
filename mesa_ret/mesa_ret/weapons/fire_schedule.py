"""Firing schedules."""
from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from numpy import histogram

if TYPE_CHECKING:
    from datetime import timedelta


class FireSchedule:
    """Wrapper around a list of fire events to simplify data management."""

    def __init__(self) -> None:
        """Create a new empty Fire Schedule."""
        self._steps: list[int] = []

    def get_round(self) -> int:
        """Get number of rounds to fire at the next time step.

        Raises:
            ValueError: The firing schedule is complete

        Returns:
            int: Number of rounds to fire
        """
        if len(self._steps) == 0:
            raise ValueError("This firing schedule is already complete")
        return self._steps.pop(0)

    def is_complete(self) -> bool:
        """Determine whether the firing schedule is complete.

        Returns:
            bool: Whether or not the firing schedule is complete
        """
        return len(self._steps) == 0

    @staticmethod
    def validate_rounds(rounds: int) -> None:
        """Validate rounds.

        Args:
            rounds (int): Number of rounds to validate

        Raises:
            ValueError: Negative number of rounds provided
        """
        if rounds <= 0:
            msg = f"Number of rounds must be greater than zero. {rounds} provided."
            raise ValueError(msg)


class TimeBasedFireSchedule(FireSchedule):
    """Fire schedule based on time between rounds and initial start-up time."""

    def __init__(
        self,
        rounds: int,
        time_before_first_round: timedelta,
        time_between_rounds: timedelta,
        time_step: timedelta,
    ):
        """Create a new TimeBasedFireSchedule.

        Args:
            rounds (int): Number of rounds to fire for
            time_before_first_round (timedelta): Time before the first round is fired
            time_between_rounds (timedelta): Time between each round after the first
                rounds fired
            time_step (timedelta): Model time_step
        """
        super().__init__()
        self.validate_rounds(rounds)

        firing_times = [
            (x * time_between_rounds) + time_before_first_round for x in range(0, rounds)
        ]

        n_time_steps = ceil(firing_times[-1] / time_step)
        bins = [(x * time_step) for x in range(0, n_time_steps + 1)]

        if bins[-1] == firing_times[-1]:
            bins.append(firing_times[-1] + time_step)

        hist = histogram(firing_times, bins=bins)[0]
        self._steps = hist.astype("int").tolist()  # type: ignore
