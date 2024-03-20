"""Tasks for tracked projectiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ret.orders.order import CompoundTask
from ret.orders.tasks.move import MoveTask, MoveToTargetTask
from ret.orders.tasks.fire import FireTask

if TYPE_CHECKING:
    from ret.types import Coordinate2dOr3d
    from typing import Optional


class DetonateTask(FireTask):
    """Task to fire at own position."""

    def __init__(self, log: bool = True) -> None:
        """Create a Detonate Task.

        Args:
            log (bool): Whether to log or not. Defaults to True.
        """
        super().__init__(log=log)

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Detonate Task"

    def get_target(self) -> Optional[Coordinate2dOr3d]:
        """Get the target to fire at.

        Returns:
            Coordinate2dOr3d: Target to fire at.
        """
        return self.firer.pos

    def get_new_instance(self) -> DetonateTask:
        """Return a new instance of a functionally identical task.

        Returns:
            DetonateTask: New instance of the task
        """
        return DetonateTask(log=self._log)


class DetonateOnImpactTask(CompoundTask):
    """Task to move to the target location and detonate."""

    def __init__(self, target_location: Coordinate2dOr3d, log: bool = True) -> None:
        """Create a DetonateOnImpact Task.

        Args:
            target_location (Coordinate2dor3d): The location to move to and detonate.
            log (bool): Whether to log or not. Defaults to True.
        """
        self.target_location = (target_location[0], target_location[1])
        _move_task = MoveTask(destination=self.target_location, tolerance=0.1, log=log)
        _fire_task = DetonateTask(log=log)
        tasks = [_move_task, _fire_task]

        super().__init__(tasks, log)  # type: ignore

    def get_new_instance(self) -> DetonateOnImpactTask:
        """Return a new instance of a functionally identical task.

        Returns:
            DetonateOnImpactTask: New instance of the task
        """
        return DetonateOnImpactTask(target_location=self.target_location, log=self._log)


class GuidedDetonateOnImpactTask(CompoundTask):
    """Task to move to the target location and detonate."""

    def __init__(
        self,
        target_location: Coordinate2dOr3d,
        target: int,
        max_weapon_steps: int,
        log: bool = True,
    ) -> None:
        """Create a DetonateOnImpact Task.

        Args:
            target_location (Coordinate2dor3d): The location to move to and detonate.
            target (int): The unique ID of the agent to be guided to.
            log (bool): Whether to log or not. Defaults to True.
            max_weapon_steps (int): The maximum number of steps the weapon can move for
                before reaching max range
        """
        self.target_location = (target_location[0], target_location[1])
        self.target = target
        self.max_weapon_steps = max_weapon_steps
        _move_task = MoveToTargetTask(
            destination=self.target_location,
            tolerance=0.01,
            target=self.target,
            log=log,
            max_weapon_steps=max_weapon_steps,
        )
        _fire_task = DetonateTask(log=log)
        tasks = [_move_task, _fire_task]

        super().__init__(tasks, log)  # type: ignore

    def get_new_instance(self) -> GuidedDetonateOnImpactTask:
        """Return a new instance of a functionally identical task.

        Returns:
            DetonateOnImpactTask: New instance of the task
        """
        return GuidedDetonateOnImpactTask(
            target_location=self.target_location,
            target=self.target,
            log=self._log,
            max_weapon_steps=self.max_weapon_steps,
        )
