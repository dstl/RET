"""Attack tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ret.orders.order import Task

if TYPE_CHECKING:
    from typing import Optional, Union

    from ret.agents.agent import RetAgent
    from ret.types import Coordinate2dOr3d


class AttackTask(Task, ABC):
    """Task that causes an agent to attack."""

    @abstractmethod
    def _do_task_step(self, doer: RetAgent) -> None:  # pragma: no cover
        """Do one time steps worth of the attack task, override in subclasses.

        Args:
            doer (RetAgent): the agent that will do the task
        """
        pass

    @abstractmethod
    def _is_task_complete(self, doer: RetAgent) -> bool:  # pragma: no cover
        """Check whether the attack task is complete, override in subclasses.

        Args:
            doer (RetAgent): The agent completing the task

        Returns:
            bool: true when the task is completed, false otherwise
        """
        pass

    @abstractmethod
    def get_target(
        self,
    ) -> Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d]]]:  # pragma: no cover
        """Return the target for the attack, override in subclasses.

        Returns:
            Optional[Coordinate2dOr3d]: Target to attack. If this is not defined by the
                task, then return None
        """
        pass
