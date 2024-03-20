"""Wait behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ret.behaviours.loggablebehaviour import LoggableBehaviour

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent


class WaitBehaviour(LoggableBehaviour):
    """An abstract class representing wait behaviour."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(self, waiter: RetAgent) -> None:
        """Do one time steps worth of the wait behaviour and log.

        Args:
            waiter (RetAgent): the agent doing the waiting
        """
        self.log(waiter)
        self._step(waiter)

    def _step(self, waiter: RetAgent) -> None:  # pragma: no cover
        """Do one time steps worth of the wait behaviour, override in subclasses.

        Args:
            waiter (RetAgent): the agent doing the waiting
        """
        pass
