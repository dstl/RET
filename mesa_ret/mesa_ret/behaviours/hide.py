"""Hide behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.behaviours.loggablebehaviour import LoggableBehaviour

if TYPE_CHECKING:
    from mesa_ret.agents.agent import RetAgent


class HideBehaviour(LoggableBehaviour):
    """An abstract class representing hide behaviour."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(self, hider: RetAgent) -> None:
        """Do one time steps worth of the hide behaviour and log.

        Args:
            hider (RetAgent): the agent hiding
        """
        self.log(hider)
        self._step(hider)

    def _step(self, hider: RetAgent) -> None:
        """Do one time steps worth of the hide behaviour, override in subclasses.

        The agent will be restricted from active sensing and firing during background orders.

        Args:
            hider (RetAgent): the agent doing the hiding
        """
        hider.hiding = True
