"""Loggable behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mesa_ret.behaviours import Behaviour

if TYPE_CHECKING:
    from typing import Any

    from mesa_ret.agents.agent import RetAgent


class LoggableBehaviour(Behaviour):
    """Class defining behaviour for logging behaviours."""

    def __init__(self, log: bool = True) -> None:
        """Create LoggableBehaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True
        """
        super().__init__()
        self._log = log

    def log(self, doer: RetAgent, **kwargs) -> None:
        """Log behaviour.

        Args:
            doer (RetAgent): The agent doing the behaviour
            kwargs: Keyword arguments to the function
        """
        if self._log:
            doer.model.logger.log_behaviour(doer, self.name, self._get_log_message(**kwargs))

    def _get_log_message(self, **kwargs: Any) -> str:
        """Get the log message, override as necessary in subclasses.

        Args:
            **kwargs(Any): Key word arguments.
                None used by this override.

        Returns:
            str: The log message
        """
        return ""
