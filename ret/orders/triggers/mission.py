"""Triggers relating the the status of missions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent

from ret.orders.order import Trigger


class MissionMessageTrigger(Trigger):
    """Trigger for where an agent has received a specific mission message."""

    def __init__(
        self, message: str, sticky: bool = False, log: bool = True, invert: bool = False
    ) -> None:
        """Create a new MissionMessageTrigger.

        Args:
            message (str): Mission message to trigger on.
            sticky (bool): Whether or not trigger is permanent. Defaults to False.
            log (bool): whether to log or not. Defaults to True.
            invert (bool): if true, trigger will invert the boolean logic output.
                Defaults to False.
        """
        super().__init__(log=log, sticky=sticky, invert=invert)
        self._message = message

    def _check_condition(self, checker: RetAgent) -> bool:
        """Return true if the checker has received the specific mission message.

        Args:
            checker (RetAgent): Agent doing the checking

        Returns:
            bool: Whether or not the mission message has been received.
        """
        return self._message in checker.mission_messages

    def __str__(self):
        """Return string representation of trigger.

        Returns:
            str: String representation of trigger.
        """
        return "Mission Message Trigger"

    def get_new_instance(self) -> MissionMessageTrigger:
        """Return a new instance of a functionally identical trigger.

        Returns:
            MissionMessageTrigger: New instance of the trigger
        """
        return MissionMessageTrigger(
            message=self._message, sticky=self._sticky, log=self._log, invert=self._invert
        )

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        return "Message: " + self._message
