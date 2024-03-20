"""Communication task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ret.orders.order import Task

if TYPE_CHECKING:
    from typing import Optional

    from ret.agents.agent import RetAgent
    from ret.agents.agentfilter import AgentFilter
    from ret.orders.order import Order
    from ret.sensing.perceivedworld import PerceivedAgentFilter
    from ret.template import Template


class CommunicateTask(Task, ABC):
    """Task for an agent to communicate to recipients in it's network."""

    @abstractmethod
    def _do_task_step(self, doer: RetAgent) -> None:  # pragma: no cover
        """Make doer complete a single time step's worth of communicating.

        Args:
            doer (RetAgent): the agent doing the task
        """
        pass

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Return true of the task is complete and false otherwise.

        Args:
            doer (RetAgent): Agent doing the communication

        Returns:
            bool: True - Communication only ever takes one time step
        """
        return True


class CommunicateOrderTask(CommunicateTask):
    """Task for communicating an order."""

    _orders: list[Template[Order]]
    _recipient_filter: Optional[AgentFilter]

    def __init__(
        self,
        orders: list[Template[Order]],
        log: bool = True,
        recipient_filter: Optional[AgentFilter] = None,
    ):
        """Create a new communicate orders task.

        Args:
            orders (list[Template[Order]]): Orders to be communicated.
            log (bool): whether to log or not. Defaults to True.
            recipient_filter (optional[AgentFilter]): Filter to apply to agents who
                could receive the communication. Defaults to None.
        """
        super().__init__(log)
        self._orders = orders
        self._recipient_filter = recipient_filter

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Communicate Order Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one step's worth of communicating orders.

        Args:
            doer (RetAgent): Agent communicating
        """
        doer.communicate_orders_step(self._orders, self._recipient_filter)

    def _get_log_message(self) -> str:
        """Get the log message.

        Returns:
            str: The log message
        """
        order_list = map(lambda o: str(o), self._orders)
        return f"Orders: ({', '.join(order_list)})"

    def get_new_instance(self) -> CommunicateOrderTask:
        """Return a new instance of a functionally identical task.

        Returns:
            CommunicateOrderTask: New instance of the task
        """
        return CommunicateOrderTask(
            orders=self._orders, log=self._log, recipient_filter=self._recipient_filter
        )


class CommunicateWorldviewTask(CommunicateTask):
    """Task for communicating a world-view."""

    _worldview_filter: Optional[PerceivedAgentFilter]

    def __init__(
        self,
        worldview_filter: Optional[PerceivedAgentFilter] = None,
        log: bool = True,
        recipient_filter: Optional[AgentFilter] = None,
    ):
        """Create a new communicate worldview task.

        Args:
            worldview_filter (Optional[PerceivedAgentFilter]): Filter for agents to
                include in communicated worldview. Defaults to None.
            log (bool): whether to log or not. Defaults to True.
            recipient_filter (Optional[AgentFilter]): Filter to apply to agents who
                could receive the communication. Defaults to None.
        """
        super().__init__(log)
        self._worldview_filter = worldview_filter
        self._recipient_filter = recipient_filter

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Communicate World View Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one time steps worth of communicating world-view.

        Args:
            doer (RetAgent): Agent communicating
        """
        doer.communicate_worldview_step(self._worldview_filter, self._recipient_filter)

    def get_new_instance(self) -> CommunicateWorldviewTask:
        """Return a new instance of a functionally identical task.

        Returns:
            CommunicateWorldviewTask: New instance of the task
        """
        return CommunicateWorldviewTask(
            worldview_filter=self._worldview_filter,
            log=self._log,
            recipient_filter=self._recipient_filter,
        )


class CommunicateMissionMessageTask(CommunicateTask):
    """Task for communicating a mission message."""

    _message: str

    def __init__(
        self,
        message: str,
        log: bool = True,
        recipient_filter: Optional[AgentFilter] = None,
    ):
        """Create a new communicate mission message task.

        Args:
            message (str): Mission message to send.
            log (bool): whether to log or not. Defaults to True.
            recipient_filter (Optional[AgentFilter]): Filter to apply to agents who
                could receive the communication. Defaults to None.
        """
        super().__init__(log)
        self._message = message
        self._recipient_filter = recipient_filter

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Communicate Mission Message Task"

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one time steps worth of communicating mission message.

        Args:
            doer (RetAgent): Agent communicating
        """
        doer.communicate_mission_message_step(self._message, self._recipient_filter)

    def get_new_instance(self) -> CommunicateMissionMessageTask:
        """Return a new instance of a functionally identical task.

        Returns:
            CommunicateMissionMessageTask: New instance of the task
        """
        return CommunicateMissionMessageTask(
            message=self._message,
            log=self._log,
            recipient_filter=self._recipient_filter,
        )
