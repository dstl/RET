"""Communication behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from mesa_ret.behaviours.loggablebehaviour import LoggableBehaviour

if TYPE_CHECKING:
    from typing import Any, Optional, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.orders.order import Order
    from mesa_ret.sensing.perceivedworld import PerceivedAgent, PerceivedAgentFilter
    from mesa_ret.template import Template


class CommunicateOrdersBehaviour(LoggableBehaviour):
    """Class defining behaviour for sending orders."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        orders: Union[Template[Order], list[Template[Order]]],
    ):
        """Do one time step's worth of communicating orders and log.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            orders (Union[Template[Order], list[Template[Order]]]): Order(s) sent from
                communicator to recipient
        """
        self.log(communicator, recipient=recipient, communicated_orders=orders)
        self._step(communicator, recipient, orders)

    def _step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        orders: Union[Template[Order], list[Template[Order]]],
    ):
        """Do one time step's worth of communicating orders.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            orders (Union[Template[Order], list[Template[Order]]]): Order(s) sent from
                communicator to recipient
        """
        payload = {"orders": orders}
        recipient.communication_network.receiver.receive(recipient, payload)

    def _get_log_message(self, **kwargs: Any) -> str:
        """Get the log message.

        This log message includes all given orders in the following format:
            [Task1]: [Trigger1]; [Task2]: [Trigger2];...
            Compound tasks are written as "Compound Task: (Task1, Task2, ...)"

        Args:
            **kwargs(Any): Key word arguments.
                communicated_orders(Union[Template[Order], list[Template[Order]]]): The
                    list of orders being communicated.
                recipient(Optional[RetAgent]): The recipient of the order communication.

        Returns:
            str: The log message
        """
        orders: Optional[Union[Template[Order], list[Template[Order]]]] = kwargs.get(
            "communicated_orders", None
        )
        if orders is not None:
            if isinstance(orders, Iterable):
                orders_string = "; ".join(
                    [o.get_order_string() for o in orders]  # type: ignore
                ).strip()
            else:
                orders_string = orders.get_order_string()  # type: ignore
        else:
            orders_string = "NONE FOUND"

        recipient: RetAgent | None = kwargs.get("recipient", None)
        if recipient is not None:
            recipient_string = f"recipient: {recipient.name} (ID: {recipient.unique_id})"
        else:
            recipient_string = "NO RECIPIENT GIVEN"

        additional_message = super()._get_log_message()

        return f"{recipient_string}; orders:[{orders_string}]" f" {additional_message}".strip()


class CommunicateWorldviewBehaviour(LoggableBehaviour):
    """Class for defining behaviour for sending world-views."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        worldview_filter: Optional[PerceivedAgentFilter] = None,
    ):
        """Do one time step's worth of world-view communication and log.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            worldview_filter (PerceivedAgentFilter, optional): Filter to apply to the
                communicator's world-view. Defaults to None.
        """
        self.log(communicator)
        self._step(communicator, recipient, worldview_filter)

    def _step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        worldview_filter: Optional[PerceivedAgentFilter] = None,
    ):
        """Do one time step's worth of world-view communication.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            worldview_filter (PerceivedAgentFilter, optional): Filter to apply to the
                communicator's world-view. Defaults to None.
        """
        new_views = self.get_new_views(communicator, worldview_filter)
        payload = {"worldview": new_views}
        recipient.communication_network.receiver.receive(recipient, payload)

    def get_new_views(
        self, communicator: RetAgent, worldview_filter: Optional[PerceivedAgentFilter]
    ) -> list[PerceivedAgent]:
        """Get new views from the communicating agent.

        Args:
            communicator (RetAgent): Agent doing the communication
            worldview_filter (Optional[PerceivedAgentFilter]): World-view filter

        Returns:
            list[PerceivedAgent]: List of new perceived views
        """
        return communicator.perceived_world.get_perceived_agents(filter=worldview_filter)


class CommunicateMissionMessageBehaviour(LoggableBehaviour):
    """Class for defining behaviour for sending mission messages."""

    def __init__(self, log: bool = True) -> None:
        """Create behaviour.

        Args:
            log (bool): If True logging occurs for this behaviour, if False it does not.
                Defaults to True.
        """
        super().__init__(log)

    def step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        message: str,
    ):
        """Do one time step's worth of mission message communication and log.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            message (str): Mission message to send
        """
        self.log(communicator)
        self._step(communicator, recipient, message)

    def _step(
        self,
        communicator: RetAgent,
        recipient: RetAgent,
        message: str,
    ):
        """Do one time step's worth of mission message communication.

        Args:
            communicator (RetAgent): Agent doing the communication
            recipient (RetAgent): Agent receiving the communication
            message (str): Mission message to send
        """
        payload = {"missionMessage": message}
        recipient.communication_network.receiver.receive(recipient, payload)
