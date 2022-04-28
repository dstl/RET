"""Communication receivers."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable

from mesa_ret.orders.order import GroupAgentOrder
from mesa_ret.sensing.perceivedworld import And, RemoveDuplicates, RemoveDuplicatesAtLocation

if TYPE_CHECKING:
    from typing import Any, Optional, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.agents.groupagent import GroupAgent
    from mesa_ret.orders.order import Order
    from mesa_ret.sensing.perceivedworld import PerceivedAgent, PerceivedAgentFilter
    from mesa_ret.template import Template


class CommunicationHandler(ABC):
    """Abstract representation of a communication handler.

    Concrete instances of communication handlers can be implemented to interpret any
    content provided in a message payload.
    """

    @abstractmethod
    def receive(self, agent: RetAgent, payload: Any) -> None:  # pragma: no cover
        """Receive communication.

        Note that the type definition of 'payload' should be specified more exactly in
        each CommunicationHandler which extends this abstract representation.

        Args:
            agent (RetAgent): The agent receiving the information.
            payload (Any): The information received.
        """
        pass


class WorldviewHandler(CommunicationHandler):
    """Worldview communication handler.

    This handler interprets a list of new worldviews and adds them to the receiving
    agent's view of the world.

    The combine_worldviews method is intended to be overridden, should a more complex
    mechanism for combining the worldviews from the receiving agent and new information
    be desired.
    """

    _combiner: PerceivedAgentFilter

    def __init__(self):
        """Create a new WorldviewHandler."""
        self._combiner = self.get_combined_worldviews_filter()

    def receive(self, receiver: RetAgent, payload: list[PerceivedAgent]) -> None:
        """Receive new worldview information.

        Args:
            receiver (RetAgent): Agent receiving the worldview.
            payload (list[PerceivedAgent]): New perceptions.
        """
        combined_views = self.combine_worldviews(
            receiver.perceived_world.get_perceived_agents(),
            payload,
        )

        receiver.perceived_world.reset_worldview()
        receiver.perceived_world.add_acquisitions(combined_views)

    def combine_worldviews(
        self, existing_knowledge: list[PerceivedAgent], new_views: list[PerceivedAgent]
    ) -> list[PerceivedAgent]:
        """Combine world-views from two sets of perceived agents.

        This filters out any duplicate agents at the same location, removing lower
        confidence agents.

        Args:
            existing_knowledge (list[PerceivedAgent]): List of perceived agents
            new_views (list[PerceivedAgent]): List of perceived agents

        Returns:
            list[PerceivedAgent]: Combined list of perceived agents
        """
        return self._combiner.run(existing_knowledge + new_views)

    def get_combined_worldviews_filter(self) -> PerceivedAgentFilter:
        """Return filter to be used for combining world-views.

        This can be overridden or extended in child classes.

        Returns:
            PerceivedAgentFilter: Perceived agent filter for combining views
        """
        return And([RemoveDuplicates(), RemoveDuplicatesAtLocation()])


class OrdersHandler(CommunicationHandler):
    """Orders communication handler.

    This handler adds the new set of orders to the receiving agent's orders.

    """

    def receive(
        self,
        receiver: RetAgent,
        payload: Union[Template[Order], list[Template[Order]]],
    ) -> None:
        """Receive orders payload.

        Args:
            receiver (RetAgent): Agent receiving the orders.
            payload (Union[Template[Order], list[Template[Order]]]): Single order, or
                list of orders.
        """
        receiver.add_orders(payload)


class MissionMessageHandler(CommunicationHandler):
    """Communication handler for mission completion.

    This handler provides a mechanism for notifying an agent that a mission has been
    completed.
    """

    def receive(self, receiver: RetAgent, payload: str) -> None:
        """Receive a mission completion notification.

        Args:
            receiver (RetAgent): Agent receiving the notification
            payload (str): Description of completed mission
        """
        receiver.mission_messages.append(payload)


class CommunicationReceiver:
    """Communication receiver."""

    def __init__(self, handlers: Optional[list[tuple[str, CommunicationHandler]]] = None) -> None:
        """Create a new communication receiver.

        Args:
            handlers (Optional[list[tuple[str, CommunicationHandlers]]]): Handlers. If None then a
                default handler will be populated for worldview, orders and missionMessage,
                otherwise, only the handlers entered will be used. If defining a custom handler for
                any of worldview, orders or missionMessage then remember to add the other handlers
                to the handler list. Defaults to None.
        """
        if handlers is None:
            self.handlers = [
                ("worldview", WorldviewHandler()),
                ("orders", OrdersHandler()),
                ("missionMessage", MissionMessageHandler()),
            ]
        else:
            self.handlers = handlers

    def receive(self, receiver: RetAgent, payload: dict[str, Any]):
        """Receive a general form of message.

        Delegates each appropriate message content to a suitable handler, if present.

        Args:
            receiver (RetAgent): The agent receiving the message.
            payload (dict[str, Any]): The content of the message.
        """
        for handler in self.handlers:
            payload_component = handler[0]
            if payload_component in payload:
                handler[1].receive(receiver, payload[payload_component])

        handled_components = [h[0] for h in self.handlers]
        unhandled_messages = [k for k in payload.keys() if k not in handled_components]

        for message in unhandled_messages:
            msg = (
                f"Message component {message} is not handled by {receiver.name}'s "
                + f"({receiver.unique_id}) inbound message receiver."
            )
            warnings.warn(msg)


class GroupAgentHandler(CommunicationHandler):
    """CommunicationHandler for GroupAgents.

    The GroupAgentHandler automatically forwards any messages onwards to all contained
    agents. The GroupAgentHandler can be used for a variety of payloads, but must
    forward each payload independently.
    """

    def __init__(self, message_type: str):
        """Create a new GroupAgentHandler.

        Args:
            message_type (str): The payload aspect that the handler is responsible for.
        """
        self.message_type = message_type

    def receive(self, agent: GroupAgent, payload: Any) -> None:
        """Receive a new communication and distribute to all agents in the group agent.

        Args:
            agent (GroupAgent): Agent receiving the message
            payload (Any): The content of the payload
        """
        agent_payload = {self.message_type: payload}

        for _agent in agent.agents:
            _agent.communication_network.receiver.receive(_agent, agent_payload)


class GroupAgentOrdersHandler(CommunicationHandler):
    """Group agent orders communication handler.

    This handler disseminates orders to the group agents subordinate agents with
    the exception of GroupAgentOrders, which are passed directly to the group agent.
    """

    def receive(
        self,
        receiver: GroupAgent,
        payload: Union[Template[Order], list[Template[Order]]],
    ) -> None:
        """Receive orders payload.

        Args:
            receiver (GroupAgent): Agent receiving the orders.
            payload (Union[Template[Order], list[Template[Order]]]): Single order, or
                list of orders.
        """
        agents_payload: dict[str, list[Template[Order]]] = {"orders": []}

        group_agent_orders: list[Template[Order]] = []

        def handle_order(order: Template[Order]):
            if isinstance(order, GroupAgentOrder):
                group_agent_orders.append(order)
            else:
                agents_payload["orders"].append(order)

        if isinstance(payload, Iterable):
            for order in payload:
                handle_order(order)
        else:
            handle_order(payload)

        for _agent in receiver.agents:
            _agent.communication_network.receiver.receive(_agent, agents_payload)

        receiver.add_orders(group_agent_orders)


class GroupAgentCommunicationReceiver(CommunicationReceiver):
    """Communication Receiver for a group of agents.

    On receipt of either orders or world-view, all content is immediately forwarded
    to all agents contained within the GroupAgent with the exception of orders of type
    GroupAgentOrders which are passed directly to the group agent.

    Assumptions:
        -   Only GroupAgent or sub-classes of GroupAgent may use this Communication
            Receiver
    """

    def __init__(self):
        """Create a new GroupAgentCommunicationReceiver.

        The GroupAgentCommunicationReceiver passes all orders, worldviews and mission
        statuses on to the agents that it stores with the exception of orders of type
        GroupAgentOrders which are passed directly to the group agent.
        """
        handlers = [
            ("orders", GroupAgentOrdersHandler()),
            ("worldview", GroupAgentHandler("worldview")),
            ("missionMessage", GroupAgentHandler("missionMessage")),
        ]
        super().__init__(handlers)
