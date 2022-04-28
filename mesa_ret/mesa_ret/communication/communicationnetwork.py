"""Communication network."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

from mesa_ret.communication.communicationreceiver import CommunicationReceiver

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Optional, Union

    from mesa_ret.agents.agent import RetAgent
    from mesa_ret.agents.agentfilter import AgentFilter
    from mesa_ret.orders.order import Order
    from mesa_ret.sensing.perceivedworld import PerceivedAgentFilter
    from mesa_ret.template import Template


class CommunicationNetworkModifier:
    """Representation of a modification to a network.

    This can either disable all communications from an agent, or can disable comms only
    to a specific source, and can last either until a specific time step or permanently.
    """

    expiry_time: Optional[datetime]
    _target: Optional[RetAgent]

    def __init__(
        self, target: Optional[RetAgent] = None, expiry_time: Optional[datetime] = None
    ) -> None:
        """Create a new CommunicationNetworkModifier.

        Args:
            target (Optional[RetAgent], optional): agent to disable comms to. Defaults
                to None.
            expiry_time (Optional[datetime], optional): time until modifier expires.
                Defaults to None.
        """
        self._target = target
        self.expiry_time = expiry_time

    def is_affected(self, agent: RetAgent) -> bool:
        """Determine if a given agent is affected by a communication modifier.

        Args:
            agent (RetAgent): agent to test.

        Returns:
            bool: Whether the agent is affected by the communication modifier.
        """
        return self._target is None or self._target == agent


class CommunicationNetwork:
    """Representation of the outgoing communication network of an agent.

    This representation can be built up either during the simulation, or after creating
    all agents in the simulation, which reduces the constraints on the order in which
    a simulation can be built.

    The communication network defines behaviour for both communicating orders and
    communicating worldviews (i.e., sensor results).
    """

    _recipients: list[RetAgent]
    _modifiers: list[CommunicationNetworkModifier]

    receiver: CommunicationReceiver

    def __init__(
        self,
        receiver: Optional[CommunicationReceiver] = None,
    ) -> None:
        """Create a new Communication Network.

        Args:
            receiver (Optional[CommunicationReceiver]): receiver for incoming comms.
                    defaults to None.
        """
        self._recipients = []
        self._modifiers = []

        if receiver is None:
            receiver = CommunicationReceiver()
        self.receiver = receiver

    def add_recipient(self, recipient: Union[RetAgent, list[RetAgent]]) -> None:
        """Add a recipient to the communication network.

        Args:
            recipient (Union[RetAgent, list[RetAgent]]): Recipient(s)
        """
        if isinstance(recipient, Iterable):
            for _recipient in recipient:
                self._add_recipient(_recipient)
        else:
            self._add_recipient(recipient)

    def _add_recipient(self, recipient: RetAgent) -> None:
        """Add a single recipient to a communication network.

        If the agent is already present, this raises a warning.

        Args:
            recipient (RetAgent): Recipient to add
        """
        if recipient in self._recipients:
            message = (
                f"{recipient.name} ({recipient.unique_id}) is already in the "
                + "communication network."
            )
            warnings.warn(message)
        else:
            self._recipients.append(recipient)

    def remove_recipient(self, recipient: Union[RetAgent, list[RetAgent]]) -> None:
        """Remove a recipient from the communication network.

        Args:
            recipient (Union[RetAgent, list[RetAgent]]): Recipient(s)
        """
        if isinstance(recipient, Iterable):
            for agent in recipient:
                self._remove_recipient(agent)
        else:
            self._remove_recipient(recipient)

    def _remove_recipient(self, recipient: RetAgent) -> None:
        """Remove a recipient from the communication network.

        If the agent is not present, raises a warning

        Args:
            recipient (RetAgent): Recipient to remove
        """
        if recipient not in self._recipients:
            message = (
                f"{recipient.name} ({recipient.unique_id}) is not in this "
                + "communication network, and therefore can't be removed."
            )
            warnings.warn(message)
        else:
            self._recipients.remove(recipient)

    def communicate_worldview_step(
        self,
        communicator: RetAgent,
        worldview_filter: Optional[PerceivedAgentFilter] = None,
        recipient_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Do one time step's worth of worldview communication behaviour.

        Args:
            communicator (RetAgent): the agent doing the communicating
            worldview_filter (PerceivedAgentFilter, optional): Optional worldview filter
            recipient_filter (AgentFilter, optional): Filter to apply to the
                communicating agent's recipients to determine appropriate recipient(s)
                of the communication
        """
        worldview_behaviour = communicator.behaviour_pool.satisfy(
            communicator.behaviour_handlers.communicate_worldview_handler,
            communicator.behaviour_handlers.communicate_worldview_type,
        )
        if worldview_behaviour:
            for recipient in self.get_recipients(recipient_filter):
                worldview_behaviour(
                    communicator=communicator,
                    recipient=recipient,
                    worldview_filter=worldview_filter,
                )
        else:
            communicator.warn_of_undefined_behaviour("Communicate Worldview")

    def communicate_orders_step(
        self,
        communicator: RetAgent,
        orders: list[Template[Order]],
        recipient_filter: Optional[AgentFilter] = None,
    ):
        """Do one time step's worth of order communication behaviour.

        Args:
            communicator (RetAgent): The agent doing the communication.
            orders (list[Template[Order]]): The order(s) to be communicated.
            recipient_filter (AgentFilter, optional): Filter to apply to the
                communicating agent's recipients to determine appropriate recipient(s)
                of the communication
        """
        orders_behaviour = communicator.behaviour_pool.satisfy(
            communicator.behaviour_handlers.communicate_orders_handler,
            communicator.behaviour_handlers.communicate_orders_type,
        )
        if orders_behaviour:
            for recipient in self.get_recipients(recipient_filter):
                orders_behaviour(communicator=communicator, recipient=recipient, orders=orders)
        else:
            communicator.warn_of_undefined_behaviour("Communicate Orders")

    def communicate_mission_message_step(
        self,
        communicator: RetAgent,
        message: str,
        recipient_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Do one time step's worth of mission message communication behaviour.

        Args:
            communicator (RetAgent): the agent doing the communicating
            message (str): The mission message to send
            recipient_filter (AgentFilter, optional): Filter to apply to the
                communicating agent's recipients to determine appropriate recipient(s)
                of the communication
        """
        mission_message_behaviour = communicator.behaviour_pool.satisfy(
            communicator.behaviour_handlers.communicate_mission_message_handler,
            communicator.behaviour_handlers.communicate_mission_message_type,
        )
        if mission_message_behaviour:
            for recipient in self.get_recipients(recipient_filter):
                mission_message_behaviour(
                    communicator=communicator,
                    recipient=recipient,
                    message=message,
                )
        else:
            communicator.warn_of_undefined_behaviour("Communicate Mission Message")

    def get_recipients(self, recipient_filter: Optional[AgentFilter] = None) -> list[RetAgent]:
        """Get list of recipients for a given order.

        This accounts for any temporary or permanent modifications to communications
        network.

        Args:
            recipient_filter (AgentFilter, optional): Filter to apply to the list of
                recipients

        Returns:
            list[RetAgent]: List of agents to communicate to.
        """
        unfiltered_recipients = [
            r for r in self._recipients if not any([m.is_affected(r) for m in self._modifiers])
        ]

        if recipient_filter:
            return recipient_filter.run(unfiltered_recipients)
        return unfiltered_recipients

    def add_modifier(self, modifier: CommunicationNetworkModifier) -> None:
        """Add a new communication modifier.

        Args:
            modifier (CommunicationNetworkModifier): New communication network modifier.
        """
        self._modifiers.append(modifier)

    def remove_expired_modifiers(self, current_time: datetime) -> None:
        """Remove all expired modifiers from the communication network.

        Args:
            current_time (datetime): The current time
        """
        self._modifiers = [
            m for m in self._modifiers if (m.expiry_time is None) or (m.expiry_time > current_time)
        ]
