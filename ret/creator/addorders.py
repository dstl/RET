"""Utility Method for giving orders to multiple Agents."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar, Union

from ret.creator.agents import create_list_of_lists

if TYPE_CHECKING:
    from ret.agents.agent import RetAgent
    from ret.orders.background_order import BackgroundOrder
    from ret.orders.order import Order
    from ret.template import Template

T = TypeVar("T")
SingleOrList = Union[T, list[T]]


def add_orders_to_agents(
    agents: SingleOrList[RetAgent],
    orders: list[SingleOrList[Template[Order]]],
) -> None:
    """Assign orders to multiple agents.

    Can use either one list of orders given to all agents, or a list of
    lists, giving different orders to each agent. In this case the list
    of lists must be the same length as the list of agents.

    Args:
        agents (SingleOrList[RetAgent]): Single or list of agent(s) to whom the orders
            are assigned.
        orders (SingleOrList[list[Template[Order]]]): Single set (list[Template[Order]])
            or list of sets (list[list[Template[Order]]]) of template orders to be given
            to agent(s)
    """
    if isinstance(agents, Iterable):
        _orders = create_list_of_lists(len(agents), orders)  # type: ignore
        for a in range(0, len(agents)):  # type: ignore
            add_orders_to_single_agent(agents[a], _orders[a])  # type: ignore

    else:
        add_orders_to_single_agent(agents, orders)  # type: ignore


def add_orders_to_single_agent(
    agent: RetAgent,
    orders: list[Template[Order]],
) -> None:
    """Assign list of orders to single agent.

    Args:
        agent (RetAgent): The agent receiving orders.
        orders (list[Template[Order]]): The orders to be assigned.
    """
    agent.add_orders(orders)


def add_background_orders_to_agents(
    agents: SingleOrList[RetAgent],
    background_orders: list[SingleOrList[Template[BackgroundOrder]]],
) -> None:
    """Assign background orders to multiple agents.

    Can use either one list of background orders given to all agents, or a list of
    lists, giving different background orders to each agent. In this case the list
    of lists must be the same length as the list of agents.

    Args:
        agents (SingleOrList[RetAgent]): Single or list of agent(s) to whom the orders
            are assigned.
        background_orders (SingleOrList[list[Template[BackgroundOrder]]]): Single set
            (list[Template[BackgroundOrder]]) or list of sets
            (list[list[Template[BackgroundOrder]]]) of template background orders to be given
            to agent(s)
    """
    if isinstance(agents, Iterable):
        _background_orders = create_list_of_lists(len(agents), background_orders)  # type: ignore
        for a in range(0, len(agents)):  # type: ignore
            add_background_orders_to_single_agent(agents[a], _background_orders[a])  # type: ignore

    else:
        add_background_orders_to_single_agent(agents, background_orders)  # type: ignore


def add_background_orders_to_single_agent(
    agent: RetAgent,
    background_orders: list[Template[BackgroundOrder]],
) -> None:
    """Assign list of background orders to single agent.

    Args:
        agent (RetAgent): The agent receiving orders.
        background_orders (list[Template[Order]]): The background orders to be assigned.
    """
    agent.add_background_orders(background_orders)
