"""Group agent."""
from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

from mesa.time import BaseScheduler, RandomActivation
from ret.agents.agent import RetAgent
from ret.agents.agenttype import AgentType
from ret.communication.communicationreceiver import GroupAgentCommunicationReceiver

if TYPE_CHECKING:
    from typing import Optional, Union

    from ret.agents.affiliation import Affiliation
    from ret.behaviours import Behaviour
    from ret.behaviours.behaviourpool import ListAdder
    from ret.behaviours.communicate import CommunicateOrdersBehaviour
    from ret.model import RetModel
    from ret.orders.background_order import BackgroundOrder
    from ret.orders.order import Order
    from ret.template import Template
    from ret.types import Coordinate2dOr3d
    from ret.space.pathfinder import GridAStarPathfinder


class GroupAgent(RetAgent):
    """A class which groups agents in a simulation to disseminate orders.

    If a path to only one icon is passed in, the default icon will be used
    for the other. If no icons are passed in, but agents are, the most common
    icon from subordinates will be used. If no icons or agents are passed in,
    the default icons will be used.
    """

    def __init__(
        self,
        model: RetModel,
        name: str,
        affiliation: Affiliation,
        critical_dimension: Optional[float] = None,
        communicate_orders_behaviour: Optional[CommunicateOrdersBehaviour] = None,
        icon_path: Optional[str] = None,
        killed_icon_path: Optional[str] = None,
        orders: Optional[list[Template[Order]]] = None,
        background_orders: Optional[list[Template[BackgroundOrder]]] = None,
        agent_orders: Optional[list[Template[Order]]] = None,
        agent_background_orders: Optional[list[Template[BackgroundOrder]]] = None,
        agents: Optional[list[RetAgent]] = None,
        behaviour_adder: Optional[type[ListAdder]] = None,
        pathfinder: Optional[GridAStarPathfinder] = None,
    ) -> None:
        """Create a group agent.

        Args:
            model (RetModel): the model the agent will be placed in
            name (str): the name of the agent
            affiliation (Affiliation): the affiliation of the agent
            critical_dimension (float): The length (m) of the longest dimension of the
                agent
            communicate_orders_behaviour (Optional[CommunicateOrdersBehaviour]): the
                agents behaviour when communicating orders. Defaults to None.
            icon_path (Optional[str], optional): path to the agent's icon
            killed_icon_path (Optional[str], optional): path to the agent's icon when
                killed
            orders (list[Template[Order]]): the group-agent's initial set of orders
            background_orders (Optional[list[Template[BackgroundOrder]]]): templates of the
                background orders the agent will act to complete automatically
            agent_orders (list[Template[Order]]): Orders to be passed to all sub-agents
            agent_background_orders (list[Template[BackgroundOrder]]): Background orders to be
                passed to all sub-agents
            agents (list[RetAgent]): the agents initial subordinate agents
            behaviour_adder (Optional[type[ListAdder]]): Behaviour adding methodology to
                be supplied to the RetAgent class. Defaults to None.
            pathfinder (Optional[GridAStarPathfinder]): Pathfinder to generate paths for movement
                tasks. Defaults to None.
        """
        behaviours: list[Behaviour] = []
        if communicate_orders_behaviour is not None:
            behaviours.append(communicate_orders_behaviour)

        super().__init__(
            pos=None,
            model=model,
            name=name,
            affiliation=affiliation,
            critical_dimension=critical_dimension if critical_dimension else 10.0,
            agent_type=AgentType.GROUP,
            behaviours=behaviours,
            icon_path=icon_path,
            killed_icon_path=killed_icon_path,
            orders=orders,
            background_orders=background_orders,
            communication_receiver=GroupAgentCommunicationReceiver(),
            behaviour_adder=behaviour_adder,
            pathfinder=pathfinder,
        )

        self.create_scheduler()
        self._in_space = False

        if agent_orders is None:
            agent_orders = []

        if agent_background_orders is None:
            agent_background_orders = []

        self.agents: list[RetAgent] = []
        if agents is not None:
            for agent in agents:
                agent.add_orders(agent_orders)
            for agent in agents:
                agent.add_background_orders(agent_background_orders)
            self.add_agents(agents)
            if icon_path is None and killed_icon_path is None:
                self._get_icons()

    def _add_to_model(self, pos: Coordinate2dOr3d) -> None:
        """Add the agent to the schedule.

        Overridden from base class so agent does not get added to the space.

        Args:
            pos (Coordinate2dOr3d): position of the agent, not used
        """
        self.model: RetModel
        self.model.schedule.add(self)

    def create_scheduler(self) -> None:
        """Create scheduler based on the scheduler used in the model.

        If scheduler used cannot be determined, uses Base Scheduler.
        """
        if isinstance(self.model.schedule, RandomActivation):
            self.schedule = RandomActivation(self.model)
        elif type(self.model.schedule) is BaseScheduler:
            self.schedule = BaseScheduler(self.model)
        else:
            warnings.warn(
                f"Unable to determine scheduler type for Group Agent {self.name}. "
                + "Using BaseScheduler",
                stacklevel=2,
            )
            self.schedule = BaseScheduler(self.model)

    def add_agents(self, new_agents: Union[RetAgent, list[RetAgent]]) -> None:
        """Add agents to the group.

        Args:
            new_agents (Union[RetAgent, list[RetAgent]]): a list of agents to be added
                to the group
        """
        if isinstance(new_agents, Iterable):
            for agent in new_agents:
                self._add_agent(agent)
        else:
            self._add_agent(new_agents)

        self.check_killed()
        self.find_pos()

    def _add_agent(self, agent: RetAgent) -> None:
        """Add a single agent to the group if agent does not already belong to a group.

        Args:
            agent (RetAgent): an agent to be added to the group
        """
        if agent.in_group is False:
            agent.in_group = True
            self.model.schedule.remove(agent)
            self.schedule.add(agent)
            self.agents.append(agent)
            self.communication_network.add_recipient(agent)

    def remove_agents(self, agents_to_remove: Union[RetAgent, list[RetAgent]]) -> None:
        """Remove all agents in the given list from the group.

        Args:
            agents_to_remove (Union[RetAgent, list[RetAgent]]): a list of agents to be
                removed from the group
        """
        if isinstance(agents_to_remove, Iterable):
            for agent in agents_to_remove:
                self._remove_agent(agent)
        else:
            self._remove_agent(agents_to_remove)

        self.find_pos()
        self.check_killed()

    def _remove_agent(self, agent: RetAgent) -> None:
        """Remove a single agent the group.

        Args:
            agent (RetAgent): an agent to be removed from the group.
        """
        if agent in self.agents:
            agent.in_group = False
            self.schedule.remove(agent)
            self.agents.remove(agent)
            self.communication_network.remove_recipient(agent)
            self.model.schedule.add(agent)

    def step(self) -> None:
        """Step the group agent for one time step.

        Check if GroupAgent should be marked as killed, then execute GroupAgent step on
        model scheduler, then execute subordinates step on internal scheduler.
        """
        self.check_killed()
        super().step()
        self.schedule.step()
        self.check_killed()
        self.find_pos()

    def check_killed(self) -> None:
        """Check if the group agent should be marked as killed.

        GroupAgent will be marked killed if all subordinates are killed, or if there are
        no subordinates in group.
        """
        if all([a.killed is True for a in self.agents]) is True:
            if self._killed is False:
                self.kill()
        else:
            if self._killed is True:
                self.revive()

    def find_pos(self):
        """Determine the position of the GroupAgent.

        Uses the average of the positions of subordinate agents.
        """
        subordinates_pos = []
        for agent in self.agents:
            if agent.killed is False:
                agent_pos = self.model.space.get_coordinate_in_correct_dimension(agent.pos)
                subordinates_pos.append(agent_pos)

        if len(subordinates_pos) > 0:
            pos = [sum(coord) / len(subordinates_pos) for coord in zip(*subordinates_pos)]
            self.pos = tuple(pos)
            self._move_agent()

    def revive(self) -> None:
        """Revive the agent and update icon."""
        self._killed = False
        self.set_icon()

    def _get_icons(self):
        """Set the icon as the most common subordinates icon.

        If there is no most common, uses first of equal most common.
        """
        icon_list = []
        killed_icon_list = []
        for agent in self.agents:
            icon_list.append(agent._icon)
            killed_icon_list.append(agent._killed_icon)

        count = Counter(icon_list)
        self._icon: str = count.most_common(1)[0][0]

        count = Counter(killed_icon_list)
        self._killed_icon: str = count.most_common(1)[0][0]

        self.set_icon()

    def _move_agent(self):
        """Move agent in space."""
        if self._in_space is False:
            self.model.space.place_agent(self, self.pos)
            self._in_space = True
        else:
            self.model.space.move_agent(self, self.pos)
