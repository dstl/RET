"""Base agent type definition."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from mesa.agent import Agent
from ret.agents.affiliation import Affiliation
from ret.agents.agenttype import AgentType
from ret.behaviours.behaviourpool import BehaviourHandlers, BehaviourPool, NoEqualityAdder
from ret.behaviours.sense import SenseSchedule
from ret.communication.communicationnetwork import CommunicationNetwork
from ret.orders.background_order import BackgroundOrder
from ret.orders.order import Order, TaskLogStatus
from ret.orders.tasks.sense import SenseTask
from ret.sensing.agentcasualtystate import AgentCasualtyState
from ret.sensing.perceivedworld import PerceivedWorld
from ret.sensing.sensor import Sensor
from ret.space.clutter.countermeasure import Countermeasure
from ret.space.feature import Area

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from typing import Optional, Union

    from ret.agents.agentfilter import AgentFilter
    from ret.behaviours import Behaviour
    from ret.behaviours.behaviourpool import ListAdder
    from ret.behaviours.sense import SenseBehaviour
    from ret.communication.communicationreceiver import CommunicationReceiver
    from ret.model import RetModel
    from ret.sensing.perceivedworld import PerceivedAgentFilter
    from ret.sensing.sensor import ArcOfRegard
    from ret.space.pathfinder import GridAStarPathfinder
    from ret.template import Template
    from ret.types import Coordinate, Coordinate2dOr3d
    from ret.weapons.weapon import Weapon


ICON_DIR = ICON_DIR = Path(__file__).parent.joinpath("icons/generic")


class RetAgent(Agent):
    """An abstract agent class.

    The RetAgent can be used to represent various agents within a model space, or it
    can be extended to provide additional control, such as limiting specific behaviours
    or making some behaviours mandatory.

    For examples of agents, see those already implemented.

    Each agent is initiated with a series of behaviours which control how the agent
    performs a task. The task tells the agent what to do, and the behaviour covers how
    to do it. If an agent does not have some behaviours defined, it will be unable to
    perform corresponding tasks.
    """

    icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("genericagent_friendly.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("genericagent_hostile.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("genericagent_neutral.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("genericagent_unknown.svg"),
    }

    killed_icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("killed/genericagent_friendly_killed.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("killed/genericagent_hostile_killed.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("killed/genericagent_neutral_killed.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("killed/genericagent_unknown_killed.svg"),
    }

    _icon: str
    _killed_icon: str

    def __init__(
        self,
        model: RetModel,
        pos: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d], Area]],
        name: str,
        affiliation: Affiliation,
        critical_dimension: float,
        reflectivity: Optional[float] = None,
        temperature: Optional[float] = None,
        temperature_std_dev: float = 0.0,
        agent_type: AgentType = AgentType.GENERIC,
        icon_path: Optional[str] = None,
        killed_icon_path: Optional[str] = None,
        orders: Optional[list[Template[Order]]] = None,
        behaviours: Optional[list[Behaviour]] = None,
        behaviour_adder: Optional[type[ListAdder]] = None,
        sensors: Optional[list[Template[Sensor]]] = None,
        communication_receiver: Optional[CommunicationReceiver] = None,
        refresh_technique: Optional[PerceivedAgentFilter] = None,
        countermeasures: Optional[list[Template[Countermeasure]]] = None,
        background_orders: Optional[list[Template[BackgroundOrder]]] = None,
        arc_of_regard: Optional[ArcOfRegard] = None,
        weapons: Optional[list[Weapon]] = None,
        pathfinder: Optional[GridAStarPathfinder] = None,
    ) -> None:
        """Create a RetAgent.

        Args:
            model (RetModel): the model the agent will be placed in
            pos (Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d], Area]]): one
                of:
                    the initial position of the agent
                    a list of possible initial positions
                    an area for the agent to be placed in
                    None
            name (str): the name of the agent
            affiliation (Affiliation): the affiliation of the agent
            critical_dimension (float): The length (m) of the longest dimension of the
                agent
            reflectivity (float): The reflectivity of the agent, greater than 0 and less
                than or equal to 1.
            temperature (float): The temperature of the agent in degrees C
            temperature_std_dev (float): The standard deviation of the agent temperature in
                degrees C, default is 0
            agent_type (AgentType): the agents type
            icon_path (Optional[str]): path to the agent's icon
            killed_icon_path (Optional[str]): path to the agent's killed icon
            orders (list[Template[Order]]): templates of the agents initial set of
                orders
            behaviours (Optional[list[Behaviour]]): Behaviours to initialise the agent
                with. Defaults to no behaviours.
            behaviour_adder (Optional[type[ListAdder]]): The type of behaviour adder
                methodology to use, where determining if new behaviours should be
                included as well as custom behaviours.
                ListAdder types can be used as follows:
                    - NoEqualityAdder - This allows duplicates of any particular
                        behaviour type to be defined, unless it is equal to an existing
                        behaviour. Where duplicate behaviours are not wanted, this can
                        be avoided by implementing __eq__ methods for the particular
                        behaviour type.
                    - NoTypeEqualityAdder - New behaviours will be added unless there is
                         already a user-defined behaviour of the same type.
                    - NoSubtypeEqualityAdder - New behaviours will be added unless there
                        is already a user-defined behaviour that is the same type, or a
                        sub-type, of the default behaviour.
                    - TwoWayNoSubTypeEqualityAdder - New behaviours will be added unless
                        there is already a user-defined behaviour that is the same type,
                        or a sub-type, or if the new behaviour is a sub-type of any of
                        the existing behaviours.
                    - NeverAdder - New behaviours are never added
                    - AlwaysAdder - New behaviours are always added
                If no ListAdder is specified, the NoEqualityAdder will be used.
                Defaults to None.
            sensors (list[Template[Sensor]]): sensor templates defining the
                sensors that will belong to the agent
            communication_receiver (CommunicationReceiver, optional): the agent's
                communication receiver
            refresh_technique (PerceivedAgentFilter, optional): Methodology for
                refreshing the perceived world
            countermeasures (list[Template[Countermeasure]], optional): templates of the
                countermeasures the agent has available
            background_orders (Optional[list[Template[BackgroundOrder]]]): templates of the
                background orders the agent will act to complete automatically
            arc_of_regard (Optional[ArcOfRegard]): A dictionary of sectors and relative probability
                ratios that determine how likely a sensor is to "look" in the direction of each
                sector relative to the sense direction. Sectors are defined as tuples of degree
                pairs, (A,B), representing the start (A) and end angle of a sector (B), where 0
                degrees is the sense direction. Sectors are compiled going clockwise from A to B.
                The relative probability ratios will be normalised. Any areas not contained in a
                sector will have a probability of 0. If no arc of regard is input then the sensor
                will sense directly in the sense direction.
            weapons (Optional[list[Weapon]]): List of weapons the agent is armed with. If None, the
                agent will be assumed to have no weapons. Defaults to None.
            pathfinder (Optional[GridAStarPathfinder]): Pathfinder to generate paths for movement
                tasks. Defaults to None.
        """
        super().__init__(model.next_id(), model)
        selected_pos = self._select_pos(pos)
        self._add_to_model(selected_pos)  # type: ignore

        self.name = name
        self.affiliation = affiliation
        self.agent_type = agent_type

        self._killed = False
        self.in_group = False
        self.hiding = False

        self._orders = Order.generate_object_list(orders)
        self.behaviour_handlers = BehaviourHandlers()

        self.active_order: Optional[Order] = None

        if not behaviours:
            behaviours = []

        if behaviour_adder is None:
            adder: ListAdder = NoEqualityAdder()
        else:
            adder = behaviour_adder()

        if weapons is None:
            self.weapons: list[Weapon] = []
        else:
            self.weapons = [w.get_new_instance() for w in weapons]

        self.behaviour_pool = BehaviourPool(self, self.random, adder, behaviours)

        self._background_orders = BackgroundOrder.generate_object_list(background_orders)
        self._sensors = Sensor.generate_object_list(sensors)
        self._countermeasures = Countermeasure.generate_object_list(countermeasures)

        self.communication_network = CommunicationNetwork(
            receiver=communication_receiver,
        )

        self.perceived_world = PerceivedWorld(model, refresh_technique)
        self.perceived_world.add_agent(self)

        self.set_icon_paths(icon_path, killed_icon_path)

        self.mission_messages: list[str] = []

        self._existing_sense_behaviour: Optional[SenseBehaviour] = None

        self.critical_dimension = critical_dimension

        if reflectivity is not None and (reflectivity <= 0 or reflectivity > 1):
            raise ValueError(
                f"Agent reflectivity ({reflectivity}) must be "
                + "greater than 0 and less than or equal to 1."
            )

        self._reflectivity = reflectivity
        self._temperature = temperature
        self.temperature_std_dev = temperature_std_dev

        self.arc_of_regard = self._normalise_arc_of_regard(arc_of_regard)

        if self.arc_of_regard is not None:
            self._check_arc_of_regard_for_overlapping_sectors(self.arc_of_regard)

        if self.model.logger.log_agents:
            self.model.logger.log_agent(self)

        self.pathfinder = pathfinder

        self._statuses: list[Status] = []

    @property
    def reflectivity(self) -> float:
        """Get reflectivity.

        Returns:
            float: The reflectivity of the agent
        """
        if self._reflectivity is None:
            raise AgentConstructionError(
                f"Reflectivity must be defined for {self.name} for this model to be valid"
            )
        else:
            return self._reflectivity

    @property
    def temperature(self) -> float:
        """Get temperature.

        Returns:
            float: The temperature of the agent
        """
        if self._temperature is None:
            raise AgentConstructionError(
                f"Temperature must be defined for {self.name} for this model to be valid"
            )
        else:
            return self._temperature

    @property
    def killed(self) -> bool:
        """Get killed status.

        Returns:
            bool: Whether or not the agent is killed
        """
        return self._killed

    @property
    def casualty_state(self) -> AgentCasualtyState:
        """Get agent casualty state.

        Returns:
            AgentCasualtyState: The casualty state of the agent
        """
        if self._killed is True:
            return AgentCasualtyState.KILLED
        else:
            return AgentCasualtyState.ALIVE

    def _add_to_model(self, pos: Coordinate2dOr3d) -> None:
        """Add the agent to the space and schedule.

        Args:
            pos (Coordinate2dOr3d): position of the agent
        """
        self.model: RetModel
        self.model.space.place_agent(self, pos)  # type: ignore
        self.model.schedule.add(self)

    def step(self) -> None:  # noqa: C901
        """Complete one time step of activity.

        The agent checks all sensors and adds valid perceived agents to its perceived
        world. The agent checks for any new orders that have been activated and then
        executes the highest priority one.

        If the agent has already been killed, it will exhibit no behaviour
        """
        self.hiding = False
        self.clear_old_statuses()

        if self.killed:
            return

        self.communication_network.remove_expired_modifiers(self.model.get_time())

        if self._background_orders:
            for background_order in self._background_orders:
                if isinstance(background_order._task, SenseTask):
                    if background_order.is_condition_met(self):
                        background_order.execute_step(self)

        for s in self._sensors:
            perceived_agents = s.get_results(self)
            self.perceived_world.add_acquisitions(perceived_agents)

        if self.active_order:
            if self.active_order.is_complete(self):
                if self.active_order.is_persistent():
                    order_index = self._orders.index(self.active_order)
                    self._orders[
                        order_index
                    ] = self.active_order.get_new_instance_sticky_status_maintained()
                else:
                    self._orders.remove(self.active_order)
                self.active_order = None

        self.check_for_new_active_order()

        if self.active_order:
            self.active_order.execute_step(self)

        if self._background_orders:
            for background_order in self._background_orders:
                if isinstance(background_order._task, SenseTask) is False:
                    if background_order.is_condition_met(self):
                        background_order.execute_step(self)

    def check_for_new_active_order(self) -> None:
        """Check if any new (and higher priority) orders have become active."""
        triggered_orders: list[Order] = [
            order for order in self._orders if order.is_condition_met(self)
        ]

        if len(triggered_orders) > 0:
            highest_priority: int = max(map(lambda o: o.priority, triggered_orders))

            if (self.active_order is None) or (self.active_order.priority < highest_priority):
                highest_priority_orders = [
                    order for order in triggered_orders if order.priority == highest_priority
                ]
                if self.active_order is not None:
                    self.active_order._task._log_task(self, TaskLogStatus.INTERRUPTED)
                self.active_order = highest_priority_orders[0]

    def add_orders(self, orders: Union[Template[Order], list[Template[Order]]]) -> None:
        """Add order(s) to the agents orders.

        Args:
            orders (list[Template[Order]]): Single order, or list of orders
        """
        if isinstance(orders, Iterable):
            self._orders.extend([o.get_new_instance() for o in orders])
        else:
            self._orders.append(orders.get_new_instance())

    def add_background_orders(
        self, background_orders: Union[Template[BackgroundOrder], list[Template[BackgroundOrder]]]
    ) -> None:
        """Add background order(s) to the agents background orders.

        Args:
            background_orders (list[Template[BackgroundOrder]]): Single background order, or list of
                background orders
        """
        if isinstance(background_orders, Iterable):
            self._background_orders.extend([o.get_new_instance() for o in background_orders])
        else:
            self._background_orders.append(background_orders.get_new_instance())

    def kill(
        self,
        shot_id: Optional[int] = None,
        killer: Optional[RetAgent] = None,
        weapon_name: Optional[str] = None,
    ) -> None:
        """Kill the agent and update icon.

        Args:
            shot_id (Optional[int]): Unique ID for the shot. Defaults to None.
            killer (Optional[RetAgent]): The agent who performed the kill. Defaults to None.
            weapon_name (Optional[str]): The weapon that fired the shot. Defaults to None.
        """
        self._killed = True
        for c in (c for c in self._countermeasures if c.deployed and not c.persist_beyond_deployer):
            c.kill(self.model.get_time())
        self.set_icon()
        self.model.logger.log_death(
            target=self, killer=killer, shot_id=shot_id, weapon_name=weapon_name
        )

    def move_step(self, destination: Coordinate) -> None:
        """Do one time steps worth of any move behaviour.

        Args:
            destination (Coordinate): the target destination
        """
        behaviour = self.behaviour_pool.satisfy(
            self.behaviour_handlers.move_handler, self.behaviour_handlers.move_type
        )
        if behaviour:
            behaviour(mover=self, destination=destination)
        else:
            self.warn_of_undefined_behaviour("Move")

    def wait_step(self) -> None:
        """Do one time steps worth of any wait behaviour."""
        behaviour = self.behaviour_pool.satisfy(
            self.behaviour_handlers.wait_handler, self.behaviour_handlers.wait_type
        )
        if behaviour:
            behaviour(waiter=self)
        else:
            self.warn_of_undefined_behaviour("Wait")

    def sense_step(self, sense_direction: Optional[Union[float, Coordinate2dOr3d, str]]) -> None:
        """Do one time steps worth of any sense behaviour.

        Args:
            sense_direction (Optional[Union[float, Coordinate2dor3d, str]]): Optional direction to
                sense in, can be input as float: degrees clockwise from y-axis, Coordinate2dor3d: a
                position used to calculate heading from sensor agent, str: the name of an Area in
                the space, a random point within the area will be chosen to look towards.
        """
        behaviour = self.behaviour_pool.satisfy(
            self.behaviour_handlers.sense_handler, self.behaviour_handlers.sense_type
        )

        if behaviour is not None:
            behaviour(senser=self, direction=sense_direction)
        else:
            self.warn_of_undefined_behaviour("Sense")

    def hide_step(self) -> None:
        """Do one time steps worth of any hide behaviour."""
        behaviour = self.behaviour_pool.satisfy(
            self.behaviour_handlers.hide_handler, self.behaviour_handlers.hide_type
        )

        if behaviour:
            behaviour(hider=self)
        else:
            self.warn_of_undefined_behaviour("Hide")

    def communicate_worldview_step(
        self,
        worldview_filter: Optional[PerceivedAgentFilter] = None,
        recipient_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Do one time steps worth of a worldview communication.

        Args:
            worldview_filter (PerceivedAgentFilter): Filter to apply to communicator's
                worldview
            recipient_filter (AgentFilter, optional): Filter to apply to the
                communicating agent's recipients to determine appropriate recipient(s)
                of the communication
        """
        self.communication_network.communicate_worldview_step(
            self, worldview_filter, recipient_filter
        )

    def communicate_orders_step(
        self,
        orders: list[Template[Order]],
        recipient_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Do one time steps worth of communicating orders.

        Args:
            orders (list[Template[Order]]): Orders to communicate
            recipient_filter (AgentFilter, optional): Filter to apply to the
                communicating agent's recipients to determine appropriate recipient(s)
                of the communication
        """
        self.communication_network.communicate_orders_step(self, orders, recipient_filter)

    def communicate_mission_message_step(
        self,
        message: str,
        recipient_filter: Optional[AgentFilter] = None,
    ) -> None:
        """Do one time steps worth of a mission message communication.

        Args:
            message (str): Mission message to communicate.
            recipient_filter (AgentFilter, optional): Filter to apply to the
                communicating agent's recipients to determine appropriate recipient(s)
                of the communication
        """
        self.communication_network.communicate_mission_message_step(self, message, recipient_filter)

    def disable_communication_step(self) -> None:
        """Do one time steps worth of disabling communication."""
        behaviour = self.behaviour_pool.satisfy(
            self.behaviour_handlers.disable_communication_handler,
            self.behaviour_handlers.disable_communication_type,
        )

        if behaviour:
            behaviour(disabler=self)
        else:
            self.warn_of_undefined_behaviour("Disable Communication")

    def fire_step(
        self,
        rounds: int,
        weapon: Weapon,
        target: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d]]] = None,
    ) -> None:
        """Do one time steps worth of fire behaviour.

        Where the target is undefined, the agents fire behaviour will determine the
        appropriate target(s).

        Args:
            rounds (int): The number of rounds to fire
            weapon (Weapon): The weapon to fire with
            target (Optional[Coordinate2dOr3d]): Target of the fire. Defaults to None.
        """
        behaviour = self.behaviour_pool.satisfy(
            self.behaviour_handlers.fire_handler, self.behaviour_handlers.fire_type
        )
        if behaviour is not None:
            behaviour(firer=self, rounds=rounds, weapon=weapon, location=target)
        else:
            self.warn_of_undefined_behaviour("Fire")

    def clear_fire_behaviour(self):
        """Clear any existing fire behaviour."""
        self._existing_fire_behaviour = None

    def clear_sense_behaviour(self) -> None:
        """Clear any existing sense behaviour."""
        self._existing_sense_behaviour = None

    def deploy_countermeasure_step(self):
        """Do one time step's worth of deploying a countermeasure."""
        countermeasure = next((c for c in self._countermeasures if not c.deployed), None)

        behaviour = self.behaviour_pool.satisfy(
            self.behaviour_handlers.deploy_countermeasure_handler,
            self.behaviour_handlers.deploy_countermeasure_type,
        )

        if behaviour:
            if countermeasure is not None:
                behaviour(deployer=self, countermeasure=countermeasure)
        else:
            self.warn_of_undefined_behaviour("Deploy Countermeasure")

    def get_sense_schedule(self, duration: timedelta) -> SenseSchedule:
        """Get the sensing schedule for the agent.

        If the agent does not have a defined sensing schedule, this will return an
        empty sense schedule.

        Args:
            duration (timedelta): The duration of the task, over which the agent
                periodically checks its sensors

        Returns:
            SenseSchedule: An array of booleans, indicating whether to sense
                or not in future time steps.
        """
        behaviour: SenseBehaviour = self.behaviour_pool.choose_behaviour(
            self.behaviour_handlers.sense_handler, self.behaviour_handlers.sense_type
        )  # type: ignore

        self._existing_sense_behaviour = behaviour
        if behaviour is not None:
            sense_schedule: SenseSchedule = behaviour.get_sense_schedule(self, duration)
            return sense_schedule
        else:
            self.warn_of_undefined_behaviour("Sense")
            return SenseSchedule()

    def warn_of_undefined_behaviour(self, behaviour_name: str) -> None:
        """Raise a warning for undefined behaviour.

        Args:
            behaviour_name (str): the name of the behaviour to raise a warning about
        """
        warnings.warn(
            (
                f"{behaviour_name} behaviour requested for agent {self.name} "
                f"({self.unique_id}) at time {self.model.get_time()} but it is not "
                f"defined."
            ),
            stacklevel=2,
        )

    def set_icon_paths(
        self,
        icon_filepath: Optional[str] = None,
        killed_icon_filepath: Optional[str] = None,
    ) -> None:
        """Set an icon for the agent.

        Args:
            icon_filepath (Optional[str]): The path to the icon .svg image.
            killed_icon_filepath (Optional[str]): The path to the icon .svg image.


        Raises:
            TypeError: Raises an exception if either file given is not a .svg file.
        """
        if icon_filepath is None:
            self._icon_filepath = str(self.icon_dict[self.affiliation])
            with open(self._icon_filepath) as i:
                self._icon = i.read()
        else:
            if icon_filepath.lower().endswith(".svg"):
                self._icon_filepath = str(icon_filepath)
                with open(self._icon_filepath) as i:
                    self._icon = i.read()
            else:
                raise TypeError("Icon file type incompatible. Please use SVG file type.")

        if killed_icon_filepath is None:
            self._killed_icon_filepath = str(self.killed_icon_dict[self.affiliation])
            with open(self._killed_icon_filepath) as i:
                self._killed_icon = i.read()
        else:
            if killed_icon_filepath.lower().endswith(".svg"):
                self._killed_icon_filepath = str(killed_icon_filepath)
                with open(self._killed_icon_filepath) as i:
                    self._killed_icon = i.read()
            else:
                raise TypeError("Icon file type incompatible. Please use SVG file type.")

        self.set_icon()

    def set_icon(self):
        """Set the icon based on the killed status."""
        if self.killed is True:
            self.icon_path = self._killed_icon_filepath
            self.icon = self._killed_icon
        else:
            self.icon_path = self._icon_filepath
            self.icon = self._icon

    def _select_pos(
        self, pos: Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d], Area]]
    ) -> Optional[Coordinate2dOr3d]:
        """Select a pos from the provided options.

        Args:
            pos (Optional[Union[Coordinate2dOr3d, list[Coordinate2dOr3d], Area]]): Either a single
                pos or a list of them

        Returns:
            Optional[Coordinate2dOr3d]: the given pos if given a single pos or a random
            one from the list if given a list
        """
        if isinstance(pos, tuple):
            return pos
        elif isinstance(pos, Area):
            return pos.get_coord_inside(self.random)
        elif pos is not None:
            coord: Coordinate2dOr3d = self.random.choice(pos)
            return coord
        else:
            return None

    def _normalise_arc_of_regard(
        self, arc_of_regard: Optional[ArcOfRegard]
    ) -> Optional[ArcOfRegard]:
        """Return a new, normalised arc of regard.

        A normalised arc of regard expresses degrees between 0 and 360 and the
        probability ratio is scaled between 0 and 1.

        Args:
            arc_of_regard (Optional[ArcOfRegard]): The arc of regard to normalise. If None return
                None.

        Returns:
            Optional[ArcOfRegard]: The normalised arc of regard. None is input arc of regard is
                None.
        """
        if arc_of_regard is None:
            return None
        normalised_arc_of_regard: ArcOfRegard = {}
        total = sum(arc_of_regard.values())
        for sector, relative_ratio in arc_of_regard.items():
            normalised_arc = (sector[0] % 360, sector[1] % 360)
            normalised_arc_of_regard[normalised_arc] = relative_ratio / total

        return normalised_arc_of_regard

    def _check_arc_of_regard_for_overlapping_sectors(self, arc_of_regard: ArcOfRegard):
        """Check that the arc of regard doesn't contain any overlapping sectors.

        Args:
            arc_of_regard (ArcOfRegard): The Arc of Regard to check

        Raises:
            ValueError: Raised if the Arc of Regard contains overlapping sectors
        """
        sectors = sorted(list(arc_of_regard.keys()), key=lambda x: x[0])
        for i in range(len(sectors)):
            next_i = (i + 1) % len(sectors)  # check the final sector against the first sector
            sector1_lower = sectors[i][0]
            sector1_upper = sectors[i][1]
            sector2_lower = sectors[next_i][0]

            if (
                sector1_upper > sector2_lower
                # Account for sectors which contain the point 0/360 degrees
                # e.g. (340, 20) and (350, 30) overlap but 20 < 350
                or (sector1_upper < sector1_lower < sector2_lower)
            ):
                raise ValueError("Arc of Regard cannot contain overlapping sectors.")

    def clear_old_statuses(self):
        """Clear agent statuses of any which occurred before the previous time step."""
        previous_step = self.model.get_time() - self.model.time_step
        self._statuses = [status for status in self._statuses if status.time_step >= previous_step]

    def get_statuses(self) -> list[Status]:
        """Get all statuses from the previous time-step.

        Only gets statuses from the previous time-step to account for agent activation order.
        """
        current_time = self.model.get_time()
        previous_step = current_time - self.model.time_step
        return [
            status for status in self._statuses if previous_step <= status.time_step < current_time
        ]

    def add_fired_status(
        self, weapon: Weapon, target_location: Coordinate2dOr3d, target_in_range: bool
    ):
        """Add weapon fired status to agent.

        Args:
            weapon (Weapon): The weapon that was fired.
            target_location (Coordinate2dOr3d): The target location the weapon was fired at.
            target_in_range (bool): The whether the target location was in range of the weapon.
        """
        self._statuses.append(
            WeaponFiredStatus(
                time_step=self.model.get_time(),
                weapon_name=weapon.name,
                target_location=target_location,
                target_in_range=target_in_range,
            )
        )

    def add_sensed_by_status(self, sense_time: datetime, senser: RetAgent):
        """Add agent sensed status to agent.

        Args:
            sense_time (datetime): The time of sensing.
            senser (RetAgent): The senser agent.
        """
        self._statuses.append(AgentSensedStatus(time_step=sense_time, sensing_agent=senser))


class AgentConstructionError(Exception):  # pragma: no cover
    """Exception for Agent construction."""

    pass


@dataclass
class Status:
    """An active status for an agent."""

    time_step: datetime


@dataclass
class WeaponFiredStatus(Status):
    """A weapon fired status."""

    weapon_name: str
    target_location: Coordinate2dOr3d
    target_in_range: bool


@dataclass
class AgentSensedStatus(Status):
    """An agent sensed status."""

    sensing_agent: RetAgent
