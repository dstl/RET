"""Air defence agent."""

from __future__ import annotations

from datetime import timedelta
from math import inf
from pathlib import Path
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.behaviours.disablecommunication import (
    DisableAllHostileCommsInRangeBehaviour,
    DisableCommunicationBehaviour,
)
from mesa_ret.behaviours.fire import FireBehaviour
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import GroundBasedMoveBehaviour, MoveBehaviour
from mesa_ret.behaviours.sense import SenseBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.sensing.sensor import (
    LineOfSightSensor,
    SensorDistanceThresholds,
    SensorSamplingDistance,
)
from mesa_ret.weapons.weapon import BasicLongRangedWeapon

if TYPE_CHECKING:
    from typing import Optional, Union

    from mesa_ret.behaviours import Behaviour
    from mesa_ret.behaviours.behaviourpool import ListAdder
    from mesa_ret.communication.communicationreceiver import CommunicationReceiver
    from mesa_ret.model import RetModel
    from mesa_ret.orders.background_order import BackgroundOrder
    from mesa_ret.orders.order import Order
    from mesa_ret.sensing.perceivedworld import PerceivedAgentFilter
    from mesa_ret.sensing.sensor import ArcOfRegard, Sensor
    from mesa_ret.space.culture import Culture
    from mesa_ret.space.feature import Area
    from mesa_ret.template import Template
    from mesa_ret.types import Coordinate2dOr3d
    from mesa_ret.weapons.weapon import Weapon

ICON_DIR = Path(__file__).parent.joinpath("icons/airDefence")


class DefaultAirDefenceAgentWeapon(BasicLongRangedWeapon):
    """Default weapon for Air Defence Agent."""

    def __init__(self):
        """Create a new DefaultAirDefenceAgentWeapon."""
        super().__init__(
            name="Default Air Defence Agent Weapon",
            radius=1000,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=5),
            kill_probability_per_round=0.05,
            max_percentage_inaccuracy=0,
        )


class AirDefenceAgent(RetAgent):
    """A class representing an air defence unit."""

    icon_dict = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("airdefenceagent_friendly.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("airdefenceagent_hostile.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("airdefenceagent_neutral.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("airdefenceagent_unknown.svg"),
    }

    killed_icon_dict = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("killed/airdefenceagent_friendly_killed.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("killed/airdefenceagent_hostile_killed.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("killed/airdefenceagent_neutral_killed.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("killed/airdefenceagent_unknown_killed.svg"),
    }

    def __init__(
        self,
        model: RetModel,
        pos: Union[Coordinate2dOr3d, list[Coordinate2dOr3d], Area],
        name: str,
        affiliation: Affiliation,
        critical_dimension: Optional[float] = None,
        reflectivity: Optional[float] = None,
        temperature: Optional[float] = None,
        temperature_std_dev: float = 0.0,
        culture_speed_modifiers: Optional[dict[Culture, float]] = None,
        icon_path: Optional[str] = None,
        killed_icon_path: Optional[str] = None,
        orders: Optional[list[Template[Order]]] = None,
        behaviours: Optional[list[Behaviour]] = None,
        behaviour_adder: Optional[type[ListAdder]] = None,
        sensors: Optional[list[Template[Sensor]]] = None,
        background_orders: Optional[list[Template[BackgroundOrder]]] = None,
        arc_of_regard: Optional[ArcOfRegard] = None,
        communication_receiver: Optional[CommunicationReceiver] = None,
        refresh_technique: Optional[PerceivedAgentFilter] = None,
        weapons: Optional[list[Weapon]] = None,
    ) -> None:
        """Create an air defence agent.

        Args:
            model (RetModel): the model the agent will be placed in
            pos (Union[Coordinate2dOr3d, list[Coordinate2dOr3d], Area]): one
                of:
                    the initial position of the agent
                    a list of possible initial positions
                    an area for the agent to be placed in
            name (str): the name of the agent
            affiliation (Affiliation): the affiliation of the agent
            critical_dimension (float): The length (m) of the longest dimension of the
                agent
            reflectivity (float): The reflectivity of the agent, greater than 0 and less
                than or equal to 1.
            temperature (float): The temperature of the agent in degrees C
            temperature_std_dev (float): The standard deviation of the agent temperature in
                degrees C, default is 0
            culture_speed_modifiers (Optional[dict[Culture, float]]): the speed
                modifiers for the agent for each culture in the space. Defaults to None.
            icon_path (Optional[str]): path to the agent's icon. Defaults to None
            killed_icon_path (Optional[str]): path to the agent's icon when killed.
                Defaults to None
            orders (list[Template[Order]]): the agents initial set of orders
            behaviours (Optional[list[Behaviour]]): Agent behaviours. Defaults to None
            behaviour_adder (Optional[type[ListAdder]]): Behaviour adding methodology to
                be supplied to the RetAgent class. Defaults to None.
            sensors (list[Template[Sensor]]): sensor templates defining the
                sensors that will belong to the agent
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
            communication_receiver (CommunicationReceiver, optional): the agent's
                    communication receiver
            refresh_technique (PerceivedAgentFilter, optional): Methodology for
                    refreshing the perceived world
            weapons (Optional[list[Weapon]]): List of weapons the agent is armed with. If None,
                a single DefaultAirAgentWeapon will be used. If the caller wishes to create an air
                defence agent without any weapons, an empty list argument (e.g., `[]` or `list()`)
                should be provided. Defaults to None.
        """
        if sensors is None:
            sensors = [
                LineOfSightSensor(
                    distance_thresholds=SensorDistanceThresholds(
                        max_detect_dist=200000,
                        max_recognise_dist=150000,
                        max_identify_dist=75000,
                    ),
                    sampling_distance=SensorSamplingDistance(sampling_distance=25000),
                )
            ]

        if weapons is None:
            weapons = [DefaultAirDefenceAgentWeapon()]

        super().__init__(
            model=model,
            pos=pos,
            name=name,
            affiliation=affiliation,
            critical_dimension=critical_dimension if critical_dimension else 4.0,
            reflectivity=reflectivity if reflectivity else 0.105,
            temperature=temperature if temperature else 25.0,
            temperature_std_dev=temperature_std_dev,
            agent_type=AgentType.AIR_DEFENCE,
            icon_path=icon_path,
            killed_icon_path=killed_icon_path,
            orders=orders,
            behaviours=behaviours,
            behaviour_adder=behaviour_adder,
            sensors=sensors,
            background_orders=background_orders,
            arc_of_regard=arc_of_regard,
            communication_receiver=communication_receiver,
            refresh_technique=refresh_technique,
            weapons=weapons,
        )

        wait_behaviour = WaitBehaviour()
        self.behaviour_pool.add_default_behaviour(wait_behaviour, WaitBehaviour)

        hide_behaviour = HideBehaviour()
        self.behaviour_pool.add_default_behaviour(hide_behaviour, HideBehaviour)

        if culture_speed_modifiers is None:
            cultures = model.space.get_cultures()
            if cultures:
                culture_speed_modifiers = dict((culture, 1.0) for culture in cultures)

        move_behaviour = GroundBasedMoveBehaviour(
            base_speed=0.015,
            gradient_speed_modifiers=[
                ((-inf, -1.1), 0.8),
                ((-1.1, 1.1), 1),
                ((1.1, inf), 0.8),
            ],
            culture_speed_modifiers=culture_speed_modifiers,
        )
        self.behaviour_pool.add_default_behaviour(move_behaviour, MoveBehaviour)

        sense_behaviour = SenseBehaviour(
            time_before_first_sense=timedelta(seconds=0), time_between_senses=timedelta(seconds=5)
        )
        self.behaviour_pool.add_default_behaviour(sense_behaviour, SenseBehaviour)

        fire_behaviour = FireBehaviour()
        self.behaviour_pool.add_default_behaviour(fire_behaviour, FireBehaviour)

        disable_communication_behaviour = DisableAllHostileCommsInRangeBehaviour(10)
        self.behaviour_pool.add_default_behaviour(
            disable_communication_behaviour, DisableCommunicationBehaviour
        )
