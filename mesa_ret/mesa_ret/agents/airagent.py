"""ret air agent representation."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from mesa_ret.behaviours.fire import FireBehaviour
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import AircraftMoveBehaviour, MoveBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.space.heightband import AbsoluteHeightBand
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
    from mesa_ret.space.clutter.countermeasure import Countermeasure
    from mesa_ret.space.feature import Area
    from mesa_ret.space.heightband import HeightBand
    from mesa_ret.template import Template
    from mesa_ret.types import Coordinate2dOr3d
    from mesa_ret.weapons.weapon import Weapon


ICON_DIR = Path(__file__).parent.joinpath("icons/airAgent")


class DefaultAirAgentWeapon(BasicLongRangedWeapon):
    """Default weapon for Air Agent."""

    def __init__(self):
        """Create a new DefaultAirAgentWeapon."""
        super().__init__(
            name="Default Air Agent Weapon",
            radius=3000,
            time_before_first_shot=timedelta(seconds=0),
            time_between_rounds=timedelta(seconds=30),
            kill_probability_per_round=0.5,
            max_percentage_inaccuracy=0,
        )


class AirAgent(RetAgent):
    """A class representing an airborn agent."""

    icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("airagent_friendly.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("airagent_hostile.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("airagent_neutral.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("airagent_unknown.svg"),
    }

    killed_icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("killed/airagent_friendly_killed.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("killed/airagent_hostile_killed.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("killed/airagent_neutral_killed.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("killed/airagent_unknown_killed.svg"),
    }

    def __init__(
        self,
        model: RetModel,
        pos: Union[Coordinate2dOr3d, list[Coordinate2dOr3d], Area],
        name: str,
        affiliation: Affiliation,
        critical_dimension: float = 40.0,
        reflectivity: float = 0.31,
        temperature: float = 30.0,
        temperature_std_dev: float = 0.0,
        height_bands: Optional[list[HeightBand]] = None,
        icon_path: Optional[str] = None,
        killed_icon_path: Optional[str] = None,
        orders: Optional[list[Template[Order]]] = None,
        behaviours: Optional[list[Behaviour]] = None,
        behaviour_adder: Optional[type[ListAdder]] = None,
        sensors: Optional[list[Template[Sensor]]] = None,
        countermeasures: Optional[list[Template[Countermeasure]]] = None,
        background_orders: Optional[list[Template[BackgroundOrder]]] = None,
        arc_of_regard: Optional[ArcOfRegard] = None,
        communication_receiver: Optional[CommunicationReceiver] = None,
        refresh_technique: Optional[PerceivedAgentFilter] = None,
        weapons: Optional[list[Weapon]] = None,
    ) -> None:
        """Create an air agent.

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
            height_bands: (Optional[list[HeightBand]]): list of height bands in which
                the aircraft can move. Defaults to None.
            icon_path (Optional[str]): path to the agent's icon. Defaults to None
            killed_icon_path (Optional[str]): path to the agent's icon when killed.
                Defaults to None.
            orders (list[Template[Order]]): the agents initial set of orders
            behaviours (Optional[list[Behaviour]]): Agent behaviours. Defaults to None
            behaviour_adder (Optional[ListAdder]): Behaviour adding methodology to be
                supplied to the RetAgent class. Defaults to None.
            sensors (list[Template[Sensor]]): sensor templates defining the
                sensors that will belong to the agent
            countermeasures (list[Template[Countermeasure]]): templates of the
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
            communication_receiver (Optional[CommunicationReceiver]): the agent's
                communication receiver
            refresh_technique (Optional[PerceivedAgentFilter[]): Methodology for
                refreshing the perceived world
            weapons (Optional[list[Weapon]]): List of weapons the agent is armed with. If None,
                a single DefaultAirAgentWeapon will be used. If the caller wishes to create an air
                agent without any weapons, an empty list argument (e.g., `[]` or `list()`) should be
                provided. Defaults to None.
        """
        if weapons is None:
            weapons = [DefaultAirAgentWeapon()]

        super().__init__(
            model=model,
            pos=pos,
            name=name,
            affiliation=affiliation,
            critical_dimension=critical_dimension,
            reflectivity=reflectivity,
            temperature=temperature,
            temperature_std_dev=temperature_std_dev,
            agent_type=AgentType.AIR,
            icon_path=icon_path,
            killed_icon_path=killed_icon_path,
            behaviours=behaviours,
            behaviour_adder=behaviour_adder,
            orders=orders,
            sensors=sensors,
            communication_receiver=communication_receiver,
            refresh_technique=refresh_technique,
            countermeasures=countermeasures,
            background_orders=background_orders,
            arc_of_regard=arc_of_regard,
            weapons=weapons,
        )

        default_wait_behaviour = WaitBehaviour()
        self.behaviour_pool.add_default_behaviour(default_wait_behaviour, WaitBehaviour)

        default_hide_behaviour = HideBehaviour()
        self.behaviour_pool.add_default_behaviour(default_hide_behaviour, HideBehaviour)

        if height_bands is None:
            height_bands = [
                AbsoluteHeightBand("500m", 500),
                AbsoluteHeightBand("2000m", 2000),
                AbsoluteHeightBand("12000m", 12000),
            ]

        default_move_behaviour = AircraftMoveBehaviour(3, height_bands)
        self.behaviour_pool.add_default_behaviour(default_move_behaviour, MoveBehaviour)

        default_fire_behaviour = FireBehaviour()
        self.behaviour_pool.add_default_behaviour(default_fire_behaviour, FireBehaviour)

        default_deploy_countermeasure_behaviour = DeployCountermeasureBehaviour()
        self.behaviour_pool.add_default_behaviour(
            default_deploy_countermeasure_behaviour, DeployCountermeasureBehaviour
        )

        self._real_pos: Coordinate2dOr3d = self.model.space.get_coordinate_in_correct_dimension(
            self.pos
        )
