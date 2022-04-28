"""Sensor fusion agent."""
from __future__ import annotations

from math import inf
from pathlib import Path
from typing import TYPE_CHECKING

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.agents.agenttype import AgentType
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import GroundBasedMoveBehaviour, MoveBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.communication.sensorfusionreceiver import SensorFusionReceiver
from mesa_ret.orders.order import Order
from mesa_ret.orders.tasks.communicate import CommunicateWorldviewTask
from mesa_ret.orders.triggers.immediate import ImmediateSensorFusionTrigger

if TYPE_CHECKING:
    from typing import Optional, Union

    from mesa_ret.behaviours import Behaviour
    from mesa_ret.behaviours.behaviourpool import ListAdder
    from mesa_ret.model import RetModel
    from mesa_ret.sensing.perceivedworld import PerceivedAgentFilter
    from mesa_ret.space.culture import Culture
    from mesa_ret.space.feature import Area
    from mesa_ret.types import Coordinate2dOr3d

ICON_DIR = Path(__file__).parent.joinpath("icons/sensorFusion")


class SensorFusionAgent(RetAgent):
    """Sensor fusion agent."""

    icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("sensorfusionagent_friendly.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("sensorfusionagent_hostile.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("sensorfusionagent_neutral.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("sensorfusionagent_unknown.svg"),
    }

    killed_icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("killed/sensorfusionagent_friendly_killed.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("killed/sensorfusionagent_hostile_killed.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("killed/sensorfusionagent_neutral_killed.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("killed/sensorfusionagent_unknown_killed.svg"),
    }

    new_info: bool

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
        refresh_technique: Optional[PerceivedAgentFilter] = None,
        behaviours: Optional[list[Behaviour]] = None,
        behaviour_adder: Optional[type[ListAdder]] = None,
    ) -> None:
        """Create a new Sensor Fusion Agent.

        Many of the standard RetAgent parameters are set to none, in all cases
        for the sensor fusion agent.

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
                Defaults to None.
            refresh_technique (Optional[PerceivedAgentFilter]): Methodology for
                refreshing the perceived world. Defaults to None.
            behaviours (Optional[list[Behaviour]]): Agent behaviours. Defaults to None
            behaviour_adder (Optional[ListAdder]): Behaviour adding methodology to be
                supplied to the RetAgent class. Defaults to None.
        """
        order = Order(ImmediateSensorFusionTrigger(), CommunicateWorldviewTask(), persistent=True)

        super().__init__(
            model=model,
            pos=pos,
            name=name,
            affiliation=affiliation,
            critical_dimension=critical_dimension if critical_dimension else 4.0,
            reflectivity=reflectivity if reflectivity else 0.081,
            temperature=temperature if temperature else 5.0,
            temperature_std_dev=temperature_std_dev,
            agent_type=AgentType.SENSOR_FUSION,
            icon_path=icon_path,
            killed_icon_path=killed_icon_path,
            orders=[order],
            behaviours=behaviours,
            behaviour_adder=behaviour_adder,
            communication_receiver=SensorFusionReceiver(),
            refresh_technique=refresh_technique,
        )

        if culture_speed_modifiers is None:
            cultures = model.space.get_cultures()
            if cultures:
                culture_speed_modifiers = dict((culture, 1.0) for culture in cultures)

        wait_behaviour = WaitBehaviour()
        self.behaviour_pool.add_default_behaviour(wait_behaviour, WaitBehaviour)

        hide_behaviour = HideBehaviour()
        self.behaviour_pool.add_default_behaviour(hide_behaviour, HideBehaviour)

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
        self.new_info = False

    def step(self) -> None:
        """Step the sensor fusion agent.

        Reset new information received to false.
        """
        super().step()
        self.active_order = None
        self.new_info = False
