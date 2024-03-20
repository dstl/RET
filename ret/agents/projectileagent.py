"""An agent class that represents a projectile in RET."""

from __future__ import annotations

from typing import TYPE_CHECKING
from math import inf
from pathlib import Path

from ret.agents.agent import RetAgent
from ret.agents.agenttype import AgentType
from ret.behaviours.move import GroundBasedMoveBehaviour
from ret.behaviours.fire import FireBehaviour
from ret.agents.affiliation import Affiliation

if TYPE_CHECKING:
    from typing import Optional, Union
    from ret.orders.order import Order
    from ret.behaviours import Behaviour
    from ret.space.feature import Area
    from ret.model import RetModel
    from ret.template import Template
    from ret.types import Coordinate2d
    from ret.weapons.weapon import Weapon

ICON_DIR = Path(__file__).parent.joinpath("icons/projectile/")


class ProjectileAgent(RetAgent):
    """A class representing a projectile."""

    icon_dict = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("projectileagent_friendly.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("projectileagent_hostile.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("projectileagent_neutral.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("projectileagent_unknown.svg"),
    }

    killed_icon_dict = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
    }

    def __init__(
        self,
        model: RetModel,
        pos: Optional[Union[Coordinate2d, list[Coordinate2d], Area]],
        name: str,
        affiliation: Affiliation,
        critical_dimension: float,
        base_speed: float,
        weapon: Weapon,
        firer: RetAgent,
        reflectivity: Optional[float] = None,
        temperature: Optional[float] = None,
        temperature_std_dev: float = 0.0,
        icon_path: Optional[str] = None,
        killed_icon_path: Optional[str] = None,
        orders: Optional[list[Template[Order]]] = None,
        agent_type: Optional[AgentType] = None,
    ) -> None:
        """Create a ProjectileAgent.

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
            base_speed (float): The base average speed across ground of the projectile.
            weapon (Weapon): The weapon representing the projectile payload.
            firer (RetAgent): The agent that fired the weapon
            reflectivity (float): The reflectivity of the agent, greater than 0 and less
                than or equal to 1.
            temperature (float): The temperature of the agent in degrees C
            temperature_std_dev (float): The standard deviation of the agent temperature in
                degrees C, default is 0
            icon_path (Optional[str]): path to the agent's icon
            killed_icon_path (Optional[str]): path to the agent's killed icon
            orders (list[Template[Order]]): templates of the agents initial set of
                orders
            agent_type (Optional[AgentType]): Specifies if the projectile
                should be reported in observations as guided or unguided.
        """
        gradient_speed_modifiers = [
            ((-inf, -1.1), 1.0),
            ((-1.1, 1.1), 1.0),
            ((1.1, inf), 1.0),
        ]
        behaviours: Optional[list[Behaviour]] = [
            GroundBasedMoveBehaviour(base_speed, gradient_speed_modifiers),
            FireBehaviour(),
        ]
        if agent_type is None:
            agent_type = AgentType.PROJECTILE
        weapons = [weapon]
        self.firer = firer
        self.pos_when_fired = firer.pos
        super().__init__(
            model=model,
            pos=pos,  # type: ignore
            name=name,
            affiliation=affiliation,
            agent_type=agent_type,
            critical_dimension=critical_dimension,
            reflectivity=reflectivity,
            temperature=temperature,
            temperature_std_dev=temperature_std_dev,
            icon_path=icon_path,
            killed_icon_path=killed_icon_path,
            orders=orders,
            behaviours=behaviours,
            weapons=weapons,
        )


class GuidedProjectileAgent(RetAgent):
    """A class representing a guided projectile."""

    icon_dict = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("projectileagent_friendly.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("projectileagent_hostile.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("projectileagent_neutral.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("projectileagent_unknown.svg"),
    }

    killed_icon_dict = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("killed/projectileagent_killed.svg"),
    }

    def __init__(
        self,
        model: RetModel,
        pos: Optional[Union[Coordinate2d, list[Coordinate2d], Area]],
        name: str,
        affiliation: Affiliation,
        critical_dimension: float,
        base_speed: float,
        weapon: Weapon,
        reflectivity: Optional[float] = None,
        temperature: Optional[float] = None,
        temperature_std_dev: float = 0.0,
        icon_path: Optional[str] = None,
        killed_icon_path: Optional[str] = None,
        orders: Optional[list[Template[Order]]] = None,
    ) -> None:
        """Create a ProjectileAgent.

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
            base_speed (float): The base average speed across ground of the projectile.
            weapon (Weapon): The weapon representing the projectile payload.
            reflectivity (float): The reflectivity of the agent, greater than 0 and less
                than or equal to 1.
            temperature (float): The temperature of the agent in degrees C
            temperature_std_dev (float): The standard deviation of the agent temperature in
                degrees C, default is 0
            icon_path (Optional[str]): path to the agent's icon
            killed_icon_path (Optional[str]): path to the agent's killed icon
            orders (list[Template[Order]]): templates of the agents initial set of
                orders
        """
        gradient_speed_modifiers = [
            ((-inf, -1.1), 1.0),
            ((-1.1, 1.1), 1.0),
            ((1.1, inf), 1.0),
        ]
        behaviours: Optional[list[Behaviour]] = [
            GroundBasedMoveBehaviour(base_speed, gradient_speed_modifiers),
            FireBehaviour(),
        ]

        weapons = [weapon]
        super().__init__(
            model=model,
            pos=pos,  # type: ignore
            name=name,
            affiliation=affiliation,
            agent_type=AgentType.GUIDED_PROJECTILE,
            critical_dimension=critical_dimension,
            reflectivity=reflectivity,
            temperature=temperature,
            temperature_std_dev=temperature_std_dev,
            icon_path=icon_path,
            killed_icon_path=killed_icon_path,
            orders=orders,
            behaviours=behaviours,
            weapons=weapons,
        )
