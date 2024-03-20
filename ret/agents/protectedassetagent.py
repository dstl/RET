"""Protected asset agent."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent

if TYPE_CHECKING:
    from typing import Optional, Union

    from ret.agents.agent import Coordinate2dOr3d
    from ret.model import RetModel
    from ret.space.feature import Area

ICON_DIR = ICON_DIR = Path(__file__).parent.joinpath("icons/protectedAsset")


class ProtectedAssetAgent(RetAgent):
    """A class representing a protected asset."""

    icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("protectedassetagent_friendly.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("protectedassetagent_hostile.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("protectedassetagent_neutral.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("protectedassetagent_unknown.svg"),
    }

    killed_icon_dict: dict[Affiliation, Path] = {
        Affiliation.FRIENDLY: ICON_DIR.joinpath("killed/protectedassetagent_friendly_killed.svg"),
        Affiliation.HOSTILE: ICON_DIR.joinpath("killed/protectedassetagent_hostile_killed.svg"),
        Affiliation.NEUTRAL: ICON_DIR.joinpath("killed/protectedassetagent_neutral_killed.svg"),
        Affiliation.UNKNOWN: ICON_DIR.joinpath("killed/protectedassetagent_unknown_killed.svg"),
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
        icon_path: Optional[str] = None,
        killed_icon_path: Optional[str] = None,
    ) -> None:
        """Create a protected asset agent.

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
            icon_path (Optional[str], optional): path to the agent's icon
            killed_icon_path (Optional[str], optional): path to the agent's icon when
                killed
        """
        super().__init__(
            model,
            pos,
            name,
            affiliation,
            critical_dimension=critical_dimension if critical_dimension else 7.0,
            reflectivity=reflectivity if reflectivity else 0.081,
            temperature=temperature if temperature else 5.0,
            temperature_std_dev=temperature_std_dev,
            icon_path=icon_path,
            killed_icon_path=killed_icon_path,
        )
