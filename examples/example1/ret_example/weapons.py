"""Weapons for the Example model."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mesa import Agent
from ret.weapons.weapon import BasicWeapon

if TYPE_CHECKING:
    from datetime import timedelta
    from typing import Optional

    from mesa.space import GridContent
    from ret.agents.agent import RetAgent
    from ret.agents.agenttype import AgentType


class TargetTypeSpecificPKillWeapon(BasicWeapon):
    """Indirect fire rocket weapon."""

    def __init__(
        self,
        name: str,
        time_between_rounds: timedelta,
        time_before_first_shot: timedelta,
        radius: float,
        p_kill_dict: dict[AgentType, float],
        min_range: Optional[float],
        max_range: Optional[float],
    ):
        """Initialise an indirect fire rocket weapon.

        Args:
            name (str): Name of the weapon
            radius (float): Radius of targets that are hit where firing at a specific location.
            time_between_rounds (timedelta): time between shots are fired
            time_before_first_shot (timedelta): Time before first shot is fired
            p_kill_dict (dict): p kill value per agent type.
            min_range (Optional[float]): Minimum range that the weapon can fire at. If None, there
                is no minimum range. Defaults to None
            max_range (Optional[float]): Maximum range that the weapon can fire at. If None, there
                is no maximum range. Defaults to None
        """
        super().__init__(
            name=name,
            radius=radius,
            time_between_rounds=time_between_rounds,
            time_before_first_shot=time_before_first_shot,
            kill_probability_per_round=0.0,
            min_range=min_range,
            max_range=max_range,
        )

        self.p_kill_dict = p_kill_dict

    def try_kill(self, firer: RetAgent, target: GridContent, shot_id: Optional[int]):
        """Try and kill the target.

        Args:
            firer (RetAgent): The agent firing the weapon
            target (GridContent): The agent being fired upon
            shot_id (Optional[int]): The ID of the shot being fired, if known

        Raises:
            AttributeError: If target is not a RetAgent or doesn't have a 'kill' method
        """

        def try_kill_agent(agent: Agent):
            try:
                kill_probability = self.p_kill_dict.get(agent.agent_type, 0.0)
                if firer.model.random.random() < kill_probability:
                    agent.kill(killer=firer, shot_id=shot_id)  # type: ignore
            except AttributeError:
                raise AttributeError("Target is not a RetAgent, or doesn't have a 'kill' method.")

        if isinstance(target, Agent):
            try_kill_agent(target)
        elif isinstance(target, list):
            for agent in target:
                try_kill_agent(agent)

    def get_new_instance(self) -> TargetTypeSpecificPKillWeapon:
        """Create a new instance of the weapon.

        Returns:
            Weapon: New instance of the weapon
        """
        return TargetTypeSpecificPKillWeapon(
            name=self.name,
            radius=self._radius,
            time_between_rounds=self._time_between_rounds,
            time_before_first_shot=self._time_before_first_shot,
            min_range=self._min_range,
            max_range=self._max_range,
            p_kill_dict=self.p_kill_dict,
        )
