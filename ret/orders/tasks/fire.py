"""Firing tasks."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ret.orders.tasks.attack import AttackTask

if TYPE_CHECKING:
    from random import Random
    from typing import Optional

    from ret.agents.agent import RetAgent
    from ret.space.feature import Area
    from ret.types import Coordinate2dOr3d
    from ret.weapons.weapon import FireSchedule, Weapon


class WeaponSelector(ABC):
    """Encapsulation of approaches for picking which weapon to use from as list of possibilities."""

    @abstractmethod
    def select_weapon(
        self, candidate_weapons: list[Weapon], rand: Random
    ) -> Optional[Weapon]:  # pragma: no cover
        """Select a weapon to use from a list of available weapons.

        Args:
            candidate_weapons (List[Weapon]): List of candidate weapons to choose from.
            rand (Random): Randomiser to use to select weapon where any random characteristics are
                needed

        Returns:
            Optional[Weapon]: Chosen weapon, or None if no suitable weapons can be chosen
        """
        raise NotImplementedError


class RandomWeaponSelector(WeaponSelector):
    """Weapon selector for picking at random with equal probability."""

    def select_weapon(self, candidate_weapons: list[Weapon], rand: Random) -> Optional[Weapon]:
        """Select a weapon to use from a list of available weapons.

        If no weapons are available to choose from, return None and raise a warning.

        The basic implementation of the weapon selector picks a random weapon.

        Args:
            candidate_weapons (List[Weapon]): List of candidate weapons to choose from.
            rand (Random): Randomiser to use to select weapon where any random characteristics are
                needed

        Returns:
            Optional[Weapon]: Chosen weapon, or None if no suitable weapons can be chosen
        """
        if len(candidate_weapons) == 0:
            warnings.warn("No weapons are available for firing.", stacklevel=2)
            return None
        return rand.choice(candidate_weapons)


class RandomWeaponNonEmptySelector(WeaponSelector):
    """Weapon selector for picking a non empty ammo capacity at random with equal probability."""

    def select_weapon(self, candidate_weapons: list[Weapon], rand: Random) -> Optional[Weapon]:
        """Select a weapon to use from a list of available weapons.

        If no weapons are available to choose from, return None and raise a warning.

        The basic implementation of the weapon selector picks a random weapon.

        Args:
            candidate_weapons (List[Weapon]): List of candidate weapons to choose from.
            rand (Random): Randomiser to use to select weapon where any random characteristics are
                needed

        Returns:
            Optional[Weapon]: Chosen weapon, or None if no suitable weapons can be chosen
        """
        if len(candidate_weapons) == 0:
            warnings.warn("No weapons are available for firing.", stacklevel=2)
            return None

        valid_weapons = []

        for weapon in candidate_weapons:
            if hasattr(weapon, "ammo_capacity"):
                if weapon.ammo_capacity is not None:
                    if weapon.ammo_capacity > 0:
                        valid_weapons.append(weapon)

        if len(valid_weapons) == 0:
            warnings.warn("No weapons with ammo capacity available.", stacklevel=2)
            return None

        return rand.choice(valid_weapons)


class HighestAmmoSelector(WeaponSelector):
    """Weapon selector for picking the weapon with the highest ammo."""

    def select_weapon(self, candidate_weapons: list[Weapon], rand: Random) -> Optional[Weapon]:
        """Select a weapon to use from a list of available weapons.

        If no weapons are available to choose from, return None and raise a warning.

        The basic implementation of the weapon selector picks a random weapon.

        Args:
            candidate_weapons (List[Weapon]): List of candidate weapons to choose from.
            rand (Random): Randomiser to use to select weapon where any random characteristics are
                needed

        Returns:
            Optional[Weapon]: Chosen weapon, or None if no suitable weapons can be chosen
        """
        if len(candidate_weapons) == 0:
            warnings.warn("No weapons are available for firing.", stacklevel=2)
            return None

        chosen_weapons: list[Weapon] = []

        for weapon in candidate_weapons:
            if chosen_weapons == []:
                if weapon.ammo_capacity is not None:
                    chosen_weapons.append(weapon)
            elif chosen_weapons[0].ammo_capacity is not None and weapon.ammo_capacity is not None:
                if chosen_weapons[0].ammo_capacity < weapon.ammo_capacity:
                    chosen_weapons = [weapon]
                elif chosen_weapons[0].ammo_capacity == weapon.ammo_capacity:
                    chosen_weapons.append(weapon)

        if len(chosen_weapons) == 0:
            warnings.warn("No weapons are available for firing.", stacklevel=2)
            return None

        return rand.choice(chosen_weapons)


class NamedWeaponSelector(WeaponSelector):
    """Weapon Selector which picks a named weapon."""

    def __init__(self, name: str):
        """Create a new NamedWeaponSelector.

        Args:
            name (str): Name of weapon to select.
        """
        self.name = name

    def select_weapon(self, candidate_weapons: list[Weapon], rand: Random) -> Optional[Weapon]:
        """Select weapon based on name.

        Returns None if no weapon matching the name is available.

        If multiple weapons with the same name are present, the first will be selected.

        Args:
            candidate_weapons (list[BasicWeapon]): List of available weapons
            rand (Random): Random selector (unused)

        Returns:
            Optional[BasicWeapon]: Selected weapon, if found
        """
        if len(candidate_weapons) == 0:
            warnings.warn("No weapons are available for firing.", stacklevel=2)
            return None
        return next((w for w in candidate_weapons if w.name == self.name))


class FireTask(AttackTask):
    """Attack task that causes an agent to fire."""

    weapon: Optional[Weapon]
    firing_schedule: Optional[FireSchedule]
    rounds_to_fire: int
    weapon_selector: WeaponSelector

    def __init__(
        self,
        rounds: int = 1,
        weapon_selector: Optional[WeaponSelector] = None,
        log: bool = True,
    ) -> None:
        """Create a new Fire task.

        Args:
            rounds (int): Number of rounds to fire. Defaults to 1.
            weapon_selector (Optional[WeaponSelector]): The mechanism used to determine which weapon
                to fire. If None or unspecified, defaults to a new WeaponSelector.
            log (bool): Whether to log or not. Defaults to True.

        Raises:
            ValueError: Negative number of rounds provided
        """
        super().__init__(log=log)

        if weapon_selector is None:
            weapon_selector = RandomWeaponSelector()

        if rounds < 1:
            raise ValueError(f"Number of rounds must be > 0. {rounds} provided.")
        self.rounds_to_fire = rounds
        self.weapon_selector = weapon_selector
        self.weapon = None
        self.firing_schedule: Optional[FireSchedule] = None

    def _do_task_step(self, doer: RetAgent) -> None:
        """Do one time-step's worth of firing.

        Args:
            doer (RetAgent): the agent doing the task
        """
        self.firer = doer
        if doer.hiding:
            warnings.warn("Cannot fire while the agent is hiding.", stacklevel=2)
            return

        if self.weapon is None:
            self.weapon = self.weapon_selector.select_weapon(doer.weapons, doer.model.random)

        if self.weapon is None:
            warnings.warn("The firer has no suitable weapons for this Task.", stacklevel=2)
            return

        if self.firing_schedule is None:
            self.firing_schedule = self.weapon.create_fire_schedule(
                rounds_to_fire=self.rounds_to_fire, time_step=doer.model.time_step
            )

        if self.firing_schedule.is_complete():
            warnings.warn(f"{str(self)} is already complete.", stacklevel=2)
            doer.clear_fire_behaviour()
            return

        self.rounds_this_step = self.firing_schedule.get_round()
        target = self.get_target()
        doer.fire_step(rounds=self.rounds_this_step, weapon=self.weapon, target=target)

    def _is_task_complete(self, doer: RetAgent) -> bool:
        """Return whether or not the task is complete.

        If it has not been possible to initialise either the fire schedule or the weapon to fire,
        then the task will be assumed to be complete.

        Args:
            doer (RetAgent): Agent doing the firing

        Returns:
            bool: True if the task is complete. False otherwise
        """
        if self.firing_schedule is None or self.weapon is None:
            return False
        return self.firing_schedule.is_complete()


class FireAtTargetTask(FireTask):
    """Task that causes an agent to fire at a specific target."""

    def __init__(
        self,
        target: Coordinate2dOr3d,
        rounds: int = 1,
        weapon_selector: Optional[WeaponSelector] = None,
        log: bool = True,
    ):
        """Create a new FireAtTargetTask.

        Args:
            target (Coordinate2dOr3d): Target
            rounds (int): Number of rounds to fire. Defaults to 1.
            weapon_selector (WeaponSelector): The mechanism used to determine which weapon to fire.
                Defaults to None.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(rounds=rounds, weapon_selector=weapon_selector, log=log)
        self.target = target

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Fire at Target Task"

    def get_target(self) -> Coordinate2dOr3d:
        """Get the target to fire at.

        Returns:
            Coordinate2dOr3d: Target to fire at.
        """
        return self.target

    def get_new_instance(self) -> FireAtTargetTask:
        """Return a new instance of a functionally identical task.

        Returns:
            FireAtTargetTask: New instance of the task
        """
        return FireAtTargetTask(
            target=self.target,
            rounds=self.rounds_to_fire,
            weapon_selector=self.weapon_selector,
            log=self._log,
        )


class DetermineTargetAndFireTask(FireTask):
    """Task that causes an agent to fire, determining their own choice of target."""

    def __init__(
        self,
        rounds: int = 1,
        weapon_selector: Optional[WeaponSelector] = None,
        log: bool = True,
    ) -> None:
        """Create a new DetermineTargetAndFireTask.

        Args:
            rounds (int): Number of rounds to fire. Defaults to 1.
            weapon_selector (Optional[WeaponSelector]): The mechanism used to determine which weapon
                to fire. If None or unspecified, defaults to None.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(rounds=rounds, weapon_selector=weapon_selector, log=log)

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Determine Target and Fire Task"

    def get_target(self) -> None:
        """Get target of fire.

        Returns:
            None: The target to fire at.
        """
        return None

    def get_new_instance(self) -> DetermineTargetAndFireTask:
        """Return a new instance of a functionally identical task.

        Returns:
            DetermineTargetAndFireTask: New instance of the task.
        """
        return DetermineTargetAndFireTask(
            rounds=self.rounds_to_fire, weapon_selector=self.weapon_selector, log=self._log
        )


class FireAtAreaTask(FireTask):
    """Task that causes an agent to fire at a random location within an area."""

    def __init__(
        self,
        target_area: Area,
        random: Random,
        rounds: int = 1,
        weapon_selector: Optional[WeaponSelector] = None,
        log: bool = True,
    ):
        """Create a new FireAtAreaTask.

        Args:
            target_area (Area): Target area from which a position to fire at will be selected.
            random (Random): A random number generator.
            rounds (int): Number of rounds to fire. Defaults to 1.
            weapon_selector (WeaponSelector): The mechanism used to determine which weapon to fire.
                Defaults to None.
            log (bool): whether to log or not. Defaults to True.
        """
        super().__init__(rounds=rounds, weapon_selector=weapon_selector, log=log)
        self.target_area = target_area
        self.random = random

    def __str__(self):
        """Output a human readable name for the task.

        Returns:
            string: brief description of the task
        """
        return "Fire at Area Task"

    def get_target(self) -> list[Coordinate2dOr3d]:
        """Get the target to fire at.

        Returns:
            Coordinate2dOr3d: Target to fire at.
        """
        return [
            self.target_area.get_coord_inside(self.random) for _ in range(0, self.rounds_this_step)
        ]

    def get_new_instance(self) -> FireAtAreaTask:
        """Return a new instance of a functionally identical task.

        Returns:
            FireAtTargetTask: New instance of the task
        """
        return FireAtAreaTask(
            target_area=self.target_area,
            random=self.random,
            rounds=self.rounds_to_fire,
            weapon_selector=self.weapon_selector,
            log=self._log,
        )
