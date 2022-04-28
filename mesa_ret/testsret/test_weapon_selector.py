"""Tests for weapon selectors."""
from __future__ import annotations

from random import Random
from typing import Generic, TypeVar
from warnings import catch_warnings

from mesa_ret.orders.tasks.fire import NamedWeaponSelector, RandomWeaponSelector
from mesa_ret.testing.mocks import MockWeapon
from mesa_ret.weapons.weapon import Weapon
from pytest import mark

T = TypeVar("T")


class MockRandom(Generic[T]):
    """Randomiser that always returns an item at a given location in a list."""

    def __init__(self, choice: int):
        """Create a new MockRandom.

        Args:
            choice (int): The index (base 0) of the item in the list to return
        """
        self._choice = choice

    def choice(self, candidates: list[T]) -> T:
        """Choose item from list.

        Args:
            candidates (list[T]): List to select from.

        Returns:
            T: The chosen item from the list
        """
        return candidates[self._choice]


@mark.parametrize("position", [0, 1, 2])
def test_random_weapon_selector(position: int):
    """Test that the random weapon selector picks a weapon at random.

    Args:
        position (int): The position in the weapon list to choose.
    """
    selector = RandomWeaponSelector()
    weapons: list[Weapon] = [MockWeapon(), MockWeapon(), MockWeapon()]

    selected_weapon = selector.select_weapon(weapons, MockRandom[Weapon](position))  # type: ignore
    assert selected_weapon is weapons[position]


def test_random_weapon_selector_for_empty_list():
    """Check error handling where RandomWeaponSelector picks from empty list."""
    selector = RandomWeaponSelector()
    with catch_warnings(record=True) as w:
        selector.select_weapon([], Random())

    assert len(w) == 1
    assert "No weapons are available for firing" in str(w[0].message)


def test_named_weapon_selector_for_empty_list():
    """Check error handling where NamedWeaponSelector picks from empty list."""
    selector = NamedWeaponSelector("N/A")
    with catch_warnings(record=True) as w:
        selector.select_weapon([], Random())

    assert len(w) == 1
    assert "No weapons are available for firing" in str(w[0].message)


def test_named_weapon_selector():
    """Test named weapon selector functionality."""
    selector = NamedWeaponSelector("W1")
    weapons: list[Weapon] = [MockWeapon(), MockWeapon(), MockWeapon()]
    weapons[2].name = "W1"

    w1 = selector.select_weapon(weapons, Random())

    assert w1 is not None
    assert w1.name == "W1"
