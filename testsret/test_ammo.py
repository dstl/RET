"""Tests for functionality around ammo."""
from __future__ import annotations

import unittest
from random import Random
from parameterized import parameterized
from warnings import catch_warnings
from typing import TYPE_CHECKING
from ret.orders.tasks.fire import RandomWeaponNonEmptySelector, HighestAmmoSelector
from ret.testing.mocks import MockWeaponWithAmmo, MockWeapon

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.behaviours.fire import FireBehaviour
from ret.orders.order import Order
from ret.orders.tasks.fire import FireAtTargetTask
from ret.orders.triggers.immediate import ImmediateTrigger
from ret.testing.mocks import MockModel2d
from ret.agents.agent import WeaponFiredStatus
import datetime
import warnings

if TYPE_CHECKING:
    from ret.weapons.weapon import Weapon


def test_random_non_empty_weapon_selector():
    """Test demonstrating the Random weapon non-empty selector."""
    selector = RandomWeaponNonEmptySelector()
    w2 = MockWeaponWithAmmo()
    w2.ammo_capacity = 0
    weapons: list[Weapon] = [MockWeaponWithAmmo(), w2, MockWeapon()]
    weapons[0].name = "W1"

    w1 = selector.select_weapon(weapons, Random())

    assert w1 is not None
    assert w1.name == "W1"


def test_random_non_empty_weapon_selector_empty_array():
    """Test demonstrating the Random weapon non-empty selector when given empty list of weapons."""
    selector = RandomWeaponNonEmptySelector()

    with warnings.catch_warnings(record=True) as w:
        assert not selector.select_weapon([], Random())
    assert "No weapons are available for firing." in str(w[0].message)


def test_random_non_empty_weapon_selector_no_ammo():
    """Test demonstrating the Random weapon non-empty selector when given weapons with no ammo."""
    selector = RandomWeaponNonEmptySelector()
    weapons: list[Weapon] = [MockWeapon(), MockWeapon(), MockWeapon()]

    with warnings.catch_warnings(record=True) as w:
        assert not selector.select_weapon(weapons, Random())
    assert "No weapons with ammo capacity available." in str(w[0].message)


def test_highest_ammo_weapon_selector():
    """Test demonstrating the highest ammo weapon selector."""
    selector = HighestAmmoSelector()
    w1 = MockWeaponWithAmmo()  # Weapon with 60 ammo_capacity
    w2 = MockWeaponWithAmmo()
    w2.ammo_capacity = 20  # weapon with 20 ammo_capacity
    w3 = MockWeaponWithAmmo()
    w3.ammo_capacity = 0  # weapon with 0 ammo_capacity
    w4 = MockWeapon()  # weapon with None ammo_capacity
    weapons: list[Weapon] = [w1, w2, w3, w4]
    weapons[0].name = "W1"

    w1 = selector.select_weapon(weapons, Random())

    assert w1 is not None
    assert w1.name == "W1"


def test_random_highest_ammo_weapon_selector_empty_array():
    """Testing the Random weapon highest ammo selector when given empty list of weapons."""
    selector = HighestAmmoSelector()

    with warnings.catch_warnings(record=True) as w:
        assert not selector.select_weapon([], Random())
    assert "No weapons are available for firing." in str(w[0].message)


def test_random_highest_ammo_weapon_selector_no_ammo():
    """Test demonstrating the highest ammo selector when given weapons with no ammo."""
    selector = HighestAmmoSelector()
    weapons: list[Weapon] = [MockWeapon(), MockWeapon(), MockWeapon()]

    with warnings.catch_warnings(record=True) as w:
        assert not selector.select_weapon(weapons, Random())
    assert "No weapons are available for firing." in str(w[0].message)


def test_ammo_after_firing():
    """Test demonstrating the ammo capacity reduces after a fire."""
    fire_order_1 = Order(ImmediateTrigger(), FireAtTargetTask(target=(1, 1)), persistent=False)
    model = MockModel2d()
    firer = RetAgent(
        affiliation=Affiliation.NEUTRAL,
        pos=(0, 0),
        name="Firer",
        model=model,
        critical_dimension=1.0,
        behaviours=[FireBehaviour()],
        weapons=[MockWeaponWithAmmo()],
        orders=[fire_order_1],
    )

    ammo_capacity = firer.weapons[0].ammo_capacity
    model.step()

    ammo_capacity_after_firing = firer.weapons[0].ammo_capacity

    assert ammo_capacity - 1 == ammo_capacity_after_firing


def test_firing_with_no_ammo():
    """Test demonstrating gun will not fire with 0 ammo but will when ammo is supplied."""
    fire_order_1 = Order(ImmediateTrigger(), FireAtTargetTask(target=(1, 1)), persistent=True)
    model = MockModel2d()
    firer = RetAgent(
        affiliation=Affiliation.NEUTRAL,
        pos=(0, 0),
        name="Firer",
        model=model,
        critical_dimension=1.0,
        behaviours=[FireBehaviour()],
        weapons=[MockWeaponWithAmmo()],
        orders=[fire_order_1],
    )
    with catch_warnings(record=True) as w:
        firer.weapons[0].ammo_capacity = 0
        model.step()
    assert len(w) == 1
    assert f"{MockWeaponWithAmmo().name} nas no ammo, thus agent cannot fire" in str(w[0].message)

    assert firer._statuses == []

    firer.weapons[0].ammo_capacity = 1
    model.step()

    assert firer._statuses == [
        WeaponFiredStatus(
            time_step=datetime.datetime(2020, 1, 1, 1, 0),
            weapon_name="Mock Weapon with ammo",
            target_location=(1, 1),
            target_in_range=True,
        )
    ]


def test_no_ammo_warning():
    """Test demonstrating warning will fire in fire_at_target if weapon has no ammo."""
    model = MockModel2d()
    fire_order_1 = Order(ImmediateTrigger(), FireAtTargetTask(target=(1, 1)), persistent=True)
    firer = RetAgent(
        affiliation=Affiliation.NEUTRAL,
        pos=(0, 0),
        name="Firer",
        model=model,
        critical_dimension=1.0,
        behaviours=[FireBehaviour()],
        weapons=[MockWeaponWithAmmo()],
        orders=[fire_order_1],
    )
    with catch_warnings(record=True) as w1:
        firer.weapons[0].ammo_capacity = 0
        model.step()
    assert len(w1) == 1
    assert f"{MockWeaponWithAmmo().name} nas no ammo, thus agent cannot fire" in str(w1[0].message)

    with catch_warnings(record=True) as w2:
        FireBehaviour.fire_at_target(
            self=FireBehaviour(), firer=firer, weapon=firer.weapons[0], location=(1, 1)
        )
    assert len(w2) == 1
    assert f"{MockWeaponWithAmmo().name} has no ammo so can not fire." in str(w2[0].message)


def test_target_not_in_range():
    """Test demonstrating ammo capacity will not be reduced if target not in range."""
    model = MockModel2d()
    fire_order_1 = Order(ImmediateTrigger(), FireAtTargetTask(target=(1, 1)), persistent=True)
    weapon = MockWeaponWithAmmo(max_range=1)
    firer = RetAgent(
        affiliation=Affiliation.NEUTRAL,
        pos=(0, 0),
        name="Firer",
        model=model,
        critical_dimension=1.0,
        behaviours=[FireBehaviour()],
        weapons=[weapon],
        orders=[fire_order_1],
    )
    original_ammo = firer.weapons[0].ammo_capacity

    with catch_warnings(record=True) as w1:
        FireBehaviour.fire_at_target(
            self=FireBehaviour, firer=firer, weapon=weapon, location=(1000, 1000)
        )
    assert (
        "'Firer' attempting to fire 'Mock Weapon with ammo' weapon at target beyond maximum range"
        in str(w1[0].message)
    )

    assert original_ammo == firer.weapons[0].ammo_capacity


def test_more_rounds_than_ammo_warning():
    """Test demonstrating warning in fire_at_target if weapon has less ammo than rounds."""
    model = MockModel2d()
    fire_order_1 = Order(ImmediateTrigger(), FireAtTargetTask(target=(1, 1)), persistent=True)
    firer = RetAgent(
        affiliation=Affiliation.NEUTRAL,
        pos=(0, 0),
        name="Firer",
        model=model,
        critical_dimension=1.0,
        behaviours=[FireBehaviour()],
        weapons=[MockWeaponWithAmmo()],
        orders=[fire_order_1],
    )
    firer.weapons[0].ammo_capacity = 1

    with catch_warnings(record=True) as w:
        FireBehaviour._step(self=FireBehaviour(), firer=firer, rounds=2, weapon=firer.weapons[0])
    assert len(w) == 1
    assert (
        f"Number of ammo on {firer.weapons[0].name} ({firer.weapons[0].ammo_capacity}) is less than"
        + f" the number of rounds ({2}) to fire. Will fire {firer.weapons[0].ammo_capacity}"
        + " rounds instead."
        in str(w[0].message)
    )


class TestIncorrectAmmoType(unittest.TestCase):
    """Unit test Class for demonstrating weapon will only take positive int as ammo_capacity."""

    @parameterized.expand([[2.5], ["5 bullets"]])
    def test_wrong_ammo_type(self, incorrect_ammo):
        """Test demonstrating weapon will not take float or string as ammo_capacity."""
        fire_order_1 = Order(ImmediateTrigger(), FireAtTargetTask(target=(1, 1)), persistent=True)
        model = MockModel2d()
        firer = RetAgent(
            affiliation=Affiliation.NEUTRAL,
            pos=(0, 0),
            name="Firer",
            model=model,
            critical_dimension=1.0,
            behaviours=[FireBehaviour()],
            weapons=[MockWeaponWithAmmo()],
            orders=[fire_order_1],
        )

        with self.assertRaises(ValueError) as e:
            firer.weapons[0].ammo_capacity = incorrect_ammo
        self.assertEqual(
            f"Expected ammo_capacity to be an int or None, got {type(incorrect_ammo)} instead.",
            str(e.exception),
        )


class TestNegativeAmmoType(unittest.TestCase):
    """Unit test Class for demonstrating weapon will only take positive int as ammo_capacity."""

    @parameterized.expand([[-1]])
    def test_negative_ammo_type(self, incorrect_ammo):
        """Test demonstrating weapon will not take negatives as ammo_capacity."""
        fire_order_1 = Order(ImmediateTrigger(), FireAtTargetTask(target=(1, 1)), persistent=True)
        model = MockModel2d()
        firer = RetAgent(
            affiliation=Affiliation.NEUTRAL,
            pos=(0, 0),
            name="Firer",
            model=model,
            critical_dimension=1.0,
            behaviours=[FireBehaviour()],
            weapons=[MockWeaponWithAmmo()],
            orders=[fire_order_1],
        )

        with self.assertRaises(ValueError) as e:
            firer.weapons[0].ammo_capacity = incorrect_ammo
        self.assertEqual(
            f"Expected ammo_capacity to be a positive int or None, got {incorrect_ammo} instead.",
            str(e.exception),
        )
