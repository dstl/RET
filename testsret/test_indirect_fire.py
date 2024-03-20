"""Test cases demonstrating indirect fire capability."""

from datetime import timedelta

from ret.agents.affiliation import Affiliation
from ret.agents.agent import RetAgent
from ret.behaviours.fire import FireBehaviour
from ret.orders.order import Order
from ret.orders.tasks.fire import FireAtTargetTask
from ret.orders.triggers.immediate import ImmediateTrigger
from ret.testing.mocks import MockModel2d
from ret.weapons.weapon import BasicWeapon


def test_indirect_fire():
    """This test demonstrates the capability of firing indirectly.

    This test is set up such that the firer has no sensors, and therefore has no way of targetting
    anything other than blindly firing at a location (e.g., indirect fire).
    """
    fire_order_1 = Order(
        ImmediateTrigger(), FireAtTargetTask(target=(27.5, 27.5)), persistent=False, priority=3
    )
    fire_order_2 = Order(
        ImmediateTrigger(), FireAtTargetTask(target=(40, 40)), persistent=False, priority=2
    )
    fire_order_3 = Order(
        ImmediateTrigger(), FireAtTargetTask(target=(52, 52)), persistent=False, priority=1
    )

    model = MockModel2d()
    RetAgent(
        affiliation=Affiliation.FRIENDLY,
        pos=(0, 0),
        name="Firer",
        model=model,
        critical_dimension=1.0,
        reflectivity=0.1,
        temperature=20,
        behaviours=[FireBehaviour()],
        weapons=[
            BasicWeapon(
                name="Mortar",
                radius=10,
                time_between_rounds=timedelta(minutes=1),
                time_before_first_shot=timedelta(seconds=0),
                kill_probability_per_round=1,
            )
        ],
        orders=[fire_order_3, fire_order_2, fire_order_1],
    )

    target1 = RetAgent(
        affiliation=Affiliation.HOSTILE,
        pos=(30, 30),
        name="Target 1",
        model=model,
        critical_dimension=1.0,
        reflectivity=0.1,
        temperature=20,
    )

    target2 = RetAgent(
        affiliation=Affiliation.FRIENDLY,
        pos=(50, 50),
        name="Target 1",
        model=model,
        critical_dimension=1.0,
        reflectivity=0.1,
        temperature=20,
    )

    target3 = RetAgent(
        affiliation=Affiliation.NEUTRAL,
        pos=(50, 50),
        name="Target 1",
        model=model,
        critical_dimension=1.0,
        reflectivity=0.1,
        temperature=20,
    )

    # Before starting to fire, both targets are alive
    assert target1.killed is False
    assert target2.killed is False
    assert target3.killed is False

    model.step()

    # first indirect fire order is at location that is within the kill radius for target1
    assert target1.killed is True
    assert target2.killed is False
    assert target3.killed is False

    model.step()

    # second indirect fire order misses both targets, so no further kills
    assert target1.killed is True
    assert target2.killed is False
    assert target3.killed is False

    model.step()

    # third indirect fire order hits target2 and target3 in single shot
    assert target1.killed is True
    assert target2.killed is True
    assert target3.killed is True
