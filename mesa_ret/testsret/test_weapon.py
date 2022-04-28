"""Tests for weapons."""
from __future__ import annotations

import math
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING
from warnings import catch_warnings

from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent
from mesa_ret.behaviours.fire import FireBehaviour
from mesa_ret.testing.mocks import MockModel2d, MockShotFiredLogger
from mesa_ret.weapons.weapon import BasicLongRangedWeapon, BasicShortRangedWeapon, BasicWeapon
from pytest import mark, raises

if TYPE_CHECKING:
    from mesa.space import GridContent
    from mesa_ret.model import RetModel


def create_agent(model: RetModel) -> RetAgent:
    """Create a Killable RET agent.

    Args:
        model (RetModel): Model to put the agent in

    Returns:
        RetAgent: The newly created agent
    """
    return RetAgent(
        model=model,
        pos=(0, 0),
        name="Killable",
        affiliation=Affiliation.NEUTRAL,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
    )


@mark.parametrize("kill_probability,killed", [[0.0, False], [1.0, True]])
def test_weapon_guaranteed_kill_probability(kill_probability: float, killed: bool):
    """Test that a weapon with kill probability will always kill.

    Args:
        kill_probability (float): Probability of the weapon killing the target
        killed (bool): Whether or not the target will be killed
    """
    weapon = BasicWeapon(
        name="Weapon",
        radius=1.0,
        time_before_first_shot=timedelta(seconds=0),
        time_between_rounds=timedelta(seconds=30),
        kill_probability_per_round=kill_probability,
    )

    model = MockModel2d()

    killable = create_agent(model)
    firer = create_agent(model)

    weapon.try_kill(firer=firer, target=killable, shot_id=1)

    assert killable.killed is killed


def test_weapon_fire_at_non_target():
    """Test error handling where weapon fires at something that cannot be killed."""
    weapon = BasicWeapon(
        name="Weapon",
        radius=1.0,
        time_before_first_shot=timedelta(seconds=0),
        time_between_rounds=timedelta(seconds=30),
        kill_probability_per_round=1,
    )

    model = MockModel2d()

    target: GridContent = None  # type: ignore
    firer = create_agent(model)

    with raises(AttributeError) as e:
        weapon.try_kill(firer=firer, target=target, shot_id=1)

    assert e.value.args[0] == "Target is not a RetAgent, or doesn't have a 'kill' method."


@mark.parametrize("radius", [0, -0.001, -10])
def test_invalid_weapon_radius(radius: float):
    """Test error handling for invalid weapon radius.

    Args:
        radius (float): Invalid weapon radius
    """
    with raises(ValueError) as e:
        BasicWeapon(
            name="test",
            radius=radius,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=0.5,
        )

    assert e.value.args[0] == "Weapon kill radius must be a positive value"


@mark.parametrize("seconds", [0, -1, -1000])
def test_invalid_time_between_shots(seconds: int):
    """Test error handling for invalid time between shots.

    Args:
        seconds(int): invalid number of seconds between shots.
    """
    with raises(ValueError) as e:
        BasicWeapon(
            name="test",
            radius=1,
            time_between_rounds=timedelta(seconds=seconds),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=0.5,
        )
    assert e.value.args[0] == "Time between rounds must be a positive value"


@mark.parametrize("seconds", [-1, -1000])
def test_invalid_time_before_first_shot(seconds: int):
    """Test error handling for invalid time before first shot.

    Args:
        seconds (int): Invalid time (in seconds) before the first shot is taken.
    """
    with raises(ValueError) as e:
        BasicWeapon(
            name="test",
            radius=1.0,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=seconds),
            kill_probability_per_round=0.5,
        )
    assert e.value.args[0] == "Time before first shot must be a non-negative timedelta"


@mark.parametrize("probability", [-0.0001, 1.00001])
def test_invalid_kill_probability(probability: float):
    """Test error handling for invalid kill probability.

    Args:
        probability (float): Invalid kill probability

    """
    with raises(ValueError) as e:
        BasicWeapon(
            name="test",
            radius=1.0,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=probability,
        )
    assert e.value.args[0] == "Kill probability per round must be in the range [0, 1]"


def test_invalid_min_range():
    """Test case of invalid minimum range."""
    with raises(ValueError) as e:
        BasicWeapon(
            name="test",
            radius=1.0,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=0.5,
            min_range=-1,
        )

    assert e.value.args[0] == "Minimum range must be a non-negative value"


def test_invalid_max_range():
    """Test cases of invalid maximum range."""
    with raises(ValueError) as e:
        BasicWeapon(
            name="test",
            radius=1.0,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=0.5,
            max_range=-1,
        )

    assert e.value.args[0] == "Maximum range must be a positive value"


def test_min_range_greater_than_max_range():
    """Test handling of case where min range is set greater than max range."""
    with raises(ValueError) as e:
        BasicWeapon(
            name="test",
            radius=1.0,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=0.5,
            min_range=5,
            max_range=4,
        )

    assert e.value.args[0] == "Minimum range cannot be greater than maximum range"


def test_ignoring_fire_at_out_of_range_target():
    """Test firing at a target that is outside allowable range.

    The weapon used here has a kill probability of 100% so that it would always kill if in range.
    """
    weapon = BasicWeapon(
        name="Always Kill",
        radius=1.0,
        time_between_rounds=timedelta(seconds=1),
        time_before_first_shot=timedelta(seconds=0),
        kill_probability_per_round=1,
        min_range=3,
        max_range=20,
    )

    model = MockModel2d()
    firer = RetAgent(
        name="firer",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        weapons=[weapon],
        behaviours=[FireBehaviour()],
    )

    too_close = RetAgent(
        name="too close",
        model=model,
        pos=(1, 1),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
    )
    in_range = RetAgent(
        name="in range",
        model=model,
        pos=(4, 4),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
    )
    too_far_away = RetAgent(
        name="too far away",
        model=model,
        pos=(20, 20),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
    )

    with catch_warnings(record=True) as w1:
        firer.fire_step(1, weapon, (1, 1))

    assert too_close.killed is False
    assert len(w1) == 1
    assert "attempting to fire 'Always Kill' weapon at target closer than minimum range." in str(
        w1[0].message
    )

    firer.fire_step(1, weapon, (4, 4))
    assert in_range.killed is True

    with catch_warnings(record=True) as w2:
        firer.fire_step(1, weapon, (20, 20))
    assert too_far_away.killed is False
    assert len(w2) == 1
    assert "attempting to fire 'Always Kill' weapon at target beyond maximum range." in str(
        w2[0].message
    )


def test_long_ranged_weapon_within_range():
    """Test whether all of the shots from the long range weapon are within expected range."""
    weapon = BasicLongRangedWeapon(
        name="Long range weapon",
        radius=1.0,
        time_between_rounds=timedelta(seconds=1),
        time_before_first_shot=timedelta(seconds=0),
        kill_probability_per_round=1,
        max_percentage_inaccuracy=10,
    )

    model = MockModel2d()
    model.logger = MockShotFiredLogger(model)

    firer = RetAgent(
        name="firer",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        weapons=[weapon],
        behaviours=[FireBehaviour()],
    )

    shots: list[tuple[float, float]] = []
    for _ in range(250):
        firer.fire_step(1, weapon, (200, 200))
        shots.append(model.logger.recorded_firing["aim location"])

    for i in range(250):
        # Initial shot is towards location (200, 200) so a an offset of up to 10 percent  in either
        # direction means the final shot will be within a box from (180, 180) to (220, 220).
        assert 200 - (10 * 200 * math.sqrt(2)) <= shots[i][0] <= 200 + (10 * 200 * math.sqrt(2))
        assert 200 - (10 * 200 * math.sqrt(2)) <= shots[i][1] <= 200 + (10 * 200 * math.sqrt(2))


def test_long_ranged_weapon_different_offsets():
    """Test whether multiple shots from the long range weapon land in different places."""
    weapon = BasicLongRangedWeapon(
        name="Long range weapon",
        radius=1.0,
        time_between_rounds=timedelta(seconds=1),
        time_before_first_shot=timedelta(seconds=0),
        kill_probability_per_round=1,
        max_percentage_inaccuracy=10,
    )

    model = MockModel2d()
    model.logger = MockShotFiredLogger(model)

    firer = RetAgent(
        name="firer",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        weapons=[weapon],
        behaviours=[FireBehaviour()],
    )

    firer.fire_step(1, weapon, (200, 200))
    shot_1 = model.logger.recorded_firing["aim location"]
    firer.fire_step(1, weapon, (200, 200))
    shot_2 = model.logger.recorded_firing["aim location"]
    firer.fire_step(1, weapon, (200, 200))
    shot_3 = model.logger.recorded_firing["aim location"]

    assert shot_1 != shot_2 != shot_3 != (200, 200)


def test_long_ranged_weapon_firing_accuracy():
    """Test whether zero inaccuracy shots from the long range weapon land in original location."""
    weapon = BasicLongRangedWeapon(
        name="Long range weapon",
        radius=1.0,
        time_between_rounds=timedelta(seconds=1),
        time_before_first_shot=timedelta(seconds=0),
        kill_probability_per_round=1,
    )

    model = MockModel2d()
    model.logger = MockShotFiredLogger(model)

    firer = RetAgent(
        name="firer",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        weapons=[weapon],
        behaviours=[FireBehaviour()],
    )

    weapon._max_percentage_inaccuracy = 0.1
    firer.fire_step(1, weapon, (200, 200))
    shot_1 = model.logger.recorded_firing["aim location"]
    assert shot_1 != (200, 200)

    weapon._max_percentage_inaccuracy = 0
    firer.fire_step(1, weapon, (200, 200))
    shot_2 = model.logger.recorded_firing["aim location"]
    assert shot_2 == (200, 200)


def test_long_ranged_weapon_invalid_accuracy():
    """Test whether invalid inaccuracies result in the correct warning."""
    with warnings.catch_warnings(record=True) as w:
        weapon = BasicLongRangedWeapon(
            name="Long range weapon",
            radius=1.0,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=1,
            max_percentage_inaccuracy=-0.1,
        )

        model = MockModel2d()
        model.logger = MockShotFiredLogger(model)

        firer = RetAgent(
            name="firer",
            model=model,
            pos=(0, 0),
            affiliation=Affiliation.UNKNOWN,
            critical_dimension=1.0,
            reflectivity=1.0,
            temperature=1.0,
            temperature_std_dev=1.0,
            weapons=[weapon],
            behaviours=[FireBehaviour()],
        )
        firer.fire_step(1, weapon, (200, 200))
    assert len(w) == 1
    assert (
        "maximum percentage of inaccuracy (given as a decimal) for weapon 'Long range weapon' must"
        " be 0 or greater."
        " A percentage of -0.1 was specified."
        " A value of 0% will be used instead."
    ) in str(w[0].message)

    assert model.logger.recorded_firing["aim location"] == (200, 200)


def test_short_ranged_weapon_within_range():
    """Test whether all of the shots from the short range weapon are within expected range."""
    weapon = BasicShortRangedWeapon(
        name="Short range weapon",
        radius=1.0,
        time_between_rounds=timedelta(seconds=1),
        time_before_first_shot=timedelta(seconds=0),
        kill_probability_per_round=1,
        max_angle_inaccuracy=45,
    )

    model = MockModel2d()
    model.logger = MockShotFiredLogger(model)

    firer = RetAgent(
        name="firer",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        weapons=[weapon],
        behaviours=[FireBehaviour()],
    )

    shots: list[tuple[float, float]] = []
    for _ in range(250):
        firer.fire_step(1, weapon, (200, 200))
        shots.append(model.logger.recorded_firing["aim location"])

    for i in range(250):
        # Initial shot is towards location (200, 200) so a an offset of up to 45 degrees means the
        # final shot will be within a quarter circle in the positive x-y quadrant.
        assert 0 <= shots[i][0] <= (200 * math.sqrt(2))
        assert 0 <= shots[i][1] <= (200 * math.sqrt(2))
        assert math.isclose((200 * math.sqrt(2)), math.sqrt(shots[i][0] ** 2 + shots[i][1] ** 2))


def test_short_ranged_weapon_different_offsets():
    """Test whether multiple shots from the short range weapon land in different places."""
    weapon = BasicShortRangedWeapon(
        name="Short range weapon",
        radius=1.0,
        time_between_rounds=timedelta(seconds=1),
        time_before_first_shot=timedelta(seconds=0),
        kill_probability_per_round=1,
        max_angle_inaccuracy=45,
    )

    model = MockModel2d()
    model.logger = MockShotFiredLogger(model)

    firer = RetAgent(
        name="firer",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        weapons=[weapon],
        behaviours=[FireBehaviour()],
    )

    firer.fire_step(1, weapon, (200, 200))
    shot_1 = model.logger.recorded_firing["aim location"]
    firer.fire_step(1, weapon, (200, 200))
    shot_2 = model.logger.recorded_firing["aim location"]
    firer.fire_step(1, weapon, (200, 200))
    shot_3 = model.logger.recorded_firing["aim location"]

    assert shot_1 != shot_2 != shot_3 != (200, 200)


def test_short_ranged_weapon_firing_accuracy():
    """Test whether zero inaccuracy shots from the short range weapon land in original location."""
    weapon = BasicShortRangedWeapon(
        name="Short range weapon",
        radius=1.0,
        time_between_rounds=timedelta(seconds=1),
        time_before_first_shot=timedelta(seconds=0),
        kill_probability_per_round=1,
    )

    model = MockModel2d()
    model.logger = MockShotFiredLogger(model)

    firer = RetAgent(
        name="firer",
        model=model,
        pos=(0, 0),
        affiliation=Affiliation.UNKNOWN,
        critical_dimension=1.0,
        reflectivity=1.0,
        temperature=1.0,
        temperature_std_dev=1.0,
        weapons=[weapon],
        behaviours=[FireBehaviour()],
    )

    weapon._max_angle_inaccuracy = 45
    firer.fire_step(1, weapon, (200, 200))
    shot_1 = model.logger.recorded_firing["aim location"]
    assert shot_1 != (200, 200)

    weapon._max_angle_inaccuracy = 0
    firer.fire_step(1, weapon, (200, 200))
    shot_2 = model.logger.recorded_firing["aim location"]
    assert shot_2 == (200, 200)


def test_short_ranged_weapon_invalid_accuracy():
    """Test whether invalid inaccuracies result in the correct warning."""
    with warnings.catch_warnings(record=True) as w:
        weapon = BasicShortRangedWeapon(
            name="Short range weapon",
            radius=1.0,
            time_between_rounds=timedelta(seconds=1),
            time_before_first_shot=timedelta(seconds=0),
            kill_probability_per_round=1,
            max_angle_inaccuracy=-45,
        )

        model = MockModel2d()
        model.logger = MockShotFiredLogger(model)

        firer = RetAgent(
            name="firer",
            model=model,
            pos=(0, 0),
            affiliation=Affiliation.UNKNOWN,
            critical_dimension=1.0,
            reflectivity=1.0,
            temperature=1.0,
            temperature_std_dev=1.0,
            weapons=[weapon],
            behaviours=[FireBehaviour()],
        )

        firer.fire_step(1, weapon, (200, 200))
    assert len(w) == 1
    assert (
        "maximum angle of inaccuracy for weapon 'Short range weapon' must be within range 0-180."
        " An angle of -45 degrees was specified."
        " An angle of 0 degrees will be used instead."
    ) in str(w[0].message)

    assert model.logger.recorded_firing["aim location"] == (200, 200)
