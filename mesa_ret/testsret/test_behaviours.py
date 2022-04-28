"""Tests for ret agent behaviours."""

from __future__ import annotations

import math
import random
import typing
import unittest
import warnings
from datetime import datetime, timedelta
from math import inf
from typing import TYPE_CHECKING
from warnings import catch_warnings

import numpy as np
from mesa_ret.agents.affiliation import Affiliation
from mesa_ret.agents.agent import RetAgent, WeaponFiredStatus
from mesa_ret.agents.airagent import AirAgent
from mesa_ret.behaviours.communicate import (
    CommunicateMissionMessageBehaviour,
    CommunicateOrdersBehaviour,
    CommunicateWorldviewBehaviour,
)
from mesa_ret.behaviours.deploycountermeasure import DeployCountermeasureBehaviour
from mesa_ret.behaviours.fire import FireBehaviour, HostileTargetResolver, TargetSelector
from mesa_ret.behaviours.hide import HideBehaviour
from mesa_ret.behaviours.move import AircraftMoveBehaviour, GroundBasedMoveBehaviour, MoveBehaviour
from mesa_ret.behaviours.sense import SenseBehaviour
from mesa_ret.behaviours.wait import WaitBehaviour
from mesa_ret.sensing.agentcasualtystate import AgentCasualtyState
from mesa_ret.sensing.perceivedworld import (
    Confidence,
    PerceivedAgent,
    RecognisedAgents,
    UnknownAgents,
)
from mesa_ret.space.culture import Culture, default_culture
from mesa_ret.space.feature import BoxFeature, LineFeature
from mesa_ret.space.heightband import AbsoluteHeightBand
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture2d
from mesa_ret.testing.mocks import (
    MockCountermeasure,
    MockFireBehaviour,
    MockModel,
    MockModel2d,
    MockModel3d,
    MockOrderWithId,
    MockSensor,
    MockShortRangedWeapon,
    MockShotFiredLogger,
    MockWeapon,
)
from mesa_ret.types import Coordinate2dOr3d
from parameterized import parameterized

if TYPE_CHECKING:
    from mesa_ret.orders.order import Order
    from mesa_ret.space.heightband import HeightBand
    from mesa_ret.template import Template

culture_red = Culture("red")
culture_green = Culture("green")
culture_blue = Culture("blue")
culture_yellow = Culture("yellow")


class TestBehaviours(unittest.TestCase):
    """Tests for ground based agent move and wait behaviours."""

    base_speed = 1

    gradient_speed_modifiers = [
        ((-inf, -10), 0.8),
        ((-10, 10), 1),
        ((10, inf), 0.8),
    ]

    culture_speed_modifiers = {
        culture_red: 0.5,
        culture_green: 1,
        culture_blue: 2,
        culture_yellow: 0.5,
        default_culture(): 1,
    }

    def setUp(self):
        """Set up test cases."""
        self.model2d = MockModel2d(time_step=timedelta(minutes=1))
        self.agent2d = RetAgent(
            model=self.model2d,
            pos=(0, 0),
            name="2D agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.model3d = MockModel3d(time_step=timedelta(minutes=1))
        self.agent3d = RetAgent(
            model=self.model3d,
            pos=(0, 0, 0),
            name="3D agent",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

    def test_ground_based_move_behaviour_step_x(self):
        """Test ground-based movement in the X-direction."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, self.culture_speed_modifiers
        )

        behaviour.step(self.agent2d, (100, 0))
        assert self.agent2d.pos is not None
        assert self.agent2d.pos[0] == self.base_speed * 1 * self.model2d.time_step.seconds
        assert self.agent2d.pos[1] == 0

        behaviour.step(self.agent2d, (100, 0))
        assert self.agent2d.pos[0] == 100
        assert self.agent2d.pos[1] == 0

        behaviour.step(self.agent2d, (100, 0))
        assert self.agent2d.pos[0] == 100
        assert self.agent2d.pos[1] == 0

        behaviour.step(self.agent2d, (0, 0))
        assert self.agent2d.pos[0] == 100 - (self.base_speed * 1 * self.model2d.time_step.seconds)
        assert self.agent2d.pos[1] == 0

    def test_ground_based_move_behaviour_step_y(self):
        """Test ground based movement in the Y-direction."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, self.culture_speed_modifiers
        )

        behaviour.step(self.agent2d, (0, 100))
        assert self.agent2d.pos is not None
        assert self.agent2d.pos[0] == 0
        assert self.agent2d.pos[1] == self.base_speed * 1 * self.model2d.time_step.seconds

        behaviour.step(self.agent2d, (0, 100))
        assert self.agent2d.pos[0] == 0
        assert self.agent2d.pos[1] == 100

        behaviour.step(self.agent2d, (0, 100))
        assert self.agent2d.pos[0] == 0
        assert self.agent2d.pos[1] == 100

        behaviour.step(self.agent2d, (0, 0))
        assert self.agent2d.pos[0] == 0
        assert self.agent2d.pos[1] == 100 - (self.base_speed * 1 * self.model2d.time_step.seconds)

    def test_ground_based_move_behaviour_step_xy(self):
        """Test ground based movement in an XY-direction."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, self.culture_speed_modifiers
        )

        behaviour.step(self.agent2d, (50, 50))
        assert self.agent2d.pos is not None
        assert self.agent2d.pos[0] == (
            self.base_speed * 1 * self.model2d.time_step.seconds
        ) / math.sqrt(2)
        assert self.agent2d.pos[1] == (
            self.base_speed * 1 * self.model2d.time_step.seconds
        ) / math.sqrt(2)

        behaviour.step(self.agent2d, (50, 50))
        assert self.agent2d.pos[0] == 50
        assert self.agent2d.pos[1] == 50

        behaviour.step(self.agent2d, (50, 50))
        assert self.agent2d.pos[0] == 50
        assert self.agent2d.pos[1] == 50

        behaviour.step(self.agent2d, (0, 0))
        assert self.agent2d.pos[0] == 50 - (
            (self.base_speed * 1 * self.model2d.time_step.seconds) / math.sqrt(2)
        )
        assert self.agent2d.pos[1] == 50 - (
            (self.base_speed * 1 * self.model2d.time_step.seconds) / math.sqrt(2)
        )

    def test_ground_based_move_behaviour_get_gradient_modifer(self):
        """Test calculation of gradient-based ground move modifiers."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, self.culture_speed_modifiers
        )

        assert behaviour.get_gradient_modifer(-10) == 0.8
        assert behaviour.get_gradient_modifer(0) == 1
        assert behaviour.get_gradient_modifer(11) == 0.8

    def test_ground_based_move_behaviour_get_culture_modifer(self):
        """Test ground based movement accounting for culture modifiers."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, self.culture_speed_modifiers
        )

        assert behaviour.get_culture_modifier(culture_red) == 0.5
        assert behaviour.get_culture_modifier(culture_green) == 1.0
        assert behaviour.get_culture_modifier(culture_blue) == 2.0
        assert behaviour.get_culture_modifier(culture_yellow) == 0.5
        assert behaviour.get_culture_modifier(default_culture()) == 1

        with self.assertRaises(KeyError) as e:
            behaviour.get_culture_modifier(Culture("###"))
        self.assertEqual(
            "'culture movement modifier not specified for culture: ###'",
            str(e.exception),
        )

    def test_wait_2d(self):
        """Test that a Wait in 2D space doesn't move the agent's position."""
        behaviour = WaitBehaviour()
        pos = tuple(self.agent2d.pos)
        behaviour.step(waiter=self.agent2d)

        assert pos == self.agent2d.pos

    def test_wait_3d(self):
        """Test that a Wait in 3D space doesn't move the agent's position."""
        behaviour = WaitBehaviour()
        pos = tuple(self.agent3d.pos)
        behaviour.step(waiter=self.agent3d)

        assert pos == self.agent3d.pos


class TestFireBehaviours(unittest.TestCase):
    """Test ret agent fire behaviour."""

    class AlwaysOnlyResolveToAffiliationResolver(HostileTargetResolver):
        """Hostile target resolver, which always picks targets by affiliation."""

        def __init__(self, affiliation: Affiliation) -> None:
            """Create a new AlwaysOnlyResolveToAffiliateResolver.

            Args:
                affiliation (Affiliation): Affiliation to always resolve
            """
            self.affiliation = affiliation

        def run(self, detector: RetAgent, views: list[PerceivedAgent]) -> list[PerceivedAgent]:
            """Filter views.

            Args:
                detector (RetAgent): Agent doing the firing
                views (list[PerceivedAgent]): Unfiltered views

            Returns:
                list[PerceivedAgent]: Filtered views
            """
            return [r for r in views if r.affiliation == self.affiliation]

    class AlwayPickTargetAtLocation(TargetSelector):
        """Location-based target selector, which always picks targets at a location."""

        def __init__(self, location: Coordinate2dOr3d) -> None:
            """Create a new AlwaysPickTargetAtLocation.

            Args:
                location (Coordinate2dOr3d): Location to pick targets at.
            """
            self.location = location

        def run(self, detector: RetAgent, views: list[PerceivedAgent]) -> list[Coordinate2dOr3d]:
            """Return list of locations to consider targets.

            Args:
                detector (RetAgent): Agent doing the targetting
                views (list[PerceivedAgent]): List of candidate views

            Returns:
                list[Coordinate2dOr3d]: Filtered list of locations to fire at
            """
            return [r.location for r in views if r.location == self.location]

    def create_model(self) -> MockModel:
        """Return model.

        Returns:
            MockModel: 2D model
        """
        return MockModel2d()

    def get_agent_1_location(self) -> Coordinate2dOr3d:
        """Return coordinate of agent 1.

        Returns:
            Coordinate2dOr3d: 2D coordinate at (0, 0)
        """
        return (0, 0)

    def get_agent_2_location(self) -> Coordinate2dOr3d:
        """Return coordinate of agent 2.

        Returns:
            Coordinate2dOr3d: 2D coordinate at (10), 10)
        """
        return (10, 10)

    def get_target(self) -> Coordinate2dOr3d:
        """Return coordinate of target.

        Returns:
            Coordinate2dOr3d: 2D coordinate at (9, 9)
        """
        return (9, 9)

    def setUp(self):
        """Set up test cases."""
        self.model = self.create_model()
        self.agent1 = RetAgent(
            model=self.model,
            pos=self.get_agent_1_location(),
            name="agent 1",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.agent2 = RetAgent(
            model=self.model,
            pos=self.get_agent_2_location(),
            name="agent 2",
            affiliation=Affiliation.UNKNOWN,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.target = self.get_target()

    def test_fire(self):
        """Test that firing directly upon an agent kills it."""
        behaviour = MockFireBehaviour()
        self.assertFalse(self.agent2.killed)
        weapon = MockWeapon(radius=1, kill_probability_per_round=1)
        behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.agent2.pos)
        self.assertTrue(self.agent2.killed)

    def test_fire_within_radius(self):
        """Test that firing at an agent within radius of fire kills it."""
        behaviour = MockFireBehaviour()
        self.assertFalse(self.agent2.killed)
        weapon = MockWeapon(radius=5, kill_probability_per_round=1)
        behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.target)
        self.assertTrue(self.agent2.killed)

    def test_fire_outside_radius(self):
        """Test that firing at an agent outside radius doesn't kill it."""
        behaviour = MockFireBehaviour()
        self.assertFalse(self.agent2.killed)
        weapon = MockWeapon(radius=0.25, kill_probability_per_round=1)
        behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.target)
        self.assertFalse(self.agent2.killed)

    def test_cannot_fire_at_self(self):
        """Test that it's not possible to fire at ones self."""
        behaviour = MockFireBehaviour()
        self.assertFalse(self.agent1.killed)
        weapon = MockWeapon(radius=1, kill_probability_per_round=1)
        behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.agent1.pos)
        self.assertFalse(self.agent1.killed)

    def test_fire_without_target(self):
        """Assert that where no location is provided, fire behaviour will choose.

        Test that agent1 fires at a location that kills agent 2, if a hostile bogey
        is added to their perceived world at the same position as agent 2.
        """
        behaviour = MockFireBehaviour()
        location = self.get_agent_2_location()
        self.agent1.perceived_world.add_acquisitions(
            PerceivedAgent(
                unique_id=self.model.next_id(),
                sense_time=self.model.get_time(),
                location=location,
                affiliation=Affiliation.HOSTILE,
                confidence=Confidence.IDENTIFY,
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )
        self.assertFalse(self.agent2.killed)
        weapon = MockWeapon(radius=10, kill_probability_per_round=1)
        behaviour.step(firer=self.agent1, weapon=weapon, rounds=1)
        self.assertTrue(self.agent2.killed)

    def test_fire_behaviour_alternative_hostile_resolver(self):
        """Assert that the fire behaviour's hostile target resolver can be set."""
        behaviour_fire_at_hostile = MockFireBehaviour(
            hostile_agent_resolver=self.AlwaysOnlyResolveToAffiliationResolver(
                affiliation=Affiliation.HOSTILE
            ),
        )
        behaviour_fire_at_neutral = MockFireBehaviour(
            hostile_agent_resolver=self.AlwaysOnlyResolveToAffiliationResolver(
                affiliation=Affiliation.NEUTRAL
            ),
        )

        self.agent1.perceived_world.add_acquisitions(
            PerceivedAgent(
                sense_time=self.model.get_time(),
                affiliation=Affiliation.NEUTRAL,
                location=self.agent2.pos,
                unique_id=self.agent2.unique_id,
                confidence=Confidence.IDENTIFY,
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )

        w1 = MockWeapon(radius=1, kill_probability_per_round=1)

        w2 = MockWeapon(
            radius=1,
            kill_probability_per_round=1,
        )

        self.assertFalse(self.agent2.killed)
        behaviour_fire_at_hostile.step(firer=self.agent1, weapon=w2, rounds=1)
        self.assertFalse(self.agent2.killed)
        behaviour_fire_at_neutral.step(firer=self.agent1, weapon=w1, rounds=1)
        self.assertTrue(self.agent2.killed)

    def test_fire_behaviour_alternative_target_selector(self):
        """Assert that the fire behaviour's target selector can be set."""
        behaviour_fire_at_wrong_location = MockFireBehaviour(
            target_selector=self.AlwayPickTargetAtLocation(location=(100, 100)),
        )

        weapon = MockWeapon(
            radius=1,
            kill_probability_per_round=1,
        )

        behaviour_fire_at_agent2_location = MockFireBehaviour(
            target_selector=self.AlwayPickTargetAtLocation(location=self.agent2.pos),
        )

        self.agent1.perceived_world.add_acquisitions(
            PerceivedAgent(
                unique_id=self.agent2.unique_id,
                sense_time=self.model.get_time(),
                affiliation=Affiliation.HOSTILE,
                location=self.agent2.pos,
                confidence=Confidence.IDENTIFY,
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )

        self.assertFalse(self.agent2.killed)
        behaviour_fire_at_wrong_location.step(firer=self.agent1, weapon=weapon, rounds=1)
        self.assertFalse(self.agent2.killed)
        behaviour_fire_at_agent2_location.step(firer=self.agent1, weapon=weapon, rounds=1)
        self.assertTrue(self.agent2.killed)

    def test_fire_status(self):
        """Test that firing weapon updates agent status."""
        behaviour = MockFireBehaviour()
        weapon = MockWeapon(radius=1, kill_probability_per_round=1)
        behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.agent2.pos)
        statuses = self.agent1._statuses
        weapon_fired_statuses = [
            status for status in statuses if isinstance(status, WeaponFiredStatus)
        ]
        assert len(weapon_fired_statuses) == 1
        assert weapon_fired_statuses[0].target_location == self.agent2.pos
        assert weapon_fired_statuses[0].weapon_name == weapon.name
        assert weapon_fired_statuses[0].target_in_range
        assert weapon_fired_statuses[0].time_step == self.agent1.model.get_time()

    def test_fire_status_target_out_of_range(self):
        """Test that firing weapon updates agent status."""
        with warnings.catch_warnings(record=True) as w:
            behaviour = MockFireBehaviour()
            weapon = MockWeapon(radius=1, kill_probability_per_round=1, max_range=5)
            behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.agent2.pos)
            statuses = self.agent1._statuses
            weapon_fired_statuses = [
                status for status in statuses if isinstance(status, WeaponFiredStatus)
            ]
        assert len(w) == 1
        assert (
            f"'{self.agent1.name}' attempting to fire '{weapon.name}' weapon at target beyond "
            + f"maximum range. Firer position = {self.agent1.pos}, Target Location "
            + f"= {self.agent2.pos}, Max range = {weapon._max_range}"
            in str(w[0].message)
        )
        assert len(weapon_fired_statuses) == 1
        assert weapon_fired_statuses[0].target_location == self.agent2.pos
        assert weapon_fired_statuses[0].weapon_name == weapon.name
        assert not weapon_fired_statuses[0].target_in_range
        assert weapon_fired_statuses[0].time_step == self.agent1.model.get_time()

    def test_multiple_fire_status(self):
        """Test that firing weapon multiple times updates agent status."""
        behaviour = MockFireBehaviour()
        weapon = MockWeapon(radius=1, kill_probability_per_round=1)
        behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.agent2.pos)
        behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=self.agent2.pos)
        statuses = self.agent1._statuses
        weapon_fired_statuses = [
            status for status in statuses if isinstance(status, WeaponFiredStatus)
        ]
        assert len(weapon_fired_statuses) == 2
        assert weapon_fired_statuses[0].target_location == self.agent2.pos
        assert weapon_fired_statuses[0].weapon_name == weapon.name
        assert weapon_fired_statuses[0].target_in_range
        assert weapon_fired_statuses[0].time_step == self.agent1.model.get_time()
        assert weapon_fired_statuses[1].target_location == self.agent2.pos
        assert weapon_fired_statuses[1].weapon_name == weapon.name
        assert weapon_fired_statuses[1].target_in_range
        assert weapon_fired_statuses[1].time_step == self.agent1.model.get_time()

    def test_multiple_locations_given(self):
        """Test that the behaviour correctly handles being given multiple target locations."""
        with catch_warnings(record=True) as w:
            self.agent3 = RetAgent(
                model=self.model,
                pos=(5, 5),
                name="agent 3",
                affiliation=Affiliation.UNKNOWN,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            behaviour = MockFireBehaviour()
            weapon = MockWeapon(radius=1, kill_probability_per_round=1)
            locations = [self.agent2.pos, self.agent3.pos]

            self.assertFalse(self.agent2.killed)
            self.assertFalse(self.agent3.killed)
            behaviour.step(firer=self.agent1, rounds=2, weapon=weapon, location=locations)
        if isinstance(self.model, MockModel3d):
            assert len(w) == 1
            assert (
                "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
                in str(w[0].message)
            )
        else:
            assert len(w) == 0
        assert self.agent2.killed
        assert self.agent3.killed

    def test_rounds_different_to_locations(self):
        """Test that a warning is raised if the number of rounds does not match locations."""
        with warnings.catch_warnings(record=True) as w:
            self.agent3 = RetAgent(
                model=self.model,
                pos=(5, 5),
                name="agent 3",
                affiliation=Affiliation.UNKNOWN,
                critical_dimension=2.0,
                reflectivity=0.1,
                temperature=20.0,
            )

            behaviour = MockFireBehaviour()
            weapon = MockWeapon(radius=1, kill_probability_per_round=1)
            locations = [self.agent2.pos, self.agent3.pos]

            behaviour.step(firer=self.agent1, rounds=1, weapon=weapon, location=locations)
        if isinstance(self.model, MockModel3d):
            assert len(w) == 2
            assert (
                "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
            ) in str(w[0].message)
            assert (
                "The number of locations given is not equal to the number of rounds"
                + "to fire. Will fire one round at each location instead."
            ) in str(w[1].message)
        else:
            assert len(w) == 1
            assert (
                "The number of locations given is not equal to the number of rounds"
                + "to fire. Will fire one round at each location instead."
            ) in str(w[0].message)


class TestFireBehaviours3D(TestFireBehaviours):
    """Test ret agent fire behaviour in 3D space.

    This extends the test behaviour for 2D space.
    """

    def create_model(self) -> MockModel:
        """Return a 3D mock model instance.

        Returns:
            MockModel: 3D Model representation
        """
        return MockModel3d()

    def get_agent_1_location(self) -> Coordinate2dOr3d:
        """Return position of agent 1 in 3d space.

        Returns:
            Coordinate2dOr3d: 3D Coordinate at (0, 0, 0)
        """
        return (0, 0, 0)

    def get_agent_2_location(self) -> Coordinate2dOr3d:
        """Return position of agent 2 in 3d space.

        Returns:
            Coordinate2dOr3d: 3D Coordinate at (10, 10, 10)
        """
        return (10, 10, 10)

    def get_target(self) -> Coordinate2dOr3d:
        """Return position of target in 3d space.

        Returns:
            Coordinate2dOr3d: 3D coordinate at (9, 9, 9)
        """
        return (9, 9, 9)

    def test_out_of_blast_in_3d_space(self):
        """Test whether agent is within blast radius where separated on Z-axis."""
        behaviour = MockFireBehaviour()
        self.assertFalse(self.agent2.killed)
        vertical_offset_target = list(self.agent2.pos)
        vertical_offset_target[2] = 0
        weapon = MockWeapon(radius=1, kill_probability_per_round=1)
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=tuple(vertical_offset_target)
        )
        self.assertFalse(self.agent2.killed)


class TestFiringAccuracy2D(TestFireBehaviours):
    """Tests for firing accuracy in 2D."""

    def test_firing_accuracy_2d(self):
        """Test that having an inaccuracy changes the shot location."""
        self.model.logger = MockShotFiredLogger(self.model)
        behaviour = MockFireBehaviour()
        original_firing_location = (5, 5)
        weapon = MockShortRangedWeapon(radius=1, kill_probability_per_round=1)

        behaviour.average_angle_inaccuracy = 10
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        self.assertFalse(
            self.model.logger.recorded_firing["aim location"] == original_firing_location
        )

        behaviour.average_angle_inaccuracy = 0
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        self.assertTrue(
            self.model.logger.recorded_firing["aim location"] == original_firing_location
        )

    def test_repeated_shots_different_2d(self):
        """Test that subsequent shots land in different places to one another."""
        self.model.logger = MockShotFiredLogger(self.model)
        behaviour = MockFireBehaviour()
        original_firing_location = (5, 5)
        weapon = MockShortRangedWeapon(radius=1, kill_probability_per_round=1)

        behaviour.average_angle_inaccuracy = 10
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )

        first_shot = self.model.logger.recorded_firing["aim location"]
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        second_shot = self.model.logger.recorded_firing["aim location"]
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        third_shot = self.model.logger.recorded_firing["aim location"]

        self.assertTrue(first_shot != second_shot != third_shot != original_firing_location)

    def test_all_within_expected_range(self):
        """Test that all shots land within expected range."""
        self.model.logger = MockShotFiredLogger(self.model)
        behaviour = MockFireBehaviour()
        original_firing_location = (5, 5)
        weapon = MockShortRangedWeapon(radius=1, kill_probability_per_round=1)

        behaviour.average_angle_inaccuracy = 15

        shots: list[tuple[float, float]] = []
        for _ in range(250):
            behaviour.step(
                firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
            )
            shots.append(self.model.logger.recorded_firing["aim location"])

        for i in range(250):
            # Initial shot is towards heading (1, 1) so a rotation of up to 45 degrees means the
            # final shot will be in a quarter circle, point at the origin.
            self.assertTrue(0 <= shots[i][0] <= original_firing_location[0] * math.sqrt(2))
            self.assertTrue(0 <= shots[i][1] <= original_firing_location[1] * math.sqrt(2))


class TestFiringAccuracy3D(TestFireBehaviours):
    """Tests for firing accuracy in 3D."""

    def create_model(self) -> MockModel:
        """Return a 3D mock model instance.

        Returns:
            MockModel: 3D Model representation
        """
        return MockModel3d()

    def get_agent_1_location(self) -> Coordinate2dOr3d:
        """Return position of agent 1 in 3d space.

        Returns:
            Coordinate2dOr3d: 3D Coordinate at (0, 0, 0)
        """
        return (0, 0, 0)

    def get_agent_2_location(self) -> Coordinate2dOr3d:
        """Return position of agent 2 in 3d space.

        Returns:
            Coordinate2dOr3d: 3D Coordinate at (10, 10, 10)
        """
        return (10, 10, 10)

    def get_target(self) -> Coordinate2dOr3d:
        """Return position of target in 3d space.

        Returns:
            Coordinate2dOr3d: 3D coordinate at (9, 9, 9)
        """
        return (9, 9, 9)

    def test_firing_accuracy_3d(self):
        """Test that having an inaccuracy changes the shot location."""
        self.model.logger = MockShotFiredLogger(self.model)
        behaviour = MockFireBehaviour()
        original_firing_location = (5, 5, 5)
        weapon = MockShortRangedWeapon(radius=1, kill_probability_per_round=1)

        behaviour.average_angle_inaccuracy = 10
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        self.assertFalse(
            self.model.logger.recorded_firing["aim location"] == original_firing_location
        )

        behaviour.average_angle_inaccuracy = 0
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        self.assertTrue(
            self.model.logger.recorded_firing["aim location"] == original_firing_location
        )

    def test_repeated_shots_different_3d(self):
        """Test that subsequent shots land in different places to one another."""
        self.model.logger = MockShotFiredLogger(self.model)
        behaviour = MockFireBehaviour()
        original_firing_location = (5, 5, 5)
        weapon = MockShortRangedWeapon(radius=1, kill_probability_per_round=1)

        behaviour.average_angle_inaccuracy = 10
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )

        first_shot = self.model.logger.recorded_firing["aim location"]
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        second_shot = self.model.logger.recorded_firing["aim location"]
        behaviour.step(
            firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
        )
        third_shot = self.model.logger.recorded_firing["aim location"]

        self.assertTrue(first_shot != second_shot != third_shot != original_firing_location)

    def test_all_within_expected_range(self):
        """Test that all shots land within expected range."""
        self.model.logger = MockShotFiredLogger(self.model)
        behaviour = MockFireBehaviour()
        original_firing_location = (5, 5, 5)
        weapon = MockShortRangedWeapon(radius=1, kill_probability_per_round=1)

        behaviour.average_angle_inaccuracy = 15

        shots: list[tuple[float, float, float]] = []
        for _ in range(250):
            behaviour.step(
                firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
            )
            shots.append(self.model.logger.recorded_firing["aim location"])

        for i in range(250):
            # Initial shot is towards heading (1, 1) so a rotation of up to 45 degrees means the
            # final shot will be in a quarter circle, point at the origin.
            self.assertTrue(0 <= shots[i][0] <= original_firing_location[0] * math.sqrt(2))
            self.assertTrue(0 <= shots[i][1] <= original_firing_location[1] * math.sqrt(2))
            self.assertTrue(shots[i][2] == 5)

    def test_invalid_accuracy(self):
        """Test that all shots land within expected range."""
        with warnings.catch_warnings(record=True) as w:
            self.model.logger = MockShotFiredLogger(self.model)
            behaviour = FireBehaviour(average_angle_inaccuracy=-15)
            original_firing_location = (5, 5, 5)
            weapon = MockShortRangedWeapon(radius=1, kill_probability_per_round=1)

            behaviour.step(
                firer=self.agent1, rounds=1, weapon=weapon, location=original_firing_location
            )
        assert len(w) == 1
        assert (
            "maximum angle of inaccuracy for firing behaviour must be within range "
            "0-180. An angle of -15 degrees was specified. "
            "An angle of 0 degrees will be used instead."
        ) in str(w[0].message)

        assert self.model.logger.recorded_firing["aim location"] == (5, 5, 5)


class TestAircraftSpecificBehaviours(unittest.TestCase):
    """Tests for aircraft specific behaviours."""

    height_bands: list[HeightBand] = [
        AbsoluteHeightBand("-10m", -10),
        AbsoluteHeightBand("0m", 0),
        AbsoluteHeightBand("5m", 5),
        AbsoluteHeightBand("100m", 100),
    ]

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel3d(time_step=timedelta(minutes=1))
        self.agent = AirAgent(
            model=self.model,
            pos=(0, 0, 0),
            name="3D Agent",
            affiliation=Affiliation.FRIENDLY,
        )

    def test_aircraft_move_behaviour_reaches_dest_3d(self):
        """Test an aircraft moving in 3D space reaches the destination.

        Where an aircraft agent moving in 3D is moving sufficiently fast to reach
        it's destination in a time step, it's position will be at the destination at
        the end of the step.
        """
        behaviour = AircraftMoveBehaviour(base_speed=100, height_bands=self.height_bands)

        behaviour.step(mover=self.agent, destination=(5, 5, "5m"))

        assert self.agent.pos[0] == 5.0
        assert self.agent.pos[1] == 5.0
        assert self.agent.pos[2] == 5.0

    def test_aircraft_move_behaviour_reaches_dest_3d_when_aircraft_initialised_in_2d(
        self,
    ):
        """Test aircraft moving in 3D reaches the destination when initialised in 2D."""
        with catch_warnings(record=True) as w:

            agent = AirAgent(
                model=self.model,
                pos=(0, 0),
                name="3D Agent Initialised in 2D",
                affiliation=Affiliation.FRIENDLY,
            )

        assert len(w) == 1
        assert (
            "2D coordinate was entered in 3D terrain, placing coordinate at terrain level"
            == str(w[0].message)
        )

        behaviour = AircraftMoveBehaviour(base_speed=100, height_bands=self.height_bands)

        behaviour.step(mover=agent, destination=(5, 5, "5m"))

        assert agent.pos[0] == 5.0
        assert agent.pos[1] == 5.0
        assert agent.pos[2] == 5.0

    def test_aircraft_move_behaviour_doesnt_reach_dest_3d(self):
        """Test the ending position of an aircraft moving in 3D space.

        Where an aircraft agent moving in 3D space is not moving sufficiently fast
        to reach it's destination in a time stp, it will move at it's base speed
        directly towards the destination.
        """
        # sqrt(3) used for base speed to simplify calculation of distance per second to
        # x=1.0, y=1.0, z=1.0
        behaviour = AircraftMoveBehaviour(base_speed=math.sqrt(3), height_bands=self.height_bands)

        behaviour.step(mover=self.agent, destination=(100, 100, "100m"))

        assert math.isclose(self.agent.pos[0], 60.0)
        assert math.isclose(self.agent.pos[1], 60.0)
        assert math.isclose(self.agent.pos[2], 100.0)

    def test_aircraft_below_ground_destination(self):
        """Check exception is thrown where an aircraft is routed underground."""
        behaviour = AircraftMoveBehaviour(base_speed=10, height_bands=self.height_bands)

        with self.assertRaises(ValueError) as e:
            behaviour.step(mover=self.agent, destination=(10, 10, "-10m"))
        self.assertEqual("Invalid destination: below ground", str(e.exception))

    def test_aircraft_below_ground_2d_destination(self):
        """Check an aircraft cannot be underground for a 2D destination."""
        behaviour = AircraftMoveBehaviour(base_speed=10, height_bands=self.height_bands)

        assert behaviour.is_below_ground(mover=self.agent, destination=(10, 10)) is False

    def test_aircraft_invalid_destination(self):
        """Check TypeError is thrown where an invalid destination is specified."""
        behaviour = AircraftMoveBehaviour(base_speed=10, height_bands=self.height_bands)

        with self.assertRaises(TypeError) as e:
            behaviour.step(mover=self.agent, destination=(10.5, 10.5, 10.5))
        self.assertEqual("Move called with invalid destination", str(e.exception))

        with self.assertRaises(TypeError) as e:
            behaviour.step(mover=self.agent, destination=(10.5, 10.5))
        self.assertEqual("Move called with invalid destination", str(e.exception))

    def test_deploy_countermeasure(self):
        """Check the deployment of countermeasures."""
        behaviour = DeployCountermeasureBehaviour()
        countermeasure = MockCountermeasure()
        behaviour.step(deployer=self.agent, countermeasure=countermeasure)
        self.assertTrue(countermeasure.deployed)


class TestMovementBehaviourUtilities(unittest.TestCase):
    """Tests for movement behaviour utilities."""

    @parameterized.expand([[2], [3]])
    def test_unit_direction_calculation(self, dimension: int):
        """Test calculation of unit direction.

        Args:
            dimension (int): Number of dimensions in model
        """
        random_array = np.random.uniform(low=-1000, high=1000, size=(dimension,))

        # Ensure that the array is not (0, 0, 0), which does not have a defined
        # unit vector
        if all(i == 0 for i in random_array):
            random_array[0] += 1.0

        random_coordinate = typing.cast(Coordinate2dOr3d, tuple(random_array.tolist()))
        unit_coordinate = MoveBehaviour.calculate_unit_direction(random_coordinate)

        ratio = random_array[0] / unit_coordinate[0]

        for i in range(1, dimension):
            assert math.isclose(ratio, random_array[i] / unit_coordinate[i])

    @parameterized.expand([[2], [3]])
    def test_unit_calculation_throws_exception_for_zero_magnitude(self, dimension: int):
        """Test exception handling for unit direction with zero magnitude.

        Args:
            dimension (int): Number of dimensions in model
        """
        coordinate: Coordinate2dOr3d = tuple((0.0) for _ in range(dimension))  # type: ignore
        with self.assertRaises(ValueError) as e:
            MoveBehaviour.calculate_unit_direction(coordinate)
        self.assertEqual("Cannot determine orientation of zero-length vector", str(e.exception))


class TestCommunicateBehaviour(unittest.TestCase):
    """Tests for communication behaviour."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()
        self.sender = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Communicator",
            model=self.model,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

        self.recipient = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            pos=(10, 10),
            name="Recipient",
            model=self.model,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

    def test_communicate_perceived_world(self):
        """Test comminication of perceived world-view."""
        behaviour = CommunicateWorldviewBehaviour()

        # Add a perceived agent to the sender's world-view
        self.sender.perceived_world.add_acquisitions(
            PerceivedAgent(
                unique_id=101,
                sense_time=self.model.get_time(),
                location=(100, 100),
                confidence=Confidence.DETECT,
                casualty_state=AgentCasualtyState.ALIVE,
            )
        )

        behaviour.step(self.sender, self.recipient)

        perceived_agents = self.recipient.perceived_world.get_perceived_agents()

        # 3x perceived agents, as the sender will know about it's self and the bogey
        # and the recipient will know about itself
        assert len(perceived_agents) == 3
        bogey = next((a for a in perceived_agents if a.confidence == Confidence.DETECT), None)

        assert bogey is not None
        assert bogey.location[0] == 100
        assert bogey.location[1] == 100

    def test_communicate_perceived_world_with_filter(self):
        """Test application of filter to sharing a worldview."""
        behaviour = CommunicateWorldviewBehaviour()

        self.sender.perceived_world.add_acquisitions(
            [
                PerceivedAgent(
                    unique_id=102,
                    sense_time=self.model.get_time(),
                    location=(100, 100),
                    confidence=Confidence.DETECT,
                    casualty_state=AgentCasualtyState.ALIVE,
                ),
                PerceivedAgent(
                    unique_id=103,
                    sense_time=self.model.get_time(),
                    location=(200, 200),
                    confidence=Confidence.IDENTIFY,
                    casualty_state=AgentCasualtyState.ALIVE,
                ),
            ]
        )

        # Only communicate the position of recognised agents
        behaviour.step(self.sender, self.recipient, RecognisedAgents())

        perceived_agents = self.recipient.perceived_world.get_perceived_agents()

        # 3x perceived agents, as the sender will know about it's self and the
        # IDENTIFIED bogey and the recipient will know about itself. The DETECT-level
        # bogey will not have been communicated
        assert len(perceived_agents) == 3

    def test_communicate_mission_message(self):
        """Test comminication of mission message."""
        behaviour = CommunicateMissionMessageBehaviour()

        behaviour.step(self.sender, self.recipient, "test")

        assert len(self.recipient.mission_messages) == 1
        assert "test" in self.recipient.mission_messages

    def test_communicate_orders(self):
        """Test communication of orders."""
        behaviour = CommunicateOrdersBehaviour()

        random_int = random.randint(0, 9999)
        orders: list[Template[Order]] = [MockOrderWithId(random_int)]

        assert len(self.recipient._orders) == 0
        behaviour.step(self.sender, self.recipient, orders)
        assert len(self.recipient._orders) == 1
        assert self.recipient._orders[0].id == random_int


class GuaranteedSensor(MockSensor):
    """Sensor that always returns a known PerceivedAgent."""

    def __init__(self) -> None:
        """Initialise guaranteed sensor.

        Boolean to flag whether detection method has been called.
        """
        super().__init__()
        self.detection_ran = False

    def _run_detection(
        self, sensor_agent: RetAgent, all_target_agents: list[RetAgent]
    ) -> list[PerceivedAgent]:
        """Set internal detection_ran boolean to true.

        Args:
            sensor_agent (RetAgent): Agent using guaranteed sensor.
            all_target_agents (list[RetAgent]): A list of potentially perceiveable agents

        """
        self.detection_ran = True
        return []

    def get_results(self, sensor_agent: RetAgent) -> list[PerceivedAgent]:
        """Return a known perceived agent.

        Args:
            sensor_agent (RetAgent): Agent using sensor for detection test

        Returns:
            list[PerceivedAgent]: Always the same PerceivedAgent
        """
        return [
            PerceivedAgent(
                sense_time=datetime(2020, 1, 1),
                confidence=Confidence.DETECT,
                location=(1, 1),
                unique_id=None,
                casualty_state=AgentCasualtyState.ALIVE,
            )
        ]

    def get_new_instance(self) -> GuaranteedSensor:
        """Return a new instance of a functionally identical sensor.

        Returns:
            GuaranteedSensor: New instance of the sensor
        """
        return GuaranteedSensor()


class TestSenseBehaviour(unittest.TestCase):
    """Tests for sense behaviour."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModel2d()
        self.agent = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
            pos=(0, 0),
            name="Agent",
            model=self.model,
            sensors=[GuaranteedSensor()],
            behaviours=[
                SenseBehaviour(
                    time_before_first_sense=timedelta(seconds=0),
                    time_between_senses=timedelta(seconds=5),
                )
            ],
        )

    def test_sense(self):
        """Test sensing behaviour."""
        perceived_agents = self.agent.perceived_world.get_perceived_agents()
        assert len(perceived_agents) == 1

        self.agent.sense_step(sense_direction=90)
        self.agent.step()

        for s in self.agent._sensors:
            assert s.detection_ran is True

        perceived_agents = self.agent.perceived_world.get_perceived_agents()
        assert len(perceived_agents) == 2

        sensed_agents = self.agent.perceived_world.get_perceived_agents(UnknownAgents())
        assert len(sensed_agents) == 1

        assert sensed_agents[0].location == (1, 1)

    def test_sense_behaviour_invalid_time_between_senses(self):
        """Test exception handling where invalid time between senses is provided."""
        with self.assertRaises(ValueError) as ve:
            SenseBehaviour(
                time_between_senses=timedelta(seconds=-1),
                time_before_first_sense=timedelta(seconds=1),
            )
        self.assertEqual(
            "Time between senses must be non-negative. -1.0s provided.",
            str(ve.exception),
        )

    def test_sense_behaviour_invalid_time_before_first_sense(self):
        """Test exception handling where invalid time before first sense is provided."""
        with self.assertRaises(ValueError) as ve:
            SenseBehaviour(
                time_between_senses=timedelta(seconds=1),
                time_before_first_sense=timedelta(seconds=-1),
            )
        self.assertEqual(
            "Time before first sense must be non-negative. -1.0s provided.",
            str(ve.exception),
        )


class TestGroundBasedMoveWithFeatures(unittest.TestCase):
    """Tests for ground based move behaviour with features."""

    base_speed = 1
    area_modifer = 1.5

    gradient_speed_modifiers: list[tuple[tuple[float, float], float]] = [((-inf, inf), 1)]

    def setUp(self):
        """Set up test cases."""
        area = BoxFeature((0, 0), (1, 1), "area")
        area2 = BoxFeature((4, 4), (6, 6), "area2")
        boundary = LineFeature((0, 0), (10, 10), "boundary")
        self.space = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=10000, y_max=10000, features=[area, area2, boundary]
        )
        self.model = MockModel(space=self.space, time_step=timedelta(minutes=1))
        self.agent_inside = RetAgent(
            model=self.model,
            pos=(0, 0),
            name="agent inside",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        self.agent_outside = RetAgent(
            model=self.model,
            pos=(5, 5),
            name="agent outside",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        self.agent_one_side = RetAgent(
            model=self.model,
            pos=(5, 0),
            name="agent one side",
            affiliation=Affiliation.FRIENDLY,
            critical_dimension=2.0,
            reflectivity=0.1,
            temperature=20.0,
        )

    def test_ground_based_move_behaviour_get_area_modifer(self):
        """Test ground-based movement get area modifer method."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed,
            self.gradient_speed_modifiers,
            area_speed_modifiers={"area": self.area_modifer},
        )
        assert behaviour.get_area_modifier(self.agent_inside) == self.area_modifer
        assert behaviour.get_area_modifier(self.agent_outside) == 1

        behaviour_2 = GroundBasedMoveBehaviour(
            self.base_speed,
            self.gradient_speed_modifiers,
            area_speed_modifiers={"###": self.area_modifer},
        )
        with self.assertRaises(KeyError) as e:
            behaviour_2.get_area_modifier(self.agent_inside)
        self.assertEqual(
            "'area movement modifier specified for an area that does not exist: ###'",
            str(e.exception),
        )

    def test_ground_based_move_behaviour_has_crossed_boundary(self):
        """Test ground-based movement has crossed boundary method."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, impassable_boundaries=["boundary"]
        )
        assert behaviour.has_crossed_boundary(self.agent_one_side, (0, 5)) is True
        assert behaviour.has_crossed_boundary(self.agent_one_side, (6, 0)) is False
        assert behaviour.has_crossed_boundary(self.agent_one_side, (5, 5)) is True
        # agent inside always crosses boundary as it starts on boundary at (0,0)
        assert behaviour.has_crossed_boundary(self.agent_inside, (5, 5)) is True
        assert behaviour.has_crossed_boundary(self.agent_inside, (0, 5)) is True
        assert behaviour.has_crossed_boundary(self.agent_inside, (5, 0)) is True

        behaviour_2 = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, impassable_boundaries=["###"]
        )
        with self.assertRaises(KeyError) as e:
            behaviour_2.has_crossed_boundary(self.agent_one_side, (0, 5))
        self.assertEqual(
            "'impassable boundary specified for an boundary that does not exist: ###'",
            str(e.exception),
        )

    def test_ground_based_move_behaviour_has_crossed_area_as_boundary(self):
        """Test ground-based movement has crossed areas as boundary method."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, impassable_boundaries=["area", "area2"]
        )
        assert behaviour.has_crossed_boundary(self.agent_one_side, (0, 5)) is False  # jumps over
        assert behaviour.has_crossed_boundary(self.agent_one_side, (6, 0)) is False
        assert behaviour.has_crossed_boundary(self.agent_one_side, (5, 5)) is True
        # moves from one area into another
        assert behaviour.has_crossed_boundary(self.agent_inside, (5, 5)) is True
        # moves out of area
        assert behaviour.has_crossed_boundary(self.agent_inside, (0, 5)) is False

    def test_ground_based_move_behaviour_with_area(self):
        """Test ground-based movement with an area."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed,
            self.gradient_speed_modifiers,
            area_speed_modifiers={"area": self.area_modifer},
        )

        behaviour.step(self.agent_inside, (100, 0))
        assert self.agent_inside.pos is not None
        assert (
            self.agent_inside.pos[0]
            == self.base_speed * 1 * self.area_modifer * self.model.time_step.seconds
        )
        assert self.agent_inside.pos[1] == 0

        behaviour.step(self.agent_outside, (105, 5))
        assert self.agent_outside.pos is not None
        assert (
            self.agent_outside.pos[0]
            == (self.base_speed * 1 * 1 * self.model.time_step.seconds) + 5
        )
        assert self.agent_outside.pos[1] == 5

    def test_ground_based_move_behaviour_with_boundary(self):
        """Test ground-based movement has crossed boundary method."""
        behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, impassable_boundaries=["boundary"]
        )

        # Cannot move across the boundary
        behaviour.step(self.agent_one_side, (0, 5))
        assert self.agent_one_side.pos is not None
        assert self.agent_one_side.pos[0] == 5
        assert self.agent_one_side.pos[1] == 0

        # Cannot move onto the boundary
        behaviour.step(self.agent_one_side, (5, 5))
        assert self.agent_one_side.pos is not None
        assert self.agent_one_side.pos[0] == 5
        assert self.agent_one_side.pos[1] == 0

        # Can move away from the boundary
        behaviour.step(self.agent_one_side, (10, 0))
        assert self.agent_one_side.pos is not None
        assert self.agent_one_side.pos[0] == 10
        assert self.agent_one_side.pos[1] == 0

        # Can move towards the boundary
        behaviour.step(self.agent_one_side, (5, 4))
        assert self.agent_one_side.pos is not None
        assert self.agent_one_side.pos[0] == 5
        assert self.agent_one_side.pos[1] == 4

    def test_ground_based_move_behaviour_with_area_as_boundary(self):
        """Test ground-based movement has crossed area as boundary method."""
        speed = 1 / 60  # move 1 space per time step
        slow_move_behaviour = GroundBasedMoveBehaviour(
            speed, self.gradient_speed_modifiers, impassable_boundaries=["area"]
        )
        fast_move_behaviour = GroundBasedMoveBehaviour(
            self.base_speed, self.gradient_speed_modifiers, impassable_boundaries=["area2"]
        )

        # Can't move into area
        fast_move_behaviour.step(self.agent_one_side, (5, 5))
        assert self.agent_one_side.pos is not None
        assert self.agent_one_side.pos[0] == 5
        assert self.agent_one_side.pos[1] == 0

        # Can jump over area
        fast_move_behaviour.step(self.agent_one_side, (10, 0))
        assert self.agent_one_side.pos is not None
        assert self.agent_one_side.pos[0] == 10
        assert self.agent_one_side.pos[1] == 0

        # Can move inside area
        slow_move_behaviour.step(self.agent_inside, (0, 1))
        assert self.agent_inside.pos is not None
        assert self.agent_inside.pos[0] == 0
        assert self.agent_inside.pos[1] == 1

        # Can move out of area
        slow_move_behaviour.step(self.agent_inside, (0, 2))
        assert self.agent_inside.pos is not None
        assert self.agent_inside.pos[0] == 0
        assert self.agent_inside.pos[1] == 2


class TestHideBehaviour(unittest.TestCase):
    """Tests for hide behaviour."""

    def test_hide(self):
        """Test hide behaviour, with regular tasks/orders."""
        model = MockModel2d()
        agent = RetAgent(
            affiliation=Affiliation.FRIENDLY,
            pos=(0, 0),
            name="Agent",
            model=model,
            critical_dimension=1.0,
            reflectivity=0.1,
            temperature=20.0,
        )
        behaviour = HideBehaviour()

        assert agent.hiding is False
        behaviour.step(agent)
        assert agent.hiding is True
        agent.step()
        assert agent.hiding is False
