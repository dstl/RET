"""Tests for clutter."""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ret.agents.affiliation import Affiliation
from ret.agents.agentfilter import FilterByAffiliation
from ret.space.clutter.clutter import ClutterField
from ret.space.clutter.cluttermodifiers.box import BoxClutterModifier
from ret.space.clutter.cluttermodifiers.groundplane import GroundPlaneClutterModifier
from ret.space.clutter.cluttermodifiers.sphere import (
    SphereClutterModifier,
    SphereFollowerClutterModifier,
)
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from ret.testing.mocks import MockAgent, MockAgentWithAffiliation
from parameterized import parameterized
from testsret.test_space.test_space import H_0, H_127, H_191, H_255

if TYPE_CHECKING:
    from ret.types import Coordinate2d, Coordinate3d


class TestClutterModifiers(unittest.TestCase):
    """Test cases for clutter modifiers."""

    def setUp(self):
        """Set up test cases."""
        self.mockAgent = MockAgent(1, (0, 0, 0))

    def test_sphere_2d(self):
        """Test how sphere modifies 2D clutter."""
        modifier = SphereClutterModifier(10, (10, 10), 10)

        assert modifier.get_value((10, 10), self.mockAgent) == 10
        assert modifier.get_value((10, 0), self.mockAgent) == 10
        assert modifier.get_value((0, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 5), self.mockAgent) == 10
        assert modifier.get_value((0, 0), self.mockAgent) == 0

    def test_sphere_3d(self):
        """Test how sphere modifies 3D clutter."""
        modifier = SphereClutterModifier(10, (10, 10, 10), 10)

        assert modifier.get_value((10, 10, 10), self.mockAgent) == 10
        assert modifier.get_value((10, 10, 0), self.mockAgent) == 10
        assert modifier.get_value((10, 0, 10), self.mockAgent) == 10
        assert modifier.get_value((0, 10, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 5, 5), self.mockAgent) == 10
        assert modifier.get_value((0, 0, 0), self.mockAgent) == 0

    def test_sphere_invalid(self):
        """Test that sphere can't be created with invalid inputs."""
        with self.assertRaises(ValueError) as e:
            SphereClutterModifier(10, (10, 10), -1)
        self.assertEqual("Radius must be positive", str(e.exception))

    def test_sphere_follower_2d(self):
        """Test how sphere attached to agent modifies 2D clutter."""
        agent = MockAgent(1, (10, 10))
        modifier = SphereFollowerClutterModifier(10, agent, 10)

        assert modifier.get_value((10, 10), self.mockAgent) == 10
        assert modifier.get_value((10, 0), self.mockAgent) == 10
        assert modifier.get_value((0, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 5), self.mockAgent) == 10
        assert modifier.get_value((0, 0), self.mockAgent) == 0

        agent.pos = (0, 0)

        assert modifier.get_value((10, 10), self.mockAgent) == 0
        assert modifier.get_value((10, 0), self.mockAgent) == 10
        assert modifier.get_value((0, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 5), self.mockAgent) == 10
        assert modifier.get_value((0, 0), self.mockAgent) == 10

    def test_sphere_follower_3d(self):
        """Test how sphere attached to agent modifies 3D clutter."""
        agent = MockAgent(1, (10, 10, 10))
        modifier = SphereFollowerClutterModifier(10, agent, 10)

        assert modifier.get_value((10, 10, 10), self.mockAgent) == 10
        assert modifier.get_value((10, 10, 0), self.mockAgent) == 10
        assert modifier.get_value((10, 0, 10), self.mockAgent) == 10
        assert modifier.get_value((0, 10, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 5, 5), self.mockAgent) == 10
        assert modifier.get_value((0, 0, 0), self.mockAgent) == 0

        agent.pos = (0, 0, 0)

        assert modifier.get_value((10, 10, 10), self.mockAgent) == 0
        assert modifier.get_value((10, 10, 0), self.mockAgent) == 0
        assert modifier.get_value((10, 0, 10), self.mockAgent) == 0
        assert modifier.get_value((0, 10, 10), self.mockAgent) == 0
        assert modifier.get_value((5, 5, 5), self.mockAgent) == 10
        assert modifier.get_value((0, 0, 0), self.mockAgent) == 10

    def test_box_2d(self):
        """Test how box modifies 2D clutter."""
        modifier = BoxClutterModifier(10, (5, 5), (10, 10))

        assert modifier.get_value((10, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 5), self.mockAgent) == 10
        assert modifier.get_value((10, 5), self.mockAgent) == 10

        assert modifier.get_value((7, 7), self.mockAgent) == 10

        assert modifier.get_value((0, 0), self.mockAgent) == 0
        assert modifier.get_value((0, 7), self.mockAgent) == 0
        assert modifier.get_value((7, 0), self.mockAgent) == 0
        assert modifier.get_value((12, 7), self.mockAgent) == 0
        assert modifier.get_value((7, 12), self.mockAgent) == 0
        assert modifier.get_value((12, 12), self.mockAgent) == 0

    def test_box_3d(self):
        """Test how box modifies 3D clutter."""
        modifier = BoxClutterModifier(10, (5, 5, 5), (10, 10, 10))

        assert modifier.get_value((10, 10, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 10, 10), self.mockAgent) == 10
        assert modifier.get_value((10, 5, 10), self.mockAgent) == 10
        assert modifier.get_value((10, 10, 5), self.mockAgent) == 10
        assert modifier.get_value((10, 5, 5), self.mockAgent) == 10
        assert modifier.get_value((5, 10, 5), self.mockAgent) == 10
        assert modifier.get_value((5, 5, 10), self.mockAgent) == 10
        assert modifier.get_value((5, 5, 5), self.mockAgent) == 10

        assert modifier.get_value((7, 7, 7), self.mockAgent) == 10

        assert modifier.get_value((0, 0, 0), self.mockAgent) == 0
        assert modifier.get_value((0, 7, 7), self.mockAgent) == 0
        assert modifier.get_value((7, 0, 7), self.mockAgent) == 0
        assert modifier.get_value((7, 7, 0), self.mockAgent) == 0
        assert modifier.get_value((12, 7, 7), self.mockAgent) == 0
        assert modifier.get_value((7, 12, 7), self.mockAgent) == 0
        assert modifier.get_value((7, 7, 12), self.mockAgent) == 0
        assert modifier.get_value((12, 12, 12), self.mockAgent) == 0

    def test_box_invalid(self):
        """Test that box can't be created with invalid inputs."""
        with self.assertRaises(ValueError) as e:
            BoxClutterModifier(10, (10, 10), (5, 5))
        self.assertEqual(
            "max_coord (5, 5) must be greater than min_coord (10, 10) in all directions",
            str(e.exception),
        )

        with self.assertRaises(ValueError) as e:
            BoxClutterModifier(10, (10, 10, 10), (5, 5, 5))
        self.assertEqual(
            "max_coord (5, 5, 5) must be greater than min_coord (10, 10, 10) in all directions",
            str(e.exception),
        )

    def test_ground_plane_2d_flat(self):
        """Test how ground plane modifies 2D clutter."""
        terrain = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=70,
            y_max=50,
        )

        modifier = GroundPlaneClutterModifier(10, terrain, 10)

        assert modifier.get_value((0, 0), self.mockAgent) == 10

    def test_ground_plane_3d_flat(self):
        """Test how ground plain modifies 3D clutter."""
        terrain = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=70,
            y_max=50,
        )

        modifier = GroundPlaneClutterModifier(10, terrain, 10)

        assert modifier.get_value((0, 0, 0), self.mockAgent) == 10
        assert modifier.get_value((0, 0, 5), self.mockAgent) == 10
        assert modifier.get_value((0, 0, 10), self.mockAgent) == 10
        assert modifier.get_value((0, 0, 12), self.mockAgent) == 0

    @parameterized.expand(
        [[(-5, -25), H_191], [(45, -25), H_127], [(45, 25), H_255], [(-5, 25), H_0]]
    )
    def test_ground_plane_2d(self, coord: Coordinate2d, terrain_height: float):
        """Test how ground plane modifies 2D clutter accounting for terrain.

        Args:
            coord (Coordinate2d): Coordinate
            terrain_height (float): Terrain height
        """
        terrain = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=70,
            y_max=50,
            x_min=-30,
            y_min=-50,
            terrain_image_path=str(Path(__file__).parent.joinpath("TestTerrain.png")),
            height_black=H_0,
            height_white=H_255,
        )

        modifier = GroundPlaneClutterModifier(10, terrain, 10)

        assert modifier.get_value(coord, self.mockAgent) == 10

    @parameterized.expand(
        [[(-5, -25), H_191], [(45, -25), H_127], [(45, 25), H_255], [(-5, 25), H_0]]
    )
    def test_ground_plane_3d(self, coord: Coordinate3d, terrain_height: float):
        """Test how ground plane modifies 3D clutter accounting for terrain.

        Args:
            coord (Coordinate3d): Coordinate
            terrain_height (float): Terrain height
        """
        terrain = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=70,
            y_max=50,
            x_min=-30,
            y_min=-50,
            terrain_image_path=str(Path(__file__).parent.joinpath("TestTerrain.png")),
            height_black=H_0,
            height_white=H_255,
        )

        modifier = GroundPlaneClutterModifier(10, terrain, 10)

        assert modifier.get_value((coord[0], coord[1], terrain_height), self.mockAgent) == 10
        assert modifier.get_value((coord[0], coord[1], terrain_height + 5), self.mockAgent) == 10
        assert (
            modifier.get_value((coord[0], coord[1], terrain_height + 9.999), self.mockAgent) == 10
        )
        assert (
            modifier.get_value((coord[0], coord[1], terrain_height + 10.001), self.mockAgent) == 0
        )
        assert modifier.get_value((coord[0], coord[1], terrain_height + 12), self.mockAgent) == 0

    def test_ground_plane_invalid(self):
        """Test that ground plane can't be created with invalid inputs."""
        terrain = ContinuousSpaceWithTerrainAndCulture2d(
            x_max=70,
            y_max=50,
        )

        with self.assertRaises(ValueError) as e:
            GroundPlaneClutterModifier(10, terrain, -1)
        self.assertEqual("Height must be positive", str(e.exception))


class TestClutterField(unittest.TestCase):
    """Test cases for clutter fields."""

    def setUp(self):
        """Set up test cases."""
        self.clutterField = ClutterField(
            1,
            [
                SphereClutterModifier(2, (0, 0, 0), 10),
                BoxClutterModifier(4, (0, 0, 0), (1, 1, 1)),
            ],
        )

        hostile_agent_filter = FilterByAffiliation(Affiliation.HOSTILE)
        neutral_agent_filter = FilterByAffiliation(Affiliation.NEUTRAL)
        self.clutterFieldWithAffiliation = ClutterField(
            1,
            [
                SphereClutterModifier(2, (0, 0, 0), 10, agent_filter=hostile_agent_filter),
                BoxClutterModifier(4, (0, 0, 0), (1, 1, 1), agent_filter=neutral_agent_filter),
            ],
        )

        self.mockAgent = MockAgent(1, (0, 0, 0))

    def test_get_value(self):
        """Test accessing clutter values at points in 3D space."""
        # no agent filter so any agent can see clutter
        assert self.clutterField.get_value((0, 0, 0), self.mockAgent) == 7
        assert self.clutterField.get_value((5, 5, 5), self.mockAgent) == 3
        assert self.clutterField.get_value((12, 12, 12), self.mockAgent) == 1

    def test_get_value_with_agent_filter(self):
        """Test accessing clutter values at points in 3D space."""
        friendly_agent = MockAgentWithAffiliation(1, Affiliation.FRIENDLY)
        neutral_agent = MockAgentWithAffiliation(2, Affiliation.NEUTRAL)
        hostile_agent = MockAgentWithAffiliation(3, Affiliation.HOSTILE)
        assert self.clutterFieldWithAffiliation.get_value((0, 0, 0), friendly_agent) == 1
        assert self.clutterFieldWithAffiliation.get_value((0, 0, 0), neutral_agent) == 5
        assert self.clutterFieldWithAffiliation.get_value((0, 0, 0), hostile_agent) == 3

    def test_modify(self):
        """Test that clutter field can be modified by adding modifiers."""
        assert len(self.clutterField._modifiers) == 2
        assert self.clutterField.get_value((0, 0, 0), self.mockAgent) == 7

        self.clutterField.modify(SphereClutterModifier(8, (0, 0, 0), 5))

        assert len(self.clutterField._modifiers) == 3
        assert self.clutterField.get_value((0, 0, 0), self.mockAgent) == 15

    def test_remove_expired_modifiers(self):
        """Test that out of date modifiers are automatically removed from field."""
        assert len(self.clutterField._modifiers) == 2

        self.clutterField.modify(SphereClutterModifier(8, (0, 0, 0), 1, datetime(2020, 1, 2)))
        assert len(self.clutterField._modifiers) == 3
        assert self.clutterField.get_value((0, 0, 0), self.mockAgent) == 15

        self.clutterField.remove_expired_modifiers(datetime(2020, 1, 1))
        assert len(self.clutterField._modifiers) == 3
        assert self.clutterField.get_value((0, 0, 0), self.mockAgent) == 15

        self.clutterField.remove_expired_modifiers(datetime(2020, 1, 2))
        assert len(self.clutterField._modifiers) == 2
        assert self.clutterField.get_value((0, 0, 0), self.mockAgent) == 7

        self.clutterField.remove_expired_modifiers(datetime(2020, 1, 3))
        assert len(self.clutterField._modifiers) == 2
        assert self.clutterField.get_value((0, 0, 0), self.mockAgent) == 7
