"""Test that culture penetration is calculated correctly."""
import os
import unittest

from mesa_ret.space.culture import Culture
from mesa_ret.space.culturemap import LineOfSightException
from mesa_ret.space.space import ContinuousSpaceWithTerrainAndCulture3d

culture_red = Culture("red culture", 100.0)
culture_green = Culture("green culture", 100.0)
culture_blue = Culture("blue culture")
culture_yellow = Culture("yellow culture", 25.0)

position_red_z1 = (75.0, 25.0, 1.0)
position_green_z1 = (25.0, 25.0, 1.0)
position_blue_z1 = (75.0, 75.0, 1.0)
position_yellow_z1 = (25.0, 75.0, 1.0)

position_red_z50 = (75.0, 25.0, 50.0)
position_green_z50 = (25.0, 25.0, 50.0)
position_blue_z50 = (75.0, 75.0, 50.0)
position_yellow_z50 = (25.0, 75.0, 50.0)


class TestCulturePenetration(unittest.TestCase):
    """Test culture penetration is calculated correctly.

    Tests applied to ContinuousSpaceWithTerrainAndCulture3d space.
    """

    def setUp(self):
        """Set up the 3d space."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            x_min=0,
            y_min=0,
            terrain_image_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "TestLineOfSight.png")
            ),
            culture_image_path=os.path.abspath(
                os.path.join(os.path.dirname(__file__), "TestCulture.png")
            ),
            culture_dictionary={
                (255, 0, 0): culture_red,
                (0, 255, 0): culture_green,
                (0, 0, 255): culture_blue,
                (255, 255, 0): culture_yellow,
            },
        )

    def test_get_culture(self):
        """Test multiple culture types can be found correctly."""
        assert self.space.get_culture(position_red_z1) == culture_red
        assert self.space.get_culture(position_green_z1) == culture_green
        assert self.space.get_culture(position_blue_z1) == culture_blue
        assert self.space.get_culture(position_yellow_z1) == culture_yellow

    def test_culture_penetration_los(self):
        """Test culture penetration calculation through two culture types."""
        culture_penetration_dictionary_z1 = self.space.check_culture_penetration(
            position_green_z1, position_yellow_z1, 0.5
        )
        assert len(culture_penetration_dictionary_z1) == 2
        assert culture_penetration_dictionary_z1.get(culture_green) == 25.0
        assert culture_penetration_dictionary_z1.get(culture_yellow) == 25.0

    def test_culture_penetration_los_above_culture(self):
        """Test line of site above culture.

        Test culture penetration through one culture type and above another culture type
        are calculated correctly
        """
        culture_penetration_dictionary_z50 = self.space.check_culture_penetration(
            position_green_z50, position_yellow_z50, 0.5
        )
        assert len(culture_penetration_dictionary_z50) == 1
        assert culture_penetration_dictionary_z50.get(culture_green) == 25.0

    def test_culture_penetration_without_los(self):
        """Test culture penetration without line of sight raises exception."""
        with self.assertRaises(LineOfSightException) as e:
            self.space.check_culture_penetration(position_green_z1, position_red_z1)
        self.assertEqual("Input positions do not have line of sight.", str(e.exception))


class TestCulturePenetrationNoCulture(unittest.TestCase):
    """Test culture penetration is calculated correctly when no culture is supplied.

    Tests applied to ContinuousSpaceWithTerrainAndCulture3d space.
    """

    def setUp(self):
        """Set up the 3d space."""
        self.space = ContinuousSpaceWithTerrainAndCulture3d(
            x_max=100,
            y_max=100,
            x_min=0,
            y_min=0,
        )

    def test_culture_penetration_no_culture(self):
        """Test culture penetration when culture not supplied.

        Ensure returns empty culture dictionary.
        """
        assert self.space.check_culture_penetration(pos_a=(0, 0, 0), pos_b=(0, 10, 0)) == {}
