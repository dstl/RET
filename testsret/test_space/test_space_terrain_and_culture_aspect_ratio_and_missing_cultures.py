"""Tests for space and terrain with missing cultures."""
import unittest
import warnings
from pathlib import Path

from ret.space.culture import Culture
from ret.space.space import (
    ContinuousSpaceWithTerrainAndCulture2d,
    ContinuousSpaceWithTerrainAndCulture3d,
)
from testsret.test_space.test_space import H_0, H_255

culture_red = Culture("red culture")
culture_green = Culture("green culture")
culture_blue = Culture("blue culture")
culture_yellow = Culture("yellow culture")


class TestTerrainAndCultureAspectRatioAndMissingCultures(unittest.TestCase):
    """Test terrain and space aspect ratio with missing cultures in 2D space."""

    def test_aspect_ratio_warning(self):
        """Test aspect ratio warning.

        A warning is raised when the image aspect ratio doesn't match the space aspect
        ratio closely enough.
        """
        with warnings.catch_warnings(record=True) as w:
            ContinuousSpaceWithTerrainAndCulture2d(
                x_max=70,
                y_max=50,
                x_min=-30,
                y_min=-50,
                terrain_image_path=str(Path(__file__).parent.joinpath("TestTerrainAspect.png")),
                height_black=H_0,
                height_white=H_255,
            )
            assert len(w) == 1
            assert "space aspect ratio by more than 5%" in str(w[0].message)

    def test_missing_culture(self):
        """Test missing culture.

        An exception is raised when the culture image contains a color that is not
        supplied in the culture dictionary.
        """
        with self.assertRaises(KeyError) as e:
            ContinuousSpaceWithTerrainAndCulture2d(
                x_max=70,
                y_max=50,
                x_min=-30,
                y_min=-50,
                culture_image_path=str(Path(__file__).parent.joinpath("TestCulture.png")),
                culture_dictionary={
                    (255, 0, 0): culture_red,
                    (0, 255, 0): culture_green,
                    (255, 255, 0): culture_yellow,
                },
            )
        self.assertEqual(
            "'Culture image has color ((0, 0, 255)) that is not defined in the culture "
            + "dictionary.'",
            str(e.exception),
        )

    def test_culture_missing_dict(self):
        """Tests exception handling where culture map is missing culture dictionary."""
        with self.assertRaises(TypeError) as e:
            ContinuousSpaceWithTerrainAndCulture2d(
                x_max=100,
                y_max=100,
                x_min=0,
                y_min=0,
                culture_image_path=str(Path(__file__).parent.joinpath("TestCulture.png")),
            )
        self.assertEqual(
            "Must provide culture dictionary if providing culture image",
            str(e.exception),
        )


class TestTerrainAndCultureAspectRatioAndMissingCultures3d(
    TestTerrainAndCultureAspectRatioAndMissingCultures
):
    """Test terrain and space aspect ratio with missing cultures in 3D space."""

    def test_aspect_ratio_warning(self):
        """Test aspect ratio warning.

        A warning is raised when the image aspect ratio doesn't match the space aspect
        ratio closely enough.
        """
        with warnings.catch_warnings(record=True) as w:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=70,
                y_max=50,
                x_min=-30,
                y_min=-50,
                terrain_image_path=str(Path(__file__).parent.joinpath("TestTerrainAspect.png")),
                height_black=H_0,
                height_white=H_255,
            )
            assert len(w) == 1
            assert "space aspect ratio by more than 5%" in str(w[0].message)

    def test_missing_culture(self):
        """Test missing culture.

        An exception is raised when the culture image contains a color that is not
        supplied in the culture dictionary.
        """
        with self.assertRaises(KeyError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=70,
                y_max=50,
                x_min=-30,
                y_min=-50,
                culture_image_path=str(Path(__file__).parent.joinpath("TestCulture.png")),
                culture_dictionary={
                    (255, 0, 0): culture_red,
                    (0, 255, 0): culture_green,
                    (255, 255, 0): culture_yellow,
                },
            )
        self.assertEqual(
            "'Culture image has color ((0, 0, 255)) that is not defined in the culture "
            + "dictionary.'",
            str(e.exception),
        )

    def test_culture_missing_dict(self):
        """Tests exception handling where culture map is missing culture dictionary."""
        with self.assertRaises(TypeError) as e:
            ContinuousSpaceWithTerrainAndCulture3d(
                x_max=100,
                y_max=100,
                x_min=0,
                y_min=0,
                culture_image_path=str(Path(__file__).parent.joinpath("TestCulture.png")),
            )
        self.assertEqual(
            "Must provide culture dictionary if providing culture image",
            str(e.exception),
        )
